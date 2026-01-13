import os
import torch
import random
import numpy as np
from pathlib import Path
from torchvision.utils import make_grid, save_image
import neptune
import torch.nn.functional as F
import matplotlib.pyplot as plt

from masked_transformer.model import MaskedLatentTransformer
from vqvae.flat.flat_models import EnhancedFlatVQVAE
from masked_transformer.utils import mask_latents
from configs.config import Config
from model_def import build_model
from data import build_dataloaders

from PIL import Image
from torchvision import transforms
import glob



def overlay_mask_on_image(image, mask, H_latent, W_latent):
    """
    Overlay a semi-transparent gray mask on decoded image.
    image: (B, 3, H_img, W_img)
    mask:  (B, H_latent * W_latent) boolean tensor
    """
    B, _, H_img, W_img = image.shape
    mask_2d = mask.view(B, 1, H_latent, W_latent).float()
    mask_resized = F.interpolate(mask_2d, size=(H_img, W_img), mode="nearest")
    mask_3ch = mask_resized.repeat(1, 3, 1, 1)
    masked_img = image * (1 - 0.5 * mask_3ch)
    return masked_img


def draw_frame_on_image(img, color=(1.0, 0.0, 0.0), thickness=4):
    """
    Draw a rectangular frame around a single image tensor.

    img: (3, H, W) or (1, 3, H, W) in range [-1, 1]
    color: (R, G, B) in [0, 1]
    """
    if img.dim() == 4:
        assert img.size(0) == 1, "Expected batch of size 1"
        img = img[0]

    C, H, W = img.shape

    # map color [0,1] -> [-1,1] to match your value_range
    mapped = [2 * c - 1 for c in color]

    # top border
    img[:, :thickness, :] = torch.tensor(mapped, device=img.device).view(3, 1, 1)
    # bottom border
    img[:, -thickness:, :] = torch.tensor(mapped, device=img.device).view(3, 1, 1)
    # left border
    img[:, :, :thickness] = torch.tensor(mapped, device=img.device).view(3, 1, 1)
    # right border
    img[:, :, -thickness:] = torch.tensor(mapped, device=img.device).view(3, 1, 1)

    return img




def build_input_transform(input_size: int):
    # to [-1, 1] like your grid expects
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5)),
    ])

def sample_one_image_per_class(root_dir: str, num_classes: int = 10, seed: int = 42):
    rnd = random.Random(seed)
    paths = []
    for c in range(num_classes):
        class_dir = os.path.join(root_dir, str(c))
        candidates = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
            candidates += glob.glob(os.path.join(class_dir, ext))
        if not candidates:
            raise FileNotFoundError(f"No images found for class {c} in {class_dir}")
        paths.append(rnd.choice(candidates))
    return paths

def load_true_class_images(root_dir: str, input_size: int, device, num_classes: int = 10, seed: int = 42):
    tfm = build_input_transform(input_size)
    paths = sample_one_image_per_class(root_dir, num_classes=num_classes, seed=seed)
    imgs = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        imgs.append(tfm(im).unsqueeze(0))
    return torch.cat(imgs, dim=0).to(device)  # (10,3,H,W)



# =============================================================================
# def sample_one_image_per_class(root_dir, num_classes=10):
#     """
#     Assumes root_dir has subfolders named 0..9 (age classes).
#     Returns: list[str] of filepaths length num_classes (one per class)
#     """
#     paths = []
#     for c in range(num_classes):
#         class_dir = os.path.join(root_dir, str(c))
#         # pick first image found (you can randomize if you want)
#         candidates = sorted(
#             glob.glob(os.path.join(class_dir, "*.jpg")) +
#             glob.glob(os.path.join(class_dir, "*.jpeg")) +
#             glob.glob(os.path.join(class_dir, "*.png"))
#         )
#         if len(candidates) == 0:
#             raise FileNotFoundError(f"No images found for class {c} in {class_dir}")
#         paths.append(random.choice(candidates))
#     return paths
# 
# 
# def build_input_transform(input_size):
#     """
#     Produces tensors in [-1, 1] (matches your make_grid value_range and typical VQ-VAE setup).
#     """
#     return transforms.Compose([
#         transforms.Resize((input_size, input_size)),
#         transforms.ToTensor(),                          # [0,1]
#         transforms.Normalize((0.5, 0.5, 0.5),          # -> [-1,1]
#                              (0.5, 0.5, 0.5)),
#     ])
# 
# 
# @torch.no_grad()
# def plot_true_images_and_vqvae_recon(vqvae, cfg, split_root, out_path, num_classes=10):
#     """
#     Makes a 2-row grid:
#       Row 1: true/original images (one per class 0..9)
#       Row 2: VQ-VAE reconstructions of those same images
#     Saves figure to out_path.
#     """
#     device = next(vqvae.parameters()).device
#     tfm = build_input_transform(cfg.input_size)
# 
#     img_paths = sample_one_image_per_class(split_root, num_classes=num_classes)
# 
#     imgs = []
#     for p in img_paths:
#         im = Image.open(p).convert("RGB")
#         imgs.append(tfm(im).unsqueeze(0))
#     imgs = torch.cat(imgs, dim=0).to(device)  # (10,3,H,W)
# 
#     # VQ-VAE recon
#     quantized, _, _ = vqvae.encode(imgs)
#     recon = vqvae.decode(quantized)
# 
#     # Build 2-row grid: [true 10] + [recon 10] => 20 images, nrow=10
#     grid = make_grid(
#         torch.cat([imgs.cpu(), recon.cpu()], dim=0),
#         nrow=num_classes,
#         normalize=True,
#         value_range=(-1, 1),
#     )
# 
#     # Save (use matplotlib so we can title it)
#     grid_np = grid.permute(1, 2, 0).numpy()
#     fig, ax = plt.subplots(figsize=(num_classes * 2, 4))
#     ax.imshow(grid_np)
#     ax.axis("off")
#     ax.set_title("Top row: True images (one per age class 0–9) | Bottom row: VQ-VAE recon", fontsize=10)
#     fig.tight_layout()
#     fig.savefig(out_path, dpi=200)
#     plt.close(fig)
# =============================================================================



@torch.no_grad()
def reconstruct_with_transformer(vqvae_run_id, transformer_run_id, num_samples=5, mask_prob=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vqvae_run = neptune.init_run(with_id=vqvae_run_id, project='tns/audience-tuning')#os.getenv("NEPTUNE_PROJECT"))
    trans_run = neptune.init_run(with_id=transformer_run_id, project='tns/audience-tuning')#os.getenv("NEPTUNE_PROJECT"))

    # ----- load VQ-VAE config from Neptune -----
    params = {}
    for k, v in vqvae_run.get_structure().get("params", {}).items():
        try:
            params[k.split("/")[-1]] = vqvae_run[f"params/{k.split('/')[-1]}"].fetch_last()
        except Exception:
            pass

    cfg = Config(
        data_root=params.get("data_root", "./data"),
        train_subdir=params.get("train_subdir", "train"),
        val_subdir=params.get("val_subdir", "val"),
        input_size=int(params.get("input_size", 128)),
        bs=int(params.get("bs", 4)),
        epochs=int(params.get("epochs", 1)),
        lr=float(params.get("lr", 3e-4)),
        num_workers=int(params.get("num_workers", 4)),
        beta=float(params.get("beta", 0.25)),
        seed=int(params.get("seed", 42)),
        model_type=params.get("model_type", "EnhancedFlatVQVAE"),
        num_levels=int(params.get("num_levels", 1)),
        codebook_size=int(params.get("codebook_size", 32)),
        codebook_dim=int(params.get("codebook_dim", 16)),
        embed_dim=int(params.get("embed_dim", 16)),
        latent_channel=int(params.get("latent_channel", 144)),
        rotation_trick=bool(params.get("rotation_trick", False)),
        kmeans_init=bool(params.get("kmeans_init", False)),
        decay=float(params.get("decay", 0.99)),
        learnable_codebook=bool(params.get("learnable_codebook", False)),
        ema_update=bool(params.get("ema_update", True)),
        threshold_dead=None,
        world_size=1,
        local_rank=0,
        run_dir="./runs",
        torch_compile=False,
    )

    # ----- load VQ-VAE checkpoint -----
    vqvae = build_model(cfg).to(device)
    vqvae.eval()
    ckpt_path_vqvae = Path(f"vqvae/outputs/{vqvae_run_id}/samples/val_best.pt")
    vqvae_run["model/val_checkpoint_best"].download(str(ckpt_path_vqvae))
    vqvae.load_state_dict(torch.load(ckpt_path_vqvae, map_location=device))





    # ----- sample images + labels from validation set -----
    _, val_loader, _ = build_dataloaders(cfg)
    dataset = val_loader.dataset
    indices = random.sample(range(len(dataset)), num_samples)
    imgs = torch.cat([dataset[i][0].unsqueeze(0) for i in indices]).to(device)
    labels = torch.tensor([dataset[i][1] for i in indices]).to(device)

    # ----- encode with VQ-VAE -----
    quantized, _, ids = vqvae.encode(imgs)  # quantized: (B, C, H, W), ids: (B, H*W)
    B, C, H, W = quantized.shape
    zq = quantized.permute(0, 2, 3, 1).reshape(B, H * W, C)

    # mask latents
    zq_masked, target, mask, mask_prob = mask_latents(zq, ids, mask_prob)  # assuming your mask_latents returns mask_prob too

    # ----- transformer config from Neptune -----
    tparams = {}
    for k, v in trans_run.get_structure().get("params", {}).items():
        try:
            tparams[k.split("/")[-1]] = trans_run[f"params/{k.split('/')[-1]}"].fetch_last()
        except Exception:
            pass
    latent_dim = int(tparams.get("latent_dim", zq_masked.shape[-1]))
    label_dim  = int(tparams.get("label_dim", 16))
    num_heads  = int(tparams.get("num_heads", 3))
    model_dim = latent_dim + label_dim
    assert model_dim % num_heads == 0, f"model_dim={model_dim} not divisible by num_heads={num_heads}"
    print("latent_dim", latent_dim, "label_dim", label_dim, "model_dim", model_dim, "num_heads", num_heads)

    trans_model = MaskedLatentTransformer(
        latent_dim=int(tparams.get("latent_dim", zq_masked.shape[-1])),
        vocab_size=int(tparams.get("vocab_size", vqvae.vocab_size)),
        num_layers=int(tparams.get("num_layers", 6)),
        num_heads=int(tparams.get("num_heads", 3)),
        hidden_dim=int(tparams.get("hidden_dim", 512)),
        dropout=float(tparams.get("dropout", 0.1)),
        max_position_embeddings=zq.shape[1],
        num_classes=int(tparams.get("num_classes", 10)),
        label_dim=int(tparams.get("label_dim", 16)),
    ).to(device)


    ckpt_path_transformer = Path(f"transformer_{transformer_run_id}_best.pt")
    trans_run["model/val_checkpoint_best"].download(str(ckpt_path_transformer))
    trans_model.load_state_dict(torch.load(ckpt_path_transformer, map_location=device))
    trans_model.eval()

    # ----- reconstruct baseline images -----
    quant_masked = vqvae.quantize_b.get_codes_from_indices(ids.view(B, -1))
    quant_masked = quant_masked.view(B, H, W, C).permute(0, 3, 1, 2)
    recon_full = vqvae.decode(quantized)
    recon_masked = vqvae.decode(quant_masked)
    recon_masked_vis = overlay_mask_on_image(recon_masked, mask, H, W)






    # ----- generate 10 transformer completions with biases 0..9 -----
    recon_by_bias = []  # length 10, each (B, 3, H_img, W_img)

    for bias_id in range(10):
        # build bias labels (same bias for all samples in batch)
        labels_bias = torch.full(
            (B,),
            bias_id,
            dtype=torch.long,
            device=zq_masked.device
        )
        
        # transformer prediction using EMBEDDING
        with torch.no_grad():
            logits = trans_model(zq_masked, labels_bias)
            pred_indices = logits.argmax(dim=-1)
        del logits
        torch.cuda.empty_cache()



        completed_indices = ids.clone()
        completed_indices[mask] = pred_indices[mask]

        # 3) decode to image
        quant_completed = vqvae.quantize_b.get_codes_from_indices(
            completed_indices.view(B, -1))

        quant_completed = quant_completed.view(B, H, W, C).permute(0, 3, 1, 2)
        recon_completed = vqvae.decode(quant_completed)  # (B, 3, H_img, W_img)

        recon_by_bias.append(recon_completed.cpu())

# =============================================================================
#     # ----- build grid: 1 row per sample, 13 columns -----
#     rows = []
#     for i in range(num_samples):
#         # original, vqvae recon, masked image
#         row_imgs = [
#             imgs[i:i+1].cpu(),
#             recon_full[i:i+1].cpu(),
#             recon_masked_vis[i:i+1].cpu(),
#         ]
# 
#         # 10 transformer outputs with biases 0..9
#         true_label = int(labels[i].item())
#         for bias_id in range(10):
#             img_b = recon_by_bias[bias_id][i:i+1].clone()  # (1,3,H,W)
# 
#             # draw circle if this bias equals the true label
#             if bias_id == true_label:
#                 img_b[0] = draw_frame_on_image(img_b[0], color=(1.0, 0.0, 0.0), thickness=3)
# 
#             row_imgs.append(img_b)
# 
#         # concatenate 13 images in this row along batch dimension
#         row = torch.cat(row_imgs, dim=0)  # (13, 3, H, W)
#         rows.append(row)
# 
#     # stack rows: (num_samples*13, 3, H, W), then make grid with 13 columns
#     grid = make_grid(
#         torch.cat(rows, dim=0),
#         nrow=13,
#         normalize=True,
#         value_range=(-1, 1),
#     )
# =============================================================================


    rows = []
    for i in range(num_samples):
        row = [
            imgs[i:i+1].cpu(),
            recon_full[i:i+1].cpu(),
            recon_masked_vis[i:i+1].cpu(),
        ]
    
        true_label = int(labels[i].item())
        for bias_id in range(10):
            img_b = recon_by_bias[bias_id][i:i+1].clone()
            if bias_id == true_label:
                img_b[0] = draw_frame_on_image(
                    img_b[0], color=(1.0, 0.0, 0.0), thickness=3
                )
            row.append(img_b)
    
        rows.append(torch.cat(row, dim=0))  # (13,3,H,W)
        
        grid = make_grid(
        torch.cat(rows, dim=0),
        nrow=13,
        normalize=True,
        value_range=(-1, 1))
        







    out_dir = Path(f"transformer_outputs/{transformer_run_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    img_path = out_dir / "masked_completion_grid_biases.png"
    #save_image(grid, img_path)
    
    # grid: (3, H_total, W_total)
    grid_cpu = grid.cpu()
    C, H_total, W_total = grid_cpu.shape
    
    # convert to HWC and [0,1] for matplotlib
    img_np = grid_cpu.permute(1, 2, 0).numpy()
    
    fig, ax = plt.subplots(figsize=(13, num_samples * 2))
    
    ax.imshow(img_np)
    ax.axis("off")
    
    # ---- Annotation for what is being plotted ----
    ax.set_title(
        "Cols: [0] original, [1] VQ-VAE recon, [2] masked image, [3-12] transformer outputs (bias 0–9)\n"
        f"Masking probability: {mask_prob * 100:.1f}%",
        fontsize=10
    )


    
    # ---- Add text above the masked image (col index 2) ----
    num_cols = 13
    patch_width = W_total / num_cols
    
    masked_col = 2  # 0-based: 0=orig,1=recon,2=masked
    x_masked = (masked_col + 0.5) * patch_width
    y_text = 10  # pixels from top
    
    ax.text(
        x_masked,
        y_text,
        f"Masked ({mask_prob * 100:.1f}%)",
        color="yellow",
        fontsize=9,
        ha="center",
        va="top",
        bbox=dict(facecolor="black", alpha=0.6, pad=2),
    )
    
    fig.tight_layout()
    
    out_dir = Path(f"transformer_outputs/{transformer_run_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    img_path = out_dir / f"masked_completion_grid_biases_masking{mask_prob}.png"
    
    fig.savefig(img_path, dpi=200)
    plt.close(fig)

    print(f"Saved result to {img_path}")
    
    
    


    
# =============================================================================
#     # --- NEW: plot true images per age class 0..9 + their VQ-VAE recon ---
#     val_root = "/local/altamabp/UTKFace_dataset_subset_10000_structured"
#     out_true_vs_recon = out_dir / "true_vs_vqvae_recon_classes_0_9.png"
#     plot_true_images_and_vqvae_recon(vqvae, cfg, val_root, out_true_vs_recon, num_classes=10)
#     print(f"Saved true-vs-recon grid to {out_true_vs_recon}")
# =============================================================================


    trans_run.stop()
    vqvae_run.stop()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--vqvae_run", type=str, default="AUD-184", help="Neptune run ID for VQ-VAE model")
    parser.add_argument("--transformer_run", type=str, default="AUD-185", help="Neptune run ID for transformer model")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--mask_prob", type=float, default=0.9)
    args = parser.parse_args()

    reconstruct_with_transformer(args.vqvae_run, args.transformer_run, args.num_samples, args.mask_prob)