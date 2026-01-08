import os
import torch
import random
import numpy as np
from pathlib import Path
from torchvision.utils import make_grid
import neptune
import torch.nn.functional as F
import matplotlib.pyplot as plt

from masked_transformer.model import MaskedLatentTransformer
from masked_transformer.utils import mask_latents
from configs.config import Config
from model_def import build_model
from data import build_dataloaders
from data_labeled_flat import build_dataloaders_subclasses



# -------------------------
# Label -> VQ-VAE run mapping
# -------------------------
LABEL_TO_VQVAE_RUN = {
    0: "AUD-147",
    1: "AUD-149",
    2: "AUD-150",
    3: "AUD-151",
    4: "AUD-152",
    5: "AUD-153",
    6: "AUD-154",
    7: "AUD-155",
    8: "AUD-156",
    9: "AUD-157",
}


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

    # map color [0,1] -> [-1,1] to match your value_range
    mapped = [2 * c - 1 for c in color]
    mapped_t = torch.tensor(mapped, device=img.device).view(3, 1, 1)

    img[:, :thickness, :] = mapped_t
    img[:, -thickness:, :] = mapped_t
    img[:, :, :thickness] = mapped_t
    img[:, :, -thickness:] = mapped_t
    return img


def _fetch_cfg_from_neptune(run_id: str) -> Config:
    """
    Fetch VQ-VAE config params from ONE Neptune run.
    Assumes all class-specific VQ-VAEs share the same architecture/hparams.
    """
    vqvae_run = neptune.init_run(with_id=run_id, project="tns/audience-tuning", mode="read-only")

    params = {}
    for k, v in vqvae_run.get_structure().get("params", {}).items():
        try:
            key = k.split("/")[-1]
            params[key] = vqvae_run[f"params/{key}"].fetch_last()
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

    vqvae_run.stop()
    return cfg


def _load_vqvae_for_run(run_id: str, cfg: Config, device: torch.device):
    """Load one VQ-VAE checkpoint from a specific Neptune run."""
    run = neptune.init_run(with_id=run_id, project="tns/audience-tuning", mode="read-only")

    model = build_model(cfg).to(device)
    model.eval()

    ckpt_path = Path(f"vqvae/outputs/{run_id}_val_best.pt")
    run["model/val_checkpoint_best"].download(str(ckpt_path))
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    run.stop()
    return model


def _encode_by_class(imgs: torch.Tensor, labels: torch.Tensor, vqvaes: dict):
    """
    Encode each sample using the VQ-VAE corresponding to its class.
    Returns:
      quantized: (B, C, H_lat, W_lat)
      ids:       (B, H_lat*W_lat)
    """
    B = imgs.size(0)
    quant_list, ids_list = [None] * B, [None] * B

    for cls in labels.unique().tolist():
        cls = int(cls)
        idxs = (labels == cls).nonzero(as_tuple=False).squeeze(1)
        x = imgs[idxs]  # (n,3,H,W)
        q, _, ids = vqvaes[cls].encode(x)  # q: (n,C,Hl,Wl), ids: (n,L)

        for j, bi in enumerate(idxs.tolist()):
            quant_list[bi] = q[j : j + 1]
            ids_list[bi] = ids[j : j + 1]

    quantized = torch.cat(quant_list, dim=0)
    ids = torch.cat(ids_list, dim=0)
    return quantized, ids


def _decode_indices_by_class(ids: torch.Tensor, labels: torch.Tensor, vqvaes: dict, H_lat: int, W_lat: int, C_lat: int):
    """
    Decode indices -> images, using the class-specific VQ-VAE codebook + decoder.
    ids: (B, L)
    """
    B, L = ids.shape
    recon_list = [None] * B

    for cls in labels.unique().tolist():
        cls = int(cls)
        idxs = (labels == cls).nonzero(as_tuple=False).squeeze(1)
        ids_cls = ids[idxs]  # (n,L)

        # indices -> codes -> (n,C,Hl,Wl)
        codes = vqvaes[cls].quantize_b.get_codes_from_indices(ids_cls.view(ids_cls.size(0), -1))  # (n,L,C)
        q = codes.view(ids_cls.size(0), H_lat, W_lat, C_lat).permute(0, 3, 1, 2)  # (n,C,Hl,Wl)

        recon = vqvaes[cls].decode(q)  # (n,3,H,W)

        for j, bi in enumerate(idxs.tolist()):
            recon_list[bi] = recon[j : j + 1]

    return torch.cat(recon_list, dim=0)


@torch.no_grad()
def reconstruct_with_transformer(vqvae_cfg_run_id, transformer_run_id, num_samples=5, mask_prob=0.15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Get cfg once (assumed shared across class-specific VQ-VAEs) ----
    cfg = _fetch_cfg_from_neptune(vqvae_cfg_run_id)

    # ---- Load transformer run ----
    trans_run = neptune.init_run(with_id=transformer_run_id, project="tns/audience-tuning", mode="read-only")

    # ---- Sample images + labels from validation set ----
    _, val_loader, _ = build_dataloaders_subclasses(cfg) #build_dataloaders(cfg)
    dataset = val_loader.dataset
# =============================================================================
#     pick = random.sample(range(len(dataset)), num_samples)
# 
#     imgs = torch.cat([dataset[i][0].unsqueeze(0) for i in pick]).to(device)
#     labels = torch.tensor([dataset[i][1] for i in pick], device=device, dtype=torch.long)
# =============================================================================
    
########################################added    
    # ----- sample images from DIFFERENT classes -----
    # This picks one random example per class until num_samples is reached.
    # Max distinct classes = 10 for your setup.
# ----- sample images (prefer different classes, but allow repeats) -----
    max_classes = 10
    k = min(num_samples, max_classes)
    
    # Build index lists per class
    per_class = {c: [] for c in range(max_classes)}
    for i in range(len(dataset)):
        _, y = dataset[i]
        if int(y) in per_class:
            per_class[int(y)].append(i)
    
    available_classes = [c for c in range(max_classes) if len(per_class[c]) > 0]
    if len(available_classes) == 0:
        raise ValueError("No classes have data in val set.")
    
    # choose as many distinct classes as possible
    chosen_classes = random.sample(available_classes, k=min(num_samples, len(available_classes)))
    
    # if we still need more samples, fill the rest by reusing available classes
    while len(chosen_classes) < num_samples:
        chosen_classes.append(random.choice(available_classes))
    
    indices = [random.choice(per_class[c]) for c in chosen_classes]
    
    imgs = torch.cat([dataset[i][0].unsqueeze(0) for i in indices]).to(device)
    labels = torch.tensor([dataset[i][1] for i in indices], device=device)


###################################

    # ---- Load VQ-VAE models needed for these labels ----
    vqvaes = {}
    for cls in labels.unique().tolist():
        cls = int(cls)
        run_id = LABEL_TO_VQVAE_RUN[cls]
        print(f"Loading VQ-VAE for class {cls} from run {run_id}")
        vqvaes[cls] = _load_vqvae_for_run(run_id, cfg, device)

    # ---- Encode per-sample using its class VQ-VAE ----
    quantized, ids = _encode_by_class(imgs, labels, vqvaes)  # quantized: (B,C,Hl,Wl), ids: (B,L)
    B, C_lat, H_lat, W_lat = quantized.shape
    zq = quantized.permute(0, 2, 3, 1).reshape(B, H_lat * W_lat, C_lat)

    # ---- Mask latents (use function argument mask_prob, not args.mask_prob) ----
    zq_masked, target, mask, mask_prob_used = mask_latents(zq, ids, mask_prob)

    # ---- Transformer config from Neptune ----
    tparams = {}
    for k, v in trans_run.get_structure().get("params", {}).items():
        try:
            key = k.split("/")[-1]
            tparams[key] = trans_run[f"params/{key}"].fetch_last()
        except Exception:
            pass

    # Use any VQ-VAE vocab_size (assumes all codebooks have same size)
    any_vqvae = vqvaes[int(labels[0].item())]

    trans_model = MaskedLatentTransformer(
        embed_dim=int(tparams.get("embed_dim", zq_masked.shape[-1] + 10 + 1)),
        vocab_size=int(tparams.get("vocab_size", any_vqvae.vocab_size)),
        num_layers=int(tparams.get("num_layers", 6)),
        num_heads=int(tparams.get("num_heads", 3)),
        hidden_dim=int(tparams.get("hidden_dim", 64)),
        dropout=float(tparams.get("dropout", 0.1)),
        max_position_embeddings=zq.shape[1],
    ).to(device)

    ckpt_path_transformer = Path(f"transformer_{transformer_run_id}_best.pt")
    trans_run["model/val_checkpoint_best"].download(str(ckpt_path_transformer))
    trans_model.load_state_dict(torch.load(ckpt_path_transformer, map_location=device))
    trans_model.eval()

    # ---- Reconstruct baseline masked image (optional visualization) ----
    ids_masked = ids.clone()
    ids_masked[mask] = 0  # choose 0 as masked token index for visualization

    recon_masked = _decode_indices_by_class(ids_masked, labels, vqvaes, H_lat, W_lat, C_lat)
    recon_masked_vis = overlay_mask_on_image(recon_masked, mask, H_lat, W_lat)

    # ---- Generate 10 transformer completions with biases 0..9 ----
    recon_by_bias = []  # length 10, each (B, 3, H_img, W_img)

    for bias_id in range(10):
        idx_t = torch.full((B,), bias_id, dtype=torch.long, device=device)  # (B,)
        N, T, _ = zq_masked.shape
        idx_t_exp = idx_t.unsqueeze(1).expand(N, T)  # (B, T)

        one_hot_labels = torch.nn.functional.one_hot(idx_t_exp, num_classes=10).to(dtype=zq_masked.dtype)
        masked_exp_quantizes = torch.cat([zq_masked, one_hot_labels], dim=2)  # (B, T, C+10)

        # If you trained with mask feature, switch to:
        # mask_feat = mask.to(masked_exp_quantizes.dtype).unsqueeze(-1)
        # masked_exp_quantizes = torch.cat([masked_exp_quantizes, mask_feat], dim=-1)

        logits = trans_model(masked_exp_quantizes)          # (B, T, vocab)
        pred_indices = logits.argmax(dim=-1)                # (B, T)

        completed_indices = ids.clone()
        completed_indices[mask] = pred_indices[mask]        # fill in masked positions only

        recon_completed = _decode_indices_by_class(completed_indices, labels, vqvaes, H_lat, W_lat, C_lat)
        recon_by_bias.append(recon_completed.cpu())

    # ---- Build grid: 1 row per sample, 13 columns ----
    # Cols: [0] original, [1] masked image, [2-11] transformer outputs (bias 0..9), [12] (optional) unused
    # If you want exactly 12 columns, remove one item and set nrow accordingly.
    rows = []
    for i in range(num_samples):
        row_imgs = [
            imgs[i:i+1].cpu(),
            recon_masked_vis[i:i+1].cpu(),
        ]

        true_label = int(labels[i].item())
        for bias_id in range(10):
            img_b = recon_by_bias[bias_id][i:i+1].clone()
            if bias_id == true_label:
                img_b[0] = draw_frame_on_image(img_b[0], color=(1.0, 0.0, 0.0), thickness=3)
            row_imgs.append(img_b)

        row = torch.cat(row_imgs, dim=0)  # (12, 3, H, W)
        rows.append(row)

    grid = make_grid(
        torch.cat(rows, dim=0),
        nrow=12,
        normalize=True,
        value_range=(-1, 1),
    )

    out_dir = Path(f"transformer_outputs/{transformer_run_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    img_path = out_dir / "masked_completion_grid_biases_class_vqvae.png"

    grid_cpu = grid.cpu()
    img_np = grid_cpu.permute(1, 2, 0).numpy()

    fig, ax = plt.subplots(figsize=(12, num_samples * 2))
    ax.imshow(img_np)
    ax.axis("off")
    ax.set_title(
        "Cols: [0] original, [1] masked image, [2-11] transformer outputs (bias 0–9)\n"
        f"Masking probability: {mask_prob_used * 100:.1f}%",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(img_path, dpi=200)
    plt.close(fig)

    print(f"Saved result to {img_path}")

    trans_run.stop()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # used ONLY to fetch cfg params (assumed common across class VQ-VAEs)
    parser.add_argument("--vqvae_run", type=str, default="AUD-145", help="Neptune run ID to fetch VQ-VAE cfg params")
    parser.add_argument("--transformer_run", type=str, default="AUD-158", help="Neptune run ID for transformer model")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--mask_prob", type=float, default=0.5)
    args = parser.parse_args()

    reconstruct_with_transformer(args.vqvae_run, args.transformer_run, args.num_samples, args.mask_prob)
