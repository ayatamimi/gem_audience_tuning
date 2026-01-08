# reconstruct_masked_with_transformer.py
import os
import torch
import random
import numpy as np
from pathlib import Path
from torchvision.utils import make_grid, save_image
import neptune
import torch.nn.functional as F

from masked_transformer.model import MaskedLatentTransformer
from vqvae.flat.flat_models import EnhancedFlatVQVAE
from masked_transformer.utils import mask_latents
from configs.config import Config
from model_def import build_model
from data import build_dataloaders

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

@torch.no_grad()
def reconstruct_with_transformer(vqvae_run_id, transformer_run_id, num_samples=5, mask_prob=0.15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vqvae_run = neptune.init_run(with_id=vqvae_run_id, project=os.getenv("NEPTUNE_PROJECT"))
    trans_run = neptune.init_run(with_id=transformer_run_id, project=os.getenv("NEPTUNE_PROJECT"))

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

    vqvae = build_model(cfg).to(device)
    vqvae.eval()
    ckpt_path_vqvae = Path(f"vqvae/outputs/{vqvae_run_id}/samples/val_best.pt")
    vqvae_run["model/val_checkpoint_best"].download(str(ckpt_path_vqvae))
    vqvae.load_state_dict(torch.load(ckpt_path_vqvae, map_location=device))

    _, val_loader, _ = build_dataloaders(cfg)
    dataset = val_loader.dataset
    indices = random.sample(range(len(dataset)), num_samples)
    imgs = torch.cat([dataset[i][0].unsqueeze(0) for i in indices]).to(device)
    labels = torch.tensor([dataset[i][1] for i in indices]).to(device)

    quantized, _, ids = vqvae.encode(imgs)
    B, C, H, W = quantized.shape
    zq = quantized.permute(0, 2, 3, 1).reshape(B, H * W, C)
    zq_masked, target, mask = mask_latents(zq, ids, mask_prob)


##### add bias /and mask feat to quantizes
    idx_t = torch.as_tensor(labels, dtype=torch.long, device=zq_masked.device)  # (N,)
    
    N, T, _ = zq_masked.shape
    idx_t_exp = idx_t.unsqueeze(1).expand(N, T)                         # (N, T)
    
    one_hot_labels = torch.nn.functional.one_hot(idx_t_exp, num_classes=10).to(dtype=zq_masked.dtype)  # (N,T,C)
    
    masked_exp_quantizes = torch.cat([zq_masked, one_hot_labels], dim=2)
    
    device = masked_exp_quantizes.device
    dtype  = masked_exp_quantizes.dtype
    
   # 1) NumPy -> Torch, cast to float (1.0 masked, 0.0 unmasked)
    mask_feat = torch.as_tensor(mask, device=device).to(dtype)   # (N, T)

    # 2) Add feature axis
    mask_feat = mask_feat.unsqueeze(-1)                                 # (N, T, 1)

    # 3) Concatenate
    masked_exp_mask_feat_quantizes = torch.cat([masked_exp_quantizes, mask_feat], dim=-1) 
#####

    tparams = {}
    for k, v in trans_run.get_structure().get("params", {}).items():
        try:
            tparams[k.split("/")[-1]] = trans_run[f"params/{k.split('/')[-1]}"].fetch_last()
        except Exception:
            pass

    trans_model = MaskedLatentTransformer(
        embed_dim=int(tparams.get("embed_dim", C)),
        vocab_size=int(tparams.get("vocab_size", vqvae.vocab_size)),
        num_layers=int(tparams.get("num_layers", 6)),
        num_heads=int(tparams.get("num_heads", 3)),
        hidden_dim=int(tparams.get("hidden_dim", 27)),
        dropout=int(tparams.get("droupout", 0.1)),
        max_position_embeddings=zq.shape[1]
    ).to(device)

    ckpt_path_transformer = Path(f"transformer_{transformer_run_id}_best.pt")
    trans_run["model/val_checkpoint_best"].download(str(ckpt_path_transformer))
    trans_model.load_state_dict(torch.load(ckpt_path_transformer, map_location=device))
    trans_model.eval()

    logits = trans_model(masked_exp_mask_feat_quantizes)#(masked_exp_quantizes)#(zq_masked)
    pred_indices = logits.argmax(dim=-1)
    completed_indices = ids.clone()
    completed_indices[mask] = pred_indices[mask]

    # Recostruct
    quant_masked = vqvae.quantize_b.get_codes_from_indices(ids.view(B, -1))
    quant_completed = vqvae.quantize_b.get_codes_from_indices(completed_indices.view(B, -1))
    quant_masked = quant_masked.view(B, H, W, C).permute(0, 3, 1, 2)
    quant_completed = quant_completed.view(B, H, W, C).permute(0, 3, 1, 2)
    recon_full = vqvae.decode(quantized)
    recon_masked = vqvae.decode(quant_masked)
    recon_completed = vqvae.decode(quant_completed)
    recon_masked_vis = overlay_mask_on_image(recon_masked, mask, H, W)

    #Visulaize
    rows = []
    for i in range(num_samples):
        row = torch.cat([
            imgs[i:i+1].cpu(),
            recon_full[i:i+1].cpu(),
            recon_masked_vis[i:i+1].cpu(),  # use masked VIS version
            recon_completed[i:i+1].cpu()
        ], dim=0)
        rows.append(row)
    grid = make_grid(torch.cat(rows), nrow=4, normalize=True, value_range=(-1, 1))

    out_dir = Path(f"transformer_outputs/{transformer_run_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    img_path = out_dir / "masked_completion_grid.png"
    save_image(grid, img_path)
    print(f"Saved result to {img_path}")
    #trans_run["artifacts/masked_completion_grid_50mask"].upload(str(img_path))
    trans_run.stop()
    vqvae_run.stop()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--vqvae_run", type=str, default="AUD-68", help="Neptune run ID for VQ-VAE model")
    parser.add_argument("--transformer_run", type=str, default="AUD-84", help="Neptune run ID for transformer model")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--mask_prob", type=float, default=0.5)
    args = parser.parse_args()

    reconstruct_with_transformer(args.vqvae_run, args.transformer_run, args.num_samples, args.mask_prob)
