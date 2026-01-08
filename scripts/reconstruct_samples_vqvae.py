import os
import argparse
import torch
from torchvision.utils import make_grid, save_image
from pathlib import Path
import random

import neptune
from configs.config import Config
from model_def import build_model
from data import build_dataloaders


def reconstruct_samples_from_neptune(run_id: str, num_samples: int = 5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = neptune.init_run(with_id=run_id, project=os.getenv("NEPTUNE_PROJECT"))
    print(f"Neptune run: {run_id}")
    params = {}
    structure = run.get_structure()
    for k, v in structure.get("params", {}).items():
        try:
            params[k.split("/")[-1]] = run[f"params/{k.split('/')[-1]}"].fetch_last()
        except Exception:
            pass

    # === Build config and model ===
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
        latent_channel=int(params.get("latent_channel", 32)),
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

    model = build_model(cfg).to(device)
    model.eval()
    out_dir = Path(f"vqvae/outputs/{run_id}/samples")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "val_best.pt"
    run["model/val_checkpoint_best"].download(str(ckpt_path))
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    _, val_loader, _ = build_dataloaders(cfg)
    dataset = val_loader.dataset

    #random samples
    indices = random.sample(range(len(dataset)), num_samples)
    imgs = [dataset[i][0].unsqueeze(0) for i in indices]
    imgs = torch.cat(imgs).to(device)

    with torch.no_grad():
        recons, _, _ = model(imgs)


    grid = make_grid(torch.cat([imgs.cpu(), recons.cpu()]), nrow=num_samples, normalize=True, value_range=(-1, 1))
    img_path = out_dir / "reconstructions_vqvae.png"
    save_image(grid, img_path)
    print(f"Saved reconstruction image to {img_path}")

    #Upload to Neptune
    #run["artifacts/reconstruction_image"].upload(str(img_path))
    run.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstruct samples from a trained VQ-VAE Neptune run.")
    parser.add_argument("--run_id", type=str, default="AUD-91", help="Neptune run ID")
    parser.add_argument("--num_samples", type=int, default=5)
    args = parser.parse_args()
    reconstruct_samples_from_neptune(args.run_id, args.num_samples)
