import os
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

import neptune
from configs.config import Config
from model_def import build_model
from data import build_dataloaders
import inspect

class DatasetWithPaths(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

        # ImageFolder
        if hasattr(dataset, "samples"):
            self.samples = dataset.samples
        # FlatImageFolder
        elif hasattr(dataset, "samples"):
            self.samples = dataset.samples
        else:
            raise RuntimeError("Dataset does not expose samples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]

        # ImageFolder: (path, class)
        if isinstance(self.samples[idx], tuple):
            path = self.samples[idx][0]
        # FlatImageFolder: Path
        else:
            path = str(self.samples[idx])

        return img, label, path


def extract_latents_from_neptune(run_id: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = neptune.init_run(with_id=run_id, project=os.getenv("NEPTUNE_PROJECT"))
    print(f"Connected to Neptune run: {run_id}")
    params = {}
    structure = run.get_structure()
    for k, v in structure.get("params", {}).items():
        try:
            params[k.split("/")[-1]] = run[f"params/{k.split('/')[-1]}"].fetch_last()
        except Exception:
            pass

    # === Build config and model ===
    # config_args = inspect.signature(Config).parameters
    # filtered_params = {k: v for k, v in params.items() if k in config_args}
    # cfg = Config(**filtered_params)
    

    # cfg = Config(**params)
    cfg = Config(**{k: v for k, v in {
    "data_root": params.get("data_root", "./data"), "train_subdir": params.get("train_subdir", "train"), "val_subdir": params.get("val_subdir", "val"),"input_size": int(params.get("input_size", 256)), "bs": int(params.get("bs", 4)), "epochs": int(params.get("epochs", 1)), "lr": float(params.get("lr", 3e-4)),"num_workers": int(params.get("num_workers", 4)), "beta": float(params.get("beta", 0.25)), "seed": int(params.get("seed", 42)), "model_type": params.get("model_type", "EnhancedFlatVQVAE"),"num_levels": int(params.get("num_levels", 1)), "codebook_size": int(params.get("codebook_size", 512)), "codebook_dim": int(params.get("codebook_dim", 64)), "embed_dim": int(params.get("embed_dim", 64)), "latent_channel": int(params.get("latent_channel", 144)), "rotation_trick": bool(params.get("rotation_trick", False)), "kmeans_init": bool(params.get("kmeans_init", False)), "decay": float(params.get("decay", 0.99)),"learnable_codebook": bool(params.get("learnable_codebook", False)), "ema_update": bool(params.get("ema_update", True)), "threshold_dead": None, "world_size": 1, "local_rank": 0,"run_dir": "./runs", "torch_compile": False
}.items()})

    model = build_model(cfg).to(device)
    model.eval()
    out_dir = Path(f"/local/altamabp/audience_tuning-gem/vqvae/{run_id}")#(f"/local/reyhasjb/aud_tuning/vqvae/{run_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "vqvae_val_best.pt"
    print("Downloading best validation checkpoint")
    run["model/val_checkpoint_best"].download(str(ckpt_path))
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    train_loader, val_loader, _ = build_dataloaders(cfg)
    
    # wrap datasets so filenames are returned
    train_loader.dataset = DatasetWithPaths(train_loader.dataset)
    val_loader.dataset   = DatasetWithPaths(val_loader.dataset)


    def extract(loader):
        all_latents = []
        all_indices = []
        all_labels = []
        all_img_names = []
    
        for images, batch_labels, paths in tqdm(loader, desc="Extracting latents + names"):
            images = images.to(device, non_blocking=True)
    
            with torch.no_grad():
                quant_b, _, id_b = model.encode(images)
    
            all_latents.append(quant_b.cpu().numpy())
            all_indices.append(id_b.cpu().numpy())
            all_labels.append(batch_labels.cpu().numpy())
    
            all_img_names.extend([os.path.basename(p) for p in paths])
    
        return (
            np.concatenate(all_latents, axis=0),
            np.concatenate(all_indices, axis=0),
            np.concatenate(all_labels, axis=0),
            np.array(all_img_names),
        )



    latents_dir = out_dir / "latents"
    latents_dir.mkdir(parents=True, exist_ok=True)
    print("Validation: ")
    val_latents, val_indices, val_labels, val_img_names = extract(val_loader)
    print("Saving val latents and labels...")
    
    np.save(latents_dir / "val_latents.npy", val_latents)
    np.save(latents_dir / "val_indices.npy", val_indices)
    np.save(latents_dir / "val_labels.npy", val_labels)
    np.save(latents_dir / "val_img_names.npy", val_img_names)

    
    print("Train: ")
    train_latents, train_indices, train_labels, train_img_names = extract(train_loader)

    print("Saving train latents and labels...")    
    np.save(latents_dir / "train_latents.npy", train_latents)
    np.save(latents_dir / "train_indices.npy", train_indices)
    np.save(latents_dir / "train_labels.npy", train_labels)
    np.save(latents_dir / "train_img_names.npy", train_img_names)


    run.stop()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract latent representations from a trained VQ-VAE Neptune run.")
    parser.add_argument("--run_id", type=str, default="AUD-184", help="Neptune run ID")
    args = parser.parse_args()
    extract_latents_from_neptune(args.run_id)
