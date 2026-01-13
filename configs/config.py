import os
import re
import socket
from dataclasses import dataclass
from typing import Optional, Any, Dict

from dotenv import load_dotenv
import yaml
import numpy as np


# =========================
# Dataclasses
# =========================
@dataclass
class Config:
    # data
    data_root: str
    train_subdir: str
    val_subdir: str
    input_size: int

    # training
    bs: int
    epochs: int
    lr: float
    num_workers: int
    beta: float
    seed: int

    # model
    model_type: str
    num_levels: int
    codebook_size: int
    codebook_dim: int
    embed_dim: Optional[int]
    latent_channel: int
    rotation_trick: bool
    kmeans_init: bool
    decay: float
    learnable_codebook: bool
    ema_update: bool
    threshold_dead: Optional[int]

    # system
    world_size: int
    local_rank: int
    run_dir: str
    torch_compile: bool


@dataclass
class TransformerConfig:
    # data
    train_latents: str
    train_indices: str
    train_labels: str
    val_latents: str
    val_indices: str
    val_labels: str

    # optional conditioning data (classifier outputs)
    train_probs: Optional[str]
    val_probs: Optional[str]

    # model/conditioning
    embed_dim: int           # kept for backward compatibility / debugging
    vocab_size: int
    num_layers: int
    num_heads: int
    hidden_dim: int
    dropout: float
    mask_prob: float

    num_classes: int         # new
    label_dim: int           # new
    p_use_probs: float       # new
    probs_temp: float        # new (optional)

    # training
    bs: int
    epochs: int
    lr: float
    scheduler: str
    warmup_steps: int
    seed: int
    loss_masked: bool
    num_workers: int         # new (so dataloader can use cfg.num_workers)

    # system
    run_dir: str
    torch_compile: bool


# =========================
# Helpers
# =========================
def _b(s: Optional[str], default: bool) -> bool:
    if s is None:
        return default
    return s.lower() in ("1", "true", "yes", "y", "on")


def _maybe_int(s: Optional[str]) -> Optional[int]:
    return int(s) if (s is not None and s != "" and str(s).lower() != "null") else None


def _load_env():
    """
    Load environment variables from:
      1) ENV_FILE if provided
      2) cluster-specific env file based on hostname
      3) .env fallback
    """
    cluster_env = "envs/ini.env"
    if load_dotenv is None:
        raise RuntimeError("Install python-dotenv to load ENV_FILE or .env")

    env_file = os.getenv("ENV_FILE")
    if env_file and os.path.exists(env_file):
        load_dotenv(dotenv_path=env_file, override=False)
        return

    hostname = socket.gethostname().lower()
    if any(x in hostname for x in ("gpu01", "gpu02", "gpu03")):
        cluster_env = "envs/ini.env"
    elif "juwels" in hostname:
        cluster_env = "envs/juwels_booster.env"

    if os.path.exists(cluster_env):
        load_dotenv(dotenv_path=cluster_env, override=False)
    elif os.path.exists(".env"):
        load_dotenv(dotenv_path=".env", override=False)


def _interpolate_dict(d: dict) -> dict:
    """
    Recursively interpolates ${...} placeholders in YAML configs.
    Supports:
      - environment variables (highest priority)
      - dotted YAML references (e.g., ${data.run_id})
    """
    pattern = re.compile(r"\$\{([a-zA-Z0-9_.]+)\}")

    def _resolve_ref(key_path: str):
        env_val = os.getenv(key_path)
        if env_val is not None:
            return env_val

        cur: Any = d
        for part in key_path.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return None
        # Avoid substituting dicts
        return cur if not isinstance(cur, dict) else None

    def _expand_once(obj):
        if isinstance(obj, str):
            matches = pattern.findall(obj)
            for match in matches:
                ref = _resolve_ref(match)
                if ref is not None:
                    obj = obj.replace(f"${{{match}}}", str(ref))
            return obj
        if isinstance(obj, dict):
            return {k: _expand_once(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_expand_once(v) for v in obj]
        return obj

    # Iterate a few times for nested references
    for _ in range(5):
        expanded = _expand_once(d)
        if str(expanded) == str(d):
            break
        d = expanded

    return d


def _load_config_file() -> dict:
    cfg_path = os.getenv("CONFIG_FILE")
    if not cfg_path:
        raise RuntimeError("CONFIG_FILE environment variable must point to a YAML config file.")

    with open(cfg_path, "r") as f:
        text = os.path.expandvars(f.read())
        cfg = yaml.safe_load(text) or {}
        cfg = _interpolate_dict(cfg)
        return cfg


def _get(d: dict, path: str, default=None):
    """dot-path getter: example: 'training.bs' returns d['training']['bs'] if present."""
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _get_any(d: dict, keys: list, default=None):
    """
    Tries multiple keys. Supports both:
      - dot paths: "data.train_latents"
      - top-level: "train_latents"
    """
    for k in keys:
        if "." in k:
            v = _get(d, k, None)
        else:
            v = d.get(k, None) if isinstance(d, dict) else None
        if v is not None:
            return v
    return default


def _require(name: str, value: Any):
    if value is None or (isinstance(value, str) and value.strip() == ""):
        raise ValueError(
            f"Missing required config value '{name}'. "
            f"Set it in YAML (nested under data/model/training/system or flat) or via env vars."
        )
    return value


# =========================
# VQ-VAE Config (unchanged behavior)
# =========================
def get_config() -> Config:
    _load_env()
    f = _load_config_file()

    # ----- data -----
    data_root    = os.getenv("DATA_ROOT", _get(f, "data.data_root", "./data"))
    train_subdir = os.getenv("TRAIN_SUBDIR", _get(f, "data.train_subdir", "train"))
    val_subdir   = os.getenv("VAL_SUBDIR",   _get(f, "data.val_subdir",   "val"))
    input_size   = int(os.getenv("INPUT_SIZE", _get(f, "data.input_size", 256)))

    # ----- training -----
    bs          = int(os.getenv("BS",          _get(f, "training.bs", 4)))
    epochs      = int(os.getenv("EPOCHS",      _get(f, "training.epochs", 150)))
    lr          = float(os.getenv("LR",        _get(f, "training.lr", 3e-4)))
    num_workers = int(os.getenv("NUM_WORKERS", _get(f, "training.num_workers", 4)))
    beta        = float(os.getenv("BETA",      _get(f, "training.beta", 0.25)))
    seed        = int(os.getenv("SEED",        _get(f, "training.seed", 42)))

    # ----- model -----
    model_type       = os.getenv("MODEL_TYPE",    _get(f, "model.model_type", "UnconditionedHVQVAE"))
    num_levels       = int(os.getenv("NUM_LEVELS",_get(f, "model.num_levels", 1)))
    codebook_size    = int(os.getenv("CODEBOOK_SIZE", _get(f, "model.codebook_size", 512)))
    codebook_dim     = int(os.getenv("CODEBOOK_DIM",  _get(f, "model.codebook_dim", 1)))
    embed_dim        = _maybe_int(os.getenv("EMBED_DIM")) if os.getenv("EMBED_DIM") is not None else _get(f, "model.embed_dim", None)
    embed_dim        = None if (embed_dim == "" or str(embed_dim).lower() == "none") else embed_dim
    latent_channel   = int(os.getenv("LATENT_CHANNEL", _get(f, "model.latent_channel", 128)))
    rotation_trick   = _b(os.getenv("ROTATION_TRICK"), _get(f, "model.rotation_trick", False))
    kmeans_init      = _b(os.getenv("KMEANS_INIT"),    _get(f, "model.kmeans_init", False))
    decay            = float(os.getenv("DECAY",        _get(f, "model.decay", 0.99)))
    learnable_codebook = _b(os.getenv("LEARNABLE_CODEBOOK"), _get(f, "model.learnable_codebook", False))
    ema_update       = _b(os.getenv("EMA_UPDATE"),     _get(f, "model.ema_update", True))
    threshold_dead   = _maybe_int(os.getenv("THRESHOLD_DEAD")) if os.getenv("THRESHOLD_DEAD") is not None else _get(f, "model.threshold_dead", None)

    # ----- system / launcher -----
    world_size    = int(os.getenv("WORLD_SIZE", "1"))
    local_rank    = int(os.getenv("LOCAL_RANK", "0"))
    run_dir       = os.getenv("RUN_DIR", _get(f, "system.run_dir", "./runs"))
    torch_compile = _b(os.getenv("TORCH_COMPILE"), _get(f, "system.torch_compile", False))

    return Config(
        data_root=data_root, train_subdir=train_subdir, val_subdir=val_subdir, input_size=input_size,
        bs=bs, epochs=epochs, lr=lr, num_workers=num_workers, beta=beta, seed=seed,
        model_type=model_type, num_levels=num_levels, codebook_size=codebook_size, codebook_dim=codebook_dim,
        embed_dim=embed_dim, latent_channel=latent_channel, rotation_trick=rotation_trick, kmeans_init=kmeans_init,
        decay=decay, learnable_codebook=learnable_codebook, ema_update=ema_update, threshold_dead=threshold_dead,
        world_size=world_size, local_rank=local_rank, run_dir=run_dir, torch_compile=torch_compile
    )


# =========================
# Transformer Config (UPDATED)
# =========================
def get_transformer_config() -> TransformerConfig:
    _load_env()
    f = _load_config_file()

    # --- paths (accept nested or flat) ---
    train_latent_path = _get_any(f, ["data.train_latents", "train_latents"])
    train_indices     = _get_any(f, ["data.train_indices", "train_indices"])
    train_labels      = _get_any(f, ["data.train_labels",  "train_labels"])
    val_latents       = _get_any(f, ["data.val_latents",   "val_latents"])
    val_indices       = _get_any(f, ["data.val_indices",   "val_indices"])
    val_labels        = _get_any(f, ["data.val_labels",    "val_labels"])

    # optional conditioning arrays
    train_probs       = _get_any(f, ["data.train_probs", "train_probs"], default=None)
    val_probs         = _get_any(f, ["data.val_probs",   "val_probs"],   default=None)

    # hard error early (prevents np.load(None))
    _require("train_latents", train_latent_path)
    _require("train_indices", train_indices)
    _require("train_labels",  train_labels)
    _require("val_latents",   val_latents)
    _require("val_indices",   val_indices)
    _require("val_labels",    val_labels)

    # --- auto-detect embed_dim from train latents (kept for backward compatibility) ---
    embed_dim_detected: Optional[int] = None
    try:
        latent_sample = np.load(train_latent_path, mmap_mode="r")
        if latent_sample.ndim == 3:
            embed_dim_detected = int(latent_sample.shape[-1])  # (N, T, D)
            print(f"[config] Detected embed_dim = {embed_dim_detected} from shape (N, T, D)")
        elif latent_sample.ndim == 4:
            embed_dim_detected = int(latent_sample.shape[1])   # (N, D, H, W)
            print(f"[config] Detected embed_dim = {embed_dim_detected} from shape (N, D, H, W)")
        else:
            print(f"[config] Warning: unexpected latent shape {latent_sample.shape}")
    except Exception as e:
        print(f"[config] Could not auto-detect embed_dim: {e}")

    # --- model hyperparams (nested or flat) ---
    embed_dim = int(_get_any(f, ["model.embed_dim", "embed_dim"], default=(embed_dim_detected or 8)))
    vocab_size = int(_get_any(f, ["model.vocab_size", "vocab_size"], default=64))
    num_layers = int(_get_any(f, ["model.num_layers", "num_layers"], default=6))
    num_heads  = int(_get_any(f, ["model.num_heads",  "num_heads"],  default=3))
    hidden_dim = int(_get_any(f, ["model.hidden_dim", "hidden_dim"], default=512))
    dropout    = float(_get_any(f, ["model.dropout",  "dropout"],    default=0.1))
    mask_prob  = float(_get_any(f, ["model.mask_prob","mask_prob"],  default=0.15))

    # --- conditioning hyperparams (new; nested or flat) ---
    num_classes = int(_get_any(f, ["conditioning.num_classes", "num_classes"], default=10))
    label_dim   = int(_get_any(f, ["conditioning.label_dim",   "label_dim"],   default=8))
    p_use_probs = float(_get_any(f, ["conditioning.p_use_probs","p_use_probs"], default=0.0))
    probs_temp  = float(_get_any(f, ["conditioning.probs_temp","probs_temp"], default=1.0))

    # --- training hyperparams (nested or flat) ---
    bs          = int(_get_any(f, ["training.bs", "bs"], default=32))
    epochs      = int(_get_any(f, ["training.epochs", "epochs"], default=50))
    lr          = float(_get_any(f, ["training.lr", "lr"], default=1e-4))
    scheduler   = str(_get_any(f, ["training.scheduler", "scheduler"], default="linear_warmup"))
    warmup_steps= int(_get_any(f, ["training.warmup_steps", "warmup_steps"], default=1000))
    seed        = int(_get_any(f, ["training.seed", "seed"], default=42))
    loss_masked = bool(_get_any(f, ["training.loss_masked", "loss_masked"], default=True))
    num_workers = int(_get_any(f, ["training.num_workers", "num_workers"], default=4))

    # --- system ---
    run_dir       = str(_get_any(f, ["system.run_dir", "run_dir"], default="./runs"))
    torch_compile = _b(os.getenv("TORCH_COMPILE"), _get_any(f, ["system.torch_compile", "torch_compile"], default=False))

    return TransformerConfig(
        train_latents=train_latent_path,
        train_indices=train_indices,
        train_labels=train_labels,
        val_latents=val_latents,
        val_indices=val_indices,
        val_labels=val_labels,
        train_probs=train_probs,
        val_probs=val_probs,

        embed_dim=embed_dim,
        vocab_size=vocab_size,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        dropout=dropout,
        mask_prob=mask_prob,

        num_classes=num_classes,
        label_dim=label_dim,
        p_use_probs=p_use_probs,
        probs_temp=probs_temp,

        bs=bs,
        epochs=epochs,
        lr=lr,
        scheduler=scheduler,
        warmup_steps=warmup_steps,
        seed=seed,
        loss_masked=loss_masked,
        num_workers=num_workers,

        run_dir=run_dir,
        torch_compile=torch_compile,
    )
