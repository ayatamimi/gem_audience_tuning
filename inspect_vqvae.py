import argparse
import inspect
from collections import defaultdict

import torch
import torch.nn as nn

from configs.config import get_config
from model_def import build_model


def load_checkpoint(model: nn.Module, ckpt_path: str, device: str = "cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)

    # Support common checkpoint formats
    state = None
    for k in ["state_dict", "model", "model_state_dict", "module", "net"]:
        if k in ckpt and isinstance(ckpt[k], dict):
            state = ckpt[k]
            break
    if state is None and isinstance(ckpt, dict):
        # maybe the dict itself is a state_dict
        state = ckpt

    # Strip possible DDP "module." prefix
    cleaned = {}
    for k, v in state.items():
        nk = k[len("module."):] if k.startswith("module.") else k
        cleaned[nk] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    return missing, unexpected


def print_conv_specs(model: nn.Module):
    print("\n" + "=" * 80)
    print("CONV / CONVTRANSPOSE LAYERS (channels, kernel, stride, padding, dilation, groups)")
    print("=" * 80)

    rows = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Conv1d, nn.ConvTranspose1d, nn.Conv3d, nn.ConvTranspose3d)):
            rows.append((
                name,
                m.__class__.__name__,
                getattr(m, "in_channels", None),
                getattr(m, "out_channels", None),
                tuple(m.kernel_size) if hasattr(m, "kernel_size") else None,
                tuple(m.stride) if hasattr(m, "stride") else None,
                tuple(m.padding) if hasattr(m, "padding") else None,
                tuple(m.dilation) if hasattr(m, "dilation") else None,
                getattr(m, "groups", None),
            ))

    if not rows:
        print("No Conv layers found.")
        return

    # Pretty print
    for r in rows:
        name, cls, in_c, out_c, k, s, p, d, g = r
        print(f"- {name:60s} | {cls:16s} | in={in_c:4} out={out_c:4} | k={k} s={s} p={p} d={d} g={g}")


def best_effort_vq_info(model: nn.Module):
    """
    Attempts to find VQ/codebook parameters (embedding count, embedding dim, etc.)
    without relying on your exact class names.
    """
    print("\n" + "=" * 80)
    print("BEST-EFFORT VQ / CODEBOOK INFO")
    print("=" * 80)

    found = False

    # Look for nn.Embedding (common for codebooks)
    for name, m in model.named_modules():
        if isinstance(m, nn.Embedding):
            found = True
            print(f"- Embedding module: {name}")
            print(f"  num_embeddings = {m.num_embeddings}")
            print(f"  embedding_dim  = {m.embedding_dim}")

    # Look for attributes that smell like codebooks
    for attr in ["codebook", "codebooks", "embedding", "embeddings", "vq", "quantizer", "quantizers"]:
        if hasattr(model, attr):
            found = True
            obj = getattr(model, attr)
            print(f"- model.{attr}: type={type(obj)}")
            # Try to print useful fields if present
            for sub_attr in ["n_embed", "num_embeddings", "K", "codebook_size", "embedding_dim", "dim", "beta"]:
                if hasattr(obj, sub_attr):
                    try:
                        print(f"  {sub_attr} = {getattr(obj, sub_attr)}")
                    except Exception:
                        pass

    if not found:
        print("No obvious codebook / embedding modules found via heuristics.")
        print("If your VQ module uses a custom class (no nn.Embedding), add a print() for its fields.")


def shape_trace_with_hooks(model: nn.Module, x: torch.Tensor):
    """
    Runs a forward pass and collects input/output shapes for leaf modules,
    then prints a compact trace focusing on conv-like + down/up sampling.
    """
    shapes = []
    hooks = []

    def register_hook(name, m):
        def hook(m, inp, out):
            def fmt(t):
                if isinstance(t, torch.Tensor):
                    return tuple(t.shape)
                if isinstance(t, (list, tuple)) and len(t) and isinstance(t[0], torch.Tensor):
                    return [tuple(tt.shape) for tt in t]
                return str(type(t))

            in_shape = fmt(inp[0]) if isinstance(inp, (tuple, list)) and len(inp) else fmt(inp)
            out_shape = fmt(out)
            shapes.append((name, m.__class__.__name__, in_shape, out_shape))
        return m.register_forward_hook(hook)

    for name, m in model.named_modules():
        # leaf modules only (less spam)
        if len(list(m.children())) == 0:
            hooks.append(register_hook(name, m))

    with torch.no_grad():
        y = model(x)

    for h in hooks:
        h.remove()

    print("\n" + "=" * 80)
    print("FORWARD SHAPE TRACE (leaf modules)")
    print("=" * 80)
    for name, cls, ins, outs in shapes:
        # prioritize interesting modules
        interesting = (
            "Conv" in cls or "Pool" in cls or "Upsample" in cls or
            "PixelShuffle" in cls or "Interpolate" in cls or "Embedding" in cls
        )
        if interesting:
            print(f"- {name:60s} | {cls:20s} | in={ins} -> out={outs}")

    print("\n" + "=" * 80)
    print("MODEL OUTPUT STRUCTURE")
    print("=" * 80)
    if isinstance(y, torch.Tensor):
        print(f"Output: Tensor shape = {tuple(y.shape)}")
    elif isinstance(y, (list, tuple)):
        print(f"Output: {type(y).__name__} (len={len(y)})")
        for i, item in enumerate(y):
            if isinstance(item, torch.Tensor):
                print(f"  [{i}] Tensor shape = {tuple(item.shape)}")
            else:
                print(f"  [{i}] {type(item)}")
    else:
        print(f"Output: {type(y)}")

    return y


def infer_latents_from_output(y):
    """
    Your training loop expects: reconstructed, quant_loss, *levels_indices
    so we try to parse that convention.
    """
    print("\n" + "=" * 80)
    print("LATENT / INDICES SUMMARY (best-effort)")
    print("=" * 80)

    if not isinstance(y, (list, tuple)) or len(y) < 2:
        print("Model output is not (reconstructed, quant_loss, ...). Can't infer latents from outputs.")
        return

    recon = y[0]
    qloss = y[1]
    rest = y[2:]

    if isinstance(recon, torch.Tensor):
        print(f"- reconstructed: {tuple(recon.shape)}")
    else:
        print(f"- reconstructed: {type(recon)}")

    if isinstance(qloss, torch.Tensor):
        print(f"- quant_loss: {tuple(qloss.shape)} (scalar-ish expected)")
    else:
        print(f"- quant_loss: {type(qloss)}")

    if rest:
        print(f"- levels_indices: {len(rest)} tensor(s)")
        for li, idx in enumerate(rest):
            if isinstance(idx, torch.Tensor):
                print(f"  level {li}: indices shape = {tuple(idx.shape)} | dtype={idx.dtype}")
            else:
                print(f"  level {li}: {type(idx)}")
        # Often indices are [B, H, W] (or [B, T]) so that’s the latent grid
        for li, idx in enumerate(rest):
            if isinstance(idx, torch.Tensor) and idx.ndim >= 2:
                print(f"  -> level {li} latent grid dims (excluding batch): {tuple(idx.shape[1:])}")
    else:
        print("No additional outputs after quant_loss.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint (.pt/.pth). Optional.")
    ap.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--h", type=int, default=None, help="Input height (defaults to cfg image size if present, else 256)")
    ap.add_argument("--w", type=int, default=None, help="Input width  (defaults to cfg image size if present, else 256)")
    ap.add_argument("--channels", type=int, default=3)
    args = ap.parse_args()

    cfg = get_config()
    model = build_model(cfg).to(args.device)
    model.eval()

    print("\n" + "=" * 80)
    print("MODEL ARCHITECTURE (repr)")
    print("=" * 80)
    print(model)

    if args.ckpt:
        missing, unexpected = load_checkpoint(model, args.ckpt, device=args.device)
        print("\n" + "=" * 80)
        print("CHECKPOINT LOAD")
        print("=" * 80)
        print(f"Loaded: {args.ckpt}")
        print(f"Missing keys   ({len(missing)}): {missing[:20]}{' ...' if len(missing) > 20 else ''}")
        print(f"Unexpected keys({len(unexpected)}): {unexpected[:20]}{' ...' if len(unexpected) > 20 else ''}")

    # Conv specs + VQ info
    print_conv_specs(model)
    best_effort_vq_info(model)

    # Dummy forward to infer latent sizes
    # Try to pull size from cfg if present
    H = args.h
    W = args.w
    for cand in ["image_size", "img_size", "resolution", "input_size"]:
        if H is None and hasattr(cfg, cand):
            v = getattr(cfg, cand)
            if isinstance(v, int):
                H = W = v
            elif isinstance(v, (tuple, list)) and len(v) >= 2:
                H, W = int(v[0]), int(v[1])
    H = H or 256
    W = W or 256

    x = torch.randn(args.batch, args.channels, H, W, device=args.device)
    y = shape_trace_with_hooks(model, x)
    infer_latents_from_output(y)


if __name__ == "__main__":
    main()
