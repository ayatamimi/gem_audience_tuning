
import signal
import time
from pathlib import Path
import torch
from typing import Optional, Dict, Any

def save_checkpoint(path: Path, model, optimizer, epoch: int, extra: Optional[Dict[str, Any]] = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }
    if extra:
        state.update(extra)
    torch.save(state, path)

def install_preemption_handler(model, optimizer, run_dir: Path, neptune_if_main=None):
    """Save a last-second checkpoint when SIGTERM is received (common on HPC)."""
    def _handler(signum, frame):
        ts = time.strftime("%Y%m%d-%H%M%S")
        ckpt = run_dir / f"signal_ckpt_{ts}.pt"
        save_checkpoint(ckpt, model, optimizer, epoch=-1, extra={"signal": int(signum), "ts": ts})
        if neptune_if_main is not None:
            try:
                neptune_if_main[f"model/preemption/{ts}"].upload(str(ckpt))
            except Exception:
                pass
    signal.signal(signal.SIGTERM, _handler)
