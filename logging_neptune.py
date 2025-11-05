# rank-0 only neptune wrapper (works online or offline)
# offline (HPC): set NEPTUNE_PROJECT + NEPTUNE_MODE=offline, then `neptune sync` later.

import os
import neptune
import os
from pathlib import Path
import torch

class NeptuneRun:
    def __init__(self):
        self.run = None

    def init(self, project=None, api_token=None):
        self.run = neptune.init_run(
            project=project or os.getenv("NEPTUNE_PROJECT"),
            api_token=api_token or os.getenv("NEPTUNE_API_TOKEN"),
            capture_stdout=False,
            capture_stderr=False,
        )

    def __getitem__(self, key):
        return self.run[key]

    def stop(self):
        if self.run is not None:
            self.run.stop()



class CheckpointManager:
    """
    Mode-aware checkpoints:
      - offline: keep files under RUN_DIR/ckpts and upload from there (safe for later 'neptune sync').
      - online:  write into RUN_DIR/tmp_ckpts, upload, then delete to save space.
    """
    def __init__(self, run):
        self.run = run
        self.mode = os.getenv("NEPTUNE_MODE", "online").lower()
        self.run_dir = Path(os.getenv("RUN_DIR", "./runs"))
        self.keep_dir = self.run_dir / "ckpts"       # canonical storage (offline)
        self.temp_dir = self.run_dir / "tmp_ckpts"   # throwaway (online)
        # ensure directories exist
        (self.keep_dir if self.is_offline else self.temp_dir).mkdir(parents=True, exist_ok=True)

    @property
    def is_offline(self) -> bool:
        return self.mode.startswith("off")

    def _save(self, model, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), path)

    def save_and_upload(self, model, loss: float, best_loss: float, phase: str, epoch: int | None = None) -> float:
        """
        Always upload a '..._last' file. If loss improved, also update '..._best'.
        - offline: save persistent files in RUN_DIR/ckpts and keep them.
        - online:  save into RUN_DIR/tmp_ckpts, upload, then delete local temp file(s).
        """
        # choose where to write the "last" file
        last_path = (self.keep_dir if self.is_offline else self.temp_dir) / f"{phase}_last.pt"
        self._save(model, last_path)

        # upload last (Neptune will copy into its own storage; in offline it queues locally)
        try:
            self.run[f"model/{phase}_checkpoint_last"].upload(str(last_path))
        except Exception:
            pass

        # best
        if loss < best_loss:
            best_loss = loss
            if self.is_offline:
                best_path = self.keep_dir / f"{phase}_best.pt"
                self._save(model, best_path)
                try:
                    self.run[f"model/{phase}_checkpoint_best"].upload(str(best_path))
                except Exception:
                    pass
            else:
                # online: we can re-upload the same temp file under the "best" key
                try:
                    self.run[f"model/{phase}_checkpoint_best"].upload(str(last_path))
                except Exception:
                    pass

        # optional epoch snapshot (kept only in offline mode)
        if self.is_offline and epoch is not None:
            try:
                snap = self.keep_dir / f"{phase}_epoch{epoch:04d}.pt"
                self._save(model, snap)
            except Exception:
                pass

        # in online mode, clean up temp files we just created
        if not self.is_offline:
            # make sure Neptune has finished using the file
            try:
                self.run.sync()
            except Exception:
                pass
            try:
                last_path.unlink(missing_ok=True)
            except Exception:
                pass

        return best_loss

    def finalize(self):
        """Optional: extra cleanup for online mode."""
        if not self.is_offline:
            try:
                # remove empty temp dir
                if self.temp_dir.exists() and not any(self.temp_dir.iterdir()):
                    self.temp_dir.rmdir()
            except Exception:
                pass
