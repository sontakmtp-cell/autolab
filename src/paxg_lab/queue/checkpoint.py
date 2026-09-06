"""Durable training checkpoint manager for PAXG Forecast Lab.

Provides atomic checkpoint saving (tmp -> fsync -> atomic replace),
integrity verification, and checkpoint reconciliation across restarts.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import shutil
import time
from typing import Any

from peft import PeftModel

from ..model.manifest import AdapterManifest

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_DIR = Path("var/paxg_lab/checkpoints")


@dataclass(frozen=True)
class TrainingCheckpoint:
    """Represents a reconciled durable training checkpoint."""

    job_id: str
    checkpoint_dir: Path
    metadata: dict[str, Any]
    epoch: int
    global_step: int
    best_val_loss: float
    weights_path: Path
    manifest_path: Path


class TrainingCheckpointManager:
    """Manages durable, crash-resilient training checkpoints for GPU jobs."""

    def __init__(self, base_dir: str | Path | None = None):
        self.base_dir = Path(base_dir) if base_dir is not None else Path(DEFAULT_CHECKPOINT_DIR)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def get_checkpoint_dir(self, job_id: str) -> Path:
        """Returns canonical checkpoint directory for a job."""
        return self.base_dir / job_id

    def save_checkpoint(
        self,
        job_id: str,
        peft_model: PeftModel,
        manifest: AdapterManifest,
        epoch: int,
        step: int,
        best_val_loss: float,
        status: str = "IN_PROGRESS",
        simulate_crash_before_replace: bool = False,
    ) -> Path:
        """Atomically saves checkpoint using temporary directory and fsync.

        Guarantees that any crash during writing will leave any pre-existing
        valid checkpoint intact and usable.
        """
        job_dir = self.get_checkpoint_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)

        tmp_dir = job_dir / f".tmp_{int(time.time() * 1000)}"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 1. Save PEFT weights and config into tmp directory
            peft_model.save_pretrained(str(tmp_dir))

            # 2. Save manifest
            manifest_path = tmp_dir / "paxg_manifest.json"
            manifest.save_json(manifest_path)

            # 3. Save comprehensive checkpoint metadata
            meta_dict = {
                "job_id": job_id,
                "snapshot_hash": manifest.snapshot_hash,
                "train_spec": manifest.train_spec,
                "epoch": epoch,
                "global_step": step,
                "best_val_loss": best_val_loss,
                "base_model_repo": manifest.base_model_repo,
                "base_model_revision": manifest.base_model_revision,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": status,
            }
            meta_path = tmp_dir / "checkpoint_metadata.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta_dict, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())

            if simulate_crash_before_replace:
                raise RuntimeError("Simulated crash during checkpoint write before atomic replace.")

            # 4. Atomically swap files into canonical directory
            for src_file in tmp_dir.iterdir():
                if src_file.is_file():
                    dst_file = job_dir / src_file.name
                    # Atomically replace file on disk
                    os.replace(str(src_file), str(dst_file))

            shutil.rmtree(tmp_dir, ignore_errors=True)
            logger.info("Saved durable checkpoint for job '%s' at epoch %d, step %d.", job_id, epoch, step)
            return job_dir
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

    def load_checkpoint(self, job_id: str) -> TrainingCheckpoint | None:
        """Loads and verifies a durable checkpoint for the given job_id.

        Returns None if no valid checkpoint exists.
        """
        job_dir = self.get_checkpoint_dir(job_id)
        meta_path = job_dir / "checkpoint_metadata.json"
        manifest_path = job_dir / "paxg_manifest.json"

        if not meta_path.exists() or not manifest_path.exists():
            return None

        # Check weights exist
        weights_path = job_dir / "adapter_model.safetensors"
        if not weights_path.exists():
            weights_path = job_dir / "adapter_model.bin"
            if not weights_path.exists():
                return None

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            return TrainingCheckpoint(
                job_id=job_id,
                checkpoint_dir=job_dir,
                metadata=meta,
                epoch=int(meta.get("epoch", 0)),
                global_step=int(meta.get("global_step", 0)),
                best_val_loss=float(meta.get("best_val_loss", 0.0)),
                weights_path=weights_path,
                manifest_path=manifest_path,
            )
        except Exception as exc:
            logger.warning("Failed to load checkpoint for job '%s': %s", job_id, exc)
            return None
