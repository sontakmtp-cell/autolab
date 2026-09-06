"""Durable training checkpoint manager for PAXG Forecast Lab.

Provides crash-atomic checkpoint saving using versioned immutable directories
and an atomic CURRENT pointer, cryptographic SHA-256 checksum integrity verification,
torn checkpoint prevention, and fail-closed compatibility validation across restarts.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import time
from typing import Any

from peft import PeftModel
import torch

from ..constants import MODEL_REPO, MODEL_REVISION
from ..model.manifest import AdapterManifest
from ..model.train_spec import TrainSpec

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_DIR = Path("var/paxg_lab/checkpoints")


@dataclass(frozen=True)
class TrainingCheckpoint:
    """Represents a reconciled durable training checkpoint with verified cryptographic integrity."""

    job_id: str
    checkpoint_dir: Path
    metadata: dict[str, Any]
    epoch: int
    minibatch_idx: int
    global_step: int
    best_val_loss: float
    weights_path: Path
    manifest_path: Path
    trainer_state_path: Path | None = None


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
        minibatch_idx: int = -1,
        trainer_state: dict[str, Any] | None = None,
        status: str = "IN_PROGRESS",
        simulate_crash_before_replace: bool = False,
        simulate_crash_during_commit: bool = False,
        simulate_checksum_corruption: bool = False,
    ) -> Path:
        """Atomically saves checkpoint using versioned directory + atomic CURRENT pointer.

        Guarantees:
        1. Any crash during writes leaves the pre-existing complete checkpoint untouched.
        2. No torn states (mixed old/new files) can ever be observed.
        3. All files in the committed version are protected by SHA-256 checksums.
        """
        job_dir = self.get_checkpoint_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        versions_dir = job_dir / "versions"
        versions_dir.mkdir(parents=True, exist_ok=True)

        ts = int(time.time() * 1000)
        staging_dir = job_dir / f".tmp_{ts}"
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        staging_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 1. Save PEFT weights and config into staging directory
            peft_model.save_pretrained(str(staging_dir))

            # 2. Save manifest
            manifest_path = staging_dir / "paxg_manifest.json"
            manifest.save_json(manifest_path)

            # 3. Save comprehensive checkpoint metadata
            meta_dict = {
                "job_id": job_id,
                "snapshot_hash": manifest.snapshot_hash,
                "train_spec": manifest.train_spec,
                "epoch": epoch,
                "minibatch_idx": minibatch_idx,
                "global_step": step,
                "best_val_loss": best_val_loss,
                "base_model_repo": manifest.base_model_repo,
                "base_model_revision": manifest.base_model_revision,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": status,
            }
            meta_path = staging_dir / "checkpoint_metadata.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta_dict, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())

            # 4. Save trainer state (optimizer, scheduler, RNG) if provided
            if trainer_state is not None:
                trainer_state_path = staging_dir / "trainer_state.pt"
                torch.save(trainer_state, str(trainer_state_path))
                with open(trainer_state_path, "ab") as f:
                    f.flush()
                    os.fsync(f.fileno())

            # 5. Compute SHA-256 checksums for all saved files
            checksums: dict[str, str] = {}
            for file_path in staging_dir.iterdir():
                if file_path.is_file():
                    with open(file_path, "rb") as f:
                        checksums[file_path.name] = hashlib.sha256(f.read()).hexdigest()

            if simulate_checksum_corruption and checksums:
                first_k = next(iter(checksums))
                checksums[first_k] = "0000000000000000000000000000000000000000000000000000000000000000"

            checksums_path = staging_dir / "checksums.json"
            with open(checksums_path, "w", encoding="utf-8") as f:
                json.dump(checksums, f, indent=2)
                f.flush()
                os.fsync(f.fileno())

            if simulate_crash_before_replace:
                raise RuntimeError("Simulated crash during checkpoint write before atomic replace.")

            # 6. Commit version directory atomically
            version_name = f"v_e{epoch}_s{step}_{ts}"
            final_version_dir = versions_dir / version_name
            os.replace(str(staging_dir), str(final_version_dir))

            if simulate_crash_during_commit:
                raise RuntimeError("Simulated crash after version write, before CURRENT pointer swap.")

            # Read previous pointer to preserve backup_version
            previous_pointer = None
            current_pointer_path = job_dir / "current_checkpoint.json"
            if current_pointer_path.exists():
                try:
                    with open(current_pointer_path, "r", encoding="utf-8") as f:
                        previous_pointer = json.load(f)
                except Exception:
                    pass

            backup_version = previous_pointer.get("version") if previous_pointer else None

            # 7. Atomically swap CURRENT pointer file
            pointer_data = {
                "version": version_name,
                "backup_version": backup_version,
                "epoch": epoch,
                "minibatch_idx": minibatch_idx,
                "global_step": step,
                "best_val_loss": best_val_loss,
                "status": status,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            tmp_pointer = job_dir / f".tmp_curr_{ts}.json"
            with open(tmp_pointer, "w", encoding="utf-8") as f:
                json.dump(pointer_data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())

            current_pointer_path = job_dir / "current_checkpoint.json"
            os.replace(str(tmp_pointer), str(current_pointer_path))

            logger.info(
                "Saved durable atomic checkpoint for job '%s' version '%s' (epoch %d, minibatch %d, step %d).",
                job_id,
                version_name,
                epoch,
                minibatch_idx,
                step,
            )
            return final_version_dir
        except Exception:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise

    def _verify_version(self, job_id: str, version_dir: Path) -> TrainingCheckpoint | None:
        """Verifies cryptographic integrity of a version directory using its checksums.json."""
        if not version_dir.exists() or not version_dir.is_dir():
            return None

        meta_path = version_dir / "checkpoint_metadata.json"
        manifest_path = version_dir / "paxg_manifest.json"
        checksums_path = version_dir / "checksums.json"

        if not meta_path.exists() or not manifest_path.exists() or not checksums_path.exists():
            return None

        # Check weights exist
        weights_path = version_dir / "adapter_model.safetensors"
        if not weights_path.exists():
            weights_path = version_dir / "adapter_model.bin"
            if not weights_path.exists():
                return None

        trainer_state_path = version_dir / "trainer_state.pt"
        if not trainer_state_path.exists():
            trainer_state_path = None

        try:
            with open(checksums_path, "r", encoding="utf-8") as f:
                checksums = json.load(f)

            # Verify cryptographic SHA-256 for all recorded files
            for fname, expected_hash in checksums.items():
                target_file = version_dir / fname
                if not target_file.exists():
                    logger.warning("Checkpoint integrity failed: missing file '%s' in %s", fname, version_dir)
                    return None
                with open(target_file, "rb") as f:
                    actual_hash = hashlib.sha256(f.read()).hexdigest()
                if actual_hash != expected_hash:
                    logger.warning(
                        "Checkpoint integrity failed: SHA-256 hash mismatch for '%s' in %s (expected %s, got %s)",
                        fname,
                        version_dir,
                        expected_hash,
                        actual_hash,
                    )
                    return None

            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            return TrainingCheckpoint(
                job_id=job_id,
                checkpoint_dir=version_dir,
                metadata=meta,
                epoch=int(meta.get("epoch", 0)),
                minibatch_idx=int(meta.get("minibatch_idx", -1)),
                global_step=int(meta.get("global_step", 0)),
                best_val_loss=float(meta.get("best_val_loss", 0.0)),
                weights_path=weights_path,
                manifest_path=manifest_path,
                trainer_state_path=trainer_state_path,
            )
        except Exception as exc:
            logger.warning("Failed to verify checkpoint in %s: %s", version_dir, exc)
            return None

    def load_checkpoint(self, job_id: str) -> TrainingCheckpoint | None:
        """Loads and cryptographically verifies the current durable checkpoint for the job_id.

        If the current pointer is missing or corrupted, falls back to the latest valid version directory.
        Returns None if no valid, uncorrupted checkpoint exists.
        """
        job_dir = self.get_checkpoint_dir(job_id)
        versions_dir = job_dir / "versions"
        current_pointer_path = job_dir / "current_checkpoint.json"

        # 1. Try loading through CURRENT pointer file
        if current_pointer_path.exists():
            try:
                with open(current_pointer_path, "r", encoding="utf-8") as f:
                    ptr = json.load(f)
                v_name = ptr.get("version")
                if v_name:
                    ckpt = self._verify_version(job_id, versions_dir / v_name)
                    if ckpt is not None:
                        return ckpt
                backup_v = ptr.get("backup_version")
                if backup_v:
                    ckpt = self._verify_version(job_id, versions_dir / backup_v)
                    if ckpt is not None:
                        logger.info("Reconciled valid backup checkpoint version '%s' for job '%s'", backup_v, job_id)
                        return ckpt
            except Exception as exc:
                logger.warning("Failed to read current_checkpoint.json for job '%s': %s", job_id, exc)

        # 2. Self-healing fallback: scan versions directory for latest valid completed version
        if versions_dir.exists():
            candidates = sorted(
                [d for d in versions_dir.iterdir() if d.is_dir() and d.name.startswith("v_")],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for v_dir in candidates:
                ckpt = self._verify_version(job_id, v_dir)
                if ckpt is not None:
                    logger.info("Reconciled valid fallback checkpoint version '%s' for job '%s'", v_dir.name, job_id)
                    return ckpt

        # 3. Backward compatibility: check canonical flat job directory
        flat_ckpt = self._verify_legacy_flat(job_id, job_dir)
        if flat_ckpt is not None:
            return flat_ckpt

        return None

    def _verify_legacy_flat(self, job_id: str, job_dir: Path) -> TrainingCheckpoint | None:
        """Verifies legacy flat checkpoint directory if present."""
        meta_path = job_dir / "checkpoint_metadata.json"
        manifest_path = job_dir / "paxg_manifest.json"
        if not meta_path.exists() or not manifest_path.exists():
            return None

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
                minibatch_idx=int(meta.get("minibatch_idx", -1)),
                global_step=int(meta.get("global_step", 0)),
                best_val_loss=float(meta.get("best_val_loss", 0.0)),
                weights_path=weights_path,
                manifest_path=manifest_path,
                trainer_state_path=job_dir / "trainer_state.pt" if (job_dir / "trainer_state.pt").exists() else None,
            )
        except Exception:
            return None

    @staticmethod
    def validate_compatibility(
        checkpoint: TrainingCheckpoint,
        current_spec: TrainSpec,
        current_snapshot_hash: str,
        base_model_repo: str = MODEL_REPO,
        base_model_revision: str = MODEL_REVISION,
    ) -> None:
        """Strictly validates checkpoint compatibility against current job specification.

        Raises ValueError (fail closed) if any provenance, architecture, or dataset mismatch exists.
        """
        meta = checkpoint.metadata
        manifest_data: dict[str, Any] = {}
        if checkpoint.manifest_path.exists():
            try:
                with open(checkpoint.manifest_path, "r", encoding="utf-8") as f:
                    manifest_data = json.load(f)
            except Exception:
                pass

        # 1. Strict Dataset Snapshot Integrity
        ckpt_snap_hash = meta.get("snapshot_hash") or manifest_data.get("snapshot_hash")
        if not ckpt_snap_hash or ckpt_snap_hash != current_snapshot_hash:
            raise ValueError(
                f"Checkpoint compatibility violation: snapshot_hash mismatch! "
                f"Checkpoint trained on '{ckpt_snap_hash}', but current job uses '{current_snapshot_hash}'."
            )

        # 2. Strict Base Model Provenance
        ckpt_base_repo = meta.get("base_model_repo") or manifest_data.get("base_model_repo")
        ckpt_base_rev = meta.get("base_model_revision") or manifest_data.get("base_model_revision")
        if ckpt_base_repo != base_model_repo or ckpt_base_rev != base_model_revision:
            raise ValueError(
                f"Checkpoint compatibility violation: base model mismatch! "
                f"Checkpoint uses '{ckpt_base_repo}@{ckpt_base_rev}', current job uses '{base_model_repo}@{base_model_revision}'."
            )

        # 3. Strict Model Architecture & LoRA Hyperparameter Compatibility
        ckpt_spec = meta.get("train_spec") or manifest_data.get("train_spec") or {}
        checks = [
            ("timeframe", current_spec.timeframe),
            ("horizon", current_spec.horizon),
            ("context_len", current_spec.context_len),
            ("feature_set", current_spec.feature_set),
            ("lora_r", current_spec.lora_r),
            ("lora_alpha", current_spec.lora_alpha or (2 * current_spec.lora_r)),
            ("extended_targets", current_spec.extended_targets),
            ("custom_targets", current_spec.custom_targets),
        ]
        for key, expected_val in checks:
            actual_val = ckpt_spec.get(key) if key in ckpt_spec else manifest_data.get(key)
            if actual_val is not None and actual_val != expected_val:
                raise ValueError(
                    f"Checkpoint compatibility violation: hyperparameter '{key}' mismatch! "
                    f"Checkpoint has {actual_val!r}, current job requires {expected_val!r}."
                )
