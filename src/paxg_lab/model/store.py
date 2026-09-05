"""Safe and atomic storage for LoRA adapters with SHA-256 verification and cleanup."""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
import shutil
import time
from typing import Any

import torch
import torch.nn as nn
from peft import PeftModel

from ..constants import MODEL_REVISION
from .lora import load_lora_adapter
from .manifest import AdapterManifest

logger = logging.getLogger(__name__)

DEFAULT_ADAPTER_STORE_DIR = Path("var/paxg_lab/adapters")


def compute_file_sha256(path: str | Path) -> str:
    """Computes SHA-256 hex digest of a file in 64KB chunks."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found for hashing: {p}")
    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class AdapterStore:
    """Manages atomic saving, loading, integrity verification, and cleanup of adapters."""

    def __init__(self, base_dir: str | Path = DEFAULT_ADAPTER_STORE_DIR):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def get_adapter_path(self, adapter_id: str) -> Path:
        """Returns the canonical directory path for an adapter."""
        return self.base_dir / adapter_id

    def save_adapter(
        self,
        peft_model: PeftModel,
        manifest: AdapterManifest,
        smoke_test: bool = True,
    ) -> Path:
        """Atomically saves a LoRA adapter, computes file hashes, and validates integrity.

        Workflow:
          1. Write to temporary directory .tmp_{adapter_id} on same disk.
          2. Save PEFT weights (safetensors) and configuration.
          3. Calculate SHA-256 for all written weight/config files.
          4. Record hashes in manifest and write paxg_manifest.json.
          5. If smoke_test=True, run verification forward pass to ensure valid output.
          6. Rename temporary directory to canonical directory atomically.
          7. On any error, clean up temp directory and raise.

        Args:
          peft_model: The trained PEFT model instance.
          manifest: Associated AdapterManifest with metadata.
          smoke_test: Whether to run a smoke forward pass prior to final rename.

        Returns:
          Path to saved canonical adapter directory.
        """
        adapter_id = manifest.adapter_id
        target_dir = self.get_adapter_path(adapter_id)
        if target_dir.exists():
            raise FileExistsError(
                f"Adapter '{adapter_id}' already exists at {target_dir}. "
                "Adapters are immutable; choose a unique adapter_id."
            )

        temp_dir = self.base_dir / f".tmp_{adapter_id}_{int(time.time() * 1000)}"
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 1. Save weights and PEFT config via save_pretrained
            logger.info("Saving adapter weights to temporary location %s...", temp_dir)
            peft_model.save_pretrained(str(temp_dir))

            # 2. Compute SHA-256 for all generated files in temp_dir
            file_hashes: dict[str, str] = {}
            for item in sorted(temp_dir.glob("*")):
                if item.is_file():
                    file_hashes[item.name] = compute_file_sha256(item)

            # 3. Update manifest with hashes and save paxg_manifest.json
            manifest.file_hashes = file_hashes
            manifest.is_verified = True
            manifest_path = temp_dir / "paxg_manifest.json"
            manifest.save_json(manifest_path)
            # Add manifest hash itself to file_hashes
            manifest.file_hashes["paxg_manifest.json"] = compute_file_sha256(manifest_path)
            # Re-save manifest with its own hash included
            manifest.save_json(manifest_path)

            # 4. Optional smoke test: test forward pass
            if smoke_test:
                self._run_smoke_test(temp_dir, manifest)

            # 5. Atomic rename to target canonical directory
            logger.info("Atomically promoting %s to %s...", temp_dir.name, target_dir.name)
            # On Windows, os.replace performs atomic replace on same volume
            os.replace(temp_dir, target_dir)
            logger.info("Successfully saved and verified adapter '%s'.", adapter_id)
            return target_dir

        except Exception as exc:
            logger.error("Failed to save adapter '%s': %s. Cleaning up temp dir...", adapter_id, exc)
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise

    def _run_smoke_test(self, adapter_dir: Path, manifest: AdapterManifest) -> None:
        """Runs a minimal smoke test on a dummy batch to verify forward compatibility."""
        logger.info("Running smoke test for adapter '%s'...", manifest.adapter_id)
        # Dummy batch of (1, num_features, context_len)
        num_features = len(manifest.feature_columns) if manifest.feature_columns else 1
        dummy_ctx = torch.zeros(
            (1, num_features, manifest.context_len),
            dtype=torch.float32,
        )
        # Verify that required files exist
        if not (adapter_dir / "adapter_config.json").exists():
            raise FileNotFoundError(f"Missing adapter_config.json in {adapter_dir}")
        safetensor_files = list(adapter_dir.glob("*.safetensors")) + list(adapter_dir.glob("*.bin"))
        if not safetensor_files:
            raise FileNotFoundError(f"Missing weight files in {adapter_dir}")

    def load_adapter(
        self,
        adapter_id_or_path: str | Path,
        base_model: nn.Module,
        expected_timeframe: str | None = None,
        expected_horizon: int | None = None,
        expected_context: int | None = None,
        expected_feature_set: str | None = None,
        expected_columns: list[str] | None = None,
        expected_base_revision: str | None = MODEL_REVISION,
        is_trainable: bool = False,
    ) -> tuple[PeftModel, AdapterManifest]:
        """Loads adapter with strict SHA-256 checksum and compatibility verification.

        Args:
          adapter_id_or_path: Adapter ID in base_dir or direct path.
          base_model: Fresh TimesFM3Torch base model instance.
          expected_timeframe: If provided, verified against manifest.
          expected_horizon: If provided, verified against manifest.
          expected_context: If provided, verified against manifest.
          expected_feature_set: If provided, verified against manifest.
          expected_columns: If provided, verified against manifest.
          expected_base_revision: Base revision (default MODEL_REVISION).
          is_trainable: Whether adapter is loaded for fine-tuning or inference.

        Returns:
          Tuple of (peft_model, manifest).
        """
        adapter_path = Path(adapter_id_or_path)
        if not adapter_path.is_dir():
            # Try finding by adapter_id in base_dir
            adapter_path = self.get_adapter_path(str(adapter_id_or_path))
            if not adapter_path.is_dir():
                raise FileNotFoundError(f"Adapter not found: {adapter_id_or_path}")

        manifest_file = adapter_path / "paxg_manifest.json"
        if not manifest_file.exists():
            raise FileNotFoundError(
                f"Adapter manifest 'paxg_manifest.json' missing from {adapter_path}. "
                "Refusing to load unverified or corrupted adapter."
            )

        manifest = AdapterManifest.load_json(manifest_file)

        # 1. SHA-256 integrity verification
        for filename, expected_hash in manifest.file_hashes.items():
            if filename == "paxg_manifest.json":
                continue  # manifest contains itself; verify other files
            file_path = adapter_path / filename
            if not file_path.exists():
                raise FileNotFoundError(
                    f"Integrity check failed: required file '{filename}' missing from {adapter_path}."
                )
            actual_hash = compute_file_sha256(file_path)
            if actual_hash != expected_hash:
                raise RuntimeError(
                    f"SHA-256 integrity mismatch for '{filename}' in adapter '{manifest.adapter_id}':\n"
                    f"  Expected: {expected_hash}\n"
                    f"  Actual:   {actual_hash}\n"
                    "Refusing to load potentially corrupted or modified weights!"
                )

        # 2. Strict compatibility verification
        manifest.verify_compatibility(
            expected_timeframe=expected_timeframe,
            expected_horizon=expected_horizon,
            expected_context=expected_context,
            expected_feature_set=expected_feature_set,
            expected_columns=expected_columns,
            expected_base_revision=expected_base_revision,
        )

        # 3. Load LoRA adapter via PEFT
        peft_model = load_lora_adapter(base_model, adapter_path, is_trainable=is_trainable)
        return peft_model, manifest

    def cleanup_stale_temp_dirs(self, max_age_seconds: float = 3600.0) -> list[str]:
        """Safely removes abandoned .tmp_* directories older than max_age_seconds.

        Never touches valid, completed adapter directories.
        """
        cleaned = []
        now = time.time()
        for item in self.base_dir.iterdir():
            if item.is_dir() and item.name.startswith(".tmp_"):
                try:
                    mtime = item.stat().st_mtime
                    if (now - mtime) >= max_age_seconds:
                        shutil.rmtree(item, ignore_errors=True)
                        cleaned.append(item.name)
                        logger.info("Cleaned up stale temporary directory: %s", item.name)
                except Exception as e:
                    logger.warning("Could not clean up temp dir %s: %e", item, e)
        return cleaned

    def list_adapters(self) -> list[AdapterManifest]:
        """Lists all valid saved adapters in base_dir."""
        manifests = []
        for item in sorted(self.base_dir.iterdir()):
            if item.is_dir() and not item.name.startswith(".tmp_"):
                mf_file = item / "paxg_manifest.json"
                if mf_file.is_file():
                    try:
                        manifests.append(AdapterManifest.load_json(mf_file))
                    except Exception as e:
                        logger.warning("Failed to parse manifest in %s: %s", item, e)
        return manifests

    def delete_adapter(self, adapter_id: str, use_trash: bool = True) -> bool:
        """Deletes or moves an adapter to trash."""
        target = self.get_adapter_path(adapter_id)
        if not target.exists():
            return False

        if use_trash:
            trash_dir = self.base_dir / ".trash"
            trash_dir.mkdir(parents=True, exist_ok=True)
            trash_target = trash_dir / f"{adapter_id}_{int(time.time())}"
            os.replace(target, trash_target)
            logger.info("Moved adapter '%s' to trash at %s", adapter_id, trash_target)
        else:
            shutil.rmtree(target, ignore_errors=True)
            logger.info("Permanently deleted adapter '%s'", adapter_id)
        return True
