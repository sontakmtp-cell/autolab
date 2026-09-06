"""Safe and atomic storage for LoRA adapters with sidecar SHA-256 verification and cleanup."""

from __future__ import annotations

import hashlib
import logging
import json
import os
from pathlib import Path
import shutil
import sqlite3
import time
from typing import Any
import zipfile

import torch
import torch.nn as nn
from peft import PeftModel

from ..constants import MODEL_REVISION
from .lora import load_lora_adapter
from .manifest import AdapterManifest

logger = logging.getLogger(__name__)

DEFAULT_ADAPTER_STORE_DIR = Path("var/paxg_lab/adapters")
DEFAULT_DB_PATH = Path("var/paxg_lab/paxg_lab.db")
CHECKSUMS_FILENAME = "checksums.sha256"


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

    def __init__(
        self,
        base_dir: str | Path = DEFAULT_ADAPTER_STORE_DIR,
        db_path: str | Path | None = DEFAULT_DB_PATH,
    ):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = Path(db_path) if db_path is not None else None
        if self.db_path is not None:
            self._init_registry()

    def get_adapter_path(self, adapter_id: str) -> Path:
        """Returns the canonical directory path for an adapter."""
        return self.base_dir / adapter_id

    def save_adapter(
        self,
        peft_model: PeftModel,
        manifest: AdapterManifest,
        base_model: nn.Module | None = None,
        smoke_test: bool = True,
    ) -> Path:
        """Atomically saves a LoRA adapter, generates sidecar SHA-256 checksums, and validates integrity.

        Workflow:
          1. Write to temporary directory .tmp_{adapter_id}_{timestamp} on same disk.
          2. Save PEFT weights (safetensors) and configuration.
          3. Calculate SHA-256 for generated weight and config files; store in manifest.
          4. Write paxg_manifest.json.
          5. Write sidecar checksums.sha256 covering ALL files (weights, configs, manifest).
          6. If smoke_test=True, run verification forward pass to assert valid shape and finite output.
          7. Rename temporary directory to canonical directory atomically.
          8. On any error, clean up temp directory and raise.

        Args:
          peft_model: The trained PEFT model instance.
          manifest: Associated AdapterManifest with metadata.
          base_model: Optional base model to test loading from disk during smoke test.
          smoke_test: Whether to run a verification forward pass prior to final rename.

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

            # 4. Generate sidecar checksums.sha256 covering all files including paxg_manifest.json
            checksums_path = temp_dir / CHECKSUMS_FILENAME
            with open(checksums_path, "w", encoding="utf-8") as f:
                for item in sorted(temp_dir.glob("*")):
                    if item.is_file() and item.name != CHECKSUMS_FILENAME:
                        digest = compute_file_sha256(item)
                        f.write(f"{digest}  {item.name}\n")

            # 5. Smoke test: run actual forward pass to verify output shape & finiteness
            if smoke_test:
                self._run_smoke_test(
                    temp_dir=temp_dir,
                    manifest=manifest,
                    peft_model=peft_model,
                    base_model=base_model,
                )

            # 6. Atomic rename to target canonical directory
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

    def _run_smoke_test(
        self,
        temp_dir: Path,
        manifest: AdapterManifest,
        peft_model: PeftModel | None = None,
        base_model: nn.Module | None = None,
    ) -> None:
        """Runs a real forward pass smoke test to verify compatibility and finite outputs."""
        logger.info("Running smoke test for adapter '%s'...", manifest.adapter_id)
        if not (temp_dir / "adapter_config.json").exists():
            raise FileNotFoundError(f"Missing adapter_config.json in {temp_dir}")
        safetensor_files = list(temp_dir.glob("*.safetensors")) + list(temp_dir.glob("*.bin"))
        if not safetensor_files:
            raise FileNotFoundError(f"Missing weight files in {temp_dir}")

        num_features = len(manifest.feature_columns) if manifest.feature_columns else 1

        # Determine device from available model
        test_model = None
        if base_model is not None:
            # Test actual loading from temporary directory
            test_model = load_lora_adapter(base_model, temp_dir)
        elif peft_model is not None:
            test_model = peft_model

        if test_model is not None:
            device = next(test_model.parameters()).device
            dummy_ctx = torch.zeros(
                (1, num_features, manifest.context_len),
                dtype=torch.float32,
                device=device,
            )

            # Execute forward pass
            core = test_model
            if hasattr(core, "base_model"):
                core = core.base_model
            if hasattr(core, "model"):
                core = core.model

            with torch.no_grad():
                test_model.eval()
                if hasattr(core, "forward_decode"):
                    out = core.forward_decode(target=dummy_ctx, horizon=manifest.horizon)
                elif hasattr(core, "decode"):
                    out = core.decode(target=dummy_ctx, horizon=manifest.horizon)
                else:
                    out = test_model(dummy_ctx)

            # Validate shape: out should have horizon and quantiles
            if hasattr(out, "shape"):
                if out.ndim >= 3:
                    out_horizon = out.shape[-2]
                    out_quantiles = out.shape[-1]
                    if out_horizon != manifest.horizon:
                        raise ValueError(
                            f"Smoke test horizon mismatch: expected {manifest.horizon}, got {out_horizon}"
                        )
                    if out_quantiles != 9:
                        raise ValueError(
                            f"Smoke test quantiles mismatch: expected 9, got {out_quantiles}"
                        )
                # Check that outputs are strictly finite
                if not torch.isfinite(out).all():
                    raise ValueError(
                        f"Smoke test failed for '{manifest.adapter_id}': output contains NaN or Inf!"
                    )
            logger.info("Smoke test passed successfully: shape=%s, finite=True", getattr(out, "shape", None))

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

        # 1. SHA-256 integrity verification via sidecar checksums.sha256 (mandatory for P3 adapters)
        checksums_file = adapter_path / CHECKSUMS_FILENAME
        if not checksums_file.exists():
            raise FileNotFoundError(
                f"Required sidecar checksum file '{CHECKSUMS_FILENAME}' missing from {adapter_path}. "
                "Refusing to load unverified or downgraded adapter without checksum manifest protection."
            )

        with open(checksums_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    expected_hash, fname = parts[0].strip(), parts[1].strip()
                    fpath = adapter_path / fname
                    if not fpath.exists():
                        raise FileNotFoundError(
                            f"Integrity check failed: required file '{fname}' missing from {adapter_path}."
                        )
                    actual_hash = compute_file_sha256(fpath)
                    if actual_hash != expected_hash:
                        raise RuntimeError(
                            f"SHA-256 integrity mismatch for '{fname}' in adapter '{adapter_path.name}':\n"
                            f"  Expected: {expected_hash}\n"
                            f"  Actual:   {actual_hash}\n"
                            "Refusing to load potentially corrupted or modified files!"
                        )

        manifest = AdapterManifest.load_json(manifest_file)

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
                    logger.warning("Could not clean up temp dir %s: %s", item, e)
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

    def _init_registry(self) -> None:
        """Initializes adapter_registry table in the database."""
        if self.db_path is None:
            return
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self.db_path), timeout=30.0) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS adapter_registry (
                    adapter_id TEXT PRIMARY KEY,
                    alias TEXT,
                    is_pinned INTEGER NOT NULL DEFAULT 0,
                    is_recommended INTEGER NOT NULL DEFAULT 0,
                    timeframe TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    notes TEXT NOT NULL DEFAULT ''
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_adapter_reg_tf ON adapter_registry(timeframe);")
            conn.commit()

    def get_registry_metadata(self, adapter_id: str) -> dict[str, Any]:
        """Gets registry metadata for an adapter (alias, is_pinned, is_recommended, etc.)."""
        if self.db_path is None:
            return {"adapter_id": adapter_id, "alias": adapter_id, "is_pinned": False, "is_recommended": False}
        with sqlite3.connect(str(self.db_path), timeout=30.0) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute("SELECT * FROM adapter_registry WHERE adapter_id = ?;", (adapter_id,))
            row = cur.fetchone()
            if row:
                return {
                    "adapter_id": row["adapter_id"],
                    "alias": row["alias"] or row["adapter_id"],
                    "is_pinned": bool(row["is_pinned"]),
                    "is_recommended": bool(row["is_recommended"]),
                    "timeframe": row["timeframe"],
                    "created_at": row["created_at"],
                    "notes": row["notes"],
                }
        return {"adapter_id": adapter_id, "alias": adapter_id, "is_pinned": False, "is_recommended": False}

    def set_pinned(self, adapter_id: str, pinned: bool = True) -> None:
        """Pins or unpins an adapter to protect from deletion."""
        if self.db_path is None:
            return
        now = time.time()
        with sqlite3.connect(str(self.db_path), timeout=30.0) as conn:
            conn.execute(
                """
                INSERT INTO adapter_registry (adapter_id, alias, is_pinned, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(adapter_id) DO UPDATE SET is_pinned = excluded.is_pinned;
                """,
                (adapter_id, adapter_id, 1 if pinned else 0, now),
            )
            conn.commit()

    def is_pinned(self, adapter_id: str) -> bool:
        """Returns True if the adapter is pinned."""
        meta = self.get_registry_metadata(adapter_id)
        return bool(meta.get("is_pinned", False))

    def set_alias(self, adapter_id: str, alias: str) -> None:
        """Sets a human-readable display alias for the adapter."""
        if self.db_path is None:
            return
        now = time.time()
        with sqlite3.connect(str(self.db_path), timeout=30.0) as conn:
            conn.execute(
                """
                INSERT INTO adapter_registry (adapter_id, alias, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(adapter_id) DO UPDATE SET alias = excluded.alias;
                """,
                (adapter_id, alias.strip(), now),
            )
            conn.commit()

    def get_alias(self, adapter_id: str) -> str:
        """Gets display alias or defaults to adapter_id."""
        meta = self.get_registry_metadata(adapter_id)
        return meta.get("alias") or adapter_id

    def set_recommended(self, adapter_id: str, timeframe: str) -> None:
        """Designates this adapter as the recommended winner for the timeframe.
        
        Automatically clears previous recommended flag for this timeframe and auto-pins it.
        """
        if self.db_path is None:
            return
        tf = timeframe.lower().strip()
        now = time.time()
        with sqlite3.connect(str(self.db_path), timeout=30.0) as conn:
            conn.execute("BEGIN IMMEDIATE;")
            # Clear previous recommended for this timeframe
            conn.execute("UPDATE adapter_registry SET is_recommended = 0 WHERE timeframe = ?;", (tf,))
            # Set this adapter as recommended and pinned
            conn.execute(
                """
                INSERT INTO adapter_registry (adapter_id, alias, is_pinned, is_recommended, timeframe, created_at)
                VALUES (?, ?, 1, 1, ?, ?)
                ON CONFLICT(adapter_id) DO UPDATE SET
                    is_recommended = 1,
                    is_pinned = 1,
                    timeframe = excluded.timeframe;
                """,
                (adapter_id, adapter_id, tf, now),
            )
            conn.commit()

    def get_recommended(self, timeframe: str) -> str | None:
        """Gets the recommended adapter ID for the given timeframe, if any."""
        if self.db_path is None:
            return None
        tf = timeframe.lower().strip()
        with sqlite3.connect(str(self.db_path), timeout=30.0) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                "SELECT adapter_id FROM adapter_registry WHERE timeframe = ? AND is_recommended = 1 LIMIT 1;",
                (tf,),
            )
            row = cur.fetchone()
            if row:
                target = self.get_adapter_path(row["adapter_id"])
                if target.is_dir():
                    return str(row["adapter_id"])
        return None

    def delete_adapter(self, adapter_id: str, use_trash: bool = True) -> bool:
        """Deletes or moves an adapter to trash, strictly protecting pinned or recommended adapters."""
        target = self.get_adapter_path(adapter_id)
        if not target.exists():
            return False

        meta = self.get_registry_metadata(adapter_id)
        if meta.get("is_recommended"):
            raise ValueError(f"Không thể xóa adapter '{adapter_id}' vì đang là adapter khuyến nghị.")

        if self.is_pinned(adapter_id):
            raise ValueError(f"Không thể xóa adapter '{adapter_id}' vì đã được ghim chống xóa.")

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

    def restore_adapter(self, adapter_id: str) -> bool:
        """Restores an adapter from .trash back to active adapters."""
        target = self.get_adapter_path(adapter_id)
        if target.exists():
            return False
        trash_dir = self.base_dir / ".trash"
        if not trash_dir.exists():
            return False

        candidates = sorted(
            [d for d in trash_dir.iterdir() if d.is_dir() and (d.name == adapter_id or d.name.startswith(f"{adapter_id}_"))],
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            return False

        chosen = candidates[0]
        os.replace(chosen, target)
        logger.info("Restored adapter '%s' from trash (%s)", adapter_id, chosen.name)
        return True

    def list_trash_adapters(self) -> list[str]:
        """Lists IDs of adapters currently in trash."""
        trash_dir = self.base_dir / ".trash"
        if not trash_dir.exists():
            return []
        items = []
        for d in sorted(trash_dir.iterdir()):
            if d.is_dir():
                items.append(d.name)
        return items

    def export_adapter_zip(self, adapter_id: str, export_zip_path: str | Path) -> Path:
        """Safely exports adapter files to a zip archive, strictly excluding any executable code."""
        target = self.get_adapter_path(adapter_id)
        if not target.is_dir():
            raise FileNotFoundError(f"Adapter '{adapter_id}' not found at {target}")

        manifest_file = target / "paxg_manifest.json"
        if not manifest_file.exists():
            raise FileNotFoundError(f"Manifest missing in adapter '{adapter_id}'")

        out_path = Path(export_zip_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        ALLOWED_EXTENSIONS = {".safetensors", ".json", ".sha256", ".md", ".txt", ".bin"}
        DISALLOWED_EXTENSIONS = {".py", ".pyc", ".pkl", ".pickle", ".exe", ".bat", ".ps1", ".sh", ".cmd", ".dll", ".so"}

        with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for item in sorted(target.iterdir()):
                if item.is_file():
                    ext = item.suffix.lower()
                    if ext in DISALLOWED_EXTENSIONS:
                        continue
                    if ext in ALLOWED_EXTENSIONS or item.name in (CHECKSUMS_FILENAME, "paxg_manifest.json", "README.md"):
                        zf.write(item, arcname=item.name)
        return out_path

    def import_adapter_zip(self, zip_path: str | Path, overwrite: bool = False) -> str:
        """Safely imports an adapter from a zip file with path traversal, extension, and SHA-256 verification."""
        zp = Path(zip_path)
        if not zp.is_file():
            raise FileNotFoundError(f"Zip file not found: {zp}")

        DANGEROUS_EXTENSIONS = {".py", ".pyc", ".pkl", ".pickle", ".exe", ".bat", ".ps1", ".sh", ".cmd", ".dll", ".so"}

        temp_dir = self.base_dir / f".tmp_import_{int(time.time() * 1000)}"
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            with zipfile.ZipFile(zp, "r") as zf:
                # 1. Path traversal and extension validation
                for member in zf.infolist():
                    name = member.filename
                    if ".." in name or name.startswith("/") or name.startswith("\\") or (len(name) > 1 and name[1] == ":"):
                        raise ValueError(f"Path traversal detected in zip member: '{name}'")
                    p_name = Path(name)
                    if p_name.suffix.lower() in DANGEROUS_EXTENSIONS:
                        raise ValueError(f"Dangerous file extension '{p_name.suffix}' in zip member: '{name}'")

                # 2. Extract into temp_dir
                zf.extractall(temp_dir)

            # Flatten if zip has a single root folder containing paxg_manifest.json
            inner_dirs = [d for d in temp_dir.iterdir() if d.is_dir()]
            if not (temp_dir / "paxg_manifest.json").exists() and len(inner_dirs) == 1:
                single_sub = inner_dirs[0]
                if (single_sub / "paxg_manifest.json").exists():
                    for item in single_sub.iterdir():
                        shutil.move(str(item), str(temp_dir))
                    shutil.rmtree(single_sub, ignore_errors=True)

            # 3. Check paxg_manifest.json
            manifest_file = temp_dir / "paxg_manifest.json"
            if not manifest_file.exists():
                raise ValueError("Import failed: 'paxg_manifest.json' missing from zip archive.")

            manifest = AdapterManifest.load_json(manifest_file)
            adapter_id = manifest.adapter_id
            target_dir = self.get_adapter_path(adapter_id)
            if target_dir.exists() and not overwrite:
                raise FileExistsError(f"Adapter '{adapter_id}' already exists. Set overwrite=True to replace.")

            # 4. Verify checksums.sha256 sidecar
            checksums_file = temp_dir / CHECKSUMS_FILENAME
            if not checksums_file.exists():
                raise ValueError(f"Import failed: required sidecar '{CHECKSUMS_FILENAME}' missing from zip archive.")

            with open(checksums_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split(maxsplit=1)
                    if len(parts) == 2:
                        exp_hash, fname = parts[0].strip(), parts[1].strip()
                        extracted_file = temp_dir / fname
                        if not extracted_file.exists():
                            raise FileNotFoundError(f"Missing file declared in checksums: '{fname}'")
                        act_hash = compute_file_sha256(extracted_file)
                        if act_hash != exp_hash:
                            raise ValueError(f"Checksum mismatch for '{fname}': expected {exp_hash}, got {act_hash}")

            # 5. Move to canonical target
            if target_dir.exists() and overwrite:
                shutil.rmtree(target_dir, ignore_errors=True)
            os.replace(temp_dir, target_dir)

            # Register in DB
            self.set_alias(adapter_id, adapter_id)
            return adapter_id

        except Exception:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise

