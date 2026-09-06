"""Safe and atomic storage for LoRA adapters with sidecar SHA-256 verification and cleanup."""

from __future__ import annotations

import hashlib
import logging
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import time
from typing import Any
import zipfile

from safetensors import safe_open
import torch
import torch.nn as nn
from peft import PeftModel

from ..constants import (
    ALLOWED_CONTEXT_LENGTHS,
    MODEL_REPO,
    MODEL_REVISION,
    get_horizon_for_timeframe,
)
from ..data.features import FEATURE_SPECS
from .lora import load_lora_adapter
from .manifest import AdapterManifest

logger = logging.getLogger(__name__)

DEFAULT_ADAPTER_STORE_DIR = Path("var/paxg_lab/adapters")
DEFAULT_DB_PATH = Path("var/paxg_lab/paxg_lab.db")
CHECKSUMS_FILENAME = "checksums.sha256"
ADAPTER_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_\-\.]{0,127}$")

MAX_MEMBER_UNCOMPRESSED_BYTES = 500 * 1024 * 1024  # 500 MB
MAX_TOTAL_UNCOMPRESSED_BYTES = 1024 * 1024 * 1024   # 1 GB
SAFE_EXTENSIONS_ALLOWLIST = {".safetensors", ".json", ".sha256", ".md", ".txt"}
FORBIDDEN_EXTENSIONS = {
    ".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle",
    ".py", ".pyc", ".exe", ".bat", ".ps1", ".sh", ".cmd", ".dll", ".so"
}


def validate_adapter_id(adapter_id: str, base_dir: Path | None = None) -> str:
    """Validates that adapter_id is a safe slug and strictly does not escape base_dir."""
    if not isinstance(adapter_id, str) or not adapter_id.strip():
        raise ValueError(f"Invalid adapter_id: must be a non-empty string, got '{adapter_id}'")
    clean_id = adapter_id.strip()
    if ".." in clean_id or "/" in clean_id or "\\" in clean_id or ":" in clean_id:
        raise ValueError(f"Path traversal characters detected in adapter_id: '{clean_id}'")
    if not ADAPTER_ID_PATTERN.match(clean_id):
        raise ValueError(f"Invalid adapter_id format: '{clean_id}'. Must be an alphanumeric slug.")
    if clean_id in (".", ".."):
        raise ValueError(f"Reserved path component cannot be used as adapter_id: '{clean_id}'")

    if base_dir is not None:
        resolved_base = Path(base_dir).resolve()
        resolved_target = (Path(base_dir) / clean_id).resolve()
        try:
            rel = resolved_target.relative_to(resolved_base)
            if rel == Path(".") or rel.parts == ():
                raise ValueError(f"adapter_id cannot resolve to base directory itself: '{clean_id}'")
        except ValueError:
            raise ValueError(f"Path traversal detected: '{clean_id}' escapes base directory {base_dir}")
    return clean_id


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


def validate_lora_weights_semantics(safetensors_path: Path, peft_cfg: dict) -> None:
    """Validates that a safetensors file contains a valid, well-formed LoRA state dict
    matching the configured rank r and target_modules."""
    expected_r = peft_cfg.get("r")
    target_modules = peft_cfg.get("target_modules", [])
    if isinstance(target_modules, str):
        target_modules = [target_modules]
    target_modules_set = set(target_modules) if target_modules else set()

    with safe_open(str(safetensors_path), framework="pt") as f:
        keys = list(f.keys())
        if not keys:
            raise ValueError(f"Safetensors file '{safetensors_path.name}' contains no tensor keys.")

        # Match LoRA parameter naming pattern, e.g.:
        # base_model.model.transformer_stack.layers.0.seq_attn.query_proj.lora_A.weight
        # base_model.model.transformer_stack.layers.0.seq_attn.query_proj.lora_A.default.weight
        lora_pattern = re.compile(r"^(.*?)\.(lora_[AB])(?:\.[^.]+)?\.weight$")
        lora_a_modules: dict[str, tuple[str, Any]] = {}
        lora_b_modules: dict[str, tuple[str, Any]] = {}

        for k in keys:
            m = lora_pattern.match(k)
            if not m:
                raise ValueError(
                    f"Weight key '{k}' does not conform to standard PEFT LoRA parameter naming pattern."
                )
            prefix, ab = m.group(1), m.group(2)
            tensor = f.get_tensor(k)
            shape = tensor.shape
            if len(shape) != 2:
                raise ValueError(
                    f"LoRA parameter '{k}' must be a 2D weight matrix, got shape {shape}."
                )
            if ab == "lora_A":
                lora_a_modules[prefix] = (k, shape)
            else:
                lora_b_modules[prefix] = (k, shape)

        if not lora_a_modules or not lora_b_modules:
            raise ValueError(
                f"Adapter weights in '{safetensors_path.name}' do not contain both lora_A and lora_B tensors."
            )

        # Verify A and B pairs match and verify rank r
        all_prefixes = set(lora_a_modules.keys()) | set(lora_b_modules.keys())
        for prefix in all_prefixes:
            if prefix not in lora_a_modules:
                raise ValueError(f"Missing matching lora_A tensor for LoRA module: '{prefix}'")
            if prefix not in lora_b_modules:
                raise ValueError(f"Missing matching lora_B tensor for LoRA module: '{prefix}'")

            k_a, shape_a = lora_a_modules[prefix]
            k_b, shape_b = lora_b_modules[prefix]

            if shape_a[0] != expected_r:
                raise ValueError(
                    f"Rank mismatch in '{k_a}': expected rank {expected_r}, got shape {shape_a}."
                )
            if shape_b[1] != expected_r:
                raise ValueError(
                    f"Rank mismatch in '{k_b}': expected rank {expected_r}, got shape {shape_b}."
                )

            # Check that module prefix targets one of the configured target_modules
            if target_modules_set:
                matched_target = any(
                    prefix.endswith(f".{tm}") or prefix == tm or f".{tm}." in prefix
                    for tm in target_modules_set
                )
                if not matched_target:
                    raise ValueError(
                        f"LoRA module '{prefix}' does not match any configured target_modules: {sorted(target_modules_set)}."
                    )

        # Verify all configured target_modules are covered in weights
        if target_modules_set:
            covered_targets = set()
            for tm in target_modules_set:
                for prefix in lora_a_modules.keys():
                    if prefix.endswith(f".{tm}") or prefix == tm or f".{tm}." in prefix:
                        covered_targets.add(tm)
            missing_targets = target_modules_set - covered_targets
            if missing_targets:
                raise ValueError(
                    f"Missing LoRA weights for configured target_modules: {sorted(missing_targets)}."
                )


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
        """Returns the canonical directory path for an adapter after strictly validating adapter_id."""
        clean_id = validate_adapter_id(adapter_id, base_dir=self.base_dir)
        return self.base_dir / clean_id

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

        with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for item in sorted(target.iterdir()):
                if item.is_file():
                    ext = item.suffix.lower()
                    if ext in FORBIDDEN_EXTENSIONS:
                        continue
                    if ext in SAFE_EXTENSIONS_ALLOWLIST or item.name in (CHECKSUMS_FILENAME, "paxg_manifest.json", "README.md"):
                        zf.write(item, arcname=item.name)
        return out_path

    def import_adapter_zip(
        self,
        zip_path: str | Path,
        overwrite: bool = False,
        base_model: nn.Module | None = None,
    ) -> str:
        """Safely imports an adapter from a zip file with path traversal, extension allowlist,
        decompression size limits, static compatibility checks, and SHA-256 verification."""
        zp = Path(zip_path)
        if not zp.is_file():
            raise FileNotFoundError(f"Zip file not found: {zp}")

        temp_dir = self.base_dir / f".tmp_import_{int(time.time() * 1000)}"
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            with zipfile.ZipFile(zp, "r") as zf:
                # 1. Inspect members: traversal, strict allowlist, and size limits (Zip bomb protection)
                total_uncompressed = 0
                for member in zf.infolist():
                    name = member.filename
                    name_norm = name.replace("\\", "/")
                    if (
                        ".." in name_norm
                        or name_norm.startswith("/")
                        or name.startswith("\\")
                        or (len(name) > 1 and name[1] == ":")
                        or name_norm.startswith("//")
                        or name.startswith("\\\\")
                    ):
                        raise ValueError(f"Path traversal detected in zip member: '{name}'")

                    if name.endswith("/") or name.endswith("\\"):
                        continue

                    p_name = Path(name)
                    ext = p_name.suffix.lower()

                    # Strict allowlist: reject forbidden and non-allowlisted extensions
                    if ext in FORBIDDEN_EXTENSIONS or ext not in SAFE_EXTENSIONS_ALLOWLIST:
                        raise ValueError(
                            f"Disallowed or dangerous file extension '{ext}' in zip member: '{name}'. "
                            "Only safetensors, json, sha256, and text files are allowed."
                        )

                    # Explicit reject for PyTorch/pickle weight files (.bin, .pt, .pth)
                    if p_name.name.lower() in ("adapter_model.bin", "pytorch_model.bin", "model.pt", "model.pth"):
                        raise ValueError(f"Pickle/PyTorch binary weights forbidden in adapter import: '{name}'")

                    # Size check per member and total
                    if member.file_size > MAX_MEMBER_UNCOMPRESSED_BYTES:
                        raise ValueError(
                            f"Zip member '{name}' exceeds uncompressed size limit "
                            f"({member.file_size} > {MAX_MEMBER_UNCOMPRESSED_BYTES} bytes)"
                        )
                    total_uncompressed += member.file_size
                    if total_uncompressed > MAX_TOTAL_UNCOMPRESSED_BYTES:
                        raise ValueError(
                            f"Total uncompressed archive size exceeds limit "
                            f"({total_uncompressed} > {MAX_TOTAL_UNCOMPRESSED_BYTES} bytes)"
                        )

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

            # 3. Check mandatory files existence and non-emptiness
            MANDATORY_ADAPTER_FILES = (
                "paxg_manifest.json",
                CHECKSUMS_FILENAME,
                "adapter_model.safetensors",
                "adapter_config.json",
            )
            for mfile in MANDATORY_ADAPTER_FILES:
                fpath = temp_dir / mfile
                if not fpath.is_file() or fpath.stat().st_size == 0:
                    raise ValueError(f"Import failed: required adapter file '{mfile}' missing or empty in zip archive.")

            # 4. Check and validate adapter_config.json
            adapter_config_file = temp_dir / "adapter_config.json"
            try:
                with open(adapter_config_file, "r", encoding="utf-8") as f:
                    peft_cfg = json.load(f)
                if not isinstance(peft_cfg, dict) or not peft_cfg:
                    raise ValueError("must be a non-empty JSON object.")
            except Exception as err:
                raise ValueError(f"Corrupted or invalid 'adapter_config.json': {err}") from err

            # Validate essential LoRA PEFT config fields
            peft_type = str(peft_cfg.get("peft_type", "")).strip().upper()
            if peft_type != "LORA":
                raise ValueError(
                    f"Invalid PEFT config in 'adapter_config.json': 'peft_type' must be 'LORA', got '{peft_cfg.get('peft_type')}'."
                )
            r_val = peft_cfg.get("r")
            if not isinstance(r_val, int) or r_val <= 0:
                raise ValueError(
                    f"Invalid LoRA config in 'adapter_config.json': 'r' (rank) must be a positive integer, got '{r_val}'."
                )
            if "lora_alpha" in peft_cfg:
                alpha_val = peft_cfg["lora_alpha"]
                if not isinstance(alpha_val, (int, float)) or alpha_val <= 0:
                    raise ValueError(
                        f"Invalid LoRA config in 'adapter_config.json': 'lora_alpha' must be positive, got '{alpha_val}'."
                    )
            if "target_modules" in peft_cfg:
                tm_val = peft_cfg["target_modules"]
                if not isinstance(tm_val, (list, tuple, str, set)) or not tm_val:
                    raise ValueError(
                        f"Invalid LoRA config in 'adapter_config.json': 'target_modules' cannot be empty, got '{tm_val}'."
                    )

            # 4.5. Validate safetensors container and tensor keys using safetensors library
            safetensors_files = [f for f in temp_dir.rglob("*.safetensors") if f.is_file()]
            for sf in safetensors_files:
                try:
                    with safe_open(str(sf), framework="pt") as f:
                        tensor_keys = list(f.keys())
                    if not tensor_keys:
                        raise ValueError(f"Safetensors file '{sf.name}' contains no tensor keys.")
                except Exception as err:
                    raise ValueError(f"Corrupted or invalid safetensors weights file '{sf.name}': {err}") from err

            # 4.6. Validate semantic structure of LoRA PEFT weights (A/B pairs, rank r, target_modules)
            validate_lora_weights_semantics(temp_dir / "adapter_model.safetensors", peft_cfg)

            # 4.7. If base_model provided, test real load compatibility
            if base_model is not None:
                try:
                    load_lora_adapter(base_model, temp_dir)
                except Exception as err:
                    raise ValueError(f"Failed to load adapter weights into base model: {err}") from err

            # 5. Check paxg_manifest.json and enforce strict base model provenance
            manifest_file = temp_dir / "paxg_manifest.json"
            try:
                with open(manifest_file, "r", encoding="utf-8") as f:
                    raw_manifest = json.load(f)
                if not isinstance(raw_manifest, dict):
                    raise ValueError("root must be a JSON object.")
            except Exception as err:
                raise ValueError(f"Corrupted or invalid 'paxg_manifest.json': {err}") from err

            # Strict provenance: base_model_repo and base_model_revision must be explicitly present and exact match
            if "base_model_repo" not in raw_manifest or not raw_manifest["base_model_repo"]:
                raise ValueError("Incompatible manifest: missing mandatory 'base_model_repo' provenance.")
            if "base_model_revision" not in raw_manifest or not raw_manifest["base_model_revision"]:
                raise ValueError("Incompatible manifest: missing mandatory 'base_model_revision' provenance.")

            if raw_manifest["base_model_repo"] != MODEL_REPO:
                raise ValueError(
                    f"Incompatible base_model_repo '{raw_manifest['base_model_repo']}'. "
                    f"Expected strictly '{MODEL_REPO}'."
                )
            if raw_manifest["base_model_revision"] != MODEL_REVISION:
                raise ValueError(
                    f"Incompatible base_model_revision '{raw_manifest['base_model_revision']}'. "
                    f"Expected strictly '{MODEL_REVISION}'."
                )

            manifest = AdapterManifest.load_json(manifest_file)
            adapter_id = validate_adapter_id(manifest.adapter_id, self.base_dir)
            target_dir = self.get_adapter_path(adapter_id)
            if target_dir.exists():
                if not overwrite:
                    raise FileExistsError(f"Adapter '{adapter_id}' already exists. Set overwrite=True to replace.")
                meta = self.get_registry_metadata(adapter_id)
                if meta.get("is_recommended"):
                    raise ValueError(f"Không thể ghi đè adapter '{adapter_id}' vì đang là adapter khuyến nghị.")
                if self.is_pinned(adapter_id):
                    raise ValueError(f"Không thể ghi đè adapter '{adapter_id}' vì đã được ghim chống xóa.")

            # 6. Static compatibility validation
            if manifest.timeframe not in ("1h", "4h"):
                raise ValueError(f"Incompatible adapter timeframe '{manifest.timeframe}'. Must be '1h' or '4h'.")
            expected_horizon = get_horizon_for_timeframe(manifest.timeframe)
            if manifest.horizon != expected_horizon:
                raise ValueError(
                    f"Incompatible adapter horizon {manifest.horizon} for timeframe '{manifest.timeframe}'. "
                    f"Expected {expected_horizon}."
                )
            if manifest.context_len not in ALLOWED_CONTEXT_LENGTHS:
                raise ValueError(
                    f"Incompatible adapter context_len {manifest.context_len}. "
                    f"Must be one of {ALLOWED_CONTEXT_LENGTHS}."
                )
            if manifest.feature_set not in FEATURE_SPECS:
                raise ValueError(
                    f"Incompatible adapter feature_set '{manifest.feature_set}'. "
                    f"Must be one of {list(FEATURE_SPECS.keys())}."
                )
            expected_feature_columns = list(FEATURE_SPECS[manifest.feature_set].columns)
            if list(manifest.feature_columns) != expected_feature_columns:
                raise ValueError(
                    f"Incompatible feature_columns for feature_set '{manifest.feature_set}': "
                    f"expected {expected_feature_columns}, got {manifest.feature_columns}."
                )

            # 7. Check forbidden weight files
            if (temp_dir / "adapter_model.bin").exists():
                raise ValueError("Pickle/PyTorch binary weights ('adapter_model.bin') forbidden in adapter import.")

            # 8. Verify checksums.sha256 sidecar
            checksums_file = temp_dir / CHECKSUMS_FILENAME
            verified_files: set[Path] = set()
            with open(checksums_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split(maxsplit=1)
                    if len(parts) == 2:
                        exp_hash, fname = parts[0].strip(), parts[1].strip()
                        # Strict filename validation: reject traversal, absolute paths, drive letters, UNC
                        fname_clean = fname.replace("\\", "/")
                        if (
                            ".." in fname_clean
                            or fname_clean.startswith("/")
                            or fname.startswith("\\")
                            or (len(fname) > 1 and fname[1] == ":")
                            or fname.startswith("//")
                            or fname.startswith("\\\\")
                        ):
                            raise ValueError(f"Path traversal detected in checksums filename: '{fname}'")

                        extracted_file = (temp_dir / fname).resolve()
                        try:
                            extracted_file.relative_to(temp_dir.resolve())
                        except ValueError:
                            raise ValueError(f"Path traversal detected in checksums filename: '{fname}'")

                        if not extracted_file.is_file():
                            raise FileNotFoundError(f"Missing file declared in checksums: '{fname}'")
                        act_hash = compute_file_sha256(extracted_file)
                        if act_hash != exp_hash:
                            raise ValueError(f"Checksum mismatch for '{fname}': expected {exp_hash}, got {act_hash}")
                        verified_files.add(extracted_file)

            # Verify that all mandatory files are covered by checksums.sha256
            for mfile in ("paxg_manifest.json", "adapter_model.safetensors", "adapter_config.json"):
                if (temp_dir / mfile).resolve() not in verified_files:
                    raise ValueError(f"Import failed: mandatory file '{mfile}' is not declared or verified in checksums.sha256.")

            # Archive completeness: ensure all files in temp_dir (except checksums.sha256) are covered by checksums
            all_files_in_temp = {f.resolve() for f in temp_dir.rglob("*") if f.is_file() and f.name != CHECKSUMS_FILENAME}
            uncovered_files = all_files_in_temp - verified_files
            if uncovered_files:
                uncovered_names = sorted([str(f.relative_to(temp_dir.resolve())) for f in uncovered_files])
                raise ValueError(f"Import failed: archive contains unverified files not covered by checksums: {uncovered_names}")

            # 9. Atomic move to canonical target and register in DB
            backup_dir = None
            if target_dir.exists() and overwrite:
                backup_dir = target_dir.parent / f".bak_replace_{adapter_id}_{int(time.time() * 1000)}"
                if backup_dir.exists():
                    shutil.rmtree(backup_dir, ignore_errors=True)
                os.replace(target_dir, backup_dir)

            try:
                os.replace(temp_dir, target_dir)

                # 10. Register in DB (strictly after validations and move have completed)
                self.set_alias(adapter_id, adapter_id)

                # 11. Success! Cleanup backup_dir only AFTER DB mutation has succeeded
                if backup_dir and backup_dir.exists():
                    shutil.rmtree(backup_dir, ignore_errors=True)
                return adapter_id

            except Exception:
                # Rollback filesystem
                if backup_dir and backup_dir.exists():
                    if target_dir.exists():
                        shutil.rmtree(target_dir, ignore_errors=True)
                    os.replace(backup_dir, target_dir)
                elif target_dir.exists() and backup_dir is None:
                    shutil.rmtree(target_dir, ignore_errors=True)
                raise

        except Exception:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise

