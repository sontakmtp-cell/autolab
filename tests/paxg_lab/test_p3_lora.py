"""Unit and integration tests for P3: Manual LoRA training, Adapter Manifest, and Store."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model

from paxg_lab.constants import (
    MODEL_REPO,
    MODEL_REVISION,
    TIMEFRAME_HORIZONS,
    get_horizon_for_timeframe,
)
from paxg_lab.data.features import FEATURE_SPECS
from paxg_lab.eval.predictor import TimesFM3Predictor
from paxg_lab.model.lora import build_lora_timesfm3, verify_lora_parameters
from paxg_lab.model.loss import combined_forecast_loss, pinball_loss
from paxg_lab.model.manifest import AdapterManifest
from paxg_lab.model.store import AdapterStore, compute_file_sha256
from paxg_lab.model.train_spec import TrainSpec
from paxg_lab.model.trainer import LoRATrainer, TrainingResult


# ---------------------------------------------------------------------------
# 1. TrainSpec Validation & Horizon Pair Enforcement
# ---------------------------------------------------------------------------


def test_train_spec_horizon_enforcement():
    """Validates that 1h is strictly locked to 24 and 4h is strictly locked to 6."""
    # Valid specs
    spec_1h = TrainSpec(timeframe="1h", horizon=24)
    assert spec_1h.timeframe == "1h"
    assert spec_1h.horizon == 24

    spec_4h = TrainSpec(timeframe="4h", horizon=6)
    assert spec_4h.timeframe == "4h"
    assert spec_4h.horizon == 6

    # Invalid cross-pairing
    with pytest.raises(ValueError, match="Horizon mismatch for timeframe '1h'"):
        TrainSpec(timeframe="1h", horizon=6)

    with pytest.raises(ValueError, match="Horizon mismatch for timeframe '4h'"):
        TrainSpec(timeframe="4h", horizon=24)

    with pytest.raises(ValueError, match="Unsupported timeframe '15m'"):
        TrainSpec(timeframe="15m", horizon=24)  # type: ignore


def test_train_spec_hyperparameter_bounds():
    """Validates bounds on rank, alpha, dropout, learning rate, batch size, etc."""
    # Invalid rank
    with pytest.raises(ValueError, match="LoRA rank 5 invalid"):
        TrainSpec(lora_r=5)

    # Valid default alpha is 2 * r
    spec = TrainSpec(lora_r=8, lora_alpha=None)
    assert spec.lora_alpha == 16

    # Invalid dropout
    with pytest.raises(ValueError, match="LoRA dropout .* out of bounds"):
        TrainSpec(lora_dropout=0.35)

    # Invalid context
    with pytest.raises(ValueError, match="Context length 100 not allowed"):
        TrainSpec(context_len=100)

    # Invalid learning rate
    with pytest.raises(ValueError, match="Learning rate .* out of safe bounds"):
        TrainSpec(learning_rate=0.01)

    # Effective batch size
    spec_bs = TrainSpec(batch_size=2, gradient_accumulation_steps=8)
    assert spec_bs.effective_batch_size == 16


# ---------------------------------------------------------------------------
# 2. Combined Pinball Loss Arithmetic
# ---------------------------------------------------------------------------


def test_combined_pinball_loss_computation():
    """Verifies combined forecast loss calculation in float32 and scale normalization."""
    b, v, h, q = 2, 1, 6, 9
    preds = torch.ones((b, v, h, q), dtype=torch.float32) * 100.0
    targets = torch.ones((b, v, h), dtype=torch.float32) * 105.0
    p0 = torch.tensor([100.0, 100.0], dtype=torch.float32)

    loss, metrics = combined_forecast_loss(
        predictions=preds,
        targets=targets,
        last_context_price=p0,
    )

    assert isinstance(loss, torch.Tensor)
    assert loss.dtype == torch.float32
    assert loss.item() > 0.0
    assert "norm_median_mae" in metrics
    assert "norm_pinball_loss" in metrics
    assert metrics["scale"] == pytest.approx(100.0, rel=1e-3)

    # Monotonicity test: larger error must strictly increase loss
    preds_worse = torch.ones((b, v, h, q), dtype=torch.float32) * 90.0
    loss_worse, _ = combined_forecast_loss(
        predictions=preds_worse,
        targets=targets,
        last_context_price=p0,
    )
    assert loss_worse.item() > loss.item()


# ---------------------------------------------------------------------------
# 3. AdapterManifest Serialization & Compatibility
# ---------------------------------------------------------------------------


def test_manifest_serialization_and_compatibility():
    """Tests saving/loading manifest JSON, compatibility verification, and rejection on mismatch."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_path = Path(tmpdir) / "paxg_manifest.json"

        manifest = AdapterManifest(
            adapter_id="test_adapter_1h",
            timeframe="1h",
            horizon=24,
            context_len=256,
            feature_set="B",
            feature_columns=list(FEATURE_SPECS["B"].columns),
            base_model_repo=MODEL_REPO,
            base_model_revision=MODEL_REVISION,
            best_epoch=3,
            best_val_loss=0.0125,
            file_hashes={"adapter_model.safetensors": "abcdef123456"},
        )

        manifest.save_json(manifest_path)
        loaded = AdapterManifest.load_json(manifest_path)

        assert loaded.adapter_id == "test_adapter_1h"
        assert loaded.timeframe == "1h"
        assert loaded.horizon == 24
        assert loaded.context_len == 256
        assert loaded.feature_set == "B"
        assert loaded.best_epoch == 3

        # Compatibility checks - success cases
        loaded.verify_compatibility(
            expected_timeframe="1h",
            expected_horizon=24,
            expected_context=256,
            expected_feature_set="B",
            expected_columns=list(FEATURE_SPECS["B"].columns),
            expected_base_revision=MODEL_REVISION,
        )

        # Incompatible timeframe
        with pytest.raises(ValueError, match="Incompatible timeframe"):
            loaded.verify_compatibility(expected_timeframe="4h")

        # Incompatible horizon
        with pytest.raises(ValueError, match="Incompatible horizon"):
            loaded.verify_compatibility(expected_horizon=6)

        # Incompatible context length
        with pytest.raises(ValueError, match="Incompatible context length"):
            loaded.verify_compatibility(expected_context=512)

        # Incompatible feature set
        with pytest.raises(ValueError, match="Incompatible feature set"):
            loaded.verify_compatibility(expected_feature_set="A")

        # Incompatible base model revision
        with pytest.raises(ValueError, match="Base model revision mismatch"):
            loaded.verify_compatibility(expected_base_revision="wrong_hash_123")


def test_manifest_in_sample_overlap_detection():
    """Tests detection and warning generation when evaluation period overlaps training data."""
    manifest = AdapterManifest(
        adapter_id="test_adapter_overlap",
        timeframe="1h",
        horizon=24,
        context_len=256,
        feature_set="B",
        feature_columns=["close"],
        training_range={
            "start_time_ms": 1700000000000,
            "end_time_ms": 1710000000000,
        },
    )

    # Case 1: Overlapping evaluation period
    is_overlap, msg = manifest.check_in_sample_overlap(
        eval_start_ms=1705000000000,
        eval_end_ms=1715000000000,
    )
    assert is_overlap is True
    assert "IN-SAMPLE OVERLAP DETECTED" in msg

    # Case 2: Strictly out-of-sample evaluation period (after training range)
    is_overlap_oos, msg_oos = manifest.check_in_sample_overlap(
        eval_start_ms=1710000000001,
        eval_end_ms=1720000000000,
    )
    assert is_overlap_oos is False
    assert "strictly out-of-sample" in msg_oos


# ---------------------------------------------------------------------------
# 4. AdapterStore: Atomic Saving, SHA-256 Verification & Tamper Detection
# ---------------------------------------------------------------------------


class MockLinearModule(nn.Module):
    """Small mock module to test PEFT saving and loading without heavy TimesFM download."""

    def __init__(self):
        super().__init__()
        self.query_proj = nn.Linear(16, 16)
        self.value_proj = nn.Linear(16, 16)

    def forward(self, x):
        return self.query_proj(x) + self.value_proj(x)


def test_adapter_store_atomic_save_and_load():
    """Tests atomic adapter saving (.tmp -> verify -> rename) and SHA-256 integrity check."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = AdapterStore(base_dir=tmpdir)

        # Build mock PEFT model
        base_mod = MockLinearModule()
        peft_cfg = LoraConfig(r=4, lora_alpha=8, target_modules=["query_proj", "value_proj"])
        peft_model = get_peft_model(base_mod, peft_cfg)

        manifest = AdapterManifest(
            adapter_id="mock_lora_1h",
            timeframe="1h",
            horizon=24,
            context_len=256,
            feature_set="B",
            feature_columns=list(FEATURE_SPECS["B"].columns),
            base_model_repo="mock_repo",
            base_model_revision=MODEL_REVISION,
        )

        # Save with smoke_test=False for mock module
        saved_dir = store.save_adapter(peft_model, manifest, smoke_test=False)
        assert saved_dir.is_dir()
        assert (saved_dir / "adapter_model.safetensors").exists()
        assert (saved_dir / "adapter_config.json").exists()
        assert (saved_dir / "paxg_manifest.json").exists()

        # Check no temporary directories remain
        tmp_dirs = [d for d in Path(tmpdir).iterdir() if d.name.startswith(".tmp_")]
        assert len(tmp_dirs) == 0

        # Load back adapter and verify SHA-256
        fresh_base = MockLinearModule()
        loaded_model, loaded_manifest = store.load_adapter(
            adapter_id_or_path="mock_lora_1h",
            base_model=fresh_base,
            expected_timeframe="1h",
            expected_horizon=24,
            expected_base_revision=MODEL_REVISION,
        )
        assert loaded_manifest.adapter_id == "mock_lora_1h"
        assert loaded_model is not None


def test_adapter_store_tamper_detection():
    """Verifies that tampering with saved weights causes SHA-256 mismatch rejection."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = AdapterStore(base_dir=tmpdir)

        base_mod = MockLinearModule()
        peft_cfg = LoraConfig(r=4, lora_alpha=8, target_modules=["query_proj", "value_proj"])
        peft_model = get_peft_model(base_mod, peft_cfg)

        manifest = AdapterManifest(
            adapter_id="tampered_lora_1h",
            timeframe="1h",
            horizon=24,
            context_len=256,
            feature_set="B",
            feature_columns=list(FEATURE_SPECS["B"].columns),
            base_model_repo="mock_repo",
            base_model_revision=MODEL_REVISION,
        )
        saved_dir = store.save_adapter(peft_model, manifest, smoke_test=False)

        # Tamper with the safetensors file: append garbage bytes
        weight_file = saved_dir / "adapter_model.safetensors"
        with open(weight_file, "ab") as f:
            f.write(b"CORRUPTED_TAMPER_BYTES")

        fresh_base = MockLinearModule()
        with pytest.raises(RuntimeError, match="SHA-256 integrity mismatch"):
            store.load_adapter("tampered_lora_1h", fresh_base)


def test_adapter_store_interrupted_process_resilience():
    """Simulates a killed process during saving: verifies existing adapter is safe and temp dir cleans up."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = AdapterStore(base_dir=tmpdir)

        # 1. Save a good adapter first
        base_mod = MockLinearModule()
        peft_cfg = LoraConfig(r=4, lora_alpha=8, target_modules=["query_proj", "value_proj"])
        peft_model = get_peft_model(base_mod, peft_cfg)

        manifest = AdapterManifest(
            adapter_id="good_adapter",
            timeframe="1h",
            horizon=24,
            context_len=256,
            feature_set="B",
            feature_columns=["close"],
            base_model_repo="mock_repo",
            base_model_revision=MODEL_REVISION,
        )
        store.save_adapter(peft_model, manifest, smoke_test=False)

        # 2. Simulate interrupted process: leave an orphaned .tmp_ directory with broken files
        orphan_dir = Path(tmpdir) / ".tmp_interrupted_adapter_123456"
        orphan_dir.mkdir()
        (orphan_dir / "partial_file.bin").write_bytes(b"unfinished garbage")

        # 3. Verify good adapter is completely intact and loads
        fresh_base = MockLinearModule()
        loaded, mf = store.load_adapter("good_adapter", fresh_base)
        assert mf.adapter_id == "good_adapter"

        # 4. Clean up stale temp dirs (using 0 age to clean immediately)
        cleaned = store.cleanup_stale_temp_dirs(max_age_seconds=0.0)
        assert orphan_dir.name in cleaned
        assert not orphan_dir.exists()
        assert (Path(tmpdir) / "good_adapter").exists()


# ---------------------------------------------------------------------------
# 5. Base Restoration Invariant Test (Base -> Adapter A -> Adapter B -> Base)
# ---------------------------------------------------------------------------


def test_base_restoration_invariant():
    """Tests invariant: Base -> Adapter A -> Adapter B -> Base produces identical output with zero weight leak."""
    base_mod = MockLinearModule()

    # Save original base state dict
    orig_state = copy.deepcopy(base_mod.state_dict())

    # Create dummy input
    dummy_input = torch.randn(2, 16)
    with torch.no_grad():
        out_base_0 = base_mod(dummy_input).clone()

    # 1. Attach Adapter A and run
    peft_a = get_peft_model(
        base_mod,
        LoraConfig(r=4, lora_alpha=8, target_modules=["query_proj", "value_proj"]),
        adapter_name="adapter_a",
    )
    with torch.no_grad():
        out_a = peft_a(dummy_input)

    # 2. Add Adapter B and run
    peft_a.add_adapter(
        "adapter_b",
        LoraConfig(r=8, lora_alpha=16, target_modules=["query_proj", "value_proj"]),
    )
    peft_a.set_adapter("adapter_b")
    with torch.no_grad():
        out_b = peft_a(dummy_input)

    # 3. Unload back to Base model
    restored_base = peft_a.unload()
    with torch.no_grad():
        out_base_1 = restored_base(dummy_input)

    # Verification: output matches exactly (|out_0 - out_1| == 0)
    diff = torch.max(torch.abs(out_base_0 - out_base_1)).item()
    assert diff == pytest.approx(0.0, abs=1e-6), f"Base restoration output difference: {diff}"

    # Verify restored weights match original weights bit-for-bit
    for k, v in orig_state.items():
        restored_v = restored_base.state_dict()[k]
        assert torch.equal(v, restored_v), f"Weight mismatch in parameter '{k}' after adapter unload!"

    # Verify no 'lora' keys exist in restored base model
    assert not any("lora" in k.lower() for k in restored_base.state_dict().keys())
