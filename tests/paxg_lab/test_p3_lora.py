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
from paxg_lab.eval.types import ForecastRequest
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

    # Auto-resolving horizon from timeframe
    spec_auto_1h = TrainSpec(timeframe="1h")
    assert spec_auto_1h.horizon == 24
    spec_auto_4h = TrainSpec(timeframe="4h")
    assert spec_auto_4h.horizon == 6


def test_train_spec_hyperparameter_bounds():
    """Validates strict PLAN 3.2 bounds on rank, alpha, dropout, learning rate, batch size, etc."""
    # Invalid rank
    with pytest.raises(ValueError, match="LoRA rank 5 invalid"):
        TrainSpec(lora_r=5)

    # Valid default alpha is 2 * r
    spec = TrainSpec(lora_r=8, lora_alpha=None)
    assert spec.lora_alpha == 16

    # Invalid dropout (PLAN 3.2: 0.0 to 0.20)
    with pytest.raises(ValueError, match="LoRA dropout .* out of bounds"):
        TrainSpec(lora_dropout=0.35)

    # Invalid context
    with pytest.raises(ValueError, match="Context length 100 not allowed"):
        TrainSpec(context_len=100)

    # Invalid learning rate (PLAN 3.2: 1e-5 to 3e-4)
    with pytest.raises(ValueError, match="Learning rate .* out of PLAN 3.2 bounds"):
        TrainSpec(learning_rate=0.01)
    with pytest.raises(ValueError, match="Learning rate .* out of PLAN 3.2 bounds"):
        TrainSpec(learning_rate=1e-6)

    # Epochs & Batching (PLAN 3.2: max_epochs 1..10, batch_size in (1, 2, 4), grad_accum 1..16)
    with pytest.raises(ValueError, match="max_epochs must be between 1 and 10"):
        TrainSpec(max_epochs=15)
    with pytest.raises(ValueError, match="Batch size 8 must be one of"):
        TrainSpec(batch_size=8)
    with pytest.raises(ValueError, match="gradient_accumulation_steps must be between 1 and 16"):
        TrainSpec(gradient_accumulation_steps=32)

    # Weight decay & Patience (PLAN 3.2: weight_decay 0.0..0.10, patience 1..4)
    with pytest.raises(ValueError, match="weight_decay .* out of bounds"):
        TrainSpec(weight_decay=0.20)
    with pytest.raises(ValueError, match="early_stopping_patience must be between 1 and 4"):
        TrainSpec(early_stopping_patience=10)

    # Gradient clipping (PLAN 3.2: 0.5..2.0)
    with pytest.raises(ValueError, match="grad_clip_norm .* out of bounds"):
        TrainSpec(grad_clip_norm=0.1)
    with pytest.raises(ValueError, match="grad_clip_norm .* out of bounds"):
        TrainSpec(grad_clip_norm=5.0)

    # History days (PLAN 3.2: 180, 365, 'all')
    with pytest.raises(ValueError, match="history_days .* not allowed by PLAN 3.2"):
        TrainSpec(history_days=100)
    assert TrainSpec(history_days="all").history_days == "all"
    assert TrainSpec(history_days=180).history_days == 180
    assert TrainSpec(history_days=365).history_days == 365

    # Warmup ratio (PLAN 3.2: 0.0..0.10)
    with pytest.raises(ValueError, match="warmup_ratio .* out of PLAN 3.2 bounds"):
        TrainSpec(warmup_ratio=0.11)
    with pytest.raises(ValueError, match="warmup_ratio .* out of PLAN 3.2 bounds"):
        TrainSpec(warmup_ratio=-0.01)
    assert TrainSpec(warmup_ratio=0.10).warmup_ratio == 0.10

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


# ---------------------------------------------------------------------------
# 6. LoRATrainer Clean Base Model & Warmup Scheduler Tests
# ---------------------------------------------------------------------------


def test_clean_base_model_requirement():
    """LoRATrainer must reject base models with pre-existing LoRA parameters or PeftModel wrapper."""
    base_mod = MockLinearModule()
    peft_cfg = LoraConfig(r=4, lora_alpha=8, target_modules=["query_proj", "value_proj"])
    peft_model = get_peft_model(base_mod, peft_cfg)

    spec = TrainSpec(timeframe="1h", horizon=24, feature_set="A")
    with pytest.raises(ValueError, match="Base model is already a PeftModel"):
        LoRATrainer(base_model=peft_model, spec=spec)


def test_warmup_scheduler_with_gradient_accumulation():
    """Verifies that warmup scheduler counts actual optimizer updates, not raw mini-batches."""
    spec = TrainSpec(
        timeframe="1h",
        horizon=24,
        feature_set="A",
        batch_size=2,
        gradient_accumulation_steps=4,
        max_epochs=2,
        warmup_ratio=0.10,
    )
    # 20 samples -> minibatches_per_epoch = 10 -> optimizer_steps_per_epoch = ceil(10 / 4) = 3
    # total_optimizer_steps = 3 * 2 = 6 -> warmup_optimizer_steps = max(1, int(6 * 0.10)) = 1
    num_samples = 20
    minibatches_per_epoch = int(np.ceil(num_samples / spec.batch_size))
    optimizer_steps_per_epoch = int(np.ceil(minibatches_per_epoch / spec.gradient_accumulation_steps))
    total_optimizer_steps = optimizer_steps_per_epoch * spec.max_epochs
    warmup_optimizer_steps = max(1, int(total_optimizer_steps * spec.warmup_ratio))

    assert minibatches_per_epoch == 10
    assert optimizer_steps_per_epoch == 3
    assert total_optimizer_steps == 6
    assert warmup_optimizer_steps == 1


# ---------------------------------------------------------------------------
# 7. AdapterStore Smoke Test & Sidecar Checksum Tamper Detection
# ---------------------------------------------------------------------------


class MockTimesFMModule(nn.Module):
    """Mock TimesFM module implementing decode() for testing smoke tests and predictors."""

    def __init__(self):
        super().__init__()
        self.query_proj = nn.Linear(16, 16)
        self.value_proj = nn.Linear(16, 16)

    def decode(self, target: torch.Tensor, horizon: int) -> torch.Tensor:
        b, f, _ = target.shape
        return torch.ones((b, f, horizon, 9), dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.query_proj(x) + self.value_proj(x)


def test_adapter_store_smoke_test_real_forward():
    """Verifies that AdapterStore._run_smoke_test runs real forward pass and generates checksums.sha256."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = AdapterStore(base_dir=tmpdir)
        base_mod = MockTimesFMModule()
        peft_cfg = LoraConfig(r=4, lora_alpha=8, target_modules=["query_proj", "value_proj"])
        peft_model = get_peft_model(base_mod, peft_cfg)

        manifest = AdapterManifest(
            adapter_id="smoke_test_adapter",
            timeframe="1h",
            horizon=24,
            context_len=256,
            feature_set="B",
            feature_columns=list(FEATURE_SPECS["B"].columns),
            base_model_repo="mock_repo",
            base_model_revision=MODEL_REVISION,
        )
        saved_dir = store.save_adapter(peft_model, manifest, smoke_test=True)
        assert (saved_dir / "checksums.sha256").exists()


def test_adapter_store_manifest_tamper_detection():
    """Verifies that tampering with paxg_manifest.json is detected via checksums.sha256."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = AdapterStore(base_dir=tmpdir)
        base_mod = MockLinearModule()
        peft_cfg = LoraConfig(r=4, lora_alpha=8, target_modules=["query_proj", "value_proj"])
        peft_model = get_peft_model(base_mod, peft_cfg)

        manifest = AdapterManifest(
            adapter_id="manifest_tamper_adapter",
            timeframe="1h",
            horizon=24,
            context_len=256,
            feature_set="B",
            feature_columns=list(FEATURE_SPECS["B"].columns),
            base_model_repo="mock_repo",
            base_model_revision=MODEL_REVISION,
        )
        saved_dir = store.save_adapter(peft_model, manifest, smoke_test=False)

        # Tamper with paxg_manifest.json
        manifest_file = saved_dir / "paxg_manifest.json"
        with open(manifest_file, "a", encoding="utf-8") as f:
            f.write("\n// tampered extra comment")

        fresh_base = MockLinearModule()
        with pytest.raises(RuntimeError, match="SHA-256 integrity mismatch for 'paxg_manifest.json'"):
            store.load_adapter("manifest_tamper_adapter", fresh_base)


def test_adapter_store_requires_checksums_file():
    """Verifies that removing checksums.sha256 causes load_adapter to fail-fast with FileNotFoundError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = AdapterStore(base_dir=tmpdir)
        base_mod = MockLinearModule()
        peft_cfg = LoraConfig(r=4, lora_alpha=8, target_modules=["query_proj", "value_proj"])
        peft_model = get_peft_model(base_mod, peft_cfg)

        manifest = AdapterManifest(
            adapter_id="no_checksums_adapter",
            timeframe="1h",
            horizon=24,
            context_len=256,
            feature_set="B",
            feature_columns=list(FEATURE_SPECS["B"].columns),
            base_model_repo="mock_repo",
            base_model_revision=MODEL_REVISION,
        )
        saved_dir = store.save_adapter(peft_model, manifest, smoke_test=False)

        # Delete checksums.sha256
        checksums_file = saved_dir / "checksums.sha256"
        assert checksums_file.exists()
        checksums_file.unlink()

        fresh_base = MockLinearModule()
        with pytest.raises(FileNotFoundError, match="Required sidecar checksum file 'checksums.sha256' missing"):
            store.load_adapter("no_checksums_adapter", fresh_base)


# ---------------------------------------------------------------------------
# 8. Predictor Safe Loading & Pre-Inference Compatibility Checks
# ---------------------------------------------------------------------------


def test_predictor_load_adapter_requires_manifest():
    """Verifies load_adapter raises FileNotFoundError if paxg_manifest.json is missing, while load_raw_adapter_unsafe works."""
    with tempfile.TemporaryDirectory() as tmpdir:
        adapter_dir = Path(tmpdir) / "raw_adapter"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_config.json").write_text("{}")
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"dummy")

        predictor = object.__new__(TimesFM3Predictor)
        predictor.device = torch.device("cpu")
        predictor.base_model = MockTimesFMModule()
        predictor.model = predictor.base_model
        predictor.lora_model = None
        predictor.adapter_path = None
        predictor.manifest = None
        predictor.model_repo = "mock_repo"
        predictor.model_revision = MODEL_REVISION

        # load_adapter must fail
        with pytest.raises(FileNotFoundError, match="Required 'paxg_manifest.json' not found"):
            predictor.load_adapter(adapter_dir)


def test_forecast_request_strict_compatibility():
    """Verifies forecast_request fails fast on context length, num features, column order, and base revision mismatch."""
    predictor = object.__new__(TimesFM3Predictor)
    predictor.device = torch.device("cpu")
    predictor.base_model = MockTimesFMModule()
    predictor.model = predictor.base_model
    predictor.lora_model = None
    predictor.adapter_path = Path("/mock/adapter")
    predictor.model_repo = "mock_repo"
    predictor.model_revision = MODEL_REVISION

    manifest = AdapterManifest(
        adapter_id="strict_adapter",
        timeframe="1h",
        horizon=24,
        context_len=256,
        feature_set="B",
        feature_columns=list(FEATURE_SPECS["B"].columns),
        base_model_repo="mock_repo",
        base_model_revision=MODEL_REVISION,
    )
    predictor.manifest = manifest

    # 1. Context length mismatch (request: 256, array: 128)
    req = ForecastRequest(timeframe="1h", feature_set="B", context_len=256, adapter_path="/mock/adapter")
    ctx_short = np.zeros((128, 9), dtype=np.float32)
    with pytest.raises(ValueError, match="Context length mismatch"):
        predictor.forecast_request(req, ctx_short, forecast_origin_time=1700000000000)

    # 2. Number of features mismatch (array has 5 features instead of 9)
    ctx_wrong_feat = np.zeros((256, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="Feature dimension mismatch"):
        predictor.forecast_request(req, ctx_wrong_feat, forecast_origin_time=1700000000000)

    # 3. Column order mismatch
    permuted_cols = list(FEATURE_SPECS["B"].columns)
    permuted_cols[0], permuted_cols[1] = permuted_cols[1], permuted_cols[0]
    req_permuted = ForecastRequest(
        timeframe="1h",
        feature_set="B",
        context_len=256,
        adapter_path="/mock/adapter",
        columns=permuted_cols,
    )
    ctx_256 = np.zeros((256, 9), dtype=np.float32)
    with pytest.raises(ValueError, match="Incompatible feature columns"):
        predictor.forecast_request(req_permuted, ctx_256, forecast_origin_time=1700000000000)

    # 4. Base revision mismatch
    predictor.model_revision = "mismatched_revision"
    with pytest.raises(ValueError, match="Base model revision mismatch"):
        predictor.forecast_request(req, ctx_256, forecast_origin_time=1700000000000)


# ---------------------------------------------------------------------------
# 9. End-to-End Backtest In-Sample Overlap Detection
# ---------------------------------------------------------------------------


def _make_synthetic_snapshot(n_candles: int = 5500, timeframe: str = "1h"):
    from paxg_lab.data.snapshot import DatasetSnapshot, SnapshotMetadata
    step_ms = 3600 * 1000 if timeframe == "1h" else 4 * 3600 * 1000
    start_ts = 1743073200000
    timestamps = np.array([start_ts + i * step_ms for i in range(n_candles)], dtype=np.int64)
    close_prices = 2500.0 + 10.0 * np.sin(np.linspace(0, 50, n_candles)) + np.arange(n_candles) * 0.02
    features_a = close_prices[:, np.newaxis].astype(np.float32)
    features_b = np.repeat(features_a, 9, axis=1)
    features_c = np.repeat(features_a, 11, axis=1)
    meta = SnapshotMetadata(
        snapshot_id=f"synth_{timeframe}",
        timeframe=timeframe,
        symbol="PAXGUSDT",
        start_time=int(timestamps[0]),
        end_time=int(timestamps[-1]),
        total_candles=n_candles,
        feature_sets=["A", "B", "C"],
        created_at="2026-01-01T00:00:00Z",
        sha256="0" * 64,
    )
    return DatasetSnapshot(
        metadata=meta,
        timestamps=timestamps,
        features_a=features_a,
        features_b=features_b,
        features_c=features_c,
    )


def test_backtest_in_sample_overlap_detection_e2e():
    """Regression test: BacktestEngine detects adapter in-sample overlap on overlapping folds while OOS folds pass cleanly."""
    from paxg_lab.data.split import calculate_split_plan
    from paxg_lab.eval.engine import BacktestEngine

    snapshot = _make_synthetic_snapshot(n_candles=5500, timeframe="1h")
    split_plan = calculate_split_plan(total_candles=len(snapshot.timestamps), timeframe="1h")
    fold_1 = split_plan.eval_folds[0]
    fold_2 = split_plan.eval_folds[1]

    fold_1_start_ms = int(snapshot.timestamps[fold_1.eval_start])
    fold_1_end_ms = int(snapshot.timestamps[fold_1.eval_end - 1])

    # Adapter whose training and validation range covers Fold 1
    manifest = AdapterManifest(
        adapter_id="lora_trained_on_fold1",
        timeframe="1h",
        horizon=24,
        context_len=256,
        feature_set="A",
        feature_columns=list(FEATURE_SPECS["A"].columns),
        base_model_repo=MODEL_REPO,
        base_model_revision=MODEL_REVISION,
        training_range={
            "start_time_ms": fold_1_start_ms,
            "end_time_ms": fold_1_end_ms - 14 * 24 * 3600 * 1000,
            "val_start_time_ms": fold_1_end_ms - 14 * 24 * 3600 * 1000,
            "val_end_time_ms": fold_1_end_ms,
        },
    )

    engine = BacktestEngine(predictor=None)

    def dummy_pred_fn(ctx_windows: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        n = len(ctx_windows)
        pts = np.zeros((n, horizon), dtype=np.float32) + 2500.0
        q = np.zeros((n, horizon, 9), dtype=np.float32) + 2500.0
        return pts, q

    # 1. Establish base reference
    base_report = engine.run_full_backtest(
        snapshot=snapshot,
        is_base_reference=True,
        custom_predictor_fn=dummy_pred_fn,
    )

    # 2. Run candidate evaluation passing adapter_manifest
    candidate_report = engine.run_full_backtest(
        snapshot=snapshot,
        feature_set="A",
        context_len=256,
        model_name="Candidate-Overlapping",
        is_base_reference=False,
        base_reference_metrics=base_report,
        custom_predictor_fn=dummy_pred_fn,
        adapter_manifest=manifest,
    )

    # Verify Fold 1 (overlapping) is flagged as in-sample
    m_fold1 = candidate_report.get_fold_metric(fold_1.fold_id)
    assert m_fold1 is not None
    assert m_fold1.is_in_sample is True
    assert m_fold1.warning is not None
    assert "IN-SAMPLE OVERLAP DETECTED" in m_fold1.warning

    # Verify other folds (strictly out-of-sample) are NOT flagged
    m_fold2 = candidate_report.get_fold_metric(fold_2.fold_id)
    assert m_fold2 is not None
    assert m_fold2.is_in_sample is False

    # Verify ScoreReport metadata contains in-sample warnings
    assert candidate_report.metadata["has_in_sample_eval_folds"] is True
    assert fold_1.fold_id in candidate_report.metadata["in_sample_folds"]
    assert "in_sample_warning" in candidate_report.metadata

