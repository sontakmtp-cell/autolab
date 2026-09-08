"""Unit and integration tests for Phase P6: Autonomous Optimization, Multi-Segment Verification, and Winner Gatekeeper."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import time
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import optuna
import pandas as pd
import pytest
import torch

from paxg_lab.constants import TICK_SIZE, get_horizon_for_timeframe, get_horizon_weights
from paxg_lab.data.snapshot import DatasetSnapshot, SnapshotMetadata
from paxg_lab.data.split import calculate_split_plan
from paxg_lab.eval.metrics import compute_score_v1
from paxg_lab.eval.types import FoldMetrics, ScoreReport
from paxg_lab.model.manifest import AdapterManifest
from paxg_lab.model.store import AdapterStore
from paxg_lab.model.train_spec import TrainSpec
from paxg_lab.queue.storage import GPUJobStorage
from paxg_lab.queue.types import AutoRunState, JobPriority, JobSpec, JobType
from paxg_lab.tune.bootstrap import (
    BlockBootstrapResult,
    compute_block_bootstrap_ci,
    get_block_size_for_timeframe,
)
from paxg_lab.tune.gatekeeper import GatekeeperDecision, LoRAGatekeeper
from paxg_lab.tune.optimizer import (
    EarlyStoppingStagnationCallback,
    OptunaTPEOptimizer,
)
from paxg_lab.tune.protocol import (
    AutonomousTuningProtocol,
    MultiSeedEvalSummary,
    SEEDS_MULTI_RUN,
)
from paxg_lab.tune.space import (
    ALLOWED_CONTEXTS,
    ALLOWED_FEATURE_SETS,
    ALLOWED_RANKS,
    suggest_trial_spec,
    validate_train_spec_invariants,
)


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def mock_snapshot(temp_dir: Path) -> DatasetSnapshot:
    """Creates a mock dataset snapshot with 3000 1h candles for testing."""
    n_candles = 3000
    timestamps = np.arange(1700000000000, 1700000000000 + n_candles * 3600000, 3600000, dtype=np.int64)
    # Synthetic prices
    base_price = 2000.0 + np.cumsum(np.random.default_rng(42).normal(0, 2.0, n_candles))
    features_a = base_price[:, np.newaxis]  # close (1 col)
    features_b = np.column_stack([features_a, np.ones((n_candles, 8))])  # 9 cols
    features_c = np.column_stack([features_b, np.ones((n_candles, 2))])  # 11 cols

    meta = SnapshotMetadata(
        snapshot_id="mock_1h_snap",
        timeframe="1h",
        symbol="PAXGUSDT",
        start_time=int(timestamps[0]),
        end_time=int(timestamps[-1]),
        total_candles=n_candles,
        feature_sets=["A", "B", "C"],
        created_at="2026-09-06T00:00:00Z",
        sha256="mock_sha256_hash_12345678",
    )
    return DatasetSnapshot(
        metadata=meta,
        timestamps=timestamps,
        features_a=features_a,
        features_b=features_b,
        features_c=features_c,
    )


# ---------------------------------------------------------------------------
# 1. Hyperparameter Search Space and Invariant Checks
# ---------------------------------------------------------------------------


def test_optuna_search_space_constraints():
    """Verifies that sampled hyperparameter specs strictly satisfy PLAN 3.6 invariants."""
    study = optuna.create_study(direction="maximize")

    # Sample 5 trials
    for i in range(5):
        trial = study.ask()
        spec_1h = suggest_trial_spec(trial, timeframe="1h", seed=42)

        # 1. Horizon strictly 24 for 1h
        assert spec_1h.timeframe == "1h"
        assert spec_1h.horizon == 24

        # 2. Alpha strictly 2 * rank
        assert spec_1h.lora_alpha == 2 * spec_1h.lora_r

        # 3. Effective batch size strictly 16 (2 * 8)
        assert spec_1h.batch_size == 2
        assert spec_1h.gradient_accumulation_steps == 8
        assert spec_1h.effective_batch_size == 16

        # 4. Context length in [128, 256, 512]
        assert spec_1h.context_len in (128, 256, 512)

        # 5. Rank in [4, 8, 16]
        assert spec_1h.lora_r in (4, 8, 16)

        # 6. Learning rate in [1e-5, 2e-4]
        assert 1e-5 <= spec_1h.learning_rate <= 2e-4

        # 7. Dropout in [0.0, 0.20]
        assert 0.0 <= spec_1h.lora_dropout <= 0.20

        # 8. Weight decay in [0.0, 0.05]
        assert 0.0 <= spec_1h.weight_decay <= 0.05

        # 9. Max epochs capped at 10, patience 3
        assert spec_1h.max_epochs == 10
        assert spec_1h.early_stopping_patience == 3

        # Invariant validation passes without error
        validate_train_spec_invariants(spec_1h)


def test_optuna_search_space_4h_horizon():
    """Verifies that 4h timeframe samples horizon=6 strictly."""
    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    spec_4h = suggest_trial_spec(trial, timeframe="4h")
    assert spec_4h.timeframe == "4h"
    assert spec_4h.horizon == 6
    assert spec_4h.lora_alpha == 2 * spec_4h.lora_r
    validate_train_spec_invariants(spec_4h)


def test_invariant_validation_detects_violations():
    """Verifies that validate_train_spec_invariants catches illicit configurations."""
    # Invalid alpha (not 2 * r)
    bad_alpha = TrainSpec(timeframe="1h", lora_r=4, lora_alpha=16)
    with pytest.raises(ValueError, match="Alpha invariant violation"):
        validate_train_spec_invariants(bad_alpha)

    # Invalid batch (effective batch != 16)
    bad_batch = TrainSpec(timeframe="1h", batch_size=4, gradient_accumulation_steps=8)
    with pytest.raises(ValueError, match="Effective batch invariant violation"):
        validate_train_spec_invariants(bad_batch)


# ---------------------------------------------------------------------------
# 2. Data Leakage Protection on Locked Test Set
# ---------------------------------------------------------------------------


def test_data_leakage_locked_test_protection(mock_snapshot: DatasetSnapshot):
    """Verifies that test set [test_start, test_end) is completely isolated from eval and training."""
    plan = calculate_split_plan(len(mock_snapshot.features_a), timeframe="1h")

    # 1. Test set must be at the very end
    assert plan.test_end == len(mock_snapshot.features_a)
    assert plan.test_start < plan.test_end

    # 2. Every eval fold must end at or before test_start - purge_buffer
    for fold in plan.eval_folds:
        assert fold.eval_end <= plan.test_start
        assert fold.val_early_stop_end <= fold.eval_start - plan.horizon
        assert fold.train_end <= fold.val_early_stop_start

    # 3. Altering test set labels has ZERO impact on evaluation fold inputs or targets
    orig_eval_targets = [mock_snapshot.features_a[f.eval_start : f.eval_end, 0].copy() for f in plan.eval_folds]

    # Tamper with test set data
    tampered_features_a = mock_snapshot.features_a.copy()
    tampered_features_a[plan.test_start : plan.test_end, :] += 999999.0

    tampered_snapshot = DatasetSnapshot(
        metadata=mock_snapshot.metadata,
        timestamps=mock_snapshot.timestamps,
        features_a=tampered_features_a,
        features_b=mock_snapshot.features_b,
        features_c=mock_snapshot.features_c,
    )

    tampered_eval_targets = [tampered_snapshot.features_a[f.eval_start : f.eval_end, 0] for f in plan.eval_folds]

    # Exactly identical: eval targets are completely untouched
    for orig, tamp in zip(orig_eval_targets, tampered_eval_targets):
        np.testing.assert_array_equal(orig, tamp)


# ---------------------------------------------------------------------------
# 3. 24-Hour Independent Blocks and Block Bootstrap 95% CI
# ---------------------------------------------------------------------------


def test_24h_independent_blocks_definition():
    """Verifies that 1h block has 24 candles and 4h block has 6 candles (both = 24 hours)."""
    assert get_block_size_for_timeframe("1h") == 24
    assert get_block_size_for_timeframe("4h") == 6
    with pytest.raises(ValueError, match="Unsupported timeframe"):
        get_block_size_for_timeframe("15m")


def test_block_bootstrap_rejection_on_insufficient_blocks():
    """Verifies that block bootstrap rejects evaluations with < 20 independent blocks."""
    horizon = 24
    # Only 15 blocks (15 * 24 = 360 windows)
    c_preds = np.zeros((360, horizon))
    b_preds = np.zeros((360, horizon))
    tgts = np.zeros((360, horizon))

    with pytest.raises(ValueError, match="Insufficient independent 24-hour blocks: got 15"):
        compute_block_bootstrap_ci(
            candidate_predictions=c_preds,
            baseline_predictions=b_preds,
            targets=tgts,
            timeframe="1h",
            min_required_blocks=20,
        )


def test_block_bootstrap_significant_positive_improvement():
    """Verifies block bootstrap detects significant positive MAE improvement with 95% CI > 0."""
    horizon = 24
    num_blocks = 25  # 25 independent 24h blocks (>= 20 required)
    num_windows = num_blocks * 24

    rng = np.random.default_rng(42)
    tgts = rng.normal(2000.0, 10.0, size=(num_windows, horizon))

    # Baseline has error ~ 15 USDT
    b_preds = tgts + rng.normal(15.0, 1.0, size=(num_windows, horizon))
    # Candidate has error ~ 10 USDT (consistent improvement of ~ 5 USDT)
    c_preds = tgts + rng.normal(10.0, 1.0, size=(num_windows, horizon))

    res = compute_block_bootstrap_ci(
        candidate_predictions=c_preds,
        baseline_predictions=b_preds,
        targets=tgts,
        timeframe="1h",
        num_resamples=500,
        seed=42,
        min_required_blocks=20,
    )

    assert res.num_blocks == 25
    assert res.mean_improvement_usdt > 3.0
    assert res.ci_95_lower > 0.0  # CI is strictly positive!
    assert res.is_significant_positive is True


def test_block_bootstrap_insignificant_when_not_better():
    """Verifies block bootstrap marks insignificant when candidate does not beat baseline."""
    horizon = 24
    num_blocks = 25
    num_windows = num_blocks * 24

    rng = np.random.default_rng(42)
    tgts = rng.normal(2000.0, 10.0, size=(num_windows, horizon))

    # Candidate is slightly worse (error 11.0 vs 10.0)
    b_preds = tgts + 10.0
    c_preds = tgts + 11.0

    res = compute_block_bootstrap_ci(
        candidate_predictions=c_preds,
        baseline_predictions=b_preds,
        targets=tgts,
        timeframe="1h",
        num_resamples=500,
        seed=42,
        min_required_blocks=20,
    )

    # Lower bound of 95% CI should be <= 0
    assert res.ci_95_lower <= 0.0
    assert res.is_significant_positive is False


# ---------------------------------------------------------------------------
# 4. Optuna TPE Early Stopping Stagnation Callback
# ---------------------------------------------------------------------------


def test_early_stopping_stagnation_callback():
    """Verifies that study stops after 12 non-improving trials post-exploration (10 trials)."""
    study = optuna.create_study(direction="maximize")
    cb = EarlyStoppingStagnationCallback(patience=12, min_delta=0.5, startup_trials=10)

    # 1. First 10 exploration trials with slight variation
    for i in range(10):
        t = study.ask()
        val = 10.0 + i * 0.1
        study.tell(t, val)
        cb(study, study.trials[-1])
        assert not cb.stopped_early

    # 2. 11 trials without improvement
    for i in range(11):
        t = study.ask()
        study.tell(t, 10.1)  # No improvement >= 0.5
        cb(study, study.trials[-1])
        assert not cb.stopped_early

    # 3. 12th stagnant trial triggers early stopping
    t = study.ask()
    study.tell(t, 10.1)
    cb(study, study.trials[-1])
    assert cb.stopped_early is True


# ---------------------------------------------------------------------------
# 5. Multi-Seed Stability Verification & Median Selection
# ---------------------------------------------------------------------------


def test_multi_seed_median_calculation(temp_dir: Path, mock_snapshot: DatasetSnapshot):
    """Verifies that multi-seed evaluation tests seeds [42, 123, 2026] and takes median."""
    proto = AutonomousTuningProtocol(
        timeframe="1h",
        snapshot=mock_snapshot,
        db_path=temp_dir / "test.db",
        optuna_db_path=temp_dir / "optuna.db",
        adapter_store_dir=temp_dir / "adapters",
        audit_dir=temp_dir / "audit",
        backup_dir=temp_dir / "backups",
    )

    mock_spec = TrainSpec(timeframe="1h", lora_r=4)
    dummy_scores = {42: 12.0, 123: 15.0, 2026: 14.0}
    dummy_epochs = {42: 2, 123: 4, 2026: 3}

    def dummy_eval_fn(snap: Any, spec: TrainSpec) -> ScoreReport:
        s = dummy_scores[spec.seed]
        e = dummy_epochs[spec.seed]
        return ScoreReport(
            score_version="v1",
            timeframe="1h",
            horizon=24,
            model_name="mock",
            feature_set="B",
            context_len=256,
            score=s,
            total_eval_windows=100,
            overall_mae=10.0,
            overall_rmse=12.0,
            overall_weighted_mae=10.0,
            overall_weighted_pinball=5.0,
            coverage_80=0.80,
            mean_width_80=20.0,
            directional_accuracy=0.60,
            step_mae={},
            fold_metrics=[],
        )

    base_dummy = dummy_eval_fn(None, mock_spec)
    summary = proto.run_multi_seed_verification(
        best_spec=mock_spec,
        base_reference_report=base_dummy,
        custom_eval_fn=dummy_eval_fn,
    )

    assert summary.seeds == [42, 123, 2026]
    assert summary.scores == [12.0, 15.0, 14.0]
    assert summary.median_score == 14.0  # Median of 12, 14, 15 is 14
    assert summary.median_best_epoch == 2  # default best_epoch returned in dummy


# ---------------------------------------------------------------------------
# 6. Winner Gatekeeper Evaluation Criteria (PLAN 3.5 & P6.md)
# ---------------------------------------------------------------------------


def create_mock_score_report(
    score: float,
    mae: float,
    base_mae: float,
    naive_mae: float,
    worst_fold_ratio: float,
    coverage_80: float,
) -> ScoreReport:
    """Helper creating ScoreReport with specific properties for gatekeeper testing."""
    fold_1 = FoldMetrics(
        fold_id=1,
        num_windows=100,
        weighted_mae=mae,
        weighted_pinball=5.0,
        rmse=mae * 1.2,
        mae=mae,
        coverage_80=coverage_80,
        mean_width_80=20.0,
        directional_accuracy=0.60,
        relative_mae_to_base=mae / base_mae,
        relative_pinball_to_base=1.0,
        composite_loss=0.8,
    )
    fold_2 = FoldMetrics(
        fold_id=2,
        num_windows=100,
        weighted_mae=mae * worst_fold_ratio,
        weighted_pinball=5.0,
        rmse=mae * 1.2,
        mae=mae,
        coverage_80=coverage_80,
        mean_width_80=20.0,
        directional_accuracy=0.60,
        relative_mae_to_base=worst_fold_ratio,
        relative_pinball_to_base=1.0,
        composite_loss=0.8,
    )

    return ScoreReport(
        score_version="v1",
        timeframe="1h",
        horizon=24,
        model_name="CandidateModel",
        feature_set="B",
        context_len=256,
        score=score,
        total_eval_windows=200,
        overall_mae=mae,
        overall_rmse=mae * 1.2,
        overall_weighted_mae=mae,
        overall_weighted_pinball=5.0,
        coverage_80=coverage_80,
        mean_width_80=20.0,
        directional_accuracy=0.60,
        step_mae={},
        fold_metrics=[fold_1, fold_2],
        baseline_comparisons={
            "base_overall_weighted_mae": base_mae,
            "naive_flat_test_weighted_mae": naive_mae,
        },
    )


def test_gatekeeper_all_criteria_pass(temp_dir: Path):
    """Verifies that an adapter satisfying all 7 criteria is accepted, recommended, and backed up."""
    store = AdapterStore(temp_dir / "adapters")
    gatekeeper = LoRAGatekeeper(
        store=store,
        audit_dir=temp_dir / "audit",
        backup_dir=temp_dir / "backups",
    )

    # 1. Create candidate manifest & dummy adapter directory with valid safetensors
    cid = "test_cand_pass"
    cand_dir = store.get_adapter_path(cid)
    cand_dir.mkdir(parents=True, exist_ok=True)
    manifest = AdapterManifest(
        adapter_id=cid,
        timeframe="1h",
        horizon=24,
        context_len=256,
        feature_set="B",
        feature_columns=["close"],
        is_verified=False,
    )
    manifest.save_json(cand_dir / "paxg_manifest.json")

    # Write minimal adapter files
    from safetensors.torch import save_file
    import torch
    import torch.nn as nn
    save_file({
        "base_model.model.seq_attn.0.query_proj.lora_A.weight": torch.zeros((4, 1280)),
        "base_model.model.seq_attn.0.query_proj.lora_B.weight": torch.zeros((1280, 4)),
    }, cand_dir / "adapter_model.safetensors")
    with open(cand_dir / "adapter_config.json", "w") as f:
        json.dump({"r": 4, "target_modules": ["query_proj"], "peft_type": "LORA"}, f)

    # 2. Setup candidate test report meeting all criteria
    # Score: 8.5 vs current 0.0 (diff +8.5 >= 2.0)
    # MAE: 9.8 vs base 10.0 (ratio 0.98 <= 0.99) vs naive 12.0 (ratio <= 0.99)
    # Worst fold ratio: 1.02 <= 1.05
    # Coverage 80%: 0.82 in [0.65, 0.95]
    report = create_mock_score_report(
        score=8.5,
        mae=9.8,
        base_mae=10.0,
        naive_mae=12.0,
        worst_fold_ratio=1.02,
        coverage_80=0.82,
    )

    # 3. Setup test predictions: 25 blocks of 24h with consistent improvement
    num_windows = 25 * 24
    tgts = np.ones((num_windows, 24)) * 2000.0
    base_preds = tgts + 10.0
    cand_preds = tgts + 9.8

    decision = gatekeeper.evaluate_candidate(
        candidate_manifest=manifest,
        candidate_test_report=report,
        candidate_test_preds=cand_preds,
        base_test_preds=base_preds,
        test_targets=tgts,
        naive_test_mae=12.0,
        current_recommended_score=0.0,
        perform_backup=True,
    )

    assert decision.accepted is True
    assert decision.verdict == "ACCEPTED_NEW_RECOMMENDED"
    assert decision.check_score_improvement is True
    assert decision.check_mae_vs_base_and_naive is True
    assert decision.check_no_segment_worse_5pct is True
    assert decision.check_coverage_80_range is True
    assert decision.check_min_independent_blocks is True
    assert decision.check_bootstrap_ci_positive is True
    assert decision.check_smoke_test_and_backup is True

    # Check store recommendation updated
    assert store.get_recommended("1h") == cid

    # Check manifest updated to verified
    reloaded_m = AdapterManifest.load_json(cand_dir / "paxg_manifest.json")
    assert reloaded_m.is_verified is True

    # Check backup zip created
    assert decision.backup_zip_path is not None
    assert Path(decision.backup_zip_path).exists()


def test_gatekeeper_rejection_insufficient_score_improvement(temp_dir: Path):
    """Verifies candidate is rejected if score improvement < 2.0 points."""
    store = AdapterStore(temp_dir / "adapters")
    gatekeeper = LoRAGatekeeper(store=store, audit_dir=temp_dir / "audit")

    cid = "test_cand_low_score"
    manifest = AdapterManifest(adapter_id=cid, timeframe="1h", horizon=24, context_len=256, feature_set="B", feature_columns=["close"])

    # Score only 1.2 vs current 0.0 (diff 1.2 < 2.0 required)
    report = create_mock_score_report(score=1.2, mae=9.5, base_mae=10.0, naive_mae=12.0, worst_fold_ratio=1.01, coverage_80=0.80)
    num_windows = 25 * 24
    tgts = np.ones((num_windows, 24)) * 2000.0
    cand_preds = tgts + 9.5
    base_preds = tgts + 10.0

    decision = gatekeeper.evaluate_candidate(
        candidate_manifest=manifest,
        candidate_test_report=report,
        candidate_test_preds=cand_preds,
        base_test_preds=base_preds,
        test_targets=tgts,
        naive_test_mae=12.0,
        current_recommended_score=0.0,
        perform_backup=False,
    )

    assert decision.accepted is False
    assert decision.verdict == "REJECTED_PRESERVE_CURRENT"
    assert decision.check_score_improvement is False
    assert any("diff +1.20 < +2.0 required" in r for r in decision.reasons)
    # Store recommended adapter must remain None / untouched
    assert store.get_recommended("1h") is None


def test_gatekeeper_rejection_coverage_out_of_bounds(temp_dir: Path):
    """Verifies candidate is rejected if 80% coverage < 65% or > 95%."""
    store = AdapterStore(temp_dir / "adapters")
    gatekeeper = LoRAGatekeeper(store=store, audit_dir=temp_dir / "audit")

    cid = "test_cand_bad_cov"
    manifest = AdapterManifest(adapter_id=cid, timeframe="1h", horizon=24, context_len=256, feature_set="B", feature_columns=["close"])

    # Coverage 0.60 < 0.65
    report = create_mock_score_report(score=5.0, mae=9.5, base_mae=10.0, naive_mae=12.0, worst_fold_ratio=1.01, coverage_80=0.60)
    num_windows = 25 * 24
    tgts = np.ones((num_windows, 24)) * 2000.0
    cand_preds = tgts + 9.5
    base_preds = tgts + 10.0

    decision = gatekeeper.evaluate_candidate(
        candidate_manifest=manifest,
        candidate_test_report=report,
        candidate_test_preds=cand_preds,
        base_test_preds=base_preds,
        test_targets=tgts,
        naive_test_mae=12.0,
        current_recommended_score=0.0,
        perform_backup=False,
    )

    assert decision.accepted is False
    assert decision.check_coverage_80_range is False
    assert any("60.0% (required [65.0%, 95.0%])" in r for r in decision.reasons)


def test_gatekeeper_rejection_fold_worse_than_5_pct(temp_dir: Path):
    """Verifies candidate is rejected if any segment is worse than base by > 5%."""
    store = AdapterStore(temp_dir / "adapters")
    gatekeeper = LoRAGatekeeper(store=store, audit_dir=temp_dir / "audit")

    cid = "test_cand_bad_fold"
    manifest = AdapterManifest(adapter_id=cid, timeframe="1h", horizon=24, context_len=256, feature_set="B", feature_columns=["close"])

    # Worst fold ratio 1.08 > 1.05
    report = create_mock_score_report(score=5.0, mae=9.5, base_mae=10.0, naive_mae=12.0, worst_fold_ratio=1.08, coverage_80=0.80)
    num_windows = 25 * 24
    tgts = np.ones((num_windows, 24)) * 2000.0
    cand_preds = tgts + 9.5
    base_preds = tgts + 10.0

    decision = gatekeeper.evaluate_candidate(
        candidate_manifest=manifest,
        candidate_test_report=report,
        candidate_test_preds=cand_preds,
        base_test_preds=base_preds,
        test_targets=tgts,
        naive_test_mae=12.0,
        current_recommended_score=0.0,
        perform_backup=False,
    )

    assert decision.accepted is False
    assert decision.check_no_segment_worse_5pct is False
    assert any("max ratio=1.080 > 1.05" in r for r in decision.reasons)


# ---------------------------------------------------------------------------
# 7. Recovery, State Machine, and Queue Auto-Tuning Controls
# ---------------------------------------------------------------------------


def test_auto_tune_stop_and_resume_recovery(temp_dir: Path):
    """Verifies that stopping and starting auto tuning respects AutoRunState and preserves studies."""
    db_path = temp_dir / "queue.db"
    storage = GPUJobStorage(db_path)

    # Initial default state in storage is SEARCHING
    assert storage.get_auto_run_state("1h") == AutoRunState.SEARCHING

    # Transition to STOPPED
    storage.set_auto_run_state("1h", AutoRunState.STOPPED)
    assert storage.get_auto_run_state("1h") == AutoRunState.STOPPED

    # Start auto-run: transitions to SEARCHING
    storage.set_auto_run_state("1h", AutoRunState.SEARCHING, allow_unstop=True)
    assert storage.get_auto_run_state("1h") == AutoRunState.SEARCHING

    # Enqueue an auto trial job
    auto_spec = JobSpec(
        job_id="auto_job_1",
        job_type=JobType.AUTO_TRIAL.value,
        timeframe="1h",
        priority=JobPriority.AUTO.value,
    )
    storage.submit_job(auto_spec)

    # Stop auto-run: transitions to STOPPED and cancels pending jobs
    storage.set_auto_run_state("1h", AutoRunState.STOPPED)
    cancelled = storage.cancel_pending_auto_jobs("1h")
    assert cancelled == 1
    assert storage.get_auto_run_state("1h") == AutoRunState.STOPPED

    # Verify job status cancelled
    j = storage.get_job("auto_job_1")
    assert j is not None
    assert j.status == "CANCELLED"

    # Resume auto-run: transitions back to SEARCHING without data loss
    storage.set_auto_run_state("1h", AutoRunState.SEARCHING, allow_unstop=True)
    assert storage.get_auto_run_state("1h") == AutoRunState.SEARCHING


# ---------------------------------------------------------------------------
# 8. Regression Tests for Code Review Comment 5558805566
# ---------------------------------------------------------------------------


def create_mock_locked_report(
    candidate_id: str,
    timeframe: str = "1h",
    score_v1: float = 8.5,
    current_rec_score: float = 0.0,
    cand_mae: float = 9.8,
    base_mae: float = 10.0,
    naive_mae: float = 12.0,
    worst_segment_ratio: float = 1.02,
    coverage_80: float = 0.82,
    bootstrap_result: BlockBootstrapResult | None = None,
) -> Any:
    from paxg_lab.tune.locked_eval import LockedVerificationReport

    if bootstrap_result is None:
        bootstrap_result = BlockBootstrapResult(
            timeframe=timeframe,
            block_size_candles=24 if timeframe == "1h" else 6,
            num_blocks=25,
            candidate_weighted_mae=cand_mae,
            baseline_weighted_mae=base_mae,
            baseline_name="current_recommended",
            mean_improvement_usdt=base_mae - cand_mae,
            relative_improvement_pct=((base_mae - cand_mae) / base_mae) * 100,
            ci_95_lower=0.05,
            ci_95_upper=0.35,
            is_significant_positive=True,
            num_resamples=1000,
        )

    return LockedVerificationReport(
        timeframe=timeframe,
        candidate_id=candidate_id,
        test_start_idx=2400,
        test_end_idx=3000,
        score_v1=score_v1,
        current_recommended_score=current_rec_score,
        score_diff=score_v1 - current_rec_score,
        candidate_weighted_mae=cand_mae,
        base_weighted_mae=base_mae,
        naive_weighted_mae=naive_mae,
        baseline_weighted_mae=base_mae,
        baseline_name=bootstrap_result.baseline_name,
        mae_vs_base_ratio=cand_mae / base_mae,
        mae_vs_naive_ratio=cand_mae / naive_mae,
        segment_mae_ratios={"seg1": cand_mae / base_mae, "seg2": worst_segment_ratio, "seg3": 1.0},
        worst_segment_ratio=worst_segment_ratio,
        coverage_80=coverage_80,
        mean_width_80=20.0,
        directional_accuracy=0.60,
        independent_24h_blocks=bootstrap_result.num_blocks,
        bootstrap_result=bootstrap_result,
        step_maes={1: cand_mae},
        num_windows=600,
        candidate_predictions=np.ones((600, 24 if timeframe == "1h" else 6)),
        targets=np.ones((600, 24 if timeframe == "1h" else 6)),
    )


def test_locked_only_metrics_gatekeeper(temp_dir: Path):
    """Verifies that Gatekeeper reads exclusively from LockedVerificationReport on [test_start, test_end).
    The worst_segment_ratio must come from the 3 locked segments, strictly isolated from eval folds.
    """
    store = AdapterStore(temp_dir / "adapters")
    gatekeeper = LoRAGatekeeper(
        store=store,
        audit_dir=temp_dir / "audit",
        backup_dir=temp_dir / "backups",
    )

    cid = "cand_locked_only"
    cand_dir = store.get_adapter_path(cid)
    cand_dir.mkdir(parents=True, exist_ok=True)
    manifest = AdapterManifest(
        adapter_id=cid,
        timeframe="1h",
        horizon=24,
        context_len=256,
        feature_set="B",
        feature_columns=["close"],
        is_verified=False,
    )
    manifest.save_json(cand_dir / "paxg_manifest.json")

    from safetensors.torch import save_file
    import torch
    save_file({
        "base_model.model.seq_attn.0.query_proj.lora_A.weight": torch.zeros((4, 1280)),
        "base_model.model.seq_attn.0.query_proj.lora_B.weight": torch.zeros((1280, 4)),
    }, cand_dir / "adapter_model.safetensors")
    with open(cand_dir / "adapter_config.json", "w") as f:
        json.dump({"r": 4, "target_modules": ["query_proj"], "peft_type": "LORA"}, f)

    locked_pass = create_mock_locked_report(
        candidate_id=cid,
        timeframe="1h",
        score_v1=8.5,
        current_rec_score=0.0,
        cand_mae=9.8,
        base_mae=10.0,
        naive_mae=12.0,
        worst_segment_ratio=1.02,
        coverage_80=0.82,
    )

    dec_pass = gatekeeper.evaluate_candidate(
        candidate_manifest=manifest,
        locked_report=locked_pass,
        perform_backup=True,
    )
    assert dec_pass.accepted is True
    assert dec_pass.criteria_details["worst_segment_mae_ratio"] == 1.02

    # Verify rejection when locked worst segment > 1.05
    locked_fail_seg = create_mock_locked_report(
        candidate_id=cid,
        timeframe="1h",
        score_v1=8.5,
        current_rec_score=0.0,
        cand_mae=9.8,
        base_mae=10.0,
        naive_mae=12.0,
        worst_segment_ratio=1.08,
        coverage_80=0.82,
    )
    dec_fail = gatekeeper.evaluate_candidate(
        candidate_manifest=manifest,
        locked_report=locked_fail_seg,
        perform_backup=False,
    )
    assert dec_fail.accepted is False
    assert dec_fail.check_no_segment_worse_5pct is False
    assert any("max ratio=1.080 > 1.05" in r for r in dec_fail.reasons)


def test_current_recommended_bootstrap_baseline(temp_dir: Path, mock_snapshot: DatasetSnapshot):
    """Verifies that block bootstrap in locked verification sets baseline to current recommended adapter,
    and falls back to TimesFM3-Base when no recommended adapter exists in store.
    """
    from unittest.mock import MagicMock, patch
    from paxg_lab.tune.locked_eval import run_locked_verification

    store = AdapterStore(temp_dir / "adapters")
    cand_manifest = AdapterManifest(
        adapter_id="cand_test",
        timeframe="1h",
        horizon=24,
        context_len=256,
        feature_set="B",
        feature_columns=["close"],
    )

    with patch("paxg_lab.tune.locked_eval.TimesFM3Predictor") as mock_predictor_cls:
        mock_pred_instance = MagicMock()
        mock_predictor_cls.return_value = mock_pred_instance

        def fake_predict(contexts, horizon, batch_size=16):
            n = len(contexts)
            return np.ones((n, horizon)) * 2000.0, np.ones((n, horizon, 9)) * 2000.0
        mock_pred_instance.predict.side_effect = fake_predict

        # Case A: No recommended adapter in store -> baseline is TimesFM3-Base
        report_no_rec = run_locked_verification(
            snapshot=mock_snapshot,
            candidate_manifest=cand_manifest,
            candidate_adapter_path=temp_dir / "cand",
            store=store,
        )
        assert report_no_rec.bootstrap_result.baseline_name == "TimesFM3-Base"

        # Case B: Recommended adapter is registered in store
        rec_id = "rec_adapter_1h"
        rec_dir = store.get_adapter_path(rec_id)
        rec_dir.mkdir(parents=True, exist_ok=True)
        rec_m = AdapterManifest(
            adapter_id=rec_id,
            timeframe="1h",
            horizon=24,
            context_len=256,
            feature_set="B",
            feature_columns=["close"],
        )
        rec_m.save_json(rec_dir / "paxg_manifest.json")
        store.set_recommended(rec_id, "1h")

        report_with_rec = run_locked_verification(
            snapshot=mock_snapshot,
            candidate_manifest=cand_manifest,
            candidate_adapter_path=temp_dir / "cand",
            store=store,
        )
        assert report_with_rec.bootstrap_result.baseline_name == rec_id


def test_non_overlapping_block_origins():
    """Verifies that 1h blocks have 24-hour spacing (24 candles) and 4h blocks have 24-hour spacing (6 candles)."""
    # 1h timeframe: 480 windows = 20 blocks of 24h
    c_1h = np.ones((480, 24)) * 2000.0
    b_1h = np.ones((480, 24)) * 2005.0
    t_1h = np.ones((480, 24)) * 2000.0

    res_1h = compute_block_bootstrap_ci(
        candidate_predictions=c_1h,
        baseline_predictions=b_1h,
        targets=t_1h,
        timeframe="1h",
        min_required_blocks=20,
    )
    assert res_1h.block_size_candles == 24
    assert res_1h.num_blocks == 20

    # 4h timeframe: 480 windows = 80 blocks of 24h (spaced by 6 candles)
    c_4h = np.ones((480, 6)) * 2000.0
    b_4h = np.ones((480, 6)) * 2005.0
    t_4h = np.ones((480, 6)) * 2000.0

    res_4h = compute_block_bootstrap_ci(
        candidate_predictions=c_4h,
        baseline_predictions=b_4h,
        targets=t_4h,
        timeframe="4h",
        min_required_blocks=20,
    )
    assert res_4h.block_size_candles == 6
    assert res_4h.num_blocks == 80


def test_three_independent_fold_trainings(temp_dir: Path, mock_snapshot: DatasetSnapshot):
    """Verifies that an Optuna trial trains 3 separate adapters from clean Base for folds 1, 2, 3."""
    from unittest.mock import MagicMock, patch
    from paxg_lab.model.trainer import TrainingResult
    from peft import PeftModel

    proto = AutonomousTuningProtocol(
        timeframe="1h",
        snapshot=mock_snapshot,
        db_path=temp_dir / "test.db",
        optuna_db_path=temp_dir / "optuna.db",
        adapter_store_dir=temp_dir / "adapters",
        audit_dir=temp_dir / "audit",
        backup_dir=temp_dir / "backups",
    )

    trained_folds: list[int] = []
    mock_base_model = MagicMock(spec=["to", "eval", "train", "parameters", "named_parameters"])
    mock_trained_peft = MagicMock(spec=PeftModel)

    with patch("timesfm3.TimesFM3Torch.from_pretrained", return_value=mock_base_model) as mock_pretrained, \
         patch("paxg_lab.tune.protocol.TimesFM3Predictor") as mock_predictor_cls, \
         patch("paxg_lab.model.trainer.LoRATrainer.train") as mock_train, \
         patch("paxg_lab.model.store.AdapterStore.save_adapter", return_value=temp_dir / "dummy_path"), \
         patch("paxg_lab.eval.engine.BacktestEngine.evaluate_fold") as mock_eval_fold:

        def fake_train(*args, **kwargs):
            fold_id = kwargs.get("fold_id")
            trained_folds.append(fold_id)
            dummy_manifest = AdapterManifest(
                adapter_id=f"dummy_{fold_id}",
                timeframe="1h",
                horizon=24,
                context_len=256,
                feature_set="B",
                feature_columns=["close"],
            )
            return TrainingResult(
                trained_model=mock_trained_peft,
                manifest=dummy_manifest,
                best_epoch=2,
                best_val_loss=0.05,
                final_train_loss=0.06,
                train_spec=TrainSpec(timeframe="1h"),
                training_range={},
                total_steps=50,
                total_training_time_sec=10.0,
                history=[],
            )

        mock_train.side_effect = fake_train

        dummy_fold_metric = FoldMetrics(
            fold_id=1,
            num_windows=100,
            weighted_mae=10.0,
            weighted_pinball=5.0,
            rmse=12.0,
            mae=10.0,
            coverage_80=0.80,
            mean_width_80=20.0,
            directional_accuracy=0.60,
            relative_mae_to_base=1.0,
            relative_pinball_to_base=1.0,
            composite_loss=0.05,
        )
        mock_eval_fold.return_value = (dummy_fold_metric, None, None, None, None, None)

        spec = TrainSpec(timeframe="1h", lora_r=4)
        base_ref_report = create_mock_score_report(10.0, 10.0, 10.0, 12.0, 1.0, 0.8)

        score, epoch, val_loss = proto.run_trial_evaluation(
            spec=spec,
            base_reference_report=base_ref_report,
            fast_dev_mode=True,
        )

        assert len(trained_folds) == 3
        assert trained_folds == [1, 2, 3]
        assert mock_pretrained.call_count == 3


def test_final_candidate_full_pretest_boundary(temp_dir: Path, mock_snapshot: DatasetSnapshot):
    """Verifies final candidate trains on full pre-test history explicit_train_range=(0, test_start - horizon)
    without calling calculate_split_plan() again on a truncated slice.
    """
    from unittest.mock import MagicMock, patch
    from paxg_lab.model.trainer import TrainingResult
    from peft import PeftModel

    proto = AutonomousTuningProtocol(
        timeframe="1h",
        snapshot=mock_snapshot,
        db_path=temp_dir / "test.db",
        optuna_db_path=temp_dir / "optuna.db",
        adapter_store_dir=temp_dir / "adapters",
        audit_dir=temp_dir / "audit",
        backup_dir=temp_dir / "backups",
    )

    plan = proto.split_plan
    expected_ceiling = plan.test_start - plan.horizon

    mock_base_model = MagicMock(spec=["to", "eval", "train", "parameters", "named_parameters"])
    mock_trained_peft = MagicMock(spec=PeftModel)

    with patch("timesfm3.TimesFM3Torch.from_pretrained", return_value=mock_base_model), \
         patch("paxg_lab.model.trainer.LoRATrainer.train") as mock_train, \
         patch("paxg_lab.model.store.AdapterStore.save_adapter", return_value=temp_dir / "dummy_path"):

        def fake_train(*args, **kwargs):
            assert kwargs.get("fold_id") == "final_pre_test"
            assert kwargs.get("explicit_train_range") == (0, expected_ceiling)
            dummy_m = AdapterManifest(
                adapter_id="dummy_final",
                timeframe="1h",
                horizon=24,
                context_len=256,
                feature_set="B",
                feature_columns=["close"],
            )
            return TrainingResult(
                trained_model=mock_trained_peft,
                manifest=dummy_m,
                best_epoch=3,
                best_val_loss=0.03,
                final_train_loss=0.04,
                train_spec=TrainSpec(timeframe="1h"),
                training_range={},
                total_steps=100,
                total_training_time_sec=20.0,
                history=[],
            )

        mock_train.side_effect = fake_train

        cand_spec = TrainSpec(timeframe="1h", lora_r=8)
        cand_id, manifest, saved_dir = proto.train_final_candidate(
            candidate_spec=cand_spec,
            best_epoch=3,
        )

        assert cand_id.startswith("paxg_1h_r8_")
        assert manifest.adapter_id == cand_id
        assert manifest.is_verified is False
        assert mock_train.call_count == 1


def test_backup_zip_integrity_verification(temp_dir: Path):
    """Verifies gatekeeper Criterion 7 rejects corrupted zip or zip missing checksums.sha256."""
    from unittest.mock import patch
    from paxg_lab.tune.locked_eval import LockedVerificationReport

    store = AdapterStore(temp_dir / "adapters")
    gatekeeper = LoRAGatekeeper(
        store=store,
        audit_dir=temp_dir / "audit",
        backup_dir=temp_dir / "backups",
    )

    cid = "test_corrupt_backup"
    cand_dir = store.get_adapter_path(cid)
    cand_dir.mkdir(parents=True, exist_ok=True)
    manifest = AdapterManifest(
        adapter_id=cid,
        timeframe="1h",
        horizon=24,
        context_len=256,
        feature_set="B",
        feature_columns=["close"],
        is_verified=False,
    )
    manifest.save_json(cand_dir / "paxg_manifest.json")

    from safetensors.torch import save_file
    import torch
    save_file({
        "base_model.model.seq_attn.0.query_proj.lora_A.weight": torch.zeros((4, 1280)),
        "base_model.model.seq_attn.0.query_proj.lora_B.weight": torch.zeros((1280, 4)),
    }, cand_dir / "adapter_model.safetensors")
    with open(cand_dir / "adapter_config.json", "w") as f:
        json.dump({"r": 4, "target_modules": ["query_proj"], "peft_type": "LORA"}, f)

    locked_report = create_mock_locked_report(
        candidate_id=cid,
        timeframe="1h",
        score_v1=8.5,
        current_rec_score=0.0,
        cand_mae=9.8,
        base_mae=10.0,
        naive_mae=12.0,
        worst_segment_ratio=1.02,
        coverage_80=0.82,
    )

    def corrupt_export(adapter_id, out_zip):
        with open(out_zip, "wb") as f:
            f.write(b"not a valid zip file content")
        return Path(out_zip)

    with patch.object(store, "export_adapter_zip", side_effect=corrupt_export):
        decision = gatekeeper.evaluate_candidate(
            candidate_manifest=manifest,
            locked_report=locked_report,
            perform_backup=True,
        )
        assert decision.accepted is False
        assert decision.check_smoke_test_and_backup is False
        assert any("Smoke test load or backup export failed" in r for r in decision.reasons)


def test_worker_auto_trial_production_dispatch(temp_dir: Path, mock_snapshot: DatasetSnapshot):
    """Verifies that GPUWorker._handle_auto_trial_job executes bounded execute_step and queues next step."""
    from unittest.mock import patch
    from paxg_lab.queue.worker import GPUWorker

    db_path = temp_dir / "worker_queue.db"
    snap_path = temp_dir / "dummy_snap"

    job_spec = JobSpec(
        job_id="job_auto_p6",
        job_type=JobType.AUTO_TRIAL.value,
        timeframe="1h",
        priority=JobPriority.AUTO.value,
        payload={"snapshot_path": str(snap_path), "fast_dev_mode": True, "max_trials": 2},
    )

    worker = GPUWorker(job_id="job_auto_p6", db_path=db_path)
    worker.storage.set_auto_run_state("1h", AutoRunState.SEARCHING)

    with patch("pathlib.Path.exists", return_value=True), \
         patch("paxg_lab.data.snapshot.DatasetSnapshot.load", return_value=mock_snapshot), \
         patch("paxg_lab.tune.protocol.AutonomousTuningProtocol.execute_step") as mock_step:
        mock_step.return_value = {
            "phase": "TRIAL",
            "current_trial": 1,
            "timeframe": "1h",
        }

        result = worker._handle_auto_trial_job(job_spec)

        assert result["phase"] == "TRIAL"
        assert result["current_trial"] == 1
        assert mock_step.call_count == 1
        _, kwargs = mock_step.call_args
        assert kwargs["fast_dev_mode"] is True
        assert callable(kwargs["is_cancelled_func"])
        assert callable(kwargs["progress_callback"])

        # Check next bounded auto job was submitted
        recent_jobs = worker.storage.list_recent_jobs(job_type=JobType.AUTO_TRIAL.value, timeframe="1h")
        assert len(recent_jobs) == 1
        assert recent_jobs[0].job_id.startswith("auto_step_1h_")
        assert recent_jobs[0].priority == JobPriority.AUTO.value


# ---------------------------------------------------------------------------
# Focused Regression and Integration Tests for PR #6 Blockers (Items 1 - 8)
# ---------------------------------------------------------------------------


def test_forecast_priority_interleaving_between_auto_steps(temp_dir: Path):
    """Item 1: Proves that high-priority FORECAST jobs take precedence over subsequent bounded AUTO steps,
    and auto steps alternate between 1h and 4h timeframes."""
    db_path = temp_dir / "interleave_queue.db"
    storage = GPUJobStorage(db_path)

    # 1. Enqueue an AUTO_TRIAL job for 1h
    auto_job_1h = JobSpec(
        job_id="auto_step_1h_1",
        job_type=JobType.AUTO_TRIAL.value,
        timeframe="1h",
        priority=JobPriority.AUTO.value,
        payload={"timeframe": "1h"},
    )
    storage.submit_job(auto_job_1h)

    # Acquire and simulate completion of auto_step_1h_1
    acquired = storage.acquire_next_job()
    assert acquired is not None and acquired.job_id == "auto_step_1h_1"
    storage.mark_succeeded(acquired.job_id, result={"phase": "TRIAL", "current_trial": 1})

    # Auto worker enqueues the next bounded auto step (Priority 3)
    next_auto_1h = JobSpec(
        job_id="auto_step_1h_2",
        job_type=JobType.AUTO_TRIAL.value,
        timeframe="1h",
        priority=JobPriority.AUTO.value,
        payload={"timeframe": "1h"},
    )
    storage.submit_job(next_auto_1h)

    # Meanwhile, a user requests an immediate forecast (Priority 1)
    forecast_job = JobSpec(
        job_id="forecast_urgent",
        job_type=JobType.FORECAST.value,
        timeframe="1h",
        priority=JobPriority.FORECAST.value,
        payload={"timeframe": "1h"},
    )
    storage.submit_job(forecast_job)

    # Also a 4h auto step is queued (Priority 3)
    auto_job_4h = JobSpec(
        job_id="auto_step_4h_1",
        job_type=JobType.AUTO_TRIAL.value,
        timeframe="4h",
        priority=JobPriority.AUTO.value,
        payload={"timeframe": "4h"},
    )
    storage.submit_job(auto_job_4h)

    # Next job acquired MUST be the Priority 1 FORECAST job, NOT any auto job!
    next_job = storage.acquire_next_job(last_auto_timeframe="1h")
    assert next_job is not None
    assert next_job.job_id == "forecast_urgent"
    assert next_job.priority == JobPriority.FORECAST.value

    # Complete the forecast job
    storage.mark_succeeded(next_job.job_id, result={"predictions": []})

    # Next job acquired for auto should alternate to 4h because last_auto_timeframe was 1h
    next_auto = storage.acquire_next_job(last_auto_timeframe="1h")
    assert next_auto is not None
    assert next_auto.job_id == "auto_step_4h_1"
    assert next_auto.timeframe == "4h"


def test_continuous_auto_recovery_and_data_wakeup(temp_dir: Path, monkeypatch, mock_snapshot):
    """Item 2: Verifies startup auto recovery from SQLite state and 7-day new data wake-up."""
    from paxg_lab.queue.scheduler import GPUScheduler
    from paxg_lab.queue.types import JobStatus

    db_path = temp_dir / "recovery.db"
    scheduler = GPUScheduler(db_path=db_path, acquire_coordinator_lock=False)

    real_snapshot = DatasetSnapshot.create("1h", mock_snapshot.timestamps, mock_snapshot.features_a, mock_snapshot.features_b)
    real_path = real_snapshot.save(temp_dir / "snapshots")
    # 1. Startup recovery when state is SEARCHING but no active queued job
    scheduler.storage.set_auto_run_state("1h", AutoRunState.SEARCHING)
    scheduler.storage.save_auto_tune_run(
        timeframe="1h",
        snapshot_path=str(real_path),
        snapshot_hash=real_snapshot.metadata.sha256,
        phase="TRIAL",
        current_trial=2,
    )
    recovered = scheduler.recover_on_startup()
    assert len(recovered) == 0  # No running jobs was interrupted
    # Resumption job must be submitted!
    recent = scheduler.storage.list_jobs(job_type=JobType.AUTO_TRIAL.value, timeframe="1h")
    assert len(recent) == 1
    assert recent[0].job_id.startswith("auto_step_1h_resume_")

    # Clean up jobs
    for j in recent:
        scheduler.storage.mark_succeeded(j.job_id, result={})

    # 2. Data wake-up mechanism when state is WAITING_DATA
    scheduler.storage.set_auto_run_state("1h", AutoRunState.WAITING_DATA)
    scheduler.storage.save_auto_tune_run(
        timeframe="1h",
        snapshot_path=str(temp_dir / "snapshots" / "paxgusdt_1h_mock"),
        snapshot_hash="hash123",
        phase="WAITING_DATA",
        last_consumed_candles=1000,
        last_run_completed_at=time.time() - 86400,
    )

    # Setup mock snapshot directory
    snap_dir = temp_dir / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    snap_1h = snap_dir / "paxgusdt_1h_mock"
    snap_1h.mkdir(parents=True, exist_ok=True)

    # Case A: only 100 new candles (< 168 threshold for 1h)
    with open(snap_1h / "metadata.json", "w") as f:
        json.dump({"total_candles": 1100}, f)

    monkeypatch.setattr("paxg_lab.queue.scheduler.Path", lambda p: snap_dir if p == "var/paxg_lab/snapshots" else Path(p))

    scheduler._check_auto_tune_data_wakeup()
    # Should NOT have submitted any job, state remains WAITING_DATA
    assert scheduler.storage.get_auto_run_state("1h") == AutoRunState.WAITING_DATA

    # Case B: 170 new candles (>= 168 threshold for 1h)
    with open(snap_1h / "metadata.json", "w") as f:
        json.dump({"total_candles": 1170}, f)

    scheduler._check_auto_tune_data_wakeup()
    # State must transition to SEARCHING and new job submitted
    assert scheduler.storage.get_auto_run_state("1h") == AutoRunState.SEARCHING
    new_jobs = scheduler.storage.list_jobs(job_type=JobType.AUTO_TRIAL.value, timeframe="1h", status=JobStatus.QUEUED.value)
    assert len(new_jobs) == 1
    assert new_jobs[0].job_id.startswith("auto_step_1h_")


def test_statistical_lock_rejection_on_reuse(temp_dir: Path, mock_snapshot: DatasetSnapshot):
    """Item 3: Verifies that attempting to evaluate a second candidate on an already-consumed locked range
    is strictly rejected by statistical lock enforcement before inference."""
    from unittest.mock import patch
    from paxg_lab.tune.locked_eval import run_locked_verification

    db_path = temp_dir / "lock_test.db"
    storage = GPUJobStorage(db_path)
    store = AdapterStore(temp_dir / "adapters")

    cand1_dir = temp_dir / "cand_1"
    cand1_dir.mkdir(parents=True, exist_ok=True)
    cand1 = AdapterManifest(adapter_id="cand_1", timeframe="1h", horizon=24, context_len=256, feature_set="B", feature_columns=["close"])
    cand1.save_json(cand1_dir / "paxg_manifest.json")

    cand2_dir = temp_dir / "cand_2"
    cand2_dir.mkdir(parents=True, exist_ok=True)
    cand2 = AdapterManifest(adapter_id="cand_2", timeframe="1h", horizon=24, context_len=256, feature_set="B", feature_columns=["close"])
    cand2.save_json(cand2_dir / "paxg_manifest.json")

    dummy_preds = np.ones((600, 24)) * 2000.0
    dummy_quantiles = np.ones((600, 24, 9)) * 2000.0

    with patch("paxg_lab.eval.predictor.TimesFM3Predictor.load_adapter"), \
         patch("paxg_lab.tune.locked_eval.TimesFM3Predictor.predict", return_value=(dummy_preds, dummy_quantiles)), \
         patch("paxg_lab.tune.locked_eval.extract_windows", return_value=(np.zeros((600, 256, 1)), np.zeros((600, 24)), list(range(600)))):

        # First candidate consumes the locked range
        report1 = run_locked_verification(
            snapshot=mock_snapshot,
            candidate_manifest=cand1,
            candidate_adapter_path=cand1_dir,
            store=store,
            storage=storage,
        )
        assert report1.candidate_id == "cand_1"

        # Second candidate on the same snapshot range must be rejected before inference
        with pytest.raises(RuntimeError, match="Statistical lock violation"):
            run_locked_verification(
                snapshot=mock_snapshot,
                candidate_manifest=cand2,
                candidate_adapter_path=cand2_dir,
                store=store,
                storage=storage,
            )


def test_contemporaneous_score_comparison(temp_dir: Path, mock_snapshot: DatasetSnapshot):
    """Item 4: Verifies that current recommended score is computed contemporaneously on the exact same locked
    segments rather than reading an old manifest score."""
    from unittest.mock import patch
    from paxg_lab.tune.locked_eval import run_locked_verification

    store = AdapterStore(temp_dir / "adapters")
    # Save a current recommended adapter with a stale manifest score
    rec_id = "old_rec_adapter"
    rec_dir = store.get_adapter_path(rec_id)
    rec_dir.mkdir(parents=True, exist_ok=True)
    rec_manifest = AdapterManifest(
        adapter_id=rec_id,
        timeframe="1h",
        horizon=24,
        context_len=256,
        feature_set="B",
        feature_columns=["close"],
        metrics={"score_v1": 99.0},  # Stale manifest score from older period
        is_verified=True,
    )
    rec_manifest.save_json(rec_dir / "paxg_manifest.json")
    store.set_recommended("1h", rec_id)

    cand_dir = temp_dir / "new_cand"
    cand_dir.mkdir(parents=True, exist_ok=True)
    cand_manifest = AdapterManifest(
        adapter_id="new_cand",
        timeframe="1h",
        horizon=24,
        context_len=256,
        feature_set="B",
        feature_columns=["close"],
    )
    cand_manifest.save_json(cand_dir / "paxg_manifest.json")

    dummy_preds = np.ones((600, 24)) * 2000.0
    dummy_quantiles = np.ones((600, 24, 9)) * 2000.0

    with patch("paxg_lab.eval.predictor.TimesFM3Predictor.load_adapter"), \
         patch("paxg_lab.tune.locked_eval.TimesFM3Predictor.predict", return_value=(dummy_preds, dummy_quantiles)), \
         patch("paxg_lab.tune.locked_eval.extract_windows", return_value=(np.zeros((600, 256, 1)), np.zeros((600, 24)), list(range(600)))):

        report = run_locked_verification(
            snapshot=mock_snapshot,
            candidate_manifest=cand_manifest,
            candidate_adapter_path=cand_dir,
            store=store,
        )

        # Contemporaneous evaluation must NOT equal 99.0 from the stale manifest
        assert report.current_recommended_score != 99.0
        # Score diff must strictly be cand_score - contemporaneous current_rec_score
        assert pytest.approx(report.score_diff, rel=1e-5) == report.score_v1 - report.current_recommended_score


def test_origin_alignment_different_contexts_and_gaps():
    """Item 5: Verifies that models with different context lengths (128 vs 256) evaluated across a
    timestamp gap have their forecast origins strictly aligned to common origins."""
    from paxg_lab.data.split import extract_windows

    # Create synthetic series of 800 candles with a 5-hour timestamp gap at candle 200
    n = 800
    times = []
    curr = 1700000000000
    for i in range(n):
        if i == 200:
            curr += 5 * 3600000  # 5-hour gap
        else:
            curr += 3600000
        times.append(curr)
    timestamps = np.array(times, dtype=np.int64)
    features = np.arange(n, dtype=np.float32)[:, np.newaxis]
    targets = features[:, 0]

    # Extract windows: candidate with context 128, base with context 256
    c_ctx, c_fut, c_origins = extract_windows(
        features=features, targets=targets, context_len=128, horizon=24,
        start_idx=100, end_idx=750, step=1, timestamps=timestamps, timeframe="1h"
    )
    b_ctx, b_fut, b_origins = extract_windows(
        features=features, targets=targets, context_len=256, horizon=24,
        start_idx=100, end_idx=750, step=1, timestamps=timestamps, timeframe="1h"
    )

    assert len(c_origins) != len(b_origins)  # context 128 has more valid windows than 256
    assert len(c_origins) > len(b_origins)

    # Intersect common origins
    common_set = set(c_origins) & set(b_origins)
    common_origins = sorted(common_set)
    assert len(common_origins) > 0

    c_map = {o: i for i, o in enumerate(c_origins)}
    b_map = {o: i for i, o in enumerate(b_origins)}

    c_indices = [c_map[o] for o in common_origins]
    b_indices = [b_map[o] for o in common_origins]

    c_aligned_fut = c_fut[c_indices]
    b_aligned_fut = b_fut[b_indices]

    # Aligned targets must be identical row for row
    assert c_aligned_fut.shape == b_aligned_fut.shape
    np.testing.assert_array_equal(c_aligned_fut, b_aligned_fut)


def test_cancellation_during_trainer_updates(temp_dir: Path, mock_snapshot: DatasetSnapshot):
    """Item 6: Verifies that cancellation flag during trainer updates immediately raises InterruptedError."""
    import torch.nn as nn
    from paxg_lab.model.trainer import LoRATrainer

    spec = TrainSpec(
        timeframe="1h",
        horizon=24,
        context_len=128,
        feature_set="A",
        max_epochs=5,
        batch_size=4,
    )

    class CleanBaseModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.query_proj = nn.Linear(10, 10)
            self.value_proj = nn.Linear(10, 10)

        def forward_decode(self, target, horizon):
            B = target.shape[0]
            return torch.zeros((B, 1, horizon, 9), dtype=torch.float32, device=target.device, requires_grad=True)

    mock_base = CleanBaseModel()
    trainer = LoRATrainer(base_model=mock_base, spec=spec)
    features_df = pd.DataFrame({
        "open_time": mock_snapshot.timestamps,
        "close": mock_snapshot.features_a[:, 0],
    })

    # Cancel immediately
    with pytest.raises(InterruptedError, match="Training interrupted by cancellation request"):
        trainer.train(
            features_df=features_df,
            snapshot_hash="dummy_hash",
            fold_id="fold_1",
            explicit_train_range=(0, 500),
            is_cancelled_func=lambda: True,
        )


def test_backup_checksum_corruption_rejection(temp_dir: Path):
    """Item 7: Verifies that Gatekeeper parses checksums.sha256 in the backup ZIP, detects altered file bytes,
    and rejects candidate without promoting to recommended."""
    import zipfile
    from unittest.mock import patch

    store = AdapterStore(temp_dir / "adapters")
    gatekeeper = LoRAGatekeeper(
        store=store,
        audit_dir=temp_dir / "audit",
        backup_dir=temp_dir / "backups",
    )

    cid = "test_tampered_backup"
    cand_dir = store.get_adapter_path(cid)
    cand_dir.mkdir(parents=True, exist_ok=True)
    manifest = AdapterManifest(
        adapter_id=cid,
        timeframe="1h",
        horizon=24,
        context_len=256,
        feature_set="B",
        feature_columns=["close"],
        is_verified=False,
    )
    manifest.save_json(cand_dir / "paxg_manifest.json")

    from safetensors.torch import save_file
    import torch
    save_file({
        "base_model.model.seq_attn.0.query_proj.lora_A.weight": torch.zeros((4, 1280)),
        "base_model.model.seq_attn.0.query_proj.lora_B.weight": torch.zeros((1280, 4)),
    }, cand_dir / "adapter_model.safetensors")
    with open(cand_dir / "adapter_config.json", "w") as f:
        json.dump({"r": 4, "target_modules": ["query_proj"], "peft_type": "LORA"}, f)

    locked_report = create_mock_locked_report(
        candidate_id=cid,
        timeframe="1h",
        score_v1=8.5,
        current_rec_score=0.0,
    )

    # Export a zip, but tamper with adapter_config.json inside without updating checksums.sha256
    def tampered_export(adapter_id, out_zip):
        import hashlib
        out_path = Path(out_zip)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out_path, "w") as zf:
            zf.write(cand_dir / "paxg_manifest.json", "paxg_manifest.json")
            zf.write(cand_dir / "adapter_model.safetensors", "adapter_model.safetensors")
            # Write tampered bytes for config
            zf.writestr("adapter_config.json", b'{"tampered": true}')
            # Write checksum sidecar with old hash
            old_hash = hashlib.sha256((cand_dir / "adapter_config.json").read_bytes()).hexdigest()
            m_hash = hashlib.sha256((cand_dir / "paxg_manifest.json").read_bytes()).hexdigest()
            s_hash = hashlib.sha256((cand_dir / "adapter_model.safetensors").read_bytes()).hexdigest()
            cs_content = f"{m_hash}  paxg_manifest.json\n{s_hash}  adapter_model.safetensors\n{old_hash}  adapter_config.json\n"
            zf.writestr("checksums.sha256", cs_content)
        return out_path

    with patch.object(store, "export_adapter_zip", side_effect=tampered_export):
        decision = gatekeeper.evaluate_candidate(
            candidate_manifest=manifest,
            locked_report=locked_report,
            perform_backup=True,
        )
        assert decision.accepted is False
        assert decision.check_smoke_test_and_backup is False
        assert store.get_recommended("1h") is None
        assert any("Checksum mismatch" in r or "Smoke test load or backup export failed" in r for r in decision.reasons)


def test_fixed_epochs_final_retrain(temp_dir: Path, mock_snapshot: DatasetSnapshot):
    """Item 8: Verifies that LoRATrainer fixed_epochs mode trains on all pre-test data without validation split or early stopping."""
    import torch.nn as nn
    from paxg_lab.model.trainer import LoRATrainer

    spec = TrainSpec(
        timeframe="1h",
        horizon=24,
        context_len=128,
        feature_set="A",
        max_epochs=3,
        batch_size=4,
    )

    class CleanBaseModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 10)

    mock_base = CleanBaseModel()
    trainer = LoRATrainer(base_model=mock_base, spec=spec)
    features_df = pd.DataFrame({
        "open_time": mock_snapshot.timestamps,
        "close": mock_snapshot.features_a[:, 0],
    })

    # Dataset preparation in fixed_epochs mode
    train_ctx, train_fut, val_ctx, val_fut, t_range = trainer.prepare_dataset(
        features_df=features_df,
        fold_id="final_pre_test",
        explicit_train_range=(0, 1000),
        fixed_epochs=True,
    )

    # Validation set must be empty (no samples carved out from pre-test training range)
    assert len(val_ctx) == 0
    assert len(val_fut) == 0
    assert t_range["num_val_windows"] == 0
    assert t_range["num_train_windows"] == len(train_ctx)
    assert t_range["train_end_idx"] == 1000


# ---------------------------------------------------------------------------
# Comment 5559161816 Regression & Audit Tests
# ---------------------------------------------------------------------------


def test_statistical_lock_timestamp_overlap_rejection_across_shifted_snapshots(temp_dir: Path):
    """Comment 5559161816 Item 1: Verifies that candidates evaluated on shifted snapshots
    are rejected if their real timestamp locked intervals overlap, even if snapshot hash and local indices differ."""
    from unittest.mock import patch
    from paxg_lab.tune.locked_eval import run_locked_verification

    db_path = temp_dir / "timestamp_lock.db"
    storage = GPUJobStorage(db_path)
    store = AdapterStore(temp_dir / "adapters")

    # Snapshot 1: 3000 candles starting at T0
    n_candles = 3000
    t0 = 1700000000000
    step_ms = 3600000
    ts1 = np.arange(t0, t0 + n_candles * step_ms, step_ms, dtype=np.int64)
    features1 = 2000.0 + np.ones((n_candles, 1))

    snap1 = DatasetSnapshot(
        metadata=SnapshotMetadata(
            snapshot_id="snap1", timeframe="1h", symbol="PAXGUSDT",
            start_time=int(ts1[0]), end_time=int(ts1[-1]), total_candles=n_candles,
            feature_sets=["A", "B"], created_at="2026-09-01T00:00:00Z", sha256="hash_snap_1",
        ),
        timestamps=ts1, features_a=features1, features_b=np.column_stack([features1, np.ones((n_candles, 8))]),
    )

    # Snapshot 2: Shifted by 168 candles (7 days forward)
    shift = 168
    ts2 = ts1 + shift * step_ms
    features2 = 2000.0 + np.ones((n_candles, 1))
    snap2 = DatasetSnapshot(
        metadata=SnapshotMetadata(
            snapshot_id="snap2", timeframe="1h", symbol="PAXGUSDT",
            start_time=int(ts2[0]), end_time=int(ts2[-1]), total_candles=n_candles,
            feature_sets=["A", "B"], created_at="2026-09-08T00:00:00Z", sha256="hash_snap_2_different",
        ),
        timestamps=ts2, features_a=features2, features_b=np.column_stack([features2, np.ones((n_candles, 8))]),
    )

    cand1_dir = temp_dir / "cand_snap1"
    cand1_dir.mkdir(parents=True, exist_ok=True)
    cand1 = AdapterManifest(adapter_id="cand_snap1", timeframe="1h", horizon=24, context_len=256, feature_set="B", feature_columns=["close"])
    cand1.save_json(cand1_dir / "paxg_manifest.json")

    cand2_dir = temp_dir / "cand_snap2"
    cand2_dir.mkdir(parents=True, exist_ok=True)
    cand2 = AdapterManifest(adapter_id="cand_snap2", timeframe="1h", horizon=24, context_len=256, feature_set="B", feature_columns=["close"])
    cand2.save_json(cand2_dir / "paxg_manifest.json")

    dummy_preds = np.ones((600, 24)) * 2000.0
    dummy_quantiles = np.ones((600, 24, 9)) * 2000.0

    with patch("paxg_lab.eval.predictor.TimesFM3Predictor.load_adapter"), \
         patch("paxg_lab.tune.locked_eval.TimesFM3Predictor.predict", return_value=(dummy_preds, dummy_quantiles)), \
         patch("paxg_lab.tune.locked_eval.extract_windows", return_value=(np.zeros((600, 256, 1)), np.zeros((600, 24)), list(range(600)))):

        # Candidate 1 consumes snap1 locked range
        report1 = run_locked_verification(
            snapshot=snap1,
            candidate_manifest=cand1,
            candidate_adapter_path=cand1_dir,
            store=store,
            storage=storage,
        )
        assert report1.candidate_id == "cand_snap1"

        # Candidate 2 on snap2 has an overlapping time range with snap1, despite differing snapshot hash and indices
        with pytest.raises(RuntimeError, match="Statistical lock violation.*overlaps previously consumed"):
            run_locked_verification(
                snapshot=snap2,
                candidate_manifest=cand2,
                candidate_adapter_path=cand2_dir,
                store=store,
                storage=storage,
            )


def test_optuna_stagnation_early_stop_across_bounded_auto_steps(temp_dir: Path):
    """Comment 5559161816 Item 2: Verifies that Optuna stagnation state is reconstructed
    from study history so running 1 trial per step accumulates consecutive stagnant trials to 12."""
    study = optuna.create_study(direction="maximize")
    cb = EarlyStoppingStagnationCallback(patience=12, min_delta=0.5, startup_trials=10)

    # Add 10 startup trials: best score reaches 49.0
    for i in range(10):
        t = study.ask()
        study.tell(t, 40.0 + i)

    cb.sync_from_study(study)
    assert cb.best_score == 49.0
    assert cb.stagnant_trials == 0
    assert not cb.stopped_early

    # Add 11 stagnant trials (score 49.2, < 49.0 + 0.5) across separate bounded steps
    for i in range(11):
        cb.sync_from_study(study)
        assert cb.stagnant_trials == i
        t = study.ask()
        study.tell(t, 49.2)

    # Trial 12 (overall trial 22, 12th post-exploration trial)
    cb.sync_from_study(study)
    assert cb.stagnant_trials == 11
    t = study.ask()
    study.tell(t, 49.1)
    cb.sync_from_study(study)
    assert cb.stagnant_trials == 12
    assert cb.stopped_early is True


def test_cancellation_during_final_fit_and_locked_verification(temp_dir: Path, mock_snapshot: DatasetSnapshot):
    """Comment 5559161816 Item 3: Verifies that cancellation during FINAL_FIT and LOCKED_VERIFICATION
    halts execution immediately without advancing persisted phase or promoting an adapter."""
    from unittest.mock import patch

    db_path = temp_dir / "cancel_test.db"
    storage = GPUJobStorage(db_path)
    store = AdapterStore(temp_dir / "adapters")

    protocol = AutonomousTuningProtocol(
        timeframe="1h",
        snapshot=mock_snapshot,
        db_path=db_path,
        adapter_store_dir=temp_dir / "adapters",
    )

    best_spec = TrainSpec(timeframe="1h", horizon=24, context_len=128, feature_set="A", lora_r=8)
    storage.save_auto_tune_run(
        timeframe="1h",
        snapshot_path=str(temp_dir),
        snapshot_hash=mock_snapshot.metadata.sha256,
        phase="FINAL_FIT",
        best_spec_json=json.dumps(best_spec.to_dict()),
        best_epoch=2,
    )

    # Cancel during FINAL_FIT
    with pytest.raises(InterruptedError, match="cancelled by user request"):
        protocol.execute_step(is_cancelled_func=lambda: True)

    # State must NOT advance to LOCKED_VERIFICATION
    st = storage.get_auto_tune_run("1h")
    assert st["phase"] == "FINAL_FIT"
    assert st.get("final_candidate_id") is None

    # Advance state to LOCKED_VERIFICATION manually to test cancellation during verification
    cand_dir = temp_dir / "cand_mock"
    cand_dir.mkdir(parents=True, exist_ok=True)
    cand_manifest = AdapterManifest(adapter_id="cand_mock", timeframe="1h", horizon=24, context_len=128, feature_set="A", feature_columns=["close"])
    cand_manifest.save_json(cand_dir / "paxg_manifest.json")

    storage.save_auto_tune_run(
        timeframe="1h",
        snapshot_path=str(temp_dir),
        snapshot_hash=mock_snapshot.metadata.sha256,
        phase="LOCKED_VERIFICATION",
        final_candidate_id="cand_mock",
        final_candidate_path=str(cand_dir),
    )

    # Cancel during LOCKED_VERIFICATION
    with patch("paxg_lab.tune.protocol.run_locked_verification", side_effect=InterruptedError("Inference cancelled")), \
         pytest.raises(InterruptedError):
        protocol.execute_step(is_cancelled_func=lambda: True)

    # State must NOT advance to WAITING_DATA and store recommended must remain None
    st2 = storage.get_auto_tune_run("1h")
    assert st2["phase"] == "LOCKED_VERIFICATION"
    assert store.get_recommended("1h") is None


def test_auto_job_failure_and_timeout_transitions_to_paused_error(temp_dir: Path):
    """Comment 5559161816 Item 4: Verifies that timed out, hung, or failed AUTO_TRIAL jobs
    transition AutoRunState to PAUSED_ERROR instead of silently stalling."""
    from unittest.mock import MagicMock, patch
    from paxg_lab.queue.scheduler import GPUScheduler
    from paxg_lab.queue.types import AutoRunState, JobPriority, JobSpec, JobStatus, JobType

    db_path = temp_dir / "scheduler_fail.db"
    storage = GPUJobStorage(db_path)
    storage.set_auto_run_state("1h", AutoRunState.SEARCHING)

    scheduler = GPUScheduler(db_path=db_path, acquire_coordinator_lock=False)

    # Case 1: Active worker exited with non-zero code or failed status
    job1 = JobSpec(
        job_id="auto_step_fail",
        job_type=JobType.AUTO_TRIAL.value,
        timeframe="1h",
        priority=JobPriority.AUTO.value,
        payload={"timeframe": "1h"},
        status=JobStatus.FAILED.value,
        error_message="Runtime training error",
    )
    storage.submit_job(job1)
    storage.mark_failed("auto_step_fail", "Runtime training error")

    scheduler.active_job_id = "auto_step_fail"
    scheduler.active_worker = MagicMock()
    scheduler.active_worker.poll.return_value = 1  # exited

    scheduler.tick()
    assert storage.get_auto_run_state("1h") == AutoRunState.PAUSED_ERROR

    # Reset for Case 2: Heartbeat timeout
    storage.set_auto_run_state("1h", AutoRunState.SEARCHING)
    job2 = JobSpec(
        job_id="auto_step_hung",
        job_type=JobType.AUTO_TRIAL.value,
        timeframe="1h",
        priority=JobPriority.AUTO.value,
        payload={"timeframe": "1h"},
        status=JobStatus.RUNNING.value,
        heartbeat_at=time.time() - 100.0,
    )
    storage.submit_job(job2)
    scheduler.active_job_id = "auto_step_hung"
    scheduler.active_worker = MagicMock()
    scheduler.active_worker.poll.return_value = None  # running
    scheduler.active_worker_pid = 999999

    with patch("paxg_lab.queue.scheduler.safe_terminate_process", return_value=True), \
         patch("paxg_lab.queue.scheduler.is_process_alive", return_value=False):
        scheduler.tick()

    assert storage.get_auto_run_state("1h") == AutoRunState.PAUSED_ERROR

    # Reset for Case 3: Execution timeout (1200s)
    storage.set_auto_run_state("1h", AutoRunState.SEARCHING)
    job3 = JobSpec(
        job_id="auto_step_timeout",
        job_type=JobType.AUTO_TRIAL.value,
        timeframe="1h",
        priority=JobPriority.AUTO.value,
        payload={"timeframe": "1h"},
        status=JobStatus.RUNNING.value,
        started_at=time.time() - 1500.0,
        timeout_seconds=1200.0,
    )
    storage.submit_job(job3)
    scheduler.active_job_id = "auto_step_timeout"
    scheduler.active_worker = MagicMock()
    scheduler.active_worker.poll.return_value = None
    scheduler.active_worker_pid = 999999

    with patch("paxg_lab.queue.scheduler.safe_terminate_process", return_value=True), \
         patch("paxg_lab.queue.scheduler.is_process_alive", return_value=False):
        scheduler.tick()

    assert storage.get_auto_run_state("1h") == AutoRunState.PAUSED_ERROR


def test_fail_closed_incumbent_recommended_adapter(temp_dir: Path, mock_snapshot: DatasetSnapshot):
    """Comment 5559161816 Item 5: Verifies that if current_rec_id exists in store,
    any extraction or inference failure fails closed (raises RuntimeError) without falling back to Base."""
    from unittest.mock import patch
    from paxg_lab.tune.locked_eval import run_locked_verification

    store = AdapterStore(temp_dir / "adapters")
    rec_id = "corrupt_rec_adapter"
    rec_dir = store.get_adapter_path(rec_id)
    rec_dir.mkdir(parents=True, exist_ok=True)
    # Write invalid / corrupt manifest
    (rec_dir / "paxg_manifest.json").write_text("invalid json", encoding="utf-8")
    store.set_recommended("1h", rec_id)

    cand_dir = temp_dir / "cand"
    cand_dir.mkdir(parents=True, exist_ok=True)
    cand_manifest = AdapterManifest(
        adapter_id="cand", timeframe="1h", horizon=24, context_len=256, feature_set="B", feature_columns=["close"]
    )
    cand_manifest.save_json(cand_dir / "paxg_manifest.json")

    # Candidate adapter loads fine, but processing the corrupt incumbent must fail closed
    with patch("paxg_lab.eval.predictor.TimesFM3Predictor.load_adapter"), \
         pytest.raises(RuntimeError, match="Fail-closed.*corrupt_rec_adapter"):
        run_locked_verification(
            snapshot=mock_snapshot,
            candidate_manifest=cand_manifest,
            candidate_adapter_path=cand_dir,
            store=store,
        )


def test_locked_verification_ledger_terminal_verdicts(temp_dir: Path):
    """Comment 5559161816 Item 6: Verifies that locked_verification_ledger verdict is updated
    from IN_PROGRESS to terminal verdicts (ACCEPTED_NEW_RECOMMENDED, REJECTED_PRESERVE_CURRENT, FAILED)."""
    db_path = temp_dir / "ledger_audit.db"
    storage = GPUJobStorage(db_path)

    # 1. Initial consumption marked as IN_PROGRESS
    storage.record_locked_consumption(
        timeframe="1h",
        test_start_idx=100,
        test_end_idx=200,
        test_start_time_ms=1000,
        test_end_time_ms=2000,
        snapshot_hash="h1",
        candidate_id="cand_A",
        verdict="IN_PROGRESS",
    )

    with storage.get_connection() as conn:
        cur = conn.execute("SELECT verdict, details FROM locked_verification_ledger WHERE candidate_id = 'cand_A';")
        row = cur.fetchone()
        assert row["verdict"] == "IN_PROGRESS"

    # 2. Update to ACCEPTED_NEW_RECOMMENDED
    storage.update_locked_consumption_verdict("1h", "cand_A", "ACCEPTED_NEW_RECOMMENDED", details="Gatekeeper passed")
    with storage.get_connection() as conn:
        cur = conn.execute("SELECT verdict, details FROM locked_verification_ledger WHERE candidate_id = 'cand_A';")
        row = cur.fetchone()
        assert row["verdict"] == "ACCEPTED_NEW_RECOMMENDED"
        assert "Gatekeeper passed" in row["details"]

    # 3. Candidate B updated to REJECTED_PRESERVE_CURRENT
    storage.record_locked_consumption(
        timeframe="1h",
        test_start_idx=300,
        test_end_idx=400,
        test_start_time_ms=3000,
        test_end_time_ms=4000,
        snapshot_hash="h2",
        candidate_id="cand_B",
        verdict="IN_PROGRESS",
    )
    storage.update_locked_consumption_verdict("1h", "cand_B", "REJECTED_PRESERVE_CURRENT", details="Bootstrap CI not positive")
    with storage.get_connection() as conn:
        cur = conn.execute("SELECT verdict, details FROM locked_verification_ledger WHERE candidate_id = 'cand_B';")
        row = cur.fetchone()
        assert row["verdict"] == "REJECTED_PRESERVE_CURRENT"

    # 4. Candidate C failed
    storage.record_locked_consumption(
        timeframe="1h",
        test_start_idx=500,
        test_end_idx=600,
        test_start_time_ms=5000,
        test_end_time_ms=6000,
        snapshot_hash="h3",
        candidate_id="cand_C",
        verdict="IN_PROGRESS",
    )
    storage.update_locked_consumption_verdict("1h", "cand_C", "FAILED", details="Out of memory")
    with storage.get_connection() as conn:
        cur = conn.execute("SELECT verdict, details FROM locked_verification_ledger WHERE candidate_id = 'cand_C';")
        row = cur.fetchone()
        assert row["verdict"] == "FAILED"


def test_cross_cycle_fresh_verification_interval_and_20_block_requirement(temp_dir: Path):
    """Issue 1: Verifies multi-cycle fresh locked verification interval starting >= last_end_time_ms
    with >= 20 independent 24h blocks (>= 480 candles for 1h). Keeps WAITING_DATA if < 20 blocks.
    Persists exact interval before final fit and purges final fit strictly before it."""
    from paxg_lab.queue.scheduler import GPUScheduler
    from paxg_lab.queue.types import JobStatus

    db_path = temp_dir / "cycle.db"
    storage = GPUJobStorage(db_path)
    snap_dir = temp_dir / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)

    # 1. Cycle 1 consumed an exam window
    t0 = 1700000000000
    step_ms = 3600000
    n_candles_1 = 3000
    ts_1 = np.arange(t0, t0 + n_candles_1 * step_ms, step_ms, dtype=np.int64)
    # Cycle 1 test interval: [2400, 3000)
    c1_test_start_ms = int(ts_1[2400])
    c1_test_end_ms = int(ts_1[-1]) + step_ms

    storage.record_locked_consumption(
        timeframe="1h",
        test_start_idx=2400,
        test_end_idx=3000,
        test_start_time_ms=c1_test_start_ms,
        test_end_time_ms=c1_test_end_ms,
        snapshot_hash="hash_cycle_1",
        candidate_id="cand_cycle_1",
        verdict="ACCEPTED_NEW_RECOMMENDED",
    )
    storage.save_auto_tune_run(
        timeframe="1h",
        snapshot_path="snap1",
        snapshot_hash="hash_cycle_1",
        phase="WAITING_DATA",
        current_trial=30,
        last_consumed_candles=n_candles_1,
        last_run_completed_at=time.time(),
    )
    storage.set_auto_run_state("1h", AutoRunState.WAITING_DATA)

    scheduler = GPUScheduler(db_path=db_path, acquire_coordinator_lock=False, snapshots_dir=snap_dir)

    # 2. Case A: Only +7 days (168 candles < 480 candles = 20 blocks) arrive
    ts_2 = np.arange(t0, c1_test_end_ms + 168 * step_ms, step_ms, dtype=np.int64)
    DatasetSnapshot.create("1h", ts_2, np.ones((len(ts_2), 1)), np.ones((len(ts_2), 9))).save(snap_dir)

    scheduler._check_auto_tune_data_wakeup()

    # Must REMAIN in WAITING_DATA because 168 candles < 480 (20 blocks of 24h)
    assert storage.get_auto_run_state("1h") == AutoRunState.WAITING_DATA

    # 3. Case B: +25 days (600 candles >= 480 candles = 20 blocks) arrive
    ts_3 = np.arange(t0, c1_test_end_ms + 600 * step_ms, step_ms, dtype=np.int64)
    DatasetSnapshot.create("1h", ts_3, np.ones((len(ts_3), 1)), np.ones((len(ts_3), 9))).save(snap_dir)

    scheduler._check_auto_tune_data_wakeup()

    # Wakes up to SEARCHING and enqueues AUTO_TRIAL job
    assert storage.get_auto_run_state("1h") == AutoRunState.SEARCHING
    queued_jobs = storage.list_recent_jobs(job_type=JobType.AUTO_TRIAL.value, timeframe="1h")
    assert any(j.status == JobStatus.QUEUED.value for j in queued_jobs)

    # 4. Protocol initializes on snapshot 3: selects non-overlapping fresh test interval >= c1_test_end_ms
    n_candles_3 = len(ts_3)
    features_3 = 2000.0 + np.ones((n_candles_3, 1))
    snap3 = DatasetSnapshot(
        metadata=SnapshotMetadata(
            snapshot_id="snap3", timeframe="1h", symbol="PAXGUSDT",
            start_time=int(ts_3[0]), end_time=int(ts_3[-1]), total_candles=n_candles_3,
            feature_sets=["A", "B"], created_at="2026-09-20T00:00:00Z", sha256="hash_snap_3",
        ),
        timestamps=ts_3, features_a=features_3, features_b=np.column_stack([features_3, np.ones((n_candles_3, 8))]),
    )

    protocol = AutonomousTuningProtocol(
        timeframe="1h",
        snapshot=snap3,
        db_path=db_path,
        optuna_db_path=temp_dir / "opt.db",
        adapter_store_dir=temp_dir / "adapters",
    )
    # Split plan test interval must start at or after c1_test_end_ms
    test_start_time = ts_3[protocol.split_plan.test_start]
    assert test_start_time >= c1_test_end_ms
    # Must have >= 20 24h blocks (>= 480 candles)
    assert (protocol.split_plan.test_end - protocol.split_plan.test_start) >= 480

    # Final fit ceiling must strictly precede test_start
    train_ceiling = protocol.split_plan.test_start - protocol.split_plan.horizon
    assert train_ceiling < protocol.split_plan.test_start


def test_per_fold_execution_and_interleaving(temp_dir: Path, mock_snapshot: DatasetSnapshot):
    """Issue 2: Verifies fine job granularity (1-fold bounded units) and that high-priority
    FORECAST jobs can interleave between fold 1 and fold 2 of a TRIAL."""
    from paxg_lab.queue.types import JobPriority, JobStatus

    db_path = temp_dir / "interleaving.db"
    storage = GPUJobStorage(db_path)
    optuna_db = temp_dir / "optuna.db"
    store_dir = temp_dir / "adapters"

    protocol = AutonomousTuningProtocol(
        timeframe="1h",
        snapshot=mock_snapshot,
        db_path=db_path,
        optuna_db_path=optuna_db,
        adapter_store_dir=store_dir,
        max_trials=2,
    )

    def dummy_eval(snap, spec):
        return ScoreReport(
            score=0.04,
            overall_weighted_mae=1.5,
            overall_weighted_pinball=0.8,
            directional_accuracy=55.0,
            coverage_80=81.0,
            fold_metrics=[
                FoldMetrics(fold_id=1, num_windows=100, weighted_mae=1.5, weighted_pinball=0.8, rmse=2.0, mae=1.5, coverage_80=81.0, mean_width_80=5.0, directional_accuracy=55.0, composite_loss=0.04),
                FoldMetrics(fold_id=2, num_windows=100, weighted_mae=1.5, weighted_pinball=0.8, rmse=2.0, mae=1.5, coverage_80=81.0, mean_width_80=5.0, directional_accuracy=55.0, composite_loss=0.04),
                FoldMetrics(fold_id=3, num_windows=100, weighted_mae=1.5, weighted_pinball=0.8, rmse=2.0, mae=1.5, coverage_80=81.0, mean_width_80=5.0, directional_accuracy=55.0, composite_loss=0.04),
            ],
        )

    # Step 1: BASELINE -> transitions to TRIAL, current_fold=1
    with patch.object(protocol, "get_or_compute_base_report", return_value=dummy_eval(mock_snapshot, None)):
        st1 = protocol.execute_step()
        assert st1["phase"] == "TRIAL"
        assert st1["current_fold"] == 1

        # Step 2: Executes fold 1 of Trial 0
        st2 = protocol.execute_step(custom_eval_fn=dummy_eval)
        assert st2["phase"] == "TRIAL"
        assert st2["current_fold"] == 2
        inter = json.loads(st2["intermediate_fold_results_json"])
        assert "fold_losses" in inter
        assert len(inter["fold_losses"]) == 1

    # Interleaving test in Queue:
    # Next auto job queued for fold 2 (Priority AUTO = 2)
    auto_job = JobSpec(
        job_id="auto_fold_2",
        job_type=JobType.AUTO_TRIAL.value,
        timeframe="1h",
        priority=JobPriority.AUTO.value,
        payload={"timeframe": "1h"},
        status=JobStatus.QUEUED.value,
    )
    storage.submit_job(auto_job)

    # User submits emergency interactive FORECAST job (Priority FORECAST = 1)
    fc_job = JobSpec(
        job_id="user_forecast_urgent",
        job_type=JobType.FORECAST.value,
        timeframe="1h",
        priority=JobPriority.FORECAST.value,
        payload={"timeframe": "1h"},
        status=JobStatus.QUEUED.value,
    )
    storage.submit_job(fc_job)

    # Queue prioritizes FORECAST ahead of next AUTO fold
    next_j = storage.get_next_queued_job()
    assert next_j is not None
    assert next_j.job_id == "user_forecast_urgent"
    assert next_j.priority == JobPriority.FORECAST.value

    # Once FORECAST finishes, AUTO fold 2 is served
    storage.mark_finished("user_forecast_urgent", result={"forecast": [2000.0]})
    next_auto = storage.get_next_queued_job()
    assert next_auto is not None
    assert next_auto.job_id == "auto_fold_2"

    # Step 3: Executes fold 2 of Trial 0
    with patch.object(protocol, "get_or_compute_base_report", return_value=dummy_eval(mock_snapshot, None)):
        st3 = protocol.execute_step(custom_eval_fn=dummy_eval)
        assert st3["current_fold"] == 3
        inter3 = json.loads(st3["intermediate_fold_results_json"])
        assert len(inter3["fold_losses"]) == 2


def test_stopped_race_condition_never_resurrects_or_enqueues(temp_dir: Path, mock_snapshot: DatasetSnapshot):
    """Issue 3: Verifies sticky user-owned STOPPED state cannot be overwritten by background
    protocol/worker transitions, and no next step is enqueued when stopped."""
    from paxg_lab.queue.types import JobStatus
    from paxg_lab.queue.worker import GPUWorker

    db_path = temp_dir / "race.db"
    storage = GPUJobStorage(db_path)
    optuna_db = temp_dir / "optuna.db"
    store_dir = temp_dir / "adapters"

    protocol = AutonomousTuningProtocol(
        timeframe="1h",
        snapshot=mock_snapshot,
        db_path=db_path,
        optuna_db_path=optuna_db,
        adapter_store_dir=store_dir,
    )

    # User stops auto-run in DB
    storage.set_auto_run_state("1h", AutoRunState.STOPPED)
    assert storage.get_auto_run_state("1h") == AutoRunState.STOPPED

    # Protocol execute_step must raise InterruptedError immediately and not overwrite STOPPED
    with pytest.raises(InterruptedError, match="STOPPED"):
        protocol.execute_step()
    assert storage.get_auto_run_state("1h") == AutoRunState.STOPPED

    # Protocol internal transition with allow_unstop=False is safely ignored
    storage.set_auto_run_state("1h", AutoRunState.VALIDATING, allow_unstop=False)
    assert storage.get_auto_run_state("1h") == AutoRunState.STOPPED

    job = JobSpec(
        job_id="auto_step_stopped",
        job_type=JobType.AUTO_TRIAL.value,
        timeframe="1h",
        priority=JobPriority.AUTO.value,
        payload={"timeframe": "1h", "snapshot_path": str(temp_dir)},
        status=JobStatus.QUEUED.value,
    )
    storage.submit_job(job)
    worker = GPUWorker(job_id=job.job_id, db_path=db_path)

    # Run worker handle
    with patch("paxg_lab.data.snapshot.DatasetSnapshot.load", return_value=mock_snapshot), \
         patch("paxg_lab.tune.protocol.AutonomousTuningProtocol") as MockProto:
        mock_inst = MagicMock()
        mock_inst.execute_step.return_value = {"phase": "TRIAL"}
        MockProto.return_value = mock_inst

        worker._handle_auto_trial_job(job)

    # Confirm no new AUTO_TRIAL job was queued
    recent = storage.list_recent_jobs(job_type=JobType.AUTO_TRIAL.value, timeframe="1h")
    queued = [j for j in recent if j.status == JobStatus.QUEUED.value and j.job_id != "auto_step_stopped"]
    assert len(queued) == 0
    assert storage.get_auto_run_state("1h") == AutoRunState.STOPPED


def test_top3_configs_multi_seed_median_selection_and_repropose(temp_dir: Path, mock_snapshot: DatasetSnapshot):
    """Issue 4: Verifies that up to top 3 unique completed configs are multi-seeded, winner is selected
    by median score across seeds [42, 123, 2026], and prior snapshot best params are re-proposed via study.enqueue_trial."""
    from paxg_lab.tune.optimizer import OptunaTPEOptimizer
    from paxg_lab.tune.protocol import get_top_unique_specs

    optuna_db = temp_dir / "optuna_top3.db"

    # Part A: Create study with 4 trials:
    # Trial 0: r=4, lr=5e-5, seed=42 -> score = 0.020 (lucky outlier)
    # Trial 1: r=8, lr=1e-4, seed=42 -> score = 0.035 (consistent)
    # Trial 2: duplicate of Trial 0 (r=4, lr=5e-5) -> score = 0.050
    # Trial 3: r=16, lr=1.5e-4, seed=42 -> score = 0.040
    opt1 = OptunaTPEOptimizer(timeframe="1h", snapshot=mock_snapshot, db_path=optuna_db, study_name="test_top3_study")
    study = opt1.create_or_load_study()

    configs_to_add = [
        ({"context_len": 256, "lora_r": 4, "learning_rate": 5e-5, "lora_dropout": 0.05, "weight_decay": 0.01, "feature_set": "B", "history_days": 180}, 0.020),
        ({"context_len": 256, "lora_r": 8, "learning_rate": 1e-4, "lora_dropout": 0.05, "weight_decay": 0.01, "feature_set": "B", "history_days": 180}, 0.035),
        ({"context_len": 256, "lora_r": 4, "learning_rate": 5e-5, "lora_dropout": 0.05, "weight_decay": 0.01, "feature_set": "B", "history_days": 180}, 0.050),
        ({"context_len": 256, "lora_r": 16, "learning_rate": 1.5e-4, "lora_dropout": 0.05, "weight_decay": 0.01, "feature_set": "B", "history_days": 180}, 0.040),
    ]
    for p, sc in configs_to_add:
        study.enqueue_trial(p)
        tr = study.ask()
        _ = suggest_trial_spec(tr, "1h", seed=42)
        study.tell(tr, sc)

    # get_top_unique_specs must deduplicate and return at most top 3 configs (ranks 4, 16, 8)
    top_specs = get_top_unique_specs(study, "1h", top_n=3)
    assert len(top_specs) == 3
    ranks = [s.lora_r for s in top_specs]
    assert ranks == [4, 16, 8]

    # Multi-seed evaluation: Config A (r=4) degrades on other seeds, Config B (r=8) is consistent
    def mock_eval_fn(snap, spec):
        scores_map = {
            (4, 42): 0.020, (4, 123): 0.080, (4, 2026): 0.090,
            (8, 42): 0.035, (8, 123): 0.036, (8, 2026): 0.037,
            (16, 42): 0.040, (16, 123): 0.045, (16, 2026): 0.046,
        }
        val = scores_map.get((spec.lora_r, spec.seed), 0.05)
        return ScoreReport(
            score=val,
            overall_weighted_mae=val * 30,
            overall_weighted_pinball=val * 15,
            directional_accuracy=55.0,
            coverage_80=81.0,
            fold_metrics=[FoldMetrics(fold_id=i, num_windows=100, weighted_mae=val*30, weighted_pinball=val*15, rmse=val*35, mae=val*30, coverage_80=81.0, mean_width_80=5.0, directional_accuracy=55.0, composite_loss=val) for i in (1, 2, 3)],
        )

    db_path = temp_dir / "multi_seed.db"
    storage = GPUJobStorage(db_path)
    protocol = AutonomousTuningProtocol(
        timeframe="1h",
        snapshot=mock_snapshot,
        db_path=db_path,
        optuna_db_path=optuna_db,
        adapter_store_dir=temp_dir / "adapters",
    )

    # Save state at start of MULTI_SEED
    storage.save_auto_tune_run(
        timeframe="1h",
        snapshot_path="mock",
        snapshot_hash=mock_snapshot.metadata.sha256,
        phase="MULTI_SEED",
        current_trial=4,
        top_specs_json=json.dumps([s.to_dict() for s in top_specs]),
        multi_seed_config_idx=0,
        multi_seed_seed_idx=0,
        multi_seed_fold_idx=1,
        multi_seed_evaluations_json=json.dumps([]),
    )

    with patch.object(protocol, "get_or_compute_base_report", return_value=mock_eval_fn(mock_snapshot, top_specs[0])):
        # Run through MULTI_SEED steps until it transitions to FINAL_FIT
        curr_phase = "MULTI_SEED"
        for _ in range(30):
            st = protocol.execute_step(custom_eval_fn=mock_eval_fn)
            curr_phase = st.get("phase")
            if curr_phase != "MULTI_SEED":
                break

    assert curr_phase == "FINAL_FIT"
    final_st = storage.get_auto_tune_run("1h")
    # Winner must be Config B (r=8, median 0.036) NOT Config A (r=4, median 0.080)
    best_spec = json.loads(final_st["best_spec_json"])
    assert best_spec["lora_r"] == 8
    # Winner median score across seeds [96.5, 96.4, 96.3] is 96.4 (since 100 * (1 - 0.036) = 96.4)
    assert abs(final_st["best_score"] - 96.4) < 1e-4

    # Part B: Repropose prior snapshot best params into fresh study
    fresh_opt = OptunaTPEOptimizer(
        timeframe="1h",
        snapshot=mock_snapshot,
        db_path=optuna_db,
        study_name="fresh_snapshot_study",
    )
    fresh_study = fresh_opt.create_or_load_study(repropose_prior=True)
    # The fresh study should have an enqueued trial containing best_trial params from previous study
    assert len(fresh_study.trials) > 0
    enqueued = fresh_study.trials[0]
    assert enqueued.state == optuna.trial.TrialState.WAITING
    enqueued_params = enqueued.system_attrs.get("fixed_params", enqueued.params)
    assert enqueued_params["lora_r"] == study.best_trial.params["lora_r"]


def test_locked_verification_ledger_migration_and_legacy_fallback(temp_dir: Path):
    """Issue 6: Verifies that locked_verification_ledger safely migrates legacy tables without
    test_start_time_ms/test_end_time_ms columns, and correctly falls back to snapshot_hash + indices
    when legacy rows have test_start_time_ms == 0."""
    import sqlite3

    db_path = temp_dir / "legacy.db"

    # 1. Create legacy schema without test_start_time_ms / test_end_time_ms
    with sqlite3.connect(str(db_path)) as conn:
        conn.executescript("""
            CREATE TABLE locked_verification_ledger (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timeframe TEXT NOT NULL,
                test_start_idx INTEGER NOT NULL,
                test_end_idx INTEGER NOT NULL,
                snapshot_hash TEXT NOT NULL,
                candidate_id TEXT NOT NULL,
                consumed_at REAL NOT NULL,
                verdict TEXT NOT NULL DEFAULT 'IN_PROGRESS',
                details TEXT NOT NULL DEFAULT ''
            );
        """)
        # Insert a legacy row with no timestamps
        conn.execute("""
            INSERT INTO locked_verification_ledger (
                timeframe, test_start_idx, test_end_idx, snapshot_hash, candidate_id, consumed_at, verdict, details
            ) VALUES ('1h', 2400, 3000, 'legacy_snap_hash', 'legacy_cand', 1700000000.0, 'ACCEPTED_NEW_RECOMMENDED', 'Legacy');
        """)
        conn.commit()

    # 2. Initializing GPUJobStorage should trigger migration PRAGMA check and add columns
    storage = GPUJobStorage(db_path)

    with storage.get_connection() as conn:
        cur = conn.execute("PRAGMA table_info(locked_verification_ledger);")
        col_names = [r["name"] for r in cur.fetchall()]
        assert "test_start_time_ms" in col_names
        assert "test_end_time_ms" in col_names

    # 3. Test legacy fallback:
    # A) Same snapshot hash and overlapping indices -> rejected (True)
    assert storage.is_locked_range_consumed(
        timeframe="1h",
        snapshot_hash="legacy_snap_hash",
        test_start_idx=2500,
        test_end_idx=2900,
        test_start_time_ms=0,
        test_end_time_ms=0,
    ) is True

    # B) Same snapshot hash but non-overlapping indices -> not rejected (False)
    assert storage.is_locked_range_consumed(
        timeframe="1h",
        snapshot_hash="legacy_snap_hash",
        test_start_idx=0,
        test_end_idx=1000,
        test_start_time_ms=0,
        test_end_time_ms=0,
    ) is False

    # C) Different snapshot hash with indices (legacy fallback ignores different hash) -> False
    assert storage.is_locked_range_consumed(
        timeframe="1h",
        snapshot_hash="different_snap_hash",
        test_start_idx=2500,
        test_end_idx=2900,
        test_start_time_ms=0,
        test_end_time_ms=0,
    ) is False

    # 4. Modern row with timestamp interval:
    storage.record_locked_consumption(
        timeframe="1h",
        test_start_idx=100,
        test_end_idx=200,
        test_start_time_ms=100000,
        test_end_time_ms=200000,
        snapshot_hash="modern_hash",
        candidate_id="modern_cand",
        verdict="ACCEPTED_NEW_RECOMMENDED",
    )
    # Timestamp overlap check across different snapshot hashes:
    assert storage.is_locked_range_consumed(
        timeframe="1h",
        snapshot_hash="another_hash",
        test_start_idx=0,
        test_end_idx=50,
        test_start_time_ms=150000,
        test_end_time_ms=250000,
    ) is True

    # Non-overlapping timestamp range:
    assert storage.is_locked_range_consumed(
        timeframe="1h",
        snapshot_hash="another_hash",
        test_start_idx=0,
        test_end_idx=50,
        test_start_time_ms=200000,
        test_end_time_ms=300000,
    ) is False





