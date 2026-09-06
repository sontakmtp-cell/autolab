"""Unit and integration tests for Phase P6: Autonomous Optimization, Multi-Segment Verification, and Winner Gatekeeper."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np
import optuna
import pytest

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
    features_a = np.column_stack([
        base_price,  # close
        base_price + 1.0,  # open
        base_price + 2.0,  # high
        base_price - 2.0,  # low
        np.ones(n_candles) * 100.0,  # volume
    ])
    features_b = np.column_stack([features_a, np.ones((n_candles, 5))])
    features_c = np.column_stack([features_b, np.ones((n_candles, 5))])

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
    storage.set_auto_run_state("1h", AutoRunState.SEARCHING)
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
    storage.set_auto_run_state("1h", AutoRunState.SEARCHING)
    assert storage.get_auto_run_state("1h") == AutoRunState.SEARCHING
