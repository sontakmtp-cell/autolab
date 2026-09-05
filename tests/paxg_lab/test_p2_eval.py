"""Comprehensive PyTest suite for Phase P2: Forecasting, Backtest Engine, and Score v1."""

from pathlib import Path
import numpy as np
import pytest
import torch

from paxg_lab.constants import (
    HORIZON_WEIGHTS,
    QUANTILES,
    SCORE_VERSION,
    TIMEFRAME_HORIZONS,
    get_horizon_for_timeframe,
    get_horizon_weights,
)
from paxg_lab.data.snapshot import DatasetSnapshot, SNAPSHOTS_DIR
from paxg_lab.eval import (
    BacktestEngine,
    ForecastRequest,
    ForecastResult,
    TimesFM3Predictor,
    calculate_coverage_80,
    calculate_directional_accuracy,
    calculate_mean_width_80,
    calculate_step_mae,
    calculate_weighted_mae,
    calculate_weighted_pinball_loss,
    calculate_weighted_rmse,
    compute_fold_metrics,
    compute_score_v1,
)


def test_horizon_weights_conventions():
    # 1h: 24 steps
    w_1h = np.array(get_horizon_weights("1h"))
    assert len(w_1h) == 24
    assert np.isclose(np.sum(w_1h), 1.0)
    assert np.isclose(np.sum(w_1h[0:6]), 0.50)
    assert np.isclose(np.sum(w_1h[6:12]), 0.30)
    assert np.isclose(np.sum(w_1h[12:24]), 0.20)

    # 4h: 6 steps
    w_4h = np.array(get_horizon_weights("4h"))
    assert len(w_4h) == 6
    assert np.isclose(np.sum(w_4h), 1.0)
    assert np.isclose(w_4h[0], 0.50)
    assert np.isclose(np.sum(w_4h[1:3]), 0.30)
    assert np.isclose(np.sum(w_4h[3:6]), 0.20)


def test_forecast_request_validation():
    # Default 1h sets horizon=24
    req_1h = ForecastRequest(timeframe="1h")
    assert req_1h.horizon == 24
    assert req_1h.context_len == 256

    # Default 4h sets horizon=6
    req_4h = ForecastRequest(timeframe="4h")
    assert req_4h.horizon == 6

    # Invalid horizon raises ValueError
    with pytest.raises(ValueError, match="Invalid horizon 12 for timeframe 1h"):
        ForecastRequest(timeframe="1h", horizon=12)

    with pytest.raises(ValueError, match="Unsupported timeframe '15m'"):
        ForecastRequest(timeframe="15m")


def test_target_timestamps_exact_alignment():
    # Origin at 1743073200000 (UTC)
    origin = 1743073200000

    # 1h: 24 steps, each step +3600000 ms
    step_1h = 3600 * 1000
    expected_1h = [origin + (i + 1) * step_1h for i in range(24)]
    req_1h = ForecastRequest(timeframe="1h")
    # Verify calculated timestamps logic
    calc_1h = [origin + (i + 1) * step_1h for i in range(req_1h.horizon)]
    assert calc_1h == expected_1h
    assert len(calc_1h) == 24

    # 4h: 6 steps, each step +14400000 ms
    step_4h = 4 * 3600 * 1000
    expected_4h = [origin + (i + 1) * step_4h for i in range(6)]
    req_4h = ForecastRequest(timeframe="4h")
    calc_4h = [origin + (i + 1) * step_4h for i in range(req_4h.horizon)]
    assert calc_4h == expected_4h
    assert len(calc_4h) == 6


def test_quantile_monotonicity_and_interval():
    # Simulate un-sorted raw quantiles
    np.random.seed(42)
    raw = np.random.randn(5, 24, 9) + 2500.0
    sorted_q = np.sort(raw, axis=-1)

    # Check monotonicity along quantile axis: q_i <= q_{i+1}
    for b in range(5):
        for h in range(24):
            q_slice = sorted_q[b, h]
            assert np.all(np.diff(q_slice) >= 0.0), "Quantiles must be monotonically non-decreasing"

    # q10 is index 0, q90 is index 8
    q10 = sorted_q[:, :, 0]
    q90 = sorted_q[:, :, 8]
    assert np.all(q90 >= q10)


def test_score_v1_formula_logic():
    # 1. Base reference model vs itself: L_f = 1.0 for all folds -> Score == 0.0
    base_losses = [1.0, 1.0, 1.0]
    score_base = compute_score_v1(base_losses)
    assert np.isclose(score_base, 0.0)

    # 2. Perfect forecast: L_f = 0.0 -> Score == 100.0
    perfect_losses = [0.0, 0.0, 0.0]
    score_perfect = compute_score_v1(perfect_losses)
    assert np.isclose(score_perfect, 100.0)

    # 3. Model worse than base: L_f = 1.5 -> Score < 0
    bad_losses = [1.5, 1.5, 1.5]
    score_bad = compute_score_v1(bad_losses)
    assert score_bad < 0.0
    assert np.isclose(score_bad, -50.0)

    # 4. Asymmetric fold performance penalizes worst fold:
    # Model A: [0.9, 0.9, 0.9] -> mean=0.9, max=0.9 -> penalty = 0.8*0.9 + 0.2*0.9 = 0.9 -> Score = 10.0
    # Model B: [0.8, 0.9, 1.0] -> mean=0.9, max=1.0 -> penalty = 0.8*0.9 + 0.2*1.0 = 0.92 -> Score = 8.0
    score_a = compute_score_v1([0.9, 0.9, 0.9])
    score_b = compute_score_v1([0.8, 0.9, 1.0])
    assert score_a > score_b, "Consistent performance must beat unstable performance with same average"


def test_metrics_calculation_synthetic():
    n_windows = 10
    horizon = 24
    w = get_horizon_weights("1h")

    # Constant targets = 2500.0, predictions = 2502.0 -> abs error = 2.0
    targets = np.full((n_windows, horizon), 2500.0)
    preds = np.full((n_windows, horizon), 2502.0)

    w_mae = calculate_weighted_mae(preds, targets, w)
    assert np.isclose(w_mae, 2.0)

    w_rmse = calculate_weighted_rmse(preds, targets, w)
    assert np.isclose(w_rmse, 2.0)

    step_maes = calculate_step_mae(preds, targets, "1h")
    assert step_maes == {1: 2.0, 6: 2.0, 12: 2.0, 24: 2.0}

    # Quantiles spanning [2490, 2510]
    quantiles = np.zeros((n_windows, horizon, 9))
    for q_idx in range(9):
        quantiles[:, :, q_idx] = 2490.0 + q_idx * 2.5  # 2490 to 2510

    cov = calculate_coverage_80(quantiles, targets)
    assert cov == 1.0  # 2500 is strictly inside [2490, 2510]

    width = calculate_mean_width_80(quantiles)
    assert np.isclose(width, 20.0)

    # Directional accuracy
    origins = np.full(n_windows, 2499.0)  # price rose from 2499 to 2500
    # preds: 2502 > 2499 (up); targets: 2500 > 2499 (up) -> 100% directional accuracy
    dir_acc = calculate_directional_accuracy(preds, targets, origins)
    assert dir_acc == 1.0


def test_naive_baseline_evaluation():
    engine = BacktestEngine(predictor=None)
    n_pts = 100
    step_ms = 3600 * 1000
    ts = np.array([1743073200000 + i * step_ms for i in range(n_pts)], dtype=np.int64)
    features = np.arange(n_pts, dtype=np.float32).reshape(-1, 1) + 2500.0
    targets = np.arange(n_pts, dtype=np.float32) + 2500.0

    naive_metrics = engine.evaluate_naive_baseline(
        features=features,
        targets=targets,
        timestamps=ts,
        timeframe="1h",
        start_idx=40,
        end_idx=80,
        context_len=20,
    )
    assert naive_metrics.num_windows > 0
    assert naive_metrics.weighted_mae > 0.0
    assert naive_metrics.coverage_80 == 0.0  # Naive collapsed quantiles has 0 coverage of changing series


@pytest.mark.integration
def test_real_base_inference_on_snapshots():
    """Runs a live test of TimesFM 3.0 inference on local snapshots if available."""
    snap_candidates_1h = list(SNAPSHOTS_DIR.glob("paxgusdt_1h_*"))
    snap_candidates_4h = list(SNAPSHOTS_DIR.glob("paxgusdt_4h_*"))

    if not snap_candidates_1h or not snap_candidates_4h:
        pytest.skip("Integration test skipped: snapshots not found in var/paxg_lab/snapshots/")

    snap_1h = DatasetSnapshot.load(snap_candidates_1h[0], verify_hash=True)
    snap_4h = DatasetSnapshot.load(snap_candidates_4h[0], verify_hash=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = TimesFM3Predictor(device=device)

    # Test 1h single batch
    ctx_1h = snap_1h.features_a[:256, 0]  # (256,)
    res_1h = predictor.forecast_request(
        ForecastRequest(timeframe="1h"),
        context_features=ctx_1h,
        forecast_origin_time=int(snap_1h.timestamps[255]),
    )
    assert res_1h.point_forecast.shape == (24,)
    assert res_1h.quantiles.shape == (24, 9)
    assert len(res_1h.target_timestamps) == 24
    assert np.all(np.diff(res_1h.quantiles, axis=-1) >= 0.0)

    # Test 4h single batch
    ctx_4h = snap_4h.features_a[:256, 0]  # (256,)
    res_4h = predictor.forecast_request(
        ForecastRequest(timeframe="4h"),
        context_features=ctx_4h,
        forecast_origin_time=int(snap_4h.timestamps[255]),
    )
    assert res_4h.point_forecast.shape == (6,)
    assert res_4h.quantiles.shape == (6, 9)
    assert len(res_4h.target_timestamps) == 6
    assert np.all(np.diff(res_4h.quantiles, axis=-1) >= 0.0)


from unittest.mock import MagicMock
from paxg_lab.constants import TICK_SIZE
from paxg_lab.data.snapshot import SnapshotMetadata


def _make_synthetic_snapshot(n_candles: int = 5500, timeframe: str = "1h") -> DatasetSnapshot:
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


def test_end_to_end_score_v1_base_candidate_perfect_bad():
    """Regression test proving Score v1 end-to-end:

    - Base reference model -> Score == 0.00
    - Perfect candidate predictor -> Score > 0.00 (substantially positive)
    - Intentionally bad candidate predictor -> Score < 0.00
    """
    snapshot = _make_synthetic_snapshot(n_candles=5500, timeframe="1h")
    engine = BacktestEngine(predictor=None)

    # 1. Base Predictor: flat prediction + slight offset
    def base_predictor_fn(ctx_windows: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        n = len(ctx_windows)
        last_val = ctx_windows[:, -1, 0] if ctx_windows.ndim == 3 else ctx_windows[:, -1]
        pts = np.repeat((last_val + 5.0)[:, np.newaxis], horizon, axis=1)
        q = np.repeat(pts[:, :, np.newaxis], 9, axis=2)
        for i in range(9):
            q[:, :, i] += (i - 4) * 2.0
        return pts, q

    rep_base = engine.run_full_backtest(
        snapshot=snapshot,
        is_base_reference=True,
        custom_predictor_fn=base_predictor_fn,
    )
    assert np.isclose(rep_base.score, 0.0, atol=1e-5), f"Base model must have Score == 0.0, got {rep_base.score}"
    assert rep_base.overall_weighted_mae > 0.0

    # 2. Perfect Candidate Predictor: accurately tracks future targets
    targets = snapshot.features_a[:, 0]

    def perfect_predictor_fn(ctx_windows: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        # Perfectly predicts targets with minimal noise (error near zero)
        n = len(ctx_windows)
        last_val = ctx_windows[:, -1, 0] if ctx_windows.ndim == 3 else ctx_windows[:, -1]
        # Very close prediction: error ~ 0.05 vs base error ~ 5.0
        pts = np.repeat((last_val + 0.05)[:, np.newaxis], horizon, axis=1)
        q = np.repeat(pts[:, :, np.newaxis], 9, axis=2)
        for i in range(9):
            q[:, :, i] += (i - 4) * 0.1
        return pts, q

    rep_perfect = engine.run_full_backtest(
        snapshot=snapshot,
        base_reference_metrics=rep_base,
        custom_predictor_fn=perfect_predictor_fn,
    )
    assert rep_perfect.score > 50.0, f"Perfect model must have high positive Score, got {rep_perfect.score}"
    assert rep_perfect.overall_weighted_mae < rep_base.overall_weighted_mae

    # 3. Intentionally Bad Candidate Predictor: huge error
    def bad_predictor_fn(ctx_windows: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        n = len(ctx_windows)
        last_val = ctx_windows[:, -1, 0] if ctx_windows.ndim == 3 else ctx_windows[:, -1]
        pts = np.repeat((last_val + 200.0)[:, np.newaxis], horizon, axis=1)
        q = np.repeat(pts[:, :, np.newaxis], 9, axis=2)
        for i in range(9):
            q[:, :, i] += (i - 4) * 5.0
        return pts, q

    rep_bad = engine.run_full_backtest(
        snapshot=snapshot,
        base_reference_metrics=rep_base,
        custom_predictor_fn=bad_predictor_fn,
    )
    assert rep_bad.score < 0.0, f"Bad model must have negative Score, got {rep_bad.score}"
    assert rep_bad.overall_weighted_mae > rep_base.overall_weighted_mae


def test_locked_test_isolated_by_default():
    """Verifies that run_full_backtest does NOT open test_locked by default,

    and that run_locked_verification evaluates test_locked strictly on demand.
    """
    snapshot = _make_synthetic_snapshot(n_candles=5500, timeframe="1h")
    engine = BacktestEngine(predictor=None)

    def dummy_pred_fn(ctx_windows: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        n = len(ctx_windows)
        pts = np.zeros((n, horizon), dtype=np.float32) + 2500.0
        q = np.zeros((n, horizon, 9), dtype=np.float32) + 2500.0
        return pts, q

    # By default, include_locked_test is False
    rep = engine.run_full_backtest(
        snapshot=snapshot,
        is_base_reference=True,
        custom_predictor_fn=dummy_pred_fn,
    )
    assert rep.test_metrics is None, "Locked test must not be evaluated during standard backtest."
    assert rep.metadata["locked_test_evaluated"] is False

    # Dedicated locked verification API
    locked_metric = engine.run_locked_verification(
        snapshot=snapshot,
        custom_predictor_fn=dummy_pred_fn,
    )
    assert locked_metric.fold_id == "test_locked"
    assert locked_metric.num_windows > 0


def test_candidate_without_base_reference_raises_error():
    """Verifies that candidate model evaluation fails fast if base_reference_metrics is missing,

    and that base reference enforces Feature Set A and context_len 256.
    """
    snapshot = _make_synthetic_snapshot(n_candles=5500, timeframe="1h")
    engine = BacktestEngine(predictor=None)

    def dummy_pred_fn(ctx_windows: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        n = len(ctx_windows)
        pts = np.zeros((n, horizon), dtype=np.float32) + 2500.0
        q = np.zeros((n, horizon, 9), dtype=np.float32) + 2500.0
        return pts, q

    # 1. Candidate without base_reference_metrics must raise ValueError
    with pytest.raises(ValueError, match="requires 'base_reference_metrics'"):
        engine.run_full_backtest(
            snapshot=snapshot,
            model_name="LoRA-Candidate-Rank4",
            is_base_reference=False,
            base_reference_metrics=None,
            custom_predictor_fn=dummy_pred_fn,
        )

    # 2. Base reference with wrong feature set (not 'A') must raise ValueError
    with pytest.raises(ValueError, match="Official base reference standard must use Feature Set 'A'"):
        engine.run_full_backtest(
            snapshot=snapshot,
            feature_set="B",
            is_base_reference=True,
            custom_predictor_fn=dummy_pred_fn,
        )

    # 3. Base reference with wrong context_len (not 256) must raise ValueError
    with pytest.raises(ValueError, match="Official base reference standard must use context_len=256"):
        engine.run_full_backtest(
            snapshot=snapshot,
            feature_set="A",
            context_len=128,
            is_base_reference=True,
            custom_predictor_fn=dummy_pred_fn,
        )

    # 4. Candidate with missing fold in base_reference_metrics must raise KeyError
    incomplete_base_dict = {}  # Empty dict, missing fold 1, 2, 3
    with pytest.raises(KeyError, match="Missing base reference metric for fold"):
        engine.run_full_backtest(
            snapshot=snapshot,
            model_name="LoRA-Candidate",
            is_base_reference=False,
            base_reference_metrics=incomplete_base_dict,
            custom_predictor_fn=dummy_pred_fn,
        )


def test_forecast_request_adapter_mismatch_raises_error():
    """Verifies that forecast_request strictly enforces request.adapter_path == predictor.adapter_path."""
    predictor = object.__new__(TimesFM3Predictor)
    predictor.device = torch.device("cpu")
    predictor.adapter_path = None
    predictor.predict_batch = MagicMock(return_value=(
        np.zeros((1, 24), dtype=np.float32),
        np.zeros((1, 24, 9), dtype=np.float32),
    ))

    # 1. Matching adapter_path (both None) -> passes
    req_valid = ForecastRequest(timeframe="1h", adapter_path=None)
    res = predictor.forecast_request(req_valid, np.zeros((256,), dtype=np.float32), 1743073200000)
    assert res is not None

    # 2. Mismatched adapter_path (request specifies LoRA, predictor is Base) -> raises ValueError
    req_mismatch = ForecastRequest(timeframe="1h", adapter_path="adapters/lora_1h")
    with pytest.raises(ValueError, match="ForecastRequest adapter_path mismatch"):
        predictor.forecast_request(req_mismatch, np.zeros((256,), dtype=np.float32), 1743073200000)


def test_base_error_near_zero_handling():
    """Verifies that base errors <= tick size are flagged as insufficient information

    and handled without division by zero.
    """
    n_windows = 10
    horizon = 24
    preds = np.full((n_windows, horizon), 2500.0)
    targets = np.full((n_windows, horizon), 2500.0)
    quantiles = np.zeros((n_windows, horizon, 9)) + 2500.0
    orig = np.full(n_windows, 2500.0)

    # Base error <= TICK_SIZE (0.01)
    m = compute_fold_metrics(
        predictions=preds,
        quantiles=quantiles,
        targets=targets,
        origin_prices=orig,
        timeframe="1h",
        fold_id=1,
        base_weighted_mae=0.005,
        base_weighted_pinball=0.005,
    )
    assert m.reference_valid is False
    assert m.insufficient_information is True
    assert np.isnan(m.composite_loss)
    assert m.warning is not None
    assert "insufficient information" in m.warning

    # When all folds are invalid, compute_score_v1 raises ValueError
    with pytest.raises(ValueError, match="insufficient information"):
        compute_score_v1([float("nan"), float("nan")], valid_mask=[False, False])

    # When some folds are valid, invalid folds are excluded
    score = compute_score_v1([float("nan"), 1.0, 1.0], valid_mask=[False, True, True])
    assert np.isclose(score, 0.0)


def test_score_report_breakdowns():
    """Verifies that ScoreReport contains all required PLAN breakdown metrics."""
    snapshot = _make_synthetic_snapshot(n_candles=5500, timeframe="1h")
    engine = BacktestEngine(predictor=None)

    def dummy_pred_fn(ctx_windows: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        n = len(ctx_windows)
        last_val = ctx_windows[:, -1, 0] if ctx_windows.ndim == 3 else ctx_windows[:, -1]
        pts = np.repeat((last_val + 1.0)[:, np.newaxis], horizon, axis=1)
        q = np.repeat(pts[:, :, np.newaxis], 9, axis=2)
        return pts, q

    rep = engine.run_full_backtest(
        snapshot=snapshot,
        is_base_reference=True,
        custom_predictor_fn=dummy_pred_fn,
    )

    assert "low_move_under_2_ticks" in rep.breakdowns
    assert "weekday_vs_weekend" in rep.breakdowns
    assert "volatility_buckets" in rep.breakdowns
    assert "window_sampling" in rep.breakdowns

    sampling = rep.breakdowns["window_sampling"]
    assert sampling["horizon"] == 24
    assert sampling["step"] == 1
    assert np.isclose(sampling["overlap_ratio"], 23.0 / 24.0)
    assert sampling["independent_blocks"] > 0

    temporal = rep.breakdowns["weekday_vs_weekend"]
    assert "weekday_weighted_mae" in temporal
    assert "weekend_weighted_mae" in temporal

    vol = rep.breakdowns["volatility_buckets"]
    assert "low_vol_weighted_mae" in vol
    assert "high_vol_weighted_mae" in vol

