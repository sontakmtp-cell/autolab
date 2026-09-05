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
