"""Comprehensive PyTest suite for P1: Data ingestion, OHLC validation, features A/B/C, snapshots, and leak-free splitting."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from paxg_lab.constants import TIMEFRAME_HORIZONS
from paxg_lab.data.features import build_features, FEATURE_SPECS
from paxg_lab.data.snapshot import DatasetSnapshot, SNAPSHOTS_DIR
from paxg_lab.data.split import calculate_split_plan, extract_windows
from paxg_lab.data.storage import MarketDataStorage, DEFAULT_DB_PATH
from paxg_lab.data.validator import (
    validate_ohlc_integrity,
    detect_time_gaps,
    cross_validate_4h_with_1h,
    generate_data_quality_report,
)


@pytest.fixture
def sample_1h_klines():
    """Generates 100 clean consecutive 1h klines."""
    np.random.seed(42)
    start = 1743073200000
    step = 3600 * 1000
    rows = []
    base_price = 2500.0

    for i in range(100):
        t = start + i * step
        op = base_price + np.random.randn() * 2.0
        cl = op + np.random.randn() * 3.0
        hi = max(op, cl) + abs(np.random.randn() * 2.0)
        lo = min(op, cl) - abs(np.random.randn() * 2.0)
        vol = 100.0 + abs(np.random.randn() * 20.0)
        q_vol = vol * cl
        trades = int(50 + abs(np.random.randn() * 10))
        tb_vol = vol * 0.55
        tb_q_vol = tb_vol * cl

        rows.append({
            "source": "binance",
            "symbol": "PAXGUSDT",
            "interval": "1h",
            "open_time": t,
            "open": op,
            "high": hi,
            "low": lo,
            "close": cl,
            "volume": vol,
            "close_time": t + step - 1,
            "quote_volume": q_vol,
            "trades": trades,
            "taker_buy_volume": tb_vol,
            "taker_buy_quote_volume": tb_q_vol,
            "is_closed": 1,
        })
        base_price = cl

    return pd.DataFrame(rows)


def test_ohlc_integrity_validation(sample_1h_klines):
    # Clean dataset must be 100% valid
    report = validate_ohlc_integrity(sample_1h_klines)
    assert report["is_valid"]
    assert report["duplicate_timestamps"] == 0
    assert report["non_positive_prices"] == 0
    assert report["invalid_high_low"] == 0
    assert report["negative_volumes"] == 0

    # Corrupt a high price: high < max(open, close)
    corrupted = sample_1h_klines.copy()
    corrupted.loc[0, "high"] = corrupted.loc[0, "low"] - 10.0
    bad_report = validate_ohlc_integrity(corrupted)
    assert not bad_report["is_valid"]
    assert bad_report["invalid_high_low"] > 0

    # Negative volume check
    corrupted_vol = sample_1h_klines.copy()
    corrupted_vol.loc[5, "volume"] = -5.0
    bad_vol_report = validate_ohlc_integrity(corrupted_vol)
    assert not bad_vol_report["is_valid"]
    assert bad_vol_report["negative_volumes"] > 0


def test_gap_detection(sample_1h_klines):
    # No gaps in consecutive data
    gaps = detect_time_gaps(sample_1h_klines, interval="1h")
    assert len(gaps) == 0

    # Drop row index 10 to simulate missing candle
    with_gap = sample_1h_klines.drop(index=[10]).reset_index(drop=True)
    detected = detect_time_gaps(with_gap, interval="1h")
    assert len(detected) == 1
    assert detected[0]["missing_candles"] == 1


def test_cross_validation_4h_with_1h(sample_1h_klines):
    # Synthesize corresponding 4h klines from the 100 1h klines
    rows_4h = []
    step_4h = 4 * 3600 * 1000

    for i in range(0, 100 - 3, 4):
        chunk = sample_1h_klines.iloc[i : i + 4]
        t = chunk.iloc[0]["open_time"]
        rows_4h.append({
            "source": "binance",
            "symbol": "PAXGUSDT",
            "interval": "4h",
            "open_time": t,
            "open": chunk.iloc[0]["open"],
            "high": chunk["high"].max(),
            "low": chunk["low"].min(),
            "close": chunk.iloc[-1]["close"],
            "volume": chunk["volume"].sum(),
            "close_time": t + step_4h - 1,
            "quote_volume": chunk["quote_volume"].sum(),
            "trades": chunk["trades"].sum(),
            "taker_buy_volume": chunk["taker_buy_volume"].sum(),
            "taker_buy_quote_volume": chunk["taker_buy_quote_volume"].sum(),
            "is_closed": 1,
        })

    df_4h = pd.DataFrame(rows_4h)
    cv = cross_validate_4h_with_1h(sample_1h_klines, df_4h)
    assert cv["total_4h_candles"] == len(df_4h)
    assert cv["discrepancies"] == 0
    assert cv["match_ratio"] == 1.0


def test_feature_sets_construction(sample_1h_klines):
    # Set A
    feat_a, spec_a, _ = build_features(sample_1h_klines, feature_set="A")
    assert spec_a.name == "A"
    assert feat_a.shape == (len(sample_1h_klines), 1)
    assert np.isfinite(feat_a).all()

    # Set B
    feat_b, spec_b, _ = build_features(sample_1h_klines, feature_set="B")
    assert spec_b.name == "B"
    assert feat_b.shape == (len(sample_1h_klines), 9)
    assert np.isfinite(feat_b).all()

    # Synthetic mark and funding for Set C
    df_mark = sample_1h_klines[["open_time", "close"]].copy()
    df_mark["close"] = df_mark["close"] + 0.5

    funding_times = [sample_1h_klines.iloc[0]["open_time"] + i * 8 * 3600 * 1000 for i in range(15)]
    df_funding = pd.DataFrame({
        "symbol": "PAXGUSDT",
        "funding_time": funding_times,
        "funding_rate": [0.0001] * len(funding_times),
        "mark_price": [2500.0] * len(funding_times),
    })

    feat_c, spec_c, _ = build_features(
        sample_1h_klines,
        df_mark=df_mark,
        df_funding=df_funding,
        feature_set="C",
    )
    assert spec_c.name == "C"
    assert feat_c.shape == (len(sample_1h_klines), 11)
    assert np.isfinite(feat_c).all()


def test_feature_set_c_missing_mark_raises_error(sample_1h_klines):
    # Missing mark candles must raise ValueError instead of silently imputing trade close
    df_mark_missing = sample_1h_klines[["open_time", "close"]].iloc[:-5].copy()
    funding_times = [sample_1h_klines.iloc[0]["open_time"] + i * 8 * 3600 * 1000 for i in range(15)]
    df_funding = pd.DataFrame({
        "symbol": "PAXGUSDT",
        "funding_time": funding_times,
        "funding_rate": [0.0001] * len(funding_times),
        "mark_price": [2500.0] * len(funding_times),
    })

    with pytest.raises(ValueError, match="Feature Set C unavailable: missing mark price data"):
        build_features(
            sample_1h_klines,
            df_mark=df_mark_missing,
            df_funding=df_funding,
            feature_set="C",
        )


def test_snapshot_creation_and_integrity(sample_1h_klines, tmp_path):
    feat_a, _, ts = build_features(sample_1h_klines, feature_set="A")
    feat_b, _, _ = build_features(sample_1h_klines, feature_set="B")

    snap = DatasetSnapshot.create(
        timeframe="1h",
        timestamps=np.array(ts),
        features_a=feat_a,
        features_b=feat_b,
        symbol="PAXGUSDT",
    )
    snap_dir = snap.save(base_dir=tmp_path)

    # Load and verify hash
    loaded = DatasetSnapshot.load(snap_dir, verify_hash=True)
    assert loaded.metadata.sha256 == snap.metadata.sha256
    assert np.array_equal(loaded.timestamps, snap.timestamps)
    assert np.allclose(loaded.features_a, snap.features_a)
    assert np.allclose(loaded.features_b, snap.features_b)

    # Corrupt data.npz on disk and ensure verify_hash raises error
    npz_file = snap_dir / "data.npz"
    with open(npz_file, "r+b") as f:
        f.seek(100)
        f.write(b"\xff\xff\xff\xff")

    with pytest.raises(RuntimeError, match="Snapshot integrity violation"):
        DatasetSnapshot.load(snap_dir, verify_hash=True)


def test_split_plan_and_purge_buffers():
    # 1h test: horizon=24, buffer=24
    plan_1h = calculate_split_plan(total_candles=12000, timeframe="1h")
    assert plan_1h.horizon == 24
    assert plan_1h.timeframe == "1h"
    for fold in plan_1h.eval_folds:
        assert fold.purge_buffer == 24
        # Assert evaluation start is strictly after training ceiling + purge buffer
        assert fold.eval_start - fold.val_early_stop_end == 24

    # 4h test: horizon=6, buffer=6
    plan_4h = calculate_split_plan(total_candles=3000, timeframe="4h")
    assert plan_4h.horizon == 6
    assert plan_4h.timeframe == "4h"
    for fold in plan_4h.eval_folds:
        assert fold.purge_buffer == 6
        assert fold.eval_start - fold.val_early_stop_end == 6


def test_extract_windows_no_future_leakage():
    n_pts = 500
    features = np.arange(n_pts, dtype=np.float32).reshape(-1, 1)
    targets = np.arange(n_pts, dtype=np.float32)

    context_len = 32
    horizon = 24
    start_idx = 0
    end_idx = 200

    ctx_w, fut_w, origins = extract_windows(
        features=features,
        targets=targets,
        context_len=context_len,
        horizon=horizon,
        start_idx=start_idx,
        end_idx=end_idx,
    )

    assert len(ctx_w) == len(fut_w) == len(origins)
    assert len(ctx_w) > 0

    # Ensure no future target window exceeded end_idx
    for fut in fut_w:
        assert fut[-1] <= end_idx - 1
        assert len(fut) == horizon


def test_extract_windows_start_idx_boundary():
    n_pts = 500
    features = np.arange(n_pts, dtype=np.float32).reshape(-1, 1)
    targets = np.arange(n_pts, dtype=np.float32)

    context_len = 32
    horizon = 24
    start_idx = 100
    end_idx = 200

    ctx_w, fut_w, origins = extract_windows(
        features=features,
        targets=targets,
        context_len=context_len,
        horizon=horizon,
        start_idx=start_idx,
        end_idx=end_idx,
    )

    assert len(ctx_w) == len(fut_w) == len(origins)
    assert len(ctx_w) > 0

    for fut, orig in zip(fut_w, origins):
        # Target must be strictly inside [start_idx, end_idx)
        assert orig + 1 >= start_idx, f"origin + 1 ({orig + 1}) must be >= start_idx ({start_idx})"
        assert fut[0] >= start_idx
        assert orig + horizon < end_idx, f"origin + horizon ({orig + horizon}) must be < end_idx ({end_idx})"
        assert fut[-1] < end_idx
        assert len(fut) == horizon


def test_extract_windows_gap_rejection():
    n_pts = 200
    step_ms = 3600 * 1000
    start_time = 1743073200000
    ts = np.array([start_time + i * step_ms for i in range(n_pts)], dtype=np.int64)

    # Introduce a gap at index 100: make ts[100] jump by 5 hours
    ts_with_gap = ts.copy()
    ts_with_gap[100:] += 5 * step_ms

    features = np.arange(n_pts, dtype=np.float32).reshape(-1, 1)
    targets = np.arange(n_pts, dtype=np.float32)

    context_len = 20
    horizon = 10

    ctx_w, fut_w, origins = extract_windows(
        features=features,
        targets=targets,
        context_len=context_len,
        horizon=horizon,
        start_idx=0,
        end_idx=n_pts,
        timestamps=ts_with_gap,
        expected_interval_ms=step_ms,
    )

    assert len(ctx_w) > 0

    # No extracted window must cross the gap transition between index 99 and 100
    for orig in origins:
        w_start = orig - context_len + 1
        w_end = orig + horizon
        assert not (w_start <= 99 < w_end), f"Window at origin {orig} encompasses the gap transition!"


def test_market_data_storage_unit(tmp_path):
    db_file = tmp_path / "test_storage.db"
    storage = MarketDataStorage(db_file)

    # Test empty queries
    assert storage.get_latest_kline_time("PAXGUSDT", "1h") is None
    assert storage.get_latest_mark_kline_time("PAXGUSDT", "1h") is None
    assert storage.get_latest_funding_time("PAXGUSDT") is None
    assert storage.load_latest_quality_report() is None

    # Test saving klines
    kline_row = [[1743073200000, "2500", "2510", "2490", "2505", "10", 1743076799999, "25000", 5, "5", "12500", "0"]]
    inserted = storage.save_klines(kline_row, symbol="PAXGUSDT", interval="1h")
    assert inserted == 1
    assert storage.get_latest_kline_time("PAXGUSDT", "1h") == 1743073200000

    # Test saving mark klines
    mark_row = [[1743073200000, "2501", "2511", "2491", "2506", "0", 1743076799999]]
    inserted_mark = storage.save_mark_klines(mark_row, symbol="PAXGUSDT", interval="1h")
    assert inserted_mark == 1
    assert storage.get_latest_mark_kline_time("PAXGUSDT", "1h") == 1743073200000

    # Test saving funding rates
    funding_data = [{"symbol": "PAXGUSDT", "fundingTime": 1743073200000, "fundingRate": "0.0001", "markPrice": "2501"}]
    inserted_funding = storage.save_funding_rates(funding_data, symbol="PAXGUSDT")
    assert inserted_funding == 1
    assert storage.get_latest_funding_time("PAXGUSDT") == 1743073200000

    # Test saving and reading data quality report
    sample_report = {
        "status": "HEALTHY",
        "1h": {"candle_count": 100, "gaps_count": 0},
        "4h": {"candle_count": 25, "gaps_count": 0},
        "cross_validation_4h_vs_1h": {"discrepancies": 0},
    }
    report_id = storage.save_quality_report(sample_report, timeframe="1h+4h")
    assert report_id > 0

    latest_rep = storage.load_latest_quality_report()
    assert latest_rep is not None
    assert latest_rep["id"] == report_id
    assert latest_rep["total_candles"] == 125
    assert latest_rep["gaps_count"] == 0
    assert latest_rep["report"]["status"] == "HEALTHY"


@pytest.mark.integration
def test_real_database_and_snapshots():
    """Validates real ingested database and snapshots on disk."""
    if not DEFAULT_DB_PATH.exists() or not list(SNAPSHOTS_DIR.glob("paxgusdt_*")):
        pytest.skip("Integration test skipped: real DB or snapshots not found in var/paxg_lab/")

    storage = MarketDataStorage()
    df_1h = storage.load_klines_df(symbol="PAXGUSDT", interval="1h")
    df_4h = storage.load_klines_df(symbol="PAXGUSDT", interval="4h")

    assert len(df_1h) >= 12000, f"Expected >= 12000 1h candles, got {len(df_1h)}"
    assert len(df_4h) >= 3000, f"Expected >= 3000 4h candles, got {len(df_4h)}"

    # Check snapshots directory
    snap_dirs = list(SNAPSHOTS_DIR.glob("paxgusdt_*"))
    assert len(snap_dirs) >= 2, f"Expected at least 2 snapshots, got {len(snap_dirs)}"

    for s_dir in snap_dirs:
        snap = DatasetSnapshot.load(s_dir, verify_hash=True)
        assert snap.total_candles > 0
        assert snap.metadata.sha256 is not None

