"""Unit and integration tests for P5 Web UI components, adapter registry, and safe export/import."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import time
import zipfile

import pytest

from paxg_lab.constants import TIMEFRAME_1H, TIMEFRAME_4H, get_horizon_for_timeframe
from paxg_lab.model.manifest import AdapterManifest
from paxg_lab.model.store import AdapterStore, compute_file_sha256
from paxg_lab.ui.charts import build_backtest_error_chart, build_candlestick_forecast_chart, build_loss_chart
from paxg_lab.ui.components.tab_forecast import _parse_forecast_result
from paxg_lab.ui.state import (
    get_candle_count,
    get_gpu_queue_summary,
    get_latest_candle,
    get_recent_candles,
    timestamp_to_vietnam_str,
)


@pytest.fixture
def temp_store(tmp_path: Path) -> AdapterStore:
    """Creates an isolated AdapterStore with its own database and directory."""
    base_dir = tmp_path / "adapters"
    base_dir.mkdir(parents=True, exist_ok=True)
    db_path = tmp_path / "paxg_lab.db"
    return AdapterStore(base_dir=base_dir, db_path=db_path)


@pytest.fixture
def mock_adapter(temp_store: AdapterStore) -> str:
    """Creates a mock valid adapter in temp_store."""
    adapter_id = "paxg_1h_test_adapter_001"
    target_dir = temp_store.get_adapter_path(adapter_id)
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. Weights dummy safetensors
    weights_path = target_dir / "adapter_model.safetensors"
    weights_path.write_bytes(b"DUMMY_SAFETENSORS_WEIGHTS_CONTENT")

    # 2. Config dummy
    config_path = target_dir / "adapter_config.json"
    config_path.write_text(json.dumps({"r": 4, "lora_alpha": 8}), encoding="utf-8")

    # 3. Manifest
    manifest = AdapterManifest(
        adapter_id=adapter_id,
        timeframe="1h",
        horizon=24,
        context_len=256,
        feature_set="B",
        feature_columns=["close", "volume"],
        best_val_loss=0.012345,
    )
    manifest.save_json(target_dir / "paxg_manifest.json")

    # 4. Checksums
    checksums_path = target_dir / "checksums.sha256"
    with open(checksums_path, "w", encoding="utf-8") as f:
        for fpath in sorted(target_dir.glob("*")):
            if fpath.name != "checksums.sha256":
                digest = compute_file_sha256(fpath)
                f.write(f"{digest}  {fpath.name}\n")

    return adapter_id


# ---------------------------------------------------------------------------
# 1. Adapter Registry: Pin, Alias, Recommended Tests
# ---------------------------------------------------------------------------


def test_adapter_registry_pin_alias_recommended(temp_store: AdapterStore, mock_adapter: str):
    """Verifies pin, alias, and recommended adapter tracking in SQLite registry."""
    # 1. Initially not pinned and not recommended
    assert temp_store.is_pinned(mock_adapter) is False
    assert temp_store.get_recommended("1h") is None

    # 2. Set alias
    temp_store.set_alias(mock_adapter, "Mô hình Vàng 1h Tốt Nhất")
    assert temp_store.get_alias(mock_adapter) == "Mô hình Vàng 1h Tốt Nhất"

    # 3. Set pinned
    temp_store.set_pinned(mock_adapter, True)
    assert temp_store.is_pinned(mock_adapter) is True

    # 4. Set recommended for 1h
    temp_store.set_recommended(mock_adapter, "1h")
    assert temp_store.get_recommended("1h") == mock_adapter
    # Setting recommended automatically pins
    assert temp_store.is_pinned(mock_adapter) is True

    # 5. Unpin
    temp_store.set_pinned(mock_adapter, False)
    assert temp_store.is_pinned(mock_adapter) is False


def test_adapter_deletion_protection(temp_store: AdapterStore, mock_adapter: str):
    """Verifies that pinned or recommended adapters cannot be deleted."""
    # Pin adapter
    temp_store.set_pinned(mock_adapter, True)
    with pytest.raises(ValueError, match="đã được ghim"):
        temp_store.delete_adapter(mock_adapter, use_trash=True)

    # Unpin but set recommended
    temp_store.set_pinned(mock_adapter, False)
    temp_store.set_recommended(mock_adapter, "1h")
    with pytest.raises(ValueError, match="adapter khuyến nghị"):
        temp_store.delete_adapter(mock_adapter, use_trash=True)

    # Clear recommended and unpin -> deletion to trash succeeds
    with sqlite3.connect(str(temp_store.db_path)) as conn:
        conn.execute("UPDATE adapter_registry SET is_recommended = 0, is_pinned = 0 WHERE adapter_id = ?;", (mock_adapter,))
        conn.commit()

    assert temp_store.delete_adapter(mock_adapter, use_trash=True) is True
    assert not temp_store.get_adapter_path(mock_adapter).exists()

    # Restore from trash
    assert temp_store.restore_adapter(mock_adapter) is True
    assert temp_store.get_adapter_path(mock_adapter).exists()


# ---------------------------------------------------------------------------
# 2. Safe Zip Export and Import Tests
# ---------------------------------------------------------------------------


def test_safe_zip_export_and_import(temp_store: AdapterStore, mock_adapter: str, tmp_path: Path):
    """Verifies safe zip export bundles required files and clean import succeeds."""
    export_zip = tmp_path / "exported_adapter.zip"
    temp_store.export_adapter_zip(mock_adapter, export_zip)
    assert export_zip.is_file()

    # Inspect zip contents: must contain safetensors, manifest, checksums
    with zipfile.ZipFile(export_zip, "r") as zf:
        names = zf.namelist()
        assert "adapter_model.safetensors" in names
        assert "paxg_manifest.json" in names
        assert "checksums.sha256" in names
        # Must not contain python files
        assert not any(n.endswith(".py") or n.endswith(".pkl") for n in names)

    # Import into a fresh store
    new_store_dir = tmp_path / "new_store"
    new_store = AdapterStore(base_dir=new_store_dir, db_path=tmp_path / "new.db")
    imported_id = new_store.import_adapter_zip(export_zip)
    assert imported_id == mock_adapter
    assert new_store.get_adapter_path(imported_id).is_dir()


def test_safe_zip_import_rejection_of_path_traversal(temp_store: AdapterStore, tmp_path: Path):
    """Verifies that zip files with path traversal are strictly rejected."""
    bad_zip = tmp_path / "bad_traversal.zip"
    with zipfile.ZipFile(bad_zip, "w") as zf:
        zf.writestr("../evil.txt", "MALICIOUS")

    with pytest.raises(ValueError, match="Path traversal detected"):
        temp_store.import_adapter_zip(bad_zip)


def test_safe_zip_import_rejection_of_code_or_pickle(temp_store: AdapterStore, tmp_path: Path):
    """Verifies that zip files with .py or .pkl files are strictly rejected."""
    bad_zip = tmp_path / "bad_code.zip"
    with zipfile.ZipFile(bad_zip, "w") as zf:
        zf.writestr("exploit.py", "import os; os.system('calc')")

    with pytest.raises(ValueError, match="Dangerous file extension"):
        temp_store.import_adapter_zip(bad_zip)


# ---------------------------------------------------------------------------
# 3. Horizon and Time Formatting Tests
# ---------------------------------------------------------------------------


def test_horizon_convention_mandate():
    """Verifies horizon is strictly 24 for 1h and 6 for 4h."""
    assert get_horizon_for_timeframe(TIMEFRAME_1H) == 24
    assert get_horizon_for_timeframe(TIMEFRAME_4H) == 6

    # 1h: 24 steps * 1 hour = 24 hours
    # 4h: 6 steps * 4 hours = 24 hours
    assert 24 * 1 == 24
    assert 6 * 4 == 24


def test_timestamp_to_vietnam_time():
    """Verifies UTC timestamp conversion to Vietnam time (+7h)."""
    # 2026-09-06 00:00:00 UTC = 1788652800000 ms
    # Vietnam time should be 2026-09-06 07:00:00
    ts_utc = 1788652800000
    formatted = timestamp_to_vietnam_str(ts_utc)
    assert "07:00" in formatted
    assert "06/09/2026" in formatted


def test_parse_forecast_result_1h_and_4h():
    """Verifies forecast result parser generates exact 24 steps for 1h and 6 steps for 4h."""
    # 1h: 24 steps
    fake_res_1h = {
        "q50": [2500.0 + i for i in range(24)],
        "q10": [2480.0 + i for i in range(24)],
        "q90": [2520.0 + i for i in range(24)],
        "timestamps": [1788652800000 + i * 3600000 for i in range(24)],
    }
    steps_1h = _parse_forecast_result(fake_res_1h, "1h")
    assert len(steps_1h) == 24
    assert steps_1h[0]["step"] == 1
    assert steps_1h[-1]["step"] == 24

    # 4h: 6 steps
    fake_res_4h = {
        "q50": [2500.0 + i * 5 for i in range(6)],
        "q10": [2470.0 + i * 5 for i in range(6)],
        "q90": [2530.0 + i * 5 for i in range(6)],
        "timestamps": [1788652800000 + i * 14400000 for i in range(6)],
    }
    steps_4h = _parse_forecast_result(fake_res_4h, "4h")
    assert len(steps_4h) == 6
    assert steps_4h[0]["step"] == 1
    assert steps_4h[-1]["step"] == 6


# ---------------------------------------------------------------------------
# 4. Plotly Charts Generation Tests
# ---------------------------------------------------------------------------


def test_plotly_chart_builders():
    """Verifies Plotly chart builders construct valid Figure objects without exceptions."""
    import pandas as pd

    # 1. Candlestick with forecast
    candles = pd.DataFrame([
        {"time_vn": "2026-09-06 01:00", "open": 2500, "high": 2510, "low": 2490, "close": 2505, "volume": 100},
        {"time_vn": "2026-09-06 02:00", "open": 2505, "high": 2515, "low": 2500, "close": 2512, "volume": 120},
    ])
    fc_steps = [
        {"step": 1, "time_vn": "2026-09-06 03:00", "q50": 2515.0, "q10": 2495.0, "q90": 2535.0},
        {"step": 2, "time_vn": "2026-09-06 04:00", "q50": 2520.0, "q10": 2498.0, "q90": 2542.0},
    ]
    fig_candle = build_candlestick_forecast_chart(candles, fc_steps, "1h")
    assert fig_candle is not None
    assert len(fig_candle.data) >= 3  # Candlestick, upper bound, lower bound ribbon, median line

    # 2. Loss chart
    fig_loss = build_loss_chart([0.02, 0.015, 0.012], [0.018, 0.014, 0.011])
    assert fig_loss is not None
    assert len(fig_loss.data) == 2

    # 3. Backtest error chart
    fig_err = build_backtest_error_chart({1: 2.5, 6: 4.8, 12: 7.1, 24: 11.2}, {1: 3.0, 6: 5.2, 12: 7.8, 24: 12.0})
    assert fig_err is not None
    assert len(fig_err.data) == 2
