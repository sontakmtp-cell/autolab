"""Unit and integration tests for P5 Web UI components, adapter registry, and safe export/import."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import time
import zipfile

import pytest
import safetensors.torch
import torch

from paxg_lab.constants import (
    ALLOWED_CONTEXT_LENGTHS,
    MODEL_REPO,
    MODEL_REVISION,
    TIMEFRAME_1H,
    TIMEFRAME_4H,
    get_horizon_for_timeframe,
)
from paxg_lab.data.features import FEATURE_SPECS
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

def make_valid_lora_safetensors_bytes(
    r: int = 4,
    in_dim: int = 1280,
    out_dim: int = 1280,
    target_modules: tuple[str, ...] = ("query_proj", "value_proj"),
    num_layers: int = 20,
) -> bytes:
    tensors = {}
    for l_idx in range(num_layers):
        for attn in ("seq_attn", "var_attn"):
            for tm in target_modules:
                tensors[f"base_model.model.transformer_stack.layers.{l_idx}.{attn}.{tm}.lora_A.weight"] = torch.zeros(r, in_dim)
                tensors[f"base_model.model.transformer_stack.layers.{l_idx}.{attn}.{tm}.lora_B.weight"] = torch.zeros(out_dim, r)
    return safetensors.torch.save(tensors)


VALID_SAFETENSORS_BYTES = make_valid_lora_safetensors_bytes(r=4)
VALID_LORA_CONFIG = {
    "peft_type": "LORA",
    "r": 4,
    "lora_alpha": 8,
    "target_modules": ["query_proj", "value_proj"],
}


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

    # 1. Weights valid safetensors
    weights_path = target_dir / "adapter_model.safetensors"
    weights_path.write_bytes(VALID_SAFETENSORS_BYTES)

    # 2. Config valid LoRA
    config_path = target_dir / "adapter_config.json"
    config_path.write_text(json.dumps(VALID_LORA_CONFIG), encoding="utf-8")

    # 3. Manifest
    manifest = AdapterManifest(
        adapter_id=adapter_id,
        timeframe="1h",
        horizon=24,
        context_len=256,
        feature_set="B",
        feature_columns=list(FEATURE_SPECS["B"].columns),
        base_model_repo=MODEL_REPO,
        base_model_revision=MODEL_REVISION,
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

    with pytest.raises(ValueError, match=r"(Dangerous|Disallowed).*file extension"):
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


# ---------------------------------------------------------------------------
# 5. Security & Schema Regression Tests (Comment ID: 5557464228)
# ---------------------------------------------------------------------------


def test_validate_adapter_id_rejections(temp_store: AdapterStore):
    """Verifies that validate_adapter_id rejects directory traversal, absolute paths, and invalid slugs."""
    from paxg_lab.model.store import validate_adapter_id

    # Valid slugs
    assert validate_adapter_id("paxg_1h_r4_setB", temp_store.base_dir) == "paxg_1h_r4_setB"
    assert validate_adapter_id("adapter-001", temp_store.base_dir) == "adapter-001"
    assert validate_adapter_id("test.lora_v1", temp_store.base_dir) == "test.lora_v1"

    # Malicious or invalid IDs
    invalid_ids = [
        "../../escape",
        "..",
        ".",
        "C:\\Windows\\System32",
        "D:/games",
        "/etc/passwd",
        "foo/../bar",
        "foo\\..\\bar",
        "nested/dir",
        "evil:drive",
        "",
        "   ",
        "bad name with spaces",
    ]
    for bad_id in invalid_ids:
        with pytest.raises(ValueError):
            validate_adapter_id(bad_id, temp_store.base_dir)


def test_safe_zip_import_rejection_of_manifest_traversal_adapter_id(temp_store: AdapterStore, tmp_path: Path):
    """Verifies that zip archives with traversal adapter_id inside paxg_manifest.json are rejected."""
    malicious_ids = ["../../escape", "C:\\Windows\\System32", "/tmp/evil", "foo/bar"]

    safetensors_bytes = VALID_SAFETENSORS_BYTES
    config_bytes = json.dumps(VALID_LORA_CONFIG).encode("utf-8")
    s_hash = hashlib.sha256(safetensors_bytes).hexdigest()
    c_hash = hashlib.sha256(config_bytes).hexdigest()

    for idx, bad_id in enumerate(malicious_ids):
        bad_zip = tmp_path / f"bad_manifest_id_{idx}.zip"
        manifest_data = {
            "adapter_id": bad_id,
            "timeframe": "1h",
            "horizon": 24,
            "context_len": 256,
            "feature_set": "A",
            "feature_columns": ["close"],
            "base_model_repo": MODEL_REPO,
            "base_model_revision": MODEL_REVISION,
            "best_val_loss": 0.012,
        }
        manifest_str = json.dumps(manifest_data)
        m_hash = hashlib.sha256(manifest_str.encode("utf-8")).hexdigest()
        checksums_content = (
            f"{m_hash}  paxg_manifest.json\n"
            f"{s_hash}  adapter_model.safetensors\n"
            f"{c_hash}  adapter_config.json\n"
        )

        with zipfile.ZipFile(bad_zip, "w") as zf:
            zf.writestr("paxg_manifest.json", manifest_str)
            zf.writestr("adapter_model.safetensors", safetensors_bytes)
            zf.writestr("adapter_config.json", config_bytes)
            zf.writestr("checksums.sha256", checksums_content)

        with pytest.raises(ValueError, match=r"Path traversal|Reserved path|Invalid adapter_id format"):
            temp_store.import_adapter_zip(bad_zip)

        # Verify nothing was created outside base_dir
        outside_escape = temp_store.base_dir.parent / "escape"
        assert not outside_escape.exists()


def test_parse_forecast_result_with_real_forecast_result_schema():
    """Verifies parser handles exact ForecastResult.to_dict() schema from worker."""
    import numpy as np
    from paxg_lab.eval.types import ForecastResult

    origin_time = 1788652800000
    target_ts = [origin_time + (i + 1) * 3600000 for i in range(24)]
    point_fc = np.array([2500.0 + i for i in range(24)], dtype=np.float32)
    lower_fc = point_fc - 15.0
    upper_fc = point_fc + 20.0
    quantiles_fc = np.repeat(point_fc[:, None], 9, axis=1)

    fc_result = ForecastResult(
        symbol="PAXGUSDT",
        timeframe="1h",
        forecast_origin_time=origin_time,
        target_timestamps=target_ts,
        point_forecast=point_fc,
        quantiles=quantiles_fc,
        uncertainty_lower=lower_fc,
        uncertainty_upper=upper_fc,
    )

    real_dict = fc_result.to_dict()
    steps = _parse_forecast_result(real_dict, "1h")

    assert len(steps) == 24
    assert steps[0]["step"] == 1
    assert steps[0]["q50"] == pytest.approx(2500.0)
    assert steps[0]["q10"] == pytest.approx(2485.0)
    assert steps[0]["q90"] == pytest.approx(2520.0)
    assert steps[0]["timestamp_ms"] == target_ts[0]
    # Vietnam time formatting must not be a raw +1h fallback
    assert "08:00" in steps[0]["time_vn"]
    assert not steps[0]["time_vn"].startswith("+")


def test_backtest_report_view_with_real_score_report_schema():
    """Verifies backtest report view handles exact ScoreReport.to_dict() schema without errors."""
    from paxg_lab.eval.types import FoldMetrics, ScoreReport
    from paxg_lab.ui.components.tab_backtest import _render_backtest_report_view

    fm1 = FoldMetrics(
        fold_id=1,
        num_windows=50,
        weighted_mae=24.5,
        weighted_pinball=11.2,
        rmse=35.0,
        mae=25.0,
        coverage_80=0.76,
        mean_width_80=95.0,
        directional_accuracy=0.53,
    )

    score_report = ScoreReport(
        timeframe="1h",
        horizon=24,
        model_name="TimesFM3-LoRA-Candidate",
        score=2.85,
        overall_weighted_mae=23.4,
        overall_weighted_pinball=10.8,
        coverage_80=0.78,
        directional_accuracy=0.54,
        step_mae={1: 8.1, 6: 21.3, 12: 32.0, 24: 45.5},
        fold_metrics=[fm1],
        baseline_comparisons={
            "naive_flat_eval_weighted_mae": 26.2,
            "base_overall_weighted_mae": 25.0,
            "base_overall_weighted_pinball": 11.5,
            "base_coverage_80": 0.76,
            "base_directional_accuracy": 0.52,
        },
    )

    report_dict = score_report.to_dict()
    assert report_dict["score"] == 2.85
    assert report_dict["overall_weighted_mae"] == 23.4
    assert 1 in report_dict["step_mae"] or "1" in report_dict["step_mae"]

    # Calling _render_backtest_report_view should execute cleanly without raising
    _render_backtest_report_view(report_dict, "1h")


def test_tab_adapter_manager_uuid_import():
    """Verifies that tab_adapter_manager defines uuid and handles import temporary naming."""
    import paxg_lab.ui.components.tab_adapter_manager as tam

    assert hasattr(tam, "uuid")
    uid = tam.uuid.uuid4().hex
    assert len(uid) == 32


# ---------------------------------------------------------------------------
# 6. Review Round 2 Regression Tests (Comment ID: 5557593819)
# ---------------------------------------------------------------------------


def test_train_spec_instantiation_from_ui_values():
    """Verifies TrainSpec builds cleanly from UI values without TypeError (patience vs early_stopping_patience),
    and validates history_days mapping (180, 365, 'all') and PLAN 3.2 hyperparameter bounds.
    """
    from paxg_lab.model.train_spec import TrainSpec

    # Simulate UI controls mapping
    ui_history_options = {
        "365 ngày (Khuyến nghị)": 365,
        "180 ngày": 180,
        "Toàn bộ lịch sử khả dụng": "all",
    }

    for opt_text, expected_val in ui_history_options.items():
        if "180" in opt_text:
            mapped_hd = 180
        elif "365" in opt_text:
            mapped_hd = 365
        else:
            mapped_hd = "all"

        assert mapped_hd == expected_val

        # Construct TrainSpec exactly as tab_training does
        spec = TrainSpec(
            timeframe="1h",
            horizon=24,
            context_len=256,
            feature_set="B",
            lora_r=4,
            lora_alpha=8,
            lora_dropout=0.10,
            learning_rate=5e-5,
            max_epochs=5,
            batch_size=2,
            gradient_accumulation_steps=8,
            weight_decay=0.01,
            early_stopping_patience=2,  # Must be early_stopping_patience, NOT patience
            grad_clip_norm=1.0,
            history_days=mapped_hd,
            seed=42,
        )

        assert spec.history_days == expected_val
        assert spec.early_stopping_patience == 2
        assert spec.effective_batch_size == 16
        assert spec.weight_decay == 0.01
        assert spec.grad_clip_norm == 1.0


def test_timeframe_state_isolation_prevents_cross_contamination():
    """Verifies that switching 1h -> 4h -> 1h results do not cross-contaminate visualizations."""
    from paxg_lab.ui.components.tab_backtest import _render_backtest_report_view

    # 1h forecast result
    fc_1h = {
        "timeframe": "1h",
        "q50": [2500.0] * 24,
        "timestamps": [1788652800000 + i * 3600000 for i in range(24)],
    }
    # Parsing 1h result under 1h should produce 24 steps
    steps_1h = _parse_forecast_result(fc_1h, "1h")
    assert len(steps_1h) == 24

    # Parsing 1h result under 4h must be rejected (returns empty list to prevent cross-contamination)
    steps_cross = _parse_forecast_result(fc_1h, "4h")
    assert len(steps_cross) == 0

    # 4h forecast result
    fc_4h = {
        "timeframe": "4h",
        "q50": [2500.0] * 6,
        "timestamps": [1788652800000 + i * 14400000 for i in range(6)],
    }
    steps_4h = _parse_forecast_result(fc_4h, "4h")
    assert len(steps_4h) == 6

    steps_cross_4h_to_1h = _parse_forecast_result(fc_4h, "1h")
    assert len(steps_cross_4h_to_1h) == 0

    # Backtest report timeframe mismatch should be safely rejected
    report_1h = {
        "timeframe": "1h",
        "model_name": "TestModel-1h",
        "score": 1.5,
    }
    # Should not raise any error and safely warn
    _render_backtest_report_view(report_1h, "4h")


def test_safe_zip_import_rejects_checksum_traversal_and_uncovered_files(temp_store: AdapterStore, tmp_path: Path):
    """Verifies that checksums.sha256 with traversal paths, absolute paths, or unverified files are strictly rejected."""
    import hashlib

    safetensors_bytes = VALID_SAFETENSORS_BYTES
    config_bytes = json.dumps(VALID_LORA_CONFIG).encode("utf-8")
    s_hash = hashlib.sha256(safetensors_bytes).hexdigest()
    c_hash = hashlib.sha256(config_bytes).hexdigest()

    base_manifest = {
        "adapter_id": "test_traversal",
        "timeframe": "1h",
        "horizon": 24,
        "context_len": 256,
        "feature_set": "A",
        "feature_columns": ["close"],
        "base_model_repo": MODEL_REPO,
        "base_model_revision": MODEL_REVISION,
        "best_val_loss": 0.01,
    }
    manifest_str = json.dumps(base_manifest)
    m_hash = hashlib.sha256(manifest_str.encode("utf-8")).hexdigest()

    base_checksums = (
        f"{m_hash}  paxg_manifest.json\n"
        f"{s_hash}  adapter_model.safetensors\n"
        f"{c_hash}  adapter_config.json\n"
    )

    # 1. Traversal path in checksums.sha256 (../outside)
    zip_traversal = tmp_path / "bad_checksum_traversal.zip"
    with zipfile.ZipFile(zip_traversal, "w") as zf:
        zf.writestr("paxg_manifest.json", manifest_str)
        zf.writestr("adapter_model.safetensors", safetensors_bytes)
        zf.writestr("adapter_config.json", config_bytes)
        zf.writestr("checksums.sha256", base_checksums + "1234567890abcdef  ../outside_file.txt\n")

    with pytest.raises(ValueError, match="Path traversal detected in checksums filename"):
        temp_store.import_adapter_zip(zip_traversal)

    # 2. Absolute / Drive path in checksums.sha256 (C:\Windows\...)
    zip_drive = tmp_path / "bad_checksum_drive.zip"
    with zipfile.ZipFile(zip_drive, "w") as zf:
        zf.writestr("paxg_manifest.json", manifest_str)
        zf.writestr("adapter_model.safetensors", safetensors_bytes)
        zf.writestr("adapter_config.json", config_bytes)
        zf.writestr("checksums.sha256", base_checksums + "1234567890abcdef  C:\\Windows\\system.ini\n")

    with pytest.raises(ValueError, match="Path traversal detected in checksums filename"):
        temp_store.import_adapter_zip(zip_drive)

    # 3. UNC path in checksums.sha256 (//server/share or \\\\server\\share)
    zip_unc = tmp_path / "bad_checksum_unc.zip"
    with zipfile.ZipFile(zip_unc, "w") as zf:
        zf.writestr("paxg_manifest.json", manifest_str)
        zf.writestr("adapter_model.safetensors", safetensors_bytes)
        zf.writestr("adapter_config.json", config_bytes)
        zf.writestr("checksums.sha256", base_checksums + "1234567890abcdef  \\\\server\\share\\evil.txt\n")

    with pytest.raises(ValueError, match="Path traversal detected in checksums filename"):
        temp_store.import_adapter_zip(zip_unc)

    # 4. Uncovered file in archive: archive contains extra file not in checksums.sha256
    zip_uncovered = tmp_path / "uncovered_file.zip"
    with zipfile.ZipFile(zip_uncovered, "w") as zf:
        zf.writestr("paxg_manifest.json", manifest_str)
        zf.writestr("adapter_model.safetensors", safetensors_bytes)
        zf.writestr("adapter_config.json", config_bytes)
        zf.writestr("secret_notes.txt", b"some unverified notes")
        # checksums only mentions manifest, safetensors, config; secret_notes.txt is unverified!
        zf.writestr("checksums.sha256", base_checksums)

    with pytest.raises(ValueError, match="archive contains unverified files not covered by checksums"):
        temp_store.import_adapter_zip(zip_uncovered)


def test_queue_idempotency_enforcement(tmp_path: Path):
    """Verifies GPUJobStorage strictly enforces idempotency_key: two submissions yield 1 job ID."""
    from paxg_lab.queue.storage import GPUJobStorage
    from paxg_lab.queue.types import JobPriority, JobSpec, JobType

    db_path = tmp_path / "test_queue.db"
    storage = GPUJobStorage(db_path)

    key = "idem_unique_key_123"
    spec1 = JobSpec(
        job_id="job_idempotency_first",
        job_type=JobType.DUMMY.value,
        timeframe="1h",
        priority=JobPriority.MANUAL.value,
        payload={"session": "tab1"},
        idempotency_key=key,
    )
    spec2 = JobSpec(
        job_id="job_idempotency_duplicate",
        job_type=JobType.DUMMY.value,
        timeframe="1h",
        priority=JobPriority.MANUAL.value,
        payload={"session": "tab2"},
        idempotency_key=key,
    )

    id1 = storage.submit_job(spec1)
    id2 = storage.submit_job(spec2)

    assert id1 == "job_idempotency_first"
    assert id2 == "job_idempotency_first"
    assert id1 == id2

    # Verify only 1 job was created in DB
    jobs = storage.list_jobs()
    matching = [j for j in jobs if j.idempotency_key == key]
    assert len(matching) == 1
    assert matching[0].job_id == "job_idempotency_first"


def test_train_job_idempotency_deduplication(tmp_path: Path):
    """Verifies training jobs submit deterministic idempotency keys and deduplicate active runs."""
    import hashlib
    import json
    import hashlib
    import json
    from paxg_lab.model.train_spec import TrainSpec
    from paxg_lab.queue.storage import GPUJobStorage
    from paxg_lab.queue.types import JobPriority, JobSpec, JobType

    db_path = tmp_path / "queue_train.db"
    storage = GPUJobStorage(db_path)

    spec_dict = {
        "timeframe": "1h",
        "feature_set": "A",
        "lora_r": 8,
        "lora_alpha": 16,
        "batch_size": 2,
        "max_epochs": 10,
        "learning_rate": 1e-4,
    }
    train_spec = TrainSpec.from_dict(spec_dict)
    timeframe = "1h"
    snapshot_path = "snapshots/paxg_20260901.parquet"

    # Compute idempotency key as tab_training.py does
    train_hash_input = {
        "timeframe": timeframe,
        "snapshot": snapshot_path,
        "spec": train_spec.to_dict(),
    }
    train_hash = hashlib.sha256(json.dumps(train_hash_input, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    idempotency_key = f"train_{timeframe}_{train_hash}"

    job1 = JobSpec(
        job_id="train_job_1",
        job_type=JobType.TRAIN.value,
        timeframe=timeframe,
        priority=JobPriority.MANUAL.value,
        payload=train_hash_input,
        idempotency_key=idempotency_key,
    )
    job2 = JobSpec(
        job_id="train_job_2",
        job_type=JobType.TRAIN.value,
        timeframe=timeframe,
        priority=JobPriority.MANUAL.value,
        payload=train_hash_input,
        idempotency_key=idempotency_key,
    )

    submitted_id_1 = storage.submit_job(job1)
    submitted_id_2 = storage.submit_job(job2)

    assert submitted_id_1 == "train_job_1"
    assert submitted_id_2 == "train_job_1"  # Deduplicated to active job 1
    assert len(storage.list_jobs()) == 1


def test_backtest_job_idempotency_deterministic_payload_no_minute_bucket():
    """Verifies backtest idempotency keys are deterministic hashes of payload without minute buckets."""
    import hashlib
    import json

    def make_key(timeframe: str, adapter: str, payload: dict) -> str:
        payload_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        return f"bt_{timeframe}_{adapter}_{payload_hash}"

    p1 = {"capital": 10000, "threshold": 0.005}
    p2 = {"capital": 20000, "threshold": 0.005}

    k1 = make_key("1h", "lora_v1", p1)
    k1_again = make_key("1h", "lora_v1", p1)
    k2 = make_key("1h", "lora_v1", p2)

    assert k1 == k1_again
    assert k1 != k2
    # Ensure no minute-level timestamp bucket in key and format matches
    prefix = "bt_1h_lora_v1_"
    assert k1.startswith(prefix)
    hash_part = k1[len(prefix):]
    assert len(hash_part) == 16
    assert int(hash_part, 16) >= 0


def test_safe_zip_import_strict_allowlist_and_pickle_rejection(tmp_path: Path):
    """Verifies that zip files with non-allowlisted or dangerous pickle/bin extensions are rejected."""
    from paxg_lab.model.store import AdapterStore

    store = AdapterStore(tmp_path / "models")
    valid_manifest = {
        "adapter_id": "test_reject",
        "timeframe": "1h",
        "horizon": 24,
        "context_len": 512,
        "feature_set": "A",
        "feature_columns": ["close"],
        "base_model_repo": MODEL_REPO,
        "base_model_revision": MODEL_REVISION,
    }
    manifest_bytes = json.dumps(valid_manifest).encode("utf-8")
    m_hash = hashlib.sha256(manifest_bytes).hexdigest()

    # 1. Reject forbidden pickle / pt / bin extensions
    for bad_ext in [".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle"]:
        bad_zip = tmp_path / f"forbidden_{bad_ext.strip('.')}.zip"
        with zipfile.ZipFile(bad_zip, "w") as zf:
            zf.writestr("paxg_manifest.json", manifest_bytes)
            zf.writestr(f"weights{bad_ext}", b"bad data")
            zf.writestr("checksums.sha256", f"{m_hash}  paxg_manifest.json\n{hashlib.sha256(b'bad data').hexdigest()}  weights{bad_ext}\n")

        with pytest.raises(ValueError, match=r"(Disallowed or dangerous file extension|Pickle/PyTorch binary)"):
            store.import_adapter_zip(bad_zip)

    # 2. Reject non-allowlisted extensions like .py or .exe
    unallowed_zip = tmp_path / "unallowed_ext.zip"
    script_content = b"print('hello')"
    script_hash = hashlib.sha256(script_content).hexdigest()
    with zipfile.ZipFile(unallowed_zip, "w") as zf:
        zf.writestr("paxg_manifest.json", manifest_bytes)
        zf.writestr("script.py", script_content)
        zf.writestr("checksums.sha256", f"{m_hash}  paxg_manifest.json\n{script_hash}  script.py\n")

    with pytest.raises(ValueError, match=r"Disallowed or dangerous file extension '\.py'"):
        store.import_adapter_zip(unallowed_zip)


def test_safe_zip_import_uncompressed_size_limit_rejection(tmp_path: Path, monkeypatch):
    """Verifies that zip files exceeding uncompressed limits are rejected before extraction."""
    import paxg_lab.model.store as store_mod
    from paxg_lab.model.store import AdapterStore

    store = AdapterStore(tmp_path / "models")
    valid_manifest = {
        "adapter_id": "test_size",
        "timeframe": "1h",
        "horizon": 24,
        "context_len": 512,
        "feature_set": "A",
        "feature_columns": ["close"],
        "base_model_repo": MODEL_REPO,
        "base_model_revision": MODEL_REVISION,
    }
    manifest_bytes = json.dumps(valid_manifest).encode("utf-8")
    m_hash = hashlib.sha256(manifest_bytes).hexdigest()

    # Monkeypatch MAX_MEMBER_UNCOMPRESSED_BYTES to 100 bytes for fast test
    monkeypatch.setattr(store_mod, "MAX_MEMBER_UNCOMPRESSED_BYTES", 100)

    oversized_member_zip = tmp_path / "oversized_member.zip"
    with zipfile.ZipFile(oversized_member_zip, "w") as zf:
        zf.writestr("paxg_manifest.json", manifest_bytes)
        big_content = b"X" * 150
        zf.writestr("notes.txt", big_content)
        zf.writestr("checksums.sha256", f"{m_hash}  paxg_manifest.json\n{hashlib.sha256(big_content).hexdigest()}  notes.txt\n")

    with pytest.raises(ValueError, match="exceeds uncompressed size limit"):
        store.import_adapter_zip(oversized_member_zip)


def test_safe_zip_import_static_compatibility_validation(tmp_path: Path):
    """Verifies that manifests with incompatible parameters or missing mandatory files are rejected."""
    from paxg_lab.model.store import AdapterStore

    store = AdapterStore(tmp_path / "models")

    def make_zip(
        filename: str,
        manifest: dict,
        include_safetensors: bool = True,
        include_config: bool = True,
        custom_config: bytes | None = None,
    ) -> Path:
        p = tmp_path / filename
        m_bytes = json.dumps(manifest).encode("utf-8")
        s_bytes = VALID_SAFETENSORS_BYTES
        c_bytes = custom_config if custom_config is not None else json.dumps(VALID_LORA_CONFIG).encode("utf-8")

        lines = [f"{hashlib.sha256(m_bytes).hexdigest()}  paxg_manifest.json"]
        if include_safetensors:
            lines.append(f"{hashlib.sha256(s_bytes).hexdigest()}  adapter_model.safetensors")
        if include_config:
            lines.append(f"{hashlib.sha256(c_bytes).hexdigest()}  adapter_config.json")
        checksums_content = "\n".join(lines) + "\n"

        with zipfile.ZipFile(p, "w") as zf:
            zf.writestr("paxg_manifest.json", m_bytes)
            if include_safetensors:
                zf.writestr("adapter_model.safetensors", s_bytes)
            if include_config:
                zf.writestr("adapter_config.json", c_bytes)
            zf.writestr("checksums.sha256", checksums_content)
        return p

    def_manifest = {
        "adapter_id": "test_compat",
        "timeframe": "1h",
        "horizon": 24,
        "context_len": 512,
        "feature_set": "A",
        "feature_columns": ["close"],
        "base_model_repo": MODEL_REPO,
        "base_model_revision": MODEL_REVISION,
    }

    # 1. Invalid timeframe
    bad_tf = make_zip("bad_tf.zip", {**def_manifest, "timeframe": "15m"})
    with pytest.raises(ValueError, match=r"(Incompatible adapter timeframe|Unsupported timeframe).*15m"):
        store.import_adapter_zip(bad_tf)

    # 2. Mismatched horizon
    bad_horizon = make_zip("bad_hz.zip", {**def_manifest, "horizon": 6})
    with pytest.raises(ValueError, match=r"(Incompatible adapter horizon|AdapterManifest horizon mismatch)"):
        store.import_adapter_zip(bad_horizon)

    # 3. Invalid context_len
    bad_ctx = make_zip("bad_ctx.zip", {**def_manifest, "context_len": 999})
    with pytest.raises(ValueError, match="Incompatible adapter context_len 999"):
        store.import_adapter_zip(bad_ctx)

    # 4. Invalid feature set
    bad_feat = make_zip("bad_feat.zip", {**def_manifest, "timeframe": "4h", "horizon": 6, "feature_set": "D"})
    with pytest.raises(ValueError, match="Incompatible adapter feature_set 'D'"):
        store.import_adapter_zip(bad_feat)

    # 5. Feature columns mismatch with canonical FeatureSpec
    bad_feat_cols = make_zip("bad_feat_cols.zip", {**def_manifest, "feature_set": "B", "feature_columns": ["close"]})
    with pytest.raises(ValueError, match="Incompatible feature_columns for feature_set 'B'"):
        store.import_adapter_zip(bad_feat_cols)

    # 6. Feature columns wrong order
    b_cols_reversed = list(reversed(FEATURE_SPECS["B"].columns))
    bad_cols_order = make_zip("bad_cols_order.zip", {**def_manifest, "feature_set": "B", "feature_columns": b_cols_reversed})
    with pytest.raises(ValueError, match="Incompatible feature_columns for feature_set 'B'"):
        store.import_adapter_zip(bad_cols_order)

    # 7. Incompatible base_model_repo (TimesFM 2.0 must be strictly rejected)
    bad_base_2 = make_zip("bad_base_2.zip", {**def_manifest, "base_model_repo": "google/timesfm-2.0-500m-pytorch"})
    with pytest.raises(ValueError, match="Incompatible base_model_repo 'google/timesfm-2.0-500m-pytorch'"):
        store.import_adapter_zip(bad_base_2)

    # 8. Non-TimesFM base model repo
    bad_base_bert = make_zip("bad_base_bert.zip", {**def_manifest, "base_model_repo": "bert-base-uncased"})
    with pytest.raises(ValueError, match="Incompatible base_model_repo 'bert-base-uncased'"):
        store.import_adapter_zip(bad_base_bert)

    # 9. Missing base_model_repo provenance
    no_repo_manifest = {k: v for k, v in def_manifest.items() if k != "base_model_repo"}
    bad_no_repo = make_zip("bad_no_repo.zip", no_repo_manifest)
    with pytest.raises(ValueError, match="missing mandatory 'base_model_repo' provenance"):
        store.import_adapter_zip(bad_no_repo)

    # 10. Missing base_model_revision provenance
    no_rev_manifest = {k: v for k, v in def_manifest.items() if k != "base_model_revision"}
    bad_no_rev = make_zip("bad_no_rev.zip", no_rev_manifest)
    with pytest.raises(ValueError, match="missing mandatory 'base_model_revision' provenance"):
        store.import_adapter_zip(bad_no_rev)

    # 11. Incompatible base_model_revision
    bad_rev = make_zip("bad_rev.zip", {**def_manifest, "base_model_revision": "invalid_commit_hash_12345"})
    with pytest.raises(ValueError, match="Incompatible base_model_revision 'invalid_commit_hash_12345'"):
        store.import_adapter_zip(bad_rev)

    # 12. Missing adapter_model.safetensors (empty adapter prevention)
    missing_weights = make_zip("missing_weights.zip", def_manifest, include_safetensors=False)
    with pytest.raises(ValueError, match="required adapter file 'adapter_model.safetensors' missing or empty"):
        store.import_adapter_zip(missing_weights)

    # 13. Missing adapter_config.json
    missing_cfg = make_zip("missing_cfg.zip", def_manifest, include_config=False)
    with pytest.raises(ValueError, match="required adapter file 'adapter_config.json' missing or empty"):
        store.import_adapter_zip(missing_cfg)

    # 14. Corrupted adapter_config.json
    bad_cfg = make_zip("bad_cfg.zip", def_manifest, custom_config=b"NOT_A_VALID_JSON{")
    with pytest.raises(ValueError, match="Corrupted or invalid 'adapter_config.json'"):
        store.import_adapter_zip(bad_cfg)


def test_safe_zip_import_minimal_valid_package_registers_only_after_success(tmp_path: Path):
    """Verifies that a valid minimal package imports cleanly and updates registry only after verification."""
    from paxg_lab.model.store import AdapterStore

    models_dir = tmp_path / "models"
    db_path = tmp_path / "test_paxg.db"
    store = AdapterStore(base_dir=models_dir, db_path=db_path)

    adapter_id = "paxg_1h_minimal_valid_pkg"
    valid_manifest = {
        "adapter_id": adapter_id,
        "timeframe": "1h",
        "horizon": 24,
        "context_len": 256,
        "feature_set": "A",
        "feature_columns": ["close"],
        "base_model_repo": MODEL_REPO,
        "base_model_revision": MODEL_REVISION,
        "best_val_loss": 0.015,
    }
    m_bytes = json.dumps(valid_manifest).encode("utf-8")
    s_bytes = VALID_SAFETENSORS_BYTES
    c_bytes = json.dumps(VALID_LORA_CONFIG).encode("utf-8")

    m_hash = hashlib.sha256(m_bytes).hexdigest()
    s_hash = hashlib.sha256(s_bytes).hexdigest()
    c_hash = hashlib.sha256(c_bytes).hexdigest()

    checksums_content = (
        f"{m_hash}  paxg_manifest.json\n"
        f"{s_hash}  adapter_model.safetensors\n"
        f"{c_hash}  adapter_config.json\n"
    )

    pkg_zip = tmp_path / "valid_minimal_adapter.zip"
    with zipfile.ZipFile(pkg_zip, "w") as zf:
        zf.writestr("paxg_manifest.json", m_bytes)
        zf.writestr("adapter_model.safetensors", s_bytes)
        zf.writestr("adapter_config.json", c_bytes)
        zf.writestr("checksums.sha256", checksums_content)

    # 1. Before import: registry table does not have the adapter record
    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.execute("SELECT 1 FROM adapter_registry WHERE adapter_id = ?;", (adapter_id,))
        assert cur.fetchone() is None

    # 2. Perform import
    imported_id = store.import_adapter_zip(pkg_zip)
    assert imported_id == adapter_id

    # Verify row was created in DB
    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.execute("SELECT 1 FROM adapter_registry WHERE adapter_id = ?;", (adapter_id,))
        assert cur.fetchone() is not None

    # 3. After import: canonical directory exists with all verified files
    target_dir = store.get_adapter_path(adapter_id)
    assert target_dir.is_dir()
    assert (target_dir / "adapter_model.safetensors").exists()
    assert (target_dir / "adapter_config.json").exists()
    assert (target_dir / "paxg_manifest.json").exists()
    assert (target_dir / "checksums.sha256").exists()

    # 4. Registry metadata populated only after validation completed
    meta_after = store.get_registry_metadata(adapter_id)
    assert meta_after["adapter_id"] == adapter_id
    assert meta_after["alias"] == adapter_id

    # 5. Manifest load matches imported data
    manifest = AdapterManifest.load_json(target_dir / "paxg_manifest.json")
    assert manifest.adapter_id == adapter_id
    assert manifest.base_model_repo == MODEL_REPO
    assert manifest.base_model_revision == MODEL_REVISION
    assert manifest.feature_columns == ["close"]
    assert any(a.adapter_id == adapter_id for a in store.list_adapters())


def test_safe_zip_import_overwrite_protection_pinned_and_recommended(tmp_path: Path):
    """Verifies that import_adapter_zip(overwrite=True) strictly refuses to overwrite
    pinned or recommended (winner) adapters, keeping existing files intact."""
    models_dir = tmp_path / "models"
    db_path = tmp_path / "test_paxg.db"
    store = AdapterStore(base_dir=models_dir, db_path=db_path)

    aid = "paxg_1h_protect_test"
    target_dir = store.get_adapter_path(aid)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Write initial adapter files with a marker file
    (target_dir / "adapter_model.safetensors").write_bytes(VALID_SAFETENSORS_BYTES)
    (target_dir / "adapter_config.json").write_text(json.dumps(VALID_LORA_CONFIG), encoding="utf-8")
    orig_manifest = {
        "adapter_id": aid,
        "timeframe": "1h",
        "horizon": 24,
        "context_len": 256,
        "feature_set": "A",
        "feature_columns": ["close"],
        "base_model_repo": MODEL_REPO,
        "base_model_revision": MODEL_REVISION,
        "best_val_loss": 0.05,
    }
    (target_dir / "paxg_manifest.json").write_text(json.dumps(orig_manifest), encoding="utf-8")
    marker_file = target_dir / "original_marker.txt"
    marker_file.write_text("ORIGINAL_CONTENT", encoding="utf-8")

    # Generate checksums for initial directory
    c_lines = []
    for f in sorted(target_dir.iterdir()):
        if f.is_file():
            c_lines.append(f"{compute_file_sha256(f)}  {f.name}")
    (target_dir / "checksums.sha256").write_text("\n".join(c_lines) + "\n", encoding="utf-8")

    # Prepare a new replacement zip for the same adapter_id
    new_manifest = {**orig_manifest, "best_val_loss": 0.001}
    m_bytes = json.dumps(new_manifest).encode("utf-8")
    s_bytes = VALID_SAFETENSORS_BYTES
    c_bytes = json.dumps(VALID_LORA_CONFIG).encode("utf-8")
    chk_str = (
        f"{hashlib.sha256(m_bytes).hexdigest()}  paxg_manifest.json\n"
        f"{hashlib.sha256(s_bytes).hexdigest()}  adapter_model.safetensors\n"
        f"{hashlib.sha256(c_bytes).hexdigest()}  adapter_config.json\n"
    )
    new_zip = tmp_path / "replacement.zip"
    with zipfile.ZipFile(new_zip, "w") as zf:
        zf.writestr("paxg_manifest.json", m_bytes)
        zf.writestr("adapter_model.safetensors", s_bytes)
        zf.writestr("adapter_config.json", c_bytes)
        zf.writestr("checksums.sha256", chk_str)

    # 1. Pinned protection: overwrite=True MUST be rejected and original preserved
    store.set_pinned(aid, True)
    assert store.is_pinned(aid) is True

    with pytest.raises(ValueError, match="đã được ghim chống xóa"):
        store.import_adapter_zip(new_zip, overwrite=True)

    # Original files must still exist and be untouched
    assert marker_file.exists()
    assert marker_file.read_text(encoding="utf-8") == "ORIGINAL_CONTENT"
    cur_m = json.loads((target_dir / "paxg_manifest.json").read_text(encoding="utf-8"))
    assert cur_m["best_val_loss"] == 0.05

    # 2. Recommended protection: overwrite=True MUST be rejected and original preserved
    store.set_pinned(aid, False)
    store.set_recommended(aid, "1h")
    assert store.get_recommended("1h") == aid

    with pytest.raises(ValueError, match="adapter khuyến nghị"):
        store.import_adapter_zip(new_zip, overwrite=True)

    assert marker_file.exists()
    assert marker_file.read_text(encoding="utf-8") == "ORIGINAL_CONTENT"

    # 3. Unpinned & unrecommended with overwrite=False -> FileExistsError
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("UPDATE adapter_registry SET is_recommended = 0, is_pinned = 0 WHERE adapter_id = ?;", (aid,))
        conn.commit()

    with pytest.raises(FileExistsError, match="already exists"):
        store.import_adapter_zip(new_zip, overwrite=False)
    assert marker_file.exists()

    # 4. Unpinned & unrecommended with overwrite=True -> succeeds and atomically replaces
    res_id = store.import_adapter_zip(new_zip, overwrite=True)
    assert res_id == aid
    assert not marker_file.exists()
    updated_m = json.loads((target_dir / "paxg_manifest.json").read_text(encoding="utf-8"))
    assert updated_m["best_val_loss"] == 0.001


def test_safe_zip_import_rejects_corrupted_or_fake_safetensors(tmp_path: Path):
    """Verifies that non-safetensors or corrupted safetensors files are rejected by safetensors.safe_open,
    and neither canonical directory nor registry entry is created."""
    models_dir = tmp_path / "models"
    db_path = tmp_path / "test_paxg.db"
    store = AdapterStore(base_dir=models_dir, db_path=db_path)

    base_manifest = {
        "adapter_id": "paxg_1h_corrupt_test",
        "timeframe": "1h",
        "horizon": 24,
        "context_len": 256,
        "feature_set": "A",
        "feature_columns": ["close"],
        "base_model_repo": MODEL_REPO,
        "base_model_revision": MODEL_REVISION,
        "best_val_loss": 0.01,
    }
    m_bytes = json.dumps(base_manifest).encode("utf-8")
    c_bytes = json.dumps(VALID_LORA_CONFIG).encode("utf-8")

    # Case A: Random corrupted bytes with .safetensors extension
    fake_safetensors_bytes = b"RANDOM_CORRUPTED_FAKE_SAFETENSORS_HEADER_BYTES"
    chk_str = (
        f"{hashlib.sha256(m_bytes).hexdigest()}  paxg_manifest.json\n"
        f"{hashlib.sha256(fake_safetensors_bytes).hexdigest()}  adapter_model.safetensors\n"
        f"{hashlib.sha256(c_bytes).hexdigest()}  adapter_config.json\n"
    )
    zip_fake = tmp_path / "fake_safetensors.zip"
    with zipfile.ZipFile(zip_fake, "w") as zf:
        zf.writestr("paxg_manifest.json", m_bytes)
        zf.writestr("adapter_model.safetensors", fake_safetensors_bytes)
        zf.writestr("adapter_config.json", c_bytes)
        zf.writestr("checksums.sha256", chk_str)

    with pytest.raises(ValueError, match="Corrupted or invalid safetensors weights file"):
        store.import_adapter_zip(zip_fake)

    # Assert neither folder nor DB entry was created
    assert not store.get_adapter_path(base_manifest["adapter_id"]).exists()
    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.execute("SELECT 1 FROM adapter_registry WHERE adapter_id = ?;", (base_manifest["adapter_id"],))
        assert cur.fetchone() is None

    # Case B: Valid safetensors container format but empty keys
    empty_safetensors_bytes = safetensors.torch.save({})
    chk_empty = (
        f"{hashlib.sha256(m_bytes).hexdigest()}  paxg_manifest.json\n"
        f"{hashlib.sha256(empty_safetensors_bytes).hexdigest()}  adapter_model.safetensors\n"
        f"{hashlib.sha256(c_bytes).hexdigest()}  adapter_config.json\n"
    )
    zip_empty = tmp_path / "empty_keys_safetensors.zip"
    with zipfile.ZipFile(zip_empty, "w") as zf:
        zf.writestr("paxg_manifest.json", m_bytes)
        zf.writestr("adapter_model.safetensors", empty_safetensors_bytes)
        zf.writestr("adapter_config.json", c_bytes)
        zf.writestr("checksums.sha256", chk_empty)

    with pytest.raises(ValueError, match="contains no tensor keys"):
        store.import_adapter_zip(zip_empty)

    assert not store.get_adapter_path(base_manifest["adapter_id"]).exists()
    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.execute("SELECT 1 FROM adapter_registry WHERE adapter_id = ?;", (base_manifest["adapter_id"],))
        assert cur.fetchone() is None


def test_safe_zip_import_rejects_invalid_lora_config(tmp_path: Path):
    """Verifies that non-LoRA PEFT configs or invalid rank configs are rejected,
    without registering in database."""
    models_dir = tmp_path / "models"
    db_path = tmp_path / "test_paxg.db"
    store = AdapterStore(base_dir=models_dir, db_path=db_path)

    manifest = {
        "adapter_id": "paxg_1h_cfg_test",
        "timeframe": "1h",
        "horizon": 24,
        "context_len": 256,
        "feature_set": "A",
        "feature_columns": ["close"],
        "base_model_repo": MODEL_REPO,
        "base_model_revision": MODEL_REVISION,
        "best_val_loss": 0.01,
    }
    m_bytes = json.dumps(manifest).encode("utf-8")
    s_bytes = VALID_SAFETENSORS_BYTES

    # Case A: peft_type is not LORA
    bad_type_cfg = json.dumps({"peft_type": "PREFIX_TUNING", "r": 4}).encode("utf-8")
    chk_bad_type = (
        f"{hashlib.sha256(m_bytes).hexdigest()}  paxg_manifest.json\n"
        f"{hashlib.sha256(s_bytes).hexdigest()}  adapter_model.safetensors\n"
        f"{hashlib.sha256(bad_type_cfg).hexdigest()}  adapter_config.json\n"
    )
    zip_bad_type = tmp_path / "bad_peft_type.zip"
    with zipfile.ZipFile(zip_bad_type, "w") as zf:
        zf.writestr("paxg_manifest.json", m_bytes)
        zf.writestr("adapter_model.safetensors", s_bytes)
        zf.writestr("adapter_config.json", bad_type_cfg)
        zf.writestr("checksums.sha256", chk_bad_type)

    with pytest.raises(ValueError, match=r"'peft_type' must be 'LORA'"):
        store.import_adapter_zip(zip_bad_type)
    assert not store.get_adapter_path(manifest["adapter_id"]).exists()
    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.execute("SELECT 1 FROM adapter_registry WHERE adapter_id = ?;", (manifest["adapter_id"],))
        assert cur.fetchone() is None

    # Case B: rank r is missing or non-positive
    bad_r_cfg = json.dumps({"peft_type": "LORA", "r": 0}).encode("utf-8")
    chk_bad_r = (
        f"{hashlib.sha256(m_bytes).hexdigest()}  paxg_manifest.json\n"
        f"{hashlib.sha256(s_bytes).hexdigest()}  adapter_model.safetensors\n"
        f"{hashlib.sha256(bad_r_cfg).hexdigest()}  adapter_config.json\n"
    )
    zip_bad_r = tmp_path / "bad_rank.zip"
    with zipfile.ZipFile(zip_bad_r, "w") as zf:
        zf.writestr("paxg_manifest.json", m_bytes)
        zf.writestr("adapter_model.safetensors", s_bytes)
        zf.writestr("adapter_config.json", bad_r_cfg)
        zf.writestr("checksums.sha256", chk_bad_r)

    with pytest.raises(ValueError, match=r"'r' \(rank\) must be a positive integer"):
        store.import_adapter_zip(zip_bad_r)
    assert not store.get_adapter_path(manifest["adapter_id"]).exists()
    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.execute("SELECT 1 FROM adapter_registry WHERE adapter_id = ?;", (manifest["adapter_id"],))
        assert cur.fetchone() is None


def test_safe_zip_import_rejects_semantic_lora_weights_mismatches(tmp_path: Path):
    """Verifies that safetensors weights with rank mismatch, missing lora_B, or wrong target
    modules are rejected by semantic validation, and a full valid TimesFM3 adapter loads cleanly."""
    models_dir = tmp_path / "models"
    db_path = tmp_path / "test_paxg.db"
    store = AdapterStore(base_dir=models_dir, db_path=db_path)

    base_manifest = {
        "adapter_id": "paxg_1h_semantic_test",
        "timeframe": "1h",
        "horizon": 24,
        "context_len": 256,
        "feature_set": "A",
        "feature_columns": ["close"],
        "base_model_repo": MODEL_REPO,
        "base_model_revision": MODEL_REVISION,
        "best_val_loss": 0.01,
    }
    m_bytes = json.dumps(base_manifest).encode("utf-8")
    c_bytes = json.dumps(VALID_LORA_CONFIG).encode("utf-8")

    def make_custom_zip(zip_name: str, tensor_dict: dict) -> Path:
        s_bytes = safetensors.torch.save(tensor_dict)
        chk = (
            f"{hashlib.sha256(m_bytes).hexdigest()}  paxg_manifest.json\n"
            f"{hashlib.sha256(s_bytes).hexdigest()}  adapter_model.safetensors\n"
            f"{hashlib.sha256(c_bytes).hexdigest()}  adapter_config.json\n"
        )
        zpath = tmp_path / zip_name
        with zipfile.ZipFile(zpath, "w") as zf:
            zf.writestr("paxg_manifest.json", m_bytes)
            zf.writestr("adapter_model.safetensors", s_bytes)
            zf.writestr("adapter_config.json", c_bytes)
            zf.writestr("checksums.sha256", chk)
        return zpath

    def assert_not_registered(adapter_id: str):
        assert not store.get_adapter_path(adapter_id).exists()
        with sqlite3.connect(str(db_path)) as conn:
            cur = conn.execute("SELECT 1 FROM adapter_registry WHERE adapter_id = ?;", (adapter_id,))
            assert cur.fetchone() is None

    # 1. Incompatible in_dim/out_dim: shapes (4, 1) and (1, 4) instead of (4, 1280) and (1280, 4)
    bad_dims_tensors = {}
    for l_idx in range(20):
        for attn in ("seq_attn", "var_attn"):
            for tm in ("query_proj", "value_proj"):
                bad_dims_tensors[f"base_model.model.transformer_stack.layers.{l_idx}.{attn}.{tm}.lora_A.weight"] = torch.zeros(4, 1)
                bad_dims_tensors[f"base_model.model.transformer_stack.layers.{l_idx}.{attn}.{tm}.lora_B.weight"] = torch.zeros(1, 4)
    zip_bad_dims = make_custom_zip("bad_dims_tensors.zip", bad_dims_tensors)
    with pytest.raises(ValueError, match="Dimension mismatch|compatibility"):
        store.import_adapter_zip(zip_bad_dims)
    assert_not_registered(base_manifest["adapter_id"])

    # 2. Rank mismatch: tensor has rank 2 but config expects rank 4
    bad_rank_tensors = {}
    for l_idx in range(20):
        for attn in ("seq_attn", "var_attn"):
            for tm in ("query_proj", "value_proj"):
                bad_rank_tensors[f"base_model.model.transformer_stack.layers.{l_idx}.{attn}.{tm}.lora_A.weight"] = torch.zeros(2, 1280)
                bad_rank_tensors[f"base_model.model.transformer_stack.layers.{l_idx}.{attn}.{tm}.lora_B.weight"] = torch.zeros(1280, 4)
    zip_bad_rank = make_custom_zip("bad_rank_tensors.zip", bad_rank_tensors)
    with pytest.raises(ValueError, match="Dimension mismatch|Rank mismatch|compatibility"):
        store.import_adapter_zip(zip_bad_rank)
    assert_not_registered(base_manifest["adapter_id"])

    # 3. Missing lora_B for value_proj
    missing_b_tensors = {}
    for l_idx in range(20):
        for attn in ("seq_attn", "var_attn"):
            missing_b_tensors[f"base_model.model.transformer_stack.layers.{l_idx}.{attn}.query_proj.lora_A.weight"] = torch.zeros(4, 1280)
            missing_b_tensors[f"base_model.model.transformer_stack.layers.{l_idx}.{attn}.query_proj.lora_B.weight"] = torch.zeros(1280, 4)
            missing_b_tensors[f"base_model.model.transformer_stack.layers.{l_idx}.{attn}.value_proj.lora_A.weight"] = torch.zeros(4, 1280)
    zip_missing_b = make_custom_zip("missing_b_tensors.zip", missing_b_tensors)
    with pytest.raises(ValueError, match="Missing matching lora_B tensor"):
        store.import_adapter_zip(zip_missing_b)
    assert_not_registered(base_manifest["adapter_id"])

    # 4. Missing canonical layers (only layer 0 provided out of 20)
    missing_layers_tensors = {
        "base_model.model.transformer_stack.layers.0.seq_attn.query_proj.lora_A.weight": torch.zeros(4, 1280),
        "base_model.model.transformer_stack.layers.0.seq_attn.query_proj.lora_B.weight": torch.zeros(1280, 4),
        "base_model.model.transformer_stack.layers.0.seq_attn.value_proj.lora_A.weight": torch.zeros(4, 1280),
        "base_model.model.transformer_stack.layers.0.seq_attn.value_proj.lora_B.weight": torch.zeros(1280, 4),
        "base_model.model.transformer_stack.layers.0.var_attn.query_proj.lora_A.weight": torch.zeros(4, 1280),
        "base_model.model.transformer_stack.layers.0.var_attn.query_proj.lora_B.weight": torch.zeros(1280, 4),
        "base_model.model.transformer_stack.layers.0.var_attn.value_proj.lora_A.weight": torch.zeros(4, 1280),
        "base_model.model.transformer_stack.layers.0.var_attn.value_proj.lora_B.weight": torch.zeros(1280, 4),
    }
    zip_missing_layers = make_custom_zip("missing_layers.zip", missing_layers_tensors)
    with pytest.raises(ValueError, match="Missing canonical LoRA weights|missing declared adapter keys"):
        store.import_adapter_zip(zip_missing_layers)
    assert_not_registered(base_manifest["adapter_id"])

    # 5. Wrong target module (unconfigured module dense_h_to_4h)
    wrong_mod_tensors = {}
    for l_idx in range(20):
        for attn in ("seq_attn", "var_attn"):
            wrong_mod_tensors[f"base_model.model.transformer_stack.layers.{l_idx}.{attn}.dense_h_to_4h.lora_A.weight"] = torch.zeros(4, 1280)
            wrong_mod_tensors[f"base_model.model.transformer_stack.layers.{l_idx}.{attn}.dense_h_to_4h.lora_B.weight"] = torch.zeros(5120, 4)
    zip_wrong_mod = make_custom_zip("wrong_mod_tensors.zip", wrong_mod_tensors)
    with pytest.raises(ValueError, match="which does not match any configured target_modules|does not conform"):
        store.import_adapter_zip(zip_wrong_mod)
    assert_not_registered(base_manifest["adapter_id"])

    # 6. Positive case: Real PEFT package generated from TimesFM3Torch goes through EXACT Web import path (base_model=None)
    from timesfm import TimesFM3Torch
    from paxg_lab.model.lora import build_lora_timesfm3, load_lora_adapter

    base_model = TimesFM3Torch()
    real_peft_model = build_lora_timesfm3(base_model, lora_r=4, lora_alpha=8)

    real_adapter_id = "paxg_1h_real_web_import"
    real_manifest_data = {**base_manifest, "adapter_id": real_adapter_id}
    real_m_bytes = json.dumps(real_manifest_data).encode("utf-8")

    with tempfile.TemporaryDirectory() as td_save:
        real_peft_model.save_pretrained(td_save)
        real_s_bytes = (Path(td_save) / "adapter_model.safetensors").read_bytes()
        real_c_bytes = (Path(td_save) / "adapter_config.json").read_bytes()

        real_chk = (
            f"{hashlib.sha256(real_m_bytes).hexdigest()}  paxg_manifest.json\n"
            f"{hashlib.sha256(real_s_bytes).hexdigest()}  adapter_model.safetensors\n"
            f"{hashlib.sha256(real_c_bytes).hexdigest()}  adapter_config.json\n"
        )
        real_zip = tmp_path / "real_peft_web_package.zip"
        with zipfile.ZipFile(real_zip, "w") as zf:
            zf.writestr("paxg_manifest.json", real_m_bytes)
            zf.writestr("adapter_model.safetensors", real_s_bytes)
            zf.writestr("adapter_config.json", real_c_bytes)
            zf.writestr("checksums.sha256", real_chk)

    # Import via EXACT DEFAULT Web path (base_model=None)
    imported_id = store.import_adapter_zip(real_zip)
    assert imported_id == real_adapter_id

    # Verified that registry row is created ONLY after compatibility validation succeeded
    reg_meta = store.get_registry_metadata(real_adapter_id)
    assert reg_meta.get("adapter_id") == real_adapter_id

    # Verify adapter loads cleanly into fresh TimesFM3 model without warning or error
    fresh_base = TimesFM3Torch()
    loaded_adapter = load_lora_adapter(fresh_base, store.get_adapter_path(imported_id))
    assert loaded_adapter is not None


def test_safe_zip_import_overwrite_rollback_on_sqlite_failure(tmp_path: Path, monkeypatch):
    """Verifies that if SQLite registry update fails during overwrite=True, the original
    adapter directory and metadata are 100% restored, and no temporary directories remain."""
    models_dir = tmp_path / "models"
    db_path = tmp_path / "test_paxg.db"
    store = AdapterStore(base_dir=models_dir, db_path=db_path)

    aid = "paxg_1h_rollback_test"
    target_dir = store.get_adapter_path(aid)
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. Setup original adapter with marker and alias
    (target_dir / "adapter_model.safetensors").write_bytes(VALID_SAFETENSORS_BYTES)
    (target_dir / "adapter_config.json").write_text(json.dumps(VALID_LORA_CONFIG), encoding="utf-8")
    orig_manifest = {
        "adapter_id": aid,
        "timeframe": "1h",
        "horizon": 24,
        "context_len": 256,
        "feature_set": "A",
        "feature_columns": ["close"],
        "base_model_repo": MODEL_REPO,
        "base_model_revision": MODEL_REVISION,
        "best_val_loss": 0.05,
    }
    (target_dir / "paxg_manifest.json").write_text(json.dumps(orig_manifest), encoding="utf-8")
    marker_file = target_dir / "original_marker.txt"
    marker_file.write_text("ORIGINAL_MARKER_CONTENT", encoding="utf-8")

    c_lines = []
    for f in sorted(target_dir.iterdir()):
        if f.is_file():
            c_lines.append(f"{compute_file_sha256(f)}  {f.name}")
    (target_dir / "checksums.sha256").write_text("\n".join(c_lines) + "\n", encoding="utf-8")

    initial_alias = "Mô hình Gốc Không Đổi"
    store.set_alias(aid, initial_alias)
    assert store.get_alias(aid) == initial_alias

    # 2. Prepare replacement zip for same aid with new loss
    new_manifest = {**orig_manifest, "best_val_loss": 0.0001}
    m_bytes = json.dumps(new_manifest).encode("utf-8")
    s_bytes = VALID_SAFETENSORS_BYTES
    c_bytes = json.dumps(VALID_LORA_CONFIG).encode("utf-8")
    chk_str = (
        f"{hashlib.sha256(m_bytes).hexdigest()}  paxg_manifest.json\n"
        f"{hashlib.sha256(s_bytes).hexdigest()}  adapter_model.safetensors\n"
        f"{hashlib.sha256(c_bytes).hexdigest()}  adapter_config.json\n"
    )
    replacement_zip = tmp_path / "rollback_replacement.zip"
    with zipfile.ZipFile(replacement_zip, "w") as zf:
        zf.writestr("paxg_manifest.json", m_bytes)
        zf.writestr("adapter_model.safetensors", s_bytes)
        zf.writestr("adapter_config.json", c_bytes)
        zf.writestr("checksums.sha256", chk_str)

    # 3. Monkeypatch set_alias to simulate database failure (DB locked / disk error)
    def failing_set_alias(adapter_id: str, alias: str):
        raise sqlite3.OperationalError("Simulated DB error: database is locked")

    monkeypatch.setattr(store, "set_alias", failing_set_alias)

    # 4. Attempt overwrite import -> must raise OperationalError
    with pytest.raises(sqlite3.OperationalError, match="Simulated DB error"):
        store.import_adapter_zip(replacement_zip, overwrite=True)

    # 5. Assert atomic rollback: original files and marker are completely restored!
    assert target_dir.exists()
    assert marker_file.exists()
    assert marker_file.read_text(encoding="utf-8") == "ORIGINAL_MARKER_CONTENT"

    restored_manifest = json.loads((target_dir / "paxg_manifest.json").read_text(encoding="utf-8"))
    assert restored_manifest["best_val_loss"] == 0.05

    # 6. Assert no temporary backup or import directories were left behind
    backup_dirs = [d for d in models_dir.iterdir() if d.name.startswith(".bak_replace_")]
    assert len(backup_dirs) == 0, f"Lingering backup dirs: {backup_dirs}"

    tmp_import_dirs = [d for d in models_dir.iterdir() if d.name.startswith(".tmp_import_")]
    assert len(tmp_import_dirs) == 0, f"Lingering tmp_import dirs: {tmp_import_dirs}"

    # 7. Assert registry metadata remains unchanged
    assert store.get_alias(aid) == initial_alias







