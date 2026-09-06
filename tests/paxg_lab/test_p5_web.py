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

    for idx, bad_id in enumerate(malicious_ids):
        bad_zip = tmp_path / f"bad_manifest_id_{idx}.zip"
        manifest_data = {
            "adapter_id": bad_id,
            "timeframe": "1h",
            "horizon": 24,
            "context_len": 256,
            "feature_set": "B",
            "feature_columns": ["close"],
            "best_val_loss": 0.012,
        }
        manifest_str = json.dumps(manifest_data)
        import hashlib
        m_hash = hashlib.sha256(manifest_str.encode("utf-8")).hexdigest()
        checksums_content = f"{m_hash}  paxg_manifest.json\n"

        with zipfile.ZipFile(bad_zip, "w") as zf:
            zf.writestr("paxg_manifest.json", manifest_str)
            zf.writestr("checksums.sha256", checksums_content)

        with pytest.raises(ValueError):
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

    # 1. Traversal path in checksums.sha256 (../outside)
    zip_traversal = tmp_path / "bad_checksum_traversal.zip"
    manifest_str = json.dumps({"adapter_id": "test_traversal", "timeframe": "1h", "horizon": 24, "context_len": 256, "feature_set": "B", "feature_columns": ["close"], "best_val_loss": 0.01})
    m_hash = hashlib.sha256(manifest_str.encode("utf-8")).hexdigest()

    with zipfile.ZipFile(zip_traversal, "w") as zf:
        zf.writestr("paxg_manifest.json", manifest_str)
        zf.writestr("checksums.sha256", f"{m_hash}  paxg_manifest.json\n1234567890abcdef  ../outside_file.txt\n")

    with pytest.raises(ValueError, match="Path traversal detected in checksums filename"):
        temp_store.import_adapter_zip(zip_traversal)

    # 2. Absolute / Drive path in checksums.sha256 (C:\Windows\...)
    zip_drive = tmp_path / "bad_checksum_drive.zip"
    with zipfile.ZipFile(zip_drive, "w") as zf:
        zf.writestr("paxg_manifest.json", manifest_str)
        zf.writestr("checksums.sha256", f"{m_hash}  paxg_manifest.json\n1234567890abcdef  C:\\Windows\\system.ini\n")

    with pytest.raises(ValueError, match="Path traversal detected in checksums filename"):
        temp_store.import_adapter_zip(zip_drive)

    # 3. UNC path in checksums.sha256 (//server/share or \\\\server\\share)
    zip_unc = tmp_path / "bad_checksum_unc.zip"
    with zipfile.ZipFile(zip_unc, "w") as zf:
        zf.writestr("paxg_manifest.json", manifest_str)
        zf.writestr("checksums.sha256", f"{m_hash}  paxg_manifest.json\n1234567890abcdef  \\\\server\\share\\evil.txt\n")

    with pytest.raises(ValueError, match="Path traversal detected in checksums filename"):
        temp_store.import_adapter_zip(zip_unc)

    # 4. Uncovered file in archive: archive contains extra file not in checksums.sha256
    zip_uncovered = tmp_path / "uncovered_file.zip"
    with zipfile.ZipFile(zip_uncovered, "w") as zf:
        zf.writestr("paxg_manifest.json", manifest_str)
        zf.writestr("secret_backdoor.bin", b"\x00\x01\x02")
        # checksums only mentions paxg_manifest.json, secret_backdoor.bin is unverified!
        zf.writestr("checksums.sha256", f"{m_hash}  paxg_manifest.json\n")

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


