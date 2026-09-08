from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from paxg_lab.data.snapshot import DatasetSnapshot
from paxg_lab.data.split import eligible_origins_from_timestamps, extract_windows
from paxg_lab.model.manifest import AdapterManifest
from paxg_lab.queue.scheduler import GPUScheduler
from paxg_lab.queue.storage import GPUJobStorage
from paxg_lab.queue.types import AutoRunState
from paxg_lab.tune.locked_eval import run_locked_verification


def _timestamps(size: int, timeframe: str) -> np.ndarray:
    interval = 3600000 if timeframe == "1h" else 4 * 3600000
    return np.arange(size, dtype=np.int64) * interval + 1700000000000


@pytest.mark.parametrize(("timeframe", "horizon", "interval"), [("1h", 24, 3600000), ("4h", 6, 4 * 3600000)])
def test_timestamp_preflight_matches_extract_windows_and_bounds(timeframe, horizon, interval):
    timestamps = _timestamps(800, timeframe)
    features = np.arange(len(timestamps), dtype=np.float32)[:, None]
    expected = eligible_origins_from_timestamps(
        timestamps,
        context_len=32,
        horizon=horizon,
        start_idx=100,
        end_idx=700,
        timeframe=timeframe,
    )
    _, futures, actual = extract_windows(
        features,
        features[:, 0],
        context_len=32,
        horizon=horizon,
        start_idx=100,
        end_idx=700,
        timestamps=timestamps,
        timeframe=timeframe,
    )

    assert actual == expected
    assert actual[0] + 1 >= 100
    assert actual[-1] + horizon < 700
    assert futures.shape == (len(actual), horizon)
    assert interval == (timestamps[1] - timestamps[0])


@pytest.mark.parametrize(("timeframe", "horizon", "interval"), [("1h", 4, 3600000), ("4h", 2, 4 * 3600000)])
def test_timestamp_preflight_gap_edges_match_full_window_semantics(timeframe, horizon, interval):
    timestamps = _timestamps(100, timeframe)
    timestamps[31:] += interval
    origins = eligible_origins_from_timestamps(
        timestamps,
        context_len=8,
        horizon=horizon,
        start_idx=0,
        end_idx=len(timestamps),
        timeframe=timeframe,
    )

    assert 30 - horizon in origins
    assert 38 in origins
    assert not set(range(31 - horizon, 38)) & set(origins)


def _candidate() -> AdapterManifest:
    return AdapterManifest(
        adapter_id="candidate",
        timeframe="1h",
        horizon=24,
        context_len=256,
        feature_set="A",
        feature_columns=["close"],
    )


class _GuardedSnapshot:
    def __init__(self, timestamps: np.ndarray, gap_at: int | None = None):
        self.timestamps = timestamps.copy()
        if gap_at is not None:
            self.timestamps[gap_at + 1 :] += 3600000
        values = np.arange(len(timestamps), dtype=np.float32)[:, None]
        self._features = values
        self.metadata = SimpleNamespace(sha256="guarded-snapshot")
        self.feature_reads = 0
        self.target_reads = 0
        self.on_data_access = None

    @property
    def features_a(self):
        if self.on_data_access is not None:
            self.on_data_access()
        self.target_reads += 1
        return self._features

    def get_features(self, feature_set: str):
        if self.on_data_access is not None:
            self.on_data_access()
        self.feature_reads += 1
        return self._features


def _set_selected_range(storage: GPUJobStorage, snapshot: _GuardedSnapshot, start: int, end: int) -> None:
    storage.save_auto_tune_run(
        "1h",
        "unused",
        snapshot.metadata.sha256,
        "LOCKED_VERIFICATION",
        selected_test_range_json=json.dumps({"test_start_idx": start, "test_end_idx": end}),
    )


def test_locked_gap_preflight_reads_no_features_and_consumes_no_ledger(tmp_path: Path):
    snapshot = _GuardedSnapshot(_timestamps(6480, "1h"), gap_at=6200)
    storage = GPUJobStorage(tmp_path / "jobs.db")
    _set_selected_range(storage, snapshot, 6000, 6480)

    with pytest.raises(ValueError, match="Insufficient independent"):
        run_locked_verification(snapshot, _candidate(), tmp_path / "candidate", storage=storage)

    assert snapshot.feature_reads == 0
    assert snapshot.target_reads == 0
    with storage.get_connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM locked_verification_ledger").fetchone()[0] == 0


def test_locked_ledger_is_recorded_before_first_feature_or_target_read(tmp_path: Path):
    snapshot = _GuardedSnapshot(_timestamps(6480, "1h"))
    storage = GPUJobStorage(tmp_path / "jobs.db")
    _set_selected_range(storage, snapshot, 6000, 6480)
    original_record = storage.record_locked_consumption

    def record_before_data(*args, **kwargs):
        assert snapshot.feature_reads == 0
        assert snapshot.target_reads == 0
        return original_record(*args, **kwargs)

    storage.record_locked_consumption = record_before_data

    def assert_ledger_exists_before_data_access():
        with storage.get_connection() as conn:
            assert conn.execute("SELECT COUNT(*) FROM locked_verification_ledger").fetchone()[0] == 1

    snapshot.on_data_access = assert_ledger_exists_before_data_access

    def predict(contexts, horizon, **kwargs):
        count = len(contexts)
        return np.zeros((count, horizon)), np.zeros((count, horizon, 9))

    with patch("paxg_lab.tune.locked_eval.TimesFM3Predictor") as predictor:
        predictor.return_value.predict.side_effect = predict
        report = run_locked_verification(snapshot, _candidate(), tmp_path / "candidate", storage=storage)

    assert report.num_windows >= 20
    assert snapshot.feature_reads > 0
    assert snapshot.target_reads > 0
    with storage.get_connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM locked_verification_ledger").fetchone()[0] == 1


def test_locked_cancellation_preflight_reads_no_features_and_consumes_no_ledger(tmp_path: Path):
    snapshot = _GuardedSnapshot(_timestamps(6480, "1h"))
    storage = GPUJobStorage(tmp_path / "jobs.db")
    _set_selected_range(storage, snapshot, 6000, 6480)

    with pytest.raises(InterruptedError, match="cancelled before consumption"):
        run_locked_verification(
            snapshot,
            _candidate(),
            tmp_path / "candidate",
            storage=storage,
            is_cancelled_func=lambda: True,
        )

    assert snapshot.feature_reads == 0
    assert snapshot.target_reads == 0
    with storage.get_connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM locked_verification_ledger").fetchone()[0] == 0


def test_scheduler_waiting_data_uses_timestamps_only(tmp_path: Path):
    timestamps = _timestamps(1200, "1h")
    values = np.arange(len(timestamps), dtype=np.float32)[:, None]
    snapshot = DatasetSnapshot.create("1h", timestamps, values, values)
    snapshot_dir = snapshot.save(tmp_path / "snapshots")
    storage = GPUJobStorage(tmp_path / "jobs.db")
    storage.record_locked_consumption(
        timeframe="1h",
        test_start_idx=300,
        test_end_idx=700,
        test_start_time_ms=int(timestamps[300]),
        test_end_time_ms=int(timestamps[700]),
        snapshot_hash="old",
        candidate_id="old-candidate",
        verdict="REJECTED",
    )
    storage.set_auto_run_state("1h", AutoRunState.WAITING_DATA)
    scheduler = GPUScheduler(
        db_path=storage.db_path,
        snapshots_dir=tmp_path / "snapshots",
        acquire_coordinator_lock=False,
    )

    with patch.object(DatasetSnapshot, "load", side_effect=AssertionError("features must not load")):
        scheduler._check_auto_tune_data_wakeup()

    assert storage.get_auto_run_state("1h") == AutoRunState.SEARCHING
    assert storage.list_jobs(timeframe="1h")


def test_scheduler_gap_preflight_stays_waiting_without_loading_features(tmp_path: Path):
    timestamps = _timestamps(1200, "1h")
    timestamps[900:] += 3600000
    values = np.arange(len(timestamps), dtype=np.float32)[:, None]
    snapshot = DatasetSnapshot.create("1h", timestamps, values, values)
    snapshot.save(tmp_path / "snapshots")
    storage = GPUJobStorage(tmp_path / "jobs.db")
    storage.record_locked_consumption(
        timeframe="1h",
        test_start_idx=300,
        test_end_idx=700,
        test_start_time_ms=int(timestamps[300]),
        test_end_time_ms=int(timestamps[700]),
        snapshot_hash="old",
        candidate_id="old-candidate",
        verdict="REJECTED",
    )
    storage.set_auto_run_state("1h", AutoRunState.WAITING_DATA)
    scheduler = GPUScheduler(
        db_path=storage.db_path,
        snapshots_dir=tmp_path / "snapshots",
        acquire_coordinator_lock=False,
    )

    with patch.object(DatasetSnapshot, "load", side_effect=AssertionError("features must not load")):
        scheduler._check_auto_tune_data_wakeup()

    assert storage.get_auto_run_state("1h") == AutoRunState.WAITING_DATA
    assert storage.list_jobs(timeframe="1h") == []


def test_load_timestamps_reads_only_timestamps_and_checks_sha(tmp_path: Path):
    timestamps = _timestamps(32, "1h")
    values = np.arange(len(timestamps), dtype=np.float32)[:, None]
    snapshot = DatasetSnapshot.create("1h", timestamps, values, values)
    snapshot_dir = snapshot.save(tmp_path / "snapshots")

    import numpy.lib.npyio

    original_getitem = numpy.lib.npyio.NpzFile.__getitem__
    accessed = []

    def guarded_getitem(data, key):
        accessed.append(key)
        if key != "timestamps":
            raise AssertionError(f"unexpected NPZ member: {key}")
        return original_getitem(data, key)

    with patch.object(numpy.lib.npyio.NpzFile, "__getitem__", guarded_getitem):
        np.testing.assert_array_equal(DatasetSnapshot.load_timestamps(snapshot_dir), timestamps)
    assert accessed == ["timestamps"]

    npz_path = snapshot_dir / "data.npz"
    raw = bytearray(npz_path.read_bytes())
    raw[-1] ^= 1
    npz_path.write_bytes(raw)
    with pytest.raises(RuntimeError, match="Snapshot integrity violation"):
        DatasetSnapshot.load_timestamps(snapshot_dir)
