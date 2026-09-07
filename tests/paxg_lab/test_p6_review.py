"""Regressions for PR #6 review 5125813196; real SQLite and snapshot files."""
import json
import time
from unittest.mock import MagicMock, patch

import numpy as np
import optuna
import pytest

from paxg_lab.data.snapshot import DatasetSnapshot
from paxg_lab.data.split import calculate_split_plan
from paxg_lab.eval.engine import BacktestEngine
from paxg_lab.model.manifest import AdapterManifest
from paxg_lab.model.train_spec import TrainSpec
from paxg_lab.queue.scheduler import GPUScheduler
from paxg_lab.queue.storage import GPUJobStorage
from paxg_lab.queue.types import AutoRunState, JobPriority, JobSpec, JobType
from paxg_lab.tune.locked_eval import run_locked_verification
from paxg_lab.tune.protocol import AutonomousTuningProtocol


def snapshot(n=6480, gap=False):
    ts = np.arange(n, dtype=np.int64) * 3600000 + 1700000000000
    if gap:
        ts[6200:] += 3600000
    prices = (2000 + np.sin(np.arange(n) / 10))[:, None]
    return DatasetSnapshot.create("1h", ts, prices, np.repeat(prices, 9, axis=1))


def consume(storage, snap):
    storage.record_locked_consumption(
        "1h", 3840, 6000, "old", "old-candidate", verdict="REJECTED",
        test_start_time_ms=int(snap.timestamps[3840]),
        test_end_time_ms=int(snap.timestamps[6000]),
    )


def test_second_cycle_base_boundaries_and_cache(tmp_path):
    snap = snapshot()
    protocol = AutonomousTuningProtocol("1h", snap, db_path=tmp_path / "jobs.db",
        optuna_db_path=tmp_path / "optuna.db", adapter_store_dir=tmp_path / "adapters")
    consume(protocol.job_storage, snap)
    protocol = AutonomousTuningProtocol("1h", snap, db_path=tmp_path / "jobs.db",
        optuna_db_path=tmp_path / "optuna.db", adapter_store_dir=tmp_path / "adapters")
    assert protocol.split_plan.test_start == 6000
    calls = []
    original = BacktestEngine.evaluate_fold

    def predict(self, contexts, horizon, **kwargs):
        p = np.repeat(contexts[:, -1, :1], horizon, axis=1)
        return p, np.repeat(p[:, :, None], 9, axis=2)

    def capture(self, **kwargs):
        calls.append((kwargs["start_idx"], kwargs["end_idx"]))
        return original(self, **kwargs)

    with patch("paxg_lab.eval.predictor.TimesFM3Predictor.predict", predict), patch.object(BacktestEngine, "evaluate_fold", capture):
        protocol.get_or_compute_base_report()
        expected = [(f.eval_start, f.eval_end) for f in protocol.split_plan.eval_folds]
        assert calls == expected
        protocol.get_or_compute_base_report()
        assert calls == expected  # cache hit
        protocol.split_plan = calculate_split_plan(len(snap.timestamps), "1h")
        protocol.get_or_compute_base_report()
        assert calls[3:] == [(f.eval_start, f.eval_end) for f in protocol.split_plan.eval_folds]
    assert len(list(tmp_path.glob("base_ref_*.json"))) == 2


def test_second_cycle_training_uses_selected_fold(tmp_path):
    snap = snapshot()
    storage = GPUJobStorage(tmp_path / "jobs.db")
    consume(storage, snap)
    protocol = AutonomousTuningProtocol("1h", snap, db_path=storage.db_path,
        optuna_db_path=tmp_path / "optuna.db", adapter_store_dir=tmp_path / "adapters")
    fold = protocol.split_plan.eval_folds[1]
    with patch("timesfm3.TimesFM3Torch.from_pretrained"), \
         patch("paxg_lab.tune.protocol.LoRATrainer") as trainer:
        trainer.return_value.train.side_effect = InterruptedError("captured training boundary")
        with pytest.raises(InterruptedError, match="captured"):
            protocol.run_trial_fold_evaluation(TrainSpec(timeframe="1h", feature_set="A"), 2, MagicMock())
        call = trainer.return_value.train.call_args.kwargs
        assert call["explicit_train_range"] == (fold.train_start, fold.train_end)
        assert call["explicit_val_range"] == (fold.val_early_stop_start, fold.val_early_stop_end)


def test_bounded_trials_finish_in_real_optuna_storage(tmp_path):
    protocol = AutonomousTuningProtocol("1h", snapshot(), db_path=tmp_path / "jobs.db",
        optuna_db_path=tmp_path / "optuna.db", adapter_store_dir=tmp_path / "adapters", max_trials=2)
    with patch.object(protocol, "get_or_compute_base_report"), \
         patch.object(protocol, "run_trial_fold_evaluation", return_value=(0.95, 1, 0.1)) as evaluate:
        protocol.execute_step()  # BASELINE
        for _ in range(6):
            state = protocol.execute_step()
    assert state["phase"] == "MULTI_SEED"
    assert evaluate.call_count == 6
    study = optuna.load_study(study_name=f"study_1h_{protocol.snapshot.metadata.sha256[:8]}",
                             storage=f"sqlite:///{tmp_path / 'optuna.db'}")
    assert len(study.trials) == 2
    assert all(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    assert all(t.value == pytest.approx(5.0) for t in study.trials)
    assert all(t.user_attrs["snapshot_sha256"] == protocol.snapshot.metadata.sha256 for t in study.trials)


@pytest.mark.parametrize("broken", [False, True])
def test_recovery_keeps_snapshot_and_fold(tmp_path, broken):
    snap = snapshot()
    path = snap.save(tmp_path / "snapshots")
    snapshot(6500).save(tmp_path / "snapshots")
    storage = GPUJobStorage(tmp_path / "jobs.db")
    storage.save_auto_tune_run("1h", str(path), "wrong" if broken else snap.metadata.sha256,
                              "TRIAL", current_trial=2, current_fold=2)
    scheduler = GPUScheduler(db_path=storage.db_path, acquire_coordinator_lock=False)
    jobs = storage.list_jobs(job_type=JobType.AUTO_TRIAL.value)
    if broken:
        assert not jobs
        assert storage.get_auto_run_state("1h") == AutoRunState.PAUSED_ERROR
    else:
        assert len(jobs) == 1
        assert jobs[0].payload["snapshot_path"] == str(path)
        assert storage.get_auto_tune_run("1h")["current_fold"] == 2
        assert DatasetSnapshot.load(path).source_path == str(path.resolve())


@pytest.mark.parametrize("state", [AutoRunState.SEARCHING, AutoRunState.VALIDATING, AutoRunState.PAUSED_ERROR])
def test_background_transition_and_enqueue_cannot_unstop(tmp_path, state):
    storage = GPUJobStorage(tmp_path / "jobs.db")
    storage.set_auto_run_state("1h", AutoRunState.STOPPED)
    storage.set_auto_run_state("1h", state)
    assert storage.get_auto_run_state("1h") == AutoRunState.STOPPED
    job = JobSpec(job_id="late", job_type=JobType.AUTO_TRIAL.value,
                  priority=JobPriority.AUTO.value, timeframe="1h")
    with pytest.raises(ValueError, match="STOPPED"):
        storage.submit_job(job, reject_if_stopped=True)
    assert not storage.list_jobs()
    storage.set_auto_run_state("1h", AutoRunState.SEARCHING, allow_unstop=True)
    storage.submit_job(job, reject_if_stopped=True)


def test_wakeup_stop_race(tmp_path):
    snap = snapshot(6600)
    snap.save(tmp_path / "snapshots")
    scheduler = GPUScheduler(db_path=tmp_path / "jobs.db", snapshots_dir=tmp_path / "snapshots",
                             acquire_coordinator_lock=False)
    consume(scheduler.storage, snap)
    scheduler.storage.set_auto_run_state("1h", AutoRunState.WAITING_DATA)
    original = scheduler.storage.set_auto_run_state

    def stop_before_write(tf, state, **kwargs):
        original(tf, AutoRunState.STOPPED)
        original(tf, state, **kwargs)

    with patch.object(scheduler.storage, "set_auto_run_state", stop_before_write):
        scheduler._check_auto_tune_data_wakeup()
    assert scheduler.storage.get_auto_run_state("1h") == AutoRunState.STOPPED
    assert not scheduler.storage.list_jobs()


def test_timeout_stop_race(tmp_path):
    scheduler = GPUScheduler(db_path=tmp_path / "jobs.db", acquire_coordinator_lock=False)
    storage = scheduler.storage
    job = JobSpec(job_id="timeout", job_type=JobType.AUTO_TRIAL.value,
                  priority=JobPriority.AUTO.value, timeframe="1h",
                  started_at=time.time() - 1300, heartbeat_at=time.time(), timeout_seconds=1200)
    storage.submit_job(job)
    scheduler.active_job_id = job.job_id
    scheduler.active_worker = MagicMock()
    scheduler.active_worker.poll.return_value = None
    scheduler.active_worker_pid = 999999

    def stop_during_timeout(*args, **kwargs):
        storage.set_auto_run_state("1h", AutoRunState.STOPPED)
        return True

    with patch("paxg_lab.queue.scheduler.safe_terminate_process", stop_during_timeout), \
         patch("paxg_lab.queue.scheduler.is_process_alive", return_value=False):
        scheduler.tick()
    assert storage.get_auto_run_state("1h") == AutoRunState.STOPPED
    assert storage.get_job(job.job_id).status == "INTERRUPTED"
    assert len(storage.list_jobs()) == 1


def test_gap_preflight_and_persisted_range_do_not_consume(tmp_path):
    snap = snapshot(gap=True)
    storage = GPUJobStorage(tmp_path / "jobs.db")
    storage.save_auto_tune_run("1h", "unused", snap.metadata.sha256, "LOCKED_VERIFICATION",
        selected_test_range_json=json.dumps({"test_start_idx": 6000, "test_end_idx": 6480}))
    manifest = AdapterManifest(adapter_id="candidate", timeframe="1h", horizon=24,
                               context_len=256, feature_set="A", feature_columns=["close"])
    with patch("paxg_lab.tune.locked_eval.TimesFM3Predictor") as predictor:
        with pytest.raises(ValueError, match="Insufficient independent"):
            run_locked_verification(snap, manifest, tmp_path / "candidate", storage=storage)
        predictor.assert_not_called()
    with storage.get_connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM locked_verification_ledger").fetchone()[0] == 0

    snap.save(tmp_path / "snapshots")
    consume(storage, snap)
    storage.set_auto_run_state("1h", AutoRunState.WAITING_DATA)
    scheduler = GPUScheduler(db_path=storage.db_path, snapshots_dir=tmp_path / "snapshots",
                             acquire_coordinator_lock=False)
    scheduler._check_auto_tune_data_wakeup()
    assert storage.get_auto_run_state("1h") == AutoRunState.WAITING_DATA
    snapshot(7100, gap=True).save(tmp_path / "snapshots")
    scheduler._check_auto_tune_data_wakeup()
    assert storage.get_auto_run_state("1h") == AutoRunState.SEARCHING
