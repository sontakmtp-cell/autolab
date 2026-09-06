"""Standalone worker process executing a single GPU task in PAXG Forecast Lab.

Each GPU job runs in its own isolated subprocess so that when the task finishes,
the OS completely reclaims all CUDA VRAM and memory allocations.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys
import threading
import time
import traceback
from typing import Any

import psutil

from .storage import GPUJobStorage
from .types import AutoRunState, JobPriority, JobSpec, JobStatus, JobType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [PID %(process)d] %(message)s",
)
logger = logging.getLogger(__name__)


class GPUWorker:
    """Executes a single GPU job with background heartbeat, cancel monitoring, and memory cleanup."""

    def __init__(
        self,
        job_id: str,
        db_path: str | Path,
        heartbeat_interval: float = 2.0,
    ):
        self.job_id = job_id
        self.storage = GPUJobStorage(db_path)
        self.heartbeat_interval = heartbeat_interval
        self.stop_event = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._is_running = False

        # Live progress state shared with heartbeat thread
        self.progress_pct = 0.0
        self.progress_message = "Initializing"

    def _heartbeat_loop(self) -> None:
        """Background thread updating heartbeat and monitoring cancellation requests."""
        while self._is_running:
            try:
                cancel_requested = self.storage.update_heartbeat(
                    job_id=self.job_id,
                    progress_pct=self.progress_pct,
                    progress_message=self.progress_message,
                )
                if cancel_requested:
                    logger.info("Worker detected cancel request from storage. Signaling stop event...")
                    self.stop_event.set()
            except Exception as exc:
                logger.warning("Heartbeat update failed: %s", exc)

            time.sleep(self.heartbeat_interval)

    def start_heartbeat(self) -> None:
        """Starts background heartbeat thread."""
        self._is_running = True
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"heartbeat-{self.job_id}",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def stop_heartbeat(self) -> None:
        """Stops background heartbeat thread."""
        self._is_running = False
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=3.0)

    def run(self) -> int:
        """Main execution entry point for the worker.
        
        Returns:
            Exit code: 0 on success/clean cancellation, 1 on error.
        """
        pid = os.getpid()
        create_time = psutil.Process(pid).create_time()
        logger.info("Worker starting for job '%s' (PID=%d, create_time=%.2f)", self.job_id, pid, create_time)

        # Register worker PID and start heartbeat
        self.storage.register_worker(self.job_id, pid, create_time)
        self.start_heartbeat()

        job = self.storage.get_job(self.job_id)
        if not job:
            logger.error("Job '%s' not found in database!", self.job_id)
            self.stop_heartbeat()
            return 1

        exit_code = 0
        try:
            # Check if cancelled before worker actually began
            if job.cancel_requested or job.status == JobStatus.CANCELLED.value:
                logger.info("Job '%s' was cancelled before execution started.", self.job_id)
                self.storage.mark_cancelled(self.job_id, "Cancelled before execution")
                return 0

            # Execute job dispatch
            self.progress_message = f"Executing {job.job_type}"
            result = self._dispatch(job)

            if self.stop_event.is_set():
                logger.info("Job '%s' stopped cleanly on cancellation.", self.job_id)
                self.storage.mark_cancelled(self.job_id, "Cancelled by user request", result=result)
                exit_code = 0
            else:
                logger.info("Job '%s' finished successfully.", self.job_id)
                self.storage.mark_succeeded(self.job_id, result, progress_message="Completed successfully")
                exit_code = 0

        except (Exception, KeyboardInterrupt) as exc:
            if self.stop_event.is_set():
                logger.info("Job '%s' stopped cleanly on cancellation.", self.job_id)
                self.storage.mark_cancelled(self.job_id, "Cancelled by user request")
                return 0

            err_msg = traceback.format_exc()
            logger.error("Job '%s' failed with error:\n%s", self.job_id, err_msg)

            # Detect CUDA Out of Memory specifically
            is_oom = "out of memory" in str(exc).lower() or "cuda out of memory" in err_msg.lower()
            if is_oom:
                prefix = "[CUDA_OOM] "
            else:
                prefix = ""

            self.storage.mark_failed(self.job_id, f"{prefix}{str(exc)}")
            if (job.job_type == JobType.AUTO_TRIAL.value or job.priority == JobPriority.AUTO.value) and not is_oom:
                tf = job.timeframe or "1h"
                logger.error("Error on AUTO job '%s'. Setting PAUSED_ERROR.", self.job_id)
                self.storage.set_auto_run_state(tf, AutoRunState.PAUSED_ERROR)
            exit_code = 1

        finally:
            self.stop_heartbeat()
            # Clean up GPU memory
            self._cleanup_gpu_memory()

        return exit_code

    def _cleanup_gpu_memory(self) -> None:
        """Flushes CUDA caches if torch is present."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("CUDA cache emptied successfully.")
        except Exception:
            pass

    def _dispatch(self, job: JobSpec) -> dict[str, Any]:
        """Dispatches job execution to corresponding domain logic."""
        job_type = job.job_type

        if job_type == JobType.DUMMY.value:
            return self._handle_dummy_job(job)
        elif job_type == JobType.FORECAST.value:
            return self._handle_forecast_job(job)
        elif job_type == JobType.TRAIN.value:
            return self._handle_train_job(job)
        elif job_type == JobType.BACKTEST.value:
            return self._handle_backtest_job(job)
        elif job_type == JobType.AUTO_TRIAL.value:
            return self._handle_auto_trial_job(job)
        else:
            raise ValueError(f"Unsupported job type: {job_type}")

    def _handle_dummy_job(self, job: JobSpec) -> dict[str, Any]:
        """Handles dummy / mock jobs for testing heartbeat, timeouts, cancellations, and errors."""
        payload = job.payload
        steps = int(payload.get("steps", 5))
        step_sleep = float(payload.get("step_sleep", 0.2))
        simulate_oom = bool(payload.get("simulate_oom", False))
        fail_at_step = payload.get("fail_at_step")
        hang_at_step = payload.get("hang_at_step")

        for step in range(1, steps + 1):
            if self.stop_event.is_set():
                logger.info("Dummy job detected stop event at step %d/%d", step, steps)
                return {"stopped_at_step": step, "status": "cancelled"}

            if hang_at_step is not None and step >= int(hang_at_step):
                logger.warning("Simulating process hang at step %d: pausing heartbeat loop...", step)
                # Temporarily stop heartbeat to simulate process hang/freeze
                self.stop_heartbeat()
                # Sleep longer than heartbeat timeout
                time.sleep(60.0)

            if fail_at_step is not None and step >= int(fail_at_step):
                if simulate_oom:
                    raise RuntimeError("CUDA out of memory. Tried to allocate 4.00 GiB")
                raise RuntimeError(f"Simulated failure at step {step}")

            self.progress_pct = (step / steps) * 100.0
            self.progress_message = f"Step {step}/{steps}"
            time.sleep(step_sleep)

        return {"completed_steps": steps, "status": "ok"}

    def _handle_forecast_job(self, job: JobSpec) -> dict[str, Any]:
        """Handles single forecast request."""
        from paxg_lab.constants import get_horizon_for_timeframe
        from paxg_lab.eval.predictor import TimesFM3Predictor
        from paxg_lab.eval.types import ForecastRequest
        import numpy as np

        payload = job.payload
        timeframe = payload.get("timeframe", job.timeframe or "1h")
        horizon = int(payload.get("horizon") or get_horizon_for_timeframe(timeframe))
        context_len = int(payload.get("context_len", 256))
        adapter_path = payload.get("adapter_path")
        feature_set = payload.get("feature_set", "B" if adapter_path else "A")
        columns = payload.get("columns")

        # Context features passed directly or loaded from snapshot - validate fail-fast before model loading
        if "contexts" in payload:
            ctx_array = np.asarray(payload["contexts"], dtype=np.float32)
            if ctx_array.ndim == 3 and ctx_array.shape[0] == 1:
                ctx_array = ctx_array[0]
            forecast_origin_time = payload.get("forecast_origin_time")
            if forecast_origin_time is None:
                if "timestamps" in payload and len(payload["timestamps"]) > 0:
                    forecast_origin_time = int(payload["timestamps"][-1])
                else:
                    raise ValueError(
                        "Forecast request with 'contexts' requires 'forecast_origin_time' or 'timestamps'"
                    )
            else:
                forecast_origin_time = int(forecast_origin_time)
        elif "snapshot_path" in payload:
            from paxg_lab.data.snapshot import DatasetSnapshot
            snapshot_path = payload["snapshot_path"]
            if not Path(snapshot_path).exists():
                raise FileNotFoundError(f"Snapshot path '{snapshot_path}' not found.")
            snapshot = DatasetSnapshot.load(snapshot_path)
            feats = snapshot.get_features(feature_set)
            if len(feats) < context_len:
                raise ValueError(f"Snapshot has only {len(feats)} candles, less than required context_len={context_len}")
            recent_ctx = feats[-context_len:]
            ctx_array = np.asarray(recent_ctx, dtype=np.float32)
            forecast_origin_time = int(payload.get("forecast_origin_time") or snapshot.timestamps[-1])
        else:
            raise ValueError(
                "Forecast request missing required 'contexts' array or valid 'snapshot_path'. "
                "Forecasts must never fallback to synthetic random noise in production."
            )

        self.progress_message = f"Loading model for {timeframe} forecast"
        predictor = TimesFM3Predictor(adapter_path=adapter_path)

        req = ForecastRequest(
            timeframe=timeframe,
            horizon=horizon,
            context_len=context_len,
            feature_set=feature_set,
            columns=columns,
            adapter_path=adapter_path,
        )

        self.progress_message = "Generating forecast"
        res = predictor.forecast_request(req, ctx_array, forecast_origin_time=forecast_origin_time)
        return res.to_dict()

    def _handle_train_job(self, job: JobSpec) -> dict[str, Any]:
        """Handles manual or auto LoRA training job."""
        import pandas as pd
        from paxg_lab.constants import MODEL_REPO, MODEL_REVISION
        from paxg_lab.data.snapshot import DatasetSnapshot
        from paxg_lab.model.store import AdapterStore
        from paxg_lab.model.train_spec import TrainSpec
        from paxg_lab.model.trainer import LoRATrainer
        from timesfm3 import TimesFM3Torch

        payload = job.payload
        raw_spec = payload.get("train_spec")
        if raw_spec is None:
            raw_spec = {k: v for k, v in payload.items() if k in TrainSpec.__dataclass_fields__}

        if isinstance(raw_spec, TrainSpec):
            spec = raw_spec
        elif isinstance(raw_spec, dict):
            spec_dict = dict(raw_spec)
            if "timeframe" not in spec_dict and job.timeframe:
                spec_dict["timeframe"] = job.timeframe
            spec = TrainSpec.from_dict(spec_dict)
        else:
            raise TypeError(f"Invalid train_spec type in payload: {type(raw_spec)}")

        snapshot_path = payload.get("snapshot_path")
        if snapshot_path and Path(snapshot_path).exists():
            snapshot = DatasetSnapshot.load(snapshot_path)
            features_df = snapshot.to_dataframe(spec.feature_set)
            snapshot_hash = snapshot.metadata.sha256
        else:
            raise FileNotFoundError(f"Snapshot path required for training, got '{snapshot_path}'")

        self.progress_message = "Initializing base model"
        base_model = TimesFM3Torch.from_pretrained(MODEL_REPO, revision=MODEL_REVISION)

        def progress_cb(info: dict[str, Any]) -> bool:
            epoch = info.get("epoch", 0)
            max_e = spec.max_epochs
            self.progress_pct = (epoch / max(1, max_e)) * 100.0
            if "step" in info:
                self.progress_message = f"Epoch {epoch}/{max_e}, step {info['step']}, loss: {info.get('loss', 0.0):.4f}"
            else:
                self.progress_message = f"Epoch {epoch}/{max_e}, val_loss: {info.get('val_loss', 0.0):.4f}"
            # Check for cancellation
            return not self.stop_event.is_set()

        checkpoint_dir = payload.get("checkpoint_dir", "var/paxg_lab/checkpoints")
        from paxg_lab.queue.checkpoint import TrainingCheckpointManager
        checkpoint_manager = TrainingCheckpointManager(checkpoint_dir)
        resume_job_id = payload.get("resume_from_job_id") or job.job_id

        trainer = LoRATrainer(base_model=base_model, spec=spec)
        train_result = trainer.train(
            features_df=features_df,
            snapshot_hash=snapshot_hash,
            fold_id=int(payload.get("fold_id", 1)),
            progress_callback=progress_cb,
            checkpoint_manager=checkpoint_manager,
            job_id=resume_job_id,
        )

        store_dir = payload.get("adapter_store_dir", "var/paxg_lab/adapters")
        store = AdapterStore(store_dir)
        smoke_test = bool(payload.get("smoke_test", True))

        if self.stop_event.is_set():
            # Stop preservation: persist loadable adapter artifact before worker exits
            saved_dir = store.save_adapter(
                peft_model=train_result.trained_model,
                manifest=train_result.manifest,
                base_model=base_model,
                smoke_test=False,
            )
            ckpt_dir = checkpoint_manager.get_checkpoint_dir(resume_job_id)
            return {
                "status": "cancelled",
                "best_epoch": train_result.best_epoch,
                "best_val_loss": train_result.best_val_loss,
                "checkpoint_path": str(ckpt_dir) if ckpt_dir.exists() else None,
                "adapter_path": str(saved_dir),
                "total_steps": train_result.total_steps,
            }

        # Save to store
        saved_dir = store.save_adapter(
            peft_model=train_result.trained_model,
            manifest=train_result.manifest,
            base_model=base_model,
            smoke_test=smoke_test,
        )

        return {
            "adapter_id": train_result.manifest.adapter_id,
            "adapter_path": str(saved_dir),
            "timeframe": spec.timeframe,
            "best_epoch": train_result.best_epoch,
            "best_val_loss": train_result.best_val_loss,
            "training_time_sec": train_result.total_training_time_sec,
            "total_steps": train_result.total_steps,
            "history": train_result.history,
        }

    def _handle_backtest_job(self, job: JobSpec) -> dict[str, Any]:
        """Handles backtest job."""
        from paxg_lab.data.snapshot import DatasetSnapshot
        from paxg_lab.eval.engine import BacktestEngine
        from paxg_lab.eval.predictor import TimesFM3Predictor
        from paxg_lab.eval.types import FoldMetrics, ScoreReport

        payload = job.payload
        timeframe = payload.get("timeframe", job.timeframe or "1h")
        snapshot_path = payload.get("snapshot_path")
        if not snapshot_path or not Path(snapshot_path).exists():
            raise FileNotFoundError(f"Snapshot path required for backtest, got '{snapshot_path}'")

        snapshot = DatasetSnapshot.load(snapshot_path)
        adapter_path = payload.get("adapter_path")
        model_type = payload.get("model_type", "lora" if adapter_path else "base")
        feature_set = payload.get("feature_set", "B" if adapter_path else "A")
        context_len = int(payload.get("context_len", 256))
        batch_size = int(payload.get("batch_size", 16))
        include_locked_test = bool(payload.get("include_locked_test", False))
        model_name = payload.get("model_name", "TimesFM3-LoRA" if adapter_path else "TimesFM3-Base")

        # Determine whether this is base reference or candidate
        is_base_reference = bool(payload.get("is_base_reference", model_type == "base" and not adapter_path))

        base_ref_data = payload.get("base_reference_metrics")
        base_reference_metrics: ScoreReport | dict[int | str, FoldMetrics] | None = None
        if base_ref_data is not None:
            if isinstance(base_ref_data, ScoreReport):
                base_reference_metrics = base_ref_data
            elif isinstance(base_ref_data, dict):
                if "fold_metrics" in base_ref_data:
                    fold_metrics = [
                        FoldMetrics(**m) if isinstance(m, dict) else m
                        for m in base_ref_data["fold_metrics"]
                    ]
                    test_m = base_ref_data.get("test_metrics")
                    if test_m and isinstance(test_m, dict):
                        test_m = FoldMetrics(**test_m)
                    base_reference_metrics = ScoreReport(
                        **{k: v for k, v in base_ref_data.items() if k not in ("fold_metrics", "test_metrics")},
                        fold_metrics=fold_metrics,
                        test_metrics=test_m,
                    )
                else:
                    base_reference_metrics = {
                        k: FoldMetrics(**v) if isinstance(v, dict) else v
                        for k, v in base_ref_data.items()
                    }

        def progress_cb(info: dict[str, Any]) -> bool:
            if "batch_idx" in info and "total_batches" in info:
                self.progress_message = f"Evaluating batch {info['batch_idx']}/{info['total_batches']}"
            elif "message" in info:
                self.progress_message = info["message"]
            return not self.stop_event.is_set()

        # If candidate run lacks base reference metrics, compute base reference first using clean base predictor
        if not is_base_reference and base_reference_metrics is None:
            logger.info("Computing base reference baseline on snapshot for candidate backtest...")
            base_predictor = TimesFM3Predictor()
            base_engine = BacktestEngine(predictor=base_predictor)
            base_report = base_engine.run_full_backtest(
                snapshot=snapshot,
                feature_set="A",
                context_len=256,
                batch_size=batch_size,
                model_name="TimesFM3-Base",
                is_base_reference=True,
                include_locked_test=include_locked_test,
                progress_callback=progress_cb,
            )
            base_reference_metrics = base_report

        predictor = TimesFM3Predictor(adapter_path=adapter_path)
        engine = BacktestEngine(predictor=predictor)

        try:
            report = engine.run_full_backtest(
                snapshot=snapshot,
                feature_set=feature_set,
                context_len=context_len,
                batch_size=batch_size,
                model_name=model_name,
                is_base_reference=is_base_reference,
                base_reference_metrics=base_reference_metrics,
                include_locked_test=include_locked_test,
                progress_callback=progress_cb,
            )
            return report.to_dict()
        except InterruptedError:
            logger.info("Backtest job interrupted by stop event.")
            return {"status": "cancelled", "message": "Backtest interrupted"}

    def _handle_auto_trial_job(self, job: JobSpec) -> dict[str, Any]:
        """Handles autonomous optimization: full P6 tuning cycle or single training trial."""
        payload = job.payload or {}
        # If payload specifically provided a train_spec, run direct train job (legacy / unit test path)
        if "train_spec" in payload:
            return self._handle_train_job(job)

        # Production P6 path: run full AutonomousTuningProtocol
        from paxg_lab.data.snapshot import DatasetSnapshot
        from paxg_lab.tune.protocol import AutonomousTuningProtocol

        timeframe = payload.get("timeframe", job.timeframe or "1h")
        snapshot_path = payload.get("snapshot_path")
        if not snapshot_path or not Path(snapshot_path).exists():
            # Discover latest snapshot for timeframe
            snap_dir = Path("var/paxg_lab/snapshots")
            candidates = sorted(snap_dir.glob(f"paxgusdt_{timeframe}_*"))
            if not candidates:
                raise FileNotFoundError(
                    f"No dataset snapshot found for timeframe '{timeframe}' to run auto tuning."
                )
            snapshot_path = str(candidates[-1])

        snapshot = DatasetSnapshot.load(snapshot_path)
        max_trials = int(payload.get("max_trials", 30))
        fast_dev_mode = bool(payload.get("fast_dev_mode", False))

        def progress_cb(info: dict[str, Any]) -> None:
            self.progress_message = info.get("message", "Running autonomous tuning...")
            pct = float(info.get("progress_pct", 50.0))
            self.progress_pct = max(0.0, min(pct, 100.0))

        protocol = AutonomousTuningProtocol(
            timeframe=timeframe,
            snapshot=snapshot,
            db_path=self.storage.db_path,
        )

        run_entire_cycle = bool(payload.get("run_entire_cycle", False))

        try:
            if run_entire_cycle:
                run_result = protocol.run_tuning_cycle(
                    max_trials=max_trials,
                    is_cancelled_func=lambda: self.stop_event.is_set(),
                    progress_callback=progress_cb,
                    fast_dev_mode=fast_dev_mode,
                )
                return run_result if isinstance(run_result, dict) else run_result.to_dict()
            else:
                step_res = protocol.execute_step(
                    max_trials=max_trials,
                    is_cancelled_func=lambda: self.stop_event.is_set(),
                    progress_callback=progress_cb,
                    fast_dev_mode=fast_dev_mode,
                )
                current_phase = step_res.get("phase")
                auto_state = self.storage.get_auto_run_state(timeframe)

                # Check if auto run was stopped by user
                if self.stop_event.is_set() or auto_state == AutoRunState.STOPPED:
                    logger.info("Auto-run for %s is STOPPED or cancelled; skipping next step submission.", timeframe)
                    return {"status": "cancelled", "message": "Auto-run stopped by user"}

                # If tuning cycle has remaining steps and auto run is active, submit next step to queue
                if current_phase != "WAITING_DATA" and auto_state in (AutoRunState.SEARCHING, AutoRunState.VALIDATING):
                    next_job_id = f"auto_step_{timeframe}_{int(time.time() * 1000)}"
                    next_spec = JobSpec(
                        job_id=next_job_id,
                        job_type=JobType.AUTO_TRIAL.value,
                        timeframe=timeframe,
                        priority=JobPriority.AUTO.value,
                        payload={
                            "timeframe": timeframe,
                            "snapshot_path": snapshot_path,
                            "max_trials": max_trials,
                            "fast_dev_mode": fast_dev_mode,
                        },
                        timeout_seconds=1200.0,
                    )
                    self.storage.submit_job(next_spec)
                    logger.info("Submitted next bounded auto job '%s' (phase=%s, priority=%d)", next_job_id, current_phase, next_spec.priority)

                return step_res
        except InterruptedError as int_err:
            logger.info("Auto tuning job '%s' cancelled: %s", job.job_id, int_err)
            return {"status": "cancelled", "message": str(int_err)}


def run_worker(job_id: str, db_path: str | Path) -> int:
    """Invokes worker for a specific job."""
    worker = GPUWorker(job_id=job_id, db_path=db_path)
    return worker.run()


def main() -> None:
    """CLI entry point for subprocess worker invocation."""
    parser = argparse.ArgumentParser(description="PAXG Forecast Lab GPU Subprocess Worker")
    parser.add_argument("--job-id", required=True, help="Job ID to execute")
    parser.add_argument("--db-path", required=True, help="Path to SQLite database")
    args = parser.parse_args()

    exit_code = run_worker(job_id=args.job_id, db_path=args.db_path)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
