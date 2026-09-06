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
from .types import JobPriority, JobSpec, JobStatus, JobType

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
                self.storage.mark_cancelled(self.job_id, "Cancelled by user request")
                exit_code = 0
            else:
                logger.info("Job '%s' finished successfully.", self.job_id)
                self.storage.mark_succeeded(self.job_id, result, progress_message="Completed successfully")
                exit_code = 0

        except Exception as exc:
            err_msg = traceback.format_exc()
            logger.error("Job '%s' failed with error:\n%s", self.job_id, err_msg)

            # Detect CUDA Out of Memory specifically
            is_oom = "out of memory" in str(exc).lower() or "cuda out of memory" in err_msg.lower()
            if is_oom:
                prefix = "[CUDA_OOM] "
            else:
                prefix = ""

            self.storage.mark_failed(self.job_id, f"{prefix}{str(exc)}")
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
        from paxg_lab.eval.predictor import TimesFM3Predictor
        from paxg_lab.eval.types import ForecastRequest
        import numpy as np

        payload = job.payload
        timeframe = payload.get("timeframe", job.timeframe or "1h")
        context_len = int(payload.get("context_len", 256))
        adapter_path = payload.get("adapter_path")
        feature_set = payload.get("feature_set", "A")

        self.progress_message = f"Loading model for {timeframe} forecast"
        predictor = TimesFM3Predictor()
        if adapter_path:
            predictor.load_adapter(adapter_path)

        # Context features passed directly or loaded
        if "contexts" in payload:
            ctx_array = np.array(payload["contexts"], dtype=np.float32)
        else:
            # Dummy inference data if running isolated
            num_features = 1 if feature_set == "A" else 9
            ctx_array = np.random.randn(1, num_features, context_len).astype(np.float32)

        req = ForecastRequest(
            timeframe=timeframe,
            context_len=context_len,
            feature_set=feature_set,
            adapter_path=adapter_path,
        )

        self.progress_message = "Generating forecast"
        res = predictor.forecast_request(req, ctx_array)
        return res.to_dict()

    def _handle_train_job(self, job: JobSpec) -> dict[str, Any]:
        """Handles manual or auto LoRA training job."""
        import pandas as pd
        from paxg_lab.data.snapshot import DatasetSnapshot
        from paxg_lab.model.store import AdapterStore
        from paxg_lab.model.train_spec import TrainSpec
        from paxg_lab.model.trainer import LoRATrainer
        from timesfm3.model import TimesFM3

        payload = job.payload
        spec_dict = payload.get("train_spec", {})
        spec = TrainSpec(
            timeframe=spec_dict.get("timeframe", job.timeframe or "1h"),
            context_len=int(spec_dict.get("context_len", 256)),
            horizon=spec_dict.get("horizon"),
            feature_set=spec_dict.get("feature_set", "B"),
            lora_r=int(spec_dict.get("lora_r", 4)),
            lora_alpha=spec_dict.get("lora_alpha"),
            lora_dropout=float(spec_dict.get("lora_dropout", 0.10)),
            learning_rate=float(spec_dict.get("learning_rate", 5e-5)),
            max_epochs=int(spec_dict.get("max_epochs", 2)),
            batch_size=int(spec_dict.get("batch_size", 2)),
            gradient_accumulation_steps=int(spec_dict.get("gradient_accumulation_steps", 8)),
            seed=int(spec_dict.get("seed", 42)),
        )

        snapshot_path = payload.get("snapshot_path")
        if snapshot_path and Path(snapshot_path).exists():
            snapshot = DatasetSnapshot.load(snapshot_path)
            features_df = snapshot.to_dataframe()
            snapshot_hash = snapshot.sha256
        else:
            raise FileNotFoundError(f"Snapshot path required for training, got '{snapshot_path}'")

        self.progress_message = "Initializing base model"
        base_model = TimesFM3()

        def progress_cb(info: dict[str, Any]) -> bool:
            epoch = info.get("epoch", 0)
            max_e = spec.max_epochs
            self.progress_pct = (epoch / max(1, max_e)) * 100.0
            self.progress_message = f"Epoch {epoch}/{max_e}, val_loss: {info.get('val_loss', 0.0):.4f}"
            # Check for cancellation
            return not self.stop_event.is_set()

        trainer = LoRATrainer(base_model=base_model, spec=spec)
        train_result = trainer.train(
            features_df=features_df,
            snapshot_hash=snapshot_hash,
            fold_id=int(payload.get("fold_id", 1)),
            progress_callback=progress_cb,
        )

        if self.stop_event.is_set():
            return {"status": "cancelled", "best_val_loss": train_result.best_val_loss}

        # Save to store
        store_dir = payload.get("adapter_store_dir", "var/paxg_lab/adapters")
        store = AdapterStore(store_dir)
        saved_dir = store.save_trained_adapter(train_result, base_model=base_model)

        return {
            "adapter_id": train_result.manifest.adapter_id,
            "adapter_path": str(saved_dir),
            "best_epoch": train_result.best_epoch,
            "best_val_loss": train_result.best_val_loss,
            "training_time_sec": train_result.total_training_time_sec,
        }

    def _handle_backtest_job(self, job: JobSpec) -> dict[str, Any]:
        """Handles backtest job."""
        from paxg_lab.data.snapshot import DatasetSnapshot
        from paxg_lab.eval.engine import BacktestEngine
        from paxg_lab.eval.predictor import TimesFM3Predictor
        from paxg_lab.eval.types import BacktestSpec

        payload = job.payload
        timeframe = payload.get("timeframe", job.timeframe or "1h")
        snapshot_path = payload.get("snapshot_path")
        if not snapshot_path or not Path(snapshot_path).exists():
            raise FileNotFoundError(f"Snapshot path required for backtest, got '{snapshot_path}'")

        snapshot = DatasetSnapshot.load(snapshot_path)
        spec = BacktestSpec(
            timeframe=timeframe,
            model_type=payload.get("model_type", "base"),
            adapter_path=payload.get("adapter_path"),
            context_len=int(payload.get("context_len", 256)),
            feature_set=payload.get("feature_set", "A"),
        )

        predictor = TimesFM3Predictor()
        if spec.adapter_path:
            predictor.load_adapter(spec.adapter_path)

        engine = BacktestEngine(predictor=predictor)

        def progress_cb(info: dict[str, Any]) -> bool:
            self.progress_message = info.get("message", "Evaluating fold")
            return not self.stop_event.is_set()

        report = engine.run_full_backtest(snapshot, spec, progress_callback=progress_cb)
        return report.to_dict()

    def _handle_auto_trial_job(self, job: JobSpec) -> dict[str, Any]:
        """Handles single autonomous optimization trial."""
        return self._handle_train_job(job)


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
