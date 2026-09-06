"""Autonomous Tuning and Candidate Verification Protocol for Phase P6.

Orchestrates the complete end-to-end pipeline:
1. Bayesian hyperparameter search using Optuna TPE (max 30 trials, 10 exploration, 12 stagnation stop).
2. Multi-seed stability verification on seeds [42, 123, 2026] with median score selection.
3. Final candidate training from base on all historical data prior to locked test set.
4. Candidate lock (checksums, frozen manifest).
5. Strict single locked verification on 90-day test set (>= 20 independent 24-hour blocks).
6. Winner gatekeeper evaluation: replaces recommended adapter ONLY if all 7 criteria pass.
7. State transition to WAITING_DATA if candidate rejected, enforcing test set non-reuse.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import logging
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np
import optuna
from optuna.trial import TrialState

from ..constants import (
    MODEL_REPO,
    MODEL_REVISION,
    get_horizon_for_timeframe,
)
from ..data.snapshot import DatasetSnapshot
from ..data.split import calculate_split_plan
from ..eval.engine import BacktestEngine
from ..eval.metrics import compute_score_v1
from ..eval.predictor import TimesFM3Predictor
from ..eval.types import FoldMetrics, ScoreReport
from ..model.manifest import AdapterManifest
from ..model.store import AdapterStore
from ..model.train_spec import TrainSpec
from ..model.trainer import LoRATrainer
from ..queue.storage import GPUJobStorage
from ..queue.types import AutoRunState
from .gatekeeper import GatekeeperDecision, LoRAGatekeeper
from .locked_eval import LockedVerificationReport, run_locked_verification
from .optimizer import DEFAULT_OPTUNA_DB_PATH, OptunaTPEOptimizer
from .space import suggest_trial_spec

logger = logging.getLogger(__name__)

SEEDS_MULTI_RUN = (42, 123, 2026)


@dataclass
class MultiSeedEvalSummary:
    """Summary of multi-seed evaluation across seeds [42, 123, 2026]."""

    seeds: list[int]
    scores: list[float]
    best_epochs: list[int]
    median_score: float
    median_best_epoch: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class P6RunResult:
    """Complete summary of a single autonomous optimization and validation run."""

    timeframe: str
    snapshot_id: str
    study_name: str
    total_trials: int
    best_trial_number: int
    best_trial_params: dict[str, Any]
    multi_seed_summary: dict[str, Any]
    final_candidate_id: str
    test_score_report: dict[str, Any]
    decision: dict[str, Any]
    resulting_auto_state: str
    finished_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def get_top_unique_specs(study: optuna.Study, timeframe: str, top_n: int = 3) -> list[TrainSpec]:
    """Extracts up to top_n unique hyperparameter configurations from completed study trials."""
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE and t.value is not None]
    completed.sort(key=lambda t: float(t.value), reverse=True)
    unique_specs: list[TrainSpec] = []
    seen_params: set[tuple] = set()
    for t in completed:
        spec = suggest_trial_spec(t, timeframe=timeframe, seed=42)
        key = (
            spec.context_len, spec.lora_r, spec.lora_alpha, spec.learning_rate,
            spec.lora_dropout, spec.weight_decay, spec.feature_set, str(spec.history_days),
            spec.extended_targets,
        )
        if key not in seen_params:
            seen_params.add(key)
            unique_specs.append(spec)
        if len(unique_specs) >= top_n:
            break
    return unique_specs


class AutonomousTuningProtocol:
    """Coordinates autonomous Optuna search, multi-seed validation, and winner gatekeeping."""

    def __init__(
        self,
        timeframe: str,
        snapshot: DatasetSnapshot,
        db_path: Path | str = "var/paxg_lab/paxg_lab.db",
        optuna_db_path: Path | str = DEFAULT_OPTUNA_DB_PATH,
        adapter_store_dir: Path | str = "var/paxg_lab/adapters",
        audit_dir: Path | str = "var/paxg_lab/audit_reports",
        backup_dir: Path | str = "var/paxg_lab/backups",
        max_trials: int = 30,
        startup_trials: int = 10,
        patience: int = 12,
    ):
        self.timeframe = str(timeframe).lower().strip()
        self.snapshot = snapshot
        self.job_storage = GPUJobStorage(db_path)
        self.optuna_db_path = Path(optuna_db_path)
        self.store = AdapterStore(adapter_store_dir)
        self.gatekeeper = LoRAGatekeeper(store=self.store, audit_dir=audit_dir, backup_dir=backup_dir)
        self.max_trials = max_trials
        self.startup_trials = startup_trials
        self.patience = patience

        # Check if there is a previously consumed verification range
        last_range = self.job_storage.get_latest_consumed_locked_range(self.timeframe)
        custom_test_start = None
        if last_range is not None:
            _, last_end_ms = last_range
            ts_arr = np.asarray(snapshot.timestamps)
            after_idxs = np.where(ts_arr >= last_end_ms)[0]
            if len(after_idxs) > 0:
                custom_test_start = int(after_idxs[0])

        self.split_plan = calculate_split_plan(
            total_candles=len(snapshot.features_a),
            timeframe=self.timeframe,
            custom_test_start=custom_test_start,
        )

    def run_trial_fold_evaluation(
        self,
        spec: TrainSpec,
        fold_id: int,
        base_reference_report: ScoreReport,
        custom_trainer_fn: Callable[[TrainSpec], Any] | None = None,
        custom_eval_fn: Callable[[Any, TrainSpec], ScoreReport] | None = None,
        fast_dev_mode: bool = False,
        is_cancelled_func: Callable[[], bool] | None = None,
    ) -> tuple[float, int, float]:
        """Trains and evaluates a candidate spec on a single evaluation fold.

        Returns:
            Tuple of (composite_loss, best_epoch, val_loss).
        """
        if custom_eval_fn is not None:
            report = custom_eval_fn(self.snapshot, spec)
            fm = report.get_fold_metric(fold_id)
            loss = fm.composite_loss if fm else 0.05
            return float(loss), 2, 0.05

        features_df = self.snapshot.to_dataframe(spec.feature_set)
        features = self.snapshot.get_features(spec.feature_set)
        targets = self.snapshot.features_a[:, 0]
        timestamps = self.snapshot.timestamps

        from timesfm3 import TimesFM3Torch

        fold = next((f for f in self.split_plan.eval_folds if f.fold_id == fold_id), None)
        if fold is None:
            raise ValueError(f"Unknown fold_id {fold_id}")

        if is_cancelled_func is not None and is_cancelled_func():
            raise InterruptedError(f"Trial execution on fold {fold_id} cancelled by user request.")

        base_model = TimesFM3Torch.from_pretrained(MODEL_REPO, revision=MODEL_REVISION)

        eval_spec = TrainSpec.from_dict(spec.to_dict())
        if fast_dev_mode:
            eval_spec.max_epochs = 1
            eval_spec.max_samples_per_epoch = 64

        def check_cancel(record: dict[str, Any] | None = None) -> bool:
            if is_cancelled_func is not None and is_cancelled_func():
                return False
            return True

        trainer = LoRATrainer(base_model=base_model, spec=eval_spec)
        train_res = trainer.train(
            features_df=features_df,
            snapshot_hash=self.snapshot.metadata.sha256,
            fold_id=fold.fold_id,
            progress_callback=check_cancel,
        )

        if is_cancelled_func is not None and is_cancelled_func():
            raise InterruptedError(f"Trial execution on fold {fold_id} cancelled by user request.")

        trial_adapter_id = f"trial_tmp_{self.timeframe}_f{fold.fold_id}_{int(time.time()*1000)}"
        manifest = train_res.manifest
        manifest.adapter_id = trial_adapter_id
        saved_dir = self.store.save_adapter(
            peft_model=train_res.trained_model,
            manifest=manifest,
            base_model=base_model,
            smoke_test=False,
        )

        predictor = TimesFM3Predictor(adapter_path=saved_dir)
        engine = BacktestEngine(predictor=predictor)
        base_m = base_reference_report.get_fold_metric(fold.fold_id)
        base_w_mae = base_m.weighted_mae if base_m else None
        base_w_pinball = base_m.weighted_pinball if base_m else None

        f_metric, _, _, _, _, _ = engine.evaluate_fold(
            features=features,
            targets=targets,
            timestamps=timestamps,
            timeframe=self.timeframe,
            start_idx=fold.eval_start,
            end_idx=fold.eval_end,
            context_len=spec.context_len,
            batch_size=16,
            fold_id=fold.fold_id,
            base_weighted_mae=base_w_mae,
            base_weighted_pinball=base_w_pinball,
            progress_callback=check_cancel,
        )
        composite_loss = float(f_metric.composite_loss)
        best_epoch = int(train_res.best_epoch)
        val_loss = float(train_res.best_val_loss)

        self.store.delete_adapter(trial_adapter_id, use_trash=False)
        return composite_loss, best_epoch, val_loss

    def run_trial_evaluation(
        self,
        spec: TrainSpec,
        base_reference_report: ScoreReport,
        custom_trainer_fn: Callable[[TrainSpec], Any] | None = None,
        custom_eval_fn: Callable[[Any, TrainSpec], ScoreReport] | None = None,
        fast_dev_mode: bool = False,
        is_cancelled_func: Callable[[], bool] | None = None,
    ) -> tuple[float, int, float]:
        """Trains and evaluates a candidate spec across 3 out-of-sample evaluation folds strictly without locked test.

        Per PLAN 3.3: For each trial, 3 independent adapters are trained from clean Base
        corresponding to Fold 1, Fold 2, and Fold 3 on pre-eval history.

        Returns:
            Tuple of (Score v1, median_best_epoch, avg_val_loss).
        """
        if custom_eval_fn is not None:
            report = custom_eval_fn(self.snapshot, spec)
            return float(report.score), 2, 0.05

        fold_losses = []
        best_epochs = []
        val_losses = []

        for fold in self.split_plan.eval_folds:
            if is_cancelled_func is not None and is_cancelled_func():
                raise InterruptedError("Trial execution cancelled by user request.")

            loss, b_epoch, val_loss = self.run_trial_fold_evaluation(
                spec=spec,
                fold_id=fold.fold_id,
                base_reference_report=base_reference_report,
                custom_trainer_fn=custom_trainer_fn,
                custom_eval_fn=custom_eval_fn,
                fast_dev_mode=fast_dev_mode,
                is_cancelled_func=is_cancelled_func,
            )
            fold_losses.append(loss)
            best_epochs.append(b_epoch)
            val_losses.append(val_loss)

        score_v1 = compute_score_v1(fold_losses)
        median_epoch = int(np.round(np.median(best_epochs))) if best_epochs else 2
        avg_val_loss = float(np.mean(val_losses)) if val_losses else 0.0

        return float(score_v1), median_epoch, avg_val_loss

    def run_multi_seed_verification(
        self,
        best_spec: TrainSpec,
        base_reference_report: ScoreReport,
        custom_trainer_fn: Callable[[TrainSpec], Any] | None = None,
        custom_eval_fn: Callable[[Any, TrainSpec], ScoreReport] | None = None,
        fast_dev_mode: bool = False,
        is_cancelled_func: Callable[[], bool] | None = None,
    ) -> MultiSeedEvalSummary:
        """Evaluates best hyperparameter configuration across seeds [42, 123, 2026] and takes median."""
        scores = []
        best_epochs = []

        logger.info("Executing multi-seed verification on seeds %s for %s...", SEEDS_MULTI_RUN, self.timeframe)
        for seed in SEEDS_MULTI_RUN:
            if is_cancelled_func is not None and is_cancelled_func():
                raise InterruptedError("Multi-seed verification cancelled by user request.")

            seed_spec = TrainSpec.from_dict(best_spec.to_dict())
            seed_spec.seed = seed

            score, b_epoch, _ = self.run_trial_evaluation(
                spec=seed_spec,
                base_reference_report=base_reference_report,
                custom_trainer_fn=custom_trainer_fn,
                custom_eval_fn=custom_eval_fn,
                fast_dev_mode=fast_dev_mode,
                is_cancelled_func=is_cancelled_func,
            )
            scores.append(score)
            best_epochs.append(b_epoch)
            logger.info("Seed %d result: score=%.2f, best_epoch=%d", seed, score, b_epoch)

        median_score = float(np.median(scores))
        median_epoch = int(np.round(np.median(best_epochs)))

        logger.info(
            "Multi-seed evaluation complete: median_score=%.2f, median_best_epoch=%d",
            median_score,
            median_epoch,
        )
        return MultiSeedEvalSummary(
            seeds=list(SEEDS_MULTI_RUN),
            scores=scores,
            best_epochs=best_epochs,
            median_score=median_score,
            median_best_epoch=median_epoch,
        )

    def train_final_candidate(
        self,
        candidate_spec: TrainSpec,
        best_epoch: int,
        custom_trainer_fn: Callable[[TrainSpec, int], Any] | None = None,
        fast_dev_mode: bool = False,
        is_cancelled_func: Callable[[], bool] | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[str, AdapterManifest, Path]:
        """Trains final candidate model from base on all historical data prior to locked test set.

        Returns:
            Tuple of (candidate_id, manifest, saved_adapter_path).
        """
        if is_cancelled_func is not None and is_cancelled_func():
            raise InterruptedError("Final candidate training cancelled by user request.")

        logger.info(
            "Training final candidate from base weights on all data prior to test_start (%d) with epochs=%d...",
            self.split_plan.test_start,
            best_epoch,
        )

        cand_id = f"paxg_{self.timeframe}_r{candidate_spec.lora_r}_{int(time.time())}"
        features_df = self.snapshot.to_dataframe(candidate_spec.feature_set)
        train_ceiling = self.split_plan.test_start - self.split_plan.horizon

        if custom_trainer_fn is not None:
            return custom_trainer_fn(candidate_spec, best_epoch)

        from timesfm3 import TimesFM3Torch

        base_model = TimesFM3Torch.from_pretrained(MODEL_REPO, revision=MODEL_REVISION)

        cand_spec = TrainSpec.from_dict(candidate_spec.to_dict())
        cand_spec.max_epochs = max(1, best_epoch)
        if fast_dev_mode:
            cand_spec.max_samples_per_epoch = 128

        trainer = LoRATrainer(base_model=base_model, spec=cand_spec)
        train_res = trainer.train(
            features_df=features_df,
            snapshot_hash=self.snapshot.metadata.sha256,
            fold_id="final_pre_test",
            explicit_train_range=(0, train_ceiling),
            fixed_epochs=True,
            is_cancelled_func=is_cancelled_func,
            progress_callback=progress_callback,
        )

        if is_cancelled_func is not None and is_cancelled_func():
            raise InterruptedError("Final candidate training cancelled by user request.")

        manifest = train_res.manifest
        manifest.adapter_id = cand_id
        manifest.is_verified = False

        saved_dir = self.store.save_adapter(
            peft_model=train_res.trained_model,
            manifest=manifest,
            base_model=base_model,
            smoke_test=True,
        )
        return cand_id, manifest, saved_dir

    def run_locked_verification_and_gatekeeper(
        self,
        candidate_manifest: AdapterManifest,
        candidate_adapter_path: Path,
        base_reference_report: ScoreReport,
        custom_test_eval_fn: Any = None,
        is_cancelled_func: Callable[[], bool] | None = None,
    ) -> tuple[Any, GatekeeperDecision]:
        """Executes strict single evaluation on 90-day locked test set and triggers gatekeeper winner check."""
        if is_cancelled_func is not None and is_cancelled_func():
            raise InterruptedError("Locked verification cancelled by user request.")

        logger.info(
            "Executing single locked verification on 90-day test set [%d, %d) for candidate '%s'...",
            self.split_plan.test_start,
            self.split_plan.test_end,
            candidate_manifest.adapter_id,
        )

        from timesfm3 import TimesFM3Torch

        base_model = TimesFM3Torch.from_pretrained(MODEL_REPO, revision=MODEL_REVISION)

        try:
            if custom_test_eval_fn is not None:
                locked_report = custom_test_eval_fn(self.snapshot, candidate_manifest)
            else:
                locked_report = run_locked_verification(
                    snapshot=self.snapshot,
                    candidate_manifest=candidate_manifest,
                    candidate_adapter_path=candidate_adapter_path,
                    store=self.store,
                    batch_size=16,
                    split_plan=self.split_plan,
                    storage=self.job_storage,
                    is_cancelled_func=is_cancelled_func,
                )

            if is_cancelled_func is not None and is_cancelled_func():
                raise InterruptedError("Locked verification cancelled by user request.")

            decision = self.gatekeeper.evaluate_candidate(
                candidate_manifest=candidate_manifest,
                locked_report=locked_report,
                base_model=base_model,
                perform_backup=True,
            )

            # Update locked_verification_ledger audit verdict to terminal outcome
            if self.job_storage is not None:
                reasons_str = "; ".join(decision.reasons) if decision.reasons else "All criteria passed."
                self.job_storage.update_locked_consumption_verdict(
                    timeframe=self.timeframe,
                    candidate_id=candidate_manifest.adapter_id,
                    verdict=decision.verdict,
                    details=f"Gatekeeper verdict: {decision.verdict}. Reasons: {reasons_str}",
                )

            return locked_report, decision
        except Exception as exc:
            if self.job_storage is not None:
                try:
                    self.job_storage.update_locked_consumption_verdict(
                        timeframe=self.timeframe,
                        candidate_id=candidate_manifest.adapter_id,
                        verdict="FAILED",
                        details=f"Verification failed: {exc}",
                    )
                except Exception:
                    pass
            raise

    def get_or_compute_base_report(
        self,
        is_cancelled_func: Callable[[], bool] | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> ScoreReport:
        """Retrieves cached Base reference report or computes it."""
        snap_hash = self.snapshot.metadata.sha256[:8]
        cache_path = Path("var/paxg_lab") / f"base_ref_{self.timeframe}_{snap_hash}.json"
        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                f_metrics = [FoldMetrics(**fm) if isinstance(fm, dict) else fm for fm in data.get("fold_metrics", [])]
                t_metric = FoldMetrics(**data["test_metrics"]) if data.get("test_metrics") and isinstance(data["test_metrics"], dict) else data.get("test_metrics")
                cleaned = {k: v for k, v in data.items() if k not in ("fold_metrics", "test_metrics")}
                return ScoreReport(**cleaned, fold_metrics=f_metrics, test_metrics=t_metric)
            except Exception as exc:
                logger.warning("Failed to load cached base reference report from %s: %s", cache_path, exc)

        if progress_callback:
            progress_callback({"message": "Computing base reference baseline on eval folds...", "progress_pct": 5.0})

        base_predictor = TimesFM3Predictor()
        base_engine = BacktestEngine(predictor=base_predictor)
        base_report = base_engine.run_full_backtest(
            snapshot=self.snapshot,
            feature_set="A",
            context_len=256,
            batch_size=16,
            model_name="TimesFM3-Base-Ref",
            is_base_reference=True,
            include_locked_test=False,
        )
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(base_report.to_dict(), f, indent=2)
        except Exception as exc:
            logger.warning("Failed to cache base reference report to %s: %s", cache_path, exc)
        return base_report

    def execute_step(
        self,
        max_trials: int | None = None,
        is_cancelled_func: Callable[[], bool] | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        fast_dev_mode: bool = False,
        custom_trainer_fn: Callable[[TrainSpec], Any] | None = None,
        custom_eval_fn: Callable[[Any, TrainSpec], ScoreReport] | None = None,
        custom_test_eval_fn: Any = None,
    ) -> dict[str, Any]:
        """Executes a single bounded step of the autonomous tuning cycle."""
        limit_trials = max_trials or self.max_trials
        snap_hash = str(self.snapshot.metadata.sha256)
        snap_path = str(getattr(self.snapshot, "source_path", ""))

        state = self.job_storage.get_auto_tune_run(self.timeframe)
        if state is None or state.get("snapshot_hash") != snap_hash:
            self.job_storage.save_auto_tune_run(
                timeframe=self.timeframe,
                snapshot_path=snap_path,
                snapshot_hash=snap_hash,
                phase="BASELINE",
                current_trial=0,
                max_trials=limit_trials,
            )
            state = self.job_storage.get_auto_tune_run(self.timeframe)
            if self.job_storage.get_auto_run_state(self.timeframe) != AutoRunState.STOPPED:
                self.job_storage.set_auto_run_state(self.timeframe, AutoRunState.SEARCHING, allow_unstop=False)

        phase = state.get("phase", "BASELINE")
        current_trial = int(state.get("current_trial", 0))

        if self.job_storage.get_auto_run_state(self.timeframe) == AutoRunState.STOPPED:
            raise InterruptedError("Auto-run is STOPPED by user.")

        if is_cancelled_func is not None and is_cancelled_func():
            raise InterruptedError(f"Step execution in phase '{phase}' cancelled by user request.")

        # Phase 1: BASELINE
        if phase == "BASELINE":
            logger.info("Executing step: BASELINE for %s", self.timeframe)
            if self.job_storage.get_auto_run_state(self.timeframe) == AutoRunState.STOPPED:
                raise InterruptedError("Auto-run is STOPPED by user.")
            if self.job_storage.get_auto_run_state(self.timeframe) != AutoRunState.STOPPED:
                self.job_storage.set_auto_run_state(self.timeframe, AutoRunState.SEARCHING, allow_unstop=False)
            if custom_eval_fn is not None:
                try:
                    self.get_or_compute_base_report(is_cancelled_func=is_cancelled_func, progress_callback=progress_callback)
                except Exception:
                    pass
            else:
                self.get_or_compute_base_report(is_cancelled_func=is_cancelled_func, progress_callback=progress_callback)

            step_ms = 3600 * 1000 if self.timeframe == "1h" else 4 * 3600 * 1000
            test_start_ms = int(self.snapshot.timestamps[self.split_plan.test_start])
            if self.split_plan.test_end < len(self.snapshot.timestamps):
                test_end_ms = int(self.snapshot.timestamps[self.split_plan.test_end])
            else:
                test_end_ms = int(self.snapshot.timestamps[self.split_plan.test_end - 1]) + step_ms

            test_range_info = {
                "test_start_idx": self.split_plan.test_start,
                "test_end_idx": self.split_plan.test_end,
                "test_start_time_ms": test_start_ms,
                "test_end_time_ms": test_end_ms,
            }

            self.job_storage.save_auto_tune_run(
                timeframe=self.timeframe,
                snapshot_path=snap_path,
                snapshot_hash=snap_hash,
                phase="TRIAL",
                current_trial=0,
                max_trials=limit_trials,
                current_fold=1,
                selected_test_range_json=json.dumps(test_range_info),
            )
            return self.job_storage.get_auto_tune_run(self.timeframe) or {}

        # Phase 2: TRIAL
        elif phase == "TRIAL":
            if self.job_storage.get_auto_run_state(self.timeframe) == AutoRunState.STOPPED:
                raise InterruptedError("Auto-run is STOPPED by user.")
            if self.job_storage.get_auto_run_state(self.timeframe) != AutoRunState.STOPPED:
                self.job_storage.set_auto_run_state(self.timeframe, AutoRunState.SEARCHING, allow_unstop=False)

            if custom_eval_fn is not None:
                try:
                    base_report = self.get_or_compute_base_report(is_cancelled_func=is_cancelled_func, progress_callback=progress_callback)
                except Exception:
                    base_report = custom_eval_fn(self.snapshot, None)
            else:
                base_report = self.get_or_compute_base_report(is_cancelled_func=is_cancelled_func, progress_callback=progress_callback)
            optimizer = OptunaTPEOptimizer(
                timeframe=self.timeframe,
                snapshot=self.snapshot,
                db_path=self.optuna_db_path,
                max_trials=limit_trials,
                startup_trials=min(self.startup_trials, max(2, limit_trials // 3)),
                patience=self.patience,
                seed=42,
            )

            study = optimizer.create_or_load_study()
            from .optimizer import EarlyStoppingStagnationCallback
            early_stop_cb = EarlyStoppingStagnationCallback(
                patience=self.patience,
                min_delta=optimizer.min_delta,
                startup_trials=optimizer.startup_trials,
            )
            early_stop_cb.sync_from_study(study)
            completed_trials = len([t for t in study.trials if t.state == TrialState.COMPLETE])

            # Check if search phase is already completed
            if early_stop_cb.stopped_early or completed_trials >= limit_trials:
                top_specs = get_top_unique_specs(study, timeframe=self.timeframe, top_n=3)
                best_t = study.best_trial if len(study.trials) > 0 else None
                best_val = study.best_value if len(study.trials) > 0 else None
                best_num = best_t.number if best_t else None
                best_spec = top_specs[0] if top_specs else suggest_trial_spec(best_t, timeframe=self.timeframe, seed=42)
                self.job_storage.save_auto_tune_run(
                    timeframe=self.timeframe,
                    snapshot_path=snap_path,
                    snapshot_hash=snap_hash,
                    phase="MULTI_SEED",
                    current_trial=completed_trials,
                    max_trials=limit_trials,
                    best_trial_num=best_num,
                    best_score=best_val,
                    best_spec_json=json.dumps(best_spec.to_dict()),
                    top_specs_json=json.dumps([s.to_dict() for s in top_specs]),
                    current_fold=1,
                    intermediate_fold_results_json=None,
                    multi_seed_config_idx=0,
                    multi_seed_seed_idx=0,
                    multi_seed_fold_idx=1,
                    multi_seed_evaluations_json=json.dumps([]),
                )
                if self.job_storage.get_auto_run_state(self.timeframe) != AutoRunState.STOPPED:
                    self.job_storage.set_auto_run_state(self.timeframe, AutoRunState.VALIDATING, allow_unstop=False)
                return self.job_storage.get_auto_tune_run(self.timeframe) or {}

            current_fold = int(state.get("current_fold") or 1)
            inter_data_raw = state.get("intermediate_fold_results_json")
            inter_data = json.loads(inter_data_raw) if inter_data_raw else None

            if inter_data is None or inter_data.get("spec") is None:
                trial = study.ask()
                trial_num = trial.number
                spec = suggest_trial_spec(trial, timeframe=self.timeframe, seed=optimizer.seed)
                inter_data = {
                    "trial_number": trial_num,
                    "spec": spec.to_dict(),
                    "fold_losses": [],
                    "best_epochs": [],
                    "val_losses": [],
                }
                current_fold = 1
            else:
                spec = TrainSpec.from_dict(inter_data["spec"])

            if is_cancelled_func is not None and is_cancelled_func():
                raise InterruptedError(f"Step execution cancelled by user request before fold {current_fold}.")

            if progress_callback:
                pct = 10.0 + (completed_trials / limit_trials) * 50.0 + (current_fold / 3.0) * (50.0 / limit_trials)
                progress_callback({
                    "message": f"Evaluating Trial #{inter_data['trial_number']}/{limit_trials}, Fold {current_fold}/3...",
                    "progress_pct": min(pct, 60.0),
                })

            # Execute single bounded fold evaluation
            loss, b_epoch, val_loss = self.run_trial_fold_evaluation(
                spec=spec,
                fold_id=current_fold,
                base_reference_report=base_report,
                custom_trainer_fn=custom_trainer_fn,
                custom_eval_fn=custom_eval_fn,
                fast_dev_mode=fast_dev_mode,
                is_cancelled_func=is_cancelled_func,
            )
            inter_data["fold_losses"].append(loss)
            inter_data["best_epochs"].append(b_epoch)
            inter_data["val_losses"].append(val_loss)

            if is_cancelled_func is not None and is_cancelled_func():
                raise InterruptedError(f"Step execution cancelled by user request after fold {current_fold}.")

            if current_fold < 3:
                # Advance to next fold for next bounded job
                self.job_storage.save_auto_tune_run(
                    timeframe=self.timeframe,
                    snapshot_path=snap_path,
                    snapshot_hash=snap_hash,
                    phase="TRIAL",
                    current_trial=completed_trials,
                    max_trials=limit_trials,
                    current_fold=current_fold + 1,
                    intermediate_fold_results_json=json.dumps(inter_data),
                )
            else:
                # Fold 3 complete -> finish trial in Optuna
                score_v1 = float(compute_score_v1(inter_data["fold_losses"]))
                t_num = int(inter_data["trial_number"])
                trial = [t for t in study.trials if t.number == t_num][0]
                study.tell(trial, score_v1)
                trial.set_user_attr("timeframe", self.timeframe)
                trial.set_user_attr("snapshot_sha256", self.snapshot.metadata.sha256)
                trial.set_user_attr("score_v1", score_v1)
                early_stop_cb.sync_from_study(study)
                new_completed = len([t for t in study.trials if t.state == TrialState.COMPLETE])

                is_finished = (new_completed >= limit_trials) or early_stop_cb.stopped_early
                if is_finished:
                    top_specs = get_top_unique_specs(study, timeframe=self.timeframe, top_n=3)
                    best_t = study.best_trial
                    best_val = study.best_value
                    best_num = best_t.number if best_t else None
                    best_spec = top_specs[0] if top_specs else suggest_trial_spec(best_t, timeframe=self.timeframe, seed=42)
                    self.job_storage.save_auto_tune_run(
                        timeframe=self.timeframe,
                        snapshot_path=snap_path,
                        snapshot_hash=snap_hash,
                        phase="MULTI_SEED",
                        current_trial=new_completed,
                        max_trials=limit_trials,
                        best_trial_num=best_num,
                        best_score=best_val,
                        best_spec_json=json.dumps(best_spec.to_dict()),
                        top_specs_json=json.dumps([s.to_dict() for s in top_specs]),
                        current_fold=1,
                        intermediate_fold_results_json=None,
                        multi_seed_config_idx=0,
                        multi_seed_seed_idx=0,
                        multi_seed_fold_idx=1,
                        multi_seed_evaluations_json=json.dumps([]),
                    )
                    if self.job_storage.get_auto_run_state(self.timeframe) != AutoRunState.STOPPED:
                        self.job_storage.set_auto_run_state(self.timeframe, AutoRunState.VALIDATING, allow_unstop=False)
                else:
                    best_val = study.best_value if len(study.trials) > 0 else None
                    best_num = study.best_trial.number if len(study.trials) > 0 else None
                    self.job_storage.save_auto_tune_run(
                        timeframe=self.timeframe,
                        snapshot_path=snap_path,
                        snapshot_hash=snap_hash,
                        phase="TRIAL",
                        current_trial=new_completed,
                        max_trials=limit_trials,
                        best_trial_num=best_num,
                        best_score=best_val,
                        current_fold=1,
                        intermediate_fold_results_json=None,
                    )
            return self.job_storage.get_auto_tune_run(self.timeframe) or {}

        # Phase 3: MULTI_SEED
        elif phase == "MULTI_SEED":
            logger.info("Executing step: MULTI_SEED for %s", self.timeframe)
            if self.job_storage.get_auto_run_state(self.timeframe) == AutoRunState.STOPPED:
                raise InterruptedError("Auto-run is STOPPED by user.")
            if self.job_storage.get_auto_run_state(self.timeframe) != AutoRunState.STOPPED:
                self.job_storage.set_auto_run_state(self.timeframe, AutoRunState.VALIDATING, allow_unstop=False)

            if custom_eval_fn is not None:
                try:
                    base_report = self.get_or_compute_base_report(is_cancelled_func=is_cancelled_func, progress_callback=progress_callback)
                except Exception:
                    base_report = custom_eval_fn(self.snapshot, None)
            else:
                base_report = self.get_or_compute_base_report(is_cancelled_func=is_cancelled_func, progress_callback=progress_callback)

            # Load top unique specs
            top_specs_raw = state.get("top_specs_json")
            if top_specs_raw:
                top_specs = [TrainSpec.from_dict(d) for d in json.loads(top_specs_raw)]
            else:
                best_dict = json.loads(state.get("best_spec_json") or "{}")
                top_specs = [TrainSpec.from_dict(best_dict)] if best_dict else []

            if not top_specs:
                top_specs = [TrainSpec(timeframe=self.timeframe, horizon=get_horizon_for_timeframe(self.timeframe))]

            config_idx = int(state.get("multi_seed_config_idx") or 0)
            seed_idx = int(state.get("multi_seed_seed_idx") or 0)
            fold_idx = int(state.get("multi_seed_fold_idx") or 1)

            evals_raw = state.get("multi_seed_evaluations_json")
            evals = json.loads(evals_raw) if evals_raw else []
            if not evals or len(evals) != len(top_specs):
                evals = [
                    {
                        "config_idx": i,
                        "seed_scores": {},
                        "seed_best_epochs": {},
                        "current_fold_losses": [],
                        "current_best_epochs": [],
                        "current_val_losses": [],
                    }
                    for i in range(len(top_specs))
                ]

            current_spec = top_specs[config_idx]
            current_seed = SEEDS_MULTI_RUN[seed_idx]
            seed_spec = TrainSpec.from_dict(current_spec.to_dict())
            seed_spec.seed = current_seed

            if is_cancelled_func is not None and is_cancelled_func():
                raise InterruptedError(f"Multi-seed step cancelled before config {config_idx+1}, seed {current_seed}, fold {fold_idx}.")

            if progress_callback:
                progress_callback({
                    "message": f"Multi-seed verification: Config {config_idx+1}/{len(top_specs)}, Seed {current_seed}, Fold {fold_idx}/3...",
                    "progress_pct": 65.0 + (config_idx * 3 + seed_idx) * 2.0,
                })

            # Execute single fold
            loss, b_epoch, val_loss = self.run_trial_fold_evaluation(
                spec=seed_spec,
                fold_id=fold_idx,
                base_reference_report=base_report,
                custom_trainer_fn=custom_trainer_fn,
                custom_eval_fn=custom_eval_fn,
                fast_dev_mode=fast_dev_mode,
                is_cancelled_func=is_cancelled_func,
            )
            evals[config_idx]["current_fold_losses"].append(loss)
            evals[config_idx]["current_best_epochs"].append(b_epoch)
            evals[config_idx]["current_val_losses"].append(val_loss)

            if is_cancelled_func is not None and is_cancelled_func():
                raise InterruptedError(f"Multi-seed step cancelled after config {config_idx+1}, seed {current_seed}, fold {fold_idx}.")

            if fold_idx < 3:
                # Next fold of same seed
                self.job_storage.save_auto_tune_run(
                    timeframe=self.timeframe,
                    snapshot_path=snap_path,
                    snapshot_hash=snap_hash,
                    phase="MULTI_SEED",
                    current_trial=current_trial,
                    max_trials=limit_trials,
                    multi_seed_config_idx=config_idx,
                    multi_seed_seed_idx=seed_idx,
                    multi_seed_fold_idx=fold_idx + 1,
                    multi_seed_evaluations_json=json.dumps(evals),
                )
            else:
                # All 3 folds done for current_seed
                seed_score = float(compute_score_v1(evals[config_idx]["current_fold_losses"]))
                seed_epoch = int(np.round(np.median(evals[config_idx]["current_best_epochs"])))
                s_key = str(current_seed)
                evals[config_idx]["seed_scores"][s_key] = seed_score
                evals[config_idx]["seed_best_epochs"][s_key] = seed_epoch
                evals[config_idx]["current_fold_losses"] = []
                evals[config_idx]["current_best_epochs"] = []
                evals[config_idx]["current_val_losses"] = []

                if seed_idx < len(SEEDS_MULTI_RUN) - 1:
                    # Advance to next seed for this config
                    self.job_storage.save_auto_tune_run(
                        timeframe=self.timeframe,
                        snapshot_path=snap_path,
                        snapshot_hash=snap_hash,
                        phase="MULTI_SEED",
                        current_trial=current_trial,
                        max_trials=limit_trials,
                        multi_seed_config_idx=config_idx,
                        multi_seed_seed_idx=seed_idx + 1,
                        multi_seed_fold_idx=1,
                        multi_seed_evaluations_json=json.dumps(evals),
                    )
                elif config_idx < len(top_specs) - 1:
                    # Advance to next config
                    self.job_storage.save_auto_tune_run(
                        timeframe=self.timeframe,
                        snapshot_path=snap_path,
                        snapshot_hash=snap_hash,
                        phase="MULTI_SEED",
                        current_trial=current_trial,
                        max_trials=limit_trials,
                        multi_seed_config_idx=config_idx + 1,
                        multi_seed_seed_idx=0,
                        multi_seed_fold_idx=1,
                        multi_seed_evaluations_json=json.dumps(evals),
                    )
                else:
                    # All configs and all seeds evaluated!
                    ranked_configs = []
                    for idx, entry in enumerate(evals):
                        scores = [float(entry["seed_scores"].get(str(s), 0.0)) for s in SEEDS_MULTI_RUN]
                        epochs = [int(entry["seed_best_epochs"].get(str(s), 2)) for s in SEEDS_MULTI_RUN]
                        med_score = float(np.median(scores))
                        med_epoch = int(np.round(np.median(epochs)))
                        min_score = float(np.min(scores))
                        ranked_configs.append({
                            "config_idx": idx,
                            "spec": top_specs[idx],
                            "median_score": med_score,
                            "median_epoch": med_epoch,
                            "min_score": min_score,
                            "scores": scores,
                            "best_epochs": epochs,
                        })

                    # Rank by median_score desc, min_score desc, config_idx asc
                    ranked_configs.sort(key=lambda c: (c["median_score"], c["min_score"], -c["config_idx"]), reverse=True)
                    winner = ranked_configs[0]
                    best_spec = winner["spec"]
                    best_epoch = winner["median_epoch"]

                    multi_seed_summary = MultiSeedEvalSummary(
                        seeds=list(SEEDS_MULTI_RUN),
                        scores=winner["scores"],
                        best_epochs=winner["best_epochs"],
                        median_score=winner["median_score"],
                        median_best_epoch=best_epoch,
                    )
                    logger.info(
                        "Multi-seed completed across %d configs. Selected winner config #%d with median_score=%.2f, median_epoch=%d",
                        len(top_specs), winner["config_idx"] + 1, winner["median_score"], best_epoch,
                    )

                    self.job_storage.save_auto_tune_run(
                        timeframe=self.timeframe,
                        snapshot_path=snap_path,
                        snapshot_hash=snap_hash,
                        phase="FINAL_FIT",
                        current_trial=current_trial,
                        max_trials=limit_trials,
                        best_spec_json=json.dumps(best_spec.to_dict()),
                        best_epoch=best_epoch,
                        best_score=winner["median_score"],
                        multi_seed_results_json=json.dumps(multi_seed_summary.to_dict()),
                    )
            return self.job_storage.get_auto_tune_run(self.timeframe) or {}

        # Phase 4: FINAL_FIT
        elif phase == "FINAL_FIT":
            logger.info("Executing step: FINAL_FIT for %s", self.timeframe)
            if self.job_storage.get_auto_run_state(self.timeframe) == AutoRunState.STOPPED:
                raise InterruptedError("Auto-run is STOPPED by user.")
            if self.job_storage.get_auto_run_state(self.timeframe) != AutoRunState.STOPPED:
                self.job_storage.set_auto_run_state(self.timeframe, AutoRunState.VALIDATING, allow_unstop=False)
            if is_cancelled_func is not None and is_cancelled_func():
                raise InterruptedError(f"Step execution in phase '{phase}' cancelled by user request.")

            best_spec_dict = json.loads(state.get("best_spec_json") or "{}")
            best_spec = TrainSpec.from_dict(best_spec_dict)
            best_epoch = int(state.get("best_epoch") or 2)

            if progress_callback:
                progress_callback({"message": "Training final candidate from Base model on full pre-test history...", "progress_pct": 80.0})

            cand_id, cand_manifest, cand_path = self.train_final_candidate(
                candidate_spec=best_spec,
                best_epoch=best_epoch,
                custom_trainer_fn=custom_trainer_fn,
                fast_dev_mode=fast_dev_mode,
                is_cancelled_func=is_cancelled_func,
                progress_callback=progress_callback,
            )

            if is_cancelled_func is not None and is_cancelled_func():
                raise InterruptedError(f"Step execution in phase '{phase}' cancelled by user request.")

            self.job_storage.save_auto_tune_run(
                timeframe=self.timeframe,
                snapshot_path=snap_path,
                snapshot_hash=snap_hash,
                phase="LOCKED_VERIFICATION",
                current_trial=current_trial,
                final_candidate_id=cand_id,
                final_candidate_path=str(cand_path),
            )
            return self.job_storage.get_auto_tune_run(self.timeframe) or {}

        # Phase 5: LOCKED_VERIFICATION
        elif phase == "LOCKED_VERIFICATION":
            logger.info("Executing step: LOCKED_VERIFICATION for %s", self.timeframe)
            if self.job_storage.get_auto_run_state(self.timeframe) == AutoRunState.STOPPED:
                raise InterruptedError("Auto-run is STOPPED by user.")
            if self.job_storage.get_auto_run_state(self.timeframe) != AutoRunState.STOPPED:
                self.job_storage.set_auto_run_state(self.timeframe, AutoRunState.VALIDATING, allow_unstop=False)
            if is_cancelled_func is not None and is_cancelled_func():
                raise InterruptedError(f"Step execution in phase '{phase}' cancelled by user request.")

            if custom_eval_fn is not None:
                try:
                    base_report = self.get_or_compute_base_report(is_cancelled_func=is_cancelled_func, progress_callback=progress_callback)
                except Exception:
                    base_report = custom_eval_fn(self.snapshot, None)
            else:
                base_report = self.get_or_compute_base_report(is_cancelled_func=is_cancelled_func, progress_callback=progress_callback)
            cand_id = state.get("final_candidate_id")
            cand_path = Path(state.get("final_candidate_path") or "")
            cand_manifest = AdapterManifest.load_json(cand_path / "paxg_manifest.json")

            if progress_callback:
                progress_callback({"message": "Executing single-pass locked test verification...", "progress_pct": 90.0})

            locked_report, decision = self.run_locked_verification_and_gatekeeper(
                candidate_manifest=cand_manifest,
                candidate_adapter_path=cand_path,
                base_reference_report=base_report,
                custom_test_eval_fn=custom_test_eval_fn,
                is_cancelled_func=is_cancelled_func,
            )

            if is_cancelled_func is not None and is_cancelled_func():
                raise InterruptedError(f"Step execution in phase '{phase}' cancelled by user request.")

            # Complete cycle -> transition to WAITING_DATA
            self.job_storage.set_auto_run_state(self.timeframe, AutoRunState.WAITING_DATA, allow_unstop=False)
            self.job_storage.save_auto_tune_run(
                timeframe=self.timeframe,
                snapshot_path=snap_path,
                snapshot_hash=snap_hash,
                phase="WAITING_DATA",
                current_trial=current_trial,
                last_consumed_candles=len(self.snapshot.features_a),
                last_run_completed_at=time.time(),
            )
            logger.info("Auto tune cycle completed with decision: %s. Transitioned to WAITING_DATA.", decision.verdict)
            return self.job_storage.get_auto_tune_run(self.timeframe) or {}

        elif phase == "WAITING_DATA":
            logger.info("Auto tune is currently in WAITING_DATA for %s.", self.timeframe)
            return state

        else:
            raise ValueError(f"Unknown phase: {phase}")

    def run_tuning_cycle(
        self,
        max_trials: int | None = None,
        is_cancelled_func: Callable[[], bool] | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        fast_dev_mode: bool = False,
        custom_trainer_fn: Callable[[TrainSpec], Any] | None = None,
        custom_eval_fn: Callable[[Any, TrainSpec], ScoreReport] | None = None,
        custom_test_eval_fn: Any = None,
    ) -> P6RunResult:
        """Executes a full autonomous tuning cycle by looping execute_step until WAITING_DATA."""
        limit_trials = max_trials or self.max_trials
        logger.info("Initiating full autonomous tuning cycle loop for %s (max_trials=%d)...", self.timeframe, limit_trials)

        # Reset any prior run state for fresh cycle
        self.job_storage.reset_auto_tune_run(self.timeframe)
        self.job_storage.set_auto_run_state(self.timeframe, AutoRunState.SEARCHING)

        while True:
            if is_cancelled_func is not None and is_cancelled_func():
                raise InterruptedError("Tuning cycle cancelled by user request.")
            st = self.execute_step(
                max_trials=limit_trials,
                is_cancelled_func=is_cancelled_func,
                progress_callback=progress_callback,
                fast_dev_mode=fast_dev_mode,
                custom_trainer_fn=custom_trainer_fn,
                custom_eval_fn=custom_eval_fn,
                custom_test_eval_fn=custom_test_eval_fn,
            )
            if st.get("phase") == "WAITING_DATA":
                break

        # Load Optuna study results
        optimizer = OptunaTPEOptimizer(
            timeframe=self.timeframe,
            snapshot=self.snapshot,
            db_path=self.optuna_db_path,
            max_trials=limit_trials,
        )
        study = optimizer.create_or_load_study()
        best_t = study.best_trial

        final_st = self.job_storage.get_auto_tune_run(self.timeframe) or {}
        multi_seed_data = json.loads(final_st.get("multi_seed_results_json") or "{}")

        # Load audit decision if exists
        audit_dir = Path("var/paxg_lab/audit_reports")
        audits = sorted(audit_dir.glob(f"audit_{self.timeframe}_*"))
        dec_data = {}
        if audits:
            try:
                with open(audits[-1], "r", encoding="utf-8") as af:
                    dec_data = json.load(af)
            except Exception:
                pass

        return P6RunResult(
            timeframe=self.timeframe,
            snapshot_id=self.snapshot.metadata.snapshot_id,
            study_name=study.study_name,
            total_trials=len(study.trials),
            best_trial_number=best_t.number if best_t else 0,
            best_trial_params=best_t.params if best_t else {},
            multi_seed_summary=multi_seed_data,
            final_candidate_id=final_st.get("final_candidate_id", ""),
            test_score_report={},
            decision=dec_data,
            resulting_auto_state=AutoRunState.WAITING_DATA.value,
        )
