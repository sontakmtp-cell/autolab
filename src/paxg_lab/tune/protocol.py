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

        self.split_plan = calculate_split_plan(
            total_candles=len(snapshot.features_a),
            timeframe=self.timeframe,
        )

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

        features_df = self.snapshot.to_dataframe(spec.feature_set)
        features = self.snapshot.get_features(spec.feature_set)
        targets = self.snapshot.features_a[:, 0]
        timestamps = self.snapshot.timestamps

        from timesfm3 import TimesFM3Torch

        fold_losses = []
        best_epochs = []
        val_losses = []

        def check_cancel(record: dict[str, Any] | None = None) -> bool:
            if is_cancelled_func is not None and is_cancelled_func():
                return False
            return True

        for fold in self.split_plan.eval_folds:
            if is_cancelled_func is not None and is_cancelled_func():
                raise InterruptedError("Trial execution cancelled by user request.")

            base_model = TimesFM3Torch.from_pretrained(MODEL_REPO, revision=MODEL_REVISION)

            eval_spec = TrainSpec.from_dict(spec.to_dict())
            if fast_dev_mode:
                eval_spec.max_epochs = 1
                eval_spec.max_samples_per_epoch = 64

            trainer = LoRATrainer(base_model=base_model, spec=eval_spec)
            train_res = trainer.train(
                features_df=features_df,
                snapshot_hash=self.snapshot.metadata.sha256,
                fold_id=fold.fold_id,
                progress_callback=check_cancel,
            )

            if is_cancelled_func is not None and is_cancelled_func():
                raise InterruptedError("Trial execution cancelled by user request.")

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
            fold_losses.append(f_metric.composite_loss)
            best_epochs.append(train_res.best_epoch)
            val_losses.append(train_res.best_val_loss)

            self.store.delete_adapter(trial_adapter_id, use_trash=False)

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
    ) -> tuple[str, AdapterManifest, Path]:
        """Trains final candidate model from base on all historical data prior to locked test set.

        Returns:
            Tuple of (candidate_id, manifest, saved_adapter_path).
        """
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
        )

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
    ) -> tuple[Any, GatekeeperDecision]:
        """Executes strict single evaluation on 90-day locked test set and triggers gatekeeper winner check."""
        logger.info(
            "Executing single locked verification on 90-day test set [%d, %d) for candidate '%s'...",
            self.split_plan.test_start,
            self.split_plan.test_end,
            candidate_manifest.adapter_id,
        )

        from timesfm3 import TimesFM3Torch

        base_model = TimesFM3Torch.from_pretrained(MODEL_REPO, revision=MODEL_REVISION)

        if custom_test_eval_fn is not None:
            locked_report = custom_test_eval_fn(self.snapshot, candidate_manifest)
        else:
            locked_report = run_locked_verification(
                snapshot=self.snapshot,
                candidate_manifest=candidate_manifest,
                candidate_adapter_path=candidate_adapter_path,
                store=self.store,
                batch_size=16,
                storage=self.job_storage,
            )

        decision = self.gatekeeper.evaluate_candidate(
            candidate_manifest=candidate_manifest,
            locked_report=locked_report,
            base_model=base_model,
            perform_backup=True,
        )

        return locked_report, decision

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
                self.job_storage.set_auto_run_state(self.timeframe, AutoRunState.SEARCHING)

        phase = state.get("phase", "BASELINE")
        current_trial = int(state.get("current_trial", 0))

        if is_cancelled_func is not None and is_cancelled_func():
            raise InterruptedError(f"Step execution in phase '{phase}' cancelled by user request.")

        # Phase 1: BASELINE
        if phase == "BASELINE":
            logger.info("Executing step: BASELINE for %s", self.timeframe)
            if self.job_storage.get_auto_run_state(self.timeframe) != AutoRunState.STOPPED:
                self.job_storage.set_auto_run_state(self.timeframe, AutoRunState.SEARCHING)
            self.get_or_compute_base_report(is_cancelled_func=is_cancelled_func, progress_callback=progress_callback)
            self.job_storage.save_auto_tune_run(
                timeframe=self.timeframe,
                snapshot_path=snap_path,
                snapshot_hash=snap_hash,
                phase="TRIAL",
                current_trial=0,
                max_trials=limit_trials,
            )
            return self.job_storage.get_auto_tune_run(self.timeframe) or {}

        # Phase 2: TRIAL
        elif phase == "TRIAL":
            if self.job_storage.get_auto_run_state(self.timeframe) != AutoRunState.STOPPED:
                self.job_storage.set_auto_run_state(self.timeframe, AutoRunState.SEARCHING)
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

            def eval_trial(spec: TrainSpec, trial: optuna.Trial) -> float:
                if is_cancelled_func is not None and is_cancelled_func():
                    raise InterruptedError(f"Trial #{trial.number} cancelled by user request.")
                if progress_callback:
                    pct = 10.0 + (trial.number / limit_trials) * 50.0
                    progress_callback({
                        "message": f"Evaluating Trial #{trial.number}/{limit_trials} across 3 eval folds...",
                        "progress_pct": min(pct, 60.0),
                    })
                score, _, _ = self.run_trial_evaluation(
                    spec=spec,
                    base_reference_report=base_report,
                    fast_dev_mode=fast_dev_mode,
                    is_cancelled_func=is_cancelled_func,
                )
                return score

            def objective(trial: optuna.Trial) -> float:
                spec = suggest_trial_spec(trial, timeframe=self.timeframe, seed=optimizer.seed)
                trial.set_user_attr("timeframe", self.timeframe)
                trial.set_user_attr("snapshot_sha256", self.snapshot.metadata.sha256)
                trial.set_user_attr("context_len", spec.context_len)
                trial.set_user_attr("lora_r", spec.lora_r)
                trial.set_user_attr("lora_alpha", spec.lora_alpha)
                trial.set_user_attr("learning_rate", spec.learning_rate)
                trial.set_user_attr("lora_dropout", spec.lora_dropout)
                trial.set_user_attr("weight_decay", spec.weight_decay)
                trial.set_user_attr("feature_set", spec.feature_set)
                trial.set_user_attr("history_days", str(spec.history_days))
                trial.set_user_attr("extended_targets", spec.extended_targets)
                score = eval_trial(spec, trial)
                trial.set_user_attr("score_v1", score)
                return score

            # Execute exactly 1 trial in this bounded step
            study.optimize(objective, n_trials=1, callbacks=[early_stop_cb])
            completed_trials = len([t for t in study.trials if t.state == TrialState.COMPLETE])
            logger.info("Completed trial step for %s: %d completed trials so far.", self.timeframe, completed_trials)

            # Check if search is complete
            is_finished = (completed_trials >= limit_trials) or early_stop_cb.stopped_early
            if is_finished and len(study.trials) > 0:
                best_t = study.best_trial
                best_spec = suggest_trial_spec(best_t, timeframe=self.timeframe, seed=42)
                self.job_storage.save_auto_tune_run(
                    timeframe=self.timeframe,
                    snapshot_path=snap_path,
                    snapshot_hash=snap_hash,
                    phase="MULTI_SEED",
                    current_trial=completed_trials,
                    max_trials=limit_trials,
                    best_trial_num=best_t.number,
                    best_score=best_t.value,
                    best_spec_json=json.dumps(best_spec.to_dict()),
                )
                self.job_storage.set_auto_run_state(self.timeframe, AutoRunState.VALIDATING)
            else:
                best_val = study.best_value if len(study.trials) > 0 else None
                best_num = study.best_trial.number if len(study.trials) > 0 else None
                self.job_storage.save_auto_tune_run(
                    timeframe=self.timeframe,
                    snapshot_path=snap_path,
                    snapshot_hash=snap_hash,
                    phase="TRIAL",
                    current_trial=completed_trials,
                    max_trials=limit_trials,
                    best_trial_num=best_num,
                    best_score=best_val,
                )
            return self.job_storage.get_auto_tune_run(self.timeframe) or {}

        # Phase 3: MULTI_SEED
        elif phase == "MULTI_SEED":
            logger.info("Executing step: MULTI_SEED for %s", self.timeframe)
            self.job_storage.set_auto_run_state(self.timeframe, AutoRunState.VALIDATING)
            base_report = self.get_or_compute_base_report(is_cancelled_func=is_cancelled_func, progress_callback=progress_callback)
            best_spec_dict = json.loads(state.get("best_spec_json") or "{}")
            best_spec = TrainSpec.from_dict(best_spec_dict)

            if progress_callback:
                progress_callback({"message": "Running multi-seed verification across seeds [42, 123, 2026]...", "progress_pct": 65.0})

            multi_seed_res = self.run_multi_seed_verification(
                best_spec=best_spec,
                base_reference_report=base_report,
                fast_dev_mode=fast_dev_mode,
                is_cancelled_func=is_cancelled_func,
            )

            self.job_storage.save_auto_tune_run(
                timeframe=self.timeframe,
                snapshot_path=snap_path,
                snapshot_hash=snap_hash,
                phase="FINAL_FIT",
                current_trial=current_trial,
                best_epoch=multi_seed_res.median_best_epoch,
                multi_seed_results_json=json.dumps(multi_seed_res.to_dict()),
            )
            return self.job_storage.get_auto_tune_run(self.timeframe) or {}

        # Phase 4: FINAL_FIT
        elif phase == "FINAL_FIT":
            logger.info("Executing step: FINAL_FIT for %s", self.timeframe)
            self.job_storage.set_auto_run_state(self.timeframe, AutoRunState.VALIDATING)
            best_spec_dict = json.loads(state.get("best_spec_json") or "{}")
            best_spec = TrainSpec.from_dict(best_spec_dict)
            best_epoch = int(state.get("best_epoch") or 2)

            if progress_callback:
                progress_callback({"message": "Training final candidate from Base model on full pre-test history...", "progress_pct": 80.0})

            cand_id, cand_manifest, cand_path = self.train_final_candidate(
                candidate_spec=best_spec,
                best_epoch=best_epoch,
                fast_dev_mode=fast_dev_mode,
            )

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
            self.job_storage.set_auto_run_state(self.timeframe, AutoRunState.VALIDATING)
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
            )

            # Complete cycle -> transition to WAITING_DATA
            self.job_storage.set_auto_run_state(self.timeframe, AutoRunState.WAITING_DATA)
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
