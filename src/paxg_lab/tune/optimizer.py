"""Optuna TPE Study runner and Bayesian hyperparameter optimizer.

Conforms strictly to PAXG Forecast Lab PLAN.md Section 3.6 & P6.md:
- Tree-structured Parzen Estimator (TPE) algorithm.
- Max 30 trials per study batch.
- 10 initial startup trials for exploration.
- Early stopping: stops after 12 consecutive trials without >= 0.5 score improvement.
- Isolated SQLite database (default: var/paxg_lab/optuna_studies.db).
- Study name bound to timeframe and snapshot hash (no cross-snapshot score mixing).
- Evaluates candidate models on 3 out-of-sample folds without touching locked test set.
- Hardware and resource errors recorded distinctly; no fake numerical scores assigned.
"""

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Any, Callable

import optuna
from optuna.samplers import TPESampler
from optuna.trial import TrialState

from ..data.snapshot import DatasetSnapshot
from ..eval.engine import BacktestEngine
from ..eval.types import ScoreReport
from ..model.train_spec import TrainSpec
from .space import suggest_trial_spec, validate_train_spec_invariants

logger = logging.getLogger(__name__)

DEFAULT_OPTUNA_DB_PATH = Path("var/paxg_lab/optuna_studies.db")


class EarlyStoppingStagnationCallback:
    """Stops study if trials stagnate for 12 consecutive iterations post-exploration."""

    def __init__(
        self,
        patience: int = 12,
        min_delta: float = 0.5,
        startup_trials: int = 10,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.startup_trials = startup_trials
        self.best_score = float("-inf")
        self.stagnant_trials = 0
        self.stopped_early = False

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state != TrialState.COMPLETE or trial.value is None:
            return

        completed_count = len([t for t in study.trials if t.state == TrialState.COMPLETE])

        # During startup exploration, just track the best score
        if completed_count <= self.startup_trials:
            if trial.value > self.best_score:
                self.best_score = trial.value
            return

        # Check if trial improved best score by at least min_delta
        if trial.value >= self.best_score + self.min_delta:
            self.best_score = trial.value
            self.stagnant_trials = 0
        else:
            self.stagnant_trials += 1

        if self.stagnant_trials >= self.patience:
            logger.info(
                "Early stopping triggered: %d consecutive trials without >= %.2f score improvement. "
                "Best score: %.2f.",
                self.stagnant_trials,
                self.min_delta,
                self.best_score,
            )
            self.stopped_early = True
            try:
                study.stop()
            except RuntimeError:
                # May be invoked outside of active optimize loop during testing
                pass


class OptunaTPEOptimizer:
    """Executes bounded Bayesian optimization for TimesFM 3.0 LoRA."""

    def __init__(
        self,
        timeframe: str,
        snapshot: DatasetSnapshot,
        db_path: Path | str = DEFAULT_OPTUNA_DB_PATH,
        max_trials: int = 30,
        startup_trials: int = 10,
        patience: int = 12,
        min_delta: float = 0.5,
        seed: int = 42,
    ):
        self.timeframe = str(timeframe).lower().strip()
        if self.timeframe not in ("1h", "4h"):
            raise ValueError(f"Unsupported timeframe '{timeframe}'. Must be '1h' or '4h'.")

        self.snapshot = snapshot
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_trials = max_trials
        self.startup_trials = startup_trials
        self.patience = patience
        self.min_delta = min_delta
        self.seed = seed

        # Study name locked to timeframe and snapshot hash
        snap_hash = snapshot.metadata.sha256[:8]
        self.study_name = f"study_{self.timeframe}_{snap_hash}"
        self.storage_url = f"sqlite:///{self.db_path.resolve()}"

    def create_or_load_study(self) -> optuna.Study:
        """Initializes or resumes persistent Optuna study."""
        sampler = TPESampler(
            n_startup_trials=self.startup_trials,
            seed=self.seed,
            multivariate=True,
        )
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage_url,
            load_if_exists=True,
            direction="maximize",
            sampler=sampler,
        )
        return study

    def optimize(
        self,
        eval_fn: Callable[[TrainSpec, optuna.Trial], float],
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> optuna.Study:
        """Runs the optimization loop up to max_trials or until early stopping."""
        study = self.create_or_load_study()
        early_stop_cb = EarlyStoppingStagnationCallback(
            patience=self.patience,
            min_delta=self.min_delta,
            startup_trials=self.startup_trials,
        )

        def objective(trial: optuna.Trial) -> float:
            # 1. Sample hyperparameters
            spec = suggest_trial_spec(trial, timeframe=self.timeframe, seed=self.seed)

            # 2. Record trial metadata
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

            # 3. Evaluate candidate objective
            try:
                score = eval_fn(spec, trial)
                trial.set_user_attr("score_v1", score)
                if progress_callback:
                    progress_callback({
                        "trial_number": trial.number,
                        "score": score,
                        "spec": spec.to_dict(),
                        "status": "COMPLETE",
                    })
                return score
            except Exception as exc:
                err_msg = str(exc)
                logger.error("Trial %d failed with exception: %s", trial.number, err_msg)
                trial.set_user_attr("error", err_msg)
                if progress_callback:
                    progress_callback({
                        "trial_number": trial.number,
                        "error": err_msg,
                        "status": "FAILED",
                    })
                # Re-raise to let Optuna mark trial as FAIL without assigning dummy numeric score
                raise

        callbacks = [early_stop_cb]
        study.optimize(
            objective,
            n_trials=self.max_trials,
            callbacks=callbacks,
            catch=(RuntimeError, ValueError),
        )

        return study
