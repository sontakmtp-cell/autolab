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
import logging
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np
import optuna

from ..constants import (
    MODEL_REPO,
    MODEL_REVISION,
    get_horizon_for_timeframe,
)
from ..data.snapshot import DatasetSnapshot
from ..data.split import calculate_split_plan
from ..eval.engine import BacktestEngine
from ..eval.predictor import TimesFM3Predictor
from ..eval.types import ScoreReport
from ..model.manifest import AdapterManifest
from ..model.store import AdapterStore
from ..model.train_spec import TrainSpec
from ..model.trainer import LoRATrainer
from ..queue.storage import GPUJobStorage
from ..queue.types import AutoRunState
from .gatekeeper import GatekeeperDecision, LoRAGatekeeper
from .optimizer import DEFAULT_OPTUNA_DB_PATH, OptunaTPEOptimizer

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
    ) -> tuple[float, int, float]:
        """Trains and evaluates a candidate spec across 3 out-of-sample evaluation folds strictly without locked test.

        Returns:
            Tuple of (Score v1, best_epoch, best_val_loss).
        """
        # Ensure locked test data is NEVER passed to training or evaluation
        if custom_eval_fn is not None:
            report = custom_eval_fn(self.snapshot, spec)
            return float(report.score), 2, 0.05

        # Standard pipeline: train on history before eval folds
        from timesfm3 import TimesFM3Torch
        base_model = TimesFM3Torch.from_pretrained(MODEL_REPO, revision=MODEL_REVISION)

        features_df = self.snapshot.to_dataframe(spec.feature_set)
        trainer = LoRATrainer(base_model=base_model, spec=spec)

        # Train on pre-eval segment
        train_res = trainer.train(
            features_df=features_df,
            snapshot_hash=self.snapshot.metadata.sha256,
            fold_id=1,
        )

        # Save temporary trial adapter to store
        trial_adapter_id = f"trial_tmp_{self.timeframe}_{int(time.time()*1000)}"
        manifest = train_res.manifest
        manifest.adapter_id = trial_adapter_id
        saved_dir = self.store.save_adapter(
            peft_model=train_res.trained_model,
            manifest=manifest,
            base_model=base_model,
            smoke_test=False,
        )

        # Evaluate on 3 evaluation folds (out-of-sample) strictly excluding locked test
        predictor = TimesFM3Predictor(adapter_path=saved_dir)
        engine = BacktestEngine(predictor=predictor)
        report = engine.run_full_backtest(
            snapshot=self.snapshot,
            feature_set=spec.feature_set,
            context_len=spec.context_len,
            batch_size=16,
            model_name=f"Trial-{trial_adapter_id}",
            base_reference_metrics=base_reference_report,
            include_locked_test=False,  # STRICTLY FALSE: test data never leaked into trials!
        )

        # Clean up temporary trial adapter from disk
        self.store.delete_adapter(trial_adapter_id, use_trash=False)

        return float(report.score), int(train_res.best_epoch), float(train_res.best_val_loss)

    def run_multi_seed_verification(
        self,
        best_spec: TrainSpec,
        base_reference_report: ScoreReport,
        custom_trainer_fn: Callable[[TrainSpec], Any] | None = None,
        custom_eval_fn: Callable[[Any, TrainSpec], ScoreReport] | None = None,
    ) -> MultiSeedEvalSummary:
        """Evaluates best hyperparameter configuration across seeds [42, 123, 2026] and takes median."""
        scores = []
        best_epochs = []

        logger.info("Executing multi-seed verification on seeds %s for %s...", SEEDS_MULTI_RUN, self.timeframe)
        for seed in SEEDS_MULTI_RUN:
            seed_spec = TrainSpec.from_dict(best_spec.to_dict())
            seed_spec.seed = seed

            score, b_epoch, _ = self.run_trial_evaluation(
                spec=seed_spec,
                base_reference_report=base_reference_report,
                custom_trainer_fn=custom_trainer_fn,
                custom_eval_fn=custom_eval_fn,
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
        # Train ceiling is strictly test_start - horizon (purge buffer)
        train_ceiling = self.split_plan.test_start - self.split_plan.horizon
        train_df = features_df.iloc[:train_ceiling].copy()

        if custom_trainer_fn is not None:
            return custom_trainer_fn(candidate_spec, best_epoch)

        from timesfm3 import TimesFM3Torch
        base_model = TimesFM3Torch.from_pretrained(MODEL_REPO, revision=MODEL_REVISION)

        cand_spec = TrainSpec.from_dict(candidate_spec.to_dict())
        cand_spec.max_epochs = max(1, best_epoch)

        trainer = LoRATrainer(base_model=base_model, spec=cand_spec)
        train_res = trainer.train(
            features_df=train_df,
            snapshot_hash=self.snapshot.metadata.sha256,
            fold_id=1,
        )

        manifest = train_res.manifest
        manifest.adapter_id = cand_id
        manifest.is_verified = False  # Not yet verified!

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
        custom_test_eval_fn: Callable[[DatasetSnapshot, AdapterManifest], tuple[ScoreReport, np.ndarray, np.ndarray, np.ndarray, float]] | None = None,
    ) -> tuple[ScoreReport, GatekeeperDecision]:
        """Executes strict single evaluation on 90-day locked test set and triggers gatekeeper winner check."""
        logger.info(
            "Executing single locked verification on 90-day test set [%d, %d) for candidate '%s'...",
            self.split_plan.test_start,
            self.split_plan.test_end,
            candidate_manifest.adapter_id,
        )

        if custom_test_eval_fn is not None:
            cand_test_report, cand_preds, base_preds, tgts, naive_mae = custom_test_eval_fn(
                self.snapshot,
                candidate_manifest,
            )
        else:
            # 1. Base reference evaluation on test set
            base_predictor = TimesFM3Predictor()
            base_engine = BacktestEngine(predictor=base_predictor)
            _, base_preds, _, tgts, _, _ = base_engine.evaluate_fold(
                features=self.snapshot.get_features("A"),
                targets=self.snapshot.features_a[:, 0],
                timestamps=self.snapshot.timestamps,
                timeframe=self.timeframe,
                start_idx=self.split_plan.test_start,
                end_idx=self.split_plan.test_end,
                context_len=256,
                batch_size=16,
                fold_id="test_locked",
            )

            # 2. Naive baseline on test set
            naive_metric = base_engine.evaluate_naive_baseline(
                features=self.snapshot.get_features("A"),
                targets=self.snapshot.features_a[:, 0],
                timestamps=self.snapshot.timestamps,
                timeframe=self.timeframe,
                start_idx=self.split_plan.test_start,
                end_idx=self.split_plan.test_end,
                context_len=256,
                fold_id="test_locked",
            )
            naive_mae = naive_metric.weighted_mae

            # 3. Candidate evaluation on locked test set
            cand_predictor = TimesFM3Predictor(adapter_path=candidate_adapter_path)
            cand_engine = BacktestEngine(predictor=cand_predictor)
            cand_test_report = cand_engine.run_full_backtest(
                snapshot=self.snapshot,
                feature_set=candidate_manifest.feature_set,
                context_len=candidate_manifest.context_len,
                batch_size=16,
                model_name=f"Candidate-{candidate_manifest.adapter_id}",
                base_reference_metrics=base_reference_report,
                include_locked_test=True,  # ONLY HERE: exactly 1 final evaluation on locked test!
                adapter_manifest=candidate_manifest,
            )
            _, cand_preds, _, _, _, _ = cand_engine.evaluate_fold(
                features=self.snapshot.get_features(candidate_manifest.feature_set),
                targets=self.snapshot.features_a[:, 0],
                timestamps=self.snapshot.timestamps,
                timeframe=self.timeframe,
                start_idx=self.split_plan.test_start,
                end_idx=self.split_plan.test_end,
                context_len=candidate_manifest.context_len,
                batch_size=16,
                fold_id="test_locked",
            )

        # 4. Determine current recommended model score
        current_rec_id = self.store.get_recommended(self.timeframe)
        current_score = 0.0
        if current_rec_id:
            try:
                rec_path = self.store.get_adapter_path(current_rec_id)
                rec_manifest = AdapterManifest.load_json(rec_path / "paxg_manifest.json")
                current_score = float(rec_manifest.metrics.get("score_v1", 0.0))
            except Exception:
                current_score = 0.0

        # 5. Evaluate winning criteria via Gatekeeper
        decision = self.gatekeeper.evaluate_candidate(
            candidate_manifest=candidate_manifest,
            candidate_test_report=cand_test_report,
            candidate_test_preds=cand_preds,
            base_test_preds=base_preds,
            test_targets=tgts,
            naive_test_mae=naive_mae,
            current_recommended_score=current_score,
            perform_backup=True,
        )

        return cand_test_report, decision
