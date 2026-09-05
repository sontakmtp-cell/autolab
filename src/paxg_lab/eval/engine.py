"""Backtest engine for evaluating TimesFM 3.0 and LoRA models on temporal folds."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

import numpy as np

from ..constants import (
    SCORE_VERSION,
    get_horizon_for_timeframe,
    get_horizon_weights,
)
from ..data.snapshot import DatasetSnapshot
from ..data.split import calculate_split_plan, extract_windows
from .metrics import (
    calculate_coverage_80,
    calculate_directional_accuracy,
    calculate_mean_width_80,
    calculate_step_mae,
    calculate_weighted_mae,
    calculate_weighted_pinball_loss,
    calculate_weighted_rmse,
    compute_fold_metrics,
    compute_score_v1,
)
from .predictor import TimesFM3Predictor
from .types import BacktestSpec, FoldMetrics, ScoreReport

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Orchestrates leak-free backtesting, fold evaluation, and Score v1 calculation."""

    def __init__(
        self,
        predictor: TimesFM3Predictor | None = None,
    ):
        self.predictor = predictor

    def predict_windows(
        self,
        contexts: np.ndarray,
        horizon: int,
        batch_size: int = 16,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Runs batched inference across windows to prevent GPU memory spikes."""
        if len(contexts) == 0:
            return np.empty((0, horizon), dtype=np.float32), np.empty((0, horizon, 9), dtype=np.float32)

        if self.predictor is None:
            raise RuntimeError("Predictor is not initialized on BacktestEngine.")

        point_preds_list = []
        quantiles_list = []

        num_windows = len(contexts)
        for i in range(0, num_windows, batch_size):
            batch_ctx = contexts[i : i + batch_size]
            p_pred, q_pred = self.predictor.predict_batch(batch_ctx, horizon=horizon)
            point_preds_list.append(p_pred)
            quantiles_list.append(q_pred)

        point_preds = np.concatenate(point_preds_list, axis=0)
        quantiles = np.concatenate(quantiles_list, axis=0)
        return point_preds, quantiles

    def evaluate_fold(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        timestamps: np.ndarray,
        timeframe: str,
        start_idx: int,
        end_idx: int,
        context_len: int = 256,
        step: int = 1,
        batch_size: int = 16,
        fold_id: int | str = 1,
        base_weighted_mae: float | None = None,
        base_weighted_pinball: float | None = None,
        custom_predictor_fn: Callable[[np.ndarray, int], tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> tuple[FoldMetrics, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Evaluates a model over a specific temporal boundary [start_idx, end_idx).

        Returns:
            Tuple of (FoldMetrics, predictions, quantiles, actual_targets, origin_prices).
        """
        horizon = get_horizon_for_timeframe(timeframe)

        ctx_windows, fut_windows, origins = extract_windows(
            features=features,
            targets=targets,
            context_len=context_len,
            horizon=horizon,
            start_idx=start_idx,
            end_idx=end_idx,
            step=step,
            timestamps=timestamps,
            timeframe=timeframe,
        )

        if len(ctx_windows) == 0:
            empty_m = FoldMetrics(
                fold_id=fold_id,
                num_windows=0,
                weighted_mae=0.0,
                weighted_pinball=0.0,
                rmse=0.0,
                mae=0.0,
                coverage_80=0.0,
                mean_width_80=0.0,
                directional_accuracy=0.5,
                relative_mae_to_base=1.0,
                relative_pinball_to_base=1.0,
                composite_loss=1.0,
            )
            return empty_m, np.empty((0, horizon)), np.empty((0, horizon, 9)), np.empty((0, horizon)), np.empty((0,))

        # Predict
        if custom_predictor_fn is not None:
            predictions, quantiles = custom_predictor_fn(ctx_windows, horizon)
        else:
            predictions, quantiles = self.predict_windows(ctx_windows, horizon=horizon, batch_size=batch_size)

        origin_prices = np.asarray([targets[orig] for orig in origins], dtype=np.float64)

        metrics = compute_fold_metrics(
            predictions=predictions,
            quantiles=quantiles,
            targets=fut_windows,
            origin_prices=origin_prices,
            timeframe=timeframe,
            fold_id=fold_id,
            base_weighted_mae=base_weighted_mae,
            base_weighted_pinball=base_weighted_pinball,
        )

        return metrics, predictions, quantiles, fut_windows, origin_prices

    def evaluate_naive_baseline(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        timestamps: np.ndarray,
        timeframe: str,
        start_idx: int,
        end_idx: int,
        context_len: int = 256,
        step: int = 1,
        fold_id: int | str = "naive",
        base_weighted_mae: float | None = None,
        base_weighted_pinball: float | None = None,
    ) -> FoldMetrics:
        """Evaluates naive flat forecast baseline (future prices = last known price)."""
        horizon = get_horizon_for_timeframe(timeframe)

        ctx_windows, fut_windows, origins = extract_windows(
            features=features,
            targets=targets,
            context_len=context_len,
            horizon=horizon,
            start_idx=start_idx,
            end_idx=end_idx,
            step=step,
            timestamps=timestamps,
            timeframe=timeframe,
        )

        if len(ctx_windows) == 0:
            return compute_fold_metrics(
                np.empty((0, horizon)),
                np.empty((0, horizon, 9)),
                np.empty((0, horizon)),
                np.empty((0,)),
                timeframe,
                fold_id,
            )

        origin_prices = np.asarray([targets[orig] for orig in origins], dtype=np.float64)
        # Flat prediction: repeat last known price for all horizon steps
        naive_preds = np.repeat(origin_prices[:, np.newaxis], horizon, axis=1)
        # Quantiles: collapsed onto point forecast
        naive_quantiles = np.repeat(naive_preds[:, :, np.newaxis], 9, axis=2)

        return compute_fold_metrics(
            predictions=naive_preds,
            quantiles=naive_quantiles,
            targets=fut_windows,
            origin_prices=origin_prices,
            timeframe=timeframe,
            fold_id=fold_id,
            base_weighted_mae=base_weighted_mae,
            base_weighted_pinball=base_weighted_pinball,
        )

    def run_full_backtest(
        self,
        snapshot: DatasetSnapshot,
        feature_set: str = "A",
        context_len: int = 256,
        batch_size: int = 16,
        model_name: str = "TimesFM3-Base",
    ) -> ScoreReport:
        """Runs full backtest across 3 evaluation folds and test lock set.

        Computes Score v1 relative to TimesFM Base Set A context 256.
        """
        timeframe = snapshot.timeframe
        horizon = get_horizon_for_timeframe(timeframe)

        features = snapshot.get_features(feature_set)
        # Target close price is the first column of features_a
        targets = snapshot.features_a[:, 0]
        timestamps = snapshot.timestamps

        split_plan = calculate_split_plan(total_candles=len(features), timeframe=timeframe)

        logger.info("Running full backtest for %s on %s (%d total candles)...", model_name, timeframe, len(features))

        # 1. Evaluate on each evaluation fold
        fold_metrics_list = []
        all_eval_preds = []
        all_eval_targets = []

        # First pass: if this is Base model, base_weighted_mae is its own; if candidate, base was precomputed
        for fold in split_plan.eval_folds:
            logger.info("Evaluating fold %d [%d, %d)...", fold.fold_id, fold.eval_start, fold.eval_end)
            f_metric, preds, _, tgts, _ = self.evaluate_fold(
                features=features,
                targets=targets,
                timestamps=timestamps,
                timeframe=timeframe,
                start_idx=fold.eval_start,
                end_idx=fold.eval_end,
                context_len=context_len,
                batch_size=batch_size,
                fold_id=fold.fold_id,
                # For base reference, relative to itself:
                base_weighted_mae=None,
                base_weighted_pinball=None,
            )
            fold_metrics_list.append(f_metric)
            if len(preds) > 0:
                all_eval_preds.append(preds)
                all_eval_targets.append(tgts)

        # 2. Evaluate locked test set
        logger.info("Evaluating test set [%d, %d)...", split_plan.test_start, split_plan.test_end)
        test_metric, test_preds, _, test_tgts, _ = self.evaluate_fold(
            features=features,
            targets=targets,
            timestamps=timestamps,
            timeframe=timeframe,
            start_idx=split_plan.test_start,
            end_idx=split_plan.test_end,
            context_len=context_len,
            batch_size=batch_size,
            fold_id="test_locked",
        )

        # 3. Evaluate Naive Baseline on same folds for comparison
        naive_fold_metrics = []
        for fold in split_plan.eval_folds:
            n_metric = self.evaluate_naive_baseline(
                features=features,
                targets=targets,
                timestamps=timestamps,
                timeframe=timeframe,
                start_idx=fold.eval_start,
                end_idx=fold.eval_end,
                context_len=context_len,
                fold_id=fold.fold_id,
            )
            naive_fold_metrics.append(n_metric)

        naive_test_metric = self.evaluate_naive_baseline(
            features=features,
            targets=targets,
            timestamps=timestamps,
            timeframe=timeframe,
            start_idx=split_plan.test_start,
            end_idx=split_plan.test_end,
            context_len=context_len,
            fold_id="test_locked",
        )

        # 4. Compute composite Score v1
        fold_losses = [m.composite_loss for m in fold_metrics_list]
        score_v1 = compute_score_v1(fold_losses)

        # 5. Aggregate overall metrics across evaluation folds
        concat_eval_preds = np.concatenate(all_eval_preds, axis=0) if all_eval_preds else np.empty((0, horizon))
        concat_eval_tgts = np.concatenate(all_eval_targets, axis=0) if all_eval_targets else np.empty((0, horizon))

        weights = get_horizon_weights(timeframe)
        overall_w_mae = calculate_weighted_mae(concat_eval_preds, concat_eval_tgts, weights)
        overall_w_rmse = calculate_weighted_rmse(concat_eval_preds, concat_eval_tgts, weights)
        raw_mae = float(np.mean(np.abs(concat_eval_preds - concat_eval_tgts))) if len(concat_eval_preds) > 0 else 0.0

        step_maes = calculate_step_mae(concat_eval_preds, concat_eval_tgts, timeframe=timeframe)

        # Average coverage and width across folds
        avg_cov80 = float(np.mean([m.coverage_80 for m in fold_metrics_list])) if fold_metrics_list else 0.0
        avg_width80 = float(np.mean([m.mean_width_80 for m in fold_metrics_list])) if fold_metrics_list else 0.0
        avg_dir_acc = float(np.mean([m.directional_accuracy for m in fold_metrics_list])) if fold_metrics_list else 0.0

        naive_eval_mae = float(np.mean([m.weighted_mae for m in naive_fold_metrics])) if naive_fold_metrics else 0.0

        report = ScoreReport(
            score_version=SCORE_VERSION,
            timeframe=timeframe,
            horizon=horizon,
            model_name=model_name,
            feature_set=feature_set,
            context_len=context_len,
            score=score_v1,
            total_eval_windows=len(concat_eval_preds),
            overall_mae=raw_mae,
            overall_rmse=overall_w_rmse,
            overall_weighted_mae=overall_w_mae,
            overall_weighted_pinball=float(np.mean([m.weighted_pinball for m in fold_metrics_list])),
            coverage_80=avg_cov80,
            mean_width_80=avg_width80,
            directional_accuracy=avg_dir_acc,
            step_mae=step_maes,
            fold_metrics=fold_metrics_list,
            test_metrics=test_metric,
            baseline_comparisons={
                "naive_flat_eval_weighted_mae": naive_eval_mae,
                "naive_flat_test_weighted_mae": naive_test_metric.weighted_mae,
                "improvement_over_naive_pct": float((naive_eval_mae - overall_w_mae) / max(naive_eval_mae, 1e-6) * 100.0),
            },
            metadata={
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "snapshot_id": snapshot.metadata.snapshot_id,
                "snapshot_sha256": snapshot.metadata.sha256,
            },
        )

        return report
