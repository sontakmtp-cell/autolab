"""Comprehensive evaluation metrics and Score v1 implementation for PAXG Forecast Lab."""

from __future__ import annotations

import numpy as np

from ..constants import (
    QUANTILES,
    TICK_SIZE,
    get_horizon_for_timeframe,
    get_horizon_weights,
)
from .types import FoldMetrics


def calculate_weighted_mae(
    predictions: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray | tuple[float, ...],
) -> float:
    """Calculates horizon-weighted MAE across prediction windows.

    Args:
        predictions: (N, horizon) array of point predictions (median).
        targets: (N, horizon) array of actual future close prices.
        weights: (horizon,) array of time-decay weights summing to 1.0.

    Returns:
        Weighted MAE in price units (USDT).
    """
    if len(predictions) == 0:
        return 0.0
    w = np.asarray(weights, dtype=np.float64)
    abs_err = np.abs(predictions - targets)  # (N, horizon)
    step_mae = np.mean(abs_err, axis=0)     # (horizon,)
    return float(np.sum(step_mae * w))


def calculate_weighted_rmse(
    predictions: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray | tuple[float, ...],
) -> float:
    """Calculates horizon-weighted RMSE across prediction windows."""
    if len(predictions) == 0:
        return 0.0
    w = np.asarray(weights, dtype=np.float64)
    sq_err = (predictions - targets) ** 2
    step_mse = np.mean(sq_err, axis=0)
    weighted_mse = np.sum(step_mse * w)
    return float(np.sqrt(max(weighted_mse, 0.0)))


def calculate_step_mae(
    predictions: np.ndarray,
    targets: np.ndarray,
    timeframe: str = "1h",
) -> dict[int, float]:
    """Calculates MAE at key milestone steps according to specification.

    Milestone steps:
      - 1h: 1, 6, 12, 24
      - 4h: 1, 2, 3, 6
    """
    if len(predictions) == 0:
        return {}
    abs_err = np.abs(predictions - targets)
    step_maes = np.mean(abs_err, axis=0)

    milestones = [1, 6, 12, 24] if timeframe == "1h" else [1, 2, 3, 6]
    horizon = get_horizon_for_timeframe(timeframe)

    result = {}
    for step in milestones:
        if step <= horizon:
            result[step] = float(step_maes[step - 1])
    return result


def calculate_weighted_pinball_loss(
    quantiles: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray | tuple[float, ...],
    quantile_levels: list[float] | tuple[float, ...] = tuple(QUANTILES),
) -> float:
    """Calculates horizon-weighted Pinball loss across 9 quantiles.

    Args:
        quantiles: (N, horizon, num_quantiles) array of quantile predictions.
        targets: (N, horizon) array of actual future close prices.
        weights: (horizon,) array of time-decay weights summing to 1.0.
        quantile_levels: list/tuple of quantile levels (e.g. 0.1 to 0.9).

    Returns:
        Weighted pinball loss.
    """
    if len(quantiles) == 0:
        return 0.0

    w = np.asarray(weights, dtype=np.float64)  # (horizon,)
    q_levels = np.asarray(quantile_levels, dtype=np.float64)  # (9,)

    # targets shape: (N, horizon, 1), quantiles shape: (N, horizon, 9)
    y = targets[:, :, np.newaxis]
    diff = y - quantiles

    # Pinball loss per quantile: max(q * diff, (q - 1) * diff)
    loss = np.maximum(q_levels * diff, (q_levels - 1.0) * diff)  # (N, horizon, 9)

    # Average over windows N: (horizon, 9)
    mean_window_loss = np.mean(loss, axis=0)

    # Average over quantiles: (horizon,)
    step_pinball = np.mean(mean_window_loss, axis=-1)

    # Horizon-weighted sum
    return float(np.sum(step_pinball * w))


def calculate_coverage_80(
    quantiles: np.ndarray,
    targets: np.ndarray,
) -> float:
    """Calculates empirical coverage of nominal 80% uncertainty interval [q10, q90]."""
    if len(quantiles) == 0:
        return 0.0
    q10 = quantiles[:, :, 0]
    q90 = quantiles[:, :, 8]
    in_interval = (targets >= q10) & (targets <= q90)
    return float(np.mean(in_interval))


def calculate_mean_width_80(
    quantiles: np.ndarray,
) -> float:
    """Calculates average width of the nominal 80% interval (q90 - q10)."""
    if len(quantiles) == 0:
        return 0.0
    widths = quantiles[:, :, 8] - quantiles[:, :, 0]
    return float(np.mean(widths))


def calculate_directional_accuracy(
    predictions: np.ndarray,
    targets: np.ndarray,
    origin_prices: np.ndarray,
    min_move_ticks: float = 2.0 * TICK_SIZE,
) -> float:
    """Calculates directional accuracy comparing sign(pred - origin) to sign(tgt - origin).

    Ignores movements smaller than min_move_ticks to prevent tick noise bias.
    """
    if len(predictions) == 0:
        return 0.0

    orig = origin_prices[:, np.newaxis]
    pred_move = predictions - orig
    actual_move = targets - orig

    # Filter out near-zero moves below min_move_ticks
    meaningful = np.abs(actual_move) >= min_move_ticks
    if not np.any(meaningful):
        return 0.5  # Neutral

    pred_sign = np.sign(pred_move[meaningful])
    actual_sign = np.sign(actual_move[meaningful])

    matches = (pred_sign == actual_sign)
    return float(np.mean(matches))


def compute_fold_metrics(
    predictions: np.ndarray,
    quantiles: np.ndarray,
    targets: np.ndarray,
    origin_prices: np.ndarray,
    timeframe: str,
    fold_id: int | str,
    base_weighted_mae: float | None = None,
    base_weighted_pinball: float | None = None,
) -> FoldMetrics:
    """Computes full suite of metrics for a single backtest evaluation fold."""
    num_windows = len(predictions)
    if num_windows == 0:
        return FoldMetrics(
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

    weights = get_horizon_weights(timeframe)
    w_mae = calculate_weighted_mae(predictions, targets, weights)
    w_rmse = calculate_weighted_rmse(predictions, targets, weights)
    w_pinball = calculate_weighted_pinball_loss(quantiles, targets, weights)
    raw_mae = float(np.mean(np.abs(predictions - targets)))
    cov80 = calculate_coverage_80(quantiles, targets)
    width80 = calculate_mean_width_80(quantiles)
    dir_acc = calculate_directional_accuracy(predictions, targets, origin_prices)

    # Relative to base reference
    if base_weighted_mae is not None and base_weighted_mae > TICK_SIZE:
        a_f = w_mae / base_weighted_mae
    else:
        a_f = 1.0

    if base_weighted_pinball is not None and base_weighted_pinball > TICK_SIZE:
        q_f = w_pinball / base_weighted_pinball
    else:
        q_f = 1.0

    # L_f = 0.70 * A_f + 0.30 * Q_f
    l_f = 0.70 * a_f + 0.30 * q_f

    return FoldMetrics(
        fold_id=fold_id,
        num_windows=num_windows,
        weighted_mae=w_mae,
        weighted_pinball=w_pinball,
        rmse=w_rmse,
        mae=raw_mae,
        coverage_80=cov80,
        mean_width_80=width80,
        directional_accuracy=dir_acc,
        relative_mae_to_base=a_f,
        relative_pinball_to_base=q_f,
        composite_loss=l_f,
    )


def compute_score_v1(fold_losses: list[float]) -> float:
    """Calculates official Score v1 across evaluation folds.

    Formula:
      Score = 100 * [1 - (0.80 * mean(L_f) + 0.20 * max(L_f))]

    Where:
      L_f = 0.70 * A_f + 0.30 * Q_f
      A_f = Weighted MAE(model) / Weighted MAE(base)
      Q_f = Weighted Pinball(model) / Weighted Pinball(base)

    Base reference model produces Score = 0.0.
    Positive scores indicate superior performance; negative scores indicate inferior.
    """
    if not fold_losses:
        return 0.0

    mean_l = float(np.mean(fold_losses))
    worst_l = float(np.max(fold_losses))

    penalty = 0.80 * mean_l + 0.20 * worst_l
    score = 100.0 * (1.0 - penalty)
    return float(score)
