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


import datetime
from typing import Any


def calculate_low_move_breakdown(
    targets: np.ndarray,
    origin_prices: np.ndarray,
    min_move_ticks: float = 2.0 * TICK_SIZE,
) -> dict[str, Any]:
    """Calculates statistics for movements under 2 ticks."""
    if len(targets) == 0:
        return {"count": 0, "pct": 0.0, "tick_size": TICK_SIZE, "threshold_usdt": min_move_ticks}

    orig = origin_prices[:, np.newaxis]
    actual_move = np.abs(targets - orig)
    low_moves = actual_move < min_move_ticks
    low_count = int(np.sum(low_moves))
    total_elements = int(actual_move.size)
    low_pct = float(low_count / max(total_elements, 1) * 100.0)

    return {
        "low_move_steps": low_count,
        "total_steps": total_elements,
        "pct": low_pct,
        "threshold_usdt": min_move_ticks,
        "tick_size": TICK_SIZE,
    }


def calculate_temporal_breakdown(
    predictions: np.ndarray,
    targets: np.ndarray,
    origin_timestamps: np.ndarray,
    weights: np.ndarray | tuple[float, ...],
) -> dict[str, Any]:
    """Calculates MAE split by weekday (Mon-Fri) vs weekend (Sat-Sun) in UTC."""
    if len(predictions) == 0 or len(origin_timestamps) == 0:
        return {"weekday_weighted_mae": 0.0, "weekend_weighted_mae": 0.0, "weekday_windows": 0, "weekend_windows": 0}

    # Extract UTC day of week: Monday=0, Sunday=6
    dows = np.array([
        datetime.datetime.fromtimestamp(ts / 1000.0, tz=datetime.timezone.utc).weekday()
        for ts in origin_timestamps
    ])

    is_weekday = dows < 5
    is_weekend = dows >= 5

    weekday_mae = calculate_weighted_mae(predictions[is_weekday], targets[is_weekday], weights) if np.any(is_weekday) else 0.0
    weekend_mae = calculate_weighted_mae(predictions[is_weekend], targets[is_weekend], weights) if np.any(is_weekend) else 0.0

    return {
        "weekday_windows": int(np.sum(is_weekday)),
        "weekend_windows": int(np.sum(is_weekend)),
        "weekday_weighted_mae": weekday_mae,
        "weekend_weighted_mae": weekend_mae,
    }


def calculate_volatility_breakdown(
    predictions: np.ndarray,
    targets: np.ndarray,
    origin_prices: np.ndarray,
    weights: np.ndarray | tuple[float, ...],
) -> dict[str, Any]:
    """Calculates MAE split by low vs high volatility (split by median realized volatility)."""
    if len(predictions) == 0:
        return {"low_vol_weighted_mae": 0.0, "high_vol_weighted_mae": 0.0, "median_volatility": 0.0}

    orig = origin_prices[:, np.newaxis]
    window_vol = np.std(targets - orig, axis=1)
    median_vol = float(np.median(window_vol))

    low_vol_mask = window_vol <= median_vol
    high_vol_mask = window_vol > median_vol

    low_vol_mae = calculate_weighted_mae(predictions[low_vol_mask], targets[low_vol_mask], weights) if np.any(low_vol_mask) else 0.0
    high_vol_mae = calculate_weighted_mae(predictions[high_vol_mask], targets[high_vol_mask], weights) if np.any(high_vol_mask) else 0.0

    return {
        "median_volatility": median_vol,
        "low_vol_windows": int(np.sum(low_vol_mask)),
        "high_vol_windows": int(np.sum(high_vol_mask)),
        "low_vol_weighted_mae": low_vol_mae,
        "high_vol_weighted_mae": high_vol_mae,
    }


def calculate_sampling_breakdown(
    num_windows: int,
    horizon: int,
    step: int = 1,
) -> dict[str, Any]:
    """Calculates window sampling, overlap ratio, and independent block statistics."""
    overlap_ratio = float((horizon - step) / horizon) if horizon > step else 0.0
    independent_blocks = int(num_windows // horizon) if horizon > 0 else 0

    return {
        "num_windows": num_windows,
        "step": step,
        "horizon": horizon,
        "overlap_ratio": overlap_ratio,
        "independent_blocks": independent_blocks,
    }


def compute_fold_metrics(
    predictions: np.ndarray,
    quantiles: np.ndarray,
    targets: np.ndarray,
    origin_prices: np.ndarray,
    timeframe: str,
    fold_id: int | str,
    base_weighted_mae: float | None = None,
    base_weighted_pinball: float | None = None,
    is_in_sample: bool = False,
    in_sample_warning: str | None = None,
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
            reference_valid=True,
            insufficient_information=False,
            is_in_sample=is_in_sample,
            warning=in_sample_warning,
        )

    weights = get_horizon_weights(timeframe)
    w_mae = calculate_weighted_mae(predictions, targets, weights)
    w_rmse = calculate_weighted_rmse(predictions, targets, weights)
    w_pinball = calculate_weighted_pinball_loss(quantiles, targets, weights)
    raw_mae = float(np.mean(np.abs(predictions - targets)))
    cov80 = calculate_coverage_80(quantiles, targets)
    width80 = calculate_mean_width_80(quantiles)
    dir_acc = calculate_directional_accuracy(predictions, targets, origin_prices)

    # Base reference evaluation and zero-error policy
    reference_valid = True
    insufficient_information = False
    warning_msg = None

    if base_weighted_mae is not None and base_weighted_pinball is not None:
        if base_weighted_mae <= TICK_SIZE or base_weighted_pinball <= TICK_SIZE:
            reference_valid = False
            insufficient_information = True
            warning_msg = (
                f"Base reference error is near zero (MAE={base_weighted_mae:.4f}, "
                f"Pinball={base_weighted_pinball:.4f} <= tick {TICK_SIZE}). Fold marked insufficient information."
            )
            a_f = float("nan")
            q_f = float("nan")
            l_f = float("nan")
        else:
            a_f = w_mae / base_weighted_mae
            q_f = w_pinball / base_weighted_pinball
            l_f = 0.70 * a_f + 0.30 * q_f
    else:
        # Self-referenced baseline (Base model): A_f = Q_f = 1.0 -> L_f = 1.0
        a_f = 1.0
        q_f = 1.0
        l_f = 1.0

    warnings = []
    if warning_msg:
        warnings.append(warning_msg)
    if in_sample_warning:
        warnings.append(in_sample_warning)
    final_warning = "; ".join(warnings) if warnings else None

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
        reference_valid=reference_valid,
        insufficient_information=insufficient_information,
        is_in_sample=is_in_sample,
        warning=final_warning,
    )


def compute_score_v1(
    fold_losses: list[float],
    valid_mask: list[bool] | None = None,
) -> float:
    """Calculates official Score v1 across evaluation folds.

    Formula:
      L_f = 0.70 * A_f + 0.30 * Q_f
      A_f = Weighted MAE(model) / Weighted MAE(base)
      Q_f = Weighted Pinball(model) / Weighted Pinball(base)

      Penalty = 0.80 * mean(L_f) + 0.20 * max(L_f)
      Score = 100 * [1 - Penalty]

    Base reference model produces Score = 0.0.
    Folds marked insufficient_information (reference_valid=False or NaN loss) are excluded.
    If all folds are invalid, raises ValueError per PLAN policy.
    """
    if not fold_losses:
        return 0.0

    if valid_mask is not None:
        active_losses = [l for l, v in zip(fold_losses, valid_mask) if v and not np.isnan(l)]
    else:
        active_losses = [l for l in fold_losses if not np.isnan(l)]

    if not active_losses:
        raise ValueError(
            "All evaluation folds have insufficient information (base error <= tick size). Cannot compute Score v1."
        )

    mean_l = float(np.mean(active_losses))
    worst_l = float(np.max(active_losses))

    penalty = 0.80 * mean_l + 0.20 * worst_l
    score = 100.0 * (1.0 - penalty)
    return float(score)

