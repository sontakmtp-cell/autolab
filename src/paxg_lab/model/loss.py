"""Loss functions for TimesFM 3.0 fine-tuning."""

from __future__ import annotations

import torch
import torch.nn as nn
from ..constants import QUANTILES, MEDIAN_QUANTILE_INDEX, EPSILON


def pinball_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    quantiles: list[float] | torch.Tensor | None = None,
) -> torch.Tensor:
    """Computes pinball (quantile) loss across specified quantiles.

    Args:
        predictions: Predicted quantiles, shape (b, v, horizon, num_quantiles).
        targets: Ground truth values, shape (b, v, horizon).
        quantiles: List or 1D tensor of quantile levels in (0, 1).

    Returns:
        Scalar pinball loss (float32).
    """
    if quantiles is None:
        quantiles = QUANTILES

    if not isinstance(quantiles, torch.Tensor):
        q_tensor = torch.tensor(quantiles, dtype=torch.float32, device=predictions.device)
    else:
        q_tensor = quantiles.to(dtype=torch.float32, device=predictions.device)

    # Cast to float32 for stable loss computation
    preds_f32 = predictions.float()
    targets_f32 = targets.float().unsqueeze(-1)  # (b, v, horizon, 1)

    errors = targets_f32 - preds_f32  # (b, v, horizon, num_quantiles)
    # Pinball loss: max(q * error, (q - 1) * error)
    # = error * (q - (error < 0))
    q_expanded = q_tensor.view(1, 1, 1, -1)
    loss = torch.max(q_expanded * errors, (q_expanded - 1.0) * errors)
    return loss.mean()


def combined_forecast_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    last_context_price: torch.Tensor | float | None = None,
    quantiles: list[float] | torch.Tensor | None = None,
    median_idx: int = MEDIAN_QUANTILE_INDEX,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Calculates combined median MAE and 9-quantile pinball loss, normalized by last price.

    Args:
        predictions: (b, v, horizon, num_quantiles).
        targets: (b, v, horizon).
        last_context_price: (b, v, 1) or scalar, last close price in context window.
        quantiles: list of quantiles (default 9 quantiles).
        median_idx: index of median quantile in predictions.

    Returns:
        (total_loss, metrics_dict)
    """
    preds_f32 = predictions.float()
    targets_f32 = targets.float()

    # Median forecast MAE
    median_pred = preds_f32[..., median_idx]
    median_mae = torch.abs(targets_f32 - median_pred).mean()

    # 9-quantile pinball loss
    q_loss = pinball_loss(preds_f32, targets_f32, quantiles=quantiles)

    # Scale normalization factor (last close price in context window)
    if last_context_price is not None:
        if isinstance(last_context_price, (int, float)):
            scale = torch.tensor(abs(last_context_price), device=preds_f32.device, dtype=torch.float32)
        else:
            scale = torch.clamp_min(torch.abs(last_context_price.float()).mean(), 1e-3)
    else:
        # Fallback to mean absolute target scale if context price not supplied
        scale = torch.clamp_min(torch.abs(targets_f32).mean(), 1e-3)

    norm_median_mae = median_mae / scale
    norm_q_loss = q_loss / scale

    # Combined loss: equal blend of median MAE and pinball loss
    total_loss = norm_median_mae + norm_q_loss

    metrics = {
        "total_loss": total_loss.item(),
        "median_mae_raw": median_mae.item(),
        "pinball_loss_raw": q_loss.item(),
        "scale": scale.item(),
        "norm_median_mae": norm_median_mae.item(),
        "norm_pinball_loss": norm_q_loss.item(),
    }
    return total_loss, metrics
