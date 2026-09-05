"""TimesFM 3.0 forecast predictor with quantile monotonicity and adapter support."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..constants import (
    MEDIAN_QUANTILE_INDEX,
    MODEL_REPO,
    MODEL_REVISION,
    QUANTILES,
    get_horizon_for_timeframe,
)
from ..model.lora import load_lora_adapter
from .types import ForecastRequest, ForecastResult
from timesfm3 import TimesFM3Torch

logger = logging.getLogger(__name__)


class TimesFM3Predictor:
    """Encapsulates TimesFM 3.0 inference, quantile sorting, and adapter execution."""

    def __init__(
        self,
        device: str | torch.device | None = None,
        adapter_path: str | Path | None = None,
        model_repo: str = MODEL_REPO,
        model_revision: str = MODEL_REVISION,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info("Initializing TimesFM3Predictor on %s...", self.device)
        self.base_model = TimesFM3Torch.from_pretrained(model_repo, revision=model_revision)
        self.base_model.to(self.device)
        self.base_model.eval()

        self.adapter_path = Path(adapter_path) if adapter_path else None
        self.model = self.base_model

        if self.adapter_path is not None:
            logger.info("Loading LoRA adapter from %s...", self.adapter_path)
            self.lora_model = load_lora_adapter(self.base_model, self.adapter_path)
            self.lora_model.to(self.device)
            self.lora_model.eval()
            self.model = self.lora_model

    def predict_batch(
        self,
        contexts: np.ndarray,
        horizon: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Runs batched inference and enforces monotonic quantile ordering.

        Args:
            contexts: Array of historical contexts:
                      Shape (batch_size, context_len) for univariate, or
                      Shape (batch_size, context_len, num_features) or
                      Shape (batch_size, num_features, context_len).
            horizon: Forecast horizon (24 for 1h, 6 for 4h).

        Returns:
            Tuple of (point_predictions, sorted_quantiles):
              - point_predictions: (batch_size, horizon)
              - sorted_quantiles: (batch_size, horizon, 9)
        """
        arr = np.asarray(contexts, dtype=np.float32)

        # Standardize shape to (batch_size, num_features, context_len)
        if arr.ndim == 2:
            # (batch_size, context_len) -> (batch_size, 1, context_len)
            arr = arr[:, np.newaxis, :]
        elif arr.ndim == 3:
            # Check if features are last dimension: (batch_size, context_len, num_features)
            # If context_len > num_features and shape is (B, L, F), transpose to (B, F, L)
            if arr.shape[1] > arr.shape[2]:
                arr = np.transpose(arr, (0, 2, 1))
        else:
            raise ValueError(f"Expected 2D or 3D context array, got shape {arr.shape}")

        tensor_ctx = torch.from_numpy(arr).to(self.device)

        with torch.no_grad():
            if hasattr(self.model, "base_model"):
                # PEFT wrapped model
                raw_out = self.model.base_model.model.decode(target=tensor_ctx, horizon=horizon)
            else:
                raw_out = self.model.decode(target=tensor_ctx, horizon=horizon)

        # raw_out shape: (batch_size, num_features, horizon, 9)
        # Target variable is primary variate (index 0)
        target_quantiles = raw_out[:, 0, :, :].cpu().numpy()  # (B, horizon, 9)

        # Enforce quantile monotonicity: sort along quantile dimension
        sorted_quantiles = np.sort(target_quantiles, axis=-1)

        # Point forecast is median (index 4)
        point_predictions = sorted_quantiles[:, :, MEDIAN_QUANTILE_INDEX]

        return point_predictions, sorted_quantiles

    def forecast_request(
        self,
        request: ForecastRequest,
        context_features: np.ndarray,
        forecast_origin_time: int,
    ) -> ForecastResult:
        """Executes a single forecast request and produces ForecastResult with aligned timestamps."""
        # Strict validation of adapter_path
        req_adapter = str(Path(request.adapter_path).resolve()) if request.adapter_path else None
        pred_adapter = str(Path(self.adapter_path).resolve()) if self.adapter_path else None
        if req_adapter != pred_adapter:
            raise ValueError(
                f"ForecastRequest adapter_path mismatch: request specifies '{request.adapter_path}' "
                f"(resolved: '{req_adapter}'), but predictor has adapter '{self.adapter_path}' "
                f"(resolved: '{pred_adapter}'). A separate predictor must be initialized for a different adapter."
            )

        horizon = request.horizon or get_horizon_for_timeframe(request.timeframe)
        step_ms = 3600 * 1000 if request.timeframe == "1h" else 4 * 3600 * 1000

        # Target future open_times match real future candle timestamps
        target_timestamps = [forecast_origin_time + (i + 1) * step_ms for i in range(horizon)]

        # Context features shape: (context_len, num_features) or (context_len,)
        batched_ctx = context_features[np.newaxis, ...]
        point_preds, quantiles = self.predict_batch(batched_ctx, horizon=horizon)

        p_pred = point_preds[0]
        q_pred = quantiles[0]

        return ForecastResult(
            symbol=request.symbol,
            timeframe=request.timeframe,
            forecast_origin_time=forecast_origin_time,
            target_timestamps=target_timestamps,
            point_forecast=p_pred,
            quantiles=q_pred,
            uncertainty_lower=q_pred[:, 0],   # q10
            uncertainty_upper=q_pred[:, -1],  # q90
            metadata={
                "feature_set": request.feature_set,
                "context_len": len(context_features),
                "horizon": horizon,
                "device": str(self.device),
                "is_lora": self.adapter_path is not None,
            },
        )
