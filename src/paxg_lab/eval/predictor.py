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
from ..model.manifest import AdapterManifest
from ..model.store import AdapterStore
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

        self.model_repo = model_repo
        self.model_revision = model_revision

        self.adapter_path: Path | None = None
        self.manifest: AdapterManifest | None = None
        self.lora_model = None
        self.model = self.base_model

        if adapter_path is not None:
            self.load_adapter(adapter_path)

    def load_adapter(self, adapter_path: str | Path) -> None:
        """Loads a LoRA adapter, verifying SHA-256 integrity and manifest compatibility."""
        p = Path(adapter_path)
        if not p.is_dir():
            raise FileNotFoundError(f"Adapter path not found: {p}")

        manifest_file = p / "paxg_manifest.json"
        if not manifest_file.exists():
            raise FileNotFoundError(
                f"Required 'paxg_manifest.json' not found in adapter directory: {p}. "
                "Production adapters must have a valid manifest. "
                "For isolated debugging/development, use 'load_raw_adapter_unsafe()'."
            )

        # If an adapter was already active, unload it first to maintain purity
        if self.lora_model is not None:
            self.unload_adapter()

        logger.info("Loading verified adapter with manifest from %s...", p)
        store = AdapterStore(base_dir=p.parent)
        self.lora_model, self.manifest = store.load_adapter(
            adapter_id_or_path=p,
            base_model=self.base_model,
        )

        self.adapter_path = p
        self.lora_model.to(self.device)
        self.lora_model.eval()
        self.model = self.lora_model

    def load_raw_adapter_unsafe(self, adapter_path: str | Path) -> None:
        """Loads a raw LoRA adapter without manifest verification.

        WARNING: This method bypasses SHA-256 tamper verification, timeframe,
        horizon, feature set, column order, and base revision compatibility checks.
        Use strictly for isolated development/debugging.
        """
        p = Path(adapter_path)
        if not p.is_dir():
            raise FileNotFoundError(f"Adapter path not found: {p}")

        if self.lora_model is not None:
            self.unload_adapter()

        logger.warning("UNSAFE LOAD: Loading unverified adapter without manifest from %s...", p)
        self.lora_model = load_lora_adapter(self.base_model, p)
        self.manifest = None
        self.adapter_path = p
        self.lora_model.to(self.device)
        self.lora_model.eval()
        self.model = self.lora_model

    def unload_adapter(self) -> None:
        """Unloads active LoRA adapter and restores clean base model."""
        if self.lora_model is not None:
            logger.info("Unloading LoRA adapter and restoring clean base model...")
            if hasattr(self.lora_model, "unload"):
                self.base_model = self.lora_model.unload()
            self.lora_model = None
            self.adapter_path = None
            self.manifest = None
            self.model = self.base_model
            self.model.eval()

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

    def predict(
        self,
        contexts: np.ndarray,
        horizon: int,
        batch_size: int = 16,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Runs batched inference over multiple contexts to avoid OOM."""
        if len(contexts) == 0:
            return np.empty((0, horizon), dtype=np.float32), np.empty((0, horizon, 9), dtype=np.float32)

        point_preds = []
        quantiles = []
        for i in range(0, len(contexts), batch_size):
            batch = contexts[i : i + batch_size]
            p, q = self.predict_batch(batch, horizon=horizon)
            point_preds.append(p)
            quantiles.append(q)

        return np.concatenate(point_preds, axis=0), np.concatenate(quantiles, axis=0)

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

        # Validate context length and feature dimension against request
        actual_context_len = len(context_features)
        actual_num_features = context_features.shape[1] if context_features.ndim > 1 else 1

        if actual_context_len != request.context_len:
            raise ValueError(
                f"Context length mismatch: request specifies context_len={request.context_len}, "
                f"but received array of length {actual_context_len}."
            )

        # Resolve columns from request or FEATURE_SPECS
        from ..data.features import FEATURE_SPECS

        resolved_columns = list(request.columns) if request.columns is not None else (
            list(FEATURE_SPECS[request.feature_set].columns) if request.feature_set in FEATURE_SPECS else []
        )
        if len(resolved_columns) != actual_num_features:
            raise ValueError(
                f"Feature dimension mismatch: context array has {actual_num_features} features, "
                f"but resolved column list has {len(resolved_columns)} ({resolved_columns})."
            )

        # Manifest compatibility check if adapter has manifest
        in_sample_warning = None
        manifest = getattr(self, "manifest", None)
        if manifest is not None:
            manifest.verify_compatibility(
                expected_timeframe=request.timeframe,
                expected_horizon=horizon,
                expected_context=actual_context_len,
                expected_feature_set=request.feature_set,
                expected_columns=resolved_columns,
                expected_num_features=actual_num_features,
                expected_base_revision=getattr(self, "model_revision", MODEL_REVISION),
            )
            # Check in-sample overlap
            eval_start = forecast_origin_time + step_ms
            eval_end = forecast_origin_time + horizon * step_ms
            is_overlap, warning_msg = manifest.check_in_sample_overlap(eval_start, eval_end)
            if is_overlap:
                in_sample_warning = warning_msg
                logger.warning(warning_msg)

        # Target future open_times match real future candle timestamps
        target_timestamps = [forecast_origin_time + (i + 1) * step_ms for i in range(horizon)]

        # Context features shape: (context_len, num_features) or (context_len,)
        batched_ctx = context_features[np.newaxis, ...]
        point_preds, quantiles = self.predict_batch(batched_ctx, horizon=horizon)

        p_pred = point_preds[0]
        q_pred = quantiles[0]

        meta = {
            "feature_set": request.feature_set,
            "context_len": len(context_features),
            "horizon": horizon,
            "device": str(self.device),
            "is_lora": self.adapter_path is not None,
        }
        if manifest is not None:
            meta["adapter_id"] = manifest.adapter_id
            meta["adapter_best_epoch"] = manifest.best_epoch
            meta["adapter_best_val_loss"] = manifest.best_val_loss
        if in_sample_warning is not None:
            meta["in_sample_warning"] = in_sample_warning

        return ForecastResult(
            symbol=request.symbol,
            timeframe=request.timeframe,
            forecast_origin_time=forecast_origin_time,
            target_timestamps=target_timestamps,
            point_forecast=p_pred,
            quantiles=q_pred,
            uncertainty_lower=q_pred[:, 0],   # q10
            uncertainty_upper=q_pred[:, -1],  # q90
            metadata=meta,
        )
