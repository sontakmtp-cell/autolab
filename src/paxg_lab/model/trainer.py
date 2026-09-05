"""Manual LoRA training pipeline for TimesFM 3.0 on PAXGUSDT."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from peft import PeftModel

from ..constants import (
    MODEL_REPO,
    MODEL_REVISION,
    get_horizon_for_timeframe,
)
from ..data.features import FEATURE_SPECS
from ..data.split import calculate_split_plan, extract_windows
from .lora import build_lora_timesfm3, verify_lora_parameters
from .loss import combined_forecast_loss
from .manifest import AdapterManifest
from .train_spec import TrainSpec

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """Artifacts, provenance, and loss trajectory produced by LoRATrainer."""

    trained_model: PeftModel
    manifest: AdapterManifest
    history: list[dict[str, Any]]
    best_epoch: int
    best_val_loss: float
    final_train_loss: float
    train_spec: TrainSpec
    training_range: dict[str, Any]
    total_training_time_sec: float


class LoRATrainer:
    """Manual LoRA trainer supporting full hyperparameter customization, early stopping,

    and strict horizon pair locking (1h=24, 4h=6).
    """

    def __init__(
        self,
        base_model: nn.Module,
        spec: TrainSpec,
        device: str | torch.device | None = None,
    ):
        self.base_model = base_model
        self.spec = spec

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.base_model.to(self.device)

        # Ensure base model is in eval mode before adapter attachment
        self.base_model.eval()

        logger.info(
            "Initialized LoRATrainer on %s for timeframe '%s', horizon=%d, feature_set='%s', rank=%d",
            self.device,
            self.spec.timeframe,
            self.spec.horizon,
            self.spec.feature_set,
            self.spec.lora_r,
        )

    def prepare_dataset(
        self,
        features_df: pd.DataFrame,
        fold_id: int = 1,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """Prepares leak-free sliding training and validation early-stopping windows.

        Returns:
            (train_contexts, train_futures, val_contexts, val_futures, metadata)
        """
        # 1. Feature selection
        f_spec = FEATURE_SPECS[self.spec.feature_set]
        missing_cols = [c for c in f_spec.columns if c not in features_df.columns]
        if missing_cols:
            raise ValueError(
                f"features_df is missing columns for feature set '{self.spec.feature_set}': {missing_cols}"
            )

        feat_matrix = features_df[list(f_spec.columns)].to_numpy(dtype=np.float32)
        target_series = features_df["close"].to_numpy(dtype=np.float32)
        timestamps = features_df["open_time"].to_numpy(dtype=np.int64)

        # 2. Split boundaries
        split_plan = calculate_split_plan(
            total_candles=len(features_df),
            timeframe=self.spec.timeframe,
        )

        selected_fold = None
        for fold in split_plan.eval_folds:
            if fold.fold_id == fold_id:
                selected_fold = fold
                break

        if selected_fold is None:
            selected_fold = split_plan.eval_folds[0]

        train_start = selected_fold.train_start
        train_end = selected_fold.train_end

        # Restrict history if requested (e.g. 180 or 365 days)
        if isinstance(self.spec.history_days, int):
            candles_per_day = 24 if self.spec.timeframe == "1h" else 6
            max_candles = self.spec.history_days * candles_per_day
            if train_end - train_start > max_candles:
                train_start = max(train_end - max_candles, 0)

        # 3. Extract training windows
        train_ctx, train_fut, train_origins = extract_windows(
            features=feat_matrix,
            targets=target_series,
            context_len=self.spec.context_len,
            horizon=self.spec.horizon,
            start_idx=train_start,
            end_idx=train_end,
            step=1,
            timestamps=timestamps,
            timeframe=self.spec.timeframe,
        )

        if len(train_ctx) == 0:
            raise ValueError(
                f"Insufficient training samples extracted for timeframe '{self.spec.timeframe}', "
                f"context={self.spec.context_len}, horizon={self.spec.horizon} in range [{train_start}, {train_end}]."
            )

        # 4. Extract validation early stopping windows
        val_ctx, val_fut, val_origins = extract_windows(
            features=feat_matrix,
            targets=target_series,
            context_len=self.spec.context_len,
            horizon=self.spec.horizon,
            start_idx=selected_fold.val_early_stop_start,
            end_idx=selected_fold.val_early_stop_end,
            step=1,
            timestamps=timestamps,
            timeframe=self.spec.timeframe,
        )

        if len(val_ctx) == 0:
            raise ValueError(
                f"Insufficient validation samples in early stop range "
                f"[{selected_fold.val_early_stop_start}, {selected_fold.val_early_stop_end}]."
            )

        train_time_range = {
            "fold_id": selected_fold.fold_id,
            "train_start_idx": train_start,
            "train_end_idx": train_end,
            "train_start_time_ms": int(timestamps[train_origins[0] - self.spec.context_len + 1]),
            "train_end_time_ms": int(timestamps[train_origins[-1] + self.spec.horizon]),
            "val_start_time_ms": int(timestamps[val_origins[0] - self.spec.context_len + 1]),
            "val_end_time_ms": int(timestamps[val_origins[-1] + self.spec.horizon]),
            "num_train_windows": len(train_ctx),
            "num_val_windows": len(val_ctx),
        }

        logger.info(
            "Prepared dataset: %d train windows (origins %d to %d), %d val windows (origins %d to %d)",
            len(train_ctx),
            train_origins[0],
            train_origins[-1],
            len(val_ctx),
            val_origins[0],
            val_origins[-1],
        )

        return train_ctx, train_fut, val_ctx, val_fut, train_time_range

    def _forward_pass(
        self,
        peft_model: PeftModel,
        batch_ctx: torch.Tensor,
    ) -> torch.Tensor:
        """Executes forward_decode through TimesFM3 with active LoRA gradients."""
        # Unpack inner model if wrapped in PEFT
        model_core = peft_model
        if hasattr(model_core, "base_model"):
            model_core = model_core.base_model
        if hasattr(model_core, "model"):
            model_core = model_core.model

        # batch_ctx shape: (batch_size, num_features, context_len)
        raw_out = model_core.forward_decode(target=batch_ctx, horizon=self.spec.horizon)
        # raw_out: (batch_size, num_features, horizon, num_quantiles)
        # Target variable is primary variate (close price at index 0)
        return raw_out[:, 0, :, :]

    def train(
        self,
        features_df: pd.DataFrame,
        snapshot_hash: str = "",
        fold_id: int = 1,
        adapter_id: str | None = None,
    ) -> TrainingResult:
        """Runs the complete training loop, early stopping, and best checkpoint restoration."""
        start_wall_time = time.time()

        # 1. Deterministic seeds
        torch.manual_seed(self.spec.seed)
        np.random.seed(self.spec.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.spec.seed)

        # 2. Attach LoRA adapter
        logger.info("Attaching LoRA adapter to base model (r=%d, alpha=%s)...", self.spec.lora_r, self.spec.lora_alpha)
        peft_model = build_lora_timesfm3(
            model=self.base_model,
            lora_r=self.spec.lora_r,
            lora_alpha=self.spec.lora_alpha or (2 * self.spec.lora_r),
            lora_dropout=self.spec.lora_dropout,
            extended_targets=self.spec.extended_targets,
            custom_targets=self.spec.custom_targets,
        )
        param_verification = verify_lora_parameters(peft_model)
        logger.info(
            "LoRA parameter verification: %d trainable params (%.3f%%), base frozen: %s",
            param_verification["trainable_params"],
            param_verification["trainable_percentage"],
            param_verification["is_base_frozen"],
        )

        # 3. Prepare data
        train_ctx, train_fut, val_ctx, val_fut, data_meta = self.prepare_dataset(
            features_df=features_df,
            fold_id=fold_id,
        )

        num_train_samples = len(train_ctx)
        effective_samples = min(num_train_samples, self.spec.max_samples_per_epoch)
        steps_per_epoch = math.ceil(effective_samples / self.spec.batch_size)
        total_steps = steps_per_epoch * self.spec.max_epochs
        warmup_steps = max(1, int(total_steps * self.spec.warmup_ratio))

        # 4. Optimizer and Warmup Scheduler
        optimizer = torch.optim.AdamW(
            peft_model.parameters(),
            lr=self.spec.learning_rate,
            weight_decay=self.spec.weight_decay,
        )

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step + 1) / float(warmup_steps)
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # 5. Training State
        best_val_loss = float("inf")
        best_epoch = 0
        patience_counter = 0
        best_lora_state: dict[str, torch.Tensor] | None = None
        history: list[dict[str, Any]] = []
        global_step = 0
        final_train_loss = 0.0

        rng = np.random.default_rng(self.spec.seed)

        logger.info(
            "Starting LoRA training: max_epochs=%d, batch_size=%d, grad_accum=%d (effective=%d), total_steps=%d",
            self.spec.max_epochs,
            self.spec.batch_size,
            self.spec.gradient_accumulation_steps,
            self.spec.effective_batch_size,
            total_steps,
        )

        for epoch in range(1, self.spec.max_epochs + 1):
            epoch_start = time.time()
            peft_model.train()

            # Randomly shuffle and cap sliding windows for this epoch
            epoch_indices = rng.permutation(num_train_samples)[:effective_samples]
            epoch_loss_sum = 0.0
            epoch_loss_batches = 0

            optimizer.zero_grad()

            for i in range(0, len(epoch_indices), self.spec.batch_size):
                batch_idx = epoch_indices[i : i + self.spec.batch_size]
                b_ctx_np = train_ctx[batch_idx]  # (B, context_len, num_features)
                b_fut_np = train_fut[batch_idx]  # (B, horizon)

                # Transpose to (B, num_features, context_len) for TimesFM
                b_ctx_tensor = torch.from_numpy(np.transpose(b_ctx_np, (0, 2, 1))).to(self.device)
                b_fut_tensor = torch.from_numpy(b_fut_np).to(self.device)

                # Last close price in context window (close is index 0)
                p0 = b_ctx_tensor[:, 0, -1]

                # Forward pass through model with active LoRA
                preds = self._forward_pass(peft_model, b_ctx_tensor)

                # Combined loss: MAE + 9-quantile Pinball in float32, normalized by p0
                loss, loss_metrics = combined_forecast_loss(
                    predictions=preds,
                    targets=b_fut_tensor,
                    last_context_price=p0,
                )

                # Gradient accumulation scaling
                loss_scaled = loss / self.spec.gradient_accumulation_steps
                loss_scaled.backward()

                epoch_loss_sum += loss.item()
                epoch_loss_batches += 1
                global_step += 1

                # Optimizer step every gradient_accumulation_steps or at epoch end
                if (i // self.spec.batch_size + 1) % self.spec.gradient_accumulation_steps == 0 or (
                    i + self.spec.batch_size >= len(epoch_indices)
                ):
                    torch.nn.utils.clip_grad_norm_(
                        peft_model.parameters(),
                        self.spec.grad_clip_norm,
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            avg_train_loss = epoch_loss_sum / max(1, epoch_loss_batches)
            final_train_loss = avg_train_loss

            # Validation phase on 14-day early stop segment
            peft_model.eval()
            val_loss_sum = 0.0
            val_batches = 0
            val_batch_size = max(self.spec.batch_size * 2, 4)

            with torch.no_grad():
                for v_i in range(0, len(val_ctx), val_batch_size):
                    v_ctx_np = val_ctx[v_i : v_i + val_batch_size]
                    v_fut_np = val_fut[v_i : v_i + val_batch_size]

                    v_ctx_tensor = torch.from_numpy(np.transpose(v_ctx_np, (0, 2, 1))).to(self.device)
                    v_fut_tensor = torch.from_numpy(v_fut_np).to(self.device)
                    v_p0 = v_ctx_tensor[:, 0, -1]

                    v_preds = self._forward_pass(peft_model, v_ctx_tensor)
                    v_loss, _ = combined_forecast_loss(
                        predictions=v_preds,
                        targets=v_fut_tensor,
                        last_context_price=v_p0,
                    )
                    val_loss_sum += v_loss.item()
                    val_batches += 1

            avg_val_loss = val_loss_sum / max(1, val_batches)
            epoch_duration = time.time() - epoch_start

            epoch_record = {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "learning_rate": scheduler.get_last_lr()[0],
                "duration_sec": epoch_duration,
            }
            history.append(epoch_record)

            logger.info(
                "Epoch %d/%d: train_loss=%.6f, val_loss=%.6f, lr=%.2e (%.1fs)",
                epoch,
                self.spec.max_epochs,
                avg_train_loss,
                avg_val_loss,
                scheduler.get_last_lr()[0],
                epoch_duration,
            )

            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                patience_counter = 0
                # Snapshot best LoRA weights to host CPU memory
                best_lora_state = {
                    k: v.cpu().clone()
                    for k, v in peft_model.state_dict().items()
                    if "lora" in k.lower()
                }
                logger.info("  --> New best validation loss: %.6f at epoch %d", best_val_loss, best_epoch)
            else:
                patience_counter += 1
                logger.info(
                    "  --> No improvement in val_loss. Patience: %d/%d",
                    patience_counter,
                    self.spec.early_stopping_patience,
                )
                if patience_counter >= self.spec.early_stopping_patience:
                    logger.info("Early stopping triggered at epoch %d.", epoch)
                    break

        # 6. Restore best checkpoint weights into model
        if best_lora_state is not None:
            logger.info("Restoring best checkpoint weights from epoch %d (val_loss=%.6f)...", best_epoch, best_val_loss)
            current_dict = peft_model.state_dict()
            current_dict.update(best_lora_state)
            peft_model.load_state_dict(current_dict)

        peft_model.eval()

        total_elapsed = time.time() - start_wall_time

        # 7. Construct Adapter Manifest
        if adapter_id is None:
            ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            adapter_id = (
                f"paxg_{self.spec.timeframe}_r{self.spec.lora_r}_set{self.spec.feature_set}_seed{self.spec.seed}_{ts_str}"
            )

        f_spec = FEATURE_SPECS[self.spec.feature_set]
        manifest = AdapterManifest(
            adapter_id=adapter_id,
            timeframe=self.spec.timeframe,
            horizon=self.spec.horizon,
            context_len=self.spec.context_len,
            feature_set=self.spec.feature_set,
            feature_columns=list(f_spec.columns),
            base_model_repo=MODEL_REPO,
            base_model_revision=MODEL_REVISION,
            lora_config={
                "r": self.spec.lora_r,
                "lora_alpha": self.spec.lora_alpha or (2 * self.spec.lora_r),
                "lora_dropout": self.spec.lora_dropout,
                "target_modules": self.spec.custom_targets or (
                    ["query_proj", "value_proj", "key_proj", "out_proj"]
                    if self.spec.extended_targets
                    else ["query_proj", "value_proj"]
                ),
                "bias": "none",
            },
            train_spec=self.spec.to_dict(),
            training_range={
                "start_time_ms": data_meta["train_start_time_ms"],
                "end_time_ms": data_meta["train_end_time_ms"],
                "val_start_time_ms": data_meta["val_start_time_ms"],
                "val_end_time_ms": data_meta["val_end_time_ms"],
                "fold_id": data_meta["fold_id"],
                "num_train_windows": data_meta["num_train_windows"],
                "num_val_windows": data_meta["num_val_windows"],
            },
            snapshot_hash=snapshot_hash,
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            metrics={
                "best_val_loss": best_val_loss,
                "final_train_loss": final_train_loss,
                "total_training_time_sec": total_elapsed,
                "param_verification": param_verification,
            },
        )

        return TrainingResult(
            trained_model=peft_model,
            manifest=manifest,
            history=history,
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            final_train_loss=final_train_loss,
            train_spec=self.spec,
            training_range=data_meta,
            total_training_time_sec=total_elapsed,
        )
