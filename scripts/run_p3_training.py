"""Empirical Phase P3 LoRA training on RTX 5060 Ti for both 1h and 4h timeframes."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
import time

# Safe encoding for Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import torch

from paxg_lab.constants import (
    MODEL_REPO,
    MODEL_REVISION,
    TIMEFRAME_HORIZONS,
)
from paxg_lab.data.snapshot import DatasetSnapshot, SNAPSHOTS_DIR
from paxg_lab.eval.predictor import TimesFM3Predictor
from paxg_lab.eval.types import ForecastRequest
from paxg_lab.model.manifest import AdapterManifest
from paxg_lab.model.store import AdapterStore, DEFAULT_ADAPTER_STORE_DIR
from paxg_lab.model.train_spec import TrainSpec
from paxg_lab.model.trainer import LoRATrainer
from timesfm3 import TimesFM3Torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def find_latest_snapshot(timeframe: str) -> DatasetSnapshot:
    """Finds and loads the latest snapshot for the given timeframe."""
    candidates = sorted(list(SNAPSHOTS_DIR.glob(f"paxgusdt_{timeframe}_*")))
    if not candidates:
        raise FileNotFoundError(f"No snapshot found for timeframe '{timeframe}' in {SNAPSHOTS_DIR}")
    latest_dir = candidates[-1]
    logger.info("Loading snapshot for %s from %s...", timeframe, latest_dir)
    return DatasetSnapshot.load(latest_dir, verify_hash=True)


def main() -> None:
    print("=" * 80)
    print("  PAXG FORECAST LAB — PHASE P3 MANUAL LoRA TRAINING (RTX 5060 Ti)")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"  Device:         {device} ({device_name})")
    print(f"  PyTorch:        {torch.__version__}")
    print(f"  Model:          {MODEL_REPO} (rev: {MODEL_REVISION[:8]})")
    print(f"  Horizons:       1h -> {TIMEFRAME_HORIZONS['1h']} steps; 4h -> {TIMEFRAME_HORIZONS['4h']} steps")
    print("=" * 80)

    # 1. Clean up any stale temp directories in store
    store = AdapterStore(base_dir=DEFAULT_ADAPTER_STORE_DIR)
    cleaned = store.cleanup_stale_temp_dirs(max_age_seconds=0.0)
    if cleaned:
        print(f"  Cleaned up {len(cleaned)} stale temp directories: {cleaned}")

    # 2. Load Snapshots
    print("\n>>> Loading 1h and 4h snapshots...")
    snap_1h = find_latest_snapshot("1h")
    snap_4h = find_latest_snapshot("4h")
    df_1h = snap_1h.to_dataframe("B")
    df_4h = snap_4h.to_dataframe("B")
    print(f"    1h snapshot candles: {len(df_1h)}, columns: {list(df_1h.columns)}")
    print(f"    4h snapshot candles: {len(df_4h)}, columns: {list(df_4h.columns)}")

    proof_data = {
        "execution_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": device,
        "device_name": device_name,
        "pytorch_version": torch.__version__,
        "base_model_repo": MODEL_REPO,
        "base_model_revision": MODEL_REVISION,
        "phase": "P3",
        "results": {},
    }

    # -----------------------------------------------------------------------
    # 3. Train 1h LoRA Adapter (Horizon = 24)
    # -----------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("  [1/2] TRAINING 1h LoRA ADAPTER (horizon = 24 steps = 24 hours)")
    print("-" * 80)

    base_model_1h = TimesFM3Torch.from_pretrained(MODEL_REPO, revision=MODEL_REVISION)
    base_model_1h.to(device)
    base_model_1h.eval()

    spec_1h = TrainSpec(
        timeframe="1h",
        horizon=24,
        feature_set="B",
        context_len=256,
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.10,
        learning_rate=5e-5,
        max_epochs=2,
        max_samples_per_epoch=512,
        batch_size=2,
        gradient_accumulation_steps=8,
        early_stopping_patience=2,
        history_days=365,
        seed=42,
    )
    print(f"  Spec: {spec_1h.to_dict()}")

    trainer_1h = LoRATrainer(base_model=base_model_1h, spec=spec_1h, device=device)
    t0_1h = time.time()
    res_1h = trainer_1h.train(
        features_df=df_1h,
        snapshot_hash=snap_1h.metadata.sha256,
        fold_id=1,
    )
    t_train_1h = time.time() - t0_1h
    print(f"    1h Training finished in {t_train_1h:.1f}s")
    print(f"    Best epoch: {res_1h.best_epoch}, Best val loss: {res_1h.best_val_loss:.6f}")

    # Atomic Save via AdapterStore
    saved_path_1h = store.save_adapter(res_1h.trained_model, res_1h.manifest)
    print(f"    Adapter saved atomically to: {saved_path_1h}")

    # Smoke Test & Verification via Predictor
    print("    Running verification forward pass with TimesFM3Predictor...")
    pred_1h = TimesFM3Predictor(device=device, adapter_path=saved_path_1h)
    ctx_sample_1h = df_1h.tail(256)[list(res_1h.manifest.feature_columns)].to_numpy(dtype=np.float32).copy()
    req_1h = ForecastRequest(
        symbol="PAXGUSDT",
        timeframe="1h",
        feature_set="B",
        context_len=256,
        horizon=24,
        adapter_path=str(saved_path_1h),
    )
    fc_1h = pred_1h.forecast_request(req_1h, ctx_sample_1h, int(df_1h["open_time"].iloc[-1]))
    assert fc_1h.point_forecast.shape == (24,)
    assert fc_1h.quantiles.shape == (24, 9)
    # Check monotonic quantiles
    assert np.all(np.diff(fc_1h.quantiles, axis=-1) >= -1e-6)
    print(f"    1h Verification passed: output shape {fc_1h.quantiles.shape}, quantiles monotonic.")

    proof_data["results"]["1h"] = {
        "adapter_id": res_1h.manifest.adapter_id,
        "adapter_path": str(saved_path_1h),
        "horizon": res_1h.manifest.horizon,
        "timeframe": res_1h.manifest.timeframe,
        "best_epoch": res_1h.best_epoch,
        "best_val_loss": res_1h.best_val_loss,
        "final_train_loss": res_1h.final_train_loss,
        "training_time_sec": t_train_1h,
        "history": res_1h.history,
        "file_hashes": res_1h.manifest.file_hashes,
        "test_forecast_point_median": float(np.mean(fc_1h.point_forecast)),
    }

    # Clean up GPU memory
    del base_model_1h, trainer_1h, pred_1h
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # 4. Train 4h LoRA Adapter (Horizon = 6)
    # -----------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("  [2/2] TRAINING 4h LoRA ADAPTER (horizon = 6 steps = 24 hours)")
    print("-" * 80)

    base_model_4h = TimesFM3Torch.from_pretrained(MODEL_REPO, revision=MODEL_REVISION)
    base_model_4h.to(device)
    base_model_4h.eval()

    spec_4h = TrainSpec(
        timeframe="4h",
        horizon=6,
        feature_set="B",
        context_len=256,
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.10,
        learning_rate=5e-5,
        max_epochs=2,
        max_samples_per_epoch=512,
        batch_size=2,
        gradient_accumulation_steps=8,
        early_stopping_patience=2,
        history_days=365,
        seed=42,
    )
    print(f"  Spec: {spec_4h.to_dict()}")

    trainer_4h = LoRATrainer(base_model=base_model_4h, spec=spec_4h, device=device)
    t0_4h = time.time()
    res_4h = trainer_4h.train(
        features_df=df_4h,
        snapshot_hash=snap_4h.metadata.sha256,
        fold_id=1,
    )
    t_train_4h = time.time() - t0_4h
    print(f"    4h Training finished in {t_train_4h:.1f}s")
    print(f"    Best epoch: {res_4h.best_epoch}, Best val loss: {res_4h.best_val_loss:.6f}")

    # Atomic Save via AdapterStore
    saved_path_4h = store.save_adapter(res_4h.trained_model, res_4h.manifest)
    print(f"    Adapter saved atomically to: {saved_path_4h}")

    # Smoke Test & Verification via Predictor
    print("    Running verification forward pass with TimesFM3Predictor...")
    pred_4h = TimesFM3Predictor(device=device, adapter_path=saved_path_4h)
    ctx_sample_4h = df_4h.tail(256)[list(res_4h.manifest.feature_columns)].to_numpy(dtype=np.float32).copy()
    req_4h = ForecastRequest(
        symbol="PAXGUSDT",
        timeframe="4h",
        feature_set="B",
        context_len=256,
        horizon=6,
        adapter_path=str(saved_path_4h),
    )
    fc_4h = pred_4h.forecast_request(req_4h, ctx_sample_4h, int(df_4h["open_time"].iloc[-1]))
    assert fc_4h.point_forecast.shape == (6,)
    assert fc_4h.quantiles.shape == (6, 9)
    # Check monotonic quantiles
    assert np.all(np.diff(fc_4h.quantiles, axis=-1) >= -1e-6)
    print(f"    4h Verification passed: output shape {fc_4h.quantiles.shape}, quantiles monotonic.")

    proof_data["results"]["4h"] = {
        "adapter_id": res_4h.manifest.adapter_id,
        "adapter_path": str(saved_path_4h),
        "horizon": res_4h.manifest.horizon,
        "timeframe": res_4h.manifest.timeframe,
        "best_epoch": res_4h.best_epoch,
        "best_val_loss": res_4h.best_val_loss,
        "final_train_loss": res_4h.final_train_loss,
        "training_time_sec": t_train_4h,
        "history": res_4h.history,
        "file_hashes": res_4h.manifest.file_hashes,
        "test_forecast_point_median": float(np.mean(fc_4h.point_forecast)),
    }

    # -----------------------------------------------------------------------
    # 5. Save Evidence Proof JSON
    # -----------------------------------------------------------------------
    proof_path = Path("docs/paxg-lab/phases/p3_lora_proof.json")
    proof_path.parent.mkdir(parents=True, exist_ok=True)
    with open(proof_path, "w", encoding="utf-8") as f:
        json.dump(proof_data, f, indent=2, ensure_ascii=False)
    print(f"\n>>> Saved P3 proof to: {proof_path}")

    print("\n" + "=" * 80)
    print("  PHASE P3 TRAINING SUMMARY COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"  1h Adapter: {res_1h.manifest.adapter_id} (val_loss: {res_1h.best_val_loss:.6f}, time: {t_train_1h:.1f}s)")
    print(f"  4h Adapter: {res_4h.manifest.adapter_id} (val_loss: {res_4h.best_val_loss:.6f}, time: {t_train_4h:.1f}s)")
    print("=" * 80)


if __name__ == "__main__":
    main()
