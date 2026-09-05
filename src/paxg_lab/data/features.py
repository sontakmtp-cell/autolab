"""Feature engineering for PAXG Forecast Lab: Feature Sets A, B, and C with zero future leakage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureSpec:
    name: str  # "A", "B", or "C"
    columns: tuple[str, ...]
    version: int = 1

    @property
    def num_variates(self) -> int:
        return len(self.columns)


FEATURE_SPECS: dict[str, FeatureSpec] = {
    "A": FeatureSpec(
        name="A",
        columns=("close",),
        version=1,
    ),
    "B": FeatureSpec(
        name="B",
        columns=(
            "close",
            "log1p_quote_volume",
            "log_hl_ratio",
            "ret_oc",
            "taker_buy_ratio",
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
        ),
        version=1,
    ),
    "C": FeatureSpec(
        name="C",
        columns=(
            "close",
            "log1p_quote_volume",
            "log_hl_ratio",
            "ret_oc",
            "taker_buy_ratio",
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "mark_close_basis",
            "realized_funding_rate",
        ),
        version=1,
    ),
}


def build_features(
    df_klines: pd.DataFrame,
    df_mark: pd.DataFrame | None = None,
    df_funding: pd.DataFrame | None = None,
    feature_set: str = "B",
) -> tuple[np.ndarray, FeatureSpec, list[int]]:
    """Constructs clean feature matrices for feature set A, B, or C.

    Args:
        df_klines: Kline DataFrame sorted by open_time.
        df_mark: Mark price kline DataFrame sorted by open_time. Required for set C.
        df_funding: Funding rate history DataFrame. Required for set C.
        feature_set: "A", "B", or "C".

    Returns:
        Tuple of:
          - values: NumPy array of shape (num_timestamps, num_features)
          - spec: FeatureSpec instance
          - timestamps: list of open_time timestamps
    """
    f_set = feature_set.upper().strip()
    if f_set not in FEATURE_SPECS:
        raise ValueError(f"Unknown feature_set '{feature_set}'. Expected 'A', 'B', or 'C'.")

    spec = FEATURE_SPECS[f_set]

    if df_klines.empty:
        raise ValueError("df_klines is empty.")

    df = df_klines.copy()
    df["open_time"] = df["open_time"].astype(np.int64)
    df = df.sort_values("open_time").reset_index(drop=True)

    # Feature Set A: Raw close price
    df["close"] = df["close"].astype(np.float64)

    if f_set in ("B", "C"):
        # 1. log1p(quote_volume)
        q_vol = np.maximum(df["quote_volume"].to_numpy(dtype=np.float64), 0.0)
        df["log1p_quote_volume"] = np.log1p(q_vol)

        # 2. log(high / low)
        h = df["high"].to_numpy(dtype=np.float64)
        l = np.maximum(df["low"].to_numpy(dtype=np.float64), 1e-8)
        hl_ratio = np.maximum(h / l, 1.0)
        df["log_hl_ratio"] = np.log(hl_ratio)

        # 3. (close - open) / open
        o = np.maximum(df["open"].to_numpy(dtype=np.float64), 1e-8)
        c = df["close"].to_numpy(dtype=np.float64)
        df["ret_oc"] = (c - o) / o

        # 4. taker_buy_ratio = taker_buy_volume / volume (neutral 0.5 if volume == 0)
        vol = df["volume"].to_numpy(dtype=np.float64)
        tb_vol = df["taker_buy_volume"].to_numpy(dtype=np.float64)
        taker_ratio = np.where(vol > 1e-8, tb_vol / np.maximum(vol, 1e-8), 0.5)
        df["taker_buy_ratio"] = np.clip(taker_ratio, 0.0, 1.0)

        # 5. Calendar cyclical encodings (UTC timestamps)
        # Convert open_time (ms) to UTC datetime
        dt = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        hours = dt.dt.hour + dt.dt.minute / 60.0
        dows = dt.dt.dayofweek

        df["hour_sin"] = np.sin(2.0 * np.pi * hours / 24.0)
        df["hour_cos"] = np.cos(2.0 * np.pi * hours / 24.0)
        df["dow_sin"] = np.sin(2.0 * np.pi * dows / 7.0)
        df["dow_cos"] = np.cos(2.0 * np.pi * dows / 7.0)

    if f_set == "C":
        if df_mark is None or df_mark.empty:
            raise ValueError("df_mark is required for feature set C.")
        if df_funding is None or df_funding.empty:
            raise ValueError("df_funding is required for feature set C.")

        # Merge mark price on open_time
        mark_clean = df_mark[["open_time", "close"]].rename(columns={"close": "mark_close"}).copy()
        mark_clean["open_time"] = mark_clean["open_time"].astype(np.int64)
        mark_clean["mark_close"] = mark_clean["mark_close"].astype(np.float64)

        df = pd.merge(df, mark_clean, on="open_time", how="left")
        # If any missing mark close, fill with regular close
        df["mark_close"] = df["mark_close"].fillna(df["close"])
        df["mark_close_basis"] = (df["mark_close"] - df["close"]) / np.maximum(df["close"], 1e-8)

        # Merge funding rate strictly using backward asof: fundingTime <= open_time (no lookahead)
        funding_sorted = df_funding.sort_values("funding_time").copy()
        funding_sorted["funding_time"] = funding_sorted["funding_time"].astype(np.int64)
        funding_sorted["realized_funding_rate"] = funding_sorted["funding_rate"].astype(np.float64)

        merged = pd.merge_asof(
            df,
            funding_sorted[["funding_time", "realized_funding_rate"]],
            left_on="open_time",
            right_on="funding_time",
            direction="backward",
        )
        df["realized_funding_rate"] = merged["realized_funding_rate"].fillna(0.0)

    # Extract columns according to specification
    feature_cols = list(spec.columns)
    for col in feature_cols:
        if col not in df.columns:
            raise RuntimeError(f"Missing calculated feature column: '{col}'")

    feature_matrix = df[feature_cols].to_numpy(dtype=np.float32)
    timestamps = df["open_time"].tolist()

    # Final sanity check: no NaNs or Infs
    if not np.isfinite(feature_matrix).all():
        raise RuntimeError(f"Non-finite values found in feature matrix for Set {spec.name}.")

    return feature_matrix, spec, timestamps
