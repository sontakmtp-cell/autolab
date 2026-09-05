"""Data validation, OHLC checks, gap detection, and 4h-vs-1h cross-validation."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

INTERVAL_MS: dict[str, int] = {
    "1h": 3600 * 1000,
    "4h": 4 * 3600 * 1000,
}


def validate_ohlc_integrity(df: pd.DataFrame) -> dict[str, Any]:
    """Validates OHLC bounds, positive prices, non-negative volumes, and uniqueness."""
    if df.empty:
        return {"total_rows": 0, "is_valid": True, "errors": []}

    errors: list[str] = []

    # Check required columns
    req_cols = ["open_time", "open", "high", "low", "close", "volume"]
    for col in req_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' for OHLC validation.")

    # 1. Uniqueness
    dup_mask = df.duplicated(subset=["open_time"])
    dup_count = int(dup_mask.sum())
    if dup_count > 0:
        errors.append(f"Found {dup_count} duplicate open_time timestamps.")

    # 2. Positive prices
    non_pos = (df["open"] <= 0) | (df["high"] <= 0) | (df["low"] <= 0) | (df["close"] <= 0)
    non_pos_count = int(non_pos.sum())
    if non_pos_count > 0:
        errors.append(f"Found {non_pos_count} rows with non-positive price.")

    # 3. High/Low logic: high >= max(open, close), low <= min(open, close)
    invalid_high = df["high"] < np.maximum(df["open"], df["close"])
    invalid_low = df["low"] > np.minimum(df["open"], df["close"])
    bad_hl_count = int((invalid_high | invalid_low).sum())
    if bad_hl_count > 0:
        errors.append(f"Found {bad_hl_count} rows where high < max(open, close) or low > min(open, close).")

    # 4. Non-negative volumes
    neg_vol = df["volume"] < 0
    if "quote_volume" in df.columns:
        neg_vol = neg_vol | (df["quote_volume"] < 0)
    neg_vol_count = int(neg_vol.sum())
    if neg_vol_count > 0:
        errors.append(f"Found {neg_vol_count} rows with negative volume.")

    return {
        "total_rows": len(df),
        "is_valid": len(errors) == 0,
        "duplicate_timestamps": dup_count,
        "non_positive_prices": non_pos_count,
        "invalid_high_low": bad_hl_count,
        "negative_volumes": neg_vol_count,
        "errors": errors,
    }


def detect_time_gaps(df: pd.DataFrame, interval: str = "1h") -> list[dict[str, Any]]:
    """Detects missing intervals between consecutive sorted candles.

    Returns list of gap dictionaries with start, end, and missing count.
    """
    if len(df) < 2:
        return []

    step_ms = INTERVAL_MS.get(interval)
    if step_ms is None:
        raise ValueError(f"Unsupported interval '{interval}' for gap detection.")

    times = df["open_time"].to_numpy()
    diffs = np.diff(times)

    gaps: list[dict[str, Any]] = []
    bad_indices = np.where(diffs != step_ms)[0]

    for idx in bad_indices:
        t_prev = int(times[idx])
        t_next = int(times[idx + 1])
        actual_diff = t_next - t_prev
        missing_count = int((actual_diff // step_ms) - 1)
        if missing_count > 0:
            gaps.append({
                "gap_start_open_time": t_prev,
                "gap_end_open_time": t_next,
                "missing_candles": missing_count,
                "missing_duration_ms": actual_diff - step_ms,
            })

    return gaps


def cross_validate_4h_with_1h(df_1h: pd.DataFrame, df_4h: pd.DataFrame) -> dict[str, Any]:
    """Cross-validates 4h candles by aggregating the 4 corresponding 1h candles."""
    if df_1h.empty or df_4h.empty:
        return {"matched_candles": 0, "discrepancies": 0, "perfect_match_ratio": 1.0}

    # Index 1h by open_time for O(1) lookup
    df_1h_indexed = df_1h.set_index("open_time")
    h1_step = INTERVAL_MS["1h"]

    matched = 0
    discrepancies = 0
    discrepancy_details = []

    for _, row_4h in df_4h.iterrows():
        t4 = int(row_4h["open_time"])
        expected_1h_times = [t4 + i * h1_step for i in range(4)]

        # Check if all 4 consecutive 1h candles exist
        if all(t in df_1h_indexed.index for t in expected_1h_times):
            matched += 1
            chunk_1h = df_1h_indexed.loc[expected_1h_times]

            exp_open = chunk_1h.iloc[0]["open"]
            exp_high = chunk_1h["high"].max()
            exp_low = chunk_1h["low"].min()
            exp_close = chunk_1h.iloc[-1]["close"]
            exp_vol = chunk_1h["volume"].sum()

            tol = 1e-4
            is_match = (
                abs(row_4h["open"] - exp_open) < tol and
                abs(row_4h["high"] - exp_high) < tol and
                abs(row_4h["low"] - exp_low) < tol and
                abs(row_4h["close"] - exp_close) < tol and
                abs(row_4h["volume"] - exp_vol) < max(tol, 0.01 * exp_vol)
            )

            if not is_match:
                discrepancies += 1
                if len(discrepancy_details) < 5:
                    discrepancy_details.append({
                        "open_time_4h": t4,
                        "actual_4h": [row_4h["open"], row_4h["high"], row_4h["low"], row_4h["close"], row_4h["volume"]],
                        "aggregated_1h": [exp_open, exp_high, exp_low, exp_close, exp_vol],
                    })

    match_ratio = (matched - discrepancies) / matched if matched > 0 else 0.0

    return {
        "total_4h_candles": len(df_4h),
        "fully_covered_by_1h": matched,
        "discrepancies": discrepancies,
        "match_ratio": match_ratio,
        "sample_discrepancies": discrepancy_details,
    }


def generate_data_quality_report(df_1h: pd.DataFrame, df_4h: pd.DataFrame) -> dict[str, Any]:
    """Compiles complete data quality audit report for 1h and 4h data."""
    report_1h = validate_ohlc_integrity(df_1h)
    report_4h = validate_ohlc_integrity(df_4h)

    gaps_1h = detect_time_gaps(df_1h, "1h")
    gaps_4h = detect_time_gaps(df_4h, "4h")

    cross_check = cross_validate_4h_with_1h(df_1h, df_4h)

    is_healthy = (
        report_1h["is_valid"] and
        report_4h["is_valid"] and
        len(gaps_1h) == 0 and
        len(gaps_4h) == 0 and
        cross_check["discrepancies"] == 0
    )

    return {
        "status": "HEALTHY" if is_healthy else "WARNING",
        "1h": {
            "candle_count": len(df_1h),
            "start_time": int(df_1h["open_time"].min()) if not df_1h.empty else None,
            "end_time": int(df_1h["open_time"].max()) if not df_1h.empty else None,
            "ohlc_valid": report_1h["is_valid"],
            "gaps_count": len(gaps_1h),
            "gaps": gaps_1h,
        },
        "4h": {
            "candle_count": len(df_4h),
            "start_time": int(df_4h["open_time"].min()) if not df_4h.empty else None,
            "end_time": int(df_4h["open_time"].max()) if not df_4h.empty else None,
            "ohlc_valid": report_4h["is_valid"],
            "gaps_count": len(gaps_4h),
            "gaps": gaps_4h,
        },
        "cross_validation_4h_vs_1h": cross_check,
    }
