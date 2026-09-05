"""Global constants and horizon conventions for PAXG Forecast Lab."""

from typing import Final

# Market and Model
SYMBOL: Final[str] = "PAXGUSDT"
MODEL_REPO: Final[str] = "google/timesfm-3.0-pytorch"
MODEL_REVISION: Final[str] = "43046b85ec22d584a13f8098c2ed39c889e129c2"

# Mandatory horizon convention
TIMEFRAME_HORIZONS: Final[dict[str, int]] = {
    "1h": 24,  # 24 candles = 24 hours
    "4h": 6,   # 6 candles = 24 hours
}

DEFAULT_CONTEXT_LENGTH: Final[int] = 256
ALLOWED_CONTEXT_LENGTHS: Final[tuple[int, ...]] = (128, 256, 512)

# Quantiles (TimesFM 3.0 outputs 9 quantiles, median at index 4)
QUANTILES: Final[list[float]] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
MEDIAN_QUANTILE_INDEX: Final[int] = 4

# LoRA Target Modules
DEFAULT_LORA_TARGETS: Final[list[str]] = ["query_proj", "value_proj"]
EXTENDED_LORA_TARGETS: Final[list[str]] = ["query_proj", "value_proj", "key_proj", "out_proj"]

# Safe float bounds
EPSILON: Final[float] = 1e-7

# Score v1 Constants
SCORE_VERSION: Final[int] = 1
TICK_SIZE: Final[float] = 0.01

# Time-decay weights for evaluation (0-6h: 50%, 6-12h: 30%, 12-24h: 20%)
# 1h: 24 steps -> 1-6 (0.50/6), 7-12 (0.30/6), 13-24 (0.20/12)
# 4h: 6 steps  -> 1 (0.50/1), 2-3 (0.30/2), 4-6 (0.20/3)
HORIZON_WEIGHTS: Final[dict[str, tuple[float, ...]]] = {
    "1h": tuple([0.50 / 6.0] * 6 + [0.30 / 6.0] * 6 + [0.20 / 12.0] * 12),
    "4h": tuple([0.50] + [0.30 / 2.0] * 2 + [0.20 / 3.0] * 3),
}


def get_horizon_for_timeframe(timeframe: str) -> int:
    """Returns mandatory forecast horizon for the given timeframe."""
    tf = timeframe.lower().strip()
    if tf not in TIMEFRAME_HORIZONS:
        raise ValueError(
            f"Unsupported timeframe '{timeframe}'. Expected one of {list(TIMEFRAME_HORIZONS.keys())}."
        )
    return TIMEFRAME_HORIZONS[tf]


def get_horizon_weights(timeframe: str) -> tuple[float, ...]:
    """Returns normalized time-decay weights across forecast steps for timeframe."""
    tf = timeframe.lower().strip()
    if tf not in HORIZON_WEIGHTS:
        raise ValueError(
            f"Unsupported timeframe '{timeframe}'. Expected one of {list(HORIZON_WEIGHTS.keys())}."
        )
    return HORIZON_WEIGHTS[tf]
