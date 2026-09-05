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


def get_horizon_for_timeframe(timeframe: str) -> int:
    """Returns mandatory forecast horizon for the given timeframe."""
    tf = timeframe.lower().strip()
    if tf not in TIMEFRAME_HORIZONS:
        raise ValueError(
            f"Unsupported timeframe '{timeframe}'. Expected one of {list(TIMEFRAME_HORIZONS.keys())}."
        )
    return TIMEFRAME_HORIZONS[tf]
