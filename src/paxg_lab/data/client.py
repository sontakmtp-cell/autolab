"""Binance Futures API client for PAXG Forecast Lab with rate limiting and exponential backoff."""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://fapi.binance.com"
MAX_RETRIES = 5
INITIAL_BACKOFF = 1.0


class BinanceFuturesClient:
    """HTTP Client for Binance USDⓈ-M Futures API."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: float = 15.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """Sends an HTTP GET request with retries and exponential backoff."""
        url = f"{self.base_url}{path}"
        if params:
            # Filter out None values
            clean_params = {k: v for k, v in params.items() if v is not None}
            url = f"{url}?{urllib.parse.urlencode(clean_params)}"

        headers = {
            "User-Agent": "PAXGForecastLab/1.0",
            "Accept": "application/json",
        }
        req = urllib.request.Request(url, headers=headers, method="GET")

        backoff = INITIAL_BACKOFF
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = resp.read().decode("utf-8")
                    return json.loads(data)
            except urllib.error.HTTPError as e:
                status_code = e.code
                err_body = e.read().decode("utf-8", errors="replace")
                logger.warning(
                    "Binance API HTTP error: %d on %s (attempt %d/%d): %s",
                    status_code, path, attempt, MAX_RETRIES, err_body
                )
                if status_code in (429, 418):
                    # Rate limit or ban - sleep longer
                    sleep_time = backoff * 4
                    logger.warning("Rate limit hit (%d). Backing off for %.1fs...", status_code, sleep_time)
                    time.sleep(sleep_time)
                elif 500 <= status_code < 600:
                    time.sleep(backoff)
                else:
                    raise RuntimeError(f"Binance API HTTP error {status_code}: {err_body}") from e
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                logger.warning("Network error on %s (attempt %d/%d): %s", path, attempt, MAX_RETRIES, e)
                time.sleep(backoff)

            backoff *= 2.0

        raise RuntimeError(f"Exceeded maximum retries ({MAX_RETRIES}) calling Binance API path {path}")

    def get_server_time(self) -> int:
        """Returns current Binance server time in milliseconds."""
        data = self._request("/fapi/v1/time")
        return int(data["serverTime"])

    def get_symbol_info(self, symbol: str = "PAXGUSDT") -> dict[str, Any]:
        """Returns exchange info and trading rules for symbol."""
        data = self._request("/fapi/v1/exchangeInfo")
        for sym_info in data.get("symbols", []):
            if sym_info.get("symbol") == symbol:
                return sym_info
        raise ValueError(f"Symbol '{symbol}' not found on Binance USDⓈ-M Futures.")

    def fetch_klines(
        self,
        symbol: str = "PAXGUSDT",
        interval: str = "1h",
        start_time: int | None = None,
        end_time: int | None = None,
        limit: int = 1500,
    ) -> list[list[Any]]:
        """Fetches market klines (candles)."""
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_time,
            "endTime": end_time,
            "limit": limit,
        }
        return self._request("/fapi/v1/klines", params)

    def fetch_mark_price_klines(
        self,
        symbol: str = "PAXGUSDT",
        interval: str = "1h",
        start_time: int | None = None,
        end_time: int | None = None,
        limit: int = 1500,
    ) -> list[list[Any]]:
        """Fetches mark price klines."""
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_time,
            "endTime": end_time,
            "limit": limit,
        }
        return self._request("/fapi/v1/markPriceKlines", params)

    def fetch_funding_rates(
        self,
        symbol: str = "PAXGUSDT",
        start_time: int | None = None,
        end_time: int | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Fetches realized funding rate history."""
        params = {
            "symbol": symbol,
            "startTime": start_time,
            "endTime": end_time,
            "limit": limit,
        }
        return self._request("/fapi/v1/fundingRate", params)
