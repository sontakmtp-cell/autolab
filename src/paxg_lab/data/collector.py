"""High-level market data collector and snapshot creator for PAXGUSDT Futures."""

from __future__ import annotations

import logging
import time
from typing import Any

import pandas as pd

from .client import BinanceFuturesClient
from .features import build_features
from .snapshot import DatasetSnapshot
from .storage import MarketDataStorage
from .validator import generate_data_quality_report

logger = logging.getLogger(__name__)

SYMBOL = "PAXGUSDT"
BATCH_SIZE = 1500


class MarketDataCollector:
    """Coordinates Binance data collection, SQLite storage, and snapshot creation."""

    def __init__(
        self,
        client: BinanceFuturesClient | None = None,
        storage: MarketDataStorage | None = None,
    ):
        self.client = client or BinanceFuturesClient()
        self.storage = storage or MarketDataStorage()

    def sync_klines(
        self,
        symbol: str = SYMBOL,
        interval: str = "1h",
        force_from_start: bool = False,
    ) -> int:
        """Fetches all missing klines from Binance and writes to SQLite."""
        server_time = self.client.get_server_time()
        sym_info = self.client.get_symbol_info(symbol)
        onboard_date = int(sym_info.get("onboardDate", 1743071400000))

        latest_saved = None if force_from_start else self.storage.get_latest_kline_time(symbol, interval)
        start_time = (latest_saved + 1) if latest_saved is not None else onboard_date

        total_saved = 0
        step_ms = 3600 * 1000 if interval == "1h" else 4 * 3600 * 1000

        logger.info("Syncing %s %s klines from %d to server time %d...", symbol, interval, start_time, server_time)

        cur_start = start_time
        while cur_start < server_time:
            batch = self.client.fetch_klines(
                symbol=symbol,
                interval=interval,
                start_time=cur_start,
                end_time=server_time,
                limit=BATCH_SIZE,
            )
            if not batch:
                break

            # Filter out incomplete first listing candle if open_time % step_ms != 0
            filtered_batch = []
            for k in batch:
                op_time = int(k[0])
                if op_time % step_ms == 0:
                    filtered_batch.append(k)
                else:
                    logger.info("Dropping incomplete first listing candle at %d", op_time)

            saved = self.storage.save_klines(
                filtered_batch,
                source="binance",
                symbol=symbol,
                interval=interval,
                server_time=server_time,
            )
            total_saved += saved

            last_in_batch = int(batch[-1][0])
            if last_in_batch <= cur_start or len(batch) < 2:
                break
            cur_start = last_in_batch + step_ms
            time.sleep(0.1)

        return total_saved

    def sync_mark_klines(
        self,
        symbol: str = SYMBOL,
        interval: str = "1h",
        force_from_start: bool = False,
    ) -> int:
        """Fetches all missing mark price klines and writes to SQLite."""
        server_time = self.client.get_server_time()
        sym_info = self.client.get_symbol_info(symbol)
        onboard_date = int(sym_info.get("onboardDate", 1743071400000))

        start_time = onboard_date
        total_saved = 0
        step_ms = 3600 * 1000 if interval == "1h" else 4 * 3600 * 1000

        cur_start = start_time
        while cur_start < server_time:
            batch = self.client.fetch_mark_price_klines(
                symbol=symbol,
                interval=interval,
                start_time=cur_start,
                end_time=server_time,
                limit=BATCH_SIZE,
            )
            if not batch:
                break

            filtered_batch = [k for k in batch if int(k[0]) % step_ms == 0]
            saved = self.storage.save_mark_klines(
                filtered_batch,
                source="binance",
                symbol=symbol,
                interval=interval,
                server_time=server_time,
            )
            total_saved += saved

            last_in_batch = int(batch[-1][0])
            if last_in_batch <= cur_start or len(batch) < 2:
                break
            cur_start = last_in_batch + step_ms
            time.sleep(0.1)

        return total_saved

    def sync_funding_rates(self, symbol: str = SYMBOL) -> int:
        """Fetches all funding rate history."""
        server_time = self.client.get_server_time()
        sym_info = self.client.get_symbol_info(symbol)
        onboard_date = int(sym_info.get("onboardDate", 1743071400000))

        total_saved = 0
        cur_start = onboard_date
        while cur_start < server_time:
            batch = self.client.fetch_funding_rates(
                symbol=symbol,
                start_time=cur_start,
                end_time=server_time,
                limit=1000,
            )
            if not batch:
                break

            saved = self.storage.save_funding_rates(batch, symbol=symbol)
            total_saved += saved

            last_f_time = int(batch[-1]["fundingTime"])
            if last_f_time <= cur_start or len(batch) < 2:
                break
            cur_start = last_f_time + 1
            time.sleep(0.1)

        return total_saved

    def collect_all_and_create_snapshots(
        self,
        symbol: str = SYMBOL,
    ) -> dict[str, Any]:
        """Runs complete ingestion pipeline and builds immutable snapshots for 1h and 4h."""
        # 1. Sync 1h and 4h klines
        saved_1h = self.sync_klines(symbol=symbol, interval="1h")
        saved_4h = self.sync_klines(symbol=symbol, interval="4h")

        # 2. Sync mark klines
        saved_mark_1h = self.sync_mark_klines(symbol=symbol, interval="1h")
        saved_mark_4h = self.sync_mark_klines(symbol=symbol, interval="4h")

        # 3. Sync funding rates
        saved_funding = self.sync_funding_rates(symbol=symbol)

        # 4. Load datasets from SQLite
        df_1h = self.storage.load_klines_df(symbol=symbol, interval="1h")
        df_4h = self.storage.load_klines_df(symbol=symbol, interval="4h")
        df_mark_1h = self.storage.load_mark_klines_df(symbol=symbol, interval="1h")
        df_mark_4h = self.storage.load_mark_klines_df(symbol=symbol, interval="4h")
        df_funding = self.storage.load_funding_rates_df(symbol=symbol)

        # 5. Run Quality Audit
        quality_report = generate_data_quality_report(df_1h, df_4h)

        # 6. Build Feature Sets and Snapshots for 1h
        feat_a_1h, _, ts_1h = build_features(df_1h, feature_set="A")
        feat_b_1h, _, _ = build_features(df_1h, feature_set="B")
        feat_c_1h = None
        if not df_mark_1h.empty and not df_funding.empty:
            feat_c_1h, _, _ = build_features(
                df_klines=df_1h,
                df_mark=df_mark_1h,
                df_funding=df_funding,
                feature_set="C",
            )

        snapshot_1h = DatasetSnapshot.create(
            timeframe="1h",
            timestamps=ts_1h,
            features_a=feat_a_1h,
            features_b=feat_b_1h,
            features_c=feat_c_1h,
            symbol=symbol,
        )
        snap_path_1h = snapshot_1h.save()

        # 7. Build Feature Sets and Snapshots for 4h
        feat_a_4h, _, ts_4h = build_features(df_4h, feature_set="A")
        feat_b_4h, _, _ = build_features(df_4h, feature_set="B")
        feat_c_4h = None
        if not df_mark_4h.empty and not df_funding.empty:
            feat_c_4h, _, _ = build_features(
                df_klines=df_4h,
                df_mark=df_mark_4h,
                df_funding=df_funding,
                feature_set="C",
            )

        snapshot_4h = DatasetSnapshot.create(
            timeframe="4h",
            timestamps=ts_4h,
            features_a=feat_a_4h,
            features_b=feat_b_4h,
            features_c=feat_c_4h,
            symbol=symbol,
        )
        snap_path_4h = snapshot_4h.save()

        return {
            "ingestion": {
                "klines_1h_saved": saved_1h,
                "klines_4h_saved": saved_4h,
                "mark_1h_saved": saved_mark_1h,
                "mark_4h_saved": saved_mark_4h,
                "funding_saved": saved_funding,
            },
            "quality_report": quality_report,
            "snapshots": {
                "1h": {
                    "snapshot_id": snapshot_1h.metadata.snapshot_id,
                    "candles": snapshot_1h.total_candles,
                    "sha256": snapshot_1h.metadata.sha256,
                    "path": str(snap_path_1h),
                },
                "4h": {
                    "snapshot_id": snapshot_4h.metadata.snapshot_id,
                    "candles": snapshot_4h.total_candles,
                    "sha256": snapshot_4h.metadata.sha256,
                    "path": str(snap_path_4h),
                },
            },
        }
