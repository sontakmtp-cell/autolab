"""SQLite storage manager for market data in PAXG Forecast Lab."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_DB_PATH = Path("var/paxg_lab/paxg_lab.db")


class MarketDataStorage:
    """Manages SQLite database storage for klines, mark price, and funding rates."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        """Returns SQLite connection with row factory and WAL mode."""
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def init_db(self) -> None:
        """Creates tables and indexes if they do not already exist."""
        with self.get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS klines (
                    source TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    open_time INTEGER NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    close_time INTEGER NOT NULL,
                    quote_volume REAL NOT NULL,
                    trades INTEGER NOT NULL,
                    taker_buy_volume REAL NOT NULL,
                    taker_buy_quote_volume REAL NOT NULL,
                    is_closed INTEGER NOT NULL DEFAULT 1,
                    PRIMARY KEY (source, symbol, interval, open_time)
                );

                CREATE INDEX IF NOT EXISTS idx_klines_lookup
                ON klines (symbol, interval, open_time);

                CREATE TABLE IF NOT EXISTS mark_klines (
                    source TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    open_time INTEGER NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    close_time INTEGER NOT NULL,
                    PRIMARY KEY (source, symbol, interval, open_time)
                );

                CREATE INDEX IF NOT EXISTS idx_mark_lookup
                ON mark_klines (symbol, interval, open_time);

                CREATE TABLE IF NOT EXISTS funding_rates (
                    symbol TEXT NOT NULL,
                    funding_time INTEGER NOT NULL,
                    funding_rate REAL NOT NULL,
                    mark_price REAL NOT NULL,
                    PRIMARY KEY (symbol, funding_time)
                );

                CREATE INDEX IF NOT EXISTS idx_funding_lookup
                ON funding_rates (symbol, funding_time);

                CREATE TABLE IF NOT EXISTS data_quality_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    total_candles INTEGER NOT NULL,
                    gaps_count INTEGER NOT NULL,
                    anomalies_count INTEGER NOT NULL,
                    report_json TEXT NOT NULL
                );
            """)

    def save_klines(
        self,
        raw_klines: list[list[Any]],
        source: str = "binance",
        symbol: str = "PAXGUSDT",
        interval: str = "1h",
        server_time: int | None = None,
    ) -> int:
        """Saves raw klines into SQLite, skipping unclosed candle if server_time is supplied.

        Returns number of inserted/updated rows.
        """
        if not raw_klines:
            return 0

        rows = []
        for k in raw_klines:
            open_time = int(k[0])
            close_time = int(k[6])

            # Exclude currently active/unclosed candle
            if server_time is not None and close_time > server_time:
                continue

            open_p = float(k[1])
            high_p = float(k[2])
            low_p = float(k[3])
            close_p = float(k[4])
            vol = float(k[5])
            q_vol = float(k[7])
            trades = int(k[8])
            tb_vol = float(k[9])
            tb_q_vol = float(k[10])

            rows.append((
                source, symbol, interval, open_time,
                open_p, high_p, low_p, close_p, vol,
                close_time, q_vol, trades, tb_vol, tb_q_vol, 1
            ))

        if not rows:
            return 0

        with self.get_connection() as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO klines (
                    source, symbol, interval, open_time,
                    open, high, low, close, volume,
                    close_time, quote_volume, trades,
                    taker_buy_volume, taker_buy_quote_volume, is_closed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, rows)

        return len(rows)

    def save_mark_klines(
        self,
        raw_mark_klines: list[list[Any]],
        source: str = "binance",
        symbol: str = "PAXGUSDT",
        interval: str = "1h",
        server_time: int | None = None,
    ) -> int:
        """Saves raw mark price klines into SQLite."""
        if not raw_mark_klines:
            return 0

        rows = []
        for k in raw_mark_klines:
            open_time = int(k[0])
            close_time = int(k[6])

            if server_time is not None and close_time > server_time:
                continue

            open_p = float(k[1])
            high_p = float(k[2])
            low_p = float(k[3])
            close_p = float(k[4])

            rows.append((source, symbol, interval, open_time, open_p, high_p, low_p, close_p, close_time))

        if not rows:
            return 0

        with self.get_connection() as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO mark_klines (
                    source, symbol, interval, open_time,
                    open, high, low, close, close_time
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, rows)

        return len(rows)

    def save_funding_rates(
        self,
        raw_funding: list[dict[str, Any]],
        symbol: str = "PAXGUSDT",
    ) -> int:
        """Saves realized funding rates into SQLite."""
        if not raw_funding:
            return 0

        rows = []
        for f in raw_funding:
            sym = str(f.get("symbol", symbol))
            f_time = int(f["fundingTime"])
            f_rate = float(f["fundingRate"])
            mark_p = float(f.get("markPrice", 0.0))
            rows.append((sym, f_time, f_rate, mark_p))

        with self.get_connection() as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO funding_rates (
                    symbol, funding_time, funding_rate, mark_price
                ) VALUES (?, ?, ?, ?);
            """, rows)

        return len(rows)

    def get_latest_kline_time(
        self,
        symbol: str = "PAXGUSDT",
        interval: str = "1h",
        source: str = "binance",
    ) -> int | None:
        """Returns the maximum open_time recorded for this timeframe."""
        with self.get_connection() as conn:
            cur = conn.execute(
                "SELECT MAX(open_time) FROM klines WHERE symbol = ? AND interval = ? AND source = ?;",
                (symbol, interval, source)
            )
            row = cur.fetchone()
            return row[0] if row and row[0] is not None else None

    def get_earliest_kline_time(
        self,
        symbol: str = "PAXGUSDT",
        interval: str = "1h",
        source: str = "binance",
    ) -> int | None:
        """Returns the minimum open_time recorded for this timeframe."""
        with self.get_connection() as conn:
            cur = conn.execute(
                "SELECT MIN(open_time) FROM klines WHERE symbol = ? AND interval = ? AND source = ?;",
                (symbol, interval, source)
            )
            row = cur.fetchone()
            return row[0] if row and row[0] is not None else None

    def load_klines_df(
        self,
        symbol: str = "PAXGUSDT",
        interval: str = "1h",
        start_time: int | None = None,
        end_time: int | None = None,
        source: str = "binance",
    ) -> pd.DataFrame:
        """Loads klines as a clean pandas DataFrame sorted by open_time."""
        query = ["SELECT * FROM klines WHERE symbol = ? AND interval = ? AND source = ?"]
        params: list[Any] = [symbol, interval, source]

        if start_time is not None:
            query.append("AND open_time >= ?")
            params.append(start_time)
        if end_time is not None:
            query.append("AND open_time <= ?")
            params.append(end_time)

        query.append("ORDER BY open_time ASC;")
        sql = " ".join(query)

        with self.get_connection() as conn:
            df = pd.read_sql_query(sql, conn, params=params)

        return df

    def load_mark_klines_df(
        self,
        symbol: str = "PAXGUSDT",
        interval: str = "1h",
        start_time: int | None = None,
        end_time: int | None = None,
        source: str = "binance",
    ) -> pd.DataFrame:
        """Loads mark price klines as a DataFrame sorted by open_time."""
        query = ["SELECT * FROM mark_klines WHERE symbol = ? AND interval = ? AND source = ?"]
        params: list[Any] = [symbol, interval, source]

        if start_time is not None:
            query.append("AND open_time >= ?")
            params.append(start_time)
        if end_time is not None:
            query.append("AND open_time <= ?")
            params.append(end_time)

        query.append("ORDER BY open_time ASC;")
        sql = " ".join(query)

        with self.get_connection() as conn:
            df = pd.read_sql_query(sql, conn, params=params)

        return df

    def load_funding_rates_df(
        self,
        symbol: str = "PAXGUSDT",
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> pd.DataFrame:
        """Loads funding rates as a DataFrame sorted by funding_time."""
        query = ["SELECT * FROM funding_rates WHERE symbol = ?"]
        params: list[Any] = [symbol]

        if start_time is not None:
            query.append("AND funding_time >= ?")
            params.append(start_time)
        if end_time is not None:
            query.append("AND funding_time <= ?")
            params.append(end_time)

        query.append("ORDER BY funding_time ASC;")
        sql = " ".join(query)

        with self.get_connection() as conn:
            df = pd.read_sql_query(sql, conn, params=params)

        return df
