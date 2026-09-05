"""CLI script to sync Binance Futures PAXGUSDT data and create initial snapshots."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Safe encoding for Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from paxg_lab.data.collector import MarketDataCollector


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("=" * 75)
    print("  BINANCE FUTURES DATA INGESTION & SNAPSHOT PIPELINE (PAXGUSDT)")
    print("=" * 75)

    collector = MarketDataCollector()
    results = collector.collect_all_and_create_snapshots()

    print("\n[Ingestion Summary]")
    ing = results["ingestion"]
    print(f"  1h klines saved:    {ing['klines_1h_saved']:,}")
    print(f"  4h klines saved:    {ing['klines_4h_saved']:,}")
    print(f"  1h mark klines:     {ing['mark_1h_saved']:,}")
    print(f"  4h mark klines:     {ing['mark_4h_saved']:,}")
    print(f"  Funding records:    {ing['funding_saved']:,}")

    print("\n[Data Quality Audit]")
    qr = results["quality_report"]
    print(f"  Overall Status:     {qr['status']}")
    print(f"  1h Candles:         {qr['1h']['candle_count']} (OHLC Valid: {qr['1h']['ohlc_valid']}, Gaps: {qr['1h']['gaps_count']})")
    print(f"  4h Candles:         {qr['4h']['candle_count']} (OHLC Valid: {qr['4h']['ohlc_valid']}, Gaps: {qr['4h']['gaps_count']})")

    cv = qr["cross_validation_4h_vs_1h"]
    print(f"  4h vs 1h Cross-Check: Matched {cv['fully_covered_by_1h']}/{cv['total_4h_candles']} "
          f"(Match Ratio: {cv['match_ratio'] * 100:.1f}%, Discrepancies: {cv['discrepancies']})")

    print("\n[Created Snapshots]")
    snaps = results["snapshots"]
    print(f"  1h Snapshot ID:     {snaps['1h']['snapshot_id']}")
    print(f"     Candles:         {snaps['1h']['candles']}")
    print(f"     SHA-256:         {snaps['1h']['sha256']}")
    print(f"     Directory:       {snaps['1h']['path']}")
    print(f"  4h Snapshot ID:     {snaps['4h']['snapshot_id']}")
    print(f"     Candles:         {snaps['4h']['candles']}")
    print(f"     SHA-256:         {snaps['4h']['sha256']}")
    print(f"     Directory:       {snaps['4h']['path']}")

    # Save summary report for documentation
    out_evidence = Path("docs/paxg-lab/phases/p1_ingestion_evidence.json")
    with open(out_evidence, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved audit report to {out_evidence}")
    print("=" * 75)


if __name__ == "__main__":
    main()
