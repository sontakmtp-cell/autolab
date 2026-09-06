"""Main Streamlit entry point for PAXG Forecast Lab.

Provides a unified Vietnamese research dashboard with 5 tabs:
1. Dự đoán (Forecast)
2. Backtest
3. Huấn luyện LoRA (LoRA Training)
4. Tự động tối ưu (Autonomous Tuning)
5. Quản lý LoRA (LoRA Management)
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import streamlit as st

from paxg_lab.queue.storage import GPUJobStorage
from paxg_lab.ui.components.header import render_header
from paxg_lab.ui.components.tab_adapter_manager import render_adapter_manager_tab
from paxg_lab.ui.components.tab_auto_tune import render_auto_tune_tab
from paxg_lab.ui.components.tab_backtest import render_backtest_tab
from paxg_lab.ui.components.tab_forecast import render_forecast_tab
from paxg_lab.ui.components.tab_training import render_training_tab
from paxg_lab.ui.state import DEFAULT_DB_PATH

# 1. Streamlit Page Configuration
st.set_page_config(
    page_title="PAXG Forecast Lab",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# 2. Custom Dark Theme Styling
st.markdown(
    """
    <style>
        /* Overall Page Dark Background & Styling */
        .stApp {
            background-color: #0e1117;
            color: #e6edf3;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }

        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 12px;
            border-bottom: 2px solid #21262d;
            padding-bottom: 4px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 48px;
            white-space: pre-wrap;
            border-radius: 6px 6px 0px 0px;
            gap: 1px;
            padding: 10px 20px;
            color: #8b949e;
            font-size: 1.02rem;
            font-weight: 600;
        }
        .stTabs [aria-selected="true"] {
            background-color: #161b22 !important;
            color: #f1c40f !important;
            border-bottom: 3px solid #f1c40f !important;
        }

        /* Metric Cards */
        [data-testid="stMetricValue"] {
            font-size: 1.35rem !important;
            font-weight: 700 !important;
            color: #f1c40f !important;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.82rem !important;
            color: #8b949e !important;
        }

        /* Buttons and Forms */
        .stButton button {
            border-radius: 6px;
            font-weight: 600;
        }

        /* Footer */
        .footer-text {
            text-align: center;
            color: #484f58;
            font-size: 0.8rem;
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid #21262d;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def main() -> None:
    """Main application loop."""
    db_path = DEFAULT_DB_PATH
    if not db_path.exists():
        storage = GPUJobStorage(db_path)
        storage.init_db()

    # Render Header (Timeframe selector, Market Telemetry, Horizon Banner, Queue status)
    current_timeframe = render_header(db_path=db_path)

    # Render 5 Tabs
    tab_fc, tab_bt, tab_tr, tab_at, tab_am = st.tabs([
        "🔮 Dự đoán Giá",
        "📊 Đánh giá Backtest",
        "🛠️ Huấn luyện LoRA",
        "🤖 Tự động Tối ưu",
        "🗄️ Quản lý LoRA",
    ])

    with tab_fc:
        render_forecast_tab(timeframe=current_timeframe, db_path=db_path)

    with tab_bt:
        render_backtest_tab(timeframe=current_timeframe, db_path=db_path)

    with tab_tr:
        render_training_tab(timeframe=current_timeframe, db_path=db_path)

    with tab_at:
        render_auto_tune_tab(timeframe=current_timeframe, db_path=db_path)

    with tab_am:
        render_adapter_manager_tab(current_timeframe=current_timeframe, db_path=db_path)

    # Footer
    st.markdown(
        """
        <div class="footer-text">
            PAXG Forecast Lab &bull; TimesFM 3.0 PyTorch &bull; Binance Futures USDⓈ-M PAXGUSDT &bull; Bind Cục bộ: 127.0.0.1:8501
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
