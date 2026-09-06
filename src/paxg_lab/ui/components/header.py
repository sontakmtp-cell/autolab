"""Header component displaying timeframe selection, market data status, horizon convention, and GPU queue telemetry."""

from __future__ import annotations

from pathlib import Path
import streamlit as st

from ...constants import TIMEFRAME_1H, TIMEFRAME_4H, get_horizon_for_timeframe
from ..state import (
    get_candle_count,
    get_gpu_queue_summary,
    get_latest_candle,
    timestamp_to_vietnam_str,
)


def render_header(db_path: Path) -> str:
    """Renders the top application header and returns the selected timeframe ('1h' or '4h')."""
    st.markdown(
        """
        <div style="padding: 0.5rem 0 1rem 0;">
            <h1 style="margin: 0; font-size: 1.9rem; font-weight: 700; color: #f1c40f;">
                🏆 PAXG Forecast Lab — Nghiên cứu Dự đoán Giá PAXG/USDT
            </h1>
            <p style="margin: 0.2rem 0 0 0; color: #8b949e; font-size: 0.95rem;">
                Mô hình nền tảng <b>google/timesfm-3.0-pytorch</b> &bull; Binance USDⓈ-M Futures &bull; Giao diện Nghiên cứu Độc lập
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Timeframe selection and telemetry bar
    col_tf, col_m1, col_m2, col_m3, col_m4 = st.columns([1.5, 1.2, 1.3, 1.3, 1.3])

    with col_tf:
        current_tf = st.session_state.get("timeframe", TIMEFRAME_1H)
        tf_index = 0 if current_tf == TIMEFRAME_1H else 1
        selected_label = st.radio(
            "Khung thời gian:",
            options=["1h (24 nến / 24h)", "4h (6 nến / 24h)"],
            index=tf_index,
            horizontal=True,
            key="timeframe_radio",
        )
        chosen_tf = TIMEFRAME_1H if "1h" in selected_label else TIMEFRAME_4H
        st.session_state["timeframe"] = chosen_tf

    latest_candle = get_latest_candle(chosen_tf, db_path=db_path)
    candle_count = get_candle_count(chosen_tf, db_path=db_path)
    queue_summary = get_gpu_queue_summary(db_path=db_path)

    with col_m1:
        if latest_candle:
            close_price = float(latest_candle["close"])
            st.metric(label="Giá gần nhất (USDT)", value=f"${close_price:,.2f}")
        else:
            st.metric(label="Giá gần nhất", value="Chưa có dữ liệu")

    with col_m2:
        if latest_candle:
            last_time_str = timestamp_to_vietnam_str(latest_candle["open_time"])
            st.metric(label="Nến đóng gần nhất", value=last_time_str)
        else:
            st.metric(label="Nến đóng gần nhất", value="N/A")

    with col_m3:
        st.metric(label=f"Tổng số nến ({chosen_tf})", value=f"{candle_count:,} nến")

    with col_m4:
        running_job = queue_summary.get("running_job")
        queued_count = queue_summary.get("queued_count", 0)
        auto_state = queue_summary.get(f"auto_state_{chosen_tf}", "STOPPED")

        if running_job:
            status_text = f"Đang chạy ({running_job.job_type})"
            badge_color = "#e67e22"
        elif queued_count > 0:
            status_text = f"Chờ: {queued_count} việc"
            badge_color = "#3498db"
        else:
            status_text = "Rảnh (Idle)"
            badge_color = "#2ecc71"

        st.markdown(
            f"""
            <div style="padding-top: 0.2rem;">
                <div style="font-size: 0.8rem; color: #8b949e;">Tiến trình GPU & Tự động:</div>
                <div style="font-weight: 600; color: {badge_color}; font-size: 1rem;">
                    ● {status_text}
                </div>
                <div style="font-size: 0.75rem; color: #8b949e;">Auto: <b>{auto_state}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Mandatory Horizon Banner
    horizon = get_horizon_for_timeframe(chosen_tf)
    st.markdown(
        f"""
        <div style="background-color: #161b22; border-left: 4px solid #f1c40f; padding: 0.4rem 0.8rem; border-radius: 4px; margin: 0.5rem 0 1rem 0; font-size: 0.88rem; color: #c9d1d9;">
            📌 <b>Quy ước Horizon bắt buộc (PLAN 2.3):</b> Khung <b>{chosen_tf}</b> &rarr; horizon = <b>{horizon} nến</b>, nhìn trước đúng <b>24 giờ</b>. Mọi dự đoán, huấn luyện và backtest đều tuân thủ chặt chẽ ranh giới này.
        </div>
        """,
        unsafe_allow_html=True,
    )

    return chosen_tf
