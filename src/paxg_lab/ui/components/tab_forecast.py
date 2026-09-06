"""Tab 1: Forecast component with interactive candlestick charts, 80% nominal uncertainty bands, and future steps table."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any
import uuid

import pandas as pd
import streamlit as st

from ...constants import get_horizon_for_timeframe
from ...model.store import AdapterStore
from ...queue.storage import GPUJobStorage
from ...queue.types import JobPriority, JobSpec, JobStatus, JobType
from ..charts import build_candlestick_forecast_chart
from ..state import (
    DEFAULT_DB_PATH,
    VIETNAM_TZ,
    get_candle_count,
    get_latest_candle,
    get_latest_snapshot_path,
    get_recent_candles,
    list_recent_jobs,
    timestamp_to_vietnam_str,
)


def render_forecast_tab(timeframe: str, db_path: Path = DEFAULT_DB_PATH) -> None:
    """Renders the Forecast tab."""
    st.markdown("### 🔮 Dự đoán Giá Tương lai (TimesFM 3.0)")
    st.caption("Dự đoán xu hướng giá đóng cửa, đường trung vị và dải bất định danh nghĩa 80% [q10-q90] nhìn trước 24 giờ.")

    horizon = get_horizon_for_timeframe(timeframe)
    adapter_store = AdapterStore(db_path=db_path)
    all_adapters = adapter_store.list_adapters()
    # Filter adapters compatible with active timeframe
    valid_adapters = [a for a in all_adapters if a.timeframe == timeframe]
    recommended_id = adapter_store.get_recommended(timeframe)

    model_options = ["TimesFM 3.0 Base (Chuẩn không adapter)"]
    adapter_id_map: dict[str, str | None] = {"TimesFM 3.0 Base (Chuẩn không adapter)": None}
    adapter_manifest_map: dict[str, Any] = {"TimesFM 3.0 Base (Chuẩn không adapter)": None}

    for a in valid_adapters:
        alias = adapter_store.get_alias(a.adapter_id)
        star = " ⭐ [Khuyến nghị]" if a.adapter_id == recommended_id else ""
        label = f"LoRA: {alias}{star} (val_loss: {a.best_val_loss:.4f})"
        model_options.append(label)
        adapter_id_map[label] = a.adapter_id
        adapter_manifest_map[label] = a

    # Form to select model and generate forecast without duplicate triggers
    with st.form(key=f"form_forecast_{timeframe}"):
        col_m, col_c, col_btn = st.columns([2.5, 1.5, 1.2])

        with col_m:
            selected_model_label = st.selectbox(
                "Lựa chọn mô hình dự đoán:",
                options=model_options,
                index=0,
                help="Chọn TimesFM 3.0 Base hoặc adapter LoRA đã huấn luyện tương thích với khung giờ này.",
            )
            selected_adapter_id = adapter_id_map[selected_model_label]
            selected_manifest = adapter_manifest_map[selected_model_label]

        with col_c:
            candles_display_count = st.slider(
                "Số nến lịch sử trên biểu đồ:",
                min_value=24,
                max_value=200,
                value=72,
                step=12,
                help="Số lượng nến lịch sử hiển thị trên biểu đồ nến để người dùng dễ quan sát bối cảnh.",
            )

        with col_btn:
            st.markdown("<div style='height: 1.7rem;'></div>", unsafe_allow_html=True)
            submit_forecast = st.form_submit_button("🚀 Tạo Dự đoán", use_container_width=True)

    # Handle forecast execution via GPU queue
    storage = GPUJobStorage(db_path)
    snapshot_path = get_latest_snapshot_path(timeframe)

    if submit_forecast:
        if not snapshot_path or not snapshot_path.exists():
            st.error(f"Lỗi: Không tìm thấy snapshot dữ liệu cho khung {timeframe}. Vui lòng kiểm tra kho dữ liệu.")
            return

        latest_candle = get_latest_candle(timeframe, db_path=db_path)
        last_candle_time = latest_candle["open_time"] if latest_candle else int(time.time() * 1000)

        # Build job specification
        job_id = f"fc_{timeframe}_{uuid.uuid4().hex[:8]}"
        idempotency_key = f"fc_{timeframe}_{selected_adapter_id or 'base'}_{last_candle_time}"

        if selected_manifest is not None:
            adapter_dir = str(adapter_store.get_adapter_path(selected_adapter_id))
            context_len = int(selected_manifest.context_len)
            feature_set = str(selected_manifest.feature_set)
            columns = list(selected_manifest.feature_columns)
        else:
            adapter_dir = None
            context_len = 256
            feature_set = "A"
            columns = ["close"]

        payload = {
            "timeframe": timeframe,
            "horizon": horizon,
            "context_len": context_len,
            "snapshot_path": str(snapshot_path),
            "adapter_path": adapter_dir,
            "feature_set": feature_set,
            "columns": columns,
        }

        job_spec = JobSpec(
            job_id=job_id,
            job_type=JobType.FORECAST.value,
            timeframe=timeframe,
            priority=JobPriority.FORECAST.value,
            payload=payload,
            idempotency_key=idempotency_key,
        )

        submitted_id = storage.submit_job(job_spec)
        st.session_state["active_forecast_job_id"] = submitted_id

    # Polling & displaying forecast result
    active_job_id = st.session_state.get("active_forecast_job_id")
    if active_job_id:
        active_job = storage.get_job(active_job_id)
        if active_job:
            if active_job.status in (JobStatus.QUEUED.value, JobStatus.RUNNING.value):
                with st.spinner(f"Đang thực hiện dự đoán qua hàng đợi GPU ({active_job.status}). Vui lòng đợi..."):
                    # Quick polling loop
                    poll_start = time.time()
                    while time.time() - poll_start < 45.0:
                        time.sleep(0.8)
                        active_job = storage.get_job(active_job_id)
                        if not active_job or active_job.status not in (JobStatus.QUEUED.value, JobStatus.RUNNING.value):
                            break
                    st.rerun()

            elif active_job.status == JobStatus.SUCCEEDED.value and active_job.result:
                st.session_state["last_forecast_data"] = active_job.result
                st.session_state["last_forecast_model"] = (
                    f"LoRA: {adapter_store.get_alias(selected_adapter_id)}" if selected_adapter_id else "TimesFM 3.0 Base"
                )
                st.session_state.pop("active_forecast_job_id", None)
                st.success(f"Dự đoán thành công! Mã công việc: `{active_job_id}`")

            elif active_job.status == JobStatus.FAILED.value:
                st.error(f"Dự đoán thất bại: {active_job.error_message}")
                st.session_state.pop("active_forecast_job_id", None)

            elif active_job.status in (JobStatus.CANCELLED.value, JobStatus.INTERRUPTED.value):
                st.warning(f"Công việc dự đoán bị hủy hoặc gián đoạn ({active_job.status}).")
                st.session_state.pop("active_forecast_job_id", None)

    # Render Visualizations and Tables if forecast data is available
    forecast_data = st.session_state.get("last_forecast_data")
    if not forecast_data:
        # Check if there was any previously succeeded forecast in DB
        prev_jobs = storage.list_jobs(job_type=JobType.FORECAST.value, timeframe=timeframe, limit=1)
        if prev_jobs and prev_jobs[0].status == JobStatus.SUCCEEDED.value and prev_jobs[0].result:
            forecast_data = prev_jobs[0].result
            st.session_state["last_forecast_data"] = forecast_data

    candles_df = get_recent_candles(timeframe, limit=candles_display_count, db_path=db_path)

    if forecast_data:
        steps_list = _parse_forecast_result(forecast_data, timeframe)
        last_close = float(candles_df.iloc[-1]["close"]) if not candles_df.empty else 0.0

        # Summary KPIs
        if steps_list:
            final_step = steps_list[-1]
            final_median = final_step["q50"]
            delta_val = final_median - last_close
            delta_pct = (delta_val / last_close) * 100.0 if last_close > 0 else 0.0
            range_width = final_step["q90"] - final_step["q10"]

            col_k1, col_k2, col_k3, col_k4 = st.columns(4)
            with col_k1:
                st.metric("Giá dự đoán sau 24h", f"${final_median:,.2f}", f"{delta_val:+,.2f} USDT")
            with col_k2:
                st.metric("Tỷ lệ thay đổi kỳ vọng", f"{delta_pct:+.2f}%")
            with col_k3:
                st.metric("Dải bất định 80% (q90-q10)", f"${range_width:,.2f}")
            with col_k4:
                st.metric("Số bước dự đoán", f"{len(steps_list)} bước ({timeframe})")

        # Plotly Candlestick with forecast overlay
        fig = build_candlestick_forecast_chart(
            candles_df=candles_df,
            forecast_steps=steps_list,
            timeframe=timeframe,
            title=f"Dự đoán PAXG/USDT — Khung {timeframe} ({horizon} nến nhìn trước 24 giờ)",
        )
        st.plotly_chart(fig, use_container_width=True)

        # 24 steps (1h) or 6 steps (4h) table
        st.markdown(f"#### 📋 Bảng số chi tiết {len(steps_list)} bước dự đoán tương lai")
        table_rows = []
        for s in steps_list:
            step_idx = s["step"]
            time_str = s["time_vn"]
            q50 = s["q50"]
            q10 = s["q10"]
            q90 = s["q90"]
            d_usd = q50 - last_close
            d_pct = (d_usd / last_close) * 100.0 if last_close > 0 else 0.0
            table_rows.append({
                "Bước": f"+{step_idx} ({step_idx * (4 if timeframe == '4h' else 1)}h)",
                "Thời gian nến mở (UTC+7)": time_str,
                "Giá trung vị q50 (USDT)": f"${q50:,.2f}",
                "Thấp 80% (q10)": f"${q10:,.2f}",
                "Cao 80% (q90)": f"${q90:,.2f}",
                "Độ rộng dải ($)": f"${q90 - q10:,.2f}",
                "Thay đổi vs Giá hiện tại": f"{d_usd:+,.2f} ({d_pct:+.2f}%)",
            })
        
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    else:
        # Render clean candlestick chart without forecast if not yet generated
        fig = build_candlestick_forecast_chart(
            candles_df=candles_df,
            forecast_steps=None,
            timeframe=timeframe,
            title=f"Biểu đồ nến lịch sử PAXG/USDT — Khung {timeframe}",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.info("Nhấn nút **Tạo Dự đoán** ở trên để bắt đầu dự đoán 24 nến (1h) hoặc 6 nến (4h) qua mô hình TimesFM 3.0.")

    # Recent Forecast History Section
    with st.expander("📜 Lịch sử các lần dự đoán trước đó", expanded=False):
        recent_fc_jobs = list_recent_jobs(job_type=JobType.FORECAST.value, timeframe=timeframe, limit=10, db_path=db_path)
        if recent_fc_jobs:
            history_data = []
            for j in recent_fc_jobs:
                history_data.append({
                    "Mã Job": j.job_id,
                    "Thời gian tạo": timestamp_to_vietnam_str(j.created_at * 1000),
                    "Mô hình": "LoRA" if j.payload.get("adapter_path") else "Base",
                    "Trạng thái": j.status,
                    "Thời lượng": f"{(j.finished_at - j.started_at):.1f}s" if (j.finished_at and j.started_at) else "N/A",
                })
            st.dataframe(pd.DataFrame(history_data), use_container_width=True, hide_index=True)
        else:
            st.caption("Chưa có lịch sử dự đoán nào được lưu trong cơ sở dữ liệu.")


def _parse_forecast_result(result_dict: dict[str, Any], timeframe: str) -> list[dict[str, Any]]:
    """Normalizes ForecastResult dictionary into a step list for charts and tables."""
    steps_list = []

    # Read standard domain ForecastResult schema keys first, with backward-compatible fallbacks
    target_timestamps = result_dict.get("target_timestamps") or result_dict.get("timestamps", [])
    point_forecast = result_dict.get("point_forecast") or result_dict.get("q50", [])
    uncertainty_lower = result_dict.get("uncertainty_lower") or result_dict.get("q10", [])
    uncertainty_upper = result_dict.get("uncertainty_upper") or result_dict.get("q90", [])
    quantiles_matrix = result_dict.get("quantiles", [])  # shape (horizon, 9)
    forecast_origin_time = result_dict.get("forecast_origin_time")

    horizon = get_horizon_for_timeframe(timeframe)
    interval_ms = 4 * 3600 * 1000 if timeframe == "4h" else 1 * 3600 * 1000

    num_steps = (
        len(point_forecast)
        if point_forecast
        else (
            len(target_timestamps)
            if target_timestamps
            else (len(quantiles_matrix) if quantiles_matrix else horizon)
        )
    )

    for i in range(num_steps):
        step_num = i + 1
        if target_timestamps and i < len(target_timestamps):
            ts_ms = int(target_timestamps[i])
        elif forecast_origin_time is not None:
            ts_ms = int(forecast_origin_time) + step_num * interval_ms
        else:
            ts_ms = None

        time_vn = timestamp_to_vietnam_str(ts_ms) if ts_ms is not None else f"Bước +{step_num}"

        if point_forecast and i < len(point_forecast):
            q50 = float(point_forecast[i])
            q10 = float(uncertainty_lower[i]) if (uncertainty_lower and i < len(uncertainty_lower)) else (
                float(quantiles_matrix[i][0]) if (quantiles_matrix and i < len(quantiles_matrix)) else q50 * 0.99
            )
            q90 = float(uncertainty_upper[i]) if (uncertainty_upper and i < len(uncertainty_upper)) else (
                float(quantiles_matrix[i][8]) if (quantiles_matrix and i < len(quantiles_matrix)) else q50 * 1.01
            )
        elif quantiles_matrix and i < len(quantiles_matrix):
            row = quantiles_matrix[i]
            q10 = float(row[0])
            q50 = float(row[4])
            q90 = float(row[8])
        else:
            q50, q10, q90 = 0.0, 0.0, 0.0

        steps_list.append({
            "step": step_num,
            "timestamp_ms": ts_ms,
            "time_vn": time_vn,
            "q50": q50,
            "q10": q10,
            "q90": q90,
        })

    return steps_list
