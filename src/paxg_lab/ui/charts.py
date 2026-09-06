"""Interactive Plotly charts for PAXG Forecast Lab: candlesticks, forecast ribbons, loss curves, and backtest evaluations."""

from __future__ import annotations

from typing import Any
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import plotly.graph_objects as go

VIETNAM_TZ = ZoneInfo("Asia/Ho_Chi_Minh")

# Dark theme palette
BG_COLOR = "#0e1117"
PANEL_COLOR = "#161b22"
GRID_COLOR = "#21262d"
TEXT_COLOR = "#e6edf3"
CANDLE_UP = "#26a69a"
CANDLE_DOWN = "#ef5350"
FORECAST_MEDIAN = "#00e5ff"
BAND_80_FILL = "rgba(0, 229, 255, 0.18)"
BAND_80_LINE = "rgba(0, 229, 255, 0.50)"
BASE_REF_COLOR = "#8b949e"


def build_candlestick_forecast_chart(
    candles_df: pd.DataFrame,
    forecast_steps: list[dict[str, Any]] | None = None,
    timeframe: str = "1h",
    title: str = "Biểu đồ Nến PAXG/USDT & Dự đoán Tương lai",
) -> go.Figure:
    """Builds interactive dark-theme candlestick chart with forecast median and 80% nominal uncertainty band."""
    fig = go.Figure()

    if candles_df.empty:
        fig.update_layout(
            title=dict(text="Chưa có dữ liệu nến", font=dict(color=TEXT_COLOR)),
            paper_bgcolor=BG_COLOR,
            plot_bgcolor=BG_COLOR,
        )
        return fig

    # 1. Historical Candlesticks
    fig.add_trace(
        go.Candlestick(
            x=candles_df["time_vn"],
            open=candles_df["open"],
            high=candles_df["high"],
            low=candles_df["low"],
            close=candles_df["close"],
            name="Nến lịch sử",
            increasing_line_color=CANDLE_UP,
            decreasing_line_color=CANDLE_DOWN,
            increasing_fillcolor=CANDLE_UP,
            decreasing_fillcolor=CANDLE_DOWN,
        )
    )

    # 2. Forecast Overlays (if available)
    if forecast_steps and len(forecast_steps) > 0:
        last_row = candles_df.iloc[-1]
        last_time = last_row["time_vn"]
        last_close = float(last_row["close"])

        # Construct time series for forecast points
        future_times = [last_time]
        q50_vals = [last_close]
        q10_vals = [last_close]
        q90_vals = [last_close]

        for s in forecast_steps:
            # Prefer numeric millisecond epoch timestamp converted to Vietnam timezone
            t_ms = s.get("timestamp_ms")
            if isinstance(t_ms, (int, float)) and t_ms > 0:
                t_dt = pd.to_datetime(t_ms, unit="ms", utc=True).tz_convert(VIETNAM_TZ)
            else:
                t_str = s.get("time_vn")
                try:
                    t_dt = pd.to_datetime(t_str)
                except Exception:
                    step_offset = len(future_times)
                    t_dt = last_time + pd.Timedelta(hours=step_offset * (4 if timeframe == "4h" else 1))

            future_times.append(t_dt)
            q50_vals.append(float(s["q50"]))
            q10_vals.append(float(s["q10"]))
            q90_vals.append(float(s["q90"]))

        # Upper bound (q90)
        fig.add_trace(
            go.Scatter(
                x=future_times,
                y=q90_vals,
                mode="lines",
                line=dict(width=1, dash="dot", color=BAND_80_LINE),
                name="Trần 80% (q90)",
                hoverinfo="skip",
            )
        )

        # Lower bound (q10) with translucent fill to upper bound
        fig.add_trace(
            go.Scatter(
                x=future_times,
                y=q10_vals,
                mode="lines",
                line=dict(width=1, dash="dot", color=BAND_80_LINE),
                fill="tonexty",
                fillcolor=BAND_80_FILL,
                name="Dải bất định 80% [q10-q90]",
                hoverinfo="skip",
            )
        )

        # Median trajectory line (q50)
        fig.add_trace(
            go.Scatter(
                x=future_times,
                y=q50_vals,
                mode="lines+markers",
                line=dict(width=2.5, color=FORECAST_MEDIAN),
                marker=dict(size=5, color=FORECAST_MEDIAN),
                name=f"Trung vị dự đoán ({len(forecast_steps)} nến / 24h)",
                hovertemplate="Thời gian: %{x}<br>Giá trung vị: $%{y:.2f}<extra></extra>",
            )
        )

        # Vertical line separating history from forecast
        fig.add_vline(
            x=last_time,
            line_width=1.5,
            line_dash="dash",
            line_color="#ffd700",
            annotation_text="Thời điểm bắt đầu dự đoán",
            annotation_position="top left",
            annotation_font=dict(color="#ffd700", size=11),
        )

    fig.update_layout(
        title=dict(text=title, font=dict(color=TEXT_COLOR, size=15)),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(color=TEXT_COLOR),
        margin=dict(l=45, r=30, t=50, b=30),
        xaxis=dict(
            gridcolor=GRID_COLOR,
            showgrid=True,
            rangeslider=dict(visible=False),
        ),
        yaxis=dict(
            gridcolor=GRID_COLOR,
            showgrid=True,
            title="Giá (USDT)",
            tickformat="$,.2f",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11),
        ),
        hovermode="x unified",
    )

    return fig


def build_loss_chart(
    train_losses: list[float],
    val_losses: list[float] | None = None,
    epochs: list[int] | None = None,
    title: str = "Đồ thị Loss Huấn luyện LoRA",
) -> go.Figure:
    """Builds interactive training loss curve."""
    fig = go.Figure()
    x_axis = epochs if epochs else list(range(1, len(train_losses) + 1))

    if train_losses:
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=train_losses,
                mode="lines+markers",
                name="Train Loss",
                line=dict(color="#f39c12", width=2),
                marker=dict(size=6),
            )
        )

    if val_losses:
        fig.add_trace(
            go.Scatter(
                x=x_axis[:len(val_losses)],
                y=val_losses,
                mode="lines+markers",
                name="Validation Loss (Dừng sớm)",
                line=dict(color="#00e5ff", width=2, dash="dash"),
                marker=dict(size=7, symbol="star"),
            )
        )

    fig.update_layout(
        title=dict(text=title, font=dict(color=TEXT_COLOR, size=14)),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(color=TEXT_COLOR),
        margin=dict(l=40, r=20, t=45, b=30),
        xaxis=dict(title="Vòng học (Epoch)", gridcolor=GRID_COLOR, showgrid=True),
        yaxis=dict(title="Loss", gridcolor=GRID_COLOR, showgrid=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def build_backtest_error_chart(
    step_errors: dict[int, float],
    base_step_errors: dict[int, float] | None = None,
    timeframe: str = "1h",
    title: str = "Sai số Tuyệt đối (MAE) theo Từng Bước Dự đoán",
) -> go.Figure:
    """Builds step-by-step MAE bar chart comparing Candidate vs Base."""
    fig = go.Figure()

    steps = sorted(step_errors.keys())
    step_labels = [f"Bước {s} (+{s * (4 if timeframe == '4h' else 1)}h)" for s in steps]
    cand_maes = [step_errors[s] for s in steps]

    fig.add_trace(
        go.Bar(
            x=step_labels,
            y=cand_maes,
            name="Mô hình Đánh giá",
            marker_color="#00e5ff",
        )
    )

    if base_step_errors:
        base_maes = [base_step_errors.get(s, 0.0) for s in steps]
        fig.add_trace(
            go.Bar(
                x=step_labels,
                y=base_maes,
                name="Base Chuẩn",
                marker_color=BASE_REF_COLOR,
            )
        )

    fig.update_layout(
        title=dict(text=title, font=dict(color=TEXT_COLOR, size=14)),
        barmode="group",
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(color=TEXT_COLOR),
        margin=dict(l=40, r=20, t=45, b=30),
        xaxis=dict(gridcolor=GRID_COLOR, showgrid=True),
        yaxis=dict(title="MAE (USDT)", gridcolor=GRID_COLOR, showgrid=True, tickformat="$,.2f"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig
