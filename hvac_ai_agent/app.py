from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from agents.analyzer import run_analysis
from agents.diagnostic import detect_anomalies_zscore
from agents.forecaster import forecast_next_24
from agents.optimizer import generate_recommendations
from report.report_generator import generate_html_report


THEME_PALETTES = {
    "Interior": {
        "primary": "#0b6e6e",
        "secondary": "#1f4f8f",
        "accent": "#c96b2c",
        "danger": "#b42318",
        "bg1": "#f6efe4",
        "bg2": "#e8f1ef",
        "card": "#fffaf2",
    },
    "Ocean": {
        "primary": "#0f766e",
        "secondary": "#1d4ed8",
        "accent": "#f97316",
        "danger": "#dc2626",
        "bg1": "#e6fffb",
        "bg2": "#eff6ff",
        "card": "#ffffff",
    },
    "Sunset": {
        "primary": "#c2410c",
        "secondary": "#7c3aed",
        "accent": "#ea580c",
        "danger": "#b91c1c",
        "bg1": "#fff7ed",
        "bg2": "#f5f3ff",
        "card": "#ffffff",
    },
    "Forest": {
        "primary": "#166534",
        "secondary": "#0369a1",
        "accent": "#ca8a04",
        "danger": "#b91c1c",
        "bg1": "#ecfdf5",
        "bg2": "#eff6ff",
        "card": "#ffffff",
    },
}


st.set_page_config(page_title="HVAC Multi-Agent AI Optimization", page_icon="📈", layout="wide")
st.title("HVAC Multi-Agent AI Optimization System")
st.caption("Analyzer + Forecaster + Diagnostic + Optimizer")


def apply_styles(palette: dict[str, str]) -> None:
    st.markdown(
        f"""
        <style>
            .stApp {{
                background:
                    radial-gradient(1200px 380px at 0% -10%, {palette['bg1']} 0%, transparent 58%),
                    radial-gradient(1200px 430px at 100% -15%, {palette['bg2']} 0%, transparent 55%),
                    {palette['bg1']};
            }}
            .block-container {{
                padding-top: 1.3rem;
            }}
            div[data-testid="stMetric"] {{
                border: 1px solid #e2e8f0;
                border-radius: 14px;
                padding: 8px 12px;
                background: {palette['card']};
                box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
            }}
            [data-testid="stSidebar"] {{
                border-right: 1px solid #e2e8f0;
                background: rgba(255, 250, 242, 0.92);
                backdrop-filter: blur(4px);
            }}
            div[data-baseweb="tab-list"] {{
                gap: 8px;
                margin-bottom: 8px;
            }}
            button[data-baseweb="tab"] {{
                border: 1px solid #d6ddd8 !important;
                border-radius: 999px !important;
                background: #fffaf2 !important;
                padding: 6px 14px !important;
                font-weight: 600 !important;
            }}
            button[data-baseweb="tab"][aria-selected="true"] {{
                background: {palette['primary']} !important;
                color: white !important;
                border-color: {palette['primary']} !important;
            }}
            .stDownloadButton > button, .stButton > button {{
                border-radius: 12px !important;
                border: 1px solid #d6ddd8 !important;
                background: #fffaf2 !important;
                font-weight: 600 !important;
            }}
            .stDownloadButton > button:hover, .stButton > button:hover {{
                border-color: {palette['primary']} !important;
                color: {palette['primary']} !important;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def apply_natural_time_axis(ax, max_ticks: int = 7, rotate: int = 0) -> None:
    """Format datetime axis with readable labels and no noisy offset text."""
    locator = mdates.AutoDateLocator(minticks=4, maxticks=max_ticks)
    formatter = mdates.ConciseDateFormatter(locator)
    formatter.formats = ["%Y", "%b %Y", "%d %b", "%H:%M", "%H:%M", "%S"]
    formatter.offset_formats = ["", "", "", "", "", ""]
    formatter.zero_formats = ["", "%Y", "%b %Y", "%d %b", "%d %b", "%d %b %H:%M"]
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    plt.setp(ax.get_xticklabels(), rotation=rotate, ha="right" if rotate else "center")


def load_input_data(uploaded_file) -> pd.DataFrame:
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)

    sample_path = Path(__file__).resolve().parent / "data" / "sample_dataset.csv"
    return pd.read_csv(sample_path)


with st.sidebar:
    st.header("Data Source")
    upload = st.file_uploader("Upload HVAC CSV", type=["csv"])
    theme = st.selectbox("Color Theme", options=list(THEME_PALETTES.keys()), index=0)
    lookback_hours = st.slider("Lookback Window (hours)", min_value=48, max_value=240, value=168, step=24)
    use_smoothing = st.toggle("Smooth Energy Trend", value=False)
    smooth_window = st.slider("Smoothing Window", min_value=2, max_value=12, value=4, step=1, disabled=not use_smoothing)
    z_threshold = st.slider("Anomaly Z-score Threshold", min_value=1.5, max_value=4.0, value=2.5, step=0.1)
    efficiency_threshold = st.slider("Efficiency Threshold (iKW_TR)", min_value=0.8, max_value=2.0, value=1.2, step=0.05)

palette = THEME_PALETTES[theme]
apply_styles(palette)

try:
    raw_df = load_input_data(upload)
    df, summary, load_kwh_corr = run_analysis(raw_df)
except Exception as exc:
    st.error(f"Failed to load/process data: {exc}")
    st.stop()

forecast_df = forecast_next_24(df, target_col="kWh", horizon=24)
anomalies_df, _ = detect_anomalies_zscore(df, target_col="kWh", threshold=float(z_threshold))
recommendations = generate_recommendations(
    summary=summary,
    forecast_df=forecast_df,
    anomalies_df=anomalies_df,
    full_df=df,
    efficiency_threshold=float(efficiency_threshold),
)

view_df = df.tail(int(lookback_hours)).copy()
if use_smoothing:
    view_df["kWh_plot"] = view_df["kWh"].rolling(window=int(smooth_window), min_periods=1).mean()
else:
    view_df["kWh_plot"] = view_df["kWh"]

forecast_avg = float(forecast_df["forecast_kWh"].mean())
forecast_delta_pct = ((forecast_avg - summary["avg_kwh"]) / summary["avg_kwh"] * 100) if summary["avg_kwh"] else 0.0
recent_eff = float(df["iKW_TR"].tail(24).mean())
eff_delta = recent_eff - float(efficiency_threshold)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total Energy (kWh)", f"{summary['total_kwh']:,.0f}")
kpi2.metric("Avg iKW_TR", f"{summary['avg_ikw_tr']:.3f}", delta=f"{eff_delta:+.3f} vs threshold")
kpi3.metric("Load-kWh Correlation", f"{load_kwh_corr:.2f}")
kpi4.metric("Forecast Next-24 Avg", f"{forecast_avg:.1f}", delta=f"{forecast_delta_pct:+.1f}% vs baseline")

overview_tab, diagnostics_tab, recs_tab = st.tabs(["📊 Overview", "🩺 Diagnostics", "💡 Recommendations"])

qa1, qa2, qa3 = st.columns(3)
with qa1:
    if st.button("🔄 Refresh View", use_container_width=True):
        st.rerun()
with qa2:
    if st.button("🎯 Focus Anomalies", use_container_width=True):
        st.info("Tip: lower Z-threshold in the sidebar to surface more anomalies.")
with qa3:
    if st.button("⚙️ Efficiency Hint", use_container_width=True):
        st.info("Tip: tune iKW_TR threshold near your baseline operating range.")

with overview_tab:
    st.subheader("Data Preview")
    st.dataframe(view_df.tail(30), use_container_width=True)
    st.download_button(
        "⬇️ Download Current View (.csv)",
        data=view_df.to_csv(index=False),
        file_name="hvac_view_slice.csv",
        mime="text/csv",
    )

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Energy Trend")
        fig, ax = plt.subplots(figsize=(8, 3.8))
        ax.plot(view_df["Timestamp"], view_df["kWh_plot"], label="kWh", color=palette["primary"], linewidth=1.8)
        ax.fill_between(view_df["Timestamp"], view_df["kWh_plot"], color=palette["primary"], alpha=0.08)
        anomalies_view = anomalies_df[anomalies_df["Timestamp"].isin(view_df["Timestamp"])]
        if not anomalies_view.empty:
            ax.scatter(
                anomalies_view["Timestamp"],
                anomalies_view["kWh"],
                color=palette["danger"],
                label="Anomalies",
                s=32,
                zorder=3,
            )
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("kWh")
        apply_natural_time_axis(ax, max_ticks=7, rotate=0)
        ax.legend()
        ax.grid(alpha=0.22)
        fig.tight_layout()
        st.pyplot(fig)

    with chart_col2:
        st.subheader("24-Step Forecast")
        fig2, ax2 = plt.subplots(figsize=(8, 3.8))
        history = df.tail(72)
        ax2.plot(history["Timestamp"], history["kWh"], label="Historical", color=palette["secondary"], linewidth=1.8)
        ax2.plot(forecast_df["Timestamp"], forecast_df["forecast_kWh"], label="Forecast", color=palette["accent"], linewidth=2)
        hist_std = float(np.std(history["kWh"].to_numpy(), ddof=0))
        low = forecast_df["forecast_kWh"] - 0.25 * hist_std
        high = forecast_df["forecast_kWh"] + 0.25 * hist_std
        ax2.fill_between(forecast_df["Timestamp"], low, high, color=palette["accent"], alpha=0.12, label="Forecast band")
        ax2.set_xlabel("Timestamp")
        ax2.set_ylabel("kWh")
        apply_natural_time_axis(ax2, max_ticks=6, rotate=0)
        ax2.legend()
        ax2.grid(alpha=0.22)
        fig2.tight_layout()
        st.pyplot(fig2)

with diagnostics_tab:
    d1, d2, d3 = st.columns(3)
    d1.metric("Anomaly Count", int(len(anomalies_df)))
    d2.metric("Recent 24h iKW_TR", f"{recent_eff:.3f}")
    d3.metric("Active Z Threshold", f"{z_threshold:.1f}")
    if anomalies_df.empty:
        st.success("No anomalies detected at the current Z-score threshold.")
    else:
        st.warning("Anomalies detected. Review the flagged intervals below.")
        st.dataframe(
            anomalies_df[["Timestamp", "kWh", "iKW_TR", "Load", "Temp", "z_score"]].sort_values("Timestamp"),
            use_container_width=True,
        )

with recs_tab:
    st.subheader("AI Recommendations")
    for rec in recommendations:
        if "high" in rec.lower() or "above threshold" in rec.lower() or "anomal" in rec.lower():
            st.warning(rec)
        else:
            st.success(rec)

report_html = generate_html_report(
    summary_metrics=summary,
    forecast_value=float(forecast_df["forecast_kWh"].mean()),
    anomaly_count=int(len(anomalies_df)),
    recommendations=recommendations,
)

st.download_button(
    label="📄 Download HTML Report",
    data=report_html,
    file_name="hvac_ai_optimization_report.html",
    mime="text/html",
)
