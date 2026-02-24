from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from agents.analyzer import run_analysis
from agents.diagnostic import detect_anomalies_zscore
from agents.forecaster import forecast_next_24
from agents.optimizer import generate_recommendations
from report.report_generator import generate_html_report


st.set_page_config(page_title="HVAC Multi-Agent AI Optimization", page_icon="📈", layout="wide")
st.title("HVAC Multi-Agent AI Optimization System")
st.caption("Analyzer + Forecaster + Diagnostic + Optimizer")


def load_input_data(uploaded_file) -> pd.DataFrame:
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)

    sample_path = Path(__file__).resolve().parent / "data" / "sample_dataset.csv"
    return pd.read_csv(sample_path)


with st.sidebar:
    st.header("Data Source")
    upload = st.file_uploader("Upload HVAC CSV", type=["csv"])
    z_threshold = st.slider("Anomaly Z-score Threshold", min_value=1.5, max_value=4.0, value=2.5, step=0.1)
    efficiency_threshold = st.slider("Efficiency Threshold (iKW_TR)", min_value=0.8, max_value=2.0, value=1.2, step=0.05)

try:
    raw_df = load_input_data(upload)
    df, summary, load_kwh_corr = run_analysis(raw_df)
except Exception as exc:
    st.error(f"Failed to load/process data: {exc}")
    st.stop()

forecast_df = forecast_next_24(df, target_col="kWh", horizon=24)
anomalies_df, scored_df = detect_anomalies_zscore(df, target_col="kWh", threshold=float(z_threshold))
recommendations = generate_recommendations(
    summary=summary,
    forecast_df=forecast_df,
    anomalies_df=anomalies_df,
    full_df=df,
    efficiency_threshold=float(efficiency_threshold),
)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total Energy (kWh)", f"{summary['total_kwh']:,.0f}")
kpi2.metric("Avg iKW_TR", f"{summary['avg_ikw_tr']:.3f}")
kpi3.metric("Load-kWh Correlation", f"{load_kwh_corr:.2f}")
kpi4.metric("Anomalies", int(len(anomalies_df)))

st.subheader("Data Preview")
st.dataframe(df.tail(30), use_container_width=True)

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("Energy Trend")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(df["Timestamp"], df["kWh"], label="kWh", color="#0f766e", linewidth=1.6)
    if not anomalies_df.empty:
        ax.scatter(anomalies_df["Timestamp"], anomalies_df["kWh"], color="#dc2626", label="Anomalies", s=28)
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("kWh")
    ax.legend()
    ax.grid(alpha=0.25)
    st.pyplot(fig)

with chart_col2:
    st.subheader("24-Step Forecast")
    fig2, ax2 = plt.subplots(figsize=(8, 3.8))
    history = df.tail(72)
    ax2.plot(history["Timestamp"], history["kWh"], label="Historical", color="#1d4ed8", linewidth=1.6)
    ax2.plot(forecast_df["Timestamp"], forecast_df["forecast_kWh"], label="Forecast", color="#f97316", linewidth=2)
    ax2.set_xlabel("Timestamp")
    ax2.set_ylabel("kWh")
    ax2.legend()
    ax2.grid(alpha=0.25)
    st.pyplot(fig2)

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
    label="Download HTML Report",
    data=report_html,
    file_name="hvac_ai_optimization_report.html",
    mime="text/html",
)
