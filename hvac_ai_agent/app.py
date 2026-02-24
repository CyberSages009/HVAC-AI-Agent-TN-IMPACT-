from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from hvac_ai_agent.agents.analyzer import analyze_data, extract_operational_profile
from hvac_ai_agent.agents.diagnostic import run_diagnostics
from hvac_ai_agent.agents.forecaster import forecast_demand
from hvac_ai_agent.agents.optimizer import generate_recommendations
from hvac_ai_agent.report.report_generator import build_html_report


st.set_page_config(page_title="HVAC AI Optimization", layout="wide")

st.markdown(
    """
<style>
:root {
    --bg: #f3f8fd;
    --panel: #ffffff;
    --ink: #1f2a44;
    --muted: #5c6b8a;
    --accent: #1d4ed8;
    --accent-2: #0ea5e9;
    --warm: #f59e0b;
    --bubble: #eef5ff;
    --border: #d5def0;
}

html, body, [data-testid="stAppViewContainer"] {
    background:
        radial-gradient(1200px 420px at 95% -15%, #c6f3e8 0%, transparent 52%),
        radial-gradient(850px 380px at -8% 8%, #e7efff 0%, transparent 57%),
        var(--bg);
    color: var(--ink);
    font-family: "Manrope", "Segoe UI", sans-serif;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #eaf0ff 0%, #dfe8ff 100%);
    border-right: 1px solid #c8d5f0;
}

.sidebar-brand {
    background: linear-gradient(135deg, #1d4ed8 0%, #0284c7 100%);
    color: #ffffff;
    border-radius: 14px;
    padding: 12px 14px;
    margin-bottom: 8px;
    box-shadow: 0 10px 22px rgba(15, 118, 110, 0.22);
}

.sidebar-brand .title {
    font-family: "Space Grotesk", "Segoe UI", sans-serif;
    font-size: 1rem;
    font-weight: 700;
    line-height: 1.2;
}

.sidebar-brand .sub {
    margin-top: 4px;
    font-size: 0.78rem;
    opacity: 0.92;
}

.sidebar-block {
    background: rgba(255, 255, 255, 0.84);
    border: 1px solid #d4e0ec;
    border-radius: 12px;
    padding: 8px 10px;
    margin-bottom: 8px;
}

.hero {
    background: linear-gradient(130deg, #1d4ed8 0%, #0ea5e9 52%, #38bdf8 100%);
    border-radius: 18px;
    padding: 24px 28px;
    color: #ffffff;
    box-shadow: 0 16px 36px rgba(15, 118, 110, 0.22);
    margin-bottom: 14px;
}

.hero h1 {
    margin: 0;
    font-size: 2.05rem;
    font-family: "Space Grotesk", "Segoe UI", sans-serif;
    letter-spacing: 0.2px;
}

.hero p {
    margin: 8px 0 0 0;
    font-size: 1rem;
    opacity: 0.94;
}

.section-card {
    background: rgba(255, 255, 255, 0.92);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 14px 16px;
    box-shadow: 0 8px 18px rgba(16, 42, 67, 0.06);
}

.kpi-card {
    background: rgba(255, 255, 255, 0.97);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 14px 16px;
    box-shadow: 0 8px 18px rgba(16, 42, 67, 0.06);
}

.kpi-label {
    color: var(--muted);
    font-size: 0.84rem;
    margin-bottom: 3px;
}

.kpi-value {
    color: var(--ink);
    font-family: "Space Grotesk", "Segoe UI", sans-serif;
    font-size: 1.9rem;
    font-weight: 700;
    line-height: 1.1;
}

.bubble-wrap {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin: 6px 0 2px 0;
}

.bubble {
    background: var(--bubble);
    border: 1px solid #c4d6ff;
    color: #1d3766;
    border-radius: 999px;
    padding: 8px 13px;
    font-size: 0.88rem;
    box-shadow: 0 6px 16px rgba(15, 118, 110, 0.12);
    animation: floaty 3.6s ease-in-out infinite;
}

.bubble:nth-child(2n) { animation-delay: 0.4s; }
.bubble:nth-child(3n) { animation-delay: 0.8s; }

@keyframes floaty {
    0% { transform: translateY(0px); }
    50% { transform: translateY(-4px); }
    100% { transform: translateY(0px); }
}

.rec-box {
    background: linear-gradient(90deg, #fff9eb 0%, #fffef8 100%);
    border: 1px solid #ffe2ac;
    border-left: 5px solid #f59e0b;
    border-radius: 12px;
    padding: 10px 12px;
    margin-bottom: 8px;
    color: #0b3e3a;
}

[data-testid="stSidebar"] [data-baseweb="radio"] label {
    background: rgba(255, 255, 255, 0.88);
    border: 1px solid #cfdae7;
    border-radius: 10px;
    padding: 8px 10px;
    margin-bottom: 6px;
}

[data-testid="stSidebar"] [data-baseweb="radio"] input:checked + div {
    color: #1d4ed8;
    font-weight: 700;
}

.icon-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-bottom: 10px;
}

.icon-chip {
    background: #ffffff;
    border: 1px solid #d5def0;
    color: #25375e;
    border-radius: 999px;
    padding: 7px 12px;
    font-size: 0.83rem;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    box-shadow: 0 4px 10px rgba(40, 70, 140, 0.1);
    transition: all 0.2s ease;
    cursor: pointer;
}

.icon-chip:hover {
    transform: translateY(-2px) scale(1.02);
    background: #edf3ff;
    border-color: #9ab5ff;
}

.icon-dot {
    width: 18px;
    height: 18px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    color: #ffffff;
    font-size: 0.68rem;
    font-weight: 700;
}

.dot-a { background: #2563eb; }
.dot-f { background: #0284c7; }
.dot-d { background: #f59e0b; }
.dot-o { background: #7c3aed; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <h1>Multi-Agent HVAC & Energy Optimization</h1>
  <p>Virtual energy engineer for forecasting, anomaly diagnostics, and explainable optimization actions.</p>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        """
<div class="sidebar-brand">
  <div class="title">HVAC AI Console</div>
  <div class="sub">Analyzer • Forecaster • Diagnostic • Optimizer</div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-block">', unsafe_allow_html=True)
    st.markdown("**Left Menu**")
    page = st.radio(
        "View",
        ["▣ Overview", "◉ Forecast & Diagnostics", "◆ Recommendations & Report"],
        index=0,
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-block">', unsafe_allow_html=True)
    st.markdown("**Dataset**")
    uploaded_file = st.file_uploader("Upload HVAC CSV", type=["csv"])
    st.caption("Required: `timestamp` + any 3 core parameters.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-block">', unsafe_allow_html=True)
    st.markdown("**Forecast Settings**")
    horizon_hours = st.slider("Horizon (hours)", min_value=24, max_value=168, value=48, step=24)
    show_raw_data = st.toggle("Show processed table", value=False)
    st.markdown("</div>", unsafe_allow_html=True)

sample_path = "hvac_ai_agent/data/sample_dataset.csv"
if uploaded_file is None:
    st.info("No file uploaded. Running with sample dataset.")
    raw_df = pd.read_csv(sample_path)
else:
    raw_df = pd.read_csv(uploaded_file)

try:
    analysis = analyze_data(raw_df)
    clean_df = analysis.cleaned_df
    diagnostic = run_diagnostics(clean_df)
    forecast = forecast_demand(clean_df, horizon_hours=horizon_hours)
    recommendations = generate_recommendations(
        analysis.kpis,
        analysis.correlations,
        diagnostic.summary,
        forecast.forecast_df,
    )
except Exception as exc:
    st.error(f"Processing error: {exc}")
    st.stop()

peak_idx = forecast.forecast_df["pred_kwh"].idxmax() if not forecast.forecast_df.empty else None
peak_time = (
    pd.to_datetime(forecast.forecast_df.loc[peak_idx, "timestamp"]).strftime("%b %d %H:%M")
    if peak_idx is not None
    else "N/A"
)
peak_value = float(forecast.forecast_df["pred_kwh"].max()) if not forecast.forecast_df.empty else 0.0

bubble_texts = [
    f"Peak forecast: {peak_value:.1f} kWh @ {peak_time}",
    f"Anomalies detected: {int(diagnostic.summary.get('anomaly_count', 0))}",
    f"Avg efficiency: {analysis.kpis.get('avg_ikw_tr', 0):.3f} iKW-TR",
]
if "ambient_temp" in analysis.correlations:
    bubble_texts.append(f"kWh-temp correlation: {analysis.correlations['ambient_temp']:.2f}")

st.markdown('<div class="bubble-wrap">' + "".join(f'<div class="bubble">{txt}</div>' for txt in bubble_texts) + "</div>", unsafe_allow_html=True)
st.markdown(
    """
<div class="icon-strip">
  <div class="icon-chip"><span class="icon-dot dot-a">A</span> Analyzer</div>
  <div class="icon-chip"><span class="icon-dot dot-f">F</span> Forecaster</div>
  <div class="icon-chip"><span class="icon-dot dot-d">D</span> Diagnostic</div>
  <div class="icon-chip"><span class="icon-dot dot-o">O</span> Optimizer</div>
</div>
""",
    unsafe_allow_html=True,
)

if page == "▣ Overview":
    metric_cols = st.columns(4, gap="large")
    kpi_values = [
        ("Average kWh", f"{analysis.kpis.get('avg_kwh', 0):.2f}"),
        ("Peak kWh", f"{analysis.kpis.get('peak_kwh', 0):.2f}"),
        ("Average iKW-TR", f"{analysis.kpis.get('avg_ikw_tr', 0):.3f}"),
        ("Anomaly Ratio", f"{diagnostic.summary.get('anomaly_ratio_pct', 0):.2f}%"),
    ]
    for col, (label, value) in zip(metric_cols, kpi_values):
        col.markdown(
            f"""
<div class="kpi-card">
  <div class="kpi-label">{label}</div>
  <div class="kpi-value">{value}</div>
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("#### Energy Profile")
    if "kwh" in clean_df.columns:
        fig_kwh = px.line(clean_df, x="timestamp", y="kwh", title="Historical Energy Consumption (kWh)")
        fig_kwh.update_traces(line=dict(width=2.4, color="#0f766e"))
        fig_kwh.update_layout(plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", margin=dict(l=10, r=10, t=46, b=10))
        st.plotly_chart(fig_kwh, use_container_width=True)

    hourly_profile, daily_profile = extract_operational_profile(clean_df)
    plot_cols = st.columns(2, gap="large")
    if not hourly_profile.empty:
        fig_hourly = px.bar(hourly_profile, x="hour", y="kwh", title="Average Hourly kWh", color_discrete_sequence=["#14b8a6"])
        fig_hourly.update_layout(plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", margin=dict(l=8, r=8, t=42, b=8))
        plot_cols[0].plotly_chart(fig_hourly, use_container_width=True)
    if not daily_profile.empty:
        fig_daily = px.line(daily_profile, x="day", y="kwh", title="Daily Energy Totals")
        fig_daily.update_traces(line=dict(width=2.4, color="#0ea5e9"))
        fig_daily.update_layout(plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", margin=dict(l=8, r=8, t=42, b=8))
        plot_cols[1].plotly_chart(fig_daily, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if show_raw_data:
        st.markdown("#### Processed Data Preview")
        st.dataframe(clean_df.head(100), use_container_width=True)

elif page == "◉ Forecast & Diagnostics":
    left, right = st.columns([1.5, 1], gap="large")
    with left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### Demand Forecast")
        fig_fc = px.line(forecast.forecast_df, x="timestamp", y="pred_kwh", title=f"{horizon_hours}-Hour Demand Forecast")
        fig_fc.update_traces(line=dict(width=2.6, color="#f97316"))
        fig_fc.update_layout(plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", margin=dict(l=10, r=10, t=46, b=10))
        st.plotly_chart(fig_fc, use_container_width=True)
        st.dataframe(forecast.forecast_df.head(24), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### Diagnostics")
        anomaly_df = diagnostic.anomaly_df[diagnostic.anomaly_df["is_anomaly"]]
        st.write(f"Detected anomalies: {int(diagnostic.summary['anomaly_count'])}")
        st.dataframe(
            anomaly_df[[c for c in ["timestamp", "kwh", "ikw_tr", "ambient_temp", "load_profile"] if c in anomaly_df.columns]].head(50),
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("#### Optimization Recommendations")
    for i, rec in enumerate(recommendations, start=1):
        st.markdown(f'<div class="rec-box"><b>Action {i}:</b> {rec}</div>', unsafe_allow_html=True)

    if analysis.correlations:
        st.markdown("#### Correlation Signals")
        st.markdown(
            '<div class="bubble-wrap">'
            + "".join(
                f'<div class="bubble">{name}: {value:.2f}</div>' for name, value in analysis.correlations.items()
            )
            + "</div>",
            unsafe_allow_html=True,
        )

    html_report = build_html_report(
        kpis=analysis.kpis,
        correlations=analysis.correlations,
        diagnostic_summary=diagnostic.summary,
        recommendations=recommendations,
        forecast_df=forecast.forecast_df,
    )

    st.markdown("#### Decision Report")
    st.download_button(
        label="Download HTML Decision Report",
        data=html_report,
        file_name="hvac_decision_report.html",
        mime="text/html",
    )
    st.markdown("</div>", unsafe_allow_html=True)
