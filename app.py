import numpy as np
import pandas as pd
import streamlit as st
from dataclasses import dataclass


REQUIRED_COLUMNS = ["timestamp", "kWh", "iKW-TR", "ambient_temp", "load_profile"]


@dataclass
class AgentFinding:
    severity: str
    title: str
    detail: str
    recommendation: str


class AnalyzerAgent:
    def __init__(self, sensitivity: str = "Medium") -> None:
        self.sensitivity = sensitivity
        self.z_threshold = {"Low": 2.7, "Medium": 2.3, "High": 1.9}.get(sensitivity, 2.3)

    def run(self, df: pd.DataFrame) -> dict:
        working = df.copy()
        working["kWh_z"] = (working["kWh"] - working["kWh"].mean()) / working["kWh"].std(ddof=0)
        anomalies = working.loc[working["kWh_z"].abs() > self.z_threshold, ["timestamp", "kWh", "iKW-TR"]]

        recent_window = working.tail(12)
        baseline_window = working.tail(72).head(60)

        recent_kwh = float(recent_window["kWh"].mean())
        baseline_kwh = float(baseline_window["kWh"].mean()) if not baseline_window.empty else recent_kwh
        demand_shift_pct = ((recent_kwh - baseline_kwh) / baseline_kwh * 100) if baseline_kwh else 0.0

        temp_kwh_corr = float(working["ambient_temp"].corr(working["kWh"]))
        peak_row = working.loc[working["kWh"].idxmax()]

        findings: list[AgentFinding] = []
        findings.append(
            AgentFinding(
                severity="high" if demand_shift_pct > 8 else "medium" if demand_shift_pct > 4 else "low",
                title="Recent demand drift",
                detail=f"Last 12h avg demand is {recent_kwh:.1f} kWh vs 72h baseline {baseline_kwh:.1f} kWh ({demand_shift_pct:+.1f}%).",
                recommendation="Tune chilled-water and AHU schedules for late-day hours if drift persists for 2+ days.",
            )
        )

        findings.append(
            AgentFinding(
                severity="medium" if temp_kwh_corr > 0.55 else "low",
                title="Weather sensitivity",
                detail=f"Ambient temperature to energy correlation is {temp_kwh_corr:.2f}.",
                recommendation="Pre-cool before peak ambient periods and verify sensor calibration around noon-afternoon intervals.",
            )
        )

        findings.append(
            AgentFinding(
                severity="high" if anomalies.shape[0] >= 8 else "medium" if anomalies.shape[0] >= 4 else "low",
                title="Anomaly concentration",
                detail=f"Detected {int(anomalies.shape[0])} anomaly points at sensitivity '{self.sensitivity}' (|z| > {self.z_threshold}).",
                recommendation="Inspect top anomaly windows for valve hunting, staging conflicts, and occupancy mismatches.",
            )
        )

        findings.append(
            AgentFinding(
                severity="medium",
                title="Peak energy event",
                detail=f"Highest demand occurred at {peak_row['timestamp']} with {peak_row['kWh']:.1f} kWh and iKW-TR {peak_row['iKW-TR']:.3f}.",
                recommendation="Use this timestamp as a replay scenario for control-strategy simulation.",
            )
        )

        findings = sorted(findings, key=lambda item: {"high": 0, "medium": 1, "low": 2}[item.severity])
        return {
            "anomalies": anomalies,
            "findings": findings,
            "demand_shift_pct": demand_shift_pct,
            "corr_temp_kwh": temp_kwh_corr,
        }


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=DM+Sans:wght@400;500;700&display=swap');

        :root {
            --bg-soft: #f4f8f7;
            --card: #ffffff;
            --ink: #12212b;
            --muted: #4b5b65;
            --accent: #0f766e;
            --accent-2: #0369a1;
            --line: #d8e3e7;
        }

        .stApp {
            background:
                radial-gradient(1200px 400px at 0% -10%, #c9f0ec 0%, transparent 55%),
                radial-gradient(1100px 450px at 100% -20%, #cfe9fb 0%, transparent 58%),
                var(--bg-soft);
            color: var(--ink);
        }

        h1, h2, h3, h4 {
            font-family: 'Space Grotesk', sans-serif !important;
            color: var(--ink);
            letter-spacing: 0.1px;
        }

        p, div, span, label {
            font-family: 'DM Sans', sans-serif !important;
        }

        .hero {
            border: 1px solid var(--line);
            background: linear-gradient(135deg, rgba(15,118,110,0.12), rgba(3,105,161,0.12));
            border-radius: 18px;
            padding: 20px 24px;
            margin-bottom: 12px;
            box-shadow: 0 8px 24px rgba(18, 33, 43, 0.08);
        }

        .chip {
            display: inline-block;
            border: 1px solid rgba(15, 118, 110, 0.35);
            color: #0b5e57;
            border-radius: 999px;
            padding: 3px 10px;
            margin-right: 6px;
            font-size: 12px;
            font-weight: 600;
            background: rgba(255,255,255,0.7);
        }

        .kpi {
            border: 1px solid var(--line);
            border-radius: 14px;
            background: var(--card);
            padding: 12px 14px;
            box-shadow: 0 6px 16px rgba(18, 33, 43, 0.06);
        }

        .kpi-label { color: var(--muted); font-size: 12px; }
        .kpi-value { color: var(--ink); font-size: 22px; font-weight: 700; }

        [data-testid="stSidebar"] {
            border-right: 1px solid var(--line);
            background: rgba(255,255,255,0.6);
            backdrop-filter: blur(6px);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def generate_demo_data(hours: int = 168) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    index = pd.date_range(end=pd.Timestamp.now().floor("h"), periods=hours, freq="h")

    base_temp = 30 + 4 * np.sin(np.arange(hours) * 2 * np.pi / 24)
    load = 60 + 25 * np.clip(np.sin((np.arange(hours) - 7) * 2 * np.pi / 24), 0, None)
    noise = rng.normal(0, 3, size=hours)

    df = pd.DataFrame(
        {
            "timestamp": index,
            "ambient_temp": (base_temp + rng.normal(0, 0.8, size=hours)).round(2),
            "load_profile": (load + rng.normal(0, 4, size=hours)).round(2),
            "kWh": (250 + 3.4 * load + 2.2 * base_temp + noise).round(2),
        }
    )
    df["iKW-TR"] = (df["kWh"] / (df["load_profile"].clip(lower=1) * 0.85)).round(3)
    return df


def parse_uploaded_data(file) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(file)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df, missing


def metric_block(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def run() -> None:
    st.set_page_config(page_title="HVAC Multi-Agent Optimizer", page_icon="??", layout="wide")
    inject_styles()

    with st.sidebar:
        st.header("Control Panel")
        st.caption("Configure dataset and analysis scope")

        uploaded = st.file_uploader("Upload HVAC CSV", type=["csv"])
        use_demo = st.toggle("Use simulated dataset", value=uploaded is None)

        st.divider()
        building_type = st.selectbox("Building Type", ["Hotel", "Mall", "Office", "Hospital"])
        forecast_horizon = st.slider("Forecast Horizon (hours)", min_value=24, max_value=168, value=72, step=24)
        anomaly_sensitivity = st.select_slider("Anomaly Sensitivity", options=["Low", "Medium", "High"], value="Medium")

        st.divider()
        st.markdown("**Agent Status**")
        st.markdown("- Analyzer: Active")
        st.markdown("- Forecaster: UI Ready")
        st.markdown("- Diagnostic: UI Ready")
        st.markdown("- Optimizer: UI Ready")
        st.markdown("- Report Generator: UI Ready")

    data_source = "Simulated"
    if uploaded is not None and not use_demo:
        df, missing = parse_uploaded_data(uploaded)
        if missing:
            st.error(f"Missing required columns: {', '.join(missing)}")
            st.info(f"Expected columns: {', '.join(REQUIRED_COLUMNS)}")
            df = generate_demo_data()
            data_source = "Simulated fallback"
        else:
            df = df.sort_values("timestamp").dropna(subset=["timestamp"]).reset_index(drop=True)
            data_source = "Uploaded CSV"
    else:
        df = generate_demo_data()

    analyzer = AnalyzerAgent(sensitivity=anomaly_sensitivity)
    analyzer_result = analyzer.run(df)

    st.markdown(
        """
        <section class="hero">
            <h2 style="margin:0;">HVAC Multi-Agent AI Optimization System</h2>
            <p style="margin:8px 0 10px 0; color:#35515f;">
                Virtual Energy Engineer for demand forecasting, anomaly diagnostics, and actionable optimization.
            </p>
            <span class="chip">TN Impact 2026</span>
            <span class="chip">Explainable AI</span>
            <span class="chip">Production-Ready UI Skeleton</span>
        </section>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    total_kwh = f"{df['kWh'].sum():,.0f}"
    avg_ikwtr = f"{df['iKW-TR'].mean():.3f}"
    avg_temp = f"{df['ambient_temp'].mean():.1f} C"
    peak_load = f"{df['load_profile'].max():.1f}%"

    with c1:
        metric_block("Total Energy (kWh)", total_kwh)
    with c2:
        metric_block("Avg iKW-TR", avg_ikwtr)
    with c3:
        metric_block("Avg Ambient Temp", avg_temp)
    with c4:
        metric_block("Peak Load Profile", peak_load)

    st.caption(
        f"Data Source: {data_source} | Building Type: {building_type} | Forecast Horizon: {forecast_horizon}h | Anomaly Sensitivity: {anomaly_sensitivity}"
    )

    tab1, tab2, tab3, tab4 = st.tabs(["Operations", "Forecast", "Diagnostics", "Recommendations"])

    with tab1:
        left, right = st.columns([2, 1])
        with left:
            st.subheader("Energy and Weather Trend")
            trend = df.set_index("timestamp")[["kWh", "ambient_temp"]]
            st.line_chart(trend, use_container_width=True)
        with right:
            st.subheader("Cooling Efficiency")
            st.area_chart(df.set_index("timestamp")[["iKW-TR"]], use_container_width=True)

        st.subheader("Data Preview")
        st.dataframe(df.tail(48), use_container_width=True, height=260)

    with tab2:
        st.subheader("Short-Term Demand Forecast")
        st.info("Forecast module will be connected to Forecaster Agent in Phase 3.")
        baseline = df[["timestamp", "kWh"]].tail(24).copy()
        baseline["forecast_kWh"] = baseline["kWh"].rolling(3, min_periods=1).mean()
        st.line_chart(
            baseline.set_index("timestamp")[["kWh", "forecast_kWh"]],
            use_container_width=True,
        )

    with tab3:
        st.subheader("Anomaly and Efficiency Diagnostics")
        st.info("Analyzer Agent is running with rule-based diagnostics.")
        flagged = analyzer_result["anomalies"]
        st.metric("Potential Anomaly Points", int(flagged.shape[0]))
        st.metric("Recent Demand Shift", f"{analyzer_result['demand_shift_pct']:+.1f}%")
        st.metric("Temp-Energy Correlation", f"{analyzer_result['corr_temp_kwh']:.2f}")
        st.dataframe(flagged.tail(10), use_container_width=True)

    with tab4:
        st.subheader("Agent Recommendations")
        st.success("Analyzer Agent generated actionable findings.")
        severity_badge = {"high": "HIGH", "medium": "MEDIUM", "low": "LOW"}
        for i, finding in enumerate(analyzer_result["findings"], start=1):
            st.markdown(
                f"**{i}. [{severity_badge[finding.severity]}] {finding.title}**\n\n"
                f"- Observation: {finding.detail}\n"
                f"- Action: {finding.recommendation}"
            )
        st.download_button(
            "Download Decision Snapshot (.txt)",
            data="\n".join(
                [
                    "HVAC Analyzer Agent Snapshot",
                    *[
                        f"{idx}. [{finding.severity.upper()}] {finding.title} | {finding.detail} | {finding.recommendation}"
                        for idx, finding in enumerate(analyzer_result["findings"], start=1)
                    ],
                ]
            ),
            file_name="hvac_decision_snapshot.txt",
            mime="text/plain",
        )


if __name__ == "__main__":
    run()
