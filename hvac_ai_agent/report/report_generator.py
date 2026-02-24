"""Report generator for HVAC optimization decisions."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List

import pandas as pd


def _fmt(v: float) -> str:
    if pd.isna(v):
        return "N/A"
    return f"{v:,.2f}"


def build_html_report(
    kpis: Dict[str, float],
    correlations: Dict[str, float],
    diagnostic_summary: Dict[str, float],
    recommendations: List[str],
    forecast_df: pd.DataFrame,
) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    forecast_rows = "".join(
        f"<tr><td>{row.timestamp}</td><td>{row.pred_kwh:.2f}</td></tr>"
        for row in forecast_df.head(24).itertuples(index=False)
    )

    rec_items = "".join(f"<li>{rec}</li>" for rec in recommendations)
    corr_items = "".join(f"<li>{k}: {v:.2f}</li>" for k, v in correlations.items()) or "<li>N/A</li>"

    return f"""
<!DOCTYPE html>
<html>
<head>
<meta charset=\"utf-8\" />
<title>HVAC Decision Report</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2937; }}
h1, h2 {{ color: #0f766e; }}
.card {{ background: #f8fafc; border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; margin-bottom: 12px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #d1d5db; padding: 8px; text-align: left; }}
th {{ background: #e2e8f0; }}
</style>
</head>
<body>
<h1>HVAC Multi-Agent Decision Report</h1>
<p>Generated: {generated_at}</p>

<h2>Key Performance Indicators</h2>
<div class=\"card\">
<p><b>Records:</b> {_fmt(kpis.get('records', float('nan')))}</p>
<p><b>Average kWh:</b> {_fmt(kpis.get('avg_kwh', float('nan')))}</p>
<p><b>Peak kWh:</b> {_fmt(kpis.get('peak_kwh', float('nan')))}</p>
<p><b>Average iKW-TR:</b> {_fmt(kpis.get('avg_ikw_tr', float('nan')))}</p>
<p><b>Average Ambient Temperature:</b> {_fmt(kpis.get('avg_ambient_temp', float('nan')))}</p>
</div>

<h2>Diagnostic Summary</h2>
<div class=\"card\">
<p><b>Anomaly Count:</b> {_fmt(diagnostic_summary.get('anomaly_count', float('nan')))}</p>
<p><b>Anomaly Ratio (%):</b> {_fmt(diagnostic_summary.get('anomaly_ratio_pct', float('nan')))}</p>
<p><b>Efficiency Degradation (%):</b> {_fmt(diagnostic_summary.get('efficiency_degradation_pct', float('nan')))}</p>
</div>

<h2>Correlation Insight</h2>
<div class=\"card\"><ul>{corr_items}</ul></div>

<h2>Optimization Recommendations</h2>
<div class=\"card\"><ol>{rec_items}</ol></div>

<h2>24-Hour Forecast Snapshot</h2>
<table>
<tr><th>Timestamp</th><th>Predicted kWh</th></tr>
{forecast_rows}
</table>
</body>
</html>
"""
