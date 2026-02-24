from __future__ import annotations

from html import escape
from typing import Any


def generate_html_report(
    summary_metrics: dict[str, Any],
    forecast_value: float,
    anomaly_count: int,
    recommendations: list[str],
) -> str:
    """Build a simple HTML report for download."""
    rows = "".join(
        [
            f"<tr><td>Total Energy (kWh)</td><td>{summary_metrics['total_kwh']:.2f}</td></tr>",
            f"<tr><td>Average Energy (kWh)</td><td>{summary_metrics['avg_kwh']:.2f}</td></tr>",
            f"<tr><td>Peak Energy (kWh)</td><td>{summary_metrics['peak_kwh']:.2f}</td></tr>",
            f"<tr><td>Average iKW_TR</td><td>{summary_metrics['avg_ikw_tr']:.3f}</td></tr>",
            f"<tr><td>Average Temperature (C)</td><td>{summary_metrics['avg_temp']:.2f}</td></tr>",
            f"<tr><td>Average Load (%)</td><td>{summary_metrics['avg_load']:.2f}</td></tr>",
            f"<tr><td>Forecast Next-24 Avg (kWh)</td><td>{forecast_value:.2f}</td></tr>",
            f"<tr><td>Detected Anomalies</td><td>{anomaly_count}</td></tr>",
        ]
    )

    rec_items = "".join([f"<li>{escape(item)}</li>" for item in recommendations])

    return f"""
<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>HVAC AI Optimization Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2937; }}
    h1 {{ color: #0f766e; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
    th, td {{ border: 1px solid #d1d5db; padding: 10px; text-align: left; }}
    th {{ background: #f3f4f6; }}
    .box {{ background: #f8fafc; border: 1px solid #dbeafe; padding: 14px; border-radius: 8px; }}
  </style>
</head>
<body>
  <h1>HVAC Multi-Agent AI Optimization Report</h1>
  <div class=\"box\">Summary window: {summary_metrics['start']} to {summary_metrics['end']}</div>

  <h2>Summary Metrics</h2>
  <table>
    <thead><tr><th>Metric</th><th>Value</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>

  <h2>Recommendations</h2>
  <ol>{rec_items}</ol>
</body>
</html>
"""
