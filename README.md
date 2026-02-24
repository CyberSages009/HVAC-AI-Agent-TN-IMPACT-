# HVAC-AI-Agent-TN-IMPACT
A multi-agent AI system that analyzes HVAC energy data from commercial buildings to forecast demand, detect inefficiencies, identify anomalies, and generate actionable optimization recommendations. The platform helps reduce energy consumption, improve operational efficiency, and support sustainable building management.

Test append: file update verification (2026-02-24)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/CyberSages009/HVAC-AI-Agent-TN-IMPACT-.git
   cd HVAC-AI-Agent-TN-IMPACT-
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Prepare your HVAC data in CSV format with columns:
   `timestamp`, `kWh`, `iKW-TR`, `ambient_temp`, `load_profile`.

2. Launch the Streamlit app:
   ```bash
   streamlit run main.py
   ```

3. Upload your CSV in the sidebar, or use the simulated dataset.

The app will:
- Load and validate HVAC data
- Run an Analyzer Agent for demand drift and anomaly diagnostics
- Show operational charts and anomaly tables
- Generate actionable recommendations and a downloadable decision snapshot

## Features

- Streamlit dashboard with HVAC operations, forecast, diagnostics, and recommendations tabs
- Rule-based Analyzer Agent with configurable anomaly sensitivity
- Z-score anomaly detection and demand drift monitoring
- Action-focused optimization recommendations from agent findings
