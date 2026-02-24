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

1. Prepare your HVAC data in CSV format with columns: `timestamp`, `temperature`, `humidity`, `energy_consumption`, etc.

2. Place the data file (e.g., `hvac_data.csv`) in the project directory.

3. Run the analysis:
   ```bash
   python main.py
   ```

The script will:
- Load and preprocess the data
- Train a machine learning model to forecast energy consumption
- Detect anomalies in energy usage
- Provide optimization recommendations

## Features

- Energy consumption forecasting using Random Forest regression
- Anomaly detection based on statistical thresholds
- Automated recommendations for system optimization
- Data visualization of predictions vs actual values
