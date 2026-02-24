"""Forecaster agent for short-term energy demand prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


@dataclass
class ForecastResult:
    forecast_df: pd.DataFrame
    model_summary: Dict[str, float]


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)
    features["hour"] = df["timestamp"].dt.hour
    features["dayofweek"] = df["timestamp"].dt.dayofweek
    features["lag_1"] = df["kwh"].shift(1)
    features["lag_24"] = df["kwh"].shift(24)
    if "ambient_temp" in df.columns:
        features["ambient_temp"] = df["ambient_temp"]
    if "load_profile" in df.columns:
        features["load_profile"] = df["load_profile"]
    return features


def forecast_demand(df: pd.DataFrame, horizon_hours: int = 24) -> ForecastResult:
    if "kwh" not in df.columns:
        raise ValueError("Forecasting requires kwh column.")

    work_df = df.copy().sort_values("timestamp").reset_index(drop=True)
    feature_df = _build_features(work_df)

    mask = ~feature_df.isna().any(axis=1)
    X = feature_df[mask]
    y = work_df.loc[mask, "kwh"]

    if len(X) < 30:
        raise ValueError("Not enough data for forecasting. Provide at least 30 valid records.")

    model = LinearRegression()
    model.fit(X, y)
    feature_defaults = X.median(numeric_only=True).to_dict()

    inferred_freq = pd.infer_freq(work_df["timestamp"])
    if inferred_freq is None:
        inferred_freq = "H"

    future_rows = []
    history = work_df.copy()

    for _ in range(horizon_hours):
        next_time = history["timestamp"].iloc[-1] + pd.tseries.frequencies.to_offset(inferred_freq)
        next_row = {"timestamp": next_time}

        if "ambient_temp" in history.columns:
            next_row["ambient_temp"] = history["ambient_temp"].iloc[-24:].mean()
        if "load_profile" in history.columns:
            next_row["load_profile"] = history["load_profile"].iloc[-24:].mean()

        history = pd.concat([history, pd.DataFrame([next_row])], ignore_index=True)
        features = _build_features(history).iloc[[-1]]
        for col, default_value in feature_defaults.items():
            if col in features.columns:
                features[col] = features[col].fillna(default_value)
        pred = float(model.predict(features)[0])
        history.loc[history.index[-1], "kwh"] = max(pred, 0.0)

        future_rows.append({"timestamp": next_time, "pred_kwh": max(pred, 0.0)})

    coef_summary = {name: float(val) for name, val in zip(X.columns, model.coef_)}
    coef_summary["intercept"] = float(model.intercept_)

    return ForecastResult(forecast_df=pd.DataFrame(future_rows), model_summary=coef_summary)
