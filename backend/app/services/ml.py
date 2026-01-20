from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error


def add_time_features(df: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    out = df.copy()
    t = pd.to_datetime(out[time_col])
    out["hour"] = t.dt.hour
    out["doy"] = t.dt.dayofyear

    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    out["doy_sin"] = np.sin(2 * np.pi * out["doy"] / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * out["doy"] / 365.25)
    return out


def build_training_frame(
    energy_hourly: pd.DataFrame,  # columns: ts, kwh
    meteo_hourly: pd.DataFrame,   # columns: time, ...
) -> pd.DataFrame:
    e = energy_hourly.copy()
    e["time"] = pd.to_datetime(e["ts"]).dt.tz_localize(None)  # drop tz for merge
    e = e.drop(columns=["ts"])

    m = meteo_hourly.copy()
    m["time"] = pd.to_datetime(m["time"])

    df = pd.merge(m, e, on="time", how="inner")
    df = df.sort_values("time")

    # simple lag features (previous day same hour)
    df["kwh_lag_24"] = df["kwh"].shift(24).fillna(0.0)
    df["gti_lag_24"] = df["global_tilted_irradiance"].shift(24).fillna(0.0)

    df = add_time_features(df, "time")
    df = df.dropna()
    return df


@dataclass
class TrainedModel:
    model: HistGradientBoostingRegressor
    features: list


class PanelModelService:
    def __init__(self, base_dir: str = "data/models") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _panel_dir(self, panel_id: str) -> Path:
        d = self.base_dir / panel_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def train(self, panel_id: str, df: pd.DataFrame) -> dict:
        # Define features
        features = [
            "global_tilted_irradiance",
            "shortwave_radiation",
            "cloud_cover",
            "temperature_2m",
            "kwh_lag_24",
            "gti_lag_24",
            "hour_sin", "hour_cos",
            "doy_sin", "doy_cos",
        ]

        # Train/val split: last 7 days as validation (168 hours)
        df = df.sort_values("time")
        val_n = min(168, max(24, int(len(df) * 0.15)))
        train_df = df.iloc[:-val_n] if len(df) > val_n else df
        val_df = df.iloc[-val_n:] if len(df) > val_n else df.iloc[0:0]

        X_train = train_df[features].to_numpy()
        y_train = train_df["kwh"].to_numpy()

        model = HistGradientBoostingRegressor(
            loss="squared_error",
            max_depth=6,
            learning_rate=0.08,
            max_iter=400,
            random_state=42,
        )
        model.fit(X_train, y_train)

        metrics = {"train_rows": int(len(train_df))}
        if len(val_df) > 0:
            X_val = val_df[features].to_numpy()
            y_val = val_df["kwh"].to_numpy()
            pred = model.predict(X_val)
            pred = np.clip(pred, 0, None)
            metrics["val_rows"] = int(len(val_df))
            metrics["val_mae_kwh"] = float(mean_absolute_error(y_val, pred))

        # Save
        d = self._panel_dir(panel_id)
        joblib.dump({"model": model, "features": features}, d / "model.joblib")
        return metrics

    def load(self, panel_id: str) -> TrainedModel:
        d = self._panel_dir(panel_id)
        obj = joblib.load(d / "model.joblib")
        return TrainedModel(model=obj["model"], features=obj["features"])

    def predict(self, trained: TrainedModel, feature_df: pd.DataFrame) -> np.ndarray:
        X = feature_df[trained.features].to_numpy()
        pred = trained.model.predict(X)
        return np.clip(pred, 0, None)
