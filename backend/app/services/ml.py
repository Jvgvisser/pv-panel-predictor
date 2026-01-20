from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
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
    # drop tz for merge (meteo times are local)
    e["time"] = pd.to_datetime(e["ts"]).dt.tz_localize(None)
    e = e.drop(columns=["ts"])

    m = meteo_hourly.copy()
    m["time"] = pd.to_datetime(m["time"])

    df = pd.merge(m, e, on="time", how="inner").sort_values("time")

    # Lag features
    df["kwh_lag_24"] = df["kwh"].shift(24).fillna(0.0)
    df["gti_lag_24"] = df["global_tilted_irradiance"].shift(24).fillna(0.0)


    df["kwh_lag_168"] = df["kwh"].shift(168).fillna(0.0)
    df["gti_lag_168"] = df["global_tilted_irradiance"].shift(168).fillna(0.0)
    df = add_time_features(df, "time")
    df = df.dropna()
    return df


@dataclass
class TrainedModel:
    model: LGBMRegressor
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

        df = df.sort_values("time")

        # Validation: last 7 days (168h) when available
        val_n = min(168, max(24, int(len(df) * 0.15)))
        train_df = df.iloc[:-val_n] if len(df) > val_n else df
        val_df = df.iloc[-val_n:] if len(df) > val_n else df.iloc[0:0]

        X_train = train_df[features]
        y_train = train_df["kwh"]

        # LightGBM settings: good default for 1 year hourly series
        model = LGBMRegressor(
            n_estimators=1200,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=12,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.0,
            reg_lambda=0.0,
            random_state=42,
            n_jobs=-1,
        )

        # Early stopping if we have a validation set
        metrics = {"train_rows": int(len(train_df))}
        if len(val_df) > 0:
            X_val = val_df[features]
            y_val = val_df["kwh"]

            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="l1",
                callbacks=[],
            )

            pred = np.clip(model.predict(X_val), 0, None)
            metrics["val_rows"] = int(len(val_df))
            metrics["val_mae_kwh"] = float(mean_absolute_error(y_val, pred))
            metrics["best_iteration"] = int(getattr(model, "best_iteration_", model.n_estimators))
        else:
            model.fit(X_train, y_train)

        # Feature importance (handig voor debug)
        try:
            imp = list(zip(features, model.feature_importances_.tolist()))
            imp.sort(key=lambda x: x[1], reverse=True)
            metrics["feature_importance"] = imp[:8]
        except Exception:
            pass

        d = self._panel_dir(panel_id)
        joblib.dump({"model": model, "features": features}, d / "model.joblib")
        return metrics

    def load(self, panel_id: str) -> TrainedModel:
        d = self._panel_dir(panel_id)
        obj = joblib.load(d / "model.joblib")
        return TrainedModel(model=obj["model"], features=obj["features"])

    def predict(self, trained: TrainedModel, feature_df: pd.DataFrame) -> np.ndarray:
        X = feature_df[trained.features]
        pred = trained.model.predict(X)
        pred = np.clip(pred, 0, None)

        # Force nights to zero (no irradiance)
        if "global_tilted_irradiance" in feature_df.columns:
            pred = np.where(feature_df["global_tilted_irradiance"].to_numpy() <= 0.5, 0.0, pred)

        return pred

def build_training_frame_idxjoin(energy: pd.DataFrame, meteo: pd.DataFrame) -> pd.DataFrame:
    """
    DST-safe training frame builder.
    Joins energy and meteo on DatetimeIndex (no merge on 'time' column).
    """
    e = energy.copy()
    m = meteo.copy()

    if "time" in e.columns and not isinstance(e.index, pd.DatetimeIndex):
        e["time"] = pd.to_datetime(e["time"], utc=True, errors="coerce")
        e = e.dropna(subset=["time"]).set_index("time")
    if "time" in m.columns and not isinstance(m.index, pd.DatetimeIndex):
        m["time"] = pd.to_datetime(m["time"], utc=True, errors="coerce")
        m = m.dropna(subset=["time"]).set_index("time")

    if not isinstance(e.index, pd.DatetimeIndex) or not isinstance(m.index, pd.DatetimeIndex):
        raise ValueError("energy/meteo must have a DatetimeIndex (or a 'time' column to set index)")

    if e.index.tz is None and m.index.tz is not None:
        e.index = e.index.tz_localize("UTC")
    if m.index.tz is None and e.index.tz is not None:
        m.index = m.index.tz_localize("UTC")

    if "time" in e.columns:
        e = e.drop(columns=["time"])
    if "time" in m.columns:
        m = m.drop(columns=["time"])

    df = e.join(m, how="inner").sort_index()

    df["kwh_lag_24"] = df["kwh"].shift(24)
    if "global_tilted_irradiance" in df.columns:
        df["gti_lag_24"] = df["global_tilted_irradiance"].shift(24)

    df = add_time_features(df)
    df = df.dropna()
    return df

