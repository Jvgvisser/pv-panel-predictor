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
    if time_col in out.columns:
        t = pd.to_datetime(out[time_col], utc=True, errors="coerce")
    elif isinstance(out.index, pd.DatetimeIndex):
        t = out.index
        t = t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")
    else:
        raise ValueError("add_time_features: nood aan 'time' kolom of DatetimeIndex")

    out["hour"] = t.dt.hour
    out["doy"] = t.dt.dayofyear
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    out["doy_sin"] = np.sin(2 * np.pi * out["doy"] / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * out["doy"] / 365.25)
    return out

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
            "global_tilted_irradiance", "shortwave_radiation", "cloud_cover",
            "temperature_2m", "kwh_lag_24", "gti_lag_24",
            "hour_sin", "hour_cos", "doy_sin", "doy_cos",
        ]

        for col in features:
            if col not in df.columns:
                df[col] = 0.0

        # Split data voor validatie (laatste 7 dagen)
        val_n = min(168, max(24, int(len(df) * 0.15)))
        train_df = df.iloc[:-val_n] if len(df) > val_n else df
        val_df = df.iloc[-val_n:] if len(df) > val_n else df.iloc[0:0]

        model = LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )

        model.fit(train_df[features], train_df["kwh"])
        
        d = self._panel_dir(panel_id)
        joblib.dump({"model": model, "features": features}, d / "model.joblib")
        return {"status": "trained with LightGBM", "rows": len(df)}

    def load(self, panel_id: str) -> TrainedModel:
        d = self._panel_dir(panel_id)
        obj = joblib.load(d / "model.joblib")
        return TrainedModel(model=obj["model"], features=obj["features"])

    def predict(self, trained: TrainedModel, feature_df: pd.DataFrame) -> np.ndarray:
        for _col in trained.features:
            if _col not in feature_df.columns:
                feature_df[_col] = 0.0
        
        pred = trained.model.predict(feature_df[trained.features])
        pred = np.clip(pred, 0, None)
        # Forceer 0 bij geen instraling
        if "global_tilted_irradiance" in feature_df.columns:
            pred = np.where(feature_df["global_tilted_irradiance"] <= 0.5, 0.0, pred)
        return pred