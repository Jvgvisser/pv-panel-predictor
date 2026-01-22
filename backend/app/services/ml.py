from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from pvlib.solarposition import get_solarposition
import os

def add_time_features(df: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    """Voegt cyclische tijdskenmerken toe voor betere AI patronen."""
    out = df.copy()

    if time_col in out.columns:
        t = pd.to_datetime(out[time_col], utc=True)
        out["hour"] = t.dt.hour
        out["doy"] = t.dt.dayofyear
    elif isinstance(out.index, pd.DatetimeIndex):
        t = out.index
        out["hour"] = t.hour
        out["doy"] = t.dayofyear
    else:
        raise ValueError("add_time_features: geen 'time' kolom of DatetimeIndex gevonden")

    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    out["doy_sin"] = np.sin(2 * np.pi * out["doy"] / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * out["doy"] / 365.25)
    
    return out

def add_solar_position(df: pd.DataFrame, lat: float, lon: float) -> pd.DataFrame:
    """Berekent de exacte zonnestand (Azimuth/Elevation) voor schaduw-detectie."""
    out = df.copy()
    times = pd.to_datetime(out.index if isinstance(out.index, pd.DatetimeIndex) else out["time"], utc=True)
    
    solpos = get_solarposition(times, lat, lon)
    out["azimuth"] = solpos["azimuth"].values
    out["elevation"] = solpos["elevation"].values
    return out

@dataclass
class TrainedModel:
    model: LGBMRegressor
    features: list

class PanelModelService:
    def __init__(self, base_dir: str = "/opt/pv-panel-predictor/models") -> None:
        # We gebruiken nu een vast absoluut pad voor stabiliteit op de LXC
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_model_path(self, panel_id: str) -> Path:
        # GEFIXT: Slaat nu op als /models/panel_123.joblib (geen submappen meer)
        return self.base_dir / f"{panel_id}.joblib"

    def train(self, panel_id: str, df: pd.DataFrame) -> dict:
        """Traint een LightGBM model inclusief zonnestand en lags."""
        if df.empty:
            raise ValueError("Training mislukt: De input dataframe is leeg.")

        df = df.copy()

        if len(df) > 24:
            df["kwh_lag_24"] = df["kwh"].shift(24)
            df["gti_lag_24"] = df["global_tilted_irradiance"].shift(24)
        
        df = df.dropna()

        features = [
            "global_tilted_irradiance", "shortwave_radiation", "cloud_cover",
            "temperature_2m", "kwh_lag_24", "gti_lag_24",
            "hour_sin", "hour_cos", "doy_sin", "doy_cos",
            "azimuth", "elevation"
        ]

        for col in features:
            if col not in df.columns:
                df[col] = 0.0

        if len(df) < 24:
            raise ValueError(f"Te weinig data na voorbereiding: {len(df)} rijen over.")

        val_n = min(168, max(24, int(len(df) * 0.15)))
        train_df = df.iloc[:-val_n]
        val_df = df.iloc[-val_n:]

        model = LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.03,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )

        model.fit(
            train_df[features], train_df["kwh"],
            eval_set=[(val_df[features], val_df["kwh"])],
            eval_metric="rmse"
        )
        
        # Gebruik de nieuwe platte pad-structuur
        target_path = self._get_model_path(panel_id)
        joblib.dump({"model": model, "features": features}, target_path)
        
        return {"status": "trained", "path": str(target_path), "rows": len(df)}

    def load(self, panel_id: str) -> TrainedModel:
        path = self._get_model_path(panel_id)
        if not path.exists():
            # In plaats van een crash, geven we None terug zodat de API 404 kan geven
            return None
        
        obj = joblib.load(path)
        return TrainedModel(model=obj["model"], features=obj["features"])

    def predict(self, trained: TrainedModel, feature_df: pd.DataFrame) -> np.ndarray:
        if trained is None:
            return np.array([])
            
        df = feature_df.copy()
        for _col in trained.features:
            if _col not in df.columns:
                df[_col] = 0.0
        
        pred = trained.model.predict(df[trained.features])
        pred = np.clip(pred, 0, None)

        if "global_tilted_irradiance" in df.columns:
            pred = np.where(df["global_tilted_irradiance"] <= 0.5, 0.0, pred)
        if "elevation" in df.columns:
            pred = np.where(df["elevation"] <= 0, 0.0, pred)
        
        return pred