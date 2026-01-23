from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from pvlib.solarposition import get_solarposition
import os

# Importeer XGBoost alleen als het nodig is, om fouten te voorkomen als het nog niet geinstalleerd is
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

def add_time_features(df: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    """Voegt cyclische tijdskenmerken toe (sin/cos) voor uren en dagen."""
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
    """Berekent Azimuth en Elevation voor schaduw-detectie."""
    out = df.copy()
    times = pd.to_datetime(out.index if isinstance(out.index, pd.DatetimeIndex) else out["time"], utc=True)
    
    solpos = get_solarposition(times, lat, lon)
    out["azimuth"] = solpos["azimuth"].values
    out["elevation"] = solpos["elevation"].values
    return out

@dataclass
class TrainedModel:
    model: any # Kan LGBM of XGB zijn
    features: list

class PanelModelService:
    def __init__(self, base_dir: str = "/opt/pv-panel-predictor/models") -> None:
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        print(f"--- PanelModelService: Modellen worden opgeslagen in: {self.base_dir} ---")

    def _get_model_path(self, panel_id: str) -> Path:
        return self.base_dir / f"{panel_id}.joblib"

    def train(self, panel_id: str, df: pd.DataFrame, model_type: str = "lightgbm") -> dict:
        """Traint een model (LightGBM of XGBoost) en slaat het op."""
        if df.empty:
            raise ValueError(f"Training mislukt voor {panel_id}: DataFrame is leeg.")

        df = df.copy()

        # Feature Engineering: Lags
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
            raise ValueError(f"Te weinig data na lag-creatie: {len(df)} rijen over.")

        X = df[features]
        y = df["kwh"]

        if model_type == "lightgbm":
            # Bestaande LightGBM logica
            val_n = min(168, max(24, int(len(df) * 0.15)))
            train_X = X.iloc[:-val_n]
            train_y = y.iloc[:-val_n]
            val_X = X.iloc[-val_n:]
            val_y = y.iloc[-val_n:]

            model = LGBMRegressor(
                n_estimators=1000,
                learning_rate=0.03,
                num_leaves=31,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            model.fit(
                train_X, train_y,
                eval_set=[(val_X, val_y)],
                eval_metric="rmse"
            )
        elif model_type == "xgboost":
            if XGBRegressor is None:
                raise ImportError("XGBoost is niet geinstalleerd. Gebruik 'pip install xgboost'")
            
            model = XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                random_state=42,
                objective='reg:squarederror'
            )
            model.fit(X, y)
        else:
            raise ValueError(f"Onbekend model_type: {model_type}")
        
        # Opslaan (gebruikt de panel_id die vanuit main.py wordt gestuurd, bijv 'p01' of 'p01_xgb')
        target_path = self._get_model_path(panel_id)
        joblib.dump({"model": model, "features": features}, target_path)
        
        print(f"✅ {model_type} model succesvol getraind: {target_path}")
        return {"status": "trained", "model": model_type, "rows": len(df)}

    def load(self, panel_id: str) -> TrainedModel | None:
        path = self._get_model_path(panel_id)
        if not path.exists():
            return None
        
        try:
            obj = joblib.load(path)
            return TrainedModel(model=obj["model"], features=obj["features"])
        except Exception as e:
            print(f"❌ Fout bij laden van model {panel_id}: {e}")
            return None

    def predict(self, trained: TrainedModel, feature_df: pd.DataFrame) -> np.ndarray:
        if trained is None:
            return np.array([])
            
        df = feature_df.copy()
        for _col in trained.features:
            if _col not in df.columns:
                df[_col] = 0.0
        
        pred = trained.model.predict(df[trained.features])
        pred = np.clip(pred, 0, None)

        # Forceer 0 opbrengst bij nacht
        if "global_tilted_irradiance" in df.columns:
            pred = np.where(df["global_tilted_irradiance"] <= 0.5, 0.0, pred)
        if "elevation" in df.columns:
            pred = np.where(df["elevation"] <= 0, 0.0, pred)
        
        return pred