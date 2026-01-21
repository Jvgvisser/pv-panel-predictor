from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

def add_time_features(df: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    """Voegt cyclische tijdskenmerken toe voor betere AI patronen."""
    out = df.copy()

    # Detecteer of we met de index of een kolom werken
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

    # Zet uren en dagen om in sinus/cosinus golven (zodat 23:00 en 00:00 dicht bij elkaar liggen)
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
        """Traint een LightGBM model op historische data."""
        if df.empty:
            raise ValueError("Training mislukt: De input dataframe is leeg.")

        df = df.copy()

        # Maak 'Lags' aan: leer van de opbrengst van gisteren
        if len(df) > 24:
            df["kwh_lag_24"] = df["kwh"].shift(24)
            df["gti_lag_24"] = df["global_tilted_irradiance"].shift(24)
        
        # Verwijder rijen die door de 'shift' leeg zijn geworden (de eerste 24 uur)
        df = df.dropna()

        # Definieer welke eigenschappen de AI moet gebruiken
        features = [
            "global_tilted_irradiance", "shortwave_radiation", "cloud_cover",
            "temperature_2m", "kwh_lag_24", "gti_lag_24",
            "hour_sin", "hour_cos", "doy_sin", "doy_cos",
        ]

        # Zorg dat alle kolommen bestaan, vul ontbrekende in met 0.0
        for col in features:
            if col not in df.columns:
                df[col] = 0.0

        if len(df) < 24:
            raise ValueError(f"Te weinig data na voorbereiding: {len(df)} rijen over.")

        # Split data: Gebruik 15% voor validatie (testen tijdens trainen)
        val_n = min(168, max(24, int(len(df) * 0.15)))
        train_df = df.iloc[:-val_n]
        val_df = df.iloc[-val_n:]

        # Initialiseer LightGBM
        model = LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.03,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )

        # Start het trainen
        model.fit(
            train_df[features], train_df["kwh"],
            eval_set=[(val_df[features], val_df["kwh"])],
            eval_metric="rmse"
        )
        
        # Sla het model op
        d = self._panel_dir(panel_id)
        joblib.dump({"model": model, "features": features}, d / "model.joblib")
        
        return {"status": "trained with LightGBM", "rows": len(df)}

    def load(self, panel_id: str) -> TrainedModel:
        d = self._panel_dir(panel_id)
        path = d / "model.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Geen getraind model gevonden voor {panel_id}")
        obj = joblib.load(path)
        return TrainedModel(model=obj["model"], features=obj["features"])

    def predict(self, trained: TrainedModel, feature_df: pd.DataFrame) -> np.ndarray:
        """Voorspelt opbrengst op basis van een getraind model."""
        # Zorg dat alle kolommen die tijdens training gebruikt zijn, ook nu bestaan
        for _col in trained.features:
            if _col not in feature_df.columns:
                feature_df[_col] = 0.0
        
        pred = trained.model.predict(feature_df[trained.features])
        pred = np.clip(pred, 0, None) # Opbrengst kan nooit negatief zijn

        # Snelkoppeling: Als er bijna geen zon is (nacht), forceer 0.0
        if "global_tilted_irradiance" in feature_df.columns:
            pred = np.where(feature_df["global_tilted_irradiance"] <= 0.5, 0.0, pred)
        
        return pred