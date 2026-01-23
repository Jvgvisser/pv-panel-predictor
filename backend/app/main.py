from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import json
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List

from backend.app.models.panel import PanelConfig
from backend.app.storage.panels_repo import PanelsRepo
from backend.app.services.ha_stats_ws import HAStatsWSClient
from backend.app.services.open_meteo_client import OpenMeteoClient
from backend.app.services.ml import PanelModelService, add_time_features, add_solar_position

app = FastAPI(title="PV Panel Predictor")

repo = PanelsRepo()
meteo = OpenMeteoClient()
ms = PanelModelService()

# --- MODELS FOR CONFIGURATION ---

class GlobalConfig(BaseModel):
    ha_base_url: str
    ha_token: str
    latitude: float
    longitude: float
    evcc_url: str = ""

# --- HELPER FUNCTIONS ---

def prepare_features_for_model(df_in, panel):
    """
    Centrale functie voor feature engineering.
    Zorgt dat training, predictie en evaluatie exact dezelfde kolommen gebruiken.
    """
    df = df_in.copy()
    if "time" not in df.columns:
        df["time"] = pd.to_datetime(df.index, utc=True)
    
    # Voeg lags en tijd features toe
    df["kwh_lag_24"] = 0.0 
    df["gti_lag_24"] = df["global_tilted_irradiance"].shift(24).fillna(0.0)
    
    df = add_time_features(df, "time")
    df = add_solar_position(df, panel.latitude, panel.longitude)
    return df

async def _fetch_panel_kwh_stats(panel, days: int):
    """Haalt de werkelijke kWh opbrengst uit Home Assistant Statistics."""
    ws = HAStatsWSClient(base_url=panel.ha_base_url, token=panel.ha_token)
    now_dt = datetime.now(timezone.utc)
    
    points = await ws.fetch_hourly_energy_kwh_from_stats(
        panel.entity_id, 
        days=days, 
        now=now_dt
    )
    
    if not points:
        return pd.DataFrame()

    df = pd.DataFrame(points)
    
    # Zoek naar de juiste kolom in de HA stats
    target_col = None
    for c in ["sum", "state", "mean"]:
        if c in df.columns and df[c].notna().any():
            target_col = c
            break
    
    if target_col is None:
        return pd.DataFrame()

    # Tijd converteren en indexeren
    df["time"] = pd.to_datetime(df["start"], unit='ms', utc=True) if df["start"].dtype != object else pd.to_datetime(df["start"], utc=True)
    df["time"] = df["time"].dt.floor("h")
    
    df = df.set_index("time").sort_index()
    df = df[~df.index.duplicated(keep='first')]
    
    # Bereken verschil per uur (kWh per uur)
    series = df[target_col].astype(float)
    hourly_diff = series.diff().clip(lower=0)
    
    result_df = (hourly_diff * panel.scale_to_kwh).to_frame(name="kwh").dropna()
    return result_df

async def perform_prediction(panel_id: str, days: int):
    """Voert de live voorspelling uit (gebruikt altijd het hoofdmodel)."""
    panel = repo.get(panel_id)
    trained = ms.load(panel_id)
    
    if not trained:
        return []

    meteo_fc = meteo.fetch_hourly_forecast(
        latitude=panel.latitude, 
        longitude=panel.longitude,
        days=days, 
        tilt_deg=panel.tilt_deg, 
        azimuth_deg=panel.azimuth_deg
    )
    
    df = prepare_features_for_model(meteo_fc, panel)
    pred = ms.predict(trained, df)
    
    return [{"time": t.isoformat(), "kwh": float(y)} for t, y in zip(df["time"], pred)]

# --- GLOBAL CONFIGURATION ---

@app.get("/api/config")
def get_global_config():
    panels = repo.list()
    if panels:
        p = panels[0]
        return {
            "ha_base_url": p.ha_base_url,
            "ha_token": p.ha_token,
            "latitude": p.latitude,
            "longitude": p.longitude,
            "evcc_url": "" 
        }
    return {"ha_base_url": "", "ha_token": "", "latitude": 52.3, "longitude": 4.9, "evcc_url": ""}

@app.post("/api/config/save")
async def save_global_config(config: GlobalConfig):
    try:
        panels = repo.list()
        for p in panels:
            p.ha_base_url = config.ha_base_url
            p.ha_token = config.ha_token
            p.latitude = config.latitude
            p.longitude = config.longitude
            repo.upsert(p)
        return {"ok": True, "message": "Instellingen opgeslagen voor alle panelen."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- PANEL MANAGEMENT ---

@app.get("/api/panels")
def list_panels():
    panels = repo.list()
    result = []
    for p in panels:
        p_dict = p.__dict__.copy()
        
        # We kijken eerst of er iets in 'friendly_name' staat
        # Daarna pas in 'name', en anders gebruiken we de panel_id
        display_name = p_dict.get('friendly_name') or p_dict.get('name')
        
        # Fallback als beide leeg zijn of de standaardwaarde hebben
        if not display_name or str(display_name).strip() in ["", "Onbekend paneel", "None"]:
            eid = p_dict.get('entity_id', '')
            if "inverter_" in eid:
                display_name = eid.split("inverter_")[1].split("_")[0]
            else:
                display_name = p_dict.get('panel_id')

        # We zorgen dat de UI altijd 'name' gebruikt als weergaveveld
        p_dict['name'] = display_name
        result.append(p_dict)
        
    return result

@app.post("/api/panels")
def add_panel(panel: PanelConfig):
    repo.upsert(panel)
    return {"ok": True}

@app.delete("/api/panels/{panel_id}")
def delete_panel(panel_id: str):
    repo.delete(panel_id)
    return {"ok": True}

# --- TRAINING ---

@app.post("/api/panels/{panel_id}/train")
async def train_panel(panel_id: str, days: int = 30):
    """Traint zowel LightGBM (standaard) als XGBoost (uitdager)."""
    try:
        panel = repo.get(panel_id)
        
        # Haal data op
        energy_df = await _fetch_panel_kwh_stats(panel, days=days)
        meteo_df = meteo.fetch_history_days(
            latitude=panel.latitude, 
            longitude=panel.longitude,
            days=days, 
            tilt_deg=panel.tilt_deg, 
            azimuth_deg=panel.azimuth_deg,
        )
        
        # Join data
        energy_df.index = pd.to_datetime(energy_df.index, utc=True).floor("h")
        meteo_df.index = pd.to_datetime(meteo_df.index, utc=True).floor("h")
        train_df = energy_df.join(meteo_df, how="inner").dropna()
        
        if len(train_df) < 24:
            raise HTTPException(status_code=400, detail="Te weinig data voor training.")

        train_df = prepare_features_for_model(train_df, panel)
        
        # 1. Train LightGBM (Hoofdmodel)
        print(f"DEBUG: Start LightGBM training voor {panel_id}")
        metrics_lgb = ms.train(panel_id, train_df, model_type="lightgbm")
        
        # 2. Train XGBoost (DIT MOET WERKEN, ANDERS ERROR)
        print(f"DEBUG: Start XGBoost training voor {panel_id}")
        ms.train(f"{panel_id}_xgb", train_df, model_type="xgboost")
        print(f"DEBUG: Alle modellen getraind voor {panel_id}")

        return {"ok": True, "metrics": metrics_lgb}
    except Exception as e:
        print(f"DEBUG TRAINING FOUT: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/train/all")
async def train_all_panels(days: int = 30):
    panels = repo.list()
    results = []
    for p in panels:
        try:
            await train_panel(p.panel_id, days)
            results.append({"panel_id": p.panel_id, "status": "ok"})
        except Exception as e:
            results.append({"panel_id": p.panel_id, "status": f"error: {str(e)}"})
    return {"trained_count": len(panels), "details": results}

# --- PREDICTION & EVALUATION ---

@app.get("/api/panels/{panel_id}/predict")
async def get_panel_prediction(panel_id: str, days: int = 7):
    return await perform_prediction(panel_id, days)

@app.get("/api/panels/{panel_id}/evaluate")
async def evaluate_panel(panel_id: str, days: int = 7):
    """Geeft vergelijking tussen Werkelijk, LightGBM en XGBoost."""
    try:
        panel = repo.get(panel_id)
        actual_df = await _fetch_panel_kwh_stats(panel, days=days)
        meteo_df = meteo.fetch_history_days(
            panel.latitude, panel.longitude, days, panel.tilt_deg, panel.azimuth_deg
        )
        
        df_eval = prepare_features_for_model(meteo_df, panel)
        
        # Modellen laden
        trained_lgb = ms.load(panel_id)
        trained_xgb = ms.load(f"{panel_id}_xgb")
        
        if not trained_lgb:
            raise HTTPException(status_code=400, detail="Model niet getraind.")
            
        preds_lgb = ms.predict(trained_lgb, df_eval)
        preds_xgb = ms.predict(trained_xgb, df_eval) if trained_xgb else [0.0] * len(preds_lgb)
        
        actual_dict = actual_df["kwh"].to_dict() if not actual_df.empty else {}
        
        comparison = []
        for i, timestamp in enumerate(df_eval["time"]):
            comparison.append({
                "time": timestamp.isoformat(),
                "actual": round(float(actual_dict.get(timestamp, 0.0)), 3),
                "lgb": round(float(preds_lgb[i]), 3),
                "xgb": round(float(preds_xgb[i]), 3)
            })
        return comparison
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/predict/total")
async def get_total_prediction(days: int = 2):
    panels = repo.list()
    total_forecast = {}
    for p in panels:
        try:
            forecast = await perform_prediction(p.panel_id, days)
            for entry in forecast:
                t, k = entry["time"], entry["kwh"]
                total_forecast[t] = total_forecast.get(t, 0.0) + k
        except:
            continue
    return [{"time": t, "kwh": round(total_forecast[t], 3)} for t in sorted(total_forecast.keys())]

@app.get("/api/predict/total/daily")
async def get_total_prediction_daily(days: int = 7):
    hourly_data = await get_total_prediction(days=days)
    daily_data = {}
    for entry in hourly_data:
        day = entry["time"].split("T")[0]
        daily_data[day] = daily_data.get(day, 0.0) + entry["kwh"]
    return [{"date": d, "kwh": round(k, 2)} for d in sorted(daily_data.keys())]

# --- STATIC FILES ---

static_path = Path("/opt/pv-panel-predictor/frontend")
if static_path.exists():
    app.mount("/ui", StaticFiles(directory=str(static_path), html=True), name="ui")