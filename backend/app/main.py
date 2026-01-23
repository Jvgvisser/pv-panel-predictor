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

async def _fetch_panel_kwh_stats(panel, days: int):
    """Fetch hourly kWh using HA long-term statistics via websocket."""
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
    
    target_col = None
    for c in ["sum", "state", "mean"]:
        if c in df.columns and df[c].notna().any():
            target_col = c
            break
    
    if target_col is None:
        return pd.DataFrame()

    df["time"] = pd.to_datetime(df["start"], unit='ms', utc=True) if df["start"].dtype != object else pd.to_datetime(df["start"], utc=True)
    df["time"] = df["time"].dt.floor("h")
    
    df = df.set_index("time").sort_index()
    df = df[~df.index.duplicated(keep='first')]
    
    series = df[target_col].astype(float)
    hourly_diff = series.diff().clip(lower=0)
    
    result_df = (hourly_diff * panel.scale_to_kwh).to_frame(name="kwh").dropna()
    return result_df

async def perform_prediction(panel_id: str, days: int):
    """Internal function to handle prediction logic for one panel."""
    panel = repo.get(panel_id)
    trained = ms.load(panel_id)
    
    if not trained:
        # Als er geen model is, geven we een lege lijst of error, 
        # zodat de UI weet dat er nog getraind moet worden.
        return []

    meteo_fc = meteo.fetch_hourly_forecast(
        latitude=panel.latitude, longitude=panel.longitude,
        days=days, tilt_deg=panel.tilt_deg, azimuth_deg=panel.azimuth_deg
    )
    
    df = meteo_fc.copy()
    df["time"] = pd.to_datetime(df.index, utc=True)
    
    # Voeg lags en tijd features toe
    df["kwh_lag_24"] = 0.0 
    df["gti_lag_24"] = df["global_tilted_irradiance"].shift(24).fillna(0.0)
    
    df = add_time_features(df, "time")
    df = add_solar_position(df, panel.latitude, panel.longitude)
    
    pred = ms.predict(trained, df)
    return [{"time": t.isoformat(), "kwh": float(y)} for t, y in zip(df["time"], pred)]

# --- GLOBAL CONFIGURATION ENDPOINTS ---

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
        return {"ok": True, "message": f"Global settings applied to {len(panels)} panels."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- PANEL ENDPOINTS ---

@app.get("/api/panels")
def list_panels():
    return repo.list()

@app.post("/api/panels")
def add_panel(panel: PanelConfig):
    repo.upsert(panel)
    return {"ok": True, "panel_id": panel.panel_id}

@app.delete("/api/panels/{panel_id}")
def delete_panel(panel_id: str):
    repo.delete(panel_id)
    return {"ok": True}

# --- TRAINING & PREDICTION ---

@app.get("/api/panels/{panel_id}/predict")
async def get_panel_prediction(panel_id: str, days: int = 7):
    """Route die de UI aanroept voor de grafiek per paneel (Lost 404 op)."""
    try:
        forecast = await perform_prediction(panel_id, days)
        return forecast
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/train/all")
async def train_all_panels(days: int = 30):
    panels = repo.list()
    results = []
    for p in panels:
        try:
            energy_df = await _fetch_panel_kwh_stats(p, days=days)
            meteo_df = meteo.fetch_history_days(
                latitude=p.latitude, longitude=p.longitude,
                days=days, tilt_deg=p.tilt_deg, azimuth_deg=p.azimuth_deg,
            )
            energy_df.index = pd.to_datetime(energy_df.index, utc=True).floor("h")
            meteo_df.index = pd.to_datetime(meteo_df.index, utc=True).floor("h")
            
            train_df = energy_df.join(meteo_df, how="inner").dropna()
            
            if len(train_df) >= 24:
                train_df = add_time_features(train_df)
                train_df = add_solar_position(train_df, p.latitude, p.longitude)
                ms.train(p.panel_id, train_df)
                results.append({"panel_id": p.panel_id, "status": "ok"})
            else:
                results.append({"panel_id": p.panel_id, "status": "insufficient_data"})
        except Exception as e:
            results.append({"panel_id": p.panel_id, "status": f"error: {str(e)}"})
    return {"trained_count": len(panels), "details": results}

@app.post("/api/panels/{panel_id}/train")
async def train_panel(panel_id: str, days: int = 30):
    try:
        panel = repo.get(panel_id)
        energy_df = await _fetch_panel_kwh_stats(panel, days=days)
        meteo_df = meteo.fetch_history_days(
            latitude=panel.latitude, longitude=panel.longitude,
            days=days, tilt_deg=panel.tilt_deg, azimuth_deg=panel.azimuth_deg,
        )
        energy_df.index = pd.to_datetime(energy_df.index, utc=True).floor("h")
        meteo_df.index = pd.to_datetime(meteo_df.index, utc=True).floor("h")
        train_df = energy_df.join(meteo_df, how="inner").dropna()
        
        if len(train_df) < 24:
            raise HTTPException(status_code=400, detail="Insufficient data.")

        train_df = add_time_features(train_df)
        train_df = add_solar_position(train_df, panel.latitude, panel.longitude)
        
        metrics = ms.train(panel_id, train_df)
        return {"ok": True, "metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/predict/total")
async def get_total_prediction(days: int = 2):
    """Voor evcc en uurbasis grafieken."""
    panels = repo.list()
    total_forecast = {}
    for p in panels:
        try:
            forecast = await perform_prediction(p.panel_id, days)
            for entry in forecast:
                t = entry["time"]
                k = entry["kwh"]
                total_forecast[t] = total_forecast.get(t, 0.0) + k
        except Exception:
            continue

    return [{"time": t, "kwh": round(total_forecast[t], 3)} for t in sorted(total_forecast.keys())]

@app.get("/api/predict/total/daily")
async def get_total_prediction_daily(days: int = 7):
    """Voor Home Assistant: dagtotalen."""
    hourly_data = await get_total_prediction(days=days)
    daily_data = {}
    for entry in hourly_data:
        day = entry["time"].split("T")[0]
        daily_data[day] = daily_data.get(day, 0.0) + entry["kwh"]
    
    return [{"date": d, "kwh": round(k, 2)} for d in sorted(daily_data.keys())]


@app.get("/api/panels/{panel_id}/evaluate")
async def evaluate_panel(panel_id: str, days: int = 7):
    """Vergelijkt historische voorspelling met werkelijke opbrengst."""
    try:
        panel = repo.get(panel_id)
        # 1. Haal werkelijke data op uit Home Assistant
        actual_df = await _fetch_panel_kwh_stats(panel, days=days)
        
        # 2. Haal historische weerdata op voor dezelfde periode
        meteo_df = meteo.fetch_history_days(
            latitude=panel.latitude, longitude=panel.longitude,
            days=days, tilt_deg=panel.tilt_deg, azimuth_deg=panel.azimuth_deg
        )
        
        # 3. Voer predictie uit op die historische weerdata
        trained = ms.load(panel_id)
        if not trained:
            raise HTTPException(status_code=400, detail="Model niet getraind voor dit paneel.")
            
        df_eval = meteo_df.copy()
        df_eval["time"] = pd.to_datetime(df_eval.index, utc=True)
        # We simuleren de lags die het model verwacht
        df_eval["kwh_lag_24"] = 0.0 
        df_eval["gti_lag_24"] = df_eval["global_tilted_irradiance"].shift(24).fillna(0.0)
        
        df_eval = add_time_features(df_eval, "time")
        df_eval = add_solar_position(df_eval, panel.latitude, panel.longitude)
        
        predictions = ms.predict(trained, df_eval)
        
        # 4. Combineer data voor de grafiek
        comparison = []
        # Maak van de actual_df een dictionary voor snelle lookup
        actual_dict = actual_df["kwh"].to_dict() if not actual_df.empty else {}
        
        for timestamp, pred in zip(df_eval["time"], predictions):
            actual = actual_dict.get(timestamp, 0.0)
            comparison.append({
                "time": timestamp.isoformat(),
                "actual": round(float(actual), 3),
                "predicted": round(float(pred), 3),
                "error": round(float(pred - actual), 3)
            })
            
        return comparison
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- STATIC FILES ---

static_path = Path("/opt/pv-panel-predictor/frontend")
if static_path.exists():
    app.mount("/ui", StaticFiles(directory=str(static_path), html=True), name="ui")