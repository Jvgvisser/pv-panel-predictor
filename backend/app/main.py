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
    df = df_in.copy()
    if "time" not in df.columns:
        df["time"] = pd.to_datetime(df.index, utc=True)
    
    df["kwh_lag_24"] = 0.0 
    df["gti_lag_24"] = df["global_tilted_irradiance"].shift(24).fillna(0.0)
    
    df = add_time_features(df, "time")
    df = add_solar_position(df, panel.latitude, panel.longitude)
    return df

async def _fetch_panel_kwh_stats(panel, days: int):
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

# --- API ENDPOINTS ---

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
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/panels")
def list_panels():
    panels = repo.list()
    return [p.__dict__ for p in panels]

@app.post("/api/panels")
def add_panel(panel: PanelConfig):
    repo.upsert(panel)
    return {"ok": True}

@app.delete("/api/panels/{panel_id}")
def delete_panel(panel_id: str):
    repo.delete(panel_id)
    return {"ok": True}

@app.get("/api/panels/{panel_id}/evaluate")
async def evaluate_panel(panel_id: str, days: int = 7):
    """Eén centrale functie voor zowel individuele panelen als het systeem-totaal."""
    
    # CASE 1: HET SYSTEEM TOTAAL (Samenvoegen van alle 28 panelen)
    if panel_id == "total_all":
        all_panels = repo.list()
        combined = {}
        for p in all_panels:
            try:
                # We roepen de evaluate logica aan voor elk paneel
                data = await evaluate_panel(p.panel_id, days)
                for d in data:
                    t = d['time']
                    if t not in combined:
                        combined[t] = {"time": t, "actual": 0.0, "lgb": 0.0, "xgb": 0.0}
                    combined[t]["actual"] += d.get("actual", 0)
                    combined[t]["lgb"] += d.get("lgb", 0)
                    combined[t]["xgb"] += d.get("xgb", 0)
            except:
                continue
        return sorted(combined.values(), key=lambda x: x['time'])

    # CASE 2: INDIVIDUEEL PANEEL (De originele ML evaluatie logica)
    try:
        panel = repo.get(panel_id)
        actual_df = await _fetch_panel_kwh_stats(panel, days=days)
        meteo_df = meteo.fetch_history_days(
            panel.latitude, panel.longitude, days, panel.tilt_deg, panel.azimuth_deg
        )
        
        df_eval = prepare_features_for_model(meteo_df, panel)
        
        trained_lgb = ms.load(panel_id)
        trained_xgb = ms.load(f"{panel_id}_xgb")
        
        if not trained_lgb:
            return [] # Model nog niet klaar
            
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
        print(f"Fout bij eval van {panel_id}: {e}")
        return []

@app.post("/api/panels/{panel_id}/train")
async def train_panel(panel_id: str, days: int = 30):
    try:
        panel = repo.get(panel_id)
        energy_df = await _fetch_panel_kwh_stats(panel, days=days)
        meteo_df = meteo.fetch_history_days(panel.latitude, panel.longitude, days, panel.tilt_deg, panel.azimuth_deg)
        
        energy_df.index = pd.to_datetime(energy_df.index, utc=True).floor("h")
        meteo_df.index = pd.to_datetime(meteo_df.index, utc=True).floor("h")
        train_df = energy_df.join(meteo_df, how="inner").dropna()
        
        if len(train_df) < 24:
            raise HTTPException(status_code=400, detail="Te weinig data.")

        train_df = prepare_features_for_model(train_df, panel)
        metrics_lgb = ms.train(panel_id, train_df, model_type="lightgbm")
        ms.train(f"{panel_id}_xgb", train_df, model_type="xgboost")

        return {"ok": True, "metrics": metrics_lgb}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/train/all")
async def train_all_panels(days: int = 30):
    panels = repo.list()
    for p in panels:
        try: await train_panel(p.panel_id, days)
        except: continue
    return {"ok": True}

@app.get("/api/panels/{panel_id}/predict")
async def get_panel_prediction(panel_id: str, days: int = 7):
    return await perform_prediction(panel_id, days)

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
        except: continue
    return [{"time": t, "kwh": round(total_forecast[t], 3)} for t in sorted(total_forecast.keys())]

@app.get("/api/predict/total/daily")
async def get_total_prediction_daily(days: int = 7):
    hourly_data = await get_total_prediction(days=days)
    daily_data = {}
    for entry in hourly_data:
        day = entry["time"].split("T")[0]
        daily_data[day] = daily_data.get(day, 0.0) + entry["kwh"]
    return [{"date": d, "kwh": round(k, 2)} for d in sorted(daily_data.keys())]

static_path = Path("/opt/pv-panel-predictor/frontend")
if static_path.exists():
    app.mount("/ui", StaticFiles(directory=str(static_path), html=True), name="ui")