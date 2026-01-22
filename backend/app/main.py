from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import json
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List

from backend.app.models.panel import PanelConfig
from backend.app.storage.panels_repo import PanelsRepo
from backend.app.services.ha_stats_ws import HAStatsWSClient
from backend.app.services.open_meteo_client import OpenMeteoClient
from backend.app.services.ml import PanelModelService, add_time_features

app = FastAPI(title="PV Panel Predictor")

repo = PanelsRepo()
meteo = OpenMeteoClient()
ms = PanelModelService()

# --- MODELS FOR CONFIGURATION ---

class GlobalConfig(BaseModel):
    ha_base_url: str
    ha_token: str
    evcc_url: str = ""

# --- HELPER FUNCTIONS ---

async def _fetch_panel_kwh_stats(panel, days: int):
    """Fetch hourly kWh using HA long-term statistics via websocket."""
    ws = HAStatsWSClient(base_url=panel.ha_base_url, token=panel.ha_token)
    now_dt = datetime.now(timezone.utc)
    
    print(f"🚀 DEBUG: Fetching data for {panel.entity_id} ({days} days)")
    
    points = await ws.fetch_hourly_energy_kwh_from_stats(
        panel.entity_id, 
        days=days, 
        now=now_dt
    )
    
    if not points:
        print("❌ DEBUG: No points received from WebSocket.")
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

# --- GLOBAL CONFIGURATION ENDPOINTS ---

@app.get("/api/config")
def get_global_config():
    """Fetches HA settings from the first panel as reference."""
    panels = repo.list()
    if panels:
        p = panels[0]
        return {
            "ha_base_url": p.ha_base_url,
            "ha_token": p.ha_token,
            "evcc_url": "" 
        }
    return {"ha_base_url": "", "ha_token": "", "evcc_url": ""}

@app.post("/api/config/save")
async def save_global_config(config: GlobalConfig):
    """Update HA credentials for ALL panels."""
    try:
        panels = repo.list()
        for p in panels:
            p.ha_base_url = config.ha_base_url
            p.ha_token = config.ha_token
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
    try:
        repo.upsert(panel)
        return {"ok": True, "panel_id": panel.panel_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/panels/{panel_id}")
def delete_panel(panel_id: str):
    repo.delete(panel_id)
    return {"ok": True}

# --- TRAINING & PREDICTION ---

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
            raise HTTPException(status_code=400, detail="Insufficient data overlap.")

        train_df = add_time_features(train_df)
        metrics = ms.train(panel_id, train_df)
        return {"ok": True, "metrics": metrics, "rows": len(train_df)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/train/all")
async def train_all_panels(days: int = 30):
    """Hulpmiddel om alle panelen in één keer te trainen."""
    panels = repo.list()
    results = []
    for p in panels:
        try:
            await train_panel(p.panel_id, days=days)
            results.append({"panel_id": p.panel_id, "status": "ok"})
        except Exception as e:
            results.append({"panel_id": p.panel_id, "status": f"error: {str(e)}"})
    return {"trained_count": len(panels), "details": results}

@app.get("/api/panels/{panel_id}/predict")
async def predict_panel(panel_id: str, days: int = 7):
    try:
        panel = repo.get(panel_id)
        trained = ms.load(panel_id)
        meteo_fc = meteo.fetch_hourly_forecast(
            latitude=panel.latitude, longitude=panel.longitude,
            days=days, tilt_deg=panel.tilt_deg, azimuth_deg=panel.azimuth_deg
        )
        df = meteo_fc.copy()
        df["time"] = pd.to_datetime(df.index, utc=True)
        df["kwh_lag_24"] = 0.0 
        df["gti_lag_24"] = df["global_tilted_irradiance"].shift(24).fillna(0.0)
        df = add_time_features(df, "time")
        pred = ms.predict(trained, df)
        out = [{"time": t.isoformat(), "kwh": float(y)} for t, y in zip(df["time"], pred)]
        return {"ok": True, "panel_id": panel_id, "forecast": out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/predict/total")
async def get_total_prediction(days: int = 2):
    """Haalt de voorspelling op voor ALLE panelen en telt ze bij elkaar op."""
    panels = repo.list()
    total_forecast = {}

    for p in panels:
        try:
            # We roepen de bestaande predict functie aan
            res = await predict_panel(p.panel_id, days=days)
            for entry in res["forecast"]:
                t = entry["time"]
                k = entry["kwh"]
                total_forecast[t] = total_forecast.get(t, 0.0) + k
        except Exception as e:
            print(f"Skipping {p.panel_id}: {e}")

    sorted_forecast = [
        {"time": t, "kwh": round(total_forecast[t], 3)} 
        for t in sorted(total_forecast.keys())
    ]
    return {"ok": True, "total_forecast": sorted_forecast}

# --- STATIC FILES ---

static_path = Path("/opt/pv-panel-predictor/frontend")
if static_path.exists():
    app.mount("/ui", StaticFiles(directory=str(static_path), html=True), name="ui")