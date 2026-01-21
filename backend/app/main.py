from pathlib import Path
from datetime import datetime, timedelta, timezone

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from backend.app.models.panel import PanelConfig
from backend.app.storage.panels_repo import PanelsRepo
from backend.app.services.ha_client import HAClient
from backend.app.services.open_meteo_client import OpenMeteoClient
from backend.app.services.ha_stats_ws import HAStatsWSClient
from backend.app.services.ml import PanelModelService, build_training_frame, build_training_frame_idxjoin, add_time_features

app = FastAPI(title="PV Panel Predictor")

repo = PanelsRepo()
meteo = OpenMeteoClient()
ms = PanelModelService()

def _fetch_panel_kwh_stats(panel, days: int):
    """
    Fetch hourly kWh using HA long-term statistics via websocket.
    """
    ws = HAStatsWSClient(base_url=panel.ha_base_url, token=panel.ha_token)
    points = ws.fetch_hourly_energy_kwh_from_stats(panel.entity_id, days=days, now=datetime.now(timezone.utc))
    if not points:
        return pd.DataFrame({"kwh": []}, index=pd.DatetimeIndex([], tz="Europe/Amsterdam"))

    df = pd.DataFrame(points)
    # Kies 'sum' indien beschikbaar, anders 'state'
    col = "sum" if "sum" in df.columns else "state"
    df["time"] = pd.to_datetime(df["start"], utc=True).dt.tz_convert("Europe/Amsterdam")
    df = df.set_index("time").sort_index()
    
    series = df[col].astype(float)
    # Bereken het verschil tussen uren (kWh productie per uur)
    hourly_kwh = series.diff().clip(lower=0)
    return hourly_kwh.to_frame(name="kwh")

@app.post("/api/panels/{panel_id}/train")
def train_panel(panel_id: str, days: int = 30):
    try:
        panel = repo.get(panel_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # 1. Haal energie data op
    energy_df = _fetch_panel_kwh_stats(panel, days=days)
    
    # 2. Haal historische weerdata op
    meteo_df = meteo.fetch_history_days(
        latitude=panel.latitude,
        longitude=panel.longitude,
        days=days,
        tilt_deg=panel.tilt_deg,
        azimuth_deg=panel.azimuth_deg,
    )

    # 3. Combineer data en train model
    train_df = build_training_frame(energy_df, meteo_df)
    metrics = ms.train(panel_id, train_df)
    
    return {"ok": True, "metrics": metrics}

@app.get("/api/panels/{panel_id}/predict")
def predict_panel(panel_id: str, days: int = 7):
    try:
        panel = repo.get(panel_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    try:
        trained = ms.load(panel_id)
    except Exception:
        raise HTTPException(status_code=404, detail="Model not trained yet")

    meteo_fc = meteo.fetch_hourly_forecast(
        latitude=panel.latitude,
        longitude=panel.longitude,
        days=days,
        tilt_deg=panel.tilt_deg,
        azimuth_deg=panel.azimuth_deg,
        timezone="Europe/Amsterdam",
    )

    if meteo_fc.empty:
        raise HTTPException(status_code=502, detail="Open-Meteo forecast returned empty data")

    df = meteo_fc.copy()
    df["kwh_lag_24"] = 0.0
    df["gti_lag_24"] = df["global_tilted_irradiance"].shift(24).fillna(0.0)
    
    # Fix voor de KeyError: zorg dat 'time' kolom bestaat vanuit de index
    if "time" not in df.columns:
        df["time"] = df.index
        
    df = add_time_features(df, "time")

    pred = ms.predict(trained, df)

    out = []
    times = pd.to_datetime(df["time"])
    
    if times.dt.tz is None:
        times = times.dt.tz_localize("Europe/Amsterdam", nonexistent="shift_forward", ambiguous="NaT")

    for t, y in zip(times, pred):
        out.append({"time": t.isoformat(), "kwh": float(y)})

    return {"ok": True, "panel_id": panel_id, "hours": len(out), "forecast": out}

@app.get("/api/panels")
def list_panels():
    return repo.list()

@app.post("/api/panels")
def add_panel(panel: PanelConfig):
    repo.upsert(panel)
    return {"ok": True}

@app.delete("/api/panels/{panel_id}")
def delete_panel(panel_id: str):
    repo.delete(panel_id)
    return {"ok": True}

# Gebruik het exacte pad waar het bestand nu staat
static_path = Path("/opt/pv-panel-predictor/frontend")

if (static_path / "index.html").exists():
    app.mount("/", StaticFiles(directory=str(static_path), html=True), name="static")