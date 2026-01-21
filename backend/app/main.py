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
from backend.app.services.ml import PanelModelService, add_time_features

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
        print(f"🚀 Start training voor {panel_id} over {days} dagen...")
        panel = repo.get(panel_id)

        # --- STAP 1: HASS DATA ---
        energy_df = _fetch_panel_kwh_stats(panel, days=days)
        
        # --- STAP 2: WEER DATA ---
        meteo_df = meteo.fetch_history_days(
            latitude=panel.latitude, longitude=panel.longitude,
            days=days, tilt_deg=panel.tilt_deg, azimuth_deg=panel.azimuth_deg,
        )

        # --- STAP 3: HIER KOMT HET NIEUWE BLOK (Copy/Paste dit) ---
        print(f"🔗 Data combineren...")
        
        energy_df.index = pd.to_datetime(energy_df.index, utc=True)
        meteo_df.index = pd.to_datetime(meteo_df.index, utc=True)
        
        energy_df = energy_df[~energy_df.index.duplicated(keep='first')]
        meteo_df = meteo_df[~meteo_df.index.duplicated(keep='first')]

        if "kwh" not in energy_df.columns and "value" in energy_df.columns:
            energy_df = energy_df.rename(columns={"value": "kwh"})

        train_df = energy_df.join(meteo_df, how="inner").dropna()
        
        print(f"📈 Match gevonden voor {len(train_df)} uren.")

        if len(train_df) < 10:
            print(f"❌ Te weinig data! Hass bereik: {energy_df.index.min()} tot {energy_df.index.max()}")
            print(f"❌ Weer bereik: {meteo_df.index.min()} tot {meteo_df.index.max()}")
            raise HTTPException(status_code=400, detail=f"Geen overlap in data. Hass heeft {len(energy_df)} rijen.")
        
        # --- EINDE NIEUWE BLOK ---

        from backend.app.services.ml import add_time_features
        train_df = add_time_features(train_df)

        print("🤖 LightGBM model trainen...")
        metrics = ms.train(panel_id, train_df)
        
        return {"ok": True, "metrics": metrics, "rows": len(train_df)}

    except Exception as e:
        import traceback
        print(f"‼️ CRASH:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/panels/{panel_id}/predict")
def predict_panel(panel_id: str, days: int = 7):
    try:
        panel = repo.get(panel_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Paneel niet gevonden")

    trained = ms.load(panel_id)

    # Haal forecast op
    meteo_fc = meteo.fetch_hourly_forecast(
        latitude=panel.latitude, longitude=panel.longitude,
        days=days, tilt_deg=panel.tilt_deg, azimuth_deg=panel.azimuth_deg,
        timezone="Europe/Amsterdam"
    )

    df = meteo_fc.copy()
    
    # Maak de features aan die LightGBM verwacht
    df["kwh_lag_24"] = 0.0  # In live predictie hebben we geen echte lag van gisteren
    df["gti_lag_24"] = df["global_tilted_irradiance"].shift(24).fillna(0.0)
    
    # Fix voor de tijd kolom
    if "time" not in df.columns:
        df["time"] = df.index

    df = add_time_features(df, "time")
    
    # Doe de voorspelling met het LightGBM model
    pred = ms.predict(trained, df)

    out = []
    times = pd.to_datetime(df["time"]).dt.tz_localize("Europe/Amsterdam", nonexistent="shift_forward", ambiguous="NaT")
    for t, y in zip(times, pred):
        out.append({"time": t.isoformat(), "kwh": float(y)})

    return {"ok": True, "panel_id": panel_id, "forecast": out}

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

# --- Helemaal onderaan in backend/app/main.py ---

# We definiëren het pad naar de map waar index.html nu staat
static_path = Path("/opt/pv-panel-predictor/frontend")

if static_path.exists():
    # De 'html=True' zorgt ervoor dat /ui/ automatisch index.html zoekt
    app.mount("/ui", StaticFiles(directory=str(static_path), html=True), name="ui")
    print(f"✅ UI gemount op http://LXC-IP:8000/ui/ (bron: {static_path})")
else:
    print(f"⚠️ Waarschuwing: Kan frontend map niet vinden op {static_path}")