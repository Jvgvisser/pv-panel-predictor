from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from backend.app.models.panel import PanelConfig
from backend.app.storage.panels_repo import PanelsRepo
from backend.app.services.ha_stats_ws import HAStatsWSClient
from backend.app.services.open_meteo_client import OpenMeteoClient
from backend.app.services.ml import PanelModelService, add_time_features

app = FastAPI(title="PV Panel Predictor")

repo = PanelsRepo()
meteo = OpenMeteoClient()
ms = PanelModelService()

async def _fetch_panel_kwh_stats(panel, days: int):
    """Fetch hourly kWh using HA long-term statistics via websocket."""
    ws = HAStatsWSClient(base_url=panel.ha_base_url, token=panel.ha_token)
    now_dt = datetime.now(timezone.utc)
    
    print(f"DEBUG: Vraag data op bij Hass voor {panel.entity_id} (Dagen: {days})")
    
    points = await ws.fetch_hourly_energy_kwh_from_stats(
        panel.entity_id, 
        days=days, 
        now=now_dt
    )
    
    if not points or len(points) < 2:
        print(f"⚠️ Hass gaf te weinig punten terug: {len(points) if points else 0}")
        return pd.DataFrame()

    df = pd.DataFrame(points)
    
    # Check welke kolom HA ons geeft (meestal 'sum' bij LTS)
    # We proberen 'sum', dan 'state', dan 'mean'
    target_col = None
    for c in ["sum", "state", "mean"]:
        if c in df.columns and df[c].notna().any():
            target_col = c
            break
    
    if target_col is None:
        print(f"❌ Geen bruikbare data-kolom gevonden. Beschikbaar: {df.columns.tolist()}")
        return pd.DataFrame()
    
    print(f"✅ Gebruik kolom '{target_col}' voor berekening.")

    # Tijd goedzetten
    df["time"] = pd.to_datetime(df["start"], utc=True).dt.floor("h")
    df = df.set_index("time").sort_index()
    df = df[~df.index.duplicated(keep='first')]
    
    # Bereken het verschil tussen de uren (omdat het een teller is)
    series = df[target_col].astype(float)
    hourly_diff = series.diff().clip(lower=0)
    
    # Schaling: we weten dat de sensor Wh is, we willen kWh
    # 1447181 Wh -> diff is bijv 500 Wh -> * 0.001 -> 0.5 kWh
    result_df = (hourly_diff * 0.001).to_frame(name="kwh").dropna()
    
    print(f"📊 Na verwerking: {len(result_df)} rijen aan uurverbruik.")
    return result_df

@app.post("/api/panels/{panel_id}/train")
async def train_panel(panel_id: str, days: int = 30):
    try:
        print(f"🚀 Start training voor {panel_id} over {days} dagen...")
        panel = repo.get(panel_id)

        # 1. HAAL DATA OP (NU MET AWAIT)
        energy_df = await _fetch_panel_kwh_stats(panel, days=days)
        meteo_df = meteo.fetch_history_days(
            latitude=panel.latitude, longitude=panel.longitude,
            days=days, tilt_deg=panel.tilt_deg, azimuth_deg=panel.azimuth_deg,
        )

        # 2. DATA COMBINEREN
        print(f"🔗 Data combineren voor {len(energy_df)} Hass rijen en {len(meteo_df)} weer rijen...")
        
        energy_df.index = pd.to_datetime(energy_df.index, utc=True).floor("h")
        meteo_df.index = pd.to_datetime(meteo_df.index, utc=True).floor("h")
        
        energy_df = energy_df[~energy_df.index.duplicated(keep='first')]
        meteo_df = meteo_df[~meteo_df.index.duplicated(keep='first')]

        train_df = energy_df.join(meteo_df, how="inner").dropna()
        
        print(f"📈 Match gevonden voor {len(train_df)} uren.")

        if len(train_df) < 24:
            print(f"❌ DEBUG: Hass range: {energy_df.index.min()} tot {energy_df.index.max()}")
            print(f"❌ DEBUG: Weer range: {meteo_df.index.min()} tot {meteo_df.index.max()}")
            raise HTTPException(status_code=400, detail=f"Te weinig overlap. Hass: {len(energy_df)}, Match: {len(train_df)}")

        # 3. FEATURES & TRAINING
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
    # (Deze blijft ongewijzigd, OpenMeteo is sync)
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

# Static files
static_path = Path("/opt/pv-panel-predictor/frontend")
if static_path.exists():
    app.mount("/ui", StaticFiles(directory=str(static_path), html=True), name="ui")