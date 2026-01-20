from pathlib import Path
from datetime import datetime, timedelta, timezone

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from backend.app.models.panel import PanelConfig
from backend.app.storage.panels_repo import PanelsRepo
from backend.app.services.ha_client import HAClient
from backend.app.services.open_meteo_client import OpenMeteoClient
from backend.app.services.ml import PanelModelService, build_training_frame, add_time_features

app = FastAPI(title="PV Panel Predictor")

repo = PanelsRepo()
ms = PanelModelService()
meteo = OpenMeteoClient()

REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = REPO_ROOT / "frontend"

app.mount("/ui", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="ui")


@app.get("/health")
def health():
    return {"ok": True, "frontend_dir": str(FRONTEND_DIR)}


@app.get("/api/panels")
def list_panels():
    return {"panels": [p.model_dump(exclude={"ha_token"}) for p in repo.list()]}


@app.post("/api/panels")
def upsert_panel(panel: PanelConfig):
    repo.upsert(panel)
    return {"ok": True}


@app.delete("/api/panels/{panel_id}")
def delete_panel(panel_id: str):
    try:
        repo.delete(panel_id)
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/panels/{panel_id}/history")
def panel_history(panel_id: str, days: int = 7):
    try:
        panel = repo.get(panel_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    client = HAClient(panel.ha_base_url, panel.ha_token)

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    try:
        hist = client.fetch_history_period(panel.entity_id, start=start, end=end)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"HA history fetch failed: {e}")

    df = client.lifetime_to_hourly_kwh(
        hist, tz="Europe/Amsterdam", scale_to_kwh=panel.scale_to_kwh
    )

    return {
        "panel_id": panel_id,
        "entity_id": panel.entity_id,
        "points": int(len(df)),
        "sample_tail": df.tail(48).to_dict(orient="records"),
    }


@app.post("/api/panels/{panel_id}/train")
def train_panel(panel_id: str, days: int = 365):
    try:
        panel = repo.get(panel_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # 1) energy history (kWh/hour)
    client = HAClient(panel.ha_base_url, panel.ha_token)
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=days)
    hist = client.fetch_history_period(panel.entity_id, start=start_dt, end=end_dt)
    energy = client.lifetime_to_hourly_kwh(
        hist, tz="Europe/Amsterdam", scale_to_kwh=panel.scale_to_kwh
    )

    # 2) weather archive
    today = datetime.now().date()
    start_day = today - timedelta(days=days)
    meteo_hist = meteo.fetch_hourly_archive(
        latitude=panel.latitude,
        longitude=panel.longitude,
        start_date=start_day,
        end_date=today,
        tilt_deg=panel.tilt_deg,
        azimuth_deg=panel.azimuth_deg,
        timezone="Europe/Amsterdam",
    )

    if meteo_hist.empty:
        raise HTTPException(status_code=502, detail="Open-Meteo archive returned empty data")

    df = build_training_frame(energy, meteo_hist)

    if len(df) < 72:
        raise HTTPException(status_code=400, detail=f"Not enough joined rows to train: {len(df)}")

    metrics = ms.train(panel_id, df)
    return {"ok": True, "panel_id": panel_id, "rows": int(len(df)), "metrics": metrics}


@app.get("/api/panels/{panel_id}/predict")
def predict_panel(panel_id: str, days: int = 7):
    try:
        panel = repo.get(panel_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    trained = ms.load(panel_id)

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
    df = add_time_features(df, "time")

    pred = ms.predict(trained, df)

    out = []
    times = pd.to_datetime(df["time"])
    # Treat Open-Meteo times as Europe/Amsterdam local time, then output with offset
    times = times.dt.tz_localize("Europe/Amsterdam", nonexistent="shift_forward", ambiguous="NaT")

    for t, y in zip(times, pred):
        out.append({"time": t.isoformat(), "kwh": float(y)})

    return {"ok": True, "panel_id": panel_id, "hours": len(out), "forecast": out}


@app.get("/api/panels/{panel_id}/predict_ha")
def predict_panel_ha(panel_id: str, days: int = 7):
    """Compact JSON for Home Assistant REST sensor."""
    res = predict_panel(panel_id, days=days)  # reuse
    compact = [{"t": x["time"], "v": x["kwh"]} for x in res["forecast"]]
    return {
        "panel_id": panel_id,
        "unit": "kWh",
        "hours": res["hours"],
        "forecast": compact,
    }


@app.post("/api/train_all")
def train_all(days: int = 365):
    results = []
    for panel in repo.list():
        try:
            r = train_panel(panel.panel_id, days=days)
            results.append({"panel_id": panel.panel_id, "ok": True, "result": r})
        except Exception as e:
            results.append({"panel_id": panel.panel_id, "ok": False, "error": str(e)})
    return {"ok": True, "days": days, "results": results}


@app.get("/api/predict_all")
def predict_all(days: int = 7):
    results = []
    for panel in repo.list():
        try:
            r = predict_panel(panel.panel_id, days=days)
            results.append({"panel_id": panel.panel_id, "ok": True, "result": r})
        except Exception as e:
            results.append({"panel_id": panel.panel_id, "ok": False, "error": str(e)})
    return {"ok": True, "days": days, "results": results}
