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

def _fetch_panel_kwh_stats(panel, days: int):
    """
    Fetch hourly kWh using HA long-term statistics via websocket.

    For energy sensors with state_class total_increasing, HA stats returns 'sum' in kWh.
    We convert to hourly deltas.
    """
    import pandas as pd
    import numpy as np
    from datetime import datetime, timezone

    ws = HAStatsWSClient(base_url=panel.ha_base_url, token=panel.ha_token)
    points = ws.fetch_hourly_energy_kwh_from_stats(panel.entity_id, days=days, now=datetime.now(timezone.utc))
    if not points:
        return pd.DataFrame({"kwh": []}, index=pd.DatetimeIndex([], tz="Europe/Amsterdam"))

    # choose 'sum' if present, else 'state', else 'mean'
    def pick(p):
        for k in ("sum", "state", "mean"):
            if k in p and p[k] is not None:
                return float(p[k])
        return None

    rows = []
    for pt in points:
        start = pt.get("start") or pt.get("start_time") or pt.get("time")
        val = pick(pt)
        if start is None or val is None:
            continue
        rows.append((pd.to_datetime(start, utc=True).tz_convert("Europe/Amsterdam"), val))

    if not rows:
        return pd.DataFrame({"kwh": []}, index=pd.DatetimeIndex([], tz="Europe/Amsterdam"))

    rows.sort(key=lambda x: x[0])
    s = pd.Series([v for _, v in rows], index=pd.DatetimeIndex([t for t, _ in rows]))
    # convert total kWh -> hourly kWh delta
    kwh = s.diff().fillna(0.0).clip(lower=0.0)

    # some sensors may be Wh in stats (rare). If your totals look 1000x too big/small we adjust later.
    return pd.DataFrame({"kwh": kwh})

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

    df = build_training_frame_idxjoin(energy, meteo_hist)

    if len(df) < 72:
        raise HTTPException(status_code=400, detail=f"Not enough joined rows to train: {len(df)}")

    metrics = ms.train(panel_id, df)
    return {"ok": True, "panel_id": panel_id, "rows": int(len(df)), "metrics": metrics}


@app.get("/api/evcc/solar")
def evcc_solar(days: int = 2, interval: str = "1h"):
    """
    evcc custom solar forecast endpoint.
    Returns JSON array: [{start,end,value}] where value is PV power in Watt.
    """
    # reuse existing logic: predict all panels and sum (existing endpoint handler likely exists)
    # If you already have a function for total forecast, call that instead.
    results = predict_all(days=days)  # expects {"ok":True,"days":...,"results":[...]}
    # Sum per-hour kWh
    totals = {}
    for r in results.get("results", []):
        if not r.get("ok"):
            continue
        fc = r["result"].get("forecast", [])
        for item in fc:
            t = item["time"]
            totals[t] = totals.get(t, 0.0) + float(item.get("kwh", 0.0))

    # Convert to evcc list
    # interval supported: "1h" or "30m"
    if interval not in ("1h", "30m"):
        interval = "1h"

    out = []
    # Sort times
    times = sorted(totals.keys())
    # Limit to requested horizon
    step_hours = 1 if interval == "1h" else 0.5
    max_points = int(days * 24 / step_hours)
    times = times[:max_points]
    for t in times:
        kwh = totals[t]
        # Parse ISO with timezone; evcc examples use Z, but offsets are fine too.
        # We'll output UTC (Z) for compatibility.
        # Convert "2026-01-20T09:00:00+01:00" -> UTC Z
        import datetime as _dt
        dt = _dt.datetime.fromisoformat(t.replace("Z", "+00:00"))
        dt_utc = dt.astimezone(_dt.timezone.utc)

        if interval == "1h":
            end_utc = dt_utc + _dt.timedelta(hours=1)
            watts = kwh * 1000.0
        else:  # 30m
            end_utc = dt_utc + _dt.timedelta(minutes=30)
            watts = kwh * 2000.0

        out.append({
            "start": dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": end_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "value": int(round(watts)),
        })

    return out


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
