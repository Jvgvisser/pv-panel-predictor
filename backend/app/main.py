from pathlib import Path
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from backend.app.models.panel import PanelConfig
from backend.app.storage.panels_repo import PanelsRepo
from backend.app.services.ha_client import HAClient

app = FastAPI(title="PV Panel Predictor")

repo = PanelsRepo()

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
        hist,
        tz="Europe/Amsterdam",
        scale_to_kwh=panel.scale_to_kwh,
    )

    return {
        "panel_id": panel_id,
        "entity_id": panel.entity_id,
        "points": int(len(df)),
        "sample_tail": df.tail(48).to_dict(orient="records"),
    }


@app.post("/api/panels/{panel_id}/train")
def train_panel(panel_id: str):
    try:
        repo.get(panel_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"ok": True, "message": "Training stub"}


@app.get("/api/panels/{panel_id}/predict")
def predict_panel(panel_id: str, days: int = 7):
    try:
        repo.get(panel_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"ok": True, "message": f"Predict stub for {days} days"}
