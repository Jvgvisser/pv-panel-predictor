# 🌞 PV Panel Predictor

**Local per-panel solar production forecasting** using Home Assistant
history, Open-Meteo weather data and machine learning.

Inspired by the ideas of EpexPredictor but focused on **solar generation
instead of energy prices**.

------------------------------------------------------------------------

## 🚀 Features

-   Per-panel forecasting for Enphase micro-inverters\
-   Support for large installations (e.g. 28 panels)\
-   Open-Meteo integration with:
    -   Global Tilted Irradiance (GTI)
    -   cloud cover
    -   temperature\
-   Correct handling of:
    -   tilt angle\
    -   azimuth orientation\
-   Automatic Wh → kWh conversion\
-   Home Assistant friendly API (`/predict_ha`)\
-   Web UI for configuration and bulk actions\
-   Fully local -- no cloud account required

------------------------------------------------------------------------

## 🧠 Concept

1.  **Home Assistant** provides lifetime energy sensors per inverter\
2.  The application converts this to hourly deltas (kWh/h)\
3.  Open-Meteo supplies historical and forecast weather\
4.  A machine learning model is trained per panel\
5.  Output: reliable 7-day hourly forecast

------------------------------------------------------------------------

## 🏗 Architecture & Data Flow

    ┌──────────────────────┐
    │  Home Assistant      │
    │  - lifetime sensors  │
    └─────────┬────────────┘
              │ history (Wh)
              ▼
    ┌──────────────────────┐
    │ PV Panel Predictor   │
    │                      │
    │ 1) History loader    │
    │ 2) Delta → kWh/h     │
    │ 3) Open‑Meteo GTI    │◀────── Weather API
    │ 4) ML training       │
    │ 5) 7‑day forecast    │
    └─────────┬────────────┘
              │ REST API
              ▼
    ┌──────────────────────┐
    │  Home Assistant      │
    │  Forecast sensors    │
    └──────────────────────┘

**Per panel workflow**

-   Each inverter is treated as an independent model\
-   Features include:
    -   global tilted irradiance\
    -   cloud cover\
    -   temperature\
    -   time features (hour / month)\
-   Training uses up to 365 days of history\
-   Prediction horizon: 168 hours

------------------------------------------------------------------------

## ⚙️ Installation

``` bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt

PYTHONPATH=. python -m uvicorn backend.app.main:app --reload --port 8000
```

Open UI:\
http://127.0.0.1:8000/ui/

------------------------------------------------------------------------

## ➕ Panel configuration

Each panel requires:

-   Home Assistant `entity_id`\
-   `tilt_deg`\
-   `azimuth_deg`
    -   0 = South\
    -   -90 = East\
    -   +90 = West\
-   latitude / longitude\
-   scale_to_kwh (Wh → kWh = 0.001)

Bulk JSON import is supported.

------------------------------------------------------------------------

## 🔌 API Examples

### Train panel

``` bash
POST /api/panels/p01/train?days=365
```

### Forecast

``` bash
GET /api/panels/p01/predict?days=7
```

### Home Assistant format

``` bash
GET /api/panels/p01/predict_ha?days=7
```

------------------------------------------------------------------------

## 🧩 Roadmap

-   Caching of Open-Meteo per roof plane\
-   UI graphs (history vs forecast)\
-   Total installation forecast\
-   LightGBM model option\
-   Docker deployment

------------------------------------------------------------------------

## 🙌 Credits

-   Weather: https://open-meteo.com\
-   Inspired by: https://github.com/b3nn0/EpexPredictor\
-   Designed for Enphase IQ7+ and Home Assistant
