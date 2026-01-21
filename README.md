# PV Panel Predictor

A smart solar energy forecasting dashboard that uses Machine Learning to predict power output based on Home Assistant data and weather forecasts.

## 🌟 Features
- **Total System Overview:** Aggregated 7-day forecast for the entire solar array (28 panels).
- **Individual Analysis:** Drill down into specific panels (p01 - p28) to inspect performance.
- **Machine Learning:** Powered by Scikit-Learn (Linear Regression) for tailored predictions per panel.
- **Home Assistant Integration:** Real-time data fetching via the Home Assistant REST API.
- **Responsive Dashboard:** Clean UI built with Tailwind CSS and Chart.js.

## 📂 Project Structure
- `backend/`: FastAPI server and Machine Learning logic.
- `frontend/`: Dashboard interface (HTML5/JavaScript).
- `data/`: Local storage for `panels.json` (panel configurations and tokens). *[Excluded from Git]*
- `models/`: Trained AI models stored per panel. *[Excluded from Git]*

## 🚀 Installation & Deployment
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Jvgvisser/pv-panel-predictor](https://github.com/Jvgvisser/pv-panel-predictor)

-   Weather: https://open-meteo.com\
-   Inspired by: https://github.com/b3nn0/EpexPredictor\
-   Designed for Enphase IQ7+ and Home Assistant
