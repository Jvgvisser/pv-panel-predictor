# PV Panel Predictor (ML-Based)

A machine learning service that predicts solar yield for multiple individual PV panels using LightGBM. It fetches historical production data from Home Assistant, weather data from Open-Meteo, and provides an aggregated forecast for evcc and Home Assistant dashboards.

## Features
* Individual Panel Modeling: Trains a separate LightGBM model for every panel to account for unique shading, tilt, and orientation.
* LightGBM Engine: Uses Gradient Boosting for high-accuracy predictions based on solar elevation, azimuth, and Global Tilted Irradiance (GTI).
* evcc Integration: Provides an hourly forecast array compatible with evcc's custom forecast type.
* Home Assistant Ready: Dedicated endpoint for daily totals to simplify Energy Dashboard integration.

---

## API Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| /api/train/all | POST | Trains models for all registered panels. |
| /api/predict/total | GET | Returns aggregated hourly forecast (Direct JSON Array for evcc). |
| /api/predict/total/daily | GET | Returns aggregated daily totals (Simplified for Hass). |

---

## Automation (Cronjob)

To keep your models sharp and your forecast fresh, set up a cronjob on your Linux host/LXC.

1. Run: crontab -e
2. Add the following lines:

# Train all panels daily at 03:00 AM using the last 30 days of data
0 3 * * * curl -X POST "http://127.0.0.1:8000/api/train/all?days=30"

# Optional: Refresh/Ping the forecast every 15 minutes to ensure responsiveness
*/15 * * * * curl -s "http://127.0.0.1:8000/api/predict/total" > /dev/null

---

## Integration Examples

### 1. evcc Configuration (evcc.yaml)
Integrate the predictor as a custom solar forecast to optimize your EV charging plan.

site:
  pv:
    - name: solar_roof
      capacity: 11.3 # Total kWp of all panels
  forecast:
    type: custom
    url: http://192.168.X.XXXX:8000/api/predict/total

### 2. Home Assistant Sensor (configuration.yaml)
Create a sensor that shows the total expected yield for the current day.

sensor:
  - platform: rest
    name: "Solar Forecast Today"
    resource: http://192.168.X.XXXXX:8000/api/predict/total/daily
    value_template: >
      {% set today = now().strftime('%Y-%m-%d') %}
      {% set day_data = value_json | selectattr('date', 'eq', today) | first %}
      {{ day_data.kwh if day_data else 0 }}
    unit_of_measurement: "kWh"
    scan_interval: 900 # Refresh every 15 minutes

---

## Technical Details: LightGBM

The system utilizes LightGBM (Light Gradient Boosting Machine) for its prediction engine:
* Inputs: Hour of day, month, clear-sky index, and Global Tilted Irradiance (GTI) calculated specifically for each panel's tilt/azimuth.
* Training: The model learns the non-linear relationship between weather parameters and your actual hardware performance, including specific shading patterns.
* Performance: Optimized for low-resource environments (like LXC containers) while maintaining high accuracy.

---

Thanks to:
-   Inspired by: https://github.com/b3nn0/EpexPredictor\
-   Designed for Enphase IQ7+ and Home Assistant
-   Weather: https://open-meteo.com\