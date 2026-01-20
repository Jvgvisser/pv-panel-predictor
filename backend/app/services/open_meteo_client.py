from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, List

import pandas as pd
import requests


@dataclass
class OpenMeteoClient:
    timeout_s: int = 30

    def fetch_hourly_archive(
        self,
        latitude: float,
        longitude: float,
        start_date: date,
        end_date: date,
        tilt_deg: float,
        azimuth_deg: float,
        timezone: str = "Europe/Amsterdam",
    ) -> pd.DataFrame:
        """
        Historical hourly weather + irradiance from Open-Meteo archive.
        """
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "timezone": timezone,
            "hourly": ",".join(
                [
                    "temperature_2m",
                    "cloud_cover",
                    "global_tilted_irradiance",
                    "shortwave_radiation",
                ]
            ),
            "tilt": tilt_deg,
            "azimuth": azimuth_deg,
        }
        r = requests.get(url, params=params, timeout=self.timeout_s)
        r.raise_for_status()
        j = r.json()
        h = j.get("hourly", {})
        df = pd.DataFrame(h)
        if df.empty:
            return df
        df["time"] = pd.to_datetime(df["time"])
        return df

    def fetch_hourly_forecast(
        self,
        latitude: float,
        longitude: float,
        days: int,
        tilt_deg: float,
        azimuth_deg: float,
        timezone: str = "Europe/Amsterdam",
    ) -> pd.DataFrame:
        """
        Forecast hourly weather + irradiance from Open-Meteo forecast.
        """
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "forecast_days": days,
            "timezone": timezone,
            "hourly": ",".join(
                [
                    "temperature_2m",
                    "cloud_cover",
                    "global_tilted_irradiance",
                    "shortwave_radiation",
                ]
            ),
            "tilt": tilt_deg,
            "azimuth": azimuth_deg,
        }
        r = requests.get(url, params=params, timeout=self.timeout_s)
        r.raise_for_status()
        j = r.json()
        h = j.get("hourly", {})
        df = pd.DataFrame(h)
        if df.empty:
            return df
        df["time"] = pd.to_datetime(df["time"])
        return df
