from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import requests


@dataclass
class OpenMeteoClient:
    """
    Training needs historical hourly GTI + weather.
    Forecast needs next N days.

    Hosts:
      - Forecast: https://api.open-meteo.com/v1/forecast
      - Historical Forecast: https://historical-forecast-api.open-meteo.com/v1/forecast
        (same params, different host)
    """
    timeout_s: int = 30

    def _get(self, host: str, params: dict) -> dict:
        url = f"{host}/v1/forecast"
        r = requests.get(url, params=params, timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()

    def fetch_hourly_range(
        self,
        latitude: float,
        longitude: float,
        start_date: date,
        end_date: date,
        tilt_deg: float,
        azimuth_deg: float,
        timezone: str = "Europe/Amsterdam",
        models: Optional[str] = None,
        historical: bool = False,
    ) -> pd.DataFrame:
        host = "https://historical-forecast-api.open-meteo.com" if historical else "https://api.open-meteo.com"

        hourly = [
            "temperature_2m",
            "cloud_cover",
            "global_tilted_irradiance",
        ]

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": ",".join(hourly),
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "timezone": timezone,
            "tilt": tilt_deg,
            "azimuth": azimuth_deg,
        }
        if models:
            params["models"] = models

        js = self._get(host, params)
        h = js.get("hourly", {}) or {}
        times = h.get("time", [])
        if not times:
            return pd.DataFrame(index=pd.DatetimeIndex([], tz=timezone))

        df = pd.DataFrame(
            {
                "time": pd.to_datetime(times),
                "temperature_2m": h.get("temperature_2m", []),
                "cloud_cover": h.get("cloud_cover", []),
                "global_tilted_irradiance": h.get("global_tilted_irradiance", []),
            }
        ).set_index("time")

        # When timezone param is set, Open-Meteo returns local timestamps; ensure tz-aware
        if df.index.tz is None:
            df.index = df.index.tz_localize(timezone, ambiguous="NaT", nonexistent="shift_forward")
        df = df[~df.index.isna()]

        return df

    def fetch_history_days(
        self,
        latitude: float,
        longitude: float,
        days: int,
        tilt_deg: float,
        azimuth_deg: float,
        timezone: str = "Europe/Amsterdam",
    ) -> pd.DataFrame:
        end = date.today()
        start = end - timedelta(days=days)
        return self.fetch_hourly_range(
            latitude=latitude,
            longitude=longitude,
            start_date=start,
            end_date=end,
            tilt_deg=tilt_deg,
            azimuth_deg=azimuth_deg,
            timezone=timezone,
            historical=True,
        )

    def fetch_forecast_days(
        self,
        latitude: float,
        longitude: float,
        days: int,
        tilt_deg: float,
        azimuth_deg: float,
        timezone: str = "Europe/Amsterdam",
    ) -> pd.DataFrame:
        start = date.today()
        end = start + timedelta(days=days)
        return self.fetch_hourly_range(
            latitude=latitude,
            longitude=longitude,
            start_date=start,
            end_date=end,
            tilt_deg=tilt_deg,
            azimuth_deg=azimuth_deg,
            timezone=timezone,
            historical=False,
        )

    def fetch_hourly_archive(
        self,
        latitude: float,
        longitude: float,
        start_date,
        end_date,
        tilt_deg: float,
        azimuth_deg: float,
        timezone: str = "Europe/Amsterdam",
        models: str | None = None,
    ):
        """Wrapper for long-term training using the historical-forecast host."""
        return self.fetch_hourly_range(
            latitude=latitude,
            longitude=longitude,
            start_date=start_date,
            end_date=end_date,
            tilt_deg=tilt_deg,
            azimuth_deg=azimuth_deg,
            timezone=timezone,
            models=models,
            historical=True,
        )

    def fetch_hourly_forecast(
        self,
        latitude: float,
        longitude: float,
        days: int = 3,
        tilt_deg: float = 35.0,
        azimuth_deg: float = 0.0,
        timezone: str = "Europe/Amsterdam",
        models: Optional[str] = None,
        start_date: date | None = None,
        end_date: date | None = None,
        **_: object,
    ) -> pd.DataFrame:
        """
        Convenience wrapper used by predict endpoint.
        Accepts either days or explicit start/end dates.
        Ignores unknown kwargs for forward-compatibility.
        """
        if start_date is None:
            start_date = date.today()
        if end_date is None:
            end_date = start_date + timedelta(days=int(days))

        return self.fetch_hourly_range(
            latitude=latitude,
            longitude=longitude,
            start_date=start_date,
            end_date=end_date,
            tilt_deg=tilt_deg,
            azimuth_deg=azimuth_deg,
            timezone=timezone,
            models=models,
            historical=False,
        )

