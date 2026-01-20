from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List

import pandas as pd
import requests


@dataclass
class HAClient:
    base_url: str
    token: str
    timeout_s: int = 30

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    def fetch_history_period(
        self,
        entity_id: str,
        start: datetime,
        end: datetime,
        minimal_response: bool = True,
    ) -> List[Dict[str, Any]]:
        start_iso = start.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        end_iso = end.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

        url = f"{self.base_url.rstrip('/')}/api/history/period/{start_iso}"
        params = {
            "filter_entity_id": entity_id,
            "end_time": end_iso,
        }
        if minimal_response:
            params["minimal_response"] = "1"

        r = requests.get(url, headers=self._headers(), params=params, timeout=self.timeout_s)
        r.raise_for_status()
        data = r.json()

        if not data:
            return []
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
            return data[0]
        return data

    @staticmethod
    def lifetime_to_hourly_kwh(
        history: List[Dict[str, Any]],
        tz: str = "Europe/Amsterdam",
        scale_to_kwh: float = 0.001,
    ) -> pd.DataFrame:
        """
        history: HA states with 'last_changed'/'last_updated' and 'state' as lifetime (Wh in your case)
        Returns hourly kWh production (delta per hour), robustly filled.
        """
        if not history:
            return pd.DataFrame(columns=["ts", "kwh"])

        rows = []
        for s in history:
            st = s.get("state")
            if st in (None, "unknown", "unavailable"):
                continue
            try:
                val = float(st)
            except Exception:
                continue
            ts = s.get("last_changed") or s.get("last_updated")
            if not ts:
                continue
            rows.append((ts, val))

        if not rows:
            return pd.DataFrame(columns=["ts", "kwh"])

        df = pd.DataFrame(rows, columns=["ts_raw", "lifetime"])
        df["ts"] = pd.to_datetime(df["ts_raw"], utc=True).dt.tz_convert(tz)
        df = df.sort_values("ts").drop_duplicates(subset=["ts"])

        s = df.set_index("ts")["lifetime"]

        # Resample hourly: take last within hour, then forward-fill across missing hours
        hourly = s.resample("1H").last().ffill()

        # Delta per hour (in lifetime-units), then scale to kWh
        kwh = hourly.diff() * scale_to_kwh

        # Counter resets/negatives -> 0
        kwh[kwh < 0] = 0.0

        out = kwh.dropna().to_frame(name="kwh").reset_index()
        return out
