from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import websockets


@dataclass
class HAStatsWSClient:
    """
    Home Assistant WebSocket client for long-term statistics.

    We use 'recorder/statistics_during_period' which returns aggregated hourly points
    for entities that have statistics (state_class total/total_increasing).
    This avoids the 10-day recorder limitation for raw history.
    """
    base_url: str          # e.g. http://192.168.1.10:8123
    token: str             # long-lived access token

    def _ws_url(self) -> str:
        u = self.base_url.strip()
        if u.startswith("https://"):
            return "wss://" + u[len("https://"):] + "/api/websocket"
        if u.startswith("http://"):
            return "ws://" + u[len("http://"):] + "/api/websocket"
        # fallback
        return "ws://" + u + "/api/websocket"

    async def _rpc(self, payload: Dict[str, Any]) -> Any:
        ws_url = self._ws_url()
        async with websockets.connect(ws_url, ping_interval=20, ping_timeout=20) as ws:
            # 1) auth_required
            msg = await ws.recv()
            # 2) auth
            await ws.send(
                __import__("json").dumps({"type": "auth", "access_token": self.token})
            )
            auth = __import__("json").loads(await ws.recv())
            if auth.get("type") != "auth_ok":
                raise RuntimeError(f"HA auth failed: {auth}")

            # 3) request
            await ws.send(__import__("json").dumps(payload))
            resp = __import__("json").loads(await ws.recv())
            if not resp.get("success", False):
                raise RuntimeError(f"HA WS call failed: {resp}")
            return resp.get("result")

    async def statistics_during_period(
        self,
        start: datetime,
        end: datetime,
        statistic_ids: List[str],
        period: str = "hour",
        types: Optional[List[str]] = None,
    ) -> Any:
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("start/end must be timezone-aware datetimes")

        # HA expects ISO strings, typically UTC is safest
        start_utc = start.astimezone(timezone.utc).isoformat()
        end_utc = end.astimezone(timezone.utc).isoformat()

        payload = {
            "id": 1,
            "type": "recorder/statistics_during_period",
            "start_time": start_utc,
            "end_time": end_utc,
            "statistic_ids": statistic_ids,
            "period": period,
        }
        if types:
            payload["types"] = types

        return await self._rpc(payload)

    def fetch_hourly_energy_kwh_from_stats(
        self,
        entity_id: str,
        days: int,
        now: Optional[datetime] = None,
    ) -> List[dict]:
        """
        Returns list of points:
          { "start": "...", "sum": <kWh total increasing> }   (or "state"/"mean" depending)
        For energy sensors it is typically 'sum' for total_increasing.
        """
        if now is None:
            now = datetime.now(timezone.utc)

        start = now - timedelta(days=days)
        end = now

        # Types: ask for sum primarily; HA will return what exists.
        result = asyncio.run(
            self.statistics_during_period(
                start=start,
                end=end,
                statistic_ids=[entity_id],
                period="hour",
                types=["sum", "state", "mean"],
            )
        )
        # result is typically a dict keyed by statistic_id -> list of points
        if isinstance(result, dict):
            return result.get(entity_id, []) or []
        return []
