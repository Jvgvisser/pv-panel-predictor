from __future__ import annotations
import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
import websockets

@dataclass
class HAStatsWSClient:
    base_url: str
    token: str

    def _ws_url(self) -> str:
        u = self.base_url.strip().replace("http://", "ws://").replace("https://", "wss://")
        return u.rstrip("/") + "/api/websocket"

    async def fetch_hourly_energy_kwh_from_stats(
        self,
        entity_id: str,
        days: int,
        now: Optional[datetime] = None,
    ) -> List[dict]:
        if now is None:
            now = datetime.now(timezone.utc)
        
        end = now.replace(minute=0, second=0, microsecond=0)
        start = end - timedelta(days=days)

        try:
            async with websockets.connect(self._ws_url()) as ws:
                # Auth
                await ws.recv()
                await ws.send(json.dumps({"type": "auth", "access_token": self.token}))
                auth_res = json.loads(await ws.recv())
                if auth_res.get("type") != "auth_ok":
                    print(f"❌ HA Auth Fout: {auth_res}")
                    return []

                # Request
                payload = {
                    "id": 1,
                    "type": "recorder/statistics_during_period",
                    "start_time": start.isoformat(),
                    "end_time": end.isoformat(),
                    "statistic_ids": [entity_id],
                    "period": "hour",
                    "types": ["sum"]
                }
                await ws.send(json.dumps(payload))
                
                resp = json.loads(await ws.recv())
                if resp.get("success"):
                    points = resp.get("result", {}).get(entity_id, [])
                    print(f"📊 HA LTS API succes: {len(points)} uren voor {entity_id}")
                    return points
                return []
        except Exception as e:
            print(f"❌ HA WebSocket Error: {e}")
            return []