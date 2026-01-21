from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import websockets

@dataclass
class HAStatsWSClient:
    """
    Client voor de Home Assistant Long Term Statistics API.
    Haalt uuroverzichten op uit de 'statistics' tabel (geen 10-dagen limiet).
    """
    base_url: str
    token: str

    def _ws_url(self) -> str:
        u = self.base_url.strip()
        if u.startswith("https://"):
            return "wss://" + u[len("https://"):] + "/api/websocket"
        if u.startswith("http://"):
            return "ws://" + u[len("http://"):] + "/api/websocket"
        return "ws://" + u + "/api/websocket"

    async def _rpc(self, payload: Dict[str, Any]) -> Any:
        ws_url = self._ws_url()
        async with websockets.connect(ws_url, ping_interval=20, ping_timeout=20) as ws:
            # 1. Ontvang auth_required
            await ws.recv()
            
            # 2. Authenticatie
            await ws.send(json.dumps({
                "type": "auth",
                "access_token": self.token
            }))
            auth_resp = json.loads(await ws.recv())
            if auth_resp.get("type") != "auth_ok":
                raise RuntimeError(f"HA Auth mislukt: {auth_resp}")

            # 3. Data verzoek
            await ws.send(json.dumps(payload))
            
            # De LTS API kan bij 365 dagen even tijd nodig hebben voor het antwoord
            resp = json.loads(await ws.recv())
            if not resp.get("success"):
                raise RuntimeError(f"HA LTS API fout: {resp}")
                
            return resp.get("result")

    def fetch_hourly_energy_kwh_from_stats(
        self,
        entity_id: str,
        days: int,
        now: Optional[datetime] = None,
    ) -> List[dict]:
        """
        Haalt de 'sum' (LTS) op voor de opgegeven entiteit.
        """
        if now is None:
            now = datetime.now(timezone.utc)

        # Voor LTS is het cruciaal om op hele uren te werken
        end = now.replace(minute=0, second=0, microsecond=0)
        start = end - timedelta(days=days)

        # De exact geformateerde payload voor de LTS API
        payload = {
            "id": 1,
            "type": "recorder/statistics_during_period",
            "start_time": start.isoformat(),
            "end_time": end.isoformat(),
            "statistic_ids": [entity_id],
            "period": "hour",
            "types": ["sum"] # Lifetime sensors gebruiken altijd 'sum' in LTS
        }

        try:
            result = asyncio.run(self._rpc(payload))
            
            if isinstance(result, dict) and entity_id in result:
                points = result[entity_id]
                print(f"📊 HA LTS API succes: {len(points)} uren opgehaald voor {entity_id}")
                return points
            
            print(f"⚠️ HA LTS API: Geen data gevonden in 'statistics' tabel voor {entity_id}")
            return []

        except Exception as e:
            print(f"‼️ HA LTS API CRASH: {e}")
            return []