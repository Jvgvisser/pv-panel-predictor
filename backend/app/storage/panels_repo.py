import json
import os
from pathlib import Path
from typing import List
from backend.app.models.panel import PanelConfig

class PanelsRepo:
    def __init__(self, path: str = "data/panels.json") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            print(f"📂 Create new storage at {self.path}")
            self._save_raw([])

    def _save_raw(self, data_list: list):
        """Schrijft de rauwe lijst naar disk."""
        with open(self.path, "w") as f:
            json.dump(data_list, f, indent=4)
        # Forceer sync naar disk (OS cache omzeilen)
        os.sync() if hasattr(os, 'sync') else None

    def list(self) -> List[PanelConfig]:
        try:
            content = self.path.read_text()
            data = json.loads(content)
            # Support voor zowel de dict {"panels": []} als de platte lijst []
            if isinstance(data, dict) and "panels" in data:
                data = data["panels"]
            return [PanelConfig(**p) for p in data]
        except Exception as e:
            print(f"❌ Error loading panels: {e}")
            return []

    def upsert(self, panel: PanelConfig) -> None:
        print(f"💾 Upserting panel: {panel.panel_id} ({panel.friendly_name})")
        panels = self.list()
        
        # Converteer alles naar dicts voor opslag
        panel_dict = panel.model_dump() if hasattr(panel, 'model_dump') else panel.dict()
        
        new_panels = []
        found = False
        for p in panels:
            p_dict = p.model_dump() if hasattr(p, 'model_dump') else p.dict()
            if p_dict["panel_id"] == panel.panel_id:
                new_panels.append(panel_dict)
                found = True
            else:
                new_panels.append(p_dict)
        
        if not found:
            new_panels.append(panel_dict)

        self._save_raw(new_panels)
        print(f"✅ Saved to {self.path}. Current count: {len(new_panels)}")

    def get(self, panel_id: str) -> PanelConfig:
        for p in self.list():
            if p.panel_id == panel_id:
                return p
        raise KeyError(f"Panel not found: {panel_id}")

    def delete(self, panel_id: str) -> None:
        panels = [p.model_dump() if hasattr(p, 'model_dump') else p.dict() 
                  for p in self.list() if p.panel_id != panel_id]
        self._save_raw(panels)

    def _save(self):
        """Dummy voor compatibiliteit met main.py"""
        pass