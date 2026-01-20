import json
from pathlib import Path
from typing import List

from backend.app.models.panel import PanelConfig


class PanelsRepo:
    def __init__(self, path: str = "data/panels.json") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text(json.dumps({"panels": []}, indent=2))

    def list(self) -> List[PanelConfig]:
        data = json.loads(self.path.read_text())
        return [PanelConfig(**p) for p in data.get("panels", [])]

    def get(self, panel_id: str) -> PanelConfig:
        for p in self.list():
            if p.panel_id == panel_id:
                return p
        raise KeyError(f"Panel not found: {panel_id}")

    def upsert(self, panel: PanelConfig) -> None:
        panels = self.list()
        replaced = False
        for i, p in enumerate(panels):
            if p.panel_id == panel.panel_id:
                panels[i] = panel
                replaced = True
                break
        if not replaced:
            panels.append(panel)

        self.path.write_text(
            json.dumps({"panels": [p.model_dump() for p in panels]}, indent=2)
        )

    def delete(self, panel_id: str) -> None:
        panels = [p for p in self.list() if p.panel_id != panel_id]
        self.path.write_text(
            json.dumps({"panels": [p.model_dump() for p in panels]}, indent=2)
        )
