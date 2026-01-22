import json
from pathlib import Path
from typing import List, Dict
from backend.app.models.panel import PanelConfig

class PanelsRepo:
    def __init__(self, path: str = "data/panels.json") -> None:
        self.path = Path(path)
        # Create data folder if it doesn't exist
        self.path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize file if it doesn't exist or is empty
        if not self.path.exists() or self.path.stat().st_size == 0:
            self._save_to_disk([])

    def _save_to_disk(self, panels_list: List[PanelConfig]) -> None:
        """Helper to write the list directly to JSON file."""
        # We use model_dump() for Pydantic v2, or .dict() for v1
        data = [p.model_dump() if hasattr(p, 'model_dump') else p.dict() for p in panels_list]
        self.path.write_text(json.dumps(data, indent=4))

    def list(self) -> List[PanelConfig]:
        """Loads and returns all panels from the JSON file."""
        try:
            content = self.path.read_text()
            data = json.loads(content)
            # Handle both formats: flat list or dict with "panels" key
            if isinstance(data, dict) and "panels" in data:
                data = data["panels"]
            return [PanelConfig(**p) for p in data]
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def get(self, panel_id: str) -> PanelConfig:
        for p in self.list():
            if p.panel_id == panel_id:
                return p
        raise KeyError(f"Panel not found: {panel_id}")

    def upsert(self, panel: PanelConfig) -> None:
        """Adds or updates a panel and immediately saves to disk."""
        panels = self.list()
        found = False
        for i, p in enumerate(panels):
            if p.panel_id == panel.panel_id:
                panels[i] = panel
                found = True
                break
        if not found:
            panels.append(panel)
        
        self._save_to_disk(panels)

    def delete(self, panel_id: str) -> None:
        """Removes a panel and immediately saves to disk."""
        panels = [p for p in self.list() if p.panel_id != panel_id]
        self._save_to_disk(panels)

    def _save(self):
        """Compatibility alias for main.py if needed."""
        pass