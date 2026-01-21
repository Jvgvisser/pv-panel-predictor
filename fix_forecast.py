from pathlib import Path
import re

p = Path("backend/app/services/open_meteo_client.py")
txt = p.read_text(encoding="utf-8")

if "def fetch_hourly_forecast" in txt:
    print("OK: fetch_hourly_forecast bestaat al")
    raise SystemExit(0)

# Zoek de class definitie (met of zonder @dataclass)
m = re.search(r"(?m)^class\s+OpenMeteoClient\s*:\s*$", txt)
if not m:
    # soms is het "class OpenMeteoClient(...):"
    m = re.search(r"(?m)^class\s+OpenMeteoClient\s*\(.*?\)\s*:\s*$", txt)

if not m:
    raise SystemExit("ERROR: kon 'class OpenMeteoClient' niet vinden (check bestandsnaam / duplicaat pad)")

class_start = m.start()

# Vind einde van de class door te zoeken naar de volgende top-level 'class ' of 'def ' op kolom 0
# (we voegen methode in op het einde van de class body, dus vlak vóór die volgende top-level)
rest = txt[m.end():]
nxt = re.search(r"(?m)^(class|def)\s+\w+", rest)
insert_pos = m.end() + (nxt.start() if nxt else len(rest))

# Zorg dat we op een nette newline zitten
before = txt[:insert_pos].rstrip() + "\n\n"
after = txt[insert_pos:].lstrip("\n")

method = '''
    def fetch_hourly_forecast(
        self,
        latitude: float,
        longitude: float,
        start_date: date,
        end_date: date,
        tilt_deg: float,
        azimuth_deg: float,
        timezone: str = "Europe/Amsterdam",
        models: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Convenience wrapper used by predict endpoint.
        Uses the regular forecast host (not historical).
        """
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
'''.lstrip("\n")

txt2 = before + method + "\n" + after
p.write_text(txt2, encoding="utf-8")
print("OK patched:", p)
