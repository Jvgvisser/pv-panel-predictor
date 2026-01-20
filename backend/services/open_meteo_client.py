from pathlib import Path

p = Path("backend/app/services/open_meteo_client.py")
txt = p.read_text()

if "def fetch_hourly_archive" in txt:
    print("✅ fetch_hourly_archive bestaat al, niets te doen:", p)
else:
    inject = """

    def fetch_hourly_archive(
        self,
        *,
        lat: float,
        lon: float,
        start_date,
        end_date,
        timezone: str = "Europe/Amsterdam",
        tilt: float | None = None,
        azimuth: float | None = None,
    ):
        \"""
        Convenience wrapper for training on long-term history.
        Uses existing fetch_hourly_range(start_date..end_date) under the hood.
        \"""
        return self.fetch_hourly_range(
            lat=lat,
            lon=lon,
            start_date=start_date,
            end_date=end_date,
            timezone=timezone,
            tilt=tilt,
            azimuth=azimuth,
        )
"""
    txt = txt.rstrip() + "\n" + inject + "\n"
    p.write_text(txt)
    print("✅ Added fetch_hourly_archive() to", p)
