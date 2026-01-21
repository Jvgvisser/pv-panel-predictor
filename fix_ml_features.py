from pathlib import Path
import re

p = Path("backend/app/services/ml.py")
txt = p.read_text()

needle = re.compile(
    r"features\s*=\s*\[[^\]]+\]\n",
    re.S
)

replacement = """features = [
            "global_tilted_irradiance",
            "shortwave_radiation",
            "cloud_cover",
            "temperature_2m",
            "kwh_lag_24",
            "gti_lag_24",
            "hour_sin", "hour_cos",
            "doy_sin", "doy_cos",
        ]

        # Ensure all expected feature columns exist
        for col in features:
            if col not in df.columns:
                df[col] = 0.0

"""

if not needle.search(txt):
    raise SystemExit("❌ Could not find features list in ml.py")

txt = needle.sub(replacement, txt, count=1)
p.write_text(txt)
print("✅ Patched:", p)
