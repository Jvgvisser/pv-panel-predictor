from pathlib import Path
import re

p = Path("backend/app/services/ml.py")
txt = p.read_text(encoding="utf-8")

# We zoeken de plek waar X = feature_df[trained.features] staat
needle = "X = feature_df[trained.features]"
if needle not in txt:
    raise SystemExit("ERROR: kon 'X = feature_df[trained.features]' niet vinden")

# Voeg vlak ervoor robust-missing-features logic toe
replacement = """\
# Ensure we always have all trained features available at prediction time.
# Open-Meteo fields may be missing (or models may have been trained with older/newer feature sets).
missing = [c for c in trained.features if c not in feature_df.columns]
if missing:
    for c in missing:
        feature_df[c] = 0.0

X = feature_df[trained.features]"""

txt2 = txt.replace(needle, replacement, 1)
p.write_text(txt2, encoding="utf-8")
print("OK patched:", p)
