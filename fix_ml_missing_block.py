from __future__ import annotations
from pathlib import Path
import re

p = Path("backend/app/services/ml.py")
txt = p.read_text(encoding="utf-8")

# We zoeken een blok dat ongeveer zo is:
# missing = [c for c in trained.features if c not in feature_df.columns]
# if missing:
#   ...
# X = feature_df[trained.features]
#
# En vervangen alles tussen "missing =" en vlak vóór "X = feature_df[trained.features]"
pattern = re.compile(
    r"""
(^[ ]{8}missing\s*=\s*\[c\s+for\s+c\s+in\s+trained\.features\s+if\s+c\s+not\s+in\s+feature_df\.columns\]\s*\n)
(?:^[^\n]*\n)*?
(?=^[ ]{8}X\s*=\s*feature_df\[trained\.features\]\s*$)
""",
    re.MULTILINE | re.VERBOSE,
)

m = pattern.search(txt)
if not m:
    raise SystemExit("ERROR: kon het missing-features blok niet vinden om te patchen.")

replacement = (
    "        # Ensure all features used at training time exist at prediction time\n"
    "        for _col in trained.features:\n"
    "            if _col not in feature_df.columns:\n"
    "                feature_df[_col] = 0.0\n\n"
)

txt2 = txt[:m.start()] + replacement + txt[m.end():]
p.write_text(txt2, encoding="utf-8")
print("OK patched:", p)
