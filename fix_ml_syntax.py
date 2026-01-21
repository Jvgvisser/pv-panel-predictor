from pathlib import Path

p = Path("backend/app/services/ml.py")
txt = p.read_text()

bad = 'out["doy"] = (t.dt.dayofyear if hasattr(t, "dt") else t.dayofyear)out["hour_sin"]'
if bad not in txt:
    raise SystemExit("Pattern not found. Open ml.py and search for the glued line around add_time_features().")

txt = txt.replace(
    bad,
    'out["doy"] = (t.dt.dayofyear if hasattr(t, "dt") else t.dayofyear)\n\n    out["hour_sin"]'
)

p.write_text(txt)
print("OK fixed:", p)
