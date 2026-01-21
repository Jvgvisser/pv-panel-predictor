from pathlib import Path
import re

p = Path("backend/app/services/ml.py")
txt = p.read_text()

# Replace the two lines that currently do: out["hour"] = t.hour / out["doy"] = t.dayofyear
# with a dt-safe variant.
pattern = r'out\["hour"\]\s*=\s*t\.hour\s*\n\s*out\["doy"\]\s*=\s*t\.dayofyear\s*'
replacement = (
    'out["hour"] = (t.dt.hour if hasattr(t, "dt") else t.hour)\n'
    '    out["doy"] = (t.dt.dayofyear if hasattr(t, "dt") else t.dayofyear)'
)

new_txt, n = re.subn(pattern, replacement, txt, count=1, flags=re.M)
if n != 1:
    raise SystemExit(f"❌ Could not patch hour/doy lines (matches={n}). Open {p} and patch manually.")

p.write_text(new_txt)
print("✅ Patched:", p)
