from pathlib import Path
import re

p = Path("backend/app/services/ml.py")
txt = p.read_text()

# Add 'ts' (and 'end' is sometimes used too) to candidates list
txt2 = re.sub(
    r'candidates\s*=\s*\[(.*?)\]',
    lambda m: (
        "candidates = ["
        "\"time\", \"ts\", \"start\", \"end\", \"datetime\", \"timestamp\", \"date\", "
        "\"last_changed\", \"last_updated\""
        "]"
    ),
    txt,
    count=1,
    flags=re.S
)

if txt2 == txt:
    raise SystemExit("❌ Could not patch candidates list (pattern not found).")

p.write_text(txt2)
print("✅ Patched:", p)
