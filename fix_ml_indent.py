from __future__ import annotations
from pathlib import Path
import re

p = Path("backend/app/services/ml.py")
txt = p.read_text(encoding="utf-8")
lines = txt.splitlines(True)  # keep \n

# Zoek "def predict(" met 4 spaties indent (method in class)
start = None
for i, line in enumerate(lines):
    if re.match(r"^ {4}def predict\(", line):
        start = i
        break

if start is None:
    raise SystemExit("ERROR: kon '    def predict(' niet vinden in ml.py")

# Header kan multi-line zijn; vind de eerste regel die eindigt met ':' op indent 4
header_end = None
for i in range(start, min(start + 80, len(lines))):
    if re.match(r"^ {4}.*:\s*$", lines[i]):
        header_end = i
        break

if header_end is None:
    raise SystemExit("ERROR: kon het einde van de predict() header niet vinden (':' ontbreekt?)")

body_start = header_end + 1

# Einde van functie: volgende method/decorator op indent 4
end = len(lines)
for i in range(body_start, len(lines)):
    if re.match(r"^ {4}(def |@)", lines[i]):
        end = i
        break

# Fix indent in body: alles wat niet leeg is, moet minstens 8 spaties hebben
new_lines = lines[:]
for i in range(body_start, end):
    line = new_lines[i]
    if line.strip() == "":
        continue

    # tel leidende spaties (tabs laten we staan, maar meestal niet aanwezig)
    m = re.match(r"^( *)", line)
    leading = len(m.group(1)) if m else 0

    if leading >= 8:
        continue
    elif leading == 4:
        new_lines[i] = " " * 4 + line  # +4 => 8
    else:
        # 0..3 spaties => forceer 8 (maar behoud rest van lijn)
        new_lines[i] = " " * 8 + line.lstrip(" ")

fixed = "".join(new_lines)

if fixed == txt:
    print("OK: niets gewijzigd (indent leek al goed?)")
else:
    p.write_text(fixed, encoding="utf-8")
    print(f"OK: indent gefixt in {p} (regels {body_start+1}..{end})")
