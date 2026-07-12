from pathlib import Path
import re

ROOT = Path("ml")

print("=" * 40)
print("Feature Usage Audit")
print("=" * 40)

python_files = sorted(ROOT.glob("*.py"))

for path in python_files:

    text = path.read_text(encoding="utf-8")

    if "from common.features import FEATURE_COLUMNS" in text:
        status = "OK   common.features"

    elif re.search(r"FEATURE_COLUMNS\s*=\s*\[", text):
        status = "WARN local FEATURE_COLUMNS"

    else:
        status = "---- not used"

    print(f"{path.name:35} {status}")