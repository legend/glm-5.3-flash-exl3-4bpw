#!/usr/bin/env bash
# Build-context check: every overlay .py must byte-compile before we hand the
# tree to docker build. Pure syntax check (in-memory compile, no imports, no
# __pycache__ writes). The image self-check RUN repeats this in-image.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
export PYTHON

exec "${PYTHON}" - <<'PY'
from pathlib import Path

root = Path("upstream-core-port")
files = sorted(root.rglob("*.py"))
assert files, "no overlay .py files found under upstream-core-port/"

for f in files:
    compile(f.read_bytes(), str(f), "exec")

# the in-image manifest must also parse and map only files we actually ship
manifest = (root / "MANIFEST.txt").read_text().splitlines()
maps = [l for l in manifest if l.strip() and not l.strip().startswith("#") and "->" in l]
for line in maps:
    src, dst = (p.strip() for p in line.split("->", 1))
    assert (root / src).is_file(), f"MANIFEST maps missing source: {src}"
assert len(maps) == 91, f"expected 91 manifest mappings, found {len(maps)}"

print(f"OVERLAY COMPILE-CHECK PASSED ({len(files)} files, {len(maps)} mappings)")
PY
