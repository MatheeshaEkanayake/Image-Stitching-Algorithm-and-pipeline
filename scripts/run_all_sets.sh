#!/usr/bin/env bash
# Render every folder under Images/ into out/<set>.tif (+ out/<set>.json report and .log).
# Usage: scripts/run_all_sets.sh [extra stitch.py options]
#   e.g. scripts/run_all_sets.sh --max-image-dimension 1200
# A set that fails (for example the wide 'images' sweep) is reported but does not stop the run.
set -u
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
EXT="${EXT:-tif}"   # lossless by default; EXT=jpg for small files
mkdir -p out
status=0
for dir in Images/*/; do
    name="$(basename "$dir")"
    echo "== $name"
    if "$PYTHON" -u stitch.py "$dir" "out/$name.$EXT" --report "out/$name.json" "$@" \
        > "out/$name.log" 2>&1; then
        tail -n 1 "out/$name.log"
    else
        echo "FAILED: $(grep -m1 ERROR "out/$name.log" || echo 'see out/'"$name"'.log')"
        status=1
    fi
done
exit $status
