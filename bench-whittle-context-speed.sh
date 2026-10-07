#!/bin/sh
set -eu

repo=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
driver=/Users/ghost233/code/zx-bench/scripts/bench-context-speed.py
output="$repo/build/context-speed-whittle-$(date +%Y%m%d-%H%M%S)"

# Same driver and measurement protocol as the original 27B / Tiel runs.
exec "$repo/.venv/bin/python" "$driver" \
    --url http://192.168.64.1:8006 \
    --output "$output" \
    --concurrency 1 2 3 4 --decode 512 "$@" \
    --contexts 8000 16000 32000 64000
