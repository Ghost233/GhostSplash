#!/bin/sh
set -eu

# Compatibility entry point: keep its original INT8 KV and queue defaults.
host=0.0.0.0
repo=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
exec sh "$repo/swift1527biq3ssplash.sh" --host "$host" --kv-format int8 \
    --queue-size 4 --prompt-lookup off --compact-draft-head off "$@"
