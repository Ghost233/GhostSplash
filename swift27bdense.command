#!/bin/sh
set -eu

# Swift 27B Q4H defaults and feature switches are set by the canonical launcher.
repo=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
exec sh "$repo/swift1527biq3ssplash.sh" "$@"
