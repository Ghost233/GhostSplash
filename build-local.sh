#!/bin/sh
set -eu

# Build the current checkout in its canonical local directories.
repo=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
cd "$repo"

make install-environment VENV=.venv BUILD=build DRAFT_BLOCK_SIZE=8
make -j4 all verify-build-identity VENV=.venv BUILD=build DRAFT_BLOCK_SIZE=8 \
    BUILD_ID_PYTHON="$repo/.venv/bin/python"

printf 'Splash source directory: %s\n' "$repo"
printf 'Python: %s\n' "$repo/.venv/bin/python"
printf 'Native engine: %s\n' "$repo/build/splash"
printf 'Metal library: %s\n' "$repo/build/splash.metallib"
