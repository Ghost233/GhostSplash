#!/bin/sh
set -eu

# Experimental Qwen3.5-4B, text only, ordinary decoding, GPU prefill.
repo=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
launcher_allow_baseline=1
. "$repo/dev/tools/launcher_kv.sh"
splash_parse_launcher_options "$@"
binary="$repo/build/splash"
if [ "$use_baseline" -eq 1 ]; then
    binary="$repo/build/baseline-qwen4b/splash"
    case "$kv_format" in
        int8|bf16) ;;
        *) echo "The earlier baseline supports only --kv-format int8 or bf16" >&2; exit 2 ;;
    esac
fi
model_dir=/Users/ghost233/.lmstudio/models/unsloth/Qwen3.5-4B-GGUF
model_id=unsloth/Qwen3.5-4B-GGUF:UD-Q4_K_XL
host=127.0.0.1
port=8011
max_context=33792
max_memory=12G
max_cache_disk=0
queue_size=1
for path in "$model_dir/Qwen3.5-4B-UD-Q4_K_XL.gguf" "$repo/.venv/bin/python" "$binary"; do
    if [ ! -e "$path" ]; then
        echo "Missing: $path" >&2
        exit 1
    fi
done
unset SPLASH_API_KEY
echo "KV format: $kv_format"
exec "$repo/.venv/bin/python" -u - "$repo" "$model_dir" "$model_id" "$host" "$port" \
    "$max_context" "$max_memory" "$max_cache_disk" "$queue_size" "$kv_format" "$binary" <<'PY'
import os
import socket
import sys
from pathlib import Path

repo, model_dir, model_id, host, port, context, memory, cache_disk, queue, kv_format, *binaries = sys.argv[1:]
binary = binaries[0] if binaries else str(Path(repo) / "build/splash")
repo = Path(repo)
sys.path.insert(0, str(repo))
from install import paths
from install.local import prepare_local_model

paths.BINARY = Path(binary)
with socket.socket() as listener:
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((host, int(port)))

root = prepare_local_model(
    Path(model_dir) / "Qwen3.5-4B-UD-Q4_K_XL.gguf", language_only=True,
    models_dir=repo / "install/models-local",
)
print(f"Qwen3.5-4B: http://{host}:{port}/v1; model root: {root}", flush=True)
command = [
    str(repo / ".venv/bin/python"), "-u", "-m", "server.server", str(root),
    "--model", model_id,
    "--binary", binary, "--host", host, "--port", str(port),
    "--max-context", context, "--max-memory", memory, "--max-cache-disk", cache_disk,
    "--queue-size", queue, "--kv-format", kv_format, "--disable-ane", "--no-webui",
]
os.chdir(repo)
os.execv(command[0], command)
PY
