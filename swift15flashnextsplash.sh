#!/bin/sh
set -eu

# Swift Flash Next IQ3_S, text only, SSD-paged target and native BF16 MTP head.
model_dir=/Users/ghost233/.lmstudio/models/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF
draft_dir=/Users/ghost233/.lmstudio/models/sh0wie/Qwen3.8-Flash-Next-MTP-Drafter-MLX-bf16
mtp_enabled=${SPLASH_MTP:-1}
export SPLASH_MTP="$mtp_enabled"
model_id=ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF:IQ3_S
host=192.168.64.1
port=8008
max_context=262144
max_memory=40G
max_cache_disk=50G
queue_size=4
repo=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
python="$repo/.venv/bin/python"

for path in "$model_dir/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf" \
    "$model_dir/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf" \
    "$draft_dir/config.json" "$draft_dir/model.safetensors" \
    "$python" "$repo/server/server.py" "$repo/build/splash"; do
    if [ ! -e "$path" ]; then
        echo "Missing: $path" >&2
        exit 1
    fi
done
if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "Port $port is already in use; stop that server first." >&2
    exit 1
fi

unset SPLASH_API_KEY
exec "$python" -u - "$repo" "$model_dir" "$model_id" "$draft_dir" "$host" "$port" \
    "$max_context" "$max_memory" "$max_cache_disk" "$queue_size" <<'PY'
import os
import socket
import sys
from pathlib import Path

repo, model_dir, model_id, draft_dir, host, port, context, memory, cache_disk, queue = sys.argv[1:]
repo = Path(repo)
sys.path.insert(0, str(repo))
from install import assembly, hub, models, upstream

# Check the address before preparing the local split model.
with socket.socket() as listener:
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((host, int(port)))

selection = models.Selection.of(repo / "install/models-local", model_id, language_only=True, draft_model=draft_dir)
if models.installation_kind(selection.link) is None:
    upstream._install(selection, hub.Repository.local_directory(Path(model_dir)), None)
else:
    record = assembly.verify(selection.link)
    if record["sources"]["target"] != {"repo": model_dir, "revision": None}:
        raise models.ModelError("Prepared target does not match the local Swift Flash Next directory")
    if record["sources"]["draft"] != {"repo": draft_dir, "revision": None}:
        raise models.ModelError("Prepared MTP does not match the local Flash Next drafter")

# Hold the verified assembly until the server exits, as the official launcher does.
root, record_lock = assembly.hold(selection.link, selection.models_root)
if record_lock is not None:
    os.set_inheritable(record_lock.fileno(), True)
print(f"Swift Flash Next Splash: http://{host}:{port} (API: /v1)", flush=True)
print(f"Native Flash Next MTP: {draft_dir}; SPLASH_MTP={os.environ['SPLASH_MTP']}; 18 GiB SSD expert cache.", flush=True)
command = [
    str(repo / ".venv/bin/python"), "-u", "-m", "server.server",
    str(root), "--tokenizer", str(root / "tokenizer"),
    "--model", model_id, "--served-model-name", "splash_" + model_id,
    "--binary", str(repo / "build/splash"), "--host", host, "--port", port,
    "--max-context", context, "--max-memory", memory, "--max-cache-disk", cache_disk,
    "--queue-size", queue,
]
os.chdir(repo)
os.execv(command[0], command)
PY
