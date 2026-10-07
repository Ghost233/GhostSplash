#!/bin/sh
set -eu

# Local Swift Bonsai PQ2_0 target in text-only mode; matching DFlash2 draft.
model_dir=/Users/ghost233/.lmstudio/models/ukisai/Swift-Bonsai-2-GGUF
model_id=ukisai/Swift-Bonsai-2-GGUF:PQ2_0
host=192.168.64.1
port=8004
max_context=262144
max_memory=40G
max_cache_disk=50G
queue_size=4
repo=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
python="$repo/.venv/bin/python"

for path in "$model_dir/Swift-Bonsai-2-PQ2_0.gguf" \
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
exec "$python" -u - "$repo" "$model_dir" "$model_id" "$host" "$port" \
    "$max_context" "$max_memory" "$max_cache_disk" "$queue_size" <<'PY'
import os
import socket
import sys
from pathlib import Path

repo, model_dir, model_id, host, port, context, memory, cache_disk, queue = sys.argv[1:]
repo = Path(repo)
sys.path.insert(0, str(repo))
from install import assembly, hub, models, upstream

# Check the address before the first-run draft download.
with socket.socket() as listener:
    listener.bind((host, int(port)))

selection = models.Selection.of(repo / "install/models-local", model_id, language_only=True)
if models.installation_kind(selection.link) is None:
    upstream._install(selection, hub.Repository.local_directory(Path(model_dir)), None)
else:
    record = assembly.verify(selection.link)
    if record["sources"]["target"] != {"repo": model_dir, "revision": None}:
        raise models.ModelError("Prepared target does not match the local Swift Bonsai directory")

# Hold the verified assembly until the server exits, as the official launcher does.
root, record_lock = assembly.hold(selection.link, selection.models_root)
if record_lock is not None:
    os.set_inheritable(record_lock.fileno(), True)
print(f"Swift Bonsai Splash: http://{host}:{port} (API: /v1)", flush=True)
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
