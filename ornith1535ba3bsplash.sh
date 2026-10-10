#!/bin/sh
set -eu

# Local Ornith 1.5 35B-A3B 3bit target, text only; matching DFlash2 draft.
model_dir=/Users/ghost233/.lmstudio/models/pfeifferj/Ornith-1.5-35B-A3B-GSQ-RCO-GGUF
model_id=pfeifferj/Ornith-1.5-35B-A3B-GSQ-RCO-GGUF:3bit
# Listen on every IPv4 interface; virtual bridge addresses can change.
host=0.0.0.0
port=8009
max_context=262144
max_memory=40G
max_cache_disk=50G
queue_size=4
repo=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
launcher_allow_baseline=0
. "$repo/dev/tools/launcher_kv.sh"
splash_parse_launcher_options "$@"
python="$repo/.venv/bin/python"

for path in "$model_dir/Ornith-1.5-35B-A3B-GSQ-RCO-3bit.gguf" \
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
echo "KV format: $kv_format"
exec "$python" -u - "$repo" "$model_dir" "$model_id" "$host" "$port" \
    "$max_context" "$max_memory" "$max_cache_disk" "$queue_size" "$kv_format" <<'PY'
import os
import socket
import sys
import tempfile
from pathlib import Path

repo, model_dir, model_id, host, port, context, memory, cache_disk, queue, kv_format = sys.argv[1:]
repo = Path(repo)
sys.path.insert(0, str(repo))
from install import assembly, hub, models, upstream

# Check the address before the first-run draft download.
with socket.socket() as listener:
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((host, int(port)))

selection = models.Selection.of(repo / "install/models-local", model_id, language_only=True)
if models.installation_kind(selection.link) is None:
    with tempfile.TemporaryDirectory(prefix="splash-model-check-") as scratch:
        upstream.inspect_target(hub.Repository.local_directory(Path(model_dir)), selection.variant, selection.language_only, Path(scratch))
    draft = hub.Repository.resolve("incoai/Qwen3.6-35B-A3B-DFlash2", revision="51ef7b6923ad6c14cb1bb41c37a9041446496ab6")
    upstream._install(selection, hub.Repository.local_directory(Path(model_dir)), None, draft=draft)
else:
    record = assembly.verify(selection.link)
    if record["sources"]["target"] != {"repo": model_dir, "revision": None}:
        raise models.ModelError("Prepared target does not match the local Ornith directory")

# Hold the verified assembly until the server exits, as the official launcher does.
root, record_lock = assembly.hold(selection.link, selection.models_root)
if record_lock is not None:
    os.set_inheritable(record_lock.fileno(), True)
print(f"Ornith 1.5 Splash: http://127.0.0.1:{port}/v1 (WebUI disabled; listening on {host})", flush=True)
command = [
    str(repo / ".venv/bin/python"), "-u", "-m", "server.server",
    str(root), "--tokenizer", str(root / "tokenizer"),
    "--model", model_id, "--served-model-name", "splash_" + model_id,
    "--binary", str(repo / "build/splash"), "--host", host, "--port", port,
    "--max-context", context, "--max-memory", memory, "--max-cache-disk", cache_disk,
    "--queue-size", queue, "--kv-format", kv_format, "--no-webui",
]
os.chdir(repo)
os.execv(command[0], command)
PY
