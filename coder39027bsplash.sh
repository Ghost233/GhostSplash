#!/bin/sh
set -eu

# Local Coder390 target with the author's FP8 draft dequantized to BF16.
model_dir=/Users/ghost233/.lmstudio/models/nerkyor/Qwen3.8-27B-Coder390-EfficientThink-Opus5.5-GPT6Astra-Grok4.7-DSV4Pro-K3-SFT-RLOO-GGUF-NInfer
model_id=nerkyor/Qwen3.8-27B-Coder390-EfficientThink-Opus5.5-GPT6Astra-Grok4.7-DSV4Pro-K3-SFT-RLOO-GGUF-NInfer:Q4LynnStyle-MTP
host=0.0.0.0
port=8007
max_context=262144
max_memory=40G
max_cache_disk=50G
queue_size=4
repo=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
launcher_allow_baseline=0
. "$repo/dev/tools/launcher_kv.sh"
splash_parse_launcher_options "$@"
python="$repo/.venv/bin/python"
draft_dir="$repo/install/models-local/drafts/coder390-dflash2-fp8/bf16"

for path in "$model_dir/Qwen3.8-27B-Coder390-EfficientThink-Q4LynnStyle-MTP.gguf" \
    "$draft_dir/model.safetensors" "$draft_dir/config.json" "$draft_dir/conversion.json" \
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
exec "$python" -u - "$repo" "$model_dir" "$draft_dir" "$model_id" "$host" "$port" \
    "$max_context" "$max_memory" "$max_cache_disk" "$queue_size" "$kv_format" <<'PY'
import os
import socket
import sys
from pathlib import Path

repo, model_dir, draft_dir, model_id, host, port, context, memory, cache_disk, queue, kv_format = sys.argv[1:]
repo = Path(repo)
sys.path.insert(0, str(repo))
from install import assembly, hub, models, upstream

# Check the address before the first-run draft download.
with socket.socket() as listener:
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((host, int(port)))

selection = models.Selection.of(
    repo / "install/models-local", model_id,
    language_only=True, draft_model=draft_dir,
)
if models.installation_kind(selection.link) is None:
    upstream._install(selection, hub.Repository.local_directory(Path(model_dir)), None)
else:
    record = assembly.verify(selection.link)
    if record["sources"]["target"] != {"repo": model_dir, "revision": None}:
        raise models.ModelError("Prepared target does not match the local Coder390 directory")
    if record["sources"]["draft"] != {"repo": draft_dir, "revision": None}:
        raise models.ModelError("Prepared draft does not match the converted Coder390 draft")

# Hold the verified assembly until the server exits, as the official launcher does.
root, record_lock = assembly.hold(selection.link, selection.models_root)
if record_lock is not None:
    os.set_inheritable(record_lock.fileno(), True)
print(f"Coder390 Splash: http://127.0.0.1:{port}/v1 (WebUI disabled; listening on {host})", flush=True)
print(f"Draft: {draft_dir} (author's FP8 weights dequantized to BF16)", flush=True)
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
