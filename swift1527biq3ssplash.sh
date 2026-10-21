#!/bin/sh
set -eu

# Swift 1.5 27B GSQ RCO mixed GGUF weights, text only, matching DFlash2 draft.
model_dir=/Users/ghost233/.lmstudio/models/ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF
model_id=ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF:IQ3_S-mtp
host=0.0.0.0
port=8004
max_context=262144
max_memory=40G
max_cache_disk=50G
queue_size=1
repo=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
python="$repo/.venv/bin/python"
binary="$repo/build/splash"
if [ ! -x "$python" ]; then
    echo "Missing: $python" >&2
    exit 1
fi

unset SPLASH_API_KEY
exec "$python" -u - "$repo" "$model_dir" "$model_id" "$host" "$port" \
    "$max_context" "$max_memory" "$max_cache_disk" "$queue_size" "q4h" "$binary" "$@" <<'PY'
import argparse
import os
import socket
import sys
from pathlib import Path

repo, model_dir, model_id, host, port, context, memory, cache_disk, queue, kv_format, binary = sys.argv[1:12]
repo = Path(repo)
sys.path.insert(0, str(repo))
from install import paths
from install.local import prepare_local_model
from server import serve_options

parser = argparse.ArgumentParser(
    prog="swift1527biq3ssplash.sh", description="Swift 1.5 27B GSQ RCO launcher",
    epilog="Swift defaults: context=262144, KV=q4h, queue=1, memory=40G, "
           "lookup=on, compact=off, fast-topk=off, ANE power recovery/reference=off.",
)
parser.add_argument("--port", type=int, default=int(port))
parser.add_argument("--binary", default=binary)
serve_options.add_serve_arguments(parser)
parser.set_defaults(
    host=host, max_context=serve_options.parse_max_context(context),
    max_memory=serve_options.parse_max_memory(memory),
    max_cache_disk=serve_options.parse_max_cache_disk(cache_disk),
    queue_size=int(queue), kv_format=kv_format, no_webui=True,
    prompt_lookup="on", compact_draft_head="off", fast_topk="off",
)
args = parser.parse_args(sys.argv[12:])
serve_options.check_serve_arguments(parser, args)
if not 0 <= args.port <= 65535:
    parser.error("--port must be in [0, 65535]")
args.binary = str(Path(args.binary).expanduser().absolute())
if not Path(args.binary).is_file():
    parser.error("native binary is missing: " + args.binary)
host, port = args.host, args.port
print(f"Swift 1.5 27B GSQ RCO: context={args.max_context}, KV={args.kv_format}, "
      f"queue={args.queue_size}, memory={args.max_memory}", flush=True)

# Check the address before the first-run draft download.
paths.BINARY = Path(args.binary)
with socket.socket() as listener:
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((host, int(port)))

from huggingface_hub import snapshot_download

# Record the default draft in the Hub cache before the offline preparation.
snapshot_download(
    "incoai/Qwen3.8-27B-DFlash2",
    allow_patterns=["config.json", "*.safetensors", "*.safetensors.index.json"],
)
root = prepare_local_model(
    Path(model_dir) / "Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf", language_only=True,
    models_dir=repo / "install/models-local",
)
print(f"Swift 1.5 Splash: http://127.0.0.1:{port}/v1 (WebUI disabled; listening on {host})", flush=True)
command = [
    str(repo / ".venv/bin/python"), "-u", "-m", "server.server",
    str(root),
    "--model", model_id, "--served-model-name", "splash_" + model_id,
    "--binary", args.binary, "--port", str(port),
    *serve_options.serve_argv(args),
]
if args.api_key is not None:
    command.extend(("--api-key", args.api_key))
os.chdir(repo)
os.execv(command[0], command)
PY
