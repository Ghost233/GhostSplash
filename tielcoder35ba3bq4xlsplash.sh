#!/bin/sh
set -eu

# Local Tiel Coder UD-Q4_K_XL, text only, with SSD-streamed experts.
model_dir=/Users/ghost233/.lmstudio/models/peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF
model_id=peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF:UD-Q4_K_XL
host=0.0.0.0
port=8019
max_context=262144
max_memory=30G
max_cache_disk=50G
queue_size=4
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
    prog="tielcoder35ba3bq4xlsplash.sh",
    description="Tiel Coder 35B-A3B UD-Q4_K_XL with a 15G streamed-expert store",
    epilog="Defaults: context=262144, KV=q4h, queue=4, memory=30G, "
           "expert-memory=15G, SSD cache=50G; lookup, fast-topk and compact off.",
)
parser.add_argument("--port", type=int, default=int(port))
parser.add_argument("--binary", default=binary)
serve_options.add_serve_arguments(parser)
parser.set_defaults(
    host=host, max_context=serve_options.parse_max_context(context),
    max_memory=serve_options.parse_max_memory(memory),
    max_cache_disk=serve_options.parse_max_cache_disk(cache_disk),
    queue_size=int(queue), kv_format=kv_format, no_webui=True,
    expert_memory=serve_options.parse_expert_memory("15G"),
    prompt_lookup="off", fast_topk="off", compact_draft_head="off",
    experimental_quantized_ffn="off",
)
args = parser.parse_args(sys.argv[12:])
serve_options.check_serve_arguments(parser, args)
if not 0 <= args.port <= 65535:
    parser.error("--port must be in [0, 65535]")
args.binary = str(Path(args.binary).expanduser().absolute())
if not Path(args.binary).is_file():
    parser.error("native binary is missing: " + args.binary)
paths.BINARY = Path(args.binary)
with socket.socket() as listener:
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((args.host, args.port))

from huggingface_hub import snapshot_download

draft_dir = snapshot_download(
    "incoai/Qwen3.6-35B-A3B-DFlash2",
    revision="51ef7b6923ad6c14cb1bb41c37a9041446496ab6",
    local_dir=repo / "install/models-local/drafts/tiel-dflash2/source",
    allow_patterns=["config.json", "*.safetensors", "*.safetensors.index.json"],
)
target = Path(model_dir) / "Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf"
root = prepare_local_model(
    target, draft=draft_dir, language_only=True, models_dir=repo / "install/models-local",
)
print(f"Main model: {target}", flush=True)
print(f"Draft: {draft_dir}", flush=True)
print(f"Tiel Coder UD-Q4_K_XL: context={args.max_context}, KV={args.kv_format}, "
      f"queue={args.queue_size}, memory={args.max_memory}, experts={args.expert_memory}", flush=True)
print(f"Tiel Coder Splash: http://127.0.0.1:{args.port}/v1 "
      f"(WebUI disabled; listening on {args.host})", flush=True)
command = [
    str(repo / ".venv/bin/python"), "-u", "-m", "server.server", str(root),
    "--model", model_id, "--served-model-name", "splash_" + model_id,
    "--binary", args.binary, "--port", str(args.port),
    *serve_options.serve_argv(args),
]
if args.api_key is not None:
    command.extend(("--api-key", args.api_key))
os.chdir(repo)
os.execv(command[0], command)
PY
