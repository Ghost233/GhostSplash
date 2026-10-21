#!/bin/sh
set -eu

# Local VAL APEX I MiniPlus V3, text only, reusing the compatible converted Qwen3.8-27B DFlash2.
model_dir=/Users/ghost233/.lmstudio/models/IsValorum/Qwen3.8-27B-VAL-APEX-I-MiniPlus-V3-GGUF
draft_gguf=/Users/ghost233/.lmstudio/models/IsValorum/Qwen3.8-27B-EfficientThink-VAL-APEX-I-MiniPlus-V2.1-GGUF/dflash2-qwen38-27b-Q4_K_M.gguf
model_id=IsValorum/Qwen3.8-27B-VAL-APEX-I-MiniPlus-V3-GGUF:V3
host=0.0.0.0
port=8008
max_context=262144
max_memory=40G
max_cache_disk=50G
queue_size=1
repo=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
python="$repo/.venv/bin/python"
binary="$repo/build/splash"
draft_dir="$repo/install/models-local/drafts/miniplus-v3-dflash2-q4km/bf16"
if [ ! -x "$python" ]; then
    echo "Missing: $python" >&2
    exit 1
fi

unset SPLASH_API_KEY
exec "$python" -u - "$repo" "$model_dir" "$draft_dir" "$draft_gguf" "$model_id" "$host" "$port" \
    "$max_context" "$max_memory" "$max_cache_disk" "$queue_size" "q4h" "$binary" "$@" <<'PY'
import argparse
import json
import os
import socket
import sys
from pathlib import Path

repo, model_dir, draft_dir, draft_gguf, model_id, host, port, context, memory, cache_disk, queue, kv_format, binary = sys.argv[1:14]
repo = Path(repo)
sys.path.insert(0, str(repo))
from install import models, paths
from install.local import prepare_local_model
from server import serve_options

parser = argparse.ArgumentParser(
    prog="qwen38miniplusv3splash.sh", description="Qwen3.8 27B VAL APEX I MiniPlus V3 with the converted Qwen3.8-27B DFlash2 draft",
    epilog="MiniPlus V3 defaults: context=262144, KV=q4h, queue=1, memory=40G, "
           "lookup=on, compact=off (explicit public ranking required), fast-topk=off, ANE power recovery/reference=off.",
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
args = parser.parse_args(sys.argv[14:])
serve_options.check_serve_arguments(parser, args)
if not 0 <= args.port <= 65535:
    parser.error("--port must be in [0, 65535]")
args.binary = str(Path(args.binary).expanduser().absolute())
if not Path(args.binary).is_file():
    parser.error("native binary is missing: " + args.binary)
host, port = args.host, args.port
print(f"Main model: {Path(model_dir) / 'Qwen3.8-27B-VAL-APEX-I-MiniPlus-V3.gguf'}", flush=True)
print(f"MiniPlus V3 27B: context={args.max_context}, KV={args.kv_format}, "
      f"queue={args.queue_size}, memory={args.max_memory}", flush=True)

# Check the address before preparing the local model assembly.
paths.BINARY = Path(args.binary)
with socket.socket() as listener:
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((host, int(port)))

draft_path = Path(draft_dir)
for name in ("config.json", "model.safetensors", "conversion.json"):
    if not (draft_path / name).is_file():
        parser.error("converted draft is missing: " + str(draft_path / name)
                     + "; prepare it with python -m dev.tools.dequantize_dflash2_gguf")
conversion = json.loads((draft_path / "conversion.json").read_text())
if conversion["source_gguf"] != draft_gguf:
    raise models.ModelError("Converted draft does not match the selected Qwen3.8-27B DFlash2 GGUF")
root = prepare_local_model(
    Path(model_dir) / "Qwen3.8-27B-VAL-APEX-I-MiniPlus-V3.gguf", draft=draft_dir, language_only=True,
    models_dir=repo / "install/models-local",
)
print(f"MiniPlus V3 Splash: http://127.0.0.1:{port}/v1 (WebUI disabled; listening on {host})", flush=True)
print(f"Draft: {draft_dir} (compatible Qwen3.8-27B Q4_K_M GGUF converted to BF16)", flush=True)
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
