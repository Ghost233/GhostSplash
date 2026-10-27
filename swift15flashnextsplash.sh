#!/bin/sh
set -eu

# Flash Next: expert/PLE shared pool, separate KV budget, text only.
model_dir=/Users/ghost233/.lmstudio/models/SC117/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF
draft_gguf=/Users/ghost233/.lmstudio/models/incoai/Qwen3.8-Flash-Next-DFlash2-GGUF/Qwen3.8-Flash-Next-DFlash2-Q8_0.gguf
model_id=SC117/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF:IQ3_S
host=0.0.0.0
port=8018
max_context=262144
max_memory=30G
max_cache_disk=50G
queue_size=1
repo=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
python="$repo/.venv/bin/python"
binary="$repo/build/splash"
draft_dir="$repo/install/models-local/drafts/flash-next-dflash2-q8_0/bf16"
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
    prog="swift15flashnextsplash.sh",
    description="Swift 1.5 Flash Next abliterated IQ3_S with its converted Q8_0 DFlash2",
    epilog="Defaults: context=262144, KV=q4h, queue=1, memory=20G, "
           "expert/PLE pool=10G with both stores auto, adaptive policy=adaptive. "
           "--ple-memory off loads the WHOLE PLE table into RAM. "
           "Lookup, fast-topk, compact and the 4B FFN experiment are off.",
)
parser.add_argument("--port", type=int, default=int(port))
parser.add_argument("--binary", default=binary)
serve_options.add_serve_arguments(parser)
parser.set_defaults(
    host=host, max_context=serve_options.parse_max_context(context),
    max_memory=serve_options.parse_max_memory(memory),
    max_cache_disk=serve_options.parse_max_cache_disk(cache_disk),
    queue_size=int(queue), kv_format=kv_format, no_webui=True,
    ple_memory=None, expert_memory=None,
    expert_ple_memory=serve_options.parse_max_memory("10G"),
    expert_ple_policy="adaptive",
    disable_ane=True, prompt_lookup="off", compact_draft_head="off", fast_topk="off",
    experimental_quantized_ffn="off",
)
args = parser.parse_args(sys.argv[14:])
serve_options.check_serve_arguments(parser, args)
if not 0 <= args.port <= 65535:
    parser.error("--port must be in [0, 65535]")
if args.kv_format not in ("int8", "bf16", "q4h"):
    parser.error("Flash Next sparse KV supports int8, bf16 or q4h, not Q4_0 formats")
args.binary = str(Path(args.binary).expanduser().absolute())
if not Path(args.binary).is_file():
    parser.error("native binary is missing: " + args.binary)
paths.BINARY = Path(args.binary)
with socket.socket() as listener:
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((args.host, args.port))

draft_path = Path(draft_dir)
for name in ("config.json", "model.safetensors", "conversion.json"):
    if not (draft_path / name).is_file():
        parser.error("converted draft is missing: " + str(draft_path / name)
                     + "; prepare it with python -m dev.tools.dequantize_dflash2_gguf")
conversion = json.loads((draft_path / "conversion.json").read_text())
if conversion["source_gguf"] != draft_gguf:
    raise models.ModelError("Converted draft does not match the selected Flash Next DFlash2 GGUF")
target = Path(model_dir) / "Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf"
print(f"Main model: {target} (two GGUF parts)", flush=True)
print(f"Draft: {draft_path} (converted from {draft_gguf})", flush=True)
print(f"Flash Next: context={args.max_context}, KV={args.kv_format}, queue={args.queue_size}, "
      f"memory={args.max_memory}; PLE cache={args.ple_memory}; expert memory={args.expert_memory}", flush=True)
root = prepare_local_model(
    target, draft=draft_path, language_only=True, models_dir=repo / "install/models-local",
)
print(f"Swift Flash Next Splash: http://127.0.0.1:{args.port}/v1 "
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
