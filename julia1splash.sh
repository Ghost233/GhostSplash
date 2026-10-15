#!/bin/sh
set -eu

# Local encoder decision model. No draft or text-generation resources are enabled.
repo=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
model_dir=/Users/ghost233/.lmstudio/models/ggml-org/Julia-1-GGUF
model_id=ggml-org/Julia-1-GGUF:Q8_0
host=127.0.0.1
port=8014
max_context=8192
max_memory=4G
max_cache_disk=0
queue_size=1
exec "$repo/.venv/bin/python" -u - "$repo" "$model_dir" "$model_id" "$host" "$port" \
    "$max_context" "$max_memory" "$max_cache_disk" "$queue_size" "bf16" "$repo/build/decision-modernbert-20261015/splash" "$@" <<'PY'
import argparse
import os
import socket
import sys
from pathlib import Path

repo, model_dir, model_id, host, port, context, memory, cache_disk, queue, kv_format, binary = sys.argv[1:12]
repo = Path(repo)
sys.path.insert(0, str(repo))
from install import assembly, hub, models, paths, upstream
from server import serve_options

parser = argparse.ArgumentParser(prog="julia1splash.sh", description="Julia-1 Q8_0 decision scoring via /v1/systemone")
parser.add_argument("--port", type=int, default=int(port))
parser.add_argument("--binary", default=binary)
parser.add_argument("--ane", choices=("off",), default="off")
serve_options.add_serve_arguments(parser)
parser.set_defaults(host=host, max_context=int(context), max_memory=serve_options.parse_max_memory(memory),
                    max_cache_disk=serve_options.parse_max_cache_disk(cache_disk), queue_size=int(queue),
                    kv_format=kv_format, prompt_lookup="off", compact_draft_head="off", fast_topk="off", no_webui=True)
args = parser.parse_args(sys.argv[12:])
args.disable_ane = args.disable_ane or args.ane == "off"
serve_options.check_serve_arguments(parser, args)
if any(value == "on" for value in (args.prompt_lookup, args.compact_draft_head, args.fast_topk)):
    parser.error("Kev has no decode; lookup, compact and fast-topk must be off")
if not 0 <= args.port <= 65535:
    parser.error("--port must be in [0, 65535]")
args.binary = str(Path(args.binary).expanduser().absolute())
if not Path(args.binary).is_file():
    parser.error("native binary is missing: " + args.binary)
paths.BINARY = Path(args.binary)
with socket.socket() as listener:
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((args.host, args.port))
selection = models.Selection.of(repo / "install/models-local", model_id, language_only=True)
if models.installation_kind(selection.link) is None:
    upstream._install(selection, hub.Repository.local_directory(Path(model_dir)), None)
record = assembly.verify(selection.link)
if record.get("family") != "Julia-1":
    raise models.ModelError("Selected binary does not support the encoder decision head")
if record["sources"] != {"target": {"repo": model_dir, "revision": None}, "draft": None}:
    raise models.ModelError("Prepared Kev target does not match the local GGUF without a draft")
root, record_lock = assembly.hold(selection.link, selection.models_root)
if record_lock is not None:
    os.set_inheritable(record_lock.fileno(), True)
print(f"Julia-1: context={args.max_context}, KV=none, queue={args.queue_size}, memory={args.max_memory}", flush=True)
print(f"Decision head=ModernBERT typed option scores; drafter=off; compact=off; lookup=off; fast-topk=off; ANE={'off' if args.disable_ane else 'on'}", flush=True)
print(f"System One: http://{args.host}:{args.port}/v1/systemone", flush=True)
command = [str(repo / ".venv/bin/python"), "-u", "-m", "server.server", str(root),
           "--tokenizer", str(root / "tokenizer"), "--model", model_id,
           "--binary", args.binary, "--host", args.host, "--port", str(args.port),
           *serve_options.serve_argv(args)]
os.chdir(repo)
os.execv(command[0], command)
PY
