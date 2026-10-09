#!/bin/sh
set -eu

# Local Tiel APEX-I-MiniPlus-V2.1 target, text only.
model_dir=/Users/ghost233/.lmstudio/models/IsValorum/Tiel-Coder-35B-A3B-APEX-I-MiniPlus-V2.1-GGUF
chat_template="$model_dir/chat_template.jinja"
model_id=IsValorum/Tiel-Coder-35B-A3B-APEX-I-MiniPlus-V2.1-GGUF
host=0.0.0.0
port=8004
max_context=262144
max_memory=40G
max_cache_disk=50G
queue_size=4
repo=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
python="$repo/.venv/bin/python"

for path in "$model_dir/Tiel-Coder-35B-A3B.APEX-I-MiniPlus-V2.1.gguf" \
    "$chat_template" "$python" "$repo/server/server.py" "$repo/build/splash"; do
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
exec "$python" -u - "$repo" "$model_dir" "$model_id" "$chat_template" "$host" "$port" \
    "$max_context" "$max_memory" "$max_cache_disk" "$queue_size" <<'PY'
import hashlib
import os
import shutil
import socket
import sys
import tempfile
from pathlib import Path

repo, model_dir, model_id, chat_template, host, port, context, memory, cache_disk, queue = sys.argv[1:]
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
        raise models.ModelError("Prepared target does not match the local Tiel MiniPlus directory")

# Hold the verified assembly until the server exits, as the official launcher does.
root, record_lock = assembly.hold(selection.link, selection.models_root)
if record_lock is not None:
    os.set_inheritable(record_lock.fileno(), True)

# Keep the author's guidance, with Splash tool calls and later agent instructions.
template = Path(chat_template).read_text(encoding="utf-8")
replacements = (
    ("return a json object with function name and arguments", "return a function call with its parameters"),
    ('{"name": <function-name>, "arguments": <args-json-object>}',
     '<function=example_function_name>\n<parameter=example_parameter_name>\nvalue\n</parameter>\n</function>'),
    ("a JSON object nested within <tool_call></tool_call> XML tags.",
     "a <function=NAME> block with <parameter=NAME> values nested within <tool_call></tool_call> XML tags."),
    ('''{{- '
<tool_call>
{"name": "' + tool_call.name + '", "arguments": ' + (tool_call.arguments | tojson) + '}
</tool_call>' }}''',
     r'''{{- '\n<tool_call>\n<function=' + tool_call.name + '>\n' }}
                {%- for name, value in tool_call.arguments.items() %}
                    {{- '<parameter=' + name + '>\n' }}
                    {{- value if value is string else value | tojson }}
                    {{- '\n</parameter>\n' }}
                {%- endfor %}
                {{- '</function>\n</tool_call>' }}'''),
    ("{{- raise_exception('System message must be at the beginning.') }}",
     r"{{- '<|im_start|>system\n' + content + '<|im_end|>\n' }}"),
)
for old, new in replacements:
    if template.count(old) != 1:
        raise models.ModelError("Local chat template changed; review its Splash tool-call adaptation")
    template = template.replace(old, new, 1)
identity = hashlib.sha256((str(root) + template).encode()).hexdigest()
tokenizer_dir = repo / "build/tokenizer-overrides/tiel-miniplus" / identity
shutil.copytree(root / "tokenizer", tokenizer_dir, dirs_exist_ok=True)
(tokenizer_dir / "chat_template.jinja").write_text(template, encoding="utf-8")
print(f"Chat template: {chat_template} (Splash XML tool calls, later system messages)", flush=True)
print(f"Tiel MiniPlus Splash: http://127.0.0.1:{port}/v1 (WebUI disabled; listening on {host})", flush=True)
command = [
    str(repo / ".venv/bin/python"), "-u", "-m", "server.server",
    str(root), "--tokenizer", str(tokenizer_dir),
    "--model", model_id, "--served-model-name", "splash_" + model_id,
    "--binary", str(repo / "build/splash"), "--host", host, "--port", port,
    "--max-context", context, "--max-memory", memory, "--max-cache-disk", cache_disk,
    "--queue-size", queue, "--no-webui",
]
os.chdir(repo)
os.execv(command[0], command)
PY
