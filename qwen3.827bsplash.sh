#!/bin/sh
set -eu

model_dir=/Users/ghost233/.lmstudio/models/incoai/Qwen3.8-27B-Splash
model_id=incoai/Qwen3.8-27B-Splash
host=0.0.0.0
port=8001
max_context=262144
max_memory=40G
max_cache_disk=50G
queue_size=4

repo=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
python="$repo/.venv/bin/python"
server="$repo/server/server.py"
binary="$repo/build/splash"

for path in "$model_dir/manifest.json" "$model_dir/target" "$model_dir/draft" "$model_dir/tokenizer" "$python" "$server" "$binary"; do
    if [ ! -e "$path" ]; then
        echo "Missing required file or directory: $path" >&2
        exit 1
    fi
done

if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "Port $port is already in use; stop the existing service first." >&2
    exit 1
fi

unset SPLASH_API_KEY
echo "Starting $model_id from $model_dir"
echo "API: http://127.0.0.1:$port/v1 (WebUI disabled; listening on $host)"
cd "$repo"
exec "$python" -u -m server.server \
    "$model_dir" \
    --tokenizer "$model_dir/tokenizer" \
    --model "$model_id" \
    --served-model-name "splash_${model_id}" \
    --binary "$binary" \
    --host "$host" --port "$port" \
    --max-context "$max_context" --max-memory "$max_memory" \
    --max-cache-disk "$max_cache_disk" \
    --queue-size "$queue_size" --no-webui
