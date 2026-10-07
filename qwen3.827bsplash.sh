#!/bin/sh
set -eu

model_dir=/Users/ghost233/.lmstudio/models/incoai/Qwen3.8-27B-Splash
model_id=incoai/Qwen3.8-27B-Splash
host=192.168.64.1
port=8001
max_context=262144
max_memory=40G
max_cache_disk=50G
queue_size=4

repo=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
python="$repo/.venv/bin/python"
server="$repo/server/server.py"
binary="$repo/build/splash"
url="http://$host:$port"

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

# Open the built-in WebUI after this server finishes loading the model.
launcher_pid=$$
(
    attempts=0
    while [ "$attempts" -lt 300 ] && kill -0 "$launcher_pid" 2>/dev/null; do
        if curl --noproxy '*' -fsS --max-time 2 "$url/ready" >/dev/null 2>&1; then
            open "$url"
            exit
        fi
        attempts=$((attempts + 1))
        sleep 1
    done
) &

unset SPLASH_API_KEY
echo "Starting $model_id from $model_dir"
echo "WebUI: $url"
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
    --queue-size "$queue_size"
