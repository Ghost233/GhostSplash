#!/bin/sh
set -eu

repo=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
driver=/Users/ghost233/code/zx-bench/scripts/bench-context-speed.py
output="$repo/build/context-speed-coder390-$(date +%Y%m%d-%H%M%S)"

# --kv-format checks the running service; use the same option in its launcher.
# Same exact-token protocol as the original 27B, Tiel and Whittle runs.
exec "$repo/.venv/bin/python" - "$driver" \
    --url http://127.0.0.1:8007 \
    --output "$output" \
    --concurrency 1 2 3 4 --decode 512 \
    --contexts 8000 16000 32000 64000 "$@" <<'PY'
import argparse
import importlib.util
import json
import sys
import time

driver = sys.argv[1]
format_parser = argparse.ArgumentParser(add_help=False)
format_parser.add_argument(
    "--kv-format", choices=("int8", "bf16", "q4_0", "q4_0_unrotated", "q4h"), default="int8"
)
format_args, bench_args = format_parser.parse_known_args(sys.argv[2:])
sys.argv = [driver, *bench_args]
spec = importlib.util.spec_from_file_location("context_speed", driver)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


read_status = bench.Client.json


def checked_status(client, path, *args, **kwargs):
    result = read_status(client, path, *args, **kwargs)
    if path == "/status":
        actual = result.get("identity", {}).get("kv", {}).get("format")
        if actual != format_args.kv_format:
            raise RuntimeError(
                f"KV format mismatch: requested {format_args.kv_format}, server reports {actual}. "
                "Start the model launcher with the same --kv-format before benchmarking."
            )
    return result


bench.Client.json = checked_status


def assert_platform_idle():
    # Read the large history response in chunks; still require complete JSON.
    for attempt in range(3):
        try:
            with bench.urlopen("http://127.0.0.1:3001/api/runs", timeout=30) as response:
                payload = b"".join(iter(lambda: response.read(65536), b""))
                runs = json.loads(payload).get("data", [])
            break
        except (bench.IncompleteRead, TimeoutError, ValueError) as error:
            if attempt == 2:
                raise RuntimeError("无法完整读取评测平台状态，未请求模型；请稍后续跑") from error
            time.sleep(0.5)
        except bench.URLError:
            return
    if any(run.get("status") == "running" for run in runs):
        raise RuntimeError("ZxBench 评测仍在运行；请先暂停并等引擎空闲，以免争用 GPU")


bench.assert_platform_idle = assert_platform_idle
try:
    bench.main()
except (Exception, KeyboardInterrupt) as error:
    print(f"测试停止：{str(error) or '用户中断'}", file=sys.stderr)
    sys.exit(1)
PY
