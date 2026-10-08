#!/bin/sh
set -eu

repo=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
driver=/Users/ghost233/code/zx-bench/scripts/bench-context-speed.py
output="$repo/build/context-speed-coder390-$(date +%Y%m%d-%H%M%S)"

# Same exact-token protocol as the original 27B, Tiel and Whittle runs.
exec "$repo/.venv/bin/python" - "$driver" \
    --url http://192.168.64.1:8007 \
    --output "$output" \
    --concurrency 1 2 3 4 --decode 512 \
    --contexts 8000 16000 32000 64000 "$@" <<'PY'
import importlib.util
import json
import sys
import time

driver = sys.argv[1]
sys.argv = [driver, *sys.argv[2:]]
spec = importlib.util.spec_from_file_location("context_speed", driver)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


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
