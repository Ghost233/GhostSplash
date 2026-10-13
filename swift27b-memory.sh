#!/bin/sh
set -eu
# Usage: sh swift27b-memory.sh SERVER_PID [OUTPUT.jsonl]
repo=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
exec "$repo/.venv/bin/python" -u - "$@" <<'PYTHON'
import ctypes
import datetime
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import urllib.request

if len(sys.argv) not in (2, 3) or not sys.argv[1].isdigit() or int(sys.argv[1]) <= 0:
    sys.exit("Usage: sh swift27b-memory.sh SERVER_PID [OUTPUT.jsonl]")
pid = int(sys.argv[1])
output = Path(sys.argv[2] if len(sys.argv) == 3 else f"swift27b-memory-{pid}.jsonl")
output.parent.mkdir(parents=True, exist_ok=True)

class Rusage(ctypes.Structure):
    # Darwin rusage_info_v2: resident size and physical footprint are bytes.
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64) for name in (
            "user_time", "system_time", "pkg_idle_wkups", "interrupt_wkups",
            "pageins", "wired_size", "resident_size", "phys_footprint",
            "proc_start_abstime", "proc_exit_abstime", "child_user_time",
            "child_system_time", "child_pkg_idle_wkups", "child_interrupt_wkups",
            "child_pageins", "child_elapsed_abstime", "diskio_bytesread",
            "diskio_byteswritten",
        )
    ]

libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
libproc.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
libproc.proc_pid_rusage.restype = ctypes.c_int

def run(*args):
    return subprocess.check_output(args, text=True, timeout=10)

def process_memory(process_pid):
    usage = Rusage()
    if libproc.proc_pid_rusage(process_pid, 2, ctypes.byref(usage)):
        raise OSError(ctypes.get_errno(), "proc_pid_rusage failed")
    return {"rss_bytes": usage.resident_size, "physical_footprint_bytes": usage.phys_footprint}

def sample():
    rows = [tuple(map(int, line.split())) for line in run("/bin/ps", "-axo", "pid=,ppid=").splitlines()]
    selected = {pid}
    while True:
        children = {child for child, parent in rows if parent in selected}
        if children <= selected:
            break
        selected.update(children)
    processes = []
    for child, parent in rows:
        if child in selected:
            item = {"pid": child, "ppid": parent}
            try:
                item.update(process_memory(child))
            except OSError as error:
                item["error"] = str(error)
            processes.append(item)
    vm = run("/usr/bin/vm_stat")
    page_size = int(re.search(r"page size of (\d+) bytes", vm).group(1))
    pages = {key.strip().strip('"'): int(value) for key, value in re.findall(r"^(.+?):\s+(\d+)\.", vm, re.M)}
    system = {"page_size_bytes": page_size, "vm_pages": pages}
    system["swap_usage"] = run("/usr/sbin/sysctl", "-n", "vm.swapusage").strip()
    result = {"processes": processes, "system": system}
    # Process sums may double-count shared mappings; keep per-process data too.
    result["rss_sum_bytes"] = sum(item.get("rss_bytes", 0) for item in processes)
    result["physical_footprint_sum_bytes"] = sum(item.get("physical_footprint_bytes", 0) for item in processes)
    result["process_measurement_complete"] = all("error" not in item for item in processes)
    try:
        with urllib.request.urlopen("http://127.0.0.1:8004/status", timeout=10) as response:
            status = json.load(response)
        if status.get("instance", {}).get("pid") != pid:
            raise ValueError("8004 belongs to a different server PID")
        # Includes engine allocation current/peak, budget, context and stale status.
        result["server_status"] = status
    except Exception as error:
        result["server_status_error"] = str(error)
    return result

signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
print(f"Memory sampler PID={os.getpid()}, server PID={pid}, interval=60s, log={output}", flush=True)
with output.open("a", buffering=1) as log:
    while True:
        started = time.monotonic()
        record = {"timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "server_pid": pid}
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            record["event"] = "server_exited"
            log.write(json.dumps(record) + "\n")
            break
        try:
            record.update(sample())
        except Exception as error:
            record["sample_error"] = str(error)
        log.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(record["timestamp"], "footprint_GiB=", round(record.get("physical_footprint_sum_bytes", 0) / 2**30, 3), flush=True)
        time.sleep(max(0, 60 - (time.monotonic() - started)))
PYTHON
