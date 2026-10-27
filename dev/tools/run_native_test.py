"""Run a native check with live output, heartbeats and verifiable result reuse."""

import argparse
import errno
import hashlib
import json
import math
import os
import platform
import pty
import selectors
import shlex
import shutil
import signal
import subprocess
import sys
import termios
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def machine_identity():
    result = {"system": platform.platform(), "machine": platform.machine()}
    for name, command in (
        ("gpu", ["system_profiler", "SPDisplaysDataType", "-json"]),
        ("os_build", ["sw_vers", "-buildVersion"]),
        ("host", ["sysctl", "-n", "hw.model", "hw.memsize"]),
        ("metal_compiler", ["xcrun", "-sdk", "macosx", "metal", "--version"]),
    ):
        try:
            result[name] = subprocess.check_output(
                command, stderr=subprocess.PIPE, timeout=30, text=True
            ).strip()
            if name == "gpu":
                devices = json.loads(result[name])["SPDisplaysDataType"]
                if not devices or any(
                    not gpu.get("sppci_model") or not gpu.get("sppci_cores")
                    for gpu in devices
                ):
                    raise ValueError("GPU model/core count unavailable")
                fields = (
                    "_name",
                    "sppci_model",
                    "sppci_cores",
                    "sppci_bus",
                    "sppci_device_type",
                    "spdisplays_vendor",
                    "spdisplays_metal",
                    "spdisplays_mtlgpufamilysupport",
                )
                result[name] = json.dumps(
                    [{k: gpu[k] for k in fields if k in gpu} for gpu in devices],
                    sort_keys=True,
                )
        except (OSError, subprocess.SubprocessError, ValueError, KeyError) as error:
            result[name] = {"unavailable": str(error)}
    result["complete"] = all(
        isinstance(result[k], str)
        for k in ("gpu", "os_build", "host", "metal_compiler")
    )
    return result


def input_files(command, extra, root):
    paths = {Path(__file__), root / "Makefile", root / "dev/native.mk"}
    for directory in ("runtime", "dev/tests", "dev/tools", "dev/benchmarks"):
        paths.update(
            p
            for p in (root / directory).rglob("*")
            if p.is_file()
            and p.suffix
            in {
                ".c",
                ".cc",
                ".cpp",
                ".mm",
                ".h",
                ".hpp",
                ".metal",
                ".py",
                ".json",
                ".bin",
                ".txt",
                ".sh",
            }
        )
    for name in (*command, *map(str, extra)):
        path = Path(name)
        if path.is_dir():
            paths.update(p for p in path.rglob("*") if p.is_file())
        elif path.is_file():
            paths.add(path)
    for path in (
        root / "build/engine/build-identity.json",
        root / "build/splash.config",
        root / "build/splash",
        Path(command[0] + ".config"),
    ):
        if path.is_file():
            paths.add(path)
    executable = shutil.which(command[0])
    if executable:
        paths.add(Path(executable))
    paths.update((root / "build").glob("*.metallib"))
    paths.update((root / "build/engine-tests").glob("*.metallib"))
    return sorted({p.resolve() for p in paths if p.is_file()})


def fingerprint(command, files, machine, *, timeout=None):
    inputs = {str(path): digest(path) for path in files}
    identity = {
        "command": command,
        "cwd": str(Path.cwd()),
        "inputs": inputs,
        "machine": machine,
        "timeout_seconds": timeout,
        "validation": {
            k: v
            for k, v in os.environ.items()
            if k.startswith(("MTL_", "METAL_"))
            or k in {"ASAN_OPTIONS", "UBSAN_OPTIONS", "TSAN_OPTIONS"}
        },
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return key, identity


def run(
    command, *, directory, extra=(), reuse=False, heartbeat=15, timeout=None, root=ROOT
):
    if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
        raise ValueError("timeout must be a finite positive number of seconds")
    directory.mkdir(parents=True, exist_ok=True)
    machine = machine_identity()
    files = input_files(command, extra, root)
    key, identity = fingerprint(command, files, machine, timeout=timeout)
    cached = directory / (key + ".passed.json")
    if reuse and machine["complete"] and cached.is_file():
        try:
            record = json.loads(cached.read_text())
            log = Path(record["log"])
            if (
                record["fingerprint"] == key
                and record["passed"]
                and record["exit_code"] == 0
                and record["inputs_unchanged"]
                and log.is_file()
                and digest(log) == record["log_sha256"]
            ):
                print(
                    f"REUSE {shlex.join(command)} result={record['result_file']}",
                    flush=True,
                )
                return record
        except (OSError, ValueError, KeyError, TypeError):
            pass
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    base = directory / f"{Path(command[0]).name}-{key[:12]}-{stamp}"
    log_path = base.with_name(base.name + ".log")
    result_path = base.with_name(base.name + ".json")
    record = {
        "command": command,
        "identity": identity,
        "fingerprint": key,
        "start_utc": datetime.now(UTC).isoformat(),
        "exit_code": None,
        "timeout_seconds": timeout,
        "log": str(log_path.resolve()),
        "result_file": str(result_path.resolve()),
        "inputs_unchanged": False,
    }
    try:
        record["source_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        record["source_dirty"] = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except subprocess.SubprocessError:
        record["source_commit"] = None
    print(f"START {shlex.join(command)} log={log_path}", flush=True)
    started = time.monotonic()
    deadline = None if timeout is None else started + timeout
    last_output = started
    next_heartbeat = started + heartbeat
    process = None
    master, slave = pty.openpty()
    attributes = termios.tcgetattr(slave)
    attributes[1] &= ~termios.ONLCR
    termios.tcsetattr(slave, termios.TCSANOW, attributes)
    try:
        with log_path.open("wb") as log, selectors.DefaultSelector() as selector:
            process = subprocess.Popen(
                command, stdout=slave, stderr=slave, start_new_session=True
            )
            record["pid"] = process.pid
            os.close(slave)
            slave = None
            selector.register(master, selectors.EVENT_READ)
            while selector.get_map() or process.poll() is None:
                now = time.monotonic()
                wait = min(0.2, max(0, next_heartbeat - now))
                if deadline is not None:
                    if now >= deadline:
                        raise subprocess.TimeoutExpired(command, timeout)
                    wait = min(wait, deadline - now)
                for event, _ in selector.select(timeout=wait):
                    try:
                        data = os.read(event.fd, 65536)
                    except OSError as error:
                        if error.errno != errno.EIO:
                            raise
                        data = b""
                    if data:
                        log.write(data)
                        log.flush()
                        sys.stdout.buffer.write(data)
                        sys.stdout.buffer.flush()
                        last_output = time.monotonic()
                    else:
                        selector.unregister(master)
                now = time.monotonic()
                if now >= next_heartbeat and process.poll() is None:
                    print(
                        f"HEARTBEAT pid={process.pid} elapsed={now - started:.1f}s "
                        f"silent={now - last_output:.1f}s command={command[0]}",
                        flush=True,
                    )
                    next_heartbeat = now + heartbeat
            record["exit_code"] = process.wait()
    except BaseException as error:
        record["tool_error"] = repr(error)
        if process is not None:
            try:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                cleanup_until = time.monotonic() + 5
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
                # The parent's exit and PTY EOF do not prove its descendants
                # are gone: a descendant may have redirected its output.
                time.sleep(max(0, cleanup_until - time.monotonic()))
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired) as cleanup_error:
                record["cleanup_error"] = repr(cleanup_error)
        record["exit_code"] = process.returncode if process else None
    finally:
        os.close(master)
        if slave is not None:
            os.close(slave)
    record["elapsed_seconds"] = time.monotonic() - started
    record["end_utc"] = datetime.now(UTC).isoformat()
    try:
        record["inputs_unchanged"] = (
            fingerprint(command, files, machine, timeout=timeout)[0] == key
        )
    except OSError as error:
        record["input_error"] = str(error)
    record["log_sha256"] = digest(log_path)
    record["passed"] = (
        record["exit_code"] == 0
        and record["inputs_unchanged"]
        and "tool_error" not in record
    )
    result_path.write_text(json.dumps(record, indent=2) + "\n")
    if record["passed"] and machine["complete"]:
        cached.write_text(json.dumps(record, indent=2) + "\n")
    print(
        f"END exit={record['exit_code']} passed={record['passed']} "
        f"elapsed={record['elapsed_seconds']:.3f}s result={result_path}",
        flush=True,
    )
    return record


def main():
    def interrupted(_signal, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "build/test-runs")
    parser.add_argument("--input", action="append", type=Path, default=[])
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="reuse only a matching, complete successful result",
    )
    parser.add_argument("--heartbeat", type=float, default=15)
    parser.add_argument(
        "--timeout",
        type=float,
        help="stop the owned command group after this many wall-clock seconds",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or not math.isfinite(args.heartbeat) or args.heartbeat <= 0:
        parser.error("a command and a positive heartbeat interval are required")
    if args.timeout is not None and (
        not math.isfinite(args.timeout) or args.timeout <= 0
    ):
        parser.error("--timeout requires a finite positive number of seconds")
    result = run(
        command,
        directory=args.directory,
        extra=args.input,
        reuse=args.reuse,
        heartbeat=args.heartbeat,
        timeout=args.timeout,
    )
    code = result["exit_code"]
    return (
        0
        if result["passed"]
        else (128 - code if code is not None and code < 0 else code or 1)
    )


if __name__ == "__main__":
    raise SystemExit(main())
