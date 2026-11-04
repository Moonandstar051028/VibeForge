"""Continuously streamed subprocess output and auditable command execution."""

import os
import signal
import subprocess
import time
from pathlib import Path
from vibeforge.storage.atomic import canonical_json


def append_record(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(canonical_json(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def run_command(argv, cwd, log_path, timeout=120, env=None, trace=None,
                process_record=None, token=None, heartbeat=None):
    start = time.monotonic()
    record = {"argv": list(argv), "cwd": str(cwd), "status": "started", "monotonic_time": start}
    if trace:
        append_record(trace, record)
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(log_path).open("ab", buffering=0) as output:
        process = subprocess.Popen(argv, cwd=cwd, stdout=output, stderr=subprocess.STDOUT,
                                   env=env, start_new_session=True)
        if process_record:
            from vibeforge.engine.process import process_identity
            from vibeforge.storage.atomic import write_json
            identity = process_identity(process.pid)
            if identity:
                write_json(process_record, {**identity, "token": token})
        try:
            deadline, next_heartbeat = time.monotonic() + timeout, 0
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    raise subprocess.TimeoutExpired(argv, timeout)
                if heartbeat and time.monotonic() >= next_heartbeat:
                    heartbeat()
                    next_heartbeat = time.monotonic() + 5
                time.sleep(0.1)
            code = process.returncode
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            code = 124
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
        finally:
            if process_record:
                Path(process_record).unlink(missing_ok=True)
    record.update(status="completed", exit_code=code, elapsed_seconds=time.monotonic() - start)
    if trace:
        append_record(trace, record)
    return record
