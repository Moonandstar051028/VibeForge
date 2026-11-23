"""Owned process groups, persistent identity, bounded startup, and cleanup."""

import os
import signal
import socket
import subprocess
import time
import uuid
from pathlib import Path
from vf_runtime.contracts.event import boot_id
from vf_runtime.engine.client import EngineClient
from vf_runtime.storage.atomic import read_json, write_json


def startup_diagnostic(log_path):
    """Return a bounded raw log tail so the next agent sees the real failure."""
    try:
        with Path(log_path).open('rb') as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 4096))
            return '\n'.join(stream.read().decode('utf-8', errors='replace').splitlines()[-12:])[-2400:]
    except OSError:
        return ''


def process_identity(pid):
    try:
        # A process name may contain spaces and parentheses.
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return {"pid": pid, "start_ticks": fields[19], "boot_id": boot_id(), "state": fields[0]}
    except (OSError, IndexError):
        return None


def same_process(record):
    current = process_identity(record["pid"])
    return bool(current and current["state"] != "Z" and
                all(current[key] == record[key] for key in ("pid", "start_ticks", "boot_id")))


def stop_record(record_path):
    record_path = Path(record_path)
    if not record_path.exists():
        return False
    record = read_json(record_path)
    if same_process(record):
        # Exit may clear cmdline between the identity and ownership reads.
        # Recheck identity before treating that race as an ownership violation.
        deadline = time.monotonic() + 1
        while True:
            try:
                command = Path(f"/proc/{record['pid']}/cmdline").read_bytes().decode(errors="replace")
                owned = record["token"] in command and os.getpgid(record["pid"]) == record["pid"]
            except (FileNotFoundError, ProcessLookupError):
                command, owned = "", False
            if not same_process(record):
                record_path.unlink(missing_ok=True)
                return True
            if owned:
                break
            if command or time.monotonic() >= deadline:
                raise RuntimeError("Refusing to signal a process without the recorded ownership token")
            time.sleep(0.05)
        try:
            os.killpg(record["pid"], signal.SIGTERM)
        except ProcessLookupError:
            pass
        end = time.monotonic() + 5
        while same_process(record) and time.monotonic() < end:
            time.sleep(0.05)
        if same_process(record):
            os.killpg(record["pid"], signal.SIGKILL)
    record_path.unlink(missing_ok=True)
    return True


class EngineProcess:
    def __init__(self, paths, campaign_id, python, startup_timeout=300, request_timeout=120, extra_paths=()):
        self.paths, self.campaign_id, self.python = paths, campaign_id, python
        self.record = paths.data / "state" / f"{campaign_id}.engine.json"
        self.startup_timeout, self.request_timeout = startup_timeout, request_timeout
        self.process = None
        self.extra_paths = extra_paths

    def start(self, directory, log_path):
        stop_record(self.record)
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        token = uuid.uuid4().hex
        command = [self.python, "-u", "-m", "vf_runtime.engine.server",
                   "--directory", str(directory), "--port", str(port), "--token", token]
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(self.paths.project / "src"), *self.extra_paths, env.get("PYTHONPATH", "")])
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["HF_HUB_OFFLINE"] = "1"
        env["TOKENIZERS_PARALLELISM"] = "false"
        # The managed runtime cannot create PyTorch Inductor's multiprocessing semaphores.
        env["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
        env["CUDA_CACHE_PATH"] = str(self.paths.data / "cache" / "cuda")
        env["TRITON_CACHE_DIR"] = str(self.paths.data / "cache" / "triton")
        env["TORCH_EXTENSIONS_DIR"] = str(self.paths.data / "cache" / "extensions")
        env["TORCHINDUCTOR_CACHE_DIR"] = str(self.paths.data / "cache" / "inductor")
        scratch = self.paths.data / "cache" / "engine-scratch"
        scratch.mkdir(parents=True, exist_ok=True)
        env["TMPDIR"] = str(scratch)
        env["HF_HOME"] = str(self.paths.data / "cache" / "huggingface")
        env["TORCH_HOME"] = str(self.paths.data / "cache" / "torch")
        env["XDG_CACHE_HOME"] = str(self.paths.data / "cache" / "xdg")
        env["VIBEFORGE_ENGINE_WRITE_ROOT"] = str(self.paths.data / "cache")
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with Path(log_path).open("ab", buffering=0) as output:
            self.process = subprocess.Popen(command, cwd=directory, env=env,
                                            stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        identity = process_identity(self.process.pid)
        if not identity:
            raise RuntimeError(f"Engine exited before process registration; see {log_path}")
        write_json(self.record, {**identity, "token": token, "port": port, "directory": str(directory)})
        client = EngineClient(f"http://127.0.0.1:{port}", self.request_timeout)
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                self.stop()
                detail = startup_diagnostic(log_path)
                reason = detail.splitlines()[-1] if detail else 'No startup diagnostic available'
                raise RuntimeError(f"Engine exited with code {self.process.returncode}: {reason}; "
                                   f"see {log_path}\nStartup diagnostic (raw tail):\n{detail}")
            try:
                ready = EngineClient(client.url, 1).health()
                if ready.get("instance_token") == token:
                    client.startup_seconds = time.monotonic() - (deadline - self.startup_timeout)
                    return client
            except (OSError, ValueError, RuntimeError):
                pass
            time.sleep(0.15)
        self.stop()
        raise TimeoutError(f"Engine startup deadline exceeded; see {log_path}\n"
                           f"Startup diagnostic (raw tail):\n{startup_diagnostic(log_path)}")

    def stop(self):
        if self.process is not None:
            self.process.poll()  # Reap an already exited child before /proc inspection.
        stop_record(self.record)
        if self.process is not None:
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass
