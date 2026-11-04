"""Owned process groups, persistent identity, bounded startup, and cleanup."""

import os
import signal
import socket
import subprocess
import time
import uuid
from pathlib import Path
from vibeforge.contracts.event import boot_id
from vibeforge.engine.client import EngineClient
from vibeforge.storage.atomic import read_json, write_json


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
        command = Path(f"/proc/{record['pid']}/cmdline").read_bytes().decode(errors="replace")
        if record["token"] not in command or os.getpgid(record["pid"]) != record["pid"]:
            raise RuntimeError("Refusing to signal a process without the recorded ownership token")
        os.killpg(record["pid"], signal.SIGTERM)
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
        command = [self.python, "-u", "-m", "vibeforge.engine.server",
                   "--directory", str(directory), "--port", str(port), "--token", token]
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(self.paths.project / "src"), *self.extra_paths, env.get("PYTHONPATH", "")])
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["HF_HUB_OFFLINE"] = "1"
        env["TOKENIZERS_PARALLELISM"] = "false"
        env["CUDA_CACHE_PATH"] = str(self.paths.data / "cache" / "cuda")
        env["TRITON_CACHE_DIR"] = str(self.paths.data / "cache" / "triton")
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
                raise RuntimeError(f"Engine exited with code {self.process.returncode}; see {log_path}")
            try:
                ready = EngineClient(client.url, 1).health()
                if ready.get("instance_token") == token:
                    return client
            except (OSError, ValueError, RuntimeError):
                pass
            time.sleep(0.15)
        self.stop()
        raise TimeoutError(f"Engine startup deadline exceeded; see {log_path}")

    def stop(self):
        stop_record(self.record)
        if self.process is not None:
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass
