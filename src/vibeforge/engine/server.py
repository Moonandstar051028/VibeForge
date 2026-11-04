"""Small local HTTP transport; inference dependencies stay inside the backend."""

import argparse
import json
import os
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from vibeforge.engine.backend import load_backend
from vibeforge.engine.sandbox import restrict_writes, prepare_device
from vibeforge.storage.atomic import read_json
from pathlib import Path


def serve(directory, port, token):
    write_root = os.environ.get("VIBEFORGE_ENGINE_WRITE_ROOT")
    if not write_root:
        raise RuntimeError("A managed engine requires VIBEFORGE_ENGINE_WRITE_ROOT")
    device = read_json(Path(directory) / 'engine.json').get('device', 'cpu')
    prepared = prepare_device(device)
    sandbox = restrict_writes([write_root])
    print(json.dumps({"event": "write_confinement_enabled", 'device_setup': prepared, **sandbox}), flush=True)
    backend, config = load_backend(directory)
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, value):
            data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self.reply(200, {"ready": True, "instance_token": token, "backend": config.get("backend")})
            else:
                self.reply(404, {"error": "not_found"})

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4 * 1024 * 1024:
                    raise ValueError("Invalid request size")
                request = json.loads(self.rfile.read(length))
                operation = self.path.lstrip("/")
                if operation not in ("generate", "chat", "logits"):
                    self.reply(404, {"error": "not_found"})
                    return
                with lock:
                    result = getattr(backend, operation)(**request)
                self.reply(200, result)
            except Exception as error:
                traceback.print_exc()
                self.reply(400, {"error": type(error).__name__, "message": str(error)})

        def log_message(self, format, *args):
            print(format % args, flush=True)

    print(json.dumps({"event": "engine_ready", "port": port, "backend": config.get("backend")}), flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--token", required=True)
    arguments = parser.parse_args()
    serve(arguments.directory, arguments.port, arguments.token)


if __name__ == "__main__":
    main()
