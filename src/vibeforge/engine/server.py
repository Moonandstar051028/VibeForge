"""Small local HTTP transport; inference dependencies stay inside the backend."""

import argparse
import json
import os
import threading
import time
import traceback
import uuid
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
                self.reply(200, {"ready": True, "instance_token": token, "backend": config.get("backend"),
                                 "capabilities": [name for name in ("generate", "logits", "chat", "chat_completion")
                                                  if callable(getattr(backend, name, None))],
                                 "precision": config.get("precision"), "device": config.get("device", "cpu")})
            else:
                self.reply(404, {"error": "not_found"})

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4 * 1024 * 1024:
                    raise ValueError("Invalid request size")
                request = json.loads(self.rfile.read(length))
                if self.path == "/v1/chat/completions":
                    with lock:
                        result = backend.chat_completion(**request)
                    self.chat_completion_reply(request, result)
                    return
                operation = self.path.lstrip("/")
                if operation not in ("generate", "chat", "logits"):
                    self.reply(404, {"error": "not_found"})
                    return
                with lock:
                    result = getattr(backend, operation)(**request)
                self.reply(200, result)
            except Exception as error:
                traceback.print_exc()
                value = {"error": {"type": type(error).__name__, "message": str(error)}} \
                    if self.path == "/v1/chat/completions" else {
                        "error": type(error).__name__, "message": str(error)
                    }
                self.reply(400, value)

        def chat_completion_reply(self, request, result):
            completion_id = "chatcmpl-" + uuid.uuid4().hex
            created = int(time.time())
            calls = result.get("tool_calls") or []
            message = {"role": "assistant", "content": result.get("text") or None}
            if calls:
                message["tool_calls"] = [
                    {
                        "id": call.get("id") or f"call_{index}_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": call["arguments"] if isinstance(call["arguments"], str)
                            else json.dumps(call["arguments"], ensure_ascii=False),
                        },
                    }
                    for index, call in enumerate(calls)
                ]
            response = {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": request.get("model", config.get("model_name", "vibeforge-local")),
                "choices": [{"index": 0, "message": message,
                             "finish_reason": "tool_calls" if calls else "stop"}],
                "usage": result.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}),
            }
            if not request.get("stream", False):
                self.reply(200, response)
                return

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()

            def event(chunk):
                data = json.dumps(chunk, ensure_ascii=False, allow_nan=False).encode()
                self.wfile.write(b"data: " + data + b"\n\n")
                self.wfile.flush()

            delta = {}
            if message["content"] is not None:
                delta["content"] = message["content"]
            if calls:
                delta["tool_calls"] = [
                    {"index": index, **call}
                    for index, call in enumerate(message["tool_calls"])
                ]
            if delta:
                event({"id": completion_id, "object": "chat.completion.chunk", "created": created,
                       "model": response["model"], "choices": [{"index": 0, "delta": delta,
                                                                    "finish_reason": None}]})
            event({"id": completion_id, "object": "chat.completion.chunk", "created": created,
                   "model": response["model"], "choices": [{"index": 0, "delta": {},
                                                                "finish_reason": response["choices"][0]["finish_reason"]}]})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

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
