"""Small local HTTP transport; inference dependencies stay inside the backend."""

import argparse
import json
import os
import threading
import time
import traceback
import uuid
from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from vibeforge.engine.backend import load_backend
from vibeforge.engine.precision import validate_loaded_quantization
from vibeforge.engine.observation import observe_call, profile_generate
from vibeforge.engine.runtime import runtime_manifest
from vibeforge.engine.responses import response_events, response_object, responses_to_chat
from vibeforge.engine.tokenization import count_tokens, load_control_tokenizer
from vibeforge.engine.session_telemetry import SessionTelemetry
from vibeforge.versioning.artifacts import source_hash
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
    control_tokenizer, tokenizer_identity = load_control_tokenizer(config, backend)
    runtime_precision = validate_loaded_quantization(backend, config)
    if config.get("request_scheduler", "serialized_lock") not in {
            "serialized_lock", "concurrent_requests", "microbatch"}:
        raise ValueError("Unknown request_scheduler")
    if config.get("request_scheduler") == "microbatch" and not getattr(backend, "MICROBATCH_SUPPORTED", False):
        raise ValueError("Candidate selected microbatch without implementing a microbatch backend")
    runtime = runtime_manifest(backend, config, runtime_precision)
    print(json.dumps({"event": "effective_runtime", **runtime}), flush=True)
    source_identity = source_hash(directory)
    if runtime_precision is not None:
        print(json.dumps({"event": "runtime_precision_verified", **runtime_precision}), flush=True)
    lock = threading.Lock()
    telemetry_sessions = SessionTelemetry()

    def inference_guard():
        return lock if config.get("request_scheduler", "serialized_lock") == "serialized_lock" else nullcontext()

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
                                                  if callable(getattr(backend, name, None))] +
                                                 (["tokenize"] if control_tokenizer is not None else []),
                                 "precision": config.get("precision"), "device": config.get("device", "cpu"),
                                 "max_context": config.get("max_context", 4096),
                                 "tokenizer": tokenizer_identity,
                                 "runtime_precision": runtime_precision, "runtime": runtime})
            else:
                self.reply(404, {"error": "not_found"})

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4 * 1024 * 1024:
                    raise ValueError("Invalid request size")
                request = json.loads(self.rfile.read(length))
                if self.path == "/implementation":
                    from vibeforge.engine.implementation import read_implementation
                    self.reply(200, {**read_implementation(backend, **request), "source_hash": source_identity})
                    return
                if self.path == "/tokenize":
                    with inference_guard():
                        self.reply(200, self.tokenize_reply(request))
                    return
                if self.path == "/telemetry/start":
                    self.reply(200, telemetry_sessions.start(
                        request["session_id"], request["max_requests"], request["retention_seconds"]))
                    return
                if self.path == "/telemetry/report":
                    self.reply(200, telemetry_sessions.report(request["session_id"]))
                    return
                if self.path == "/telemetry/finish":
                    self.reply(200, telemetry_sessions.finish(request["session_id"]))
                    return
                if self.path == "/v1/chat/completions":
                    queued = time.perf_counter()
                    with inference_guard():
                        result, observation = observe_call(backend, config, "chat_completion", request,
                                                          time.perf_counter() - queued)
                    print(json.dumps({"event": "request_observation", "source_hash": source_identity,
                                      **observation}, allow_nan=False), flush=True)
                    telemetry_sessions.record({"protocol": "chat_completions", **observation})
                    self.chat_completion_reply(request, result)
                    return
                if self.path == "/v1/responses":
                    arguments = responses_to_chat(request)
                    queued = time.perf_counter()
                    with inference_guard():
                        result, observation = observe_call(backend, config, "chat_completion", arguments,
                                                          time.perf_counter() - queued)
                    print(json.dumps({"event": "request_observation", "source_hash": source_identity,
                                      "protocol": "responses", **observation}, allow_nan=False), flush=True)
                    telemetry_sessions.record({"protocol": "responses", **observation})
                    self.responses_reply(request, result)
                    return
                operation = self.path.lstrip("/")
                if operation not in ("generate", "chat", "logits", "profile"):
                    self.reply(404, {"error": "not_found"})
                    return
                observation_mode = request.pop("observation_mode", "detailed")
                if observation_mode not in ("detailed", "wall"):
                    raise ValueError("Unknown observation mode")
                queued = time.perf_counter()
                with (lock if operation == "profile" else inference_guard()):
                    if operation == "profile":
                        result = profile_generate(backend, config, **request)
                        result["source_hash"] = source_identity
                        result["runtime"] = runtime
                    else:
                        result, observation = observe_call(backend, config, operation, request,
                                                          time.perf_counter() - queued,
                                                          detailed=observation_mode == "detailed")
                        result["host_observation"] = observation
                        print(json.dumps({"event": "request_observation", "source_hash": source_identity,
                                          **observation}, allow_nan=False), flush=True)
                        telemetry_sessions.record({"protocol": "native", **observation})
                self.reply(200, result)
            except Exception as error:
                traceback.print_exc()
                value = {"error": {"type": type(error).__name__, "message": str(error)}} \
                    if self.path in ("/v1/chat/completions", "/v1/responses") else {
                        "error": type(error).__name__, "message": str(error)
                    }
                self.reply(400, value)

        def tokenize_reply(self, request):
            if control_tokenizer is None:
                raise ValueError("The active backend does not expose its tokenizer")
            token_ids = count_tokens(control_tokenizer, request)
            return {"count": len(token_ids), "token_ids": token_ids if request.get("return_ids") else None,
                    "max_context": config.get("max_context", 4096),
                    "tokenizer": tokenizer_identity}

        def responses_reply(self, request, result):
            response = response_object(request, result)
            if not request.get("stream", False):
                self.reply(200, response)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            for value in response_events(response):
                kind = value["type"]
                data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
                self.wfile.write(b"event: " + kind.encode() + b"\n")
                self.wfile.write(b"data: " + data + b"\n\n")
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

        def chat_completion_reply(self, request, result):
            completion_id = "chatcmpl-" + uuid.uuid4().hex
            created = int(time.time())
            calls = result.get("tool_calls") or []
            message = {"role": "assistant", "content": result.get("text") or None}
            if result.get("reasoning_content"):
                message["reasoning_content"] = result["reasoning_content"]
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
                             "finish_reason": "tool_calls" if calls else result.get("finish_reason", "stop")}],
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
            if message.get("reasoning_content"):
                delta["reasoning_content"] = message["reasoning_content"]
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
