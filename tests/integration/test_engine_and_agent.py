import json
import shutil
import sys
import threading
from collections import UserDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from vibeforge.agent.runner import run_agent, smoke_agent
from vibeforge.engine.process import EngineProcess
from vibeforge.engine.tokenization import count_tokens
from vibeforge.evaluation.reference import capture_reference
from vibeforge.evaluation.correctness import evaluate_correctness


def test_chat_token_count_extracts_batch_encoding_input_ids():
    class Tokenizer:
        def apply_chat_template(self, messages, **arguments):
            return UserDict({"input_ids": list(range(37)), "attention_mask": [1] * 37})

    request = {"messages": [{"role": "user", "content": "x"}]}
    assert len(count_tokens(Tokenizer(), request)) == 37


def test_engine_http_reference_and_qwen_agent_protocol(paths, project):
    engine = EngineProcess(paths, "transport", sys.executable, startup_timeout=10)
    try:
        client = engine.start(project / "templates/engine/toy", paths.campaign("transport") / "engine.log")
        started = client.telemetry_start(session_id="integration-agent", max_requests=4,
                                         retention_seconds=60)
        assert started["request_count"] == 0
        smoke = smoke_agent(client)
        timing = client.generate(prompt="profile fixture", max_new_tokens=4, fixed_length=True)["host_observation"]
        assert timing["output_tokens"] == 4 and timing["backend_wall_seconds"] > 0
        assert timing["ttft_seconds"] is None and timing["output_tokens_per_second"] > 0
        profile = client.profile(prompt="profile fixture", max_new_tokens=4)
        assert profile["kind"] == "host_inference_profile" and profile["top_cpu_ops"]
        assert "cProfile" in profile["profiler"] and not profile["top_device_ops"]
        assert profile["source_hash"] and profile["normal_request"]["output_tokens"] == 4
        with pytest.raises(RuntimeError, match="between 1 and 128"):
            client.profile(prompt="profile fixture", max_new_tokens=129)
        assert smoke["passed"]
        assert smoke["framework"] == "Qwen-Agent"
        plain_tokens = client.tokenize(text="hello")
        chat_tokens = client.tokenize(
            messages=[{"role": "user", "content": "hello"}],
            tools=[{"type": "function", "function": {"name": "ping", "description": "Ping",
                    "parameters": {"type": "object", "properties": {}}}}],
        )
        assert plain_tokens["count"] == 5
        assert chat_tokens["count"] > plain_tokens["count"]
        response = client.request("v1/responses", {
            "model": "toy", "input": "verify responses",
            "tools": [{"type": "function", "name": "submit_candidate", "description": "Submit",
                       "parameters": {"type": "object", "properties": {}}}],
        })
        assert response["object"] == "response" and response["status"] == "completed"
        assert response["output"][0]["type"] == "function_call"
        live = client.telemetry_report("integration-agent")
        assert live["request_count"] >= 3
        assert live["operations"]["chat_completion"] >= 2
        assert live["output_tokens"] >= 0
        assert client.telemetry_finish("integration-agent")["finished_at"]
        reference = capture_reference(client, [{"id": "x", "prompt": "hello", "max_new_tokens": 2}],
                                      paths.campaign("transport") / "reference.json", {"fixture": True})
        assert evaluate_correctness(client, reference, {"atol": 0, "rtol": 0})["passed"]
    finally:
        engine.stop()
    assert not engine.record.exists()


def test_engine_reports_candidate_concurrent_scheduler(paths, project, tmp_path):
    source = tmp_path / "concurrent-engine"
    shutil.copytree(project / "templates/engine/toy", source)
    config = json.loads((source / "engine.json").read_text())
    config["request_scheduler"] = "concurrent_requests"
    (source / "engine.json").write_text(json.dumps(config))
    engine = EngineProcess(paths, "concurrent-transport", sys.executable, startup_timeout=10)
    try:
        client = engine.start(source, paths.campaign("concurrent-transport") / "engine.log")
        assert client.health()["runtime"]["request_scheduler"] == "concurrent_requests"
        observation = client.generate(
            prompt="scheduler", max_new_tokens=2, fixed_length=True)["host_observation"]
        assert observation["instrumentation"] == "host wall clock only"
        assert "not thread-safe" in observation["token_timing_unavailable_reason"]
    finally:
        engine.stop()


class MockQwenEndpoint(BaseHTTPRequestHandler):
    actions = [
        ("replace_text", {"path": "engine.py", "old": "value = 1", "new": "value = 2"}),
        ("check_python", {"path": "engine.py"}),
        ("submit_candidate", {"hypothesis": "Missing required previous analysis"}),
        ("submit_candidate", {
            "hypothesis": "The fixture replaces the target value while preserving valid Python syntax.",
            "previous_analysis": "The previous benchmark showed no candidate had yet been measured; this edit only tests the tool pipeline.",
            "summary": "Changed one candidate constant and parsed the file.",
        }),
    ]

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append(request)
        name, arguments = self.actions[min(len(self.server.requests) - 1, len(self.actions) - 1)]
        call = {
            "index": 0,
            "id": f"call_{len(self.server.requests)}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }
        completion_id = f"chatcmpl-test-{len(self.server.requests)}"
        payloads = [
            {"id": completion_id, "object": "chat.completion.chunk", "created": 1,
             "model": "test", "choices": [{"index": 0, "delta": {"tool_calls": [call]},
                                                   "finish_reason": None}]},
            {"id": completion_id, "object": "chat.completion.chunk", "created": 1,
             "model": "test", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
        ]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        for payload in payloads:
            self.wfile.write(b"data: " + json.dumps(payload).encode() + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def log_message(self, format, *args):
        pass


def test_qwen_agent_assistant_executes_audited_candidate_tools(tmp_path, project):
    server = ThreadingHTTPServer(("127.0.0.1", 0), MockQwenEndpoint)
    server.requests = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "engine.py").write_text("value = 1\n")
    trace = tmp_path / "commands.jsonl"
    context = {
        "agent": {"mode": "local", "max_new_tokens": 128, "request_timeout": 10},
        "iteration": 2,
        "previous": {"analysis_pending": True},
        "source": "value = 1\n",
    }
    try:
        result = run_agent(
            f"http://127.0.0.1:{server.server_port}", context, workspace, trace,
            project / "templates/prompts/implementation.md", api_key="test-key",
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert result["mode"] == "qwen_agent"
    assert result["complete"]
    assert result["previous_analysis"]
    assert (workspace / "engine.py").read_text() == "value = 2\n"
    assert [request["tools"][0]["type"] for request in server.requests] == ["function"] * 4
    assert all(request["stream"] and request["model"] == "vibeforge-local" for request in server.requests)
    actions = [json.loads(line) for line in trace.read_text().splitlines()]
    assert [item["tool"] for item in actions if item.get("tool")] == [
        "replace_text", "check_python", "submit_candidate", "submit_candidate"
    ]
    assert any(
        "previous_analysis" in action.get("error", "")
        and not action.get("result", {}).get("submitted", True)
        for action in actions
    )
