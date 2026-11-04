import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from vibeforge.agent.runner import run_agent, smoke_agent
from vibeforge.engine.process import EngineProcess
from vibeforge.evaluation.reference import capture_reference
from vibeforge.evaluation.correctness import evaluate_correctness


def test_engine_http_reference_and_qwen_agent_protocol(paths, project):
    engine = EngineProcess(paths, "transport", sys.executable, startup_timeout=10)
    try:
        client = engine.start(project / "templates/engine/toy", paths.campaign("transport") / "engine.log")
        smoke = smoke_agent(client)
        assert smoke["passed"]
        assert smoke["framework"] == "Qwen-Agent"
        reference = capture_reference(client, [{"id": "x", "prompt": "hello", "max_new_tokens": 2}],
                                      paths.campaign("transport") / "reference.json", {"fixture": True})
        assert evaluate_correctness(client, reference, {"atol": 0, "rtol": 0})["passed"]
    finally:
        engine.stop()
    assert not engine.record.exists()


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
    assert any("Analyze the previous" in action.get("error", "") for action in actions)
