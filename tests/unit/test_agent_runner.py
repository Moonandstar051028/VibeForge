import json
import pytest

import qwen_agent.agents
from qwen_agent.llm.base import ModelServiceError

import vibeforge.agent.runner as agent_runner
from vibeforge.agent.interface import agent_capabilities
from vibeforge.agent.runner import _agent_input_budget, _assistant_tools, _run_qwen_agent


def test_profiling_tool_is_hidden_when_profiling_is_disabled(tmp_path):
    context = {
        "engine_url": "http://127.0.0.1:9000", "candidate_contract": {},
        "loop_protocol": {"profiling": {"enabled": False}},
    }

    capabilities = agent_capabilities(context)
    tools = _assistant_tools(tmp_path, tmp_path / "trace.jsonl", context, {})

    assert "profile_inference" not in capabilities["tools"]
    assert "profile_inference" not in {tool.name for tool in tools}


def test_agent_input_budget_reserves_output_and_chat_tool_overhead():
    assert _agent_input_budget({"model": {"max_context": 32768}}, 1024) == 25191
    assert _agent_input_budget({"model": {"max_context": 16384}}, 512) == 12596
    with pytest.raises(ValueError, match="too small"):
        _agent_input_budget({"model": {"max_context": 1024}}, 1024)


def test_read_file_tool_returns_source_without_a_second_json_encoding(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = 'def load(config):\n    model_path = config["model_path"]\n'
    (workspace / "engine.py").write_text(source)
    tools = {tool.name: tool for tool in _assistant_tools(workspace, tmp_path / "trace.jsonl", {}, {})}
    result = tools["read_file"].call(json.dumps({"path": "engine.py"}))
    assert result == source
    replacement = source.replace('config["model_path"]', 'config.get("model_path")')
    response = tools["replace_text"].call(json.dumps({
        "path": "engine.py", "old": result, "new": replacement,
    }))
    assert json.loads(response) == {"updated": "engine.py"}
    assert (workspace / "engine.py").read_text() == replacement


def test_pending_previous_analysis_rejects_placeholder(tmp_path):
    context = {"previous": {"analysis_pending": True}}
    submission = {}
    submit = next(tool for tool in _assistant_tools(
        tmp_path / "workspace", tmp_path / "trace.jsonl", context, submission
    ) if tool.name == "submit_candidate")

    placeholder = json.dumps({
        "hypothesis": "Try a different attention setting.",
        "previous_analysis": "Analysis pending. No causal explanation has been established.",
        "summary": "No source change.",
    })
    with pytest.raises(ValueError, match="placeholder"):
        submit.call(placeholder)

    analyzed = json.dumps({
        "hypothesis": "Test an independent kernel mechanism.",
        "previous_analysis": "The measured gain was 0.1%, with a confidence interval crossing zero, so the prior change was not supported.",
        "summary": "No source change.",
    })
    assert json.loads(submit.call(analyzed)) == {"submitted": True}
    assert submission["previous_analysis"].startswith("The measured gain")


def test_previous_analysis_cannot_be_submitted_for_another_iteration(tmp_path):
    context = {"previous": {"iteration": 24, "analysis_pending": True}}
    submission = {}
    submit = next(tool for tool in _assistant_tools(
        tmp_path / "workspace", tmp_path / "trace.jsonl", context, submission
    ) if tool.name == "submit_candidate")
    arguments = {"hypothesis": "Test a distinct mechanism.", "summary": "No change.",
                 "previous_analysis": "Iteration 26 was unfinished; iteration 2 had no significant gain."}
    result = json.loads(submit.call(json.dumps(arguments)))
    assert result["submitted"] is False
    assert "assigned to iteration 24" in result["error"]
    assert not submission
    arguments["previous_analysis"] = "Iteration 24: no code changed, so no performance conclusion was established."
    assert json.loads(submit.call(json.dumps(arguments))) == {"submitted": True}
    assert submission["analysis_iteration"] == 24


def test_candidate_submission_rejects_unregistered_attention_backend(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    engine = workspace / "engine.py"
    engine.write_text('arguments = {"attn_implementation": "xformers"}\n')
    context = {"model_features": {"model_type": "qwen3_5_text"},
               "environment": {"inference": {"packages": {"torch": "2.7.1", "xformers": "0.0.31"}}}}
    submission = {}
    submit = next(tool for tool in _assistant_tools(
        workspace, tmp_path / "trace.jsonl", context, submission
    ) if tool.name == "submit_candidate")
    arguments = json.dumps({"hypothesis": "Try an optimized attention backend.",
                            "previous_analysis": "No previous iteration.", "summary": "Set backend."})

    rejected = json.loads(submit.call(arguments))
    assert rejected["submitted"] is False
    assert "not supported" in rejected["error"]
    assert "complete" not in submission

    engine.write_text('arguments = {"attn_implementation": "sdpa"}\n')
    assert json.loads(submit.call(arguments)) == {"submitted": True}

    engine.write_text("arguments = {}\n")
    (workspace / "engine.json").write_text('{"attn_implementation": "xformers"}')
    configured = json.loads(submit.call(arguments))
    assert configured["submitted"] is False
    assert "not supported" in configured["error"]


def test_candidate_submission_rejects_missing_attention_dependency(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "engine.py").write_text('arguments = {"attn_implementation": "flash_attention_2"}\n')
    context = {"model_features": {"model_type": "qwen3_5_text"},
               "environment": {"inference": {"packages": {"torch": "2.7.1", "xformers": "0.0.31"}}}}
    submission = {}
    submit = next(tool for tool in _assistant_tools(
        workspace, tmp_path / "trace.jsonl", context, submission
    ) if tool.name == "submit_candidate")
    response = json.loads(submit.call(json.dumps({"hypothesis": "Test Flash Attention.",
                                                  "previous_analysis": "No previous iteration.",
                                                  "summary": "Set attention backend."})))
    assert response["submitted"] is False
    assert "requires flash-attn" in response["error"]


def test_history_does_not_globally_disable_supported_attention_backends(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    engine = workspace / "engine.py"
    engine.write_text('arguments = {"attn_implementation": "sdpa"}\n')
    context = {
        "model_features": {"model_type": "qwen3_5_text"},
        "environment": {"inference": {"packages": {"torch": "2.7.1"}}},
        "history": [{"decision": {"outcome": "reject", "reason": "no_significant_gain"},
                     "attention_backends": ["sdpa"]}],
    }
    submission = {}
    submit = next(tool for tool in _assistant_tools(
        workspace, tmp_path / "trace.jsonl", context, submission
    ) if tool.name == "submit_candidate")
    arguments = json.dumps({"hypothesis": "Try SDPA again.", "previous_analysis": "No gain.",
                            "summary": "Set SDPA."})
    assert json.loads(submit.call(arguments)) == {"submitted": True}

    engine.write_text('arguments = {"attn_implementation": "eager"}\n')
    assert json.loads(submit.call(arguments)) == {"submitted": True}

    context["history"].append({"decision": {"outcome": "reject", "reason": "correctness_failed"},
                               "attention_backends": ["eager"]})
    engine.write_text('arguments = {"attn_implementation": "paged|eager"}\n')
    next_submission = {}
    submit = next(tool for tool in _assistant_tools(
        workspace, tmp_path / "trace2.jsonl", context, next_submission
    ) if tool.name == "submit_candidate")
    response = json.loads(submit.call(arguments))
    assert not response["submitted"] and "paged-cache" in response["error"]


def test_agent_prompt_uses_captured_backend_availability(tmp_path, monkeypatch):
    calls = []

    class FakeAssistant:
        def __init__(self, llm, function_list, system_message):
            self.llm = llm
            self.function_list = function_list
            self.system_message = system_message
            calls.append(self)

        def run(self, messages):
            if not self.llm["generate_cfg"].get("tool_choice"):
                yield [{"role": "assistant", "content": "I will submit a candidate."}]
                return
            self.function_list[0].call(json.dumps({
                "hypothesis": "Test an available backend.",
                "previous_analysis": "No previous iteration.",
                "summary": "No source change.",
            }))
            yield [{"role": "function", "name": "submit_candidate", "content": "{\"submitted\": true}"}]

    monkeypatch.setattr(qwen_agent.agents, "Assistant", FakeAssistant)
    _run_qwen_agent(
        "http://127.0.0.1:1234", {"iteration": 1, "requirements": {"objective": "throughput"},
         "model": {"name": "test"}, "environment": {"inference": {"packages": {"torch": "2.7.1"}}},
         "agent": {"max_new_tokens": 256, "request_timeout": 5, "timeout_seconds": 30, "max_turns": 4}},
        tmp_path / "workspace", tmp_path / "trace.jsonl", "Optimize throughput.", "token",
    )
    assert "does not contain `flash-attn`" in calls[0].system_message
    assert "Missing flash-attn rules out only flash attention" in calls[0].system_message
    assert "assess every modification_surfaces.allowed value" in calls[0].system_message
    audit = json.loads((tmp_path / "agent-input.json").read_text())
    assert audit["schema_version"] == 1 and audit["calls"]
    assert audit["calls"][0]["system_prompt"] == calls[0].system_message
    assert audit["calls"][0]["messages"][0]["role"] == "user"
    assert audit["calls"][0]["tools"] and len(audit["calls"][0]["sha256"]) == 64


def test_agent_prompt_warns_against_rejected_backend_family(tmp_path, monkeypatch):
    calls = []

    class FakeAssistant:
        def __init__(self, llm, function_list, system_message):
            self.llm = llm
            self.function_list = function_list
            self.system_message = system_message
            calls.append(self)

        def run(self, messages):
            if not self.llm["generate_cfg"].get("tool_choice"):
                submit = next(tool for tool in self.function_list if tool.name == "submit_candidate")
                submit.call(json.dumps({
                    "hypothesis": "Tune an unrelated decode setting.",
                    "previous_analysis": "The SDPA result had no significant gain.",
                    "summary": "No change.",
                }))
                yield [{"role": "function", "name": "submit_candidate", "content": "{\"submitted\": true}"}]

    monkeypatch.setattr(qwen_agent.agents, "Assistant", FakeAssistant)
    _run_qwen_agent(
        "http://127.0.0.1:1234",
        {"iteration": 3, "requirements": {"objective": "throughput"},
         "model": {"name": "test"}, "model_features": {"model_type": "qwen3_5_text"},
         "environment": {"inference": {"packages": {"torch": "2.7.1"}}},
         "history": [{"decision": {"outcome": "reject"}, "attention_backends": ["sdpa"]},
                     *[{"decision": {"outcome": "reject", "reason": "no_change"}} for _ in range(3)]],
         "agent": {"max_new_tokens": 256, "request_timeout": 5, "timeout_seconds": 30, "max_turns": 4}},
        tmp_path / "workspace", tmp_path / "trace.jsonl", "Optimize throughput.", "token",
    )
    assert "Prior rejected candidates used these attention backend families: sdpa" in calls[0].system_message
    assert "Evidence policy for this session supersedes any earlier restriction" in calls[0].system_message
    assert "Search stalled: 3 consecutive iterations" in calls[0].system_message
    assert "read a startup/runtime log through read_evidence" in calls[0].system_message


def test_agent_retries_with_required_submission_tool(tmp_path, monkeypatch):
    calls = []

    class FakeAssistant:
        def __init__(self, llm, function_list, system_message):
            self.llm = llm
            self.function_list = function_list
            calls.append(self)

        def run(self, messages):
            if not self.llm["generate_cfg"].get("tool_choice"):
                yield [{"role": "assistant", "content": "A concise hypothesis without a tool call."}]
                return
            arguments = json.dumps({
                "hypothesis": "Reduce per-token synchronization overhead.",
                "previous_analysis": "No previous iteration.",
                "summary": "No source change was made before the retry.",
            })
            self.function_list[0].call(arguments)
            yield [{"role": "function", "name": "submit_candidate", "content": "{\"submitted\": true}"}]

    monkeypatch.setattr(qwen_agent.agents, "Assistant", FakeAssistant)
    result = _run_qwen_agent(
        "http://127.0.0.1:1234",
        {"iteration": 1, "requirements": {"objective": "throughput"},
         "model": {"name": "test"}, "agent": {"max_new_tokens": 1024,
         "request_timeout": 5, "timeout_seconds": 30, "max_turns": 20}},
        tmp_path / "workspace",
        tmp_path / "trace.jsonl",
        "Optimize throughput.",
        "token",
    )

    assert result["complete"] is True
    assert result["recovered_submission"] is True
    assert calls[1].llm["generate_cfg"]["tool_choice"]["function"]["name"] == "submit_candidate"


def test_candidate_planning_reserves_last_tool_call_for_submission(tmp_path, monkeypatch):
    calls = []
    workspace = tmp_path / "workspace"
    parent = tmp_path / "parent"
    workspace.mkdir()
    parent.mkdir()
    (parent / "engine.py").write_text("cache = 'dynamic'\n")
    (workspace / "engine.py").write_text("cache = 'static'\n")
    monkeypatch.setattr(agent_runner, "candidate_gate", lambda *args: {"valid": True})

    class FakeAssistant:
        def __init__(self, llm, function_list, system_message):
            self.llm, self.tools = llm, function_list
            calls.append(self)

        def run(self, messages):
            if not self.llm["generate_cfg"].get("tool_choice"):
                for count in range(1, 4):
                    yield [{"role": "function", "name": "read_file", "content": "page"}
                           for _ in range(count)]
                return
            assert [tool.name for tool in self.tools] == ["submit_candidate"]
            final = json.loads(messages[0]["content"])
            assert "cache = 'static'" in final["candidate_engine"]
            assert final["candidate_contract"]["observation_paths"] == ["iterations/0001/observation.json"]
            arguments = {
                "hypothesis": "A static cache may reduce decode allocation overhead.",
                "previous_analysis": "This is the initial candidate.",
                "summary": "Use the already written static-cache candidate.",
                "experiment_plan": {
                    "measured_bottleneck": "Decode forward time",
                    "evidence_paths": ["iterations/0001/observation.json"],
                    "cause": "Dynamic cache growth may add decode overhead.",
                    "change": "Use a static cache.",
                    "metric": "throughput", "direction": "increase", "expected_change_pct": 3,
                    "validation": "Run frozen correctness and paired performance checks.",
                    "history_refs": [],
                    "bottleneck_reference": {"evidence_path": "iterations/0001/observation.json",
                                             "metric": "decode_forward_seconds_p95", "value": 3.0,
                                             "unit": "seconds"},
                    "target_profile_ids": ["agent_2k"],
                    "runtime_preconditions": [{"field": "execution_path", "equals": "transformers_generate"}],
                    "expected_bottleneck_change_pct": 3, "bottleneck_direction": "decrease",
                    "expected_change_basis": "Avoid cache growth during decode.",
                },
            }
            response = self.tools[0].call(json.dumps(arguments))
            yield [{"role": "function", "name": "submit_candidate", "content": response}]

    monkeypatch.setattr(qwen_agent.agents, "Assistant", FakeAssistant)
    result = _run_qwen_agent(
        "http://127.0.0.1:1234",
        {"iteration": 1, "requirements": {"objective": "throughput"},
         "precision": {"quantization": "bnb-nf4"},
         "performance_observation": {"runtime": {"execution_path": "transformers_generate"},
                                     "measurements": {"decode_forward_seconds_p95": {
                                         "value": 3.0, "unit": "seconds"}}},
         "candidate_contract": {"version": 2, "parent_source": str(parent),
                                "observation_paths": ["iterations/0001/observation.json"]},
         "evidence_root": str(tmp_path),
         "model": {"name": "test", "max_context": 32768},
         "agent": {"max_new_tokens": 512, "request_timeout": 5,
                   "timeout_seconds": 30, "max_turns": 4}},
        workspace, tmp_path / "trace.jsonl", "Optimize throughput.", "token")

    assert result["complete"] and result["experiment_plan"]["change"] == "Use a static cache."
    assert result["recovered_submission"] is True
    assert calls[1].llm["generate_cfg"]["tool_choice"]["function"]["name"] == "submit_candidate"
    trace = (tmp_path / "trace.jsonl").read_text()
    assert "qwen_agent_planning_finalization_started" in trace


def test_candidate_planning_finalizes_after_incomplete_continuation(tmp_path, monkeypatch):
    calls = []
    workspace = tmp_path / "workspace"
    parent = tmp_path / "parent"
    workspace.mkdir()
    parent.mkdir()
    (parent / "engine.py").write_text("cache = 'dynamic'\n")
    (workspace / "engine.py").write_text("cache = 'static'\n")
    monkeypatch.setattr(agent_runner, "candidate_gate", lambda *args: {"valid": True})

    class FakeAssistant:
        def __init__(self, llm, function_list, system_message):
            self.llm, self.tools = llm, function_list
            calls.append(self)

        def run(self, messages):
            if not self.llm["generate_cfg"].get("tool_choice"):
                compact = json.loads(messages[0]["content"])
                if len(calls) == 2:
                    assert compact["workspace_changed"] is True
                    assert compact["candidate_engine"] == "cache = 'static'\n"
                yield [{"role": "assistant", "content": "More analysis without a final action."}]
                return
            assert self.llm["generate_cfg"]["tool_choice"]["function"]["name"] == "submit_candidate"
            submit = self.tools[0]
            arguments = {
                "hypothesis": "A static cache may reduce decode allocation overhead.",
                "previous_analysis": "This is the initial candidate.",
                "summary": "Use the already written static-cache candidate.",
                "experiment_plan": {
                    "measured_bottleneck": "Decode forward time",
                    "evidence_paths": ["iterations/0001/observation.json"],
                    "cause": "Dynamic cache growth may add decode overhead.",
                    "change": "Use a static cache.",
                    "metric": "throughput", "direction": "increase", "expected_change_pct": 3,
                    "validation": "Run frozen correctness and paired performance checks.",
                    "history_refs": [],
                    "bottleneck_reference": {"evidence_path": "iterations/0001/observation.json",
                                             "metric": "decode_forward_seconds_p95", "value": 3.0,
                                             "unit": "seconds"},
                    "target_profile_ids": ["agent_2k"],
                    "runtime_preconditions": [{"field": "execution_path", "equals": "transformers_generate"}],
                    "expected_bottleneck_change_pct": 3, "bottleneck_direction": "decrease",
                    "expected_change_basis": "Avoid cache growth during decode.",
                },
            }
            response = submit.call(json.dumps(arguments))
            yield [{"role": "function", "name": "submit_candidate", "content": response}]

    monkeypatch.setattr(qwen_agent.agents, "Assistant", FakeAssistant)
    result = _run_qwen_agent(
        "http://127.0.0.1:1234",
        {"iteration": 1, "requirements": {"objective": "throughput"},
         "precision": {"quantization": "bnb-nf4"},
         "performance_observation": {"runtime": {"execution_path": "transformers_generate"},
                                     "measurements": {"decode_forward_seconds_p95": {
                                         "value": 3.0, "unit": "seconds"}}},
         "candidate_contract": {"version": 2, "parent_source": str(parent),
                                "observation_paths": ["iterations/0001/observation.json"]},
         "evidence_root": str(tmp_path), "model": {"name": "test", "max_context": 32768},
         "agent": {"max_new_tokens": 512, "request_timeout": 5,
                   "timeout_seconds": 30, "max_turns": 4}},
        workspace, tmp_path / "trace.jsonl", "Optimize throughput.", "token")

    assert result["complete"] and result["experiment_plan"]["change"] == "Use a static cache."
    assert len(calls) == 3
    trace = (tmp_path / "trace.jsonl").read_text()
    assert '"reason":"continuation_incomplete"' in trace


def test_candidate_planning_reports_blocked_after_incomplete_no_change(tmp_path, monkeypatch):
    calls = []
    workspace = tmp_path / "workspace"
    parent = tmp_path / "parent"
    workspace.mkdir()
    parent.mkdir()
    (parent / "engine.py").write_text("cache = 'dynamic'\n")
    (workspace / "engine.py").write_text("cache = 'dynamic'\n")

    class FakeAssistant:
        def __init__(self, llm, function_list, system_message):
            self.llm, self.tools = llm, function_list
            calls.append(self)

        def run(self, messages):
            if not self.llm["generate_cfg"].get("tool_choice"):
                yield [{"role": "assistant", "content": "No justified source change was found."}]
                return
            assert self.llm["generate_cfg"]["tool_choice"]["function"]["name"] == "report_blocked"
            response = self.tools[0].call(json.dumps({
                "reason": "The required optimized kernel dependency is absent from the frozen environment.",
                "previous_analysis": "This is the initial candidate.",
            }))
            yield [{"role": "function", "name": "report_blocked", "content": response}]

    monkeypatch.setattr(qwen_agent.agents, "Assistant", FakeAssistant)
    result = _run_qwen_agent(
        "http://127.0.0.1:1234",
        {"iteration": 1, "requirements": {"objective": "throughput"},
         "candidate_contract": {"version": 2, "parent_source": str(parent),
                                "observation_paths": ["iterations/0001/observation.json"]},
         "evidence_root": str(tmp_path), "model": {"name": "test", "max_context": 32768},
         "agent": {"max_new_tokens": 512, "request_timeout": 5,
                   "timeout_seconds": 30, "max_turns": 4}},
        workspace, tmp_path / "trace.jsonl", "Optimize throughput.", "token")

    assert result["complete"] and result["blocked"] is True
    assert result["blocked_reason"].startswith("The required optimized kernel")
    assert len(calls) == 3


def test_exploration_reserves_calls_for_edit_check_and_submit(tmp_path, monkeypatch):
    workspace, parent = tmp_path / 'workspace', tmp_path / 'parent'
    workspace.mkdir()
    parent.mkdir()
    source = '# padding\n' * 350 + 'value = 1\n'
    for folder in (workspace, parent):
        (folder / 'engine.py').write_text(source)
    calls, reads = [], []
    rule = {'rule_id': 'local-component', 'implementation': {'recipe': 'Change value'}}
    monkeypatch.setattr(agent_runner, 'candidate_gate', lambda *args: {'valid': True})

    class FakeAssistant:
        def __init__(self, llm, function_list, system_message):
            self.tools = {tool.name: tool for tool in function_list}
            calls.append(self)

        def run(self, messages):
            history = []
            if len(calls) == 1:
                for _ in range(8):
                    response = self.tools['read_file'].call(json.dumps({'path': 'engine.py'}))
                    reads.append(True)
                    history.append({'role': 'function', 'name': 'read_file', 'content': response})
                    yield list(history)
                return
            compact = json.loads(messages[0]['content'])
            assert compact['knowledge'] == [rule]
            assert 'value = 1' in compact['candidate_engine']
            assert 'read_file' not in self.tools and 'read_evidence' not in self.tools
            actions = [
                ('replace_text', {'path': 'engine.py', 'old': 'value = 1', 'new': 'value = 2'}),
                ('check_python', {'path': 'engine.py'}),
                ('submit_candidate', {'hypothesis': 'Test measured mechanism', 'experiment_plan': {
                    'measured_bottleneck': 'decode', 'cause': 'overhead', 'change': 'Change value',
                    'expected_change_pct': 3, 'validation': 'Full gates', 'history_refs': []}}),
            ]
            for name, arguments in actions:
                response = self.tools[name].call(json.dumps(arguments))
                history.append({'role': 'function', 'name': name, 'content': response})
                yield list(history)

    monkeypatch.setattr(qwen_agent.agents, 'Assistant', FakeAssistant)
    result = _run_qwen_agent('http://127.0.0.1:1234', {
        'iteration': 1, 'knowledge': [rule], 'evidence_root': str(tmp_path),
        'candidate_contract': {'version': 3, 'parent_source': str(parent)},
        'model': {'max_context': 32768},
        'agent': {'max_new_tokens': 512, 'timeout_seconds': 30, 'max_turns': 8},
    }, workspace, tmp_path / 'trace.jsonl', 'Optimize', 'token')
    assert len(reads) == 4 and len(calls) == 2
    assert result['complete'] and not result.get('blocked')
    assert (workspace / 'engine.py').read_text().endswith('value = 2\n')


def test_candidate_planning_compacts_after_two_rejected_submissions(tmp_path, monkeypatch):
    calls = []
    gate_calls = []
    workspace = tmp_path / "workspace"
    parent = tmp_path / "parent"
    workspace.mkdir()
    parent.mkdir()
    (parent / "engine.py").write_text("cache = 'dynamic'\n")
    (workspace / "engine.py").write_text("cache = 'static'\n")

    def fake_gate(_workspace, _context, plan):
        gate_calls.append(plan)
        if len(gate_calls) <= 2:
            return {"valid": False, "reason": "runtime_precondition_failed",
                    "field": "precision.quantization",
                    "available_runtime_preconditions": [
                        {"field": "execution_path", "equals": "transformers_generate"}
                    ]}
        return {"valid": True}

    monkeypatch.setattr(agent_runner, "candidate_gate", fake_gate)

    def arguments(field):
        return {
            "hypothesis": "A static cache may reduce decode allocation overhead.",
            "previous_analysis": "This is the initial candidate.",
            "summary": "Use the already written static-cache candidate.",
            "experiment_plan": {
                "measured_bottleneck": "Decode forward time",
                "evidence_paths": ["iterations/0001/observation.json"],
                "cause": "Dynamic cache growth may add decode overhead.",
                "change": "Use a static cache.",
                "metric": "throughput", "direction": "increase", "expected_change_pct": 3,
                "validation": "Run frozen correctness and paired performance checks.",
                "history_refs": [],
                "bottleneck_reference": {"evidence_path": "iterations/0001/observation.json",
                                         "metric": "decode_forward_seconds_p95", "value": 3.0,
                                         "unit": "seconds"},
                "target_profile_ids": ["agent_2k"],
                "runtime_preconditions": [{"field": field, "equals": "bnb-nf4"}],
                "expected_bottleneck_change_pct": 3, "bottleneck_direction": "decrease",
                "expected_change_basis": "Avoid cache growth during decode.",
            },
        }

    class FakeAssistant:
        def __init__(self, llm, function_list, system_message):
            self.llm, self.tools = llm, function_list
            calls.append(self)

        def run(self, messages):
            submit = next(tool for tool in self.tools if tool.name == "submit_candidate")
            if not self.llm["generate_cfg"].get("tool_choice"):
                snapshots = []
                for field in ("quantization", "precision.quantization"):
                    response = submit.call(json.dumps(arguments(field)))
                    snapshots.append({"role": "function", "name": "submit_candidate", "content": response})
                    yield list(snapshots)
                pytest.fail("The open-ended agent stream was not closed after two rejected submissions")
            final = json.loads(messages[0]["content"])
            assert final["last_gate_result"]["error"]["available_runtime_preconditions"]
            corrected = arguments("execution_path")
            corrected["experiment_plan"]["runtime_preconditions"][0]["equals"] = "transformers_generate"
            response = submit.call(json.dumps(corrected))
            yield [{"role": "function", "name": "submit_candidate", "content": response}]

    monkeypatch.setattr(qwen_agent.agents, "Assistant", FakeAssistant)
    result = _run_qwen_agent(
        "http://127.0.0.1:1234",
        {"iteration": 1, "requirements": {"objective": "throughput"},
         "precision": {"quantization": "bnb-nf4"},
         "performance_observation": {"runtime": {"execution_path": "transformers_generate"},
                                     "measurements": {"decode_forward_seconds_p95": {
                                         "value": 3.0, "unit": "seconds"}}},
         "candidate_contract": {"version": 2, "parent_source": str(parent),
                                "observation_paths": ["iterations/0001/observation.json"]},
         "evidence_root": str(tmp_path), "model": {"name": "test", "max_context": 32768},
         "agent": {"max_new_tokens": 2048, "request_timeout": 5,
                   "timeout_seconds": 30, "max_turns": 20}},
        workspace, tmp_path / "trace.jsonl", "Optimize throughput.", "token")

    assert result["complete"] and result["recovered_submission"] is True
    assert len(gate_calls) == 3
    assert calls[1].llm["generate_cfg"]["max_tokens"] == 1024
    trace = (tmp_path / "trace.jsonl").read_text()
    assert "qwen_agent_rejected_submission_limit_reached" in trace
    assert '"reason":"rejected_submission_limit"' in trace


@pytest.mark.parametrize("server_error", [
    "Error code: 400 - Malformed or truncated Qwen tool call",
    "Error code: 400 - Request exceeds configured context",
])
def test_agent_recovers_protocol_or_context_error_with_structured_submission(tmp_path, monkeypatch, server_error):
    calls = []
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "engine.py").write_text("candidate_change = True\n")

    class FakeAssistant:
        def __init__(self, llm, function_list, system_message):
            self.llm = llm
            self.function_list = function_list
            self.system_message = system_message
            calls.append(self)
            assert llm["generate_cfg"]["max_input_tokens"] == 25959

        def run(self, messages):
            if not self.llm["generate_cfg"].get("tool_choice"):
                raise ModelServiceError(exception=RuntimeError(server_error))
            recovery_context = json.loads(messages[0]["content"])
            assert "candidate_change = True" in recovery_context["candidate_engine"]
            assert "stable_engine = True" in recovery_context["stable_engine"]
            arguments = json.dumps({"hypothesis": "Test the existing candidate change.",
                                    "previous_analysis": "This is the first candidate.",
                                    "summary": "Keep the already written candidate change."})
            self.function_list[0].call(arguments)
            yield [{"role": "function", "name": "submit_candidate", "content": "{\"submitted\": true}"}]

    monkeypatch.setattr(qwen_agent.agents, "Assistant", FakeAssistant)
    result = _run_qwen_agent(
        "http://127.0.0.1:1234",
        {"iteration": 1, "requirements": {"objective": "throughput"},
         "source": "stable_engine = True", "model": {"name": "test", "max_context": 32768},
         "agent": {"max_new_tokens": 256, "request_timeout": 5, "timeout_seconds": 30, "max_turns": 4}},
        workspace, tmp_path / "trace.jsonl", "Optimize throughput.", "token",
    )

    assert result["complete"] is True
    assert result["recovered_submission"] is True
    assert ("malformed or truncated tool call" if "Malformed" in server_error else "context limit") in calls[1].system_message
    assert [tool.name for tool in calls[1].function_list] == ["submit_candidate"]


def test_agent_does_not_mask_other_model_service_errors(tmp_path, monkeypatch):
    class FakeAssistant:
        def __init__(self, llm, function_list, system_message):
            pass

        def run(self, messages):
            raise ModelServiceError(exception=RuntimeError("connection refused"))
            yield

    monkeypatch.setattr(qwen_agent.agents, "Assistant", FakeAssistant)
    with pytest.raises(ModelServiceError, match="connection refused"):
        _run_qwen_agent(
            "http://127.0.0.1:1234", {"iteration": 1, "requirements": {"objective": "throughput"},
             "model": {"name": "test"}, "agent": {"max_new_tokens": 256, "request_timeout": 5,
             "timeout_seconds": 30, "max_turns": 4}},
            tmp_path / "workspace", tmp_path / "trace.jsonl", "Optimize throughput.", "token",
        )


def test_agent_emergency_submission_after_two_malformed_tool_calls(tmp_path, monkeypatch):
    calls = []

    class FakeAssistant:
        def __init__(self, llm, function_list, system_message):
            self.llm = llm
            self.function_list = function_list
            calls.append(self)

        def run(self, messages):
            raise ModelServiceError(exception=RuntimeError(
                "Error code: 400 - Malformed or truncated Qwen tool call"))
            yield

    monkeypatch.setattr(qwen_agent.agents, "Assistant", FakeAssistant)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "engine.py").write_text("candidate_change = True\n")
    result = _run_qwen_agent(
        "http://127.0.0.1:1234",
        {"iteration": 2, "requirements": {"objective": "throughput"},
         "source": "stable_change = False", "model": {"name": "test"},
         "agent": {"max_new_tokens": 256, "request_timeout": 5, "timeout_seconds": 30, "max_turns": 4}},
        workspace, tmp_path / "trace.jsonl", "Optimize throughput.", "token",
    )
    assert result["complete"] is True
    assert result["recovered_submission"] is True
    assert len(calls) == 2
    assert result["submission_origin"] == "controller_recovery"
    assert "Malformed or truncated" in result["recovery_error"]


def test_agent_rejects_static_context_that_would_starve_tool_history(tmp_path, monkeypatch):
    calls = []
    class FakeAssistant:
        def __init__(self, **kwargs):
            calls.append(kwargs)
    monkeypatch.setattr(qwen_agent.agents, "Assistant", FakeAssistant)
    with pytest.raises(ValueError, match="insufficient tool-history space"):
        _run_qwen_agent("http://127.0.0.1:1234", {
            "model": {"name": "test", "max_context": 32768}, "extra": "unbounded diagnostic " * 20000,
            "agent": {"max_new_tokens": 2048}}, tmp_path / "workspace", tmp_path / "trace.jsonl",
            "Read evidence before editing.", "token")
    assert not calls
    budget = json.loads((tmp_path / "trace.jsonl").read_text().splitlines()[-1])
    assert budget["event"] == "agent_context_budget"
    assert budget["remaining_tokens"] < budget["tool_history_reserve"]
    assert budget["planning_ratio_target_tokens"] == int(budget["max_input_tokens"] * 0.55)


@pytest.mark.parametrize("first_response", ["prose", "truncated"])
@pytest.mark.parametrize("reflection_status", ["inconclusive", "supported"])
def test_analysis_has_separate_context_and_bounded_model_finalization(tmp_path, monkeypatch, first_response, reflection_status):
    from vibeforge.storage.atomic import write_json
    evidence = tmp_path / "campaign"
    write_json(evidence / "iterations/0001/decision.json", {"outcome": "reject", "reason": "interrupted_iteration"})
    context = {"previous": {"iteration": 1, "analysis_pending": True,
                            "decision": {"outcome": "reject", "reason": "interrupted_iteration"}},
               "model": {"name": "test", "max_context": 32768}, "source": "CURRENT_SOURCE_CANARY",
               "performance_observation": {"current_profile": "CURRENT_PROFILE_CANARY"},
               "candidate_contract": {"version": 2}, "evidence_root": str(evidence),
               "instructions": "Choose a new optimization and edit code.",
               "agent": {"max_new_tokens": 2048, "timeout_seconds": 30, "max_turns": 4}}
    calls = []
    class FakeAssistant:
        def __init__(self, llm, function_list, system_message):
            self.llm, self.tools = llm, function_list
            calls.append(self)
            assert "ANOTHER_OPTIMIZATION_PROMPT" not in system_message
            assert "This is an analysis-only session" in system_message
            assert not {"replace_text", "write_file", "profile_inference"} & {tool.name for tool in function_list}
        def run(self, messages):
            payload = json.loads(messages[0]["content"])
            assert "source" not in payload and "performance_observation" not in payload
            assert payload["previous"]["iteration"] == 1
            if not self.llm["generate_cfg"].get("tool_choice"):
                if first_response == "truncated":
                    raise ModelServiceError(exception=RuntimeError("Malformed or truncated Qwen tool call"))
                yield [{"role": "assistant", "content": "The experiment was interrupted; no performance result exists."}]
                return
            assert [tool.name for tool in self.tools] == ["submit_candidate"]
            arguments = {"hypothesis": "Review iteration 1", "previous_analysis": "Iteration 1: interrupted before a candidate experiment; no gain was measured.",
                         "reflection": {"iteration": 1, "hypothesis_status": reflection_status,
                            "evidence_paths": ["iterations/0001/decision.json"], "explanation": "Execution was interrupted.",
                            "next_action": "Resume evidence-based planning after the host fix."}}
            response = self.tools[0].call(json.dumps(arguments))
            yield [{"role": "function", "name": "submit_candidate", "content": response}]
    monkeypatch.setattr(qwen_agent.agents, "Assistant", FakeAssistant)
    if reflection_status == "supported":
        with pytest.raises(RuntimeError, match="without calling submit_candidate"):
            _run_qwen_agent("http://127.0.0.1:1234", context, tmp_path / "workspace", tmp_path / "trace.jsonl",
                            "ANOTHER_OPTIMIZATION_PROMPT", "token", analysis_only=True)
    else:
        result = _run_qwen_agent("http://127.0.0.1:1234", context, tmp_path / "workspace", tmp_path / "trace.jsonl",
                                 "ANOTHER_OPTIMIZATION_PROMPT", "token", analysis_only=True)
        assert result["complete"] and result["analysis_iteration"] == 1
        assert result["reflection"]["hypothesis_status"] == "inconclusive"
        assert result["submission_origin"] == "agent" and result["recovered_submission"]
    assert len(calls) == 2
    assert context["source"] == "CURRENT_SOURCE_CANARY"


def test_history_growth_compacts_retrievable_details_before_model_call(tmp_path, monkeypatch):
    from vibeforge.agent.runner import _fit_model_context
    from qwen_agent.utils.tokenization_qwen import tokenizer
    context = {"model": {"max_context": 32768}, "source": "current_source " * 10000,
               "history": [{"iteration": i, "hypothesis": "candidate", "change_patch": "old patch " * 3000,
                            "experiment_plan": {"cause": "long cause " * 5000},
                            "decision": {"outcome": "reject", "reason": "correctness_failed"}}
                           for i in range(1, 11)],
               "previous": {"iteration": 10, "experiment_plan": {"cause": "duplicate " * 10000},
                            "decision": {"reason": "correctness_failed"}},
               "performance_observation": {"measurements": {"normal.ttft_seconds": {"value": 1.2, "unit": "seconds"}}}}
    budget = _agent_input_budget(context, 2048)
    value, steps, count = _fit_model_context(context, "Read current code.", budget, 4096, tokenizer)
    assert count + 4096 <= budget
    assert "history_details_retrievable" in steps and "source_via_read_file" in steps
    assert [r["iteration"] for r in value["history"]] == list(range(1, 11))
    assert value["previous"]["iteration"] == 10
    assert value["review_measurement_status"]["formal_comparison_available"] is False
    assert value["performance_observation"]["measurements"] == context["performance_observation"]["measurements"]
    assert context["history"][0]["experiment_plan"]["cause"].startswith("long cause")


def test_refinement_context_summarizes_probe_without_token_or_operator_arrays():
    from vibeforge.agent.runner import _model_context
    context = {
        "candidate_feedback_path": "iterations/0004/probe/0000/probe.json",
        "candidate_feedback": {
            "candidate": "v-candidate", "needs_revision": True,
            "correctness_precheck": {
                "passed": False,
                "checks": [{"case": "sequential_request_isolation", "passed": False,
                            "actual_tokens": list(range(10000)), "repeat_tokens": list(range(10000))}],
                "runtime_observation": {"normal_request": {
                    "observed_cache_class": "StaticCache", "backend_wall_seconds": 4.6}},
            },
            "mechanism": {"metric": "normal.phases.decode_forward_seconds",
                          "before": 3.2, "after": 3.5, "improvement_pct": -8.3},
            "profile_reports": [{"normal_request": {"backend_wall_seconds": 4.7,
                                 "output_tokens_per_second": 6.9},
                                 "top_cpu_ops": [{"name": "large"}] * 1000}],
            "profile_id": "agent_2k", "prompt_index": 0,
            "limitations": ["Diagnostic only"],
        },
    }

    value = _model_context(context)

    feedback = value["candidate_feedback"]
    assert feedback["correctness_precheck"]["checks"] == [
        {"case": "sequential_request_isolation", "passed": False}]
    assert feedback["runtime_effect"]["observed_cache_class"] == "StaticCache"
    assert feedback["candidate_profile_request"]["output_tokens_per_second"] == 6.9
    assert feedback["evidence_path"] == "iterations/0004/probe/0000/probe.json"
    assert "profile_reports" not in feedback and "actual_tokens" not in json.dumps(feedback)


def test_optimization_brief_prioritizes_constraints_and_compacts_model_details():
    from vibeforge.agent.brief import build_optimization_brief
    from vibeforge.agent.interface import bounded_context_view
    from vibeforge.agent.runner import _model_context
    context = {
        "requirements": {"objective": "throughput"},
        "protocol": {"min_gain": 0.02, "max_profile_regression": 0.03},
        "model_features": {"model_type": "qwen3_5_text", "hidden_size": 5120,
                           "num_hidden_layers": 4,
                           "layer_types": ["linear_attention"] * 3 + ["full_attention"]},
        "hardware": {"gpu": "A100"},
        "environment": {"inference": {"packages": {"torch": "2", "triton": "3.3.1", "unrelated": "1"}}},
        "precision": {"quantization": "bnb-nf4"},
        "workload": {"profiles": [{"id": "concurrent_2", "max_new_tokens": 64,
                                    "concurrency": 2, "weight": 1}]},
        "modification_surfaces": {"allowed": ["scheduler", "execution_adapter", "native_kernel"]},
        "candidate_contract": {"observation_paths": ["iterations/0002/observation.json"]},
        "performance_observation": {
            "measurements": {
                "normal.backend_wall_seconds": {"value": 4.0, "unit": "seconds"},
                "normal.phases.decode_forward_seconds": {"value": 3.0, "unit": "seconds"},
            },
            "normal_request": {"unavailable_metrics": ["SM occupancy"]},
            "top_cpu_ops": [{"name": "cudaLaunchKernel", "calls": 180000,
                              "self_cpu_seconds": 0.9}],
            "top_device_operator_ops": [{"name": "bitsandbytes::gemm_4bit", "calls": 15000,
                                          "self_device_seconds": 0.8}],
            "runtime": {"execution_path": "transformers_generate", "request_scheduler": "serialized_lock",
                        "continuous_batching": False, "model_device": "cuda:0",
                        "packages": {"torch": "2.7", "bitsandbytes": "0.50", "flash-attn": None}},
        },
    }
    context["optimization_brief"] = build_optimization_brief(context)

    brief = context["optimization_brief"]
    assert brief["runtime_facts"]["dependencies"]["triton"] == "3.3.1"
    assert brief["runtime_facts"]["dependencies"]["torch"] == "2.7"
    assert brief["primary_bottleneck"]["metric"] == "normal.phases.decode_forward_seconds"
    assert brief["primary_bottleneck"]["request_time_share_pct"] == 75.0
    assert brief["model_summary"]["layer_type_counts"] == {
        "linear_attention": 3, "full_attention": 1}
    assert brief["hard_constraints"][0]["fact"] == "continuous_batching_disabled"
    assert brief["modification_directions"][0]["status"] == "blocked_missing_concurrency_evidence"
    assert [row["family"] for row in brief["top_operator_families"][:2]] == [
        "bitsandbytes_nf4_gemm", "cuda_kernel_launch"]

    no_phase = json.loads(json.dumps(context))
    no_phase["performance_observation"]["measurements"][
        "normal.phases.decode_forward_seconds"]["value"] = 0
    assert build_optimization_brief(no_phase)["primary_bottleneck"] is None

    qwen_view = _model_context(context)
    codex_view = bounded_context_view(context)
    for view in (qwen_view, codex_view):
        assert view["optimization_brief"] == brief
        assert "model_features" not in view and "hardware" not in view
        assert "layer_types" not in json.dumps(view["optimization_brief"])


def test_optimization_brief_marks_unresolved_scheduler_precondition_blocked():
    from vibeforge.agent.brief import build_optimization_brief

    context = {
        "workload": {"profiles": [{"id": "single", "concurrency": 1}]},
        "modification_surfaces": {"allowed": ["scheduler"]},
        "candidate_contract": {"attempted_strategies": [{
            "history_ref": "campaign:2", "mechanism_id": "request_scheduler_change",
            "decision": {"reason": "missing_scheduler_concurrency_evidence"},
        }]},
        "performance_observation": {"measurements": {}, "runtime": {
            "request_scheduler": "serialized_lock", "continuous_batching": False}},
    }

    brief = build_optimization_brief(context)

    assert brief["modification_directions"][0]["status"] == "blocked_missing_concurrency_evidence"
    assert brief["unresolved_preconditions"] == [{
        "history_ref": "campaign:2", "mechanism_id": "request_scheduler_change",
        "reason": "missing_scheduler_concurrency_evidence",
        "required_evidence": "diagnostic.concurrent.*.throughput from a workload profile with concurrency >= 2",
        "instruction": "Do not submit this mechanism again until the required evidence is present.",
    }]


def test_actual_performance_misattribution_is_rejected_before_analysis_submission(tmp_path):
    from vibeforge.storage.atomic import write_json
    evidence = tmp_path / "campaign"
    write_json(evidence / "iterations/0003/decision.json", {"reason": "correctness_failed"})
    write_json(evidence / "iterations/0003/correctness/report.json", {
        "passed": False, "checks": [{"case": "fixed", "passed": False,
        "generation_passed": False, "numerical": {"passed": True}}]})
    context = {"previous": {"iteration": 3, "analysis_pending": True, "decision": {"reason": "correctness_failed"}},
               "candidate_contract": {"version": 2}, "evidence_root": str(evidence)}
    submission = {}
    tool = next(t for t in _assistant_tools(tmp_path, tmp_path / "trace.jsonl", context, submission, analysis_only=True)
                if t.name == "submit_candidate")
    args = {"hypothesis": "Review iteration 3", "previous_analysis": "Iteration 3: throughput decreased by 7.6%.",
            "reflection": {"iteration": 3, "hypothesis_status": "refuted", "explanation": "Correctness failed.",
                           "next_action": "Retain SDPA.", "evidence_paths": ["iterations/0003/decision.json",
                           "iterations/0003/correctness/report.json"]}}
    result = json.loads(tool.call(json.dumps(args)))
    assert not result["submitted"] and "no formal candidate" in result["error"] and not submission
    args["previous_analysis"] = "Iteration 3: correctness failed; candidate performance was not measured."
    assert json.loads(tool.call(json.dumps(args)))["submitted"]


def test_analysis_context_summarizes_correctness_without_raw_token_arrays(tmp_path):
    from vibeforge.agent.runner import _model_context
    from vibeforge.storage.atomic import write_json
    evidence = tmp_path / "campaign"
    write_json(evidence / "iterations/0003/correctness/report.json", {
        "passed": False, "failure_summary": {"failed_checks": 1, "operational_errors": 0},
        "checks": [{"case": "fixed", "passed": False, "actual_tokens": list(range(10000)),
                    "generation_passed": False, "repeat_passed": False, "fixed_length": True,
                    "numerical": {"passed": True, "normalized_rms": 0.0}}],
    })
    context = {"evidence_root": str(evidence),
               "previous": {"iteration": 3, "analysis_pending": True,
                            "decision": {"reason": "correctness_failed"}}}
    compact = _model_context(context, analysis_only=True)
    summary = compact["correctness_summary"]
    assert summary["failure_summary"]["operational_errors"] == 0
    assert summary["failed_checks"] == [{"case": "fixed", "generation_passed": False,
        "repeat_passed": False, "fixed_length": True,
        "numerical": {"passed": True, "normalized_rms": 0.0}}]
    assert "actual_tokens" not in json.dumps(summary)


def test_analysis_context_recognizes_verification_pairing_when_profile_gate_has_no_gain(tmp_path):
    from vibeforge.agent.runner import _model_context
    context = {"evidence_root": str(tmp_path), "previous": {
        "iteration": 3, "analysis_pending": True,
        "decision": {"outcome": "invalid", "reason": "profile_regression_not_excluded:concurrent_2",
                     "gain": None},
        "experiment_verification": {"gain": 0.0017, "lower": -0.016, "upper": 0.020,
                                    "pairs": 5, "method": "paired_run_bootstrap"},
    }}
    status = _model_context(context, analysis_only=True)["review_measurement_status"]
    assert status["formal_comparison_available"] is True
    assert status["formal_comparison_source"] == "experiment_verification"
