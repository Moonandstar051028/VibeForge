import json
from pathlib import Path

import pytest

from vibeforge.agent.codex import (
    _brief,
    _bind_analysis_target,
    _bind_forced_candidate_status,
    _expand_forced_submission,
    _normalize_history_refs,
    _result_schema,
    _sync_candidate,
    run_codex_agent,
    summarize_codex_events,
)
from vibeforge.engine.responses import response_object, responses_to_chat


def test_codex_candidate_sync_rejects_campaign_artifact_paths(tmp_path):
    workspace, staging = tmp_path / "workspace", tmp_path / "staging"
    workspace.mkdir()
    (workspace / "engine.py").write_text("value = 1\n")
    (staging / "iterations" / "0002").mkdir(parents=True)
    (staging / "engine.py").write_text("value = 1\n")
    (staging / "iterations" / "0002" / "candidate.json").write_text("{}\n")

    with pytest.raises(RuntimeError, match="VibeForge artifacts"):
        _sync_candidate(staging, workspace, tmp_path / "trace.jsonl")

    assert not (workspace / "iterations").exists()


def test_shared_prompts_require_authorized_surface_audit_before_blocking(project):
    prompt = (Path(project) / "templates/prompts/codex-implementation.md").read_text()
    assert "modification_surfaces.allowed" in prompt
    assert "bitsandbytes NF4 GEMM" in prompt
    assert "native_kernel" in prompt and "build_command" in prompt
    assert "read_telemetry" in prompt


@pytest.mark.parametrize("implementation_retry,timeout_after_submission", [(False, False), (True, False), (False, True)])
def test_codex_runner_uses_local_provider_and_returns_structured_result(
        tmp_path, monkeypatch, implementation_retry, timeout_after_submission):
    executable = tmp_path / "fake-codex"
    executable.write_text("""#!/usr/bin/env python3
import json
import sys
from pathlib import Path
args = sys.argv[1:]
prompt = sys.stdin.read()
if IMPLEMENTATION_RETRY:
    marker = Path(sys.argv[0]).parent / 'first-pass-complete'
    if not marker.exists():
        marker.touch()
        print(json.dumps({'type': 'turn.completed'}))
        sys.exit(0)
    assert 'IMPLEMENTATION STAGE' in prompt
    assert 'Current candidate source' in prompt and 'value = 1' in prompt
    assert 'mcp_servers.vibeforge_submit.enabled_tools=["read_candidate","replace_candidate","write_candidate","search_candidate","check_candidate","record_result"]' in args
assert Path('.vibeforge/context.json').is_file()
mounted_context = json.loads(Path('.vibeforge/context.json').read_text())
assert mounted_context['iteration'] == 1
Path('argv.json').write_text(json.dumps(args))
Path('prompt.txt').write_text(prompt)
Path('engine.py').write_text("value = 2\\n")
output = Path(args[args.index('--output-last-message') + 1])
submission = {
    'hypothesis': 'Reduce fixture overhead.',
    'previous_analysis': 'Initial iteration.',
    'summary': 'Changed the candidate fixture.',
    'blocked': False,
    'blocked_reason': '',
    'reflection': {'iteration': 1, 'hypothesis_status': 'inconclusive',
                   'explanation': 'Await host measurement.', 'next_action': 'Measure the candidate.',
                   'evidence_paths': []},
    'experiment_plan': {
        'measured_bottleneck': 'fixture overhead', 'cause': 'constant work',
        'change': 'replace the constant', 'expected_change_pct': 1,
        'validation': 'run frozen checks', 'history_refs': [], 'bottleneck_metric': 'normal.backend_wall_seconds'
    }
}
mcp_config = next(args[index + 1] for index, value in enumerate(args[:-1])
                  if value == '-c' and args[index + 1].startswith('mcp_servers.vibeforge_submit.args='))
mcp_args = json.loads(mcp_config.split('=', 1)[1])
submission_path = Path(mcp_args[mcp_args.index('--output') + 1])
submission_path.write_text(json.dumps(submission))
output.write_text('Submission recorded.')
print(json.dumps({'type': 'turn.completed'}))
""".replace("IMPLEMENTATION_RETRY", repr(implementation_retry)))
    executable.chmod(0o755)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "engine.py").write_text("value = 1\n")
    trace = tmp_path / "artifacts" / "commands.jsonl"
    context = {
        "model": {"name": "local-model", "max_context": 32768},
        "knowledge_scope": "current_campaign",
        "agent": {"framework": "codex", "executable": str(executable), "timeout_seconds": 10,
                  "max_new_tokens": 4096},
        "iteration": 1,
    }

    if timeout_after_submission:
        import vibeforge.agent.codex as codex_module
        original = codex_module._run_codex_process
        def timed_out_after_record(*args, **kwargs):
            completed, _ = original(*args, **kwargs)
            completed.returncode = 124
            return completed, True
        monkeypatch.setattr(codex_module, '_run_codex_process', timed_out_after_record)
    result = run_codex_agent("http://127.0.0.1:1234", context, workspace, trace, "Optimize.")
    assert "features.shell_tool=false" in json.loads((workspace / "argv.json").read_text())

    assert result["mode"] == "codex" and result["complete"]
    assert result["submission_origin"] == "mcp_submit"
    assert (workspace / "engine.py").read_text() == "value = 2\n"
    arguments = json.loads((workspace / "argv.json").read_text())
    assert "model_providers.vibeforge_local.base_url=\"http://127.0.0.1:1234/v1\"" in arguments
    assert arguments[arguments.index("--sandbox") + 1] == "workspace-write"
    assert "--output-schema" not in arguments
    assert "mcp_servers.vibeforge_submit.required=true" in arguments
    bridge = next(arg.split('=', 1)[1] for arg in arguments
                  if arg.startswith('mcp_servers.vibeforge_submit.args='))
    bridge_args = json.loads(bridge)
    assert bridge_args[0] == '-c'
    assert 'sys.path.insert(0,' in bridge_args[1]
    assert str(Path(__file__).resolve().parents[2] / 'src') in bridge_args[1]
    enabled = next(arg.split('=', 1)[1] for arg in arguments
                   if arg.startswith('mcp_servers.vibeforge_submit.enabled_tools='))
    assert set(json.loads(enabled)) == {'record_result', 'read_candidate', 'search_candidate', 'replace_candidate', 'write_candidate', 'check_candidate'}
    assert "model_auto_compact_token_limit=23552" in arguments
    assert "tool_output_token_limit=4096" in arguments
    rendered_prompt = (workspace / "prompt.txt").read_text()
    if not implementation_retry:
        assert ".vibeforge/brief.json" in rendered_prompt
        assert ".vibeforge/context.json" in rendered_prompt
        assert "never cat or print that entire file" in rendered_prompt
    assert "record_result" in rendered_prompt
    assert "complete bounded brief is supplied below" in rendered_prompt
    assert not (workspace / ".vibeforge").exists()
    summary = json.loads((tmp_path / "artifacts" / "codex-context.json").read_text())
    assert summary["context_owner"] == "codex"
    assert summary["vibeforge_prompt_compaction"] is False
    assert summary["codex_auto_compact_token_limit"] == 23552


def test_codex_event_summary_records_compaction_and_usage():
    stdout = "\n".join([
        json.dumps({"type": "thread.started"}),
        json.dumps({"type": "context.compacted", "usage": {"input_tokens": 9000, "output_tokens": 400}}),
        json.dumps({"type": "item.completed", "item": {"type": "error",
            "message": "Heads up: Long threads and multiple compactions can reduce accuracy."}}),
        json.dumps({"type": "turn.completed", "usage": {"input_tokens": 1200, "output_tokens": 80}}),
    ])

    summary = summarize_codex_events(stdout)

    assert summary["compaction_count"] == 2
    assert summary["compaction_events"][0]["kind"] == "context.compacted"
    assert len(summary["usage_snapshots"]) == 2


def test_codex_brief_is_a_bounded_decision_packet_with_counterexamples():
    context = {
        "context_schema_version": 4, "campaign_id": "parity", "iteration": 3,
        "requirements": {"objective": "throughput"},
        "loop_protocol": {"profiling": {"enabled": True}},
        "knowledge": [{"rule_id": "component", "implementation": {"scope": "component_only"}}],
        "transfer_hints": [{"campaign_id": "older", "scope": "advisory"}],
        "optimization_brief": {
            "current_state": {"stable_ref": "stable-v1"},
            "do_not_repeat": [{"mechanism_id": "cache_implementation_switch",
                               "history_ref": "parity:1"}],
            "model_summary": {"model_type": "qwen3_5"},
        },
        "candidate_contract": {"observation_paths": ["iterations/0003/observation.json"],
                               "history_refs": ["parity:1", "parity:2"]},
    }

    brief = _brief(context)

    assert brief["decision_packet"]["current_state"]["stable_ref"] == "stable-v1"
    assert brief["candidate_contract"]["allowed_history_refs"] == ["parity:1", "parity:2"]
    assert "static" in brief["candidate_constraints"]["transformers_cache_implementations"]
    assert brief["knowledge"] == context["knowledge"]
    assert brief["transfer_hints"] == context["transfer_hints"]
    assert "performance_observation" not in brief and "history" not in brief
    assert len(json.dumps(brief)) < 12_000
    assert set(brief["agent_interface"]["tools"]) >= {
        "record_result", "list_evidence", "read_evidence", "search_history",
    }


def test_codex_schema_and_host_normalize_history_paths_to_canonical_ids():
    context = {"candidate_contract": {"history_refs": ["campaign-a:1", "campaign-a:2"]}}
    result = {"experiment_plan": {
        "history_refs": ["iterations/0002/decision.json", "campaign-a:0001", "unknown:9"]}}

    normalized = _normalize_history_refs(result, context)
    schema = _result_schema(False, False, ["campaign-a:1", "campaign-a:2"])

    assert normalized["experiment_plan"]["history_refs"] == ["campaign-a:2", "campaign-a:1"]
    assert schema["properties"]["experiment_plan"]["properties"]["history_refs"]["items"]["enum"] == [
        "campaign-a:1", "campaign-a:2"]


def test_analysis_reflection_uses_host_selected_pending_iteration():
    result = {"reflection": {"iteration": 99, "hypothesis_status": "untested",
                             "evidence_paths": []}}

    bound = _bind_analysis_target(
        result, {"previous": {"iteration": 7}}, analysis_only=True
    )
    assert bound["reflection"]["iteration"] == 7
    assert bound["reflection"]["hypothesis_status"] == "inconclusive"
    assert bound["reflection"]["evidence_paths"] == ["iterations/0007/decision.json"]
    assert _bind_analysis_target(
        result, {"previous": {"iteration": 3}}, analysis_only=False
    )["reflection"]["iteration"] == 7


def test_analysis_reflection_binds_correctness_report_when_required():
    result = {"reflection": {"iteration": 1, "evidence_paths": []}}
    context = {"previous": {"iteration": 4, "decision": {"reason": "correctness_failed"}}}

    bound = _bind_analysis_target(result, context, analysis_only=True)

    assert bound["reflection"]["evidence_paths"] == [
        "iterations/0004/correctness/report.json"
    ]


def test_forced_submission_treats_existing_diff_as_testable_candidate():
    result = {"blocked": True, "blocked_reason": "No post-change benchmark exists yet."}

    bound = _bind_forced_candidate_status(result, ["engine.json"], analysis_only=False)

    assert bound["blocked"] is False
    assert bound["blocked_reason"] == ""


def test_forced_submission_blocks_when_no_candidate_diff_exists():
    result = {"blocked": False, "blocked_reason": ""}

    bound = _bind_forced_candidate_status(result, [], analysis_only=False)

    assert bound["blocked"] is True
    assert "no candidate file changes" in bound["blocked_reason"]


def test_codex_submission_mcp_validates_and_records_once(tmp_path):
    import subprocess
    import sys

    schema = {
        "type": "object",
        "properties": {"ready": {"type": "boolean"}},
        "required": ["ready"],
        "additionalProperties": False,
    }
    schema_path = tmp_path / "schema.json"
    output_path = tmp_path / "submission.json"
    workspace = tmp_path / "workspace"
    evidence = tmp_path / "campaign"
    workspace.mkdir()
    (evidence / "iterations" / "0001").mkdir(parents=True)
    (evidence / "iterations" / "0001" / "decision.json").write_text('{"outcome":"reject"}')
    (evidence / "memory").mkdir()
    (evidence / "memory" / "index.json").write_text(json.dumps({"entries": [{
        "iteration": 1, "hypothesis": "remove synchronization", "facts": {"decision": {"outcome": "reject"}}
    }]}))
    context_path = tmp_path / "context.json"
    context_path.write_text(json.dumps({"evidence_root": str(evidence)}))
    schema_path.write_text(json.dumps(schema))
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2025-06-18"}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {
         "name": "read_evidence", "arguments": {"path": "iterations/0001/decision.json"}}},
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {
         "name": "search_history", "arguments": {"query": "synchronization"}}},
        {"jsonrpc": "2.0", "id": 5, "method": "tools/call",
         "params": {"name": "record_result", "arguments": {"ready": True}}},
        {"jsonrpc": "2.0", "id": 6, "method": "tools/call",
         "params": {"name": "record_result", "arguments": {"ready": False}}},
    ]

    completed = subprocess.run(
        [sys.executable, "-m", "vibeforge.agent.codex_submit",
         "--schema", str(schema_path), "--output", str(output_path),
         "--context", str(context_path), "--workspace", str(workspace),
         "--trace", str(tmp_path / "commands.jsonl")],
        input="\n".join(json.dumps(value) for value in requests) + "\n",
        text=True, capture_output=True, check=True,
    )
    responses = [json.loads(line) for line in completed.stdout.splitlines()]

    assert responses[0]["result"]["protocolVersion"] == "2025-06-18"
    assert {tool["name"] for tool in responses[1]["result"]["tools"]} == {
        "record_result", "list_evidence", "read_evidence", "search_history",
        "read_candidate", "search_candidate", "replace_candidate", "write_candidate", "check_candidate",
    }
    assert '"outcome":"reject"' in responses[2]["result"]["content"][0]["text"]
    assert "remove synchronization" in responses[3]["result"]["content"][0]["text"]
    assert responses[4]["result"]["isError"] is False
    assert responses[5]["result"]["isError"] is True
    assert json.loads(output_path.read_text()) == {"ready": True}


def test_forced_submission_expands_flat_local_model_arguments():
    result = _expand_forced_submission({
        "hypothesis": "Remove a redundant synchronization.",
        "previous_analysis": "", "summary": "Removed one synchronization call.",
        "blocked": False, "blocked_reason": "", "reflection_iteration": 1,
        "reflection_hypothesis_status": "inconclusive",
        "reflection_explanation": "Await paired measurement.",
        "reflection_next_action": "Run the frozen evaluation.",
        "reflection_evidence_paths_json": "[]",
        "plan_measured_bottleneck": "host synchronization overhead",
        "plan_cause": "an extra synchronization before generation",
        "plan_change": "remove the pre-generation synchronization",
        "plan_expected_change_pct": 1.0,
        "plan_validation": "paired correctness and throughput evaluation",
        "plan_history_refs_json": "[]", "plan_bottleneck_metric": "normal.backend_wall_seconds",
    }, analysis_only=False)

    assert result["blocked"] is False
    assert result["reflection"]["evidence_paths"] == []
    assert result["experiment_plan"]["expected_change_pct"] == 1.0


def test_codex_recovers_when_local_model_ignores_output_schema(tmp_path):
    executable = tmp_path / "fake-codex"
    executable.write_text("""#!/usr/bin/env python3
import json
import sys
from pathlib import Path
args = sys.argv[1:]
output = Path(args[args.index('--output-last-message') + 1])
output.write_text(json.dumps({
    'hypothesis': 'This final message must not be accepted.',
    'previous_analysis': '', 'summary': 'Not submitted.', 'blocked': False,
    'blocked_reason': '',
    'reflection': {'iteration': 1, 'hypothesis_status': 'accepted',
                   'explanation': 'invalid authority', 'next_action': 'none',
                   'evidence_paths': []}
}))
print('{"type":"turn.completed","usage":{"input_tokens":10,"output_tokens":2}}')
""")
    executable.chmod(0o755)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "engine.py").write_text("value = 1\n")
    context = {
        "model": {"name": "local-model", "max_context": 32768},
        "agent": {"framework": "codex", "executable": str(executable), "max_new_tokens": 4096},
        "iteration": 2,
        "previous": {"iteration": 1},
    }

    result = run_codex_agent("http://127.0.0.1:1234", context, workspace,
                             tmp_path / "artifacts" / "commands.jsonl", "Review.", analysis_only=True)

    assert result["submission_origin"] == "host_recovery"
    assert result["blocked"] is True
    assert result["reflection"]["iteration"] == 1
    assert "Connection refused" in result["blocked_reason"]
    records = [json.loads(line) for line in
               (tmp_path / "artifacts" / "commands.jsonl").read_text().splitlines()]
    assert len([row for row in records if row.get("event") == "codex_pass_completed"]) == 1


def test_analysis_timeout_uses_host_structured_submission_fallback(tmp_path, monkeypatch):
    executable = tmp_path / "fake-codex"
    executable.write_text("#!/usr/bin/env python3\nimport time\ntime.sleep(2)\n")
    executable.chmod(0o755)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "engine.py").write_text("value = 1\n")
    context = {
        "model": {"name": "local-model", "max_context": 32768},
        "agent": {"framework": "codex", "executable": str(executable),
                  "timeout_seconds": 0.05, "max_new_tokens": 512},
        "iteration": 2,
        "previous": {"iteration": 1, "decision": {"reason": "no_significant_gain"}},
    }
    fallback = {
        "hypothesis": "Review the rejected cache experiment.",
        "previous_analysis": "The paired interval crossed zero.",
        "summary": "The predicted cache improvement was not observed.",
        "blocked": False,
        "blocked_reason": "",
        "reflection": {
            "iteration": 1, "hypothesis_status": "refuted",
            "explanation": "The mechanism diagnostic regressed.",
            "next_action": "Do not repeat this cache strategy.",
            "evidence_paths": ["iterations/0001/decision.json"],
        },
    }
    monkeypatch.setattr("vibeforge.agent.codex._forced_submission",
                        lambda *args, **kwargs: fallback)

    result = run_codex_agent(
        "http://127.0.0.1:1234", context, workspace,
        tmp_path / "artifacts" / "commands.jsonl", "Review.", analysis_only=True)

    assert result["submission_origin"] == "forced_tool_submit"
    assert result["reflection"]["hypothesis_status"] == "refuted"
    records = [json.loads(line) for line in
               (tmp_path / "artifacts" / "commands.jsonl").read_text().splitlines()]
    assert any(row.get("event") == "codex_analysis_timeout_fallback" for row in records)


def test_responses_adapter_combines_codex_system_constraints():
    converted = responses_to_chat({
        "instructions": "Use the provided tools.",
        "text": {"format": {"type": "json_schema", "schema": {
            "type": "object", "properties": {"ready": {"type": "boolean"}}
        }}},
        "input": [
            {"type": "message", "role": "developer", "content": "Respect the workspace."},
            {"type": "message", "role": "user", "content": "Return readiness."},
        ],
    })

    assert [message["role"] for message in converted["messages"]] == ["system", "user"]
    assert "Use the provided tools." in converted["messages"][0]["content"]
    assert "Respect the workspace." in converted["messages"][0]["content"]
    assert "Return only one JSON object" in converted["messages"][0]["content"]


def test_responses_adapter_marks_unclosed_thinking_failed():
    response = response_object({"model": "local"}, {
        "text": "", "reasoning_content": "still reasoning",
        "finish_reason": "incomplete_thinking",
        "thinking": {"enabled": True, "complete": False},
        "usage": {"prompt_tokens": 10, "completion_tokens": 4,
                  "completion_tokens_details": {"reasoning_tokens": 4}},
    })

    assert response["status"] == "failed"
    assert response["incomplete_details"] is None
    assert response["error"]["code"] == "server_error"
    assert response["error"]["type"] == "incomplete_thinking"


def test_responses_namespace_tools_roundtrip_and_forced_choice():
    request = {
        'tools': [{'type': 'namespace', 'name': 'mcp__vibeforge_submit', 'tools': [
            {'type': 'function', 'name': 'read_candidate', 'parameters': {'type': 'object'}}]}],
        'input': [{'type': 'function_call', 'namespace': 'mcp__vibeforge_submit',
                   'name': 'read_candidate', 'call_id': 'call_read', 'arguments': '{"path":"engine.py"}'},
                  {'type': 'function_call_output', 'call_id': 'call_read', 'output': 'value = 1'}],
        'tool_choice': {'type': 'function', 'namespace': 'mcp__vibeforge_submit', 'name': 'read_candidate'},
    }
    name = 'mcp__vibeforge_submit__read_candidate'
    chat = responses_to_chat(request)
    assert chat['tools'][0]['function']['name'] == name
    assert chat['messages'][0]['tool_calls'][0]['function']['name'] == name
    assert chat['messages'][1]['content'] == 'value = 1'
    assert chat['tool_choice']['function']['name'] == name
    response = response_object(request, {'tool_calls': [{'id': 'call_read', 'name': name,
                                                       'arguments': {'path': 'engine.py'}}]})
    assert response['output'][0]['namespace'] == 'mcp__vibeforge_submit'
    assert response['output'][0]['name'] == 'read_candidate'
    assert response['output'][0]['call_id'] == 'call_read'


def test_forced_no_edit_does_not_recycle_a_historical_environment_failure():
    result = _bind_forced_candidate_status({
        'blocked_reason': 'Sandbox still broken', 'summary': 'Cannot read files',
        'reflection': {'hypothesis_status': 'refuted', 'explanation': 'Sandbox still broken'},
    }, [], False)
    assert result['blocked'] is True
    assert 'Sandbox' not in result['blocked_reason']
    assert 'incomplete implementation' in result['reflection']['explanation']
    assert result['reflection']['hypothesis_status'] == 'inconclusive'


def test_codex_smoke_json_allows_only_a_complete_outer_fence(tmp_path):
    from vibeforge.agent.codex import _read_json_message
    path = tmp_path / 'message.json'
    path.write_text('```json\n{"ready":true}\n```')
    assert _read_json_message(path) == {'ready': True}
    path.write_text('Some commentary\n```json\n{"ready":true}\n```')
    with pytest.raises(ValueError):
        _read_json_message(path)


def test_codex_mcp_reserves_edit_and_submission_after_exploration(tmp_path, monkeypatch, capsys):
    import io
    from vibeforge.agent.codex_submit import serve
    work = tmp_path / 'work'
    work.mkdir()
    (work / 'engine.py').write_text('value = 1')
    schema = tmp_path / 'schema.json'
    schema.write_text('{"type":"object"}')
    context = tmp_path / 'context.json'
    context.write_text('{"agent":{"max_turns":6}}')
    calls = [('read_candidate', {'path': 'engine.py'})] * 3 + [
        ('replace_candidate', {'path': 'engine.py', 'old': 'value = 1', 'new': 'value = 2'}),
        ('check_candidate', {'path': 'engine.py'}), ('record_result', {})]
    monkeypatch.setattr('sys.stdin', io.StringIO('\n'.join(json.dumps({
        'id': i, 'method': 'tools/call', 'params': {'name': name, 'arguments': args}
    }) for i, (name, args) in enumerate(calls))))
    serve(schema, tmp_path / 'result.json', context, work, tmp_path / 'trace.jsonl')
    replies = [json.loads(line)['result'] for line in capsys.readouterr().out.splitlines()]
    assert [r['isError'] for r in replies] == [False, False, True, False, False, False]
    assert (work / 'engine.py').read_text() == 'value = 2'


def test_codex_submission_schema_rejects_negative_gain_and_unmeasured_success():
    from jsonschema import validate, ValidationError
    schema = _result_schema(bottleneck_metrics=['normal.phases.decode_forward_seconds'])
    result = {
        'hypothesis': 'Compile helper', 'previous_analysis': '', 'summary': 'Candidate only',
        'blocked': False, 'blocked_reason': '',
        'reflection': {'iteration': 1, 'hypothesis_status': 'inconclusive', 'explanation': 'Unmeasured',
                       'next_action': 'Evaluate', 'evidence_paths': []},
        'experiment_plan': {'measured_bottleneck': 'decode', 'cause': 'launches', 'change': 'fuse',
                            'expected_change_pct': 10, 'validation': 'paired', 'history_refs': [],
                            'bottleneck_metric': 'normal.phases.decode_forward_seconds'},
    }
    validate(result, schema)
    result['experiment_plan']['bottleneck_metric'] = 'throughput'
    with pytest.raises(ValidationError):
        validate(result, schema)
    result['experiment_plan']['bottleneck_metric'] = 'normal.phases.decode_forward_seconds'
    for value in (-50, 0):
        result['experiment_plan']['expected_change_pct'] = value
        with pytest.raises(ValidationError):
            validate(result, schema)
    result['experiment_plan']['expected_change_pct'] = 10
    result['reflection']['hypothesis_status'] = 'implemented_pending_validation'
    with pytest.raises(ValidationError):
        validate(result, schema)


@pytest.mark.parametrize('analysis_only', [False, True])
def test_codex_mcp_candidate_edits_are_confined_and_analysis_is_read_only(tmp_path, monkeypatch, capsys, analysis_only):
    import io
    from vibeforge.agent.codex_submit import serve
    workspace = tmp_path / 'candidate'
    workspace.mkdir()
    (workspace / 'engine.py').write_text('value = 1\n')
    (workspace / '.vibeforge').mkdir()
    (workspace / '.vibeforge' / 'brief.json').write_text('original')
    (workspace / 'alias').symlink_to(workspace / '.vibeforge', target_is_directory=True)
    schema = tmp_path / 'schema.json'
    schema.write_text('{"type":"object"}')
    context = tmp_path / 'context.json'
    context.write_text(json.dumps({'analysis_only': analysis_only}))
    calls = [('read_candidate', {'path': '.vibeforge/brief.json'}),
             ('replace_candidate', {'path': 'engine.py', 'old': 'value = 1', 'new': 'value = 2'}),
             ('replace_candidate', {'path': 'alias/brief.json', 'old': 'original', 'new': 'bad'}),
             ('replace_candidate', {'path': '.vibeforge/../.vibeforge/brief.json', 'old': 'original', 'new': 'bad'}),
             ('read_candidate', {'path': '../schema.json'}),
             ('replace_candidate', {'path': 'engine.py', 'old': 'value = 2', 'new': 'value = ('}),
             ('check_candidate', {'path': 'engine.py'}),
             ('record_result', {}),
             ('replace_candidate', {'path': 'engine.py', 'old': 'value = (', 'new': 'value = 3'})]
    requests = [{'jsonrpc': '2.0', 'id': i, 'method': 'tools/call',
                 'params': {'name': name, 'arguments': args}} for i, (name, args) in enumerate(calls)]
    monkeypatch.setattr('sys.stdin', io.StringIO('\n'.join(json.dumps(r) for r in requests)))
    serve(schema, tmp_path / 'result.json', context, workspace, tmp_path / 'trace.jsonl')
    replies = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert replies[0]['result']['isError'] is False
    assert replies[4]['result']['isError'] is True
    assert (workspace / '.vibeforge/brief.json').read_text() == 'original'
    if analysis_only:
        assert all('error' in replies[i] for i in [1, 2, 3, 5, 6, 8])
        assert (workspace / 'engine.py').read_text() == 'value = 1\n'
    else:
        assert replies[1]['result']['isError'] is False
        assert all(replies[i]['result']['isError'] for i in [2, 3, 6, 8])
        assert replies[7]['result']['isError'] is False  # syntax error did not kill MCP


@pytest.mark.parametrize('analysis_only', [False, True])
def test_codex_write_helper_is_confined_and_frozen_after_submit(tmp_path, monkeypatch, capsys, analysis_only):
    import io
    from vibeforge.agent.codex_submit import serve
    work = tmp_path / 'work'
    work.mkdir()
    (work / '.vibeforge').mkdir()
    (work / '.vibeforge/brief.json').write_text('protected')
    (work / 'alias').symlink_to(work / '.vibeforge', target_is_directory=True)
    schema = tmp_path / 'schema.json'
    schema.write_text('{"type":"object"}')
    context = tmp_path / 'context.json'
    context.write_text(json.dumps({'analysis_only': analysis_only, 'agent': {'max_turns': 5}}))
    calls = [('read_candidate', {'path': '.vibeforge/brief.json'})] * 2
    calls += [('write_candidate', {'path': name, 'content': 'value = 2\n'}) for name in (
        '../escape.py', 'alias/brief.json', '.vibeforge/brief.json', 'helpers/new.py')]
    calls += [('check_candidate', {'path': 'helpers/new.py'}), ('record_result', {}),
              ('write_candidate', {'path': 'helpers/new.py', 'content': 'value = 3\n'})]
    monkeypatch.setattr('sys.stdin', io.StringIO('\n'.join(json.dumps({
        'id': i, 'method': 'tools/call', 'params': {'name': name, 'arguments': args}
    }) for i, (name, args) in enumerate(calls))))
    serve(schema, tmp_path / 'result.json', context, work, tmp_path / 'trace.jsonl')
    replies = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert replies[1]['result']['isError']  # exploration exhausted
    assert not (tmp_path / 'escape.py').exists()
    assert (work / '.vibeforge/brief.json').read_text() == 'protected'
    if analysis_only:
        assert all('error' in replies[i] for i in (2, 3, 4, 5, 6, 8))
        assert not (work / 'helpers/new.py').exists()
    else:
        assert all(replies[i]['result']['isError'] for i in (2, 3, 4, 8))
        assert all(not replies[i]['result']['isError'] for i in (5, 6, 7))
        assert (work / 'helpers/new.py').read_text() == 'value = 2\n'


def test_revision_brief_keeps_decision_without_profiler_dump():
    from vibeforge.agent.codex import _probe_feedback_summary
    context = {'candidate_feedback_path': 'iterations/0003/probe/probe.json',
        'candidate_feedback': {'candidate': 'candidate-v1', 'admission': False,
            'feedback_state': 'revise', 'needs_revision': True,
            'mechanism': {'metric': 'decode_seconds', 'improvement_pct': -4.5},
            'correctness_precheck': {'passed': False, 'checks': [
                {'case': 'concurrent', 'passed': False, 'error': 'wrong output',
                 'actual_tokens': list(range(10000))}],
                'runtime_observation': {'top_cpu_ops': ['large kernel'] * 10000}},
            'profile_reports': [{'top_cpu_ops': ['large kernel'] * 10000}]}}
    summary = _probe_feedback_summary(context)
    assert summary['mechanism']['improvement_pct'] == -4.5
    assert summary['correctness_precheck']['checks'][0]['error'] == 'wrong output'
    assert summary['evidence_path'] == context['candidate_feedback_path']
    assert len(json.dumps(_brief(context))) < 5000
    assert len(context['candidate_feedback']['profile_reports'][0]['top_cpu_ops']) == 10000


def test_submission_fallback_receives_current_evidence_not_only_previous(tmp_path, monkeypatch):
    from vibeforge.agent.codex import _forced_submission
    work, staging = tmp_path / 'original', tmp_path / 'staging'
    work.mkdir(); staging.mkdir()
    (work / 'engine.py').write_text('value = 1')
    (staging / 'engine.py').write_text('value = 2')
    class Captured(Exception):
        pass
    def request(self, endpoint, payload):
        evidence = json.loads(payload['messages'][1]['content'])
        assert 'value = 2' in evidence['candidate_diff']
        assert evidence['current_evidence']['decision_packet']['measured_fact'] == 12
        assert 'CURRENT diff' in evidence['instruction']
        assert evidence['previous']['hypothesis'] == 'old unrelated hypothesis'
        raise Captured
    monkeypatch.setattr('vibeforge.agent.codex.EngineClient.request', request)
    with pytest.raises(Captured):
        _forced_submission('http://localhost:1', {
            'optimization_brief': {'measured_fact': 12},
            'previous': {'hypothesis': 'old unrelated hypothesis'},
        }, staging, work, False, '', 1)
