import pytest
import vibeforge.controller.supervisor as supervisor_module

from vibeforge.controller.supervisor import Supervisor, probe_prediction
from vibeforge.storage.atomic import digest, read_json, write_json


def test_invalid_candidate_config_does_not_count_as_infrastructure_failure(paths, toy_config):
    toy_config['budget'].update(max_iterations=1, max_proposals=1)
    toy_config['config_hash'] = digest({key: value for key, value in toy_config.items() if key != 'config_hash'})
    controller = Supervisor(paths, 'invalid-runtime')
    original = controller.agent_session

    def corrupt_configuration(client, workspace, iteration, analysis_only=False, **kwargs):
        result = original(client, workspace, iteration, analysis_only, **kwargs)
        if not analysis_only:
            config = read_json(workspace / 'engine.json')
            config.pop('device')
            write_json(workspace / 'engine.json', config)
        return result

    controller.agent_session = corrupt_configuration
    try:
        controller.create(toy_config)
        controller.run()
        state = controller.database.get('invalid-runtime')
        decision = read_json(paths.campaign('invalid-runtime') / 'iterations/0001/decision.json')
        assert decision['reason'] == 'invalid_candidate_runtime'
        assert 'device' in decision['error']
        assert state['completed_experiments'] == 0
        assert state['consecutive_failures'] == 0
        assert state['proposal_attempts'] == 1
        assert state['stable_ref'] == state['initial_ref']
        assert state['analysis_pending'] is False
        assert state['state'] == 'STOPPED'
    finally:
        controller.close()


def test_two_rounds_accept_good_change_reject_wrong_change_and_resume(paths, toy_config, monkeypatch):
    controller = Supervisor(paths, "closed-loop")
    original_run = supervisor_module.run_command

    def annotate_planning_result(command, *args, **kwargs):
        result = original_run(command, *args, **kwargs)
        result_path = paths.campaign("closed-loop") / "iterations/0002/agent-result.json"
        if str(result_path) in command and result_path.is_file():
            value = read_json(result_path)
            value["previous_analysis"] += " Planning-only note."
            write_json(result_path, value)
        return result

    monkeypatch.setattr(supervisor_module, "run_command", annotate_planning_result)
    try:
        controller.create(toy_config)
        controller.run()
        state = controller.database.get("closed-loop")
        iterations = controller.database.iterations("closed-loop")
        assert state["state"] == "STOPPED"
        assert state["analysis_pending"] is False
        assert iterations[0]["decision"]["outcome"] == "accept"
        assert iterations[1]["decision"]["reason"] == "correctness_failed"
        assert iterations[0]["experiment_verification"]["status"] == "supported"
        assert (paths.campaign("closed-loop") / "iterations/0001/verification.json").is_file()
        assert state["stable_ref"] == iterations[0]["candidate"]
        assert state["stable_ref"] != iterations[1]["candidate"]
        assert state["previous_stable_ref"] == state["initial_ref"]
        second_context = read_json(paths.campaign("closed-loop") / "iterations/0002/context.json")
        assert second_context["model"]["max_context"] == read_json(
            paths.workspace("closed-loop", 2) / "engine.json").get("max_context", 4096)
        assert "# Iteration 1" in second_context["previous_analysis"]
        assert second_context["history"][0]["hypothesis"] == iterations[0]["hypothesis"]
        assert "119999 * 120000 // 2" in (paths.campaign("closed-loop") / second_context["history"][0]["evidence_path"] / "change.patch").read_text()
        assert "Planning-only note." not in iterations[0]["interpretation"]
        assert "Planning-only note." in read_json(paths.campaign("closed-loop") / "iterations/0002/agent-result.json")["previous_analysis"]
        controller.run(resume=True)
        assert controller.database.get("closed-loop")["completed_iterations"] == 2
        events = list(controller.database.events("closed-loop"))
        sessions = [e for e in events if e["event_type"] == "agent_completed"]
        assert len({e["payload"]["session_id"] for e in sessions}) == 4
        assert state["proposal_attempts"] == state["completed_experiments"] == 2
        assert not second_context["pending_analyses"]
        assert second_context["history"][0]["hypothesis_status"] == iterations[0]["reflection"]["hypothesis_status"]
        assert (paths.campaign("closed-loop") / "summary.md").exists()
        from vibeforge.controller.comparison import compare_initial
        comparison = compare_initial(paths, "closed-loop")
        assert comparison["decision"]["outcome"] == "accept"
        assert comparison["initial_ref"] == state["initial_ref"]
        assert comparison["stable_ref"] == state["stable_ref"]
        assert read_json(paths.campaign("closed-loop") / "progress.json")["cumulative_comparison"]
        assert controller.database.get("closed-loop")["completed_experiments"] == 2
    finally:
        controller.close()


def test_controller_starts_qwen_agent_worker_and_final_analysis(paths, toy_config):
    toy_config["budget"]["max_proposals"] = 1
    toy_config["agent"] = {
        "mode": "local", "framework": "qwen-agent", "max_turns": 4,
        "max_new_tokens": 256, "timeout_seconds": 20, "request_timeout": 10,
        "prompt": "templates/prompts/implementation.md",
    }
    toy_config["budget"]["max_iterations"] = 1
    toy_config["evaluation"]["max_retests"] = 0
    toy_config["config_hash"] = digest({key: value for key, value in toy_config.items() if key != "config_hash"})
    controller = Supervisor(paths, "qwen-agent-worker")
    try:
        controller.create(toy_config)
        state = controller.run()
        assert state["state"] == "STOPPED" and not state["analysis_pending"]
        directory = paths.campaign("qwen-agent-worker")
        for path in (directory / "iterations/0001/agent-result.json", directory / "final-analysis/agent-result.json"):
            result = read_json(path)
            assert result["mode"] == "qwen_agent" and result["complete"]
        telemetry = read_json(directory / "iterations/0001/agent-telemetry.json")
        assert telemetry["request_count"] > 0
        assert telemetry["operations"]["chat_completion"] > 0
        context = read_json(directory / "iterations/0001/context.json")
        assert context["telemetry_session"]["admission"] is False
        assert read_json(directory / "bootstrap/agent-smoke.json")["framework"] == "Qwen-Agent"
        assert (directory / "bootstrap/agent-smoke.jsonl").exists()
        assert not (paths.data / "state/qwen-agent-worker.agent.json").exists()
    finally:
        controller.close()


def test_probe_feedback_drives_one_revision_before_formal_evaluation(paths, toy_config):
    toy_config["agent"] = {
        "mode": "scripted", "prompt": "templates/prompts/implementation.md",
        "actions": [{
            "hypothesis": "Deliberately perturb logits to trigger the quick correctness probe.",
            "edits": [{"tool": "replace_text", "arguments": {
                "path": "engine.py", "old": "seed = sum(", "new": "seed = 1 + sum("}}],
            "revision": {
                "hypothesis": "Restore logits and remove the measured fixed arithmetic overhead.",
                "edits": [
                    {"tool": "replace_text", "arguments": {
                        "path": "engine.py", "old": "seed = 1 + sum(", "new": "seed = sum("}},
                    {"tool": "replace_text", "arguments": {
                        "path": "engine.py", "old": "work = sum(range(120000))",
                        "new": "work = 119999 * 120000 // 2"}},
                ],
            },
        }],
    }
    toy_config["probe"].update(enabled=True, repeats=1, warmup=1, max_attempts=2)
    toy_config["refinement"].update(enabled=True, max_revisions=1)
    toy_config["budget"].update(max_iterations=1, max_proposals=1)
    toy_config["evaluation"]["max_retests"] = 0
    toy_config["config_hash"] = digest({key: value for key, value in toy_config.items()
                                        if key != "config_hash"})
    controller = Supervisor(paths, "probe-refinement")
    try:
        controller.create(toy_config)
        state = controller.run()
        folder = paths.campaign("probe-refinement") / "iterations/0001"

        assert state["state"] == "STOPPED"
        assert state["probe_attempts"] == 2 and state["refinement_attempts"] == 1
        first = read_json(folder / "probe/0000/probe.json")
        second = read_json(folder / "probe/0001/probe.json")
        assert first["correctness_precheck"]["passed"] is False and first["needs_revision"]
        assert first["feedback_state"] == "revise" and first["admission"] is False
        assert second["correctness_precheck"]["passed"] is True
        assert first["warmup_runs"] == 0 and second["warmup_runs"] == 1
        assert first["profile_reports"] == []
        assert read_json(folder / "refinement/0001/outcome.json")["applied"] is True
        revision_context = read_json(folder / "refinement/0001/context.json")
        target = revision_context["refinement_target"]
        assert target["candidate"] == first["candidate"]
        assert target["stable_parent"] == state["initial_ref"]
        assert "seed = 1 + sum(" in target["source_patch"]
        assert target["hypothesis"] and target["experiment_plan"]["change"]
        assert target["source_patch_truncated"] is False
        assert read_json(folder / "decision.json")["outcome"] == "accept"
        patch = (folder / "change.patch").read_text()
        assert "119999 * 120000 // 2" in patch and "seed = 1 + sum(" not in patch
    finally:
        controller.close()


def test_refinement_failure_does_not_make_probe_an_admission_gate(paths, toy_config):
    toy_config["agent"] = {
        "mode": "scripted", "prompt": "templates/prompts/implementation.md",
        "actions": [{
            "hypothesis": "Perturb logits so the quick probe recommends revision.",
            "edits": [{"tool": "replace_text", "arguments": {
                "path": "engine.py", "old": "seed = sum(", "new": "seed = 1 + sum("}}],
        }],
    }
    toy_config["probe"].update(enabled=True, repeats=1, max_attempts=2)
    toy_config["refinement"].update(enabled=True, max_revisions=1)
    toy_config["budget"].update(max_iterations=1, max_proposals=1)
    toy_config["config_hash"] = digest({key: value for key, value in toy_config.items()
                                        if key != "config_hash"})
    controller = Supervisor(paths, "probe-refinement-failure")
    original = controller.agent_session

    def fail_refinement(*args, **kwargs):
        if kwargs.get("phase") == "refinement":
            raise RuntimeError("refinement transport failed")
        return original(*args, **kwargs)

    controller.agent_session = fail_refinement
    try:
        controller.create(toy_config)
        state = controller.run()
        folder = paths.campaign("probe-refinement-failure") / "iterations/0001"
        decision = read_json(folder / "decision.json")

        assert state["stable_ref"] == state["initial_ref"] and state["state"] == "STOPPED"
        assert state["probe_attempts"] == 1 and state["refinement_attempts"] == 1
        assert decision["reason"] == "correctness_failed"
        assert read_json(folder / "correctness/report.json")["passed"] is False
        summary = read_json(folder / "probe-summary.json")
        assert summary["admission"] is False and summary["feedback_state"] == "revise"
        assert summary["probe_evidence_path"] == "iterations/0001/probe/0000/probe.json"
        assert summary["refinement_error"] == "refinement transport failed"
    finally:
        controller.close()


def test_unmeasurable_probe_prediction_is_inconclusive_and_continues(paths, toy_config, monkeypatch):
    toy_config["agent"] = {
        "mode": "scripted", "prompt": "templates/prompts/implementation.md",
        "actions": [{
            "hypothesis": "Remove the fixture arithmetic overhead.",
            "edits": [{"tool": "replace_text", "arguments": {
                "path": "engine.py", "old": "work = sum(range(120000))",
                "new": "work = 119999 * 120000 // 2"}}],
        }],
    }
    toy_config["probe"].update(enabled=True, repeats=1, max_attempts=1)
    toy_config["refinement"].update(enabled=False, max_revisions=0)
    toy_config["budget"].update(max_iterations=1, max_proposals=1)
    toy_config["config_hash"] = digest({key: value for key, value in toy_config.items()
                                        if key != "config_hash"})
    controller = Supervisor(paths, "probe-unavailable")
    original_profile = None

    def hide_bound_metric(client, version, plan, folder, deadline=None):
        report = original_profile(client, version, plan, folder, deadline=deadline)
        report["mechanism"].update(
            after=None, observed_change_pct=None, improvement_pct=None,
            prediction_observed=None, status="inconclusive")
        report.update(feedback_state="inconclusive",
                      recommended_action="continue_formal_validation",
                      needs_revision=False)
        write_json(folder / "probe.json", report)
        return report

    try:
        controller.create(toy_config)
        original_profile = controller.quick_candidate_probe
        monkeypatch.setattr(controller, "quick_candidate_probe", hide_bound_metric)
        state = controller.run()
        folder = paths.campaign("probe-unavailable") / "iterations/0001"
        probe = read_json(folder / "probe/0000/probe.json")
        decision = read_json(folder / "decision.json")

        assert probe["mechanism"]["status"] == "inconclusive"
        assert probe["feedback_state"] == "inconclusive" and probe["admission"] is False
        assert probe["needs_revision"] is False
        assert decision["outcome"] == "accept"
        assert (folder / "benchmarks").exists()
        assert state["stable_ref"] != state["initial_ref"]
    finally:
        controller.close()


def test_probe_transport_failure_is_inconclusive_and_continues(paths, toy_config, monkeypatch):
    toy_config["agent"] = {
        "mode": "scripted", "prompt": "templates/prompts/implementation.md",
        "actions": [{
            "hypothesis": "Remove the fixture arithmetic overhead.",
            "edits": [{"tool": "replace_text", "arguments": {
                "path": "engine.py", "old": "work = sum(range(120000))",
                "new": "work = 119999 * 120000 // 2"}}],
        }],
    }
    toy_config["probe"].update(enabled=True, repeats=1, max_attempts=1)
    toy_config["refinement"].update(enabled=False, max_revisions=0)
    toy_config["budget"].update(max_iterations=1, max_proposals=1)
    toy_config["config_hash"] = digest({key: value for key, value in toy_config.items()
                                        if key != "config_hash"})
    controller = Supervisor(paths, "probe-transport-failure")
    monkeypatch.setattr(controller, "quick_candidate_probe",
                        lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError("probe deadline")))
    try:
        controller.create(toy_config)
        state = controller.run()
        folder = paths.campaign("probe-transport-failure") / "iterations/0001"
        probe = read_json(folder / "probe/0000/probe.json")
        decision = read_json(folder / "decision.json")

        assert probe["feedback_state"] == "inconclusive" and probe["admission"] is False
        assert probe["error"] == "probe deadline"
        assert decision["outcome"] == "accept"
        assert (folder / "benchmarks").exists()
        assert state["stable_ref"] != state["initial_ref"]
    finally:
        controller.close()


def test_probe_prediction_is_three_state():
    assert probe_prediction(3.0, 2.0) == (True, "supported", "promising")
    assert probe_prediction(1.0, 2.0) == (False, "directionally_positive", "promising")
    assert probe_prediction(-1.0, 2.0) == (False, "refuted", "revise")
    assert probe_prediction(None, 2.0) == (None, "inconclusive", "inconclusive")


def test_stop_request_skips_pending_final_agent_analysis(paths, toy_config, monkeypatch):
    toy_config["budget"]["max_iterations"] = 2
    toy_config["config_hash"] = digest({key: value for key, value in toy_config.items() if key != "config_hash"})
    controller = Supervisor(paths, "stop-pending-analysis")
    original = controller.agent_session

    def request_stop_after_iteration(client, workspace, iteration, analysis_only=False, **kwargs):
        result = original(client, workspace, iteration, analysis_only)
        if not analysis_only:
            controller.database.update("stop-pending-analysis", {"stop_requested": True}, "stop_requested")
            controller.events.sync()
        return result

    monkeypatch.setattr(controller, "agent_session", request_stop_after_iteration)
    try:
        controller.create(toy_config)
        state = controller.run()
        directory = paths.campaign("stop-pending-analysis")
        assert state["state"] == "STOPPED"
        assert state["stop_reason"] == "stop_requested"
        assert state["analysis_pending"] is True
        assert not (directory / "final-analysis/agent-result.json").exists()
        resumed = controller.run(resume=True)
        assert resumed["state"] == "STOPPED"
        assert resumed["completed_iterations"] == 2
        assert resumed["stop_reason"] == "stop_requested"
    finally:
        controller.close()


def test_no_change_candidate_skips_build_and_benchmark(paths, toy_config):
    toy_config["budget"]["max_proposals"] = 1
    toy_config["budget"]["max_iterations"] = 1
    toy_config["evaluation"]["max_retests"] = 0
    toy_config["agent"]["actions"] = [{
        "hypothesis": "Confirm that a no-change submission is rejected before evaluation.",
        "edits": [],
    }]
    toy_config["config_hash"] = digest({key: value for key, value in toy_config.items() if key != "config_hash"})
    controller = Supervisor(paths, "no-change-candidate")
    try:
        controller.create(toy_config)
        state = controller.run()
        row = controller.database.iterations("no-change-candidate")[0]
        folder = paths.campaign("no-change-candidate") / "iterations" / "0001"

        assert state["state"] == "STOPPED"
        assert state["stable_ref"] == state["initial_ref"]
        assert row["decision"] == {"outcome": "reject", "reason": "no_change"}
        assert state["completed_experiments"] == 0 and state["proposal_attempts"] == 1
        assert "candidate" not in row
        assert not (folder / "build.json").exists()
        assert not (folder / "benchmarks" / "samples.jsonl").exists()
    finally:
        controller.close()


def test_exact_rejected_patch_is_not_benchmarked_again(paths, toy_config):
    toy_config["budget"]["max_proposals"] = 2
    toy_config["budget"]["max_iterations"] = 2
    toy_config["evaluation"]["max_retests"] = 0
    repeated_edit = {
        "tool": "replace_text",
        "arguments": {"path": "engine.py", "old": "seed = sum(", "new": "seed = 1 + sum("},
    }
    toy_config["agent"]["actions"] = [
        {"hypothesis": "Apply a known incorrect seed change.", "edits": [repeated_edit]},
        {"hypothesis": "Repeat the identical seed change.", "edits": [repeated_edit]},
    ]
    toy_config["config_hash"] = digest({key: value for key, value in toy_config.items() if key != "config_hash"})
    controller = Supervisor(paths, "duplicate-rejected-patch")
    try:
        controller.create(toy_config)
        state = controller.run()
        rows = controller.database.iterations("duplicate-rejected-patch")
        duplicate_folder = paths.campaign("duplicate-rejected-patch") / "iterations" / "0002"

        assert state["state"] == "STOPPED"
        assert rows[0]["decision"]["reason"] == "correctness_failed"
        assert rows[1]["decision"]["reason"] == "duplicate_change"
        assert rows[1]["decision"]["previous_iteration"] == 1
        assert "candidate" not in rows[1]
        assert not (duplicate_folder / "build.json").exists()
        assert not (duplicate_folder / "benchmarks" / "samples.jsonl").exists()
        assert state["stable_ref"] == state["initial_ref"]
    finally:
        controller.close()


def test_changed_numeric_parameter_is_a_distinct_experiment(paths, toy_config):
    toy_config["budget"]["max_iterations"] = 2
    toy_config["evaluation"]["max_retests"] = 0
    toy_config["agent"]["actions"] = [
        {"hypothesis": "Add one to the toy logit seed.", "edits": [{
            "tool": "replace_text",
            "arguments": {"path": "engine.py", "old": "seed = sum(", "new": "seed = 1 + sum("},
        }]},
        {"hypothesis": "Add two to the toy logit seed.", "edits": [{
            "tool": "replace_text",
            "arguments": {"path": "engine.py", "old": "seed = sum(", "new": "seed = 2 + sum("},
        }]},
    ]
    toy_config["config_hash"] = digest({key: value for key, value in toy_config.items() if key != "config_hash"})
    controller = Supervisor(paths, "similar-rejected-patch")
    try:
        controller.create(toy_config)
        state = controller.run()
        rows = controller.database.iterations("similar-rejected-patch")
        duplicate_folder = paths.campaign("similar-rejected-patch") / "iterations" / "0002"

        assert state["state"] == "STOPPED"
        assert rows[0]["decision"]["reason"] == "correctness_failed"
        assert rows[1]["decision"]["reason"] == "correctness_failed"
        assert "candidate" in rows[1]
        assert (duplicate_folder / "build.json").exists()
        assert (duplicate_folder / "correctness" / "report.json").exists()
        assert not (duplicate_folder / "benchmarks" / "samples.jsonl").exists()
    finally:
        controller.close()


def test_protocol_recovery_does_not_evaluate_unsubmitted_candidate(paths, toy_config, monkeypatch):
    toy_config["budget"]["max_proposals"] = 1
    toy_config["budget"]["max_iterations"] = 1
    toy_config["config_hash"] = digest({key: value for key, value in toy_config.items() if key != "config_hash"})
    controller = Supervisor(paths, "protocol-recovery")
    original = controller.agent_session

    def failed_submission(client, workspace, iteration, analysis_only=False, **kwargs):
        if analysis_only:
            return original(client, workspace, iteration, analysis_only=True)
        (workspace / "engine.py").write_text((workspace / "engine.py").read_text() + "\n# incomplete edit\n")
        return {"hypothesis": "No valid optimization submission was recovered.",
                "submission_origin": "controller_recovery", "recovery_error": "Malformed tool call"}

    monkeypatch.setattr(controller, "agent_session", failed_submission)
    try:
        controller.create(toy_config)
        state = controller.run()
        row = controller.database.iterations("protocol-recovery")[0]
        folder = paths.campaign("protocol-recovery") / "iterations" / "0001"
        assert state["state"] == "STOPPED"
        assert state["consecutive_failures"] == 1
        assert state["stable_ref"] == state["initial_ref"]
        assert row["decision"]["reason"] == "agent_protocol_error"
        assert "candidate" not in row
        assert not (folder / "build.json").exists()
        assert not (folder / "benchmarks" / "samples.jsonl").exists()
    finally:
        controller.close()


def test_controller_independently_rejects_missing_plan_from_scripted_worker(paths, toy_config):
    toy_config["budget"].update(max_proposals=1)
    toy_config["agent"]["actions"][0]["experiment_plan"] = None
    toy_config["config_hash"] = digest({key: value for key, value in toy_config.items() if key != "config_hash"})
    controller = Supervisor(paths, "missing-plan")
    try:
        controller.create(toy_config)
        state = controller.run()
        row = controller.database.iterations("missing-plan")[0]
        assert row["decision"]["reason"] == "missing_experiment_plan"
        assert not state["valid_candidates"] and not state["completed_experiments"]
        assert not (paths.campaign("missing-plan") / "iterations/0001/build.json").exists()
    finally:
        controller.close()


def test_analysis_barrier_prevents_new_candidate_if_reflection_cannot_finish(paths, toy_config, monkeypatch):
    controller = Supervisor(paths, "analysis-barrier")
    original = controller.agent_session

    def no_analysis_progress(client, workspace, iteration, analysis_only=False, **kwargs):
        if analysis_only:
            return {"complete": True, "previous_analysis": ""}
        return original(client, workspace, iteration, analysis_only)

    monkeypatch.setattr(controller, "agent_session", no_analysis_progress)
    try:
        controller.create(toy_config)
        with pytest.raises(RuntimeError, match="Analysis barrier made no progress"):
            controller.run()
        state = controller.database.get("analysis-barrier")
        assert state["state"] == "HALTED" and state["analysis_pending"]
        assert len(controller.database.iterations("analysis-barrier")) == 1
        assert not paths.workspace("analysis-barrier", 2).exists()
        assert state["active_ref"] is None
    finally:
        controller.close()


def test_keyboard_interrupt_keeps_incomplete_attempt_out_of_experiment_count(paths, toy_config, monkeypatch):
    controller = Supervisor(paths, "interrupted-attempt")

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(controller, "agent_session", interrupt)
    try:
        controller.create(toy_config)
        state = controller.run()
        assert state["state"] == "STOPPED" and state["stop_requested"]
        assert state["stop_reason"] == "keyboard_interrupt" and state["active_ref"] is None
        assert state["proposal_attempts"] == 1 and state["completed_experiments"] == 0
        row = controller.database.iteration("interrupted-attempt", 1)
        assert row["decision"]["reason"] == "interrupted_iteration" and row["analysis_pending"]
        assert not (paths.data / "state/interrupted-attempt.engine.json").exists()
    finally:
        controller.close()


def test_keyboard_interrupt_during_rollback_preserves_admission_decision(paths, toy_config, monkeypatch):
    toy_config["budget"].update(max_iterations=1, max_proposals=1)
    toy_config["agent"]["actions"] = [{"hypothesis": "Break the deterministic seed.", "edits": [{
        "tool": "replace_text", "arguments": {"path": "engine.py", "old": "seed = sum(",
                                                   "new": "seed = 1 + sum("}}]}]
    toy_config["config_hash"] = digest({key: value for key, value in toy_config.items() if key != "config_hash"})
    controller = Supervisor(paths, "interrupt-after-decision")

    def interrupt_restore(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(controller, "restore", interrupt_restore)
    try:
        controller.create(toy_config)
        state = controller.run()
        row = controller.database.iteration("interrupt-after-decision", 1)
        assert state["state"] == "STOPPED"
        assert row["decision"]["reason"] == "correctness_failed"
        assert row["interrupted_after_decision"] and not row.get("interrupted")
    finally:
        controller.close()
