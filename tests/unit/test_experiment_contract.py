import json
import shutil
from pathlib import Path

import pytest

from vibeforge.agent.candidate import (
    candidate_gate, candidate_strategy, observation_benefit_bounds,
    runtime_effect_fingerprint, semantic_fingerprint,
)
from vibeforge.agent.memory import (
    archived_candidates, experience_ledger, older_history_summary,
    reflection_error, refresh_memory, search_memory,
)
from vibeforge.agent.runner import _assistant_tools, _model_context
from vibeforge.agent.tools import AgentTools
from vibeforge.storage.atomic import read_json, write_json
from vibeforge.storage.database import Database
from vibeforge.versioning.artifacts import source_hash


@pytest.fixture
def proposal(tmp_path):
    parent, workspace, evidence = tmp_path / "stable", tmp_path / "workspace", tmp_path / "campaign"
    parent.mkdir()
    (parent / "engine.py").write_text("value = 1\n")
    write_json(parent / "engine.json", {"device": "cpu"})
    shutil.copytree(parent, workspace)
    (workspace / "engine.py").write_text("value = 2\n")
    profile = "iterations/0001/observation.json"
    write_json(evidence / profile, {"kind": "host_inference_profile", "source_hash": source_hash(parent),
                                    "profile_hash": "p1", "stable_ref": "s1"})
    context = {"stable_ref": "s1", "profile_hash": "p1", "evidence_root": str(evidence),
               "candidate_contract": {"parent_source": str(parent), "observation_paths": [profile],
                                      "tested_candidates": [], "history_iterations": []}}
    plan = {"measured_bottleneck": "CPU Python sum in the diagnostic profile", "cause": "Repeated arithmetic",
            "change": "Use equivalent closed form", "metric": "throughput", "direction": "increase",
            "expected_change_pct": 3, "validation": "Frozen correctness plus paired measurements",
            "evidence_paths": [profile], "history_refs": []}
    return workspace, context, plan


def test_valid_candidate_requires_actual_change_and_bound_measurement(proposal):
    workspace, context, plan = proposal
    gate = candidate_gate(workspace, context, plan)
    assert gate["valid"] and gate["changed_files"] == ["engine.py"]
    assert not candidate_gate(workspace, context, None)["valid"]
    context["profile_hash"] = "other-profile"
    assert candidate_gate(workspace, context, plan)["reason"] == "unbound_measurement_evidence"


def test_candidate_gate_rejects_invalid_literal_cache_implementation(proposal):
    workspace, context, plan = proposal
    (workspace / "engine.py").write_text('options = {"cache_implementation": "sdpa"}\n')
    gate = candidate_gate(workspace, context, plan)
    assert gate["reason"] == "incompatible_or_ineffective_configuration"
    assert gate["issues"][0][0] == "sdpa"


def test_candidate_gate_enforces_scheduler_and_native_surfaces(proposal):
    workspace, context, plan = proposal
    parent = Path(context["candidate_contract"]["parent_source"])
    parent_config = read_json(parent / "engine.json")
    write_json(workspace / "engine.json", {**parent_config,
                                            "request_scheduler": "concurrent_requests"})
    context["optimization"] = {"allowed_surfaces": ["execution_adapter"]}
    gate = candidate_gate(workspace, context, plan)
    assert gate["reason"] == "unauthorized_modification_surface"
    assert gate["surfaces"] == ["scheduler"]

    context["optimization"]["allowed_surfaces"].append("scheduler")
    context["candidate_contract"]["version"] = 3
    context["requirements"] = {"objective": "throughput"}
    context["workload"] = {"profiles": [{"id": "concurrent_2", "prompts": ["hello"], "concurrency": 2}]}
    observation = read_json(Path(context["evidence_root"]) /
                            context["candidate_contract"]["observation_paths"][0])
    observation.update(runtime={"execution_path": "transformers_generate",
                                "request_scheduler": "serialized_lock", "continuous_batching": False},
                       workload=context["workload"], profile_id="concurrent_2",
                       measurements={"diagnostic.concurrent.concurrent_2.throughput": {
                           "value": 10.0, "unit": "output_tokens_per_second"}})
    write_json(Path(context["evidence_root"]) /
               context["candidate_contract"]["observation_paths"][0], observation)
    context["performance_observation"] = observation
    gate = candidate_gate(workspace, context, plan)
    assert gate["valid"]
    assert gate["experiment_plan"]["bottleneck_reference"]["metric"] == (
        "diagnostic.concurrent.concurrent_2.throughput")
    assert gate["experiment_plan"]["mechanism_surface"] == "scheduler"
    # Agent-selected matrix evidence need not include the aggregate concurrency record.
    other_path = "matrix.json"
    write_json(Path(context["evidence_root"]) / other_path, observation)
    context["candidate_contract"]["observation_paths"].append(other_path)
    plan["evidence_paths"] = [other_path]
    rebound = candidate_gate(workspace, context, plan)
    assert rebound["valid"]
    assert rebound["experiment_plan"]["bottleneck_reference"]["evidence_path"] in rebound["experiment_plan"]["evidence_paths"]
    observation["runtime"]["scheduler_capabilities"] = {
        "available": ["serialized_lock", "concurrent_requests", "microbatch"]}
    context["performance_observation"] = observation
    rejected = candidate_gate(workspace, context, plan)
    assert rejected["reason"] == "unbatched_concurrent_scheduler"
    assert rejected["recommended"] == "microbatch"
    context["candidate_contract"]["version"] = 1
    assert candidate_gate(workspace, context, plan)["reason"] == "unbatched_concurrent_scheduler"
    context["candidate_contract"]["version"] = 3
    write_json(workspace / "engine.json", {**parent_config,
                                            "request_scheduler": "microbatch"})
    assert candidate_gate(workspace, context, plan)["valid"]
    write_json(workspace / "engine.json", {**parent_config,
                                            "request_scheduler": "concurrent_requests"})
    observation["runtime"].pop("scheduler_capabilities")
    context["performance_observation"] = observation
    (workspace / "kernel.cu").write_text("extern \"C\" __global__ void noop() {}\n")
    gate = candidate_gate(workspace, context, plan)
    assert gate["reason"] == "unauthorized_modification_surface"
    context["optimization"]["allowed_surfaces"].append("native_kernel")
    gate = candidate_gate(workspace, context, plan)
    assert gate["reason"] == "native_kernel_without_frozen_build"
    context["engine_adapter"] = {"build_command": ["python3", "build_native.py"]}
    assert candidate_gate(workspace, context, plan)["valid"]


def test_scheduler_classification_ignores_candidate_sidecar_json():
    strategy = candidate_strategy(
        ["engine.json", "iterations/0002/candidate.json"], {"request_scheduler"})
    assert strategy == {"strategy_family": "scheduler", "mechanism_id": "request_scheduler_change"}


def test_scheduler_precondition_stays_blocked_until_concurrency_evidence_exists(proposal):
    workspace, context, plan = proposal
    write_json(workspace / "engine.json", {"device": "cpu", "request_scheduler": "microbatch"})
    context["optimization"] = {"allowed_surfaces": ["scheduler"]}
    context["candidate_contract"].update(version=3, attempted_strategies=[{
        "history_ref": "prior-campaign:2",
        "mechanism_id": "request_scheduler_change",
        "decision": {"reason": "missing_scheduler_concurrency_evidence"},
    }])

    gate = candidate_gate(workspace, context, plan)

    assert gate["reason"] == "unresolved_candidate_precondition"
    assert gate["previous_history_ref"] == "prior-campaign:2"
    assert gate["required_evidence"].startswith("diagnostic.concurrent.")

    write_json(workspace / "engine.json", {
        "device": "cpu", "request_scheduler": "microbatch", "cache_implementation": "sdpa",
    })
    mixed_gate = candidate_gate(workspace, context, plan)
    assert mixed_gate["reason"] == "unresolved_candidate_precondition"
    assert mixed_gate["mechanism_id"] == "request_scheduler_change"


def test_candidate_gate_rejects_reserved_campaign_artifacts(proposal):
    workspace, context, plan = proposal
    write_json(workspace / "iterations" / "0002" / "candidate.json", {"note": "agent scratch"})

    gate = candidate_gate(workspace, context, plan)

    assert gate["reason"] == "reserved_candidate_artifact"
    assert gate["changed_files"] == ["iterations/0002/candidate.json"]


def test_comments_whitespace_and_docs_cannot_create_an_experiment(proposal):
    workspace, context, plan = proposal
    (workspace / "engine.py").write_text("# cosmetic only\nvalue=1\n\n")
    (workspace / "README.md").write_text("new description")
    assert candidate_gate(workspace, context, plan)["reason"] == "no_change"


def test_observation_from_another_workload_cannot_justify_candidate(proposal):
    workspace, context, plan = proposal
    context["workload"] = {"prompts": ["different request"], "max_new_tokens": 64}
    assert candidate_gate(workspace, context, plan)["reason"] == "unbound_measurement_evidence"


def test_semantic_duplicate_cannot_evade_gate_by_comments_or_formatting(proposal):
    workspace, context, plan = proposal
    context["candidate_contract"]["tested_candidates"] = [{"iteration": 7, "fingerprint": semantic_fingerprint(workspace)}]
    (workspace / "engine.py").write_text("# try again\nvalue=2\n")
    gate = candidate_gate(workspace, context, plan)
    assert gate["reason"] == "duplicate_change" and gate["previous_iteration"] == 7


def test_inconclusive_configuration_does_not_exclude_other_configurations(proposal):
    workspace, context, plan = proposal
    parent = Path(context["candidate_contract"]["parent_source"])
    write_json(workspace / "engine.json", {**read_json(parent / "engine.json"),
                                            "cache_implementation": "static"})
    (workspace / "engine.py").write_text((parent / "engine.py").read_text())
    context["optimization"] = {"allowed_surfaces": ["cache_policy"]}
    context["candidate_contract"]["attempted_strategies"] = [{
        "history_ref": "prior:1", "strategy_family": "cache_policy",
        "mechanism_id": "cache_implementation_switch",
        "decision": {"outcome": "reject", "reason": "no_significant_gain"},
    }]

    gate = candidate_gate(workspace, context, plan)

    assert gate["valid"]


@pytest.mark.parametrize("field,value,reason", [
    ("expected_change_pct", float("nan"), "invalid_experiment_plan"),
    ("expected_change_pct", True, "invalid_experiment_plan"),
    ("metric", "invented_tps", "invalid_experiment_plan"),
    ("metric", [], "invalid_experiment_plan"),
    ("direction", [], "invalid_experiment_plan"),
    ("evidence_paths", ["../outside"], "invalid_measurement_evidence"),
    ("evidence_paths", [], "missing_measurement_evidence"),
    ("history_refs", [999], "invalid_history_refs"),
])
def test_plan_rejects_nonfalsifiable_or_untraceable_fields(proposal, field, value, reason):
    workspace, context, plan = proposal
    plan[field] = value
    assert candidate_gate(workspace, context, plan)["reason"] == reason


def test_history_comparison_is_required_after_first_attempt(proposal):
    workspace, context, plan = proposal
    context["candidate_contract"]["history_iterations"] = [1]
    assert candidate_gate(workspace, context, plan)["reason"] == "missing_history_comparison"
    plan["history_refs"] = [1]
    assert candidate_gate(workspace, context, plan)["valid"]


def test_v3_uses_global_history_refs_and_controller_fills_bookkeeping(proposal):
    workspace, context, plan = proposal
    context["requirements"] = {"objective": "throughput"}
    context["workload"] = {"profiles": [{"id": "short", "prompts": ["hello"]}]}
    context["candidate_contract"].update(version=3, history_refs=["earlier-campaign:1"])
    path = context["candidate_contract"]["observation_paths"][0]
    record = read_json(Path(context["evidence_root"]) / path)
    record.update(workload=context["workload"], profile_id="short",
                  runtime={"execution_path": "transformers_generate"},
                  measurements={"normal.phases.decode_forward_seconds": {"value": 1.25, "unit": "seconds"}})
    write_json(Path(context["evidence_root"]) / path, record)
    concise = {key: plan[key] for key in (
        "measured_bottleneck", "cause", "change", "expected_change_pct", "validation")}
    concise["history_refs"] = ["earlier-campaign:1"]
    gate = candidate_gate(workspace, context, concise)
    assert gate["valid"]
    completed = gate["experiment_plan"]
    assert completed["evidence_paths"] == [path]
    assert completed["metric"] == "throughput" and completed["direction"] == "increase"
    assert completed["bottleneck_reference"] == {
        "evidence_path": path, "metric": "normal.phases.decode_forward_seconds",
        "value": 1.25, "unit": "seconds"}
    assert completed["target_profile_ids"] == ["short"]
    assert completed["runtime_preconditions"] == [
        {"field": "execution_path", "equals": "transformers_generate"}]
    concise["history_refs"] = ["unknown:1"]
    rejected = candidate_gate(workspace, context, concise)
    assert rejected["reason"] == "invalid_history_refs"
    assert rejected["allowed_history_refs"] == ["earlier-campaign:1"]


def test_v3_campaign_objective_overrides_measurement_key_in_metric_field(proposal):
    workspace, context, plan = proposal
    context["requirements"] = {"objective": "throughput"}
    context["workload"] = {"profiles": [{"id": "default", "prompts": ["hello"]}]}
    context["candidate_contract"]["version"] = 3
    path = context["candidate_contract"]["observation_paths"][0]
    record = read_json(Path(context["evidence_root"]) / path)
    record.update(workload=context["workload"], profile_id="default",
                  runtime={"execution_path": "transformers_generate"}, stable_ref="s1")
    record["measurements"] = {
        "normal.backend_wall_seconds": {"value": 2.0, "unit": "seconds"},
        "normal.phases.decode_forward_seconds": {"value": 1.5, "unit": "seconds"},
    }
    write_json(Path(context["evidence_root"]) / path, record)
    plan.update(metric="normal.backend_wall_seconds", direction="lower",
                bottleneck_metric="normal.phases.decode_forward_seconds")

    gate = candidate_gate(workspace, context, plan)

    assert gate["valid"]
    assert gate["experiment_plan"]["metric"] == "throughput"
    assert gate["experiment_plan"]["direction"] == "increase"
    assert gate["experiment_plan"]["bottleneck_reference"]["metric"] == "normal.phases.decode_forward_seconds"


def test_v3_binds_controller_facts_to_observation_regardless_of_evidence_order(proposal):
    workspace, context, plan = proposal
    context.update(requirements={"objective": "throughput"},
                   workload={"profiles": [{"id": "default", "prompts": ["hello"]}]})
    context["candidate_contract"]["version"] = 3
    observation = context["candidate_contract"]["observation_paths"][0]
    auxiliary = "iterations/0001/change.patch"
    (Path(context["evidence_root"]) / auxiliary).parent.mkdir(parents=True, exist_ok=True)
    (Path(context["evidence_root"]) / auxiliary).write_text("patch")
    record = read_json(Path(context["evidence_root"]) / observation)
    record.update(workload=context["workload"], profile_id="default", stable_ref="s1",
                  runtime={"execution_path": "transformers_generate"}, measurements={
                      "normal.phases.decode_forward_seconds": {"value": 1.5, "unit": "seconds"}})
    write_json(Path(context["evidence_root"]) / observation, record)
    plan.update(evidence_paths=[auxiliary, observation],
                bottleneck_metric="normal.phases.decode_forward_seconds",
                expected_bottleneck_change_pct=-3, bottleneck_direction="lower")

    gate = candidate_gate(workspace, context, plan)

    assert gate["valid"]
    completed = gate["experiment_plan"]
    assert completed["bottleneck_reference"]["evidence_path"] == observation
    assert completed["runtime_preconditions"] == [
        {"field": "execution_path", "equals": "transformers_generate"}]
    assert completed["expected_bottleneck_change_pct"] == plan["expected_change_pct"]
    assert completed["bottleneck_direction"] == "decrease"


def test_amdahl_gate_rejects_gain_above_measured_phase_upper_bound(proposal):
    workspace, context, plan = proposal
    context["candidate_contract"]["version"] = 3
    context["requirements"] = {"objective": "throughput"}
    context["workload"] = {"prompts": ["hello"], "max_new_tokens": 8}
    path = context["candidate_contract"]["observation_paths"][0]
    record = read_json(Path(context["evidence_root"]) / path)
    record.update(
        workload=context["workload"], profile_id="default",
        runtime={"execution_path": "transformers_generate"},
        measurements={
            "normal.backend_wall_seconds": {"value": 1.0, "unit": "seconds"},
            "normal.phases.decode_forward_seconds": {"value": 0.05, "unit": "seconds"},
        },
    )
    write_json(Path(context["evidence_root"]) / path, record)
    plan.update(bottleneck_metric="normal.phases.decode_forward_seconds",
                expected_change_pct=20)

    rejected = candidate_gate(workspace, context, plan)

    assert rejected["reason"] == "implausible_expected_gain"
    bound = rejected["benefit_bound"]
    assert bound["bottleneck_time_share"] == pytest.approx(0.05)
    assert bound["maximum_overall_improvement_pct"] == pytest.approx(5.2631579)
    assert bound["required_bottleneck_elimination_pct"] > 100


def test_amdahl_gate_records_feasible_prediction_and_full_wall_is_unbounded(proposal):
    workspace, context, plan = proposal
    context["candidate_contract"]["version"] = 3
    context["requirements"] = {"objective": "throughput"}
    context["workload"] = {"prompts": ["hello"], "max_new_tokens": 8}
    path = context["candidate_contract"]["observation_paths"][0]
    record = read_json(Path(context["evidence_root"]) / path)
    record.update(
        workload=context["workload"], profile_id="default",
        runtime={"execution_path": "transformers_generate"},
        measurements={
            "normal.backend_wall_seconds": {"value": 1.0, "unit": "seconds"},
            "normal.phases.decode_forward_seconds": {"value": 0.20, "unit": "seconds"},
        },
    )
    write_json(Path(context["evidence_root"]) / path, record)
    plan.update(bottleneck_metric="normal.phases.decode_forward_seconds",
                expected_change_pct=5)

    accepted = candidate_gate(workspace, context, plan)

    assert accepted["valid"] and accepted["experiment_plan"]["benefit_bound"]["feasible"]
    assert accepted["experiment_plan"]["benefit_bound"]["required_bottleneck_elimination_pct"] == pytest.approx(23.8095238)
    plan["bottleneck_metric"] = "normal.backend_wall_seconds"
    plan["expected_change_pct"] = 80
    whole_request = candidate_gate(workspace, context, plan)
    assert whole_request["valid"]
    assert whole_request["experiment_plan"]["benefit_bound"]["maximum_overall_improvement_pct"] is None


def test_observation_lists_phase_benefit_bounds_without_profiled_operator_claims():
    bounds = observation_benefit_bounds({"measurements": {
        "normal.backend_wall_seconds": {"value": 2.0, "unit": "seconds"},
        "normal.phases.prefill_forward_seconds": {"value": 0.5, "unit": "seconds"},
        "operator.aten::mm.calls": {"value": 100, "unit": "count"},
    }})

    assert len(bounds) == 1
    assert bounds[0]["metric"] == "normal.phases.prefill_forward_seconds"
    assert bounds[0]["request_time_share"] == 0.25
    assert bounds[0]["maximum_throughput_improvement_pct"] == pytest.approx(100 / 3)


def test_runtime_effect_fingerprint_ignores_timing_and_request_length():
    first = {"runtime": {"execution_path": "transformers_generate",
                         "attention_implementations": {"text": "sdpa"}},
             "normal_request": {"observed_cache_class": "StaticCache",
                                "observed_generate_kwargs": [{"use_cache": True, "max_new_tokens": 2}]}}
    second = json.loads(json.dumps(first))
    second["normal_request"]["backend_wall_seconds"] = 99
    second["normal_request"]["observed_generate_kwargs"][0]["max_new_tokens"] = 128
    assert runtime_effect_fingerprint(first) == runtime_effect_fingerprint(second)
    second["normal_request"]["observed_cache_class"] = "DynamicCache"
    assert runtime_effect_fingerprint(first) != runtime_effect_fingerprint(second)
    assert runtime_effect_fingerprint({"runtime": {"execution_path": "toy"}}) is None


def test_v2_binds_numeric_evidence_runtime_and_profile(proposal):
    workspace, context, plan = proposal
    context["candidate_contract"]["version"] = 2
    context["workload"] = {"prompts": ["hello"], "max_new_tokens": 32}
    path = context["candidate_contract"]["observation_paths"][0]
    record = json.loads((Path(context["evidence_root"]) / path).read_text())
    record.update(workload=context["workload"], profile_id="default", runtime={"execution_path": "transformers_generate"},
                  measurements={"normal.phases.decode_forward_seconds": {"value": .4, "unit": "seconds", "scope": "ordinary"}})
    write_json(Path(context["evidence_root"]) / path, record)
    plan.update(bottleneck_reference={"evidence_path": path, "metric": "normal.phases.decode_forward_seconds", "value": .4, "unit": "seconds"},
                runtime_preconditions=[{"field": "execution_path", "equals": "transformers_generate"}],
                target_profile_ids=["default"], expected_bottleneck_change_pct=5, bottleneck_direction="decrease",
                expected_change_basis="Remove redundant arithmetic within measured decode path")
    assert candidate_gate(workspace, context, plan)["valid"]
    original_reference = plan["bottleneck_reference"]
    for malformed in (["not an object"], {"evidence_path": [], "metric": "x"}):
        plan["bottleneck_reference"] = malformed
        assert candidate_gate(workspace, context, plan)["reason"] == "unbound_numeric_bottleneck"
    plan["bottleneck_reference"] = original_reference
    plan["target_profile_ids"] = [{}]
    assert candidate_gate(workspace, context, plan)["reason"] == "invalid_target_profiles"
    plan["target_profile_ids"] = ["default"]
    plan["bottleneck_reference"]["value"] = .04
    assert candidate_gate(workspace, context, plan)["reason"] == "unbound_numeric_bottleneck"
    plan["bottleneck_reference"]["value"] = .4
    plan["runtime_preconditions"][0]["equals"] = "continuous_batching"
    rejected = candidate_gate(workspace, context, plan)
    assert rejected["reason"] == "runtime_precondition_failed"
    assert rejected["observed_value"] == "transformers_generate"
    assert rejected["available_runtime_preconditions"] == [
        {"field": "execution_path", "equals": "transformers_generate"}
    ]
    plan["runtime_preconditions"][0]["equals"] = "transformers_generate"
    plan["target_profile_ids"] = ["nonexistent"]
    assert candidate_gate(workspace, context, plan)["reason"] == "invalid_target_profiles"


def test_long_workloads_do_not_flood_agent_context_or_mutate_host_inputs():
    workload = {"prompts": ["x" * 30000], "profiles": [{"id": "long", "prompts": ["y" * 50000]}]}
    context = {"workload": workload, "performance_observation": {"workload": workload}}
    assert len(json.dumps(_model_context(context))) < 1500
    assert len(context["workload"]["profiles"][0]["prompts"][0]) == 50000


def test_submission_rejection_can_end_as_explicit_blocked_proposal(proposal, tmp_path):
    workspace, context, plan = proposal
    result = {}
    tools = {tool.name: tool for tool in _assistant_tools(workspace, tmp_path / "trace.jsonl", context, result)}
    response = json.loads(tools["submit_candidate"].call(json.dumps({"hypothesis": "guess without measurement"})))
    assert not response["submitted"] and not result
    response = json.loads(tools["report_blocked"].call(json.dumps({"reason": "No supported mechanism justified by available evidence"})))
    assert response["blocked"] and result["blocked"] and result["complete"]


def test_reflection_cannot_claim_supported_after_infrastructure_failure(tmp_path):
    write_json(tmp_path / "iterations/0004/decision.json", {"reason": "execution_failed"})
    previous = {"iteration": 4, "analysis_pending": True, "decision": {"outcome": "reject", "reason": "execution_failed"}}
    reflection = {"iteration": 4, "hypothesis_status": "supported", "explanation": "fast", "next_action": "try something else",
                  "evidence_paths": ["iterations/0004/decision.json"]}
    assert reflection_error(previous, reflection, tmp_path)
    reflection["hypothesis_status"] = "inconclusive"
    assert reflection_error(previous, reflection, tmp_path) is None
    reflection["hypothesis_status"] = "refuted"
    assert reflection_error(previous, reflection, tmp_path)
    reflection["hypothesis_status"] = "inconclusive"
    reflection["iteration"] = 5
    assert reflection_error(previous, reflection, tmp_path)


def test_acceptance_does_not_prove_an_unmet_prediction(tmp_path):
    write_json(tmp_path / "iterations/0004/decision.json", {"reason": "measured_gain"})
    previous = {"iteration": 4, "analysis_pending": True, "decision": {"outcome": "accept", "reason": "measured_gain"},
                "experiment_verification": {"status": "refuted", "expected_change_pct": 50}}
    reflection = {"iteration": 4, "hypothesis_status": "supported", "explanation": "Some measured gain",
                  "next_action": "Revisit the predicted magnitude", "evidence_paths": ["iterations/0004/decision.json"]}
    assert reflection_error(previous, reflection, tmp_path)
    reflection["hypothesis_status"] = "refuted"
    assert reflection_error(previous, reflection, tmp_path) is None


def test_correctness_reflection_cites_report_and_cannot_invent_numerical_failure(tmp_path):
    report_path = "iterations/0004/correctness/report.json"
    write_json(tmp_path / report_path, {
        "passed": False,
        "failure_summary": {"failed_checks": 1, "operational_errors": 0},
        "checks": [{"case": "fixed", "passed": False, "generation_passed": False,
                    "repeat_passed": False, "fixed_length": True,
                    "numerical": {"passed": True, "normalized_rms": 0.0},
                    "continuation_numerical": {"passed": True, "normalized_rms": 0.0}}],
    })
    previous = {"iteration": 4, "analysis_pending": True,
                "decision": {"outcome": "reject", "reason": "correctness_failed"}}
    reflection = {"iteration": 4, "hypothesis_status": "refuted",
                  "explanation": "The fixed-length generation and repeat token sequences failed; numerical checks passed.",
                  "next_action": "Avoid this cache implementation.",
                  "evidence_paths": ["iterations/0004/decision.json"]}
    write_json(tmp_path / "iterations/0004/decision.json", previous["decision"])
    assert "correctness/report.json" in reflection_error(previous, reflection, tmp_path)
    reflection["evidence_paths"].append(report_path)
    assert reflection_error(previous, reflection, tmp_path) is None
    reflection["explanation"] = "The output exceeded the max_normalized_rms threshold."
    assert "numerical/logit checks passed" in reflection_error(previous, reflection, tmp_path)


def test_history_retrieves_old_analysis_and_evidence_reads_can_reach_tail(tmp_path):
    database = Database(tmp_path / "state.sqlite")
    try:
        database.put("c", {})
        for number in range(1, 51):
            database.iteration("c", number, {"hypothesis": f"change {number}",
                               "interpretation": "quantized dequantization hotspot" if number == 2 else "other result",
                               "decision": {"outcome": "reject"}, "complete": True})
        refresh_memory(database, "c", tmp_path)
        tools = AgentTools(tmp_path / "workspace", tmp_path / "trace.jsonl", tmp_path)
        assert tools.execute("search_history", {"query": "dequantization"})[0]["iteration"] == 2
        log = tmp_path / "long.log"
        log.write_text("x" * 9000 + "ACTUAL ERROR AT TAIL")
        assert "ACTUAL ERROR" in tools.execute("read_evidence", {"path": "long.log", "offset": 9000})
        assert len(search_memory(tmp_path, "", 100)) == 10
    finally:
        database.close()


def test_older_history_summary_and_experience_keep_pre_six_round_information():
    rows = []
    for iteration in range(1, 10):
        outcome = "accept" if iteration == 1 else "reject"
        reason = "measured_gain" if iteration == 1 else (
            "correctness_failed" if iteration == 2 else "no_significant_gain")
        rows.append({
            "iteration": iteration, "hypothesis": f"strategy {iteration}",
            "plan": {"change": "static cache" if iteration < 4 else "kernel fusion"},
            "facts": {"decision": {"outcome": outcome, "reason": reason},
                      "verification": {"status": "supported" if iteration == 1 else "refuted"}},
            "reflection": {"hypothesis_status": "supported" if iteration == 1 else "refuted"},
            "evidence_path": f"iterations/{iteration:04d}",
        })

    summary = older_history_summary(rows, "campaign", recent=6)
    experience = experience_ledger(rows, "campaign")

    assert summary["attempts"] == 3 and summary["iteration_range"] == [1, 3]
    assert summary["decision_reasons"] == {"measured_gain": 1, "correctness_failed": 1,
                                            "no_significant_gain": 1}
    assert {ref for item in summary["strategy_ledger"] for ref in item["history_refs"]} == {
        "campaign:1", "campaign:2", "campaign:3"}
    assert experience["verified_strategies"][0]["history_ref"] == "campaign:1"
    assert {item["history_ref"] for item in experience["counterexamples"]} >= {
        "campaign:2", "campaign:3"}
    assert experience["verified_strategies"][0]["status"] == "local_validated"


def test_quick_probe_failure_becomes_a_reusable_counterexample():
    rows = [{
        "iteration": 4,
        "hypothesis": "replace DynamicCache with StaticCache",
        "plan": {"change": "enable StaticCache", "measured_bottleneck": "decode"},
        "facts": {"decision": {"outcome": "reject", "reason": "candidate_probe_failed"}},
        "evidence_path": "iterations/0004",
    }]

    experience = experience_ledger(rows, "campaign")

    assert experience["verified_strategies"] == []
    counterexample = experience["counterexamples"][0]
    assert counterexample["history_ref"] == "campaign:4"
    assert counterexample["failure"] == "candidate_probe_failed"
    assert counterexample["status"] == "counterexample"


def test_search_history_includes_archived_campaigns_without_claiming_compatibility(tmp_path):
    current = tmp_path / "current"
    archive = current / "archives" / "old"
    (current / "memory").mkdir(parents=True)
    (archive / "memory").mkdir(parents=True)
    write_json(current / "memory/index.json", {"entries": [{
        "iteration": 1, "hypothesis": "current cache idea", "facts": {},
        "evidence_path": "iterations/0001",
    }]})
    write_json(archive / "memory/index.json", {"entries": [{
        "iteration": 7, "hypothesis": "fused dequantization kernel", "facts": {
            "decision": {"outcome": "reject", "reason": "no_significant_gain"}},
        "evidence_path": "iterations/0007",
    }]})
    write_json(current / "memory/archived-campaigns.json", {"entries": [{
        "campaign_id": "old", "memory_index": "archives/old/memory/index.json",
        "evidence_directory": "archives/old", "config_hash": "different-config",
        "profile_hash": "different-profile", "stable_ref": "old-parent",
    }]})

    result = search_memory(current, "dequantization", 5)

    assert result[0]["history_ref"] == "old:7"
    assert result[0]["evidence_path"] == "archives/old/iterations/0007"
    assert result[0]["compatibility"] == {
        "config_hash": "different-config", "profile_hash": "different-profile",
        "stable_ref": "old-parent",
    }


def test_full_host_admission_index_does_not_grow_the_model_prompt():
    context = {"history": [{"iteration": 1}], "candidate_contract": {
        "version": 1, "observation_paths": ["iterations/0002/observation.json"],
        "history_iterations": [1], "tested_candidates": [{"iteration": 1, "fingerprint": "f"}],
    }}
    before = json.dumps(_model_context(context))
    context["candidate_contract"]["history_iterations"] = list(range(1, 10001))
    context["candidate_contract"]["tested_candidates"] *= 10000
    assert json.dumps(_model_context(context)) == before
    assert len(context["candidate_contract"]["history_iterations"]) == 10000


def test_profile_summary_preserves_numeric_evidence_and_original_artifact():
    import copy
    profile = {
        "measurements": {"vibeforge.module.linear_attention.cpu_total_seconds":
                         {"value": 2.5, "unit": "seconds", "scope": "inclusive"}},
        "runtime": {"execution_path": "transformers_generate"},
        "phases": [{"name": "vibeforge.module.linear_attention", "calls": 1536}],
        "normal_request": {"ttft_seconds": .7, "inter_token_seconds": [.1, .2]},
        "timing": {"backend_wall_seconds": 5},
        "workload": {"prompts": ["x" * 30000]},
        **{key: [{"name": "kernel<" + "bf16, " * 1000 + ">", "calls": i + 1} for i in range(15)]
           for key in ("top_cpu_ops", "top_device_ops", "top_device_operator_ops")},
    }
    original = copy.deepcopy(profile)
    summary = _model_context({"performance_observation": profile})["performance_observation"]
    assert profile == original
    for key in ("measurements", "runtime", "phases"):
        assert summary[key] == profile[key]
    assert len(summary["top_device_ops"]) == 5
    assert summary["top_device_ops"][0]["raw_index"] == 0
    assert summary["top_device_ops"][0]["name_truncated"]
    assert summary["normal_request"]["inter_token_summary"]["count"] == 2
    assert summary["diagnostic_wall_seconds"] == 5
    assert "workload" not in summary and "timing" not in summary


def test_compact_profile_leaves_recent_source_tool_result_available():
    from qwen_agent.llm.base import _truncate_input_messages_roughly
    from qwen_agent.llm.schema import Message
    from vibeforge.agent.runner import _agent_input_budget
    source = "value = model.generate(**options)\n" * 300
    profile = {"measurements": {"normal.backend_wall_seconds": {"value": 2, "unit": "seconds"}},
               **{key: [{"name": "gemm<" + "bfloat16, " * 250 + ">", "calls": 500} for _ in range(15)]
                  for key in ("top_cpu_ops", "top_device_ops", "top_device_operator_ops")}}
    context = {"model": {"max_context": 32768}, "source": source, "performance_observation": profile}
    messages = [Message(role="system", content="Read evidence, edit code, then submit a measured hypothesis."),
                Message(role="user", content=json.dumps(_model_context(context))),
                Message(role="assistant", content="", function_call={"name": "read_file", "arguments": '{"path":"engine.py"}'}),
                Message(role="function", name="read_file", content=source)]
    retained = _truncate_input_messages_roughly(messages, _agent_input_budget(context, 2048))
    assert retained[-1].content == source
    assert retained[1].content == messages[1].content


def test_evidence_root_can_be_listed_without_allowing_parent_escape(tmp_path):
    write_json(tmp_path / "campaign/iterations/0001/decision.json", {"reason": "interrupted_iteration"})
    tools = AgentTools(tmp_path / "workspace", tmp_path / "trace.jsonl", tmp_path / "campaign")
    assert tools.execute("list_evidence", {"path": "."})["files"] == ["iterations/0001/decision.json"]
    with pytest.raises(ValueError, match="escapes"):
        tools.execute("list_evidence", {"path": ".."})


def test_invalidated_review_is_not_reinjected_as_trusted_model_memory():
    context = {"previous": {"iteration": 3, "decision": {"reason": "correctness_failed"},
                            "prior_analysis_validation_error": "unsupported candidate metrics",
                            "interpretation": "INVALID_METRIC_CANARY", "reflection": {"explanation": "INVALID_METRIC_CANARY"}},
               "previous_analysis": "INVALID_METRIC_CANARY"}
    result = _model_context(context, analysis_only=True)
    assert "INVALID_METRIC_CANARY" not in json.dumps(result)
    assert result["previous"]["iteration"] == 3
    assert result["review_measurement_status"]["observation_sections"]["timing"].startswith("same stable parent")
    assert context["previous"]["interpretation"] == "INVALID_METRIC_CANARY"


@pytest.mark.parametrize("make_change", [True, False])
def test_planning_continuation_keeps_evidence_and_does_not_bypass_candidate_gate(proposal, tmp_path, monkeypatch, make_change):
    import qwen_agent.agents
    from vibeforge.agent.runner import _run_qwen_agent
    workspace, context, plan = proposal
    # Start from an unchanged workspace; the model must make the change itself.
    (workspace / "engine.py").write_text("value = 1\n")
    context.update(model={"max_context": 32768}, agent={"max_new_tokens": 512, "max_turns": 4, "timeout_seconds": 30})
    calls = []
    class FakeAssistant:
        def __init__(self, llm, function_list, system_message):
            self.tools = {t.name: t for t in function_list}
            calls.append(self)
        def run(self, messages):
            if len(calls) == 1:
                yield [{"role": "assistant", "content": "I have read evidence but have not implemented the idea."}]
                return
            assert len(calls) == 2
            assert json.loads(messages[0]["content"])["stable_ref"] == context["stable_ref"]
            assert messages[-2]["content"].startswith("I have read evidence")
            assert "only continuation" in messages[-1]["content"]
            updates = []
            if make_change:
                result = self.tools["write_file"].call(json.dumps({"path": "engine.py", "content": "value = 2\n"}))
                updates.append({"role": "function", "name": "write_file", "content": result})
            result = self.tools["submit_candidate"].call(json.dumps({"hypothesis": "Measured CPU change", "experiment_plan": plan}))
            updates.append({"role": "function", "name": "submit_candidate", "content": result})
            if not make_change:
                assert not json.loads(result)["submitted"]
                result = self.tools["report_blocked"].call(json.dumps({"reason": "No justified implementation was made."}))
                updates.append({"role": "function", "name": "report_blocked", "content": result})
            yield updates
    monkeypatch.setattr(qwen_agent.agents, "Assistant", FakeAssistant)
    result = _run_qwen_agent("http://127.0.0.1:1", context, workspace, tmp_path / "continued.jsonl", "Act on evidence.", "offline")
    assert len(calls) == 2 and result["complete"]
    assert bool(result.get("blocked")) is not make_change
    assert result["submission_origin"] == "agent" and result["recovered_submission"]


@pytest.mark.parametrize("exhausted_tools", [False, True])
def test_planning_continuation_is_bounded_by_count_and_original_tool_budget(tmp_path, monkeypatch, exhausted_tools):
    import qwen_agent.agents
    from vibeforge.agent.runner import _run_qwen_agent
    calls = []
    class FakeAssistant:
        def __init__(self, **kwargs):
            calls.append(self)
        def run(self, messages):
            if exhausted_tools:
                yield [{"role": "function", "name": "read_file", "content": "source"}]
            else:
                yield [{"role": "assistant", "content": "Still only discussing."}]
    monkeypatch.setattr(qwen_agent.agents, "Assistant", FakeAssistant)
    context = {"candidate_contract": {"version": 2}, "model": {"max_context": 32768},
               "agent": {"max_new_tokens": 512, "max_turns": 1, "timeout_seconds": 30}}
    result = _run_qwen_agent("http://127.0.0.1:1", context, tmp_path, tmp_path / "bounded.jsonl", "Act.", "offline")
    assert len(calls) == (1 if exhausted_tools else 2)
    assert result["blocked"] and result["submission_origin"] == "controller_recovery"


def test_planning_schema_rejects_flat_plan_and_allows_corrected_nested_submission(proposal, tmp_path):
    from vibeforge.agent.runner import TOOL_SCHEMAS
    workspace, context, plan = proposal
    submission = {}
    tool = next(t for t in _assistant_tools(workspace, tmp_path / "schema.jsonl", context, submission)
                if t.name == "submit_candidate")
    assert "experiment_plan" in tool.parameters["required"]
    assert "nested JSON object" in tool.description
    result = json.loads(tool.call(json.dumps({"hypothesis": "Measured change", **plan})))
    assert result["error"]["reason"] == "invalid_tool_arguments"
    assert "experiment_plan" in result["error"]["message"]
    assert not submission
    result = json.loads(tool.call(json.dumps({"hypothesis": "Measured change", "experiment_plan": plan})))
    assert result["submitted"] and submission["complete"]
    assert TOOL_SCHEMAS["submit_candidate"][2] == ["hypothesis"]


def test_submission_schema_varies_by_session_without_leaking_planning_requirements(tmp_path):
    import copy
    from vibeforge.agent.runner import TOOL_SCHEMAS
    original = copy.deepcopy(TOOL_SCHEMAS)
    context = {"candidate_contract": {"version": 2}, "previous": {"iteration": 1, "analysis_pending": True}}
    def submit(**kwargs):
        return next(t for t in _assistant_tools(tmp_path, tmp_path / "schema.jsonl", context, {}, **kwargs)
                    if t.name == "submit_candidate")
    planning = submit()
    assert set(planning.parameters["required"]) == {"hypothesis", "experiment_plan", "previous_analysis", "reflection"}
    nested = planning.parameters["properties"]["experiment_plan"]["required"]
    assert {"bottleneck_reference", "runtime_preconditions", "target_profile_ids", "expected_change_basis"} <= set(nested)
    analysis = submit(analysis_only=True)
    assert set(analysis.parameters["required"]) == {"hypothesis", "previous_analysis", "reflection"}
    smoke = submit(smoke=True)
    assert smoke.parameters["required"] == ["hypothesis"]
    assert TOOL_SCHEMAS == original


def test_archived_measurement_cannot_replace_current_bound_observation(proposal):
    workspace, context, plan = proposal
    archive = "archives/earlier/iterations/0001/observation.json"
    current = Path(context["evidence_root"]) / plan["evidence_paths"][0]
    write_json(Path(context["evidence_root"]) / archive, json.loads(current.read_text()))
    plan["evidence_paths"] = [archive]
    assert not candidate_gate(workspace, context, plan)["valid"]


@pytest.fixture
def archived_proposal(proposal):
    workspace, context, plan = proposal
    directory = Path(context["evidence_root"])
    entry = {"campaign_id": "earlier", "config_hash": "config1", "profile_hash": "p1",
             "evidence_directory": "archives/earlier", "memory_index": "archives/earlier/memory/index.json"}
    row = {"iteration": 3, "parent": "s1", "candidate": "failed-candidate",
           "hypothesis": "Previously evaluated change", "evidence_path": "iterations/0003",
           "facts": {"complete": True, "candidate_gate": {"valid": True, "fingerprint": semantic_fingerprint(workspace)},
                     "decision": {"outcome": "reject", "reason": "correctness_failed"}}}
    write_json(directory / entry["memory_index"], {"entries": [row]})
    return workspace, context, plan, directory, entry, row


def test_archived_duplicate_is_rejected_with_evidence_but_distinct_change_is_allowed(archived_proposal):
    workspace, context, plan, directory, entry, row = archived_proposal
    context["candidate_contract"]["tested_candidates"] = archived_candidates(directory, entry, "s1", "p1", "config1")
    gate = candidate_gate(workspace, context, plan)
    assert gate["reason"] == "duplicate_change"
    assert gate["previous_campaign"] == "earlier" and gate["previous_iteration"] == 3
    assert gate["previous_evidence_path"] == "archives/earlier/iterations/0003"
    assert gate["previous_decision"]["reason"] == "correctness_failed"
    (workspace / "engine.py").write_text("value = 3\n")
    assert candidate_gate(workspace, context, plan)["valid"]


@pytest.mark.parametrize("changed", ["config", "profile", "parent", "incomplete", "interrupted"])
def test_archived_duplicate_index_respects_frozen_conditions_and_completion(archived_proposal, changed):
    workspace, context, plan, directory, entry, row = archived_proposal
    if changed == "config":
        entry["config_hash"] = "different"
    elif changed == "profile":
        entry["profile_hash"] = "different"
    elif changed == "parent":
        row["parent"] = "different"
    elif changed == "incomplete":
        row["facts"]["complete"] = False
    else:
        row["facts"]["decision"]["reason"] = "interrupted_iteration"
    write_json(directory / entry["memory_index"], {"entries": [row]})
    assert archived_candidates(directory, entry, "s1", "p1", "config1") == []


def test_archived_candidate_evidence_cannot_escape_its_archive(archived_proposal):
    workspace, context, plan, directory, entry, row = archived_proposal
    row["evidence_path"] = "../outside"
    write_json(directory / entry["memory_index"], {"entries": [row]})
    with pytest.raises(ValueError, match="escapes workspace"):
        archived_candidates(directory, entry, "s1", "p1", "config1")


def test_archived_invalid_candidate_with_completed_pairing_is_deduplicated(archived_proposal):
    workspace, context, plan, directory, entry, row = archived_proposal
    row["facts"]["decision"] = {
        "outcome": "invalid", "reason": "profile_regression_not_excluded:concurrent_2",
    }
    row["facts"]["verification"] = {"gain": 0.001, "pairs": 5, "method": "paired_run_bootstrap"}
    write_json(directory / entry["memory_index"], {"entries": [row]})
    archived = archived_candidates(directory, entry, "s1", "p1", "config1")
    assert archived[0]["verification"]["pairs"] == 5
    context["candidate_contract"]["tested_candidates"] = archived
    assert candidate_gate(workspace, context, plan)["reason"] == "duplicate_change"


def test_restart_context_imports_archive_dedup_without_current_history(archived_proposal, paths, toy_config):
    from vibeforge.agent.context import create_context
    workspace, context, plan, directory, entry, row = archived_proposal
    archive_root = paths.campaign("current")
    shutil.copytree(directory, archive_root)
    entry["config_hash"] = toy_config["config_hash"]
    entries = [entry]
    for number in range(6):
        another = {**entry, "campaign_id": f"other-{number}",
                   "evidence_directory": f"archives/other-{number}",
                   "memory_index": f"archives/other-{number}/memory/index.json"}
        another_row = json.loads(json.dumps(row))
        another_row["facts"]["candidate_gate"]["fingerprint"] = f"different-source-{number}"
        write_json(archive_root / another["memory_index"], {"entries": [another_row]})
        entries.append(another)
    write_json(archive_root / "memory/archived-campaigns.json", {"entries": entries})
    write_json(paths.data / "profiles/p1.json", {"model": {}, "hardware": {}, "environment": {}})
    shutil.copytree(context["candidate_contract"]["parent_source"], paths.release("s1"))
    database = Database(paths.data / "state/context-test.sqlite3")
    try:
        database.put("current", {"stable_ref": "s1", "profile_hash": "p1"})
        rebuilt = create_context(database, paths, "current", toy_config, workspace, 1)
        assert rebuilt["context_schema_version"] == 4
        assert rebuilt["loop_protocol"]["profiling"] == toy_config["profiling"]
        assert rebuilt["workload_roles"]["deployment"]["profile_ids"] == ["default"]
        assert rebuilt["workload_roles"]["agent_self_hosting"]["admission"] is False
        assert rebuilt["history_total"] == 0 and rebuilt["previous"] is None
        assert rebuilt["pending_analyses"] == []
        gate = candidate_gate(workspace, rebuilt, plan)
        assert gate["reason"] == "duplicate_change" and gate["previous_campaign"] == "earlier"
        assert len(rebuilt["candidate_contract"]["tested_candidates"]) == 7
        assert len(rebuilt["archived_campaign_memory"]["entries"]) == 4
        archived_experience = rebuilt["archived_experience_knowledge"]
        assert archived_experience["verified_strategies"] == []
        assert archived_experience["counterexamples"][0]["history_ref"] == "earlier:3"
        assert archived_experience["counterexamples"][0]["failure"] == "correctness_failed"
        assert archived_experience["counterexamples"][0]["compatibility"]["profile_hash"] == "p1"
        summary = _model_context(rebuilt)["archived_candidate_results"]
        assert len(summary) == 6
        assert summary[0]["decision"]["reason"] == "correctness_failed"
        assert "fingerprint" not in summary[0]
    finally:
        database.close()


def test_restart_context_imports_equivalent_completed_database_campaign(paths, toy_config, tmp_path):
    from vibeforge.agent.context import create_context
    parent = tmp_path / "parent"
    workspace = tmp_path / "workspace"
    parent.mkdir()
    workspace.mkdir()
    (parent / "engine.py").write_text("value = 1\n")
    (parent / "engine.json").write_text('{"precision": {}}')
    (workspace / "engine.py").write_text("value = 2\n")
    (workspace / "engine.json").write_text('{"precision": {}}')
    shutil.copytree(parent, paths.release("s1"))
    write_json(paths.data / "profiles/p1.json", {"model": {}, "hardware": {}, "environment": {}})
    database = Database(paths.data / "state/context-database-history.sqlite3")
    previous = {
        "iteration": 3, "parent": "s1", "candidate": "candidate-3", "complete": True,
        "hypothesis": "Previously tested change", "candidate_gate": {"valid": True, "fingerprint": "same"},
        "decision": {"outcome": "invalid", "reason": "profile_regression_not_excluded:concurrent_2"},
        "experiment_verification": {"gain": 0.001, "pairs": 5, "method": "paired_run_bootstrap"},
    }
    ignored = {**previous, "iteration": 4,
               "candidate_gate": {"valid": True, "fingerprint": "interrupted"},
               "decision": {"outcome": "reject", "reason": "interrupted_iteration"}}
    try:
        common = {"stable_ref": "s1", "profile_hash": "p1", "config_hash": toy_config["config_hash"]}
        database.put("current", {"campaign_id": "current", **common})
        database.put("earlier", {"campaign_id": "earlier", **common})
        database.iteration("earlier", 3, previous)
        database.iteration("earlier", 4, ignored)
        prior_evidence = paths.campaign("earlier") / "iterations/0003"
        write_json(prior_evidence / "decision.json", previous["decision"])
        (prior_evidence / "change.patch").write_text("previous patch")

        rebuilt = create_context(database, paths, "current", toy_config, workspace, 1)
        imported = rebuilt["candidate_contract"]["tested_candidates"]
        assert [(row["campaign_id"], row["fingerprint"]) for row in imported] == [("earlier", "same")]
        assert rebuilt["archived_candidate_results"][0]["decision"]["reason"] == "profile_regression_not_excluded:concurrent_2"
        assert rebuilt["archived_candidate_results"][0]["verification"]["pairs"] == 5
        evidence = rebuilt["archived_candidate_results"][0]["evidence_path"]
        assert evidence == "memory/prior-campaigns/earlier/iterations/0003"
        assert read_json(paths.campaign("current") / evidence / "decision.json") == previous["decision"]
        assert (paths.campaign("current") / evidence / "change.patch").read_text() == "previous patch"
    finally:
        database.close()
