import json

from vibeforge.contracts.workload import normalize_workload
from vibeforge.controller.supervisor import Supervisor
from vibeforge.storage.atomic import digest, read_json, write_json


def test_matrix_reference_validation_benchmark_and_post_profile(paths, toy_config, monkeypatch):
    # Deterministic fixture supplies a v2 plan from real host evidence. This tests
    # the orchestration/verification contract, not an LLM's reasoning ability.
    import vibeforge.controller.supervisor as supervisor_module
    original_context, original_session = supervisor_module.create_context, Supervisor.agent_session
    def context(*args, **kwargs):
        value = original_context(*args, **kwargs)
        if value.get("candidate_contract"):
            value["candidate_contract"]["version"] = 2
        return value
    def session(self, client, workspace, iteration, analysis_only=False, **kwargs):
        result = original_session(self, client, workspace, iteration, analysis_only)
        if not analysis_only:
            plan = result["experiment_plan"]
            name = plan["evidence_paths"][0]
            observation = read_json(self.directory / name)
            measurement = observation["measurements"]["normal.backend_wall_seconds"]
            plan.update(bottleneck_reference={"evidence_path": name, "metric": "normal.backend_wall_seconds",
                        "value": measurement["value"], "unit": measurement["unit"]},
                        target_profile_ids=["medium"],
                        runtime_preconditions=[{"field": "request_scheduler", "equals": "serialized_lock"}],
                        expected_bottleneck_change_pct=1, bottleneck_direction="decrease",
                        expected_change_basis="Eliminate fixture's repeated CPU arithmetic")
            # V3 Agent submissions deliberately omit controller-owned fields.
            # The controller gate must persist its completed plan before the
            # decision and verification stages consume it.
            plan.pop("metric", None)
            plan.pop("direction", None)
            write_json(self.directory / f"iterations/{iteration:04d}/agent-result.json", result)
        return result
    monkeypatch.setattr(supervisor_module, "create_context", context)
    monkeypatch.setattr(Supervisor, "agent_session", session)
    toy_config["workload"] = normalize_workload({"cases": "benchmarks/cases/toy.json", "observation_profile": "medium",
        "profiles": [{"id": "short", "prompts": ["hello"], "max_new_tokens": 4, "weight": 1},
                     {"id": "medium", "prompts": ["hello world repeated"], "max_new_tokens": 8, "weight": 2, "concurrency": 2}]})
    toy_config["evaluation"].update(concurrency=2, max_retests=0)
    toy_config["profiling"].update(coverage="all", max_matrix_profiles=4)
    toy_config["budget"].update(max_iterations=1, max_proposals=1)
    toy_config["config_hash"] = digest({k: v for k, v in toy_config.items() if k != "config_hash"})
    controller = Supervisor(paths, "matrix-validation")
    try:
        controller.create(toy_config)
        archive = controller.directory / "archives" / "earlier-campaign" / "memory" / "index.json"
        write_json(archive, {"entries": [{"iteration": 3, "facts": {"decision": {"reason": "correctness_failed"}}}]})
        write_json(controller.directory / "memory" / "archived-campaigns.json", {"entries": [{
            "campaign_id": "earlier-campaign", "memory_index": "archives/earlier-campaign/memory/index.json",
            "evidence_directory": "archives/earlier-campaign", "completed_attempts": 3,
        }]})
        state = controller.run()
        folder = paths.campaign("matrix-validation")
        iteration = controller.database.iterations("matrix-validation")[0]
        assert state["completed_experiments"] == 1
        inherited = read_json(folder / "iterations/0001/context.json")["archived_campaign_memory"]
        agent_context = read_json(folder / "iterations/0001/context.json")
        matrix = agent_context["performance_observation"]["profile_matrix"]
        assert matrix["coverage"] == "all" and matrix["selected_cases"] == 2
        assert {entry["profile_id"] for entry in matrix["entries"]} == {"short", "medium"}
        assert {entry["evidence_path"] for entry in matrix["entries"]}.issubset(
            set(agent_context["candidate_contract"]["observation_paths"]))
        assert read_json(folder / "iterations/0001/agent-telemetry.json")["kind"] == "agent_inference_telemetry"
        from vibeforge.agent.tools import AgentTools
        evidence = AgentTools(paths.release(state["stable_ref"]), folder / "archive-read.jsonl", folder)
        archived = evidence.execute("read_evidence", {"path": inherited["entries"][0]["memory_index"]})
        assert json.loads(archived)["entries"][0]["facts"]["decision"]["reason"] == "correctness_failed"
        assert len(controller.database.iterations("matrix-validation")) == 1
        assert iteration["decision"]["outcome"] == "accept"
        assert {c["profile_id"] for c in read_json(folder / "reference.json")["benchmark_cases"]} == {"short", "medium"}
        result = read_json(folder / "iterations/0001/benchmarks/summary-0.json")
        assert set(result["baseline"][0]["profile_records"]) == {"short", "medium"}
        assert read_json(folder / "iterations/0001/candidate-observation.json")["profile_id"] == "medium"
        verification = read_json(folder / "iterations/0001/verification.json")
        normalized_plan = read_json(folder / "iterations/0001/agent-result.json")["experiment_plan"]
        assert normalized_plan["metric"] == "throughput"
        assert normalized_plan["direction"] == "increase"
        assert verification["mechanism"]["metric"] == "normal.backend_wall_seconds"
        assert verification["mechanism"]["status"] != "unavailable"
        assert verification["mechanism"]["causal_proof"] is False
        memory = read_json(folder / "iterations/0001/memory.json")
        assert memory["facts"]["verification"]["mechanism"] == verification["mechanism"]
    finally:
        controller.close()
