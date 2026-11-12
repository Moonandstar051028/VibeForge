import pytest
from vibeforge.contracts.campaign import load_config
from vibeforge.contracts.precision import normalize_precision, precision_fields
from vibeforge.controller.supervisor import Supervisor
from vibeforge.storage.atomic import read_json, write_json


@pytest.mark.parametrize("reject", [False, True])
def test_precision_search_is_gated_and_restores_or_promotes(paths, project, toy_config, tmp_path, reject):
    precision = normalize_precision({"compute_dtype": "float32"})
    config = toy_config
    config["optimization"] = {"allow_precision_changes": True, "allowed_precisions": [precision]}
    quality = {"command": ["{python}", "{project}/tests/fixtures/quality_evaluator.py",
                           "--request", "{request}", "--output", "{output}"],
               "files": ["tests/fixtures/quality_evaluator.py"],
               "min_scores": {"accuracy": .9}, "max_regressions": {"accuracy": .01}}
    if reject:
        quality["command"].append("--reject-float32")
    config["evaluation"].update(correctness_mode="approximate", quality=quality, max_retests=0)
    config["workload"]["concurrency"] = 2
    candidate = read_json(project / "templates/engine/toy/engine.json")
    candidate.update(device="cpu", model_loader="auto", **precision_fields(precision))
    config["agent"]["actions"] = [{"hypothesis": "Fixture tests a declared precision change with equivalent arithmetic.", "edits": [
        {"tool": "write_file", "arguments": {"path": "engine.json", "content": __import__('json').dumps(candidate)}},
        {"tool": "replace_text", "arguments": {"path": "engine.py", "old": "work = sum(range(120000))",
                                               "new": "work = 120000 * 119999 // 2"}},
    ]}]
    config["budget"]["max_iterations"] = 1
    file = tmp_path / "campaign.json"
    write_json(file, config)
    config = load_config(file, project)
    controller = Supervisor(paths, "precision-search")
    try:
        controller.create(config)
        state = controller.run()
        row = controller.database.iteration("precision-search", 1)
        assert state["state"] == "STOPPED" and not state["analysis_pending"]
        assert row["decision"]["outcome"] == ("reject" if reject else "accept")
        if reject:
            assert row["decision"]["reason"] == "quality_failed"
            assert state["stable_ref"] == state["initial_ref"]
        else:
            assert state["stable_ref"] == row["candidate"]
            context = read_json(paths.campaign("precision-search") / "final-analysis/0001/context.json")
            assert context["precision"] == precision
            assert context["profile_hash"] == row["profile_hash"] != state["profile_hash"]
        quality_report = read_json(paths.campaign("precision-search") / "iterations/0001/quality/report.json")
        assert quality_report["passed"] is not reject
        assert quality_report["binding"]["precision"] == precision
        controller.run(resume=True)
        assert controller.database.get("precision-search")["stable_ref"] == state["stable_ref"]
    finally:
        controller.close()
