import pytest

from vibeforge.contracts.campaign import load_config
from vibeforge.controller.supervisor import Supervisor
from vibeforge.storage.atomic import read_json, write_json, digest


def test_repeatable_but_wrong_initial_engine_cannot_be_promoted(paths, project, toy_config, tmp_path):
    # Toy generates deterministic integer tokens; the human-owned expected answer
    # is intentionally different. Perfect self-agreement must not pass the oracle.
    cases = tmp_path / "truth.json"
    write_json(cases, [{"id": "addition", "mode": "generate", "prompt": "2 + 2 =",
                       "max_new_tokens": 1, "expected": "independent correct answer", "match": "exact"}])
    toy_config["evaluation"]["quality"] = {
        "command": ["{python}", "{project}/scripts/evaluation/quality_cases.py",
                    "--request", "{request}", "--output", "{output}", "--cases", str(cases)],
        "files": ["scripts/evaluation/quality_cases.py", str(cases)],
        "repeats": 2, "min_scores": {"accuracy": 1.0}, "max_regressions": {"accuracy": 0},
    }
    config_path = tmp_path / "campaign.json"
    write_json(config_path, toy_config)
    config = load_config(config_path, project)
    controller = Supervisor(paths, "wrong-initial-engine")
    try:
        controller.create(config)
        with pytest.raises(RuntimeError, match="correctness/quality gates failed"):
            controller.run()
        state = controller.database.get("wrong-initial-engine")
        assert state["state"] == "HALTED" and not state.get("stable_ref")
        report = read_json(paths.campaign("wrong-initial-engine") / "bootstrap/correctness.json")
        assert all(check["passed"] for check in report["checks"])
        assert not report["passed"] and not report["quality"]["passed"]
        assert report["quality"]["metrics"]["accuracy"] == 0
        assert not controller.database.iterations("wrong-initial-engine")
    finally:
        controller.close()


def test_old_campaign_cannot_resume_under_new_loop_contract(paths, toy_config):
    toy_config["schema_version"] = 2
    toy_config["config_hash"] = digest({k: v for k, v in toy_config.items() if k != "config_hash"})
    controller = Supervisor(paths, "old-contract")
    try:
        controller.create(toy_config)
        with pytest.raises(ValueError, match="schema v4"):
            controller.run(resume=True)
        assert not controller.database.get("old-contract").get("stable_ref")
    finally:
        controller.close()
