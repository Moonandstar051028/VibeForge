"""Rebuild restart context from committed state and evidence paths."""

from vibeforge.storage.atomic import write_json, read_json


def create_context(database, paths, campaign_id, config, workspace, iteration):
    state = database.get(campaign_id)
    history = database.iterations(campaign_id)
    previous = next((row for row in reversed(history) if row["iteration"] < iteration), None)
    pending = [row for row in history if row["iteration"] < iteration and row.get("analysis_pending")]
    if pending:
        previous = pending[0]
    profile_hash = state.get("version_profiles", {}).get(state.get("stable_ref"), state["profile_hash"])
    profile = read_json(paths.data / "profiles" / f"{profile_hash}.json")
    value = {
        "campaign_id": campaign_id, "iteration": iteration, "stable_ref": state.get("stable_ref"),
        "config_hash": config["config_hash"], "profile_hash": profile_hash,
        "precision": read_json(workspace / "engine.json").get("precision", config["precision"]),
        "optimization": config["optimization"], "requirements": config["requirements"],
        "engine_adapter": config["engine"],
        "pending_analyses": [row["iteration"] for row in pending],
        "workload": config["workload"], "protocol": config["evaluation"],
        "workspace": str(workspace), "previous": previous,
        "history": [{"iteration": row["iteration"], "candidate": row.get("candidate"),
                     "decision": row.get("decision"), "error": row.get("error")} for row in history],
        "budget": config["budget"], "agent": config["agent"],
        "model": {"name": config["model"].get("name", "vibeforge-local")},
        "source": (workspace / "engine.py").read_text()[:28000],
        "instructions": "Explain previous results, propose one falsifiable hypothesis, edit only the workspace, then submit.",
        "model_features": profile["model"].get("text_config", profile["model"]),
        "hardware": profile["hardware"], "environment": profile["environment"],
        "evidence_root": str(paths.campaign(campaign_id)),
    }
    write_json(paths.campaign(campaign_id) / "resume" / "context.json", value)
    return value
