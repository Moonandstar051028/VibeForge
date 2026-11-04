"""Rebuild restart context from committed state and evidence paths."""

from vibeforge.storage.atomic import write_json, read_json


def create_context(database, paths, campaign_id, config, workspace, iteration):
    state = database.get(campaign_id)
    history = database.iterations(campaign_id)
    previous = next((row for row in reversed(history) if row["iteration"] < iteration), None)
    profile = read_json(paths.data / "profiles" / f"{state['profile_hash']}.json")
    value = {
        "campaign_id": campaign_id, "iteration": iteration, "stable_ref": state.get("stable_ref"),
        "config_hash": config["config_hash"], "profile_hash": state.get("profile_hash"),
        "workload": config["workload"], "protocol": config["evaluation"],
        "workspace": str(workspace), "previous": previous,
        "history": [{"iteration": row["iteration"], "candidate": row.get("candidate"),
                     "decision": row.get("decision"), "error": row.get("error")} for row in history],
        "budget": config["budget"], "agent": config["agent"],
        "source": (workspace / "engine.py").read_text()[:28000],
        "instructions": "Explain previous results, propose one falsifiable hypothesis, edit only the workspace, then submit.",
        "model_features": profile["model"].get("text_config", profile["model"]),
        "hardware": profile["hardware"], "environment": profile["environment"],
        "evidence_root": str(paths.campaign(campaign_id)),
    }
    write_json(paths.campaign(campaign_id) / "resume" / "context.json", value)
    return value
