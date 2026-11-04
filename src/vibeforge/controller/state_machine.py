"""Explicit normal transitions; recovery is a separate audited entry."""


TRANSITIONS = {
    "READY": {"AGENT_RUNNING", "STOPPED", "ACTIVATING"},
    "AGENT_RUNNING": {"CANDIDATE_FROZEN", "RECOVERING"},
    "CANDIDATE_FROZEN": {"BUILDING", "DECIDING"},
    "BUILDING": {"VALIDATING", "DECIDING"},
    "VALIDATING": {"BENCHMARKING", "DECIDING"},
    "BENCHMARKING": {"DECIDING", "RECOVERING"},
    "DECIDING": {"BENCHMARKING", "ACTIVATING"},
    "ACTIVATING": {"COMMITTING", "READY", "ROLLING_BACK"},
    "COMMITTING": {"READY", "RECOVERING"},
    "ROLLING_BACK": {"READY", "HALTED"},
    "RECOVERING": {"READY", "BENCHMARKING", "HALTED"},
    "STOPPED": {"RECOVERING"},
    "HALTED": {"RECOVERING"},
    "BLOCKED": {"RECOVERING"},
}


def transition(database, campaign_id, target, **changes):
    current = database.get(campaign_id)["state"]
    if target not in TRANSITIONS.get(current, set()):
        raise ValueError(f"Invalid transition: {current} -> {target}")
    return database.update(campaign_id, {"state": target, **changes})
