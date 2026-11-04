"""Evidence requirements for knowledge states."""


def validate_rule(rule):
    required = ("rule_id", "status", "conditions", "implementation", "validation", "evidence")
    missing = [field for field in required if not rule.get(field)]
    if missing:
        raise ValueError(f"Knowledge rule missing fields: {missing}")
    if rule["status"] not in ("draft", "local_validated", "cross_config_validated", "invalidated"):
        raise ValueError("Unknown knowledge status")
    if rule["status"] == "cross_config_validated":
        profiles = {entry["profile_hash"] for entry in rule["evidence"] if entry.get("profile_hash")}
        if len(profiles) < 2:
            raise ValueError("Cross-configuration claims require distinct profile evidence")
    return rule
