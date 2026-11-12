"""Condition-first retrieval with explicit rule status."""

import json


def retrieve(rules, profile, query="", limit=5):
    matches = []
    words = set(query.lower().split())
    for rule in rules:
        # Invalidated rules are measured counterevidence. Keep them visible to
        # planning while their status prevents treating them as recommendations.
        if rule["status"] not in ("local_validated", "cross_config_validated", "invalidated"):
            continue
        if not all(profile.get(key) == value for key, value in rule["conditions"].items()):
            continue
        content = json.dumps(rule, ensure_ascii=False).lower()
        score = sum(word in content for word in words)
        matches.append((score, rule))
    return [rule for _, rule in sorted(matches, key=lambda entry: entry[0], reverse=True)[:limit]]
