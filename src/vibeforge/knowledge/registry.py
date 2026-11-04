"""Load versioned knowledge while excluding schemas and non-rule documents."""

from vibeforge.knowledge.validator import validate_rule
from vibeforge.storage.atomic import read_json, write_json


def load_rules(directory):
    result = []
    for path in sorted(directory.rglob("*.json")):
        value = read_json(path)
        if "rule_id" in value:
            result.append({**validate_rule(value), "file": str(path)})
    return result


def build_index(directory, output):
    rules = load_rules(directory)
    write_json(output, {"rules": rules})
    return rules
