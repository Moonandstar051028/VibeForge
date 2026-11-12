"""Load versioned knowledge while excluding schemas and non-rule documents."""

import copy
import hashlib
import shutil
from pathlib import Path

from vibeforge.knowledge.validator import validate_rule
from vibeforge.storage.atomic import read_json, write_json
from vibeforge.storage.paths import confined


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


def materialize_evidence(rules, project, evidence_root):
    """Copy referenced global evidence into the campaign's confined evidence tree."""
    project, evidence_root = Path(project).resolve(), Path(evidence_root).resolve()
    result = copy.deepcopy(rules)
    for rule in result:
        rule_key = hashlib.sha256(rule["rule_id"].encode()).hexdigest()[:16]
        for index, entry in enumerate(rule.get("evidence", [])):
            reference = entry.get("path")
            if not isinstance(reference, str) or not reference or Path(reference).is_absolute():
                continue
            try:
                source = confined(project, reference)
            except (TypeError, ValueError):
                continue
            if not source.is_file():
                continue
            relative = f"knowledge/evidence/{rule_key}/{index:04d}-{source.name}"
            destination = confined(evidence_root, relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            entry["source_path"] = reference
            entry["path"] = relative
    return result
