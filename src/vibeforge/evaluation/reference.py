"""Capture reference data before candidate edits and verify its hash thereafter."""

from vibeforge.storage.atomic import digest, read_json, write_json


def capture_reference(client, cases, path, provenance):
    results = []
    for case in cases:
        output = client.generate(prompt=case["prompt"], max_new_tokens=case.get("max_new_tokens", 4))
        numeric = client.logits(input_ids=output["input_ids"], indices=list(range(32)))
        continuation = client.logits(input_ids=output["input_ids"] + output["token_ids"][:2],
                                     indices=list(range(32)))
        results.append({"id": case["id"], "prompt": case["prompt"], "input_ids": output["input_ids"],
                        "max_new_tokens": case.get("max_new_tokens", 4),
                        "token_ids": output["token_ids"], "logits": numeric,
                        "continuation_logits": continuation,
                        "expected_contains": case.get("expected_contains")})
    result = {"provenance": provenance, "cases": results}
    result["hash"] = digest(result)
    write_json(path, result)
    return result


def load_reference(path):
    value = read_json(path)
    if value["hash"] != digest({k: v for k, v in value.items() if k != "hash"}):
        raise ValueError("Frozen reference was modified")
    return value
