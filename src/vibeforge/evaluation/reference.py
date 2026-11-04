"""Capture reference data before candidate edits and verify its hash thereafter."""

from vibeforge.storage.atomic import digest, read_json, write_json
import random


def capture_reference(client, cases, path, provenance, protocol=None, workload=None):
    protocol = protocol or {}
    vocab_size = provenance.get("vocab_size", 32)
    count = min(protocol.get("logit_samples", 64), vocab_size)
    indices = sorted(random.Random(protocol.get("seed", 17)).sample(range(vocab_size), count))
    results = []
    benchmark_cases = [{"id": f"benchmark-{i}", "prompt": prompt, "max_new_tokens": workload["max_new_tokens"],
                        "fixed_length": True} for i, prompt in enumerate(workload["prompts"])] if workload else []
    if not cases:
        raise ValueError("A reference requires nonempty correctness cases")
    for case in [*cases, *benchmark_cases]:
        output = client.generate(prompt=case["prompt"], max_new_tokens=case.get("max_new_tokens", 4),
                                 **({"fixed_length": True} if case.get("fixed_length") else {}))
        numeric = client.logits(input_ids=output["input_ids"], indices=indices)
        continuation = client.logits(input_ids=output["input_ids"] + output["token_ids"][:2],
                                     indices=indices)
        results.append({"id": case["id"], "prompt": case["prompt"], "input_ids": output["input_ids"],
                        "max_new_tokens": case.get("max_new_tokens", 4),
                        "token_ids": output["token_ids"], "logits": numeric,
                        "continuation_logits": continuation,
                        "fixed_length": bool(case.get("fixed_length")),
                        "expected_contains": case.get("expected_contains")})
    result = {"schema_version": 2, "provenance": provenance, "cases": results,
              "benchmark_cases": [case for case in results if case["fixed_length"]], "vocab_size": vocab_size}
    result["hash"] = digest(result)
    write_json(path, result)
    return result


def load_reference(path):
    value = read_json(path)
    if value["hash"] != digest({k: v for k, v in value.items() if k != "hash"}):
        raise ValueError("Frozen reference was modified")
    return value
