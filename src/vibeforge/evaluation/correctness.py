"""Differential generation and sampled-logit checks with explicit coverage."""

import math


def compare_values(actual, expected, atol, rtol):
    if len(actual) != len(expected):
        return False
    return all(math.isfinite(float(a)) and math.isfinite(float(b)) and
               abs(a - b) <= atol + rtol * abs(b) for a, b in zip(actual, expected))


def evaluate_correctness(client, reference, protocol):
    checks = []
    for case in reference["cases"]:
        try:
            output = client.generate(prompt=case["prompt"], max_new_tokens=case["max_new_tokens"])
            numeric = client.logits(input_ids=case["input_ids"], indices=case["logits"]["indices"])
            passed = compare_values(numeric["values"], case["logits"]["values"],
                                    protocol.get("atol", 1e-4), protocol.get("rtol", 1e-4))
            passed = passed and output["input_ids"] == case["input_ids"]
            passed = passed and all(isinstance(token, int) for token in output["token_ids"])
            if "continuation_logits" in case:
                continuation = client.logits(input_ids=case["input_ids"] + case["token_ids"][:2],
                                             indices=case["continuation_logits"]["indices"])
                passed = passed and compare_values(continuation["values"], case["continuation_logits"]["values"],
                                                  protocol.get("atol", 1e-4), protocol.get("rtol", 1e-4))
            # Interleave another request, then repeat the original request to
            # expose request-state contamination in generated cache code.
            client.generate(prompt="request isolation sentinel", max_new_tokens=1)
            repeated = client.generate(prompt=case["prompt"], max_new_tokens=case["max_new_tokens"])
            passed = passed and repeated["token_ids"] == output["token_ids"]
            if protocol.get("exact_tokens", True):
                passed = passed and output["token_ids"] == case["token_ids"]
            if case.get("expected_contains"):
                passed = passed and case["expected_contains"] in output["text"]
            checks.append({"case": case["id"], "passed": bool(passed), "actual_tokens": output["token_ids"]})
        except Exception as error:
            checks.append({"case": case["id"], "passed": False, "error": str(error)})
    return {"passed": bool(checks) and all(c["passed"] for c in checks), "checks": checks,
            "coverage": ["deterministic_generation", "tokenization", "sampled_prefill_logits",
                         "sampled_continuation_logits", "sequential_request_isolation"],
            "not_covered": ["full_logits", "per_layer_numerics", "multimodal", "concurrency",
                            "long_context", "long_duration_stability", "task_quality_benchmark"]}
