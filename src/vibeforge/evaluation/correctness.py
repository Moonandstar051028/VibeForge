"""Differential generation and sampled-logit checks with explicit coverage."""

import math
from concurrent.futures import ThreadPoolExecutor


def compare_values(actual, expected, atol, rtol):
    if not actual or len(actual) != len(expected):
        return False
    return all(math.isfinite(float(a)) and math.isfinite(float(b)) and
               abs(a - b) <= atol + rtol * abs(b) for a, b in zip(actual, expected))


def compare_generation(output, case, protocol):
    tokens = output.get("token_ids", [])
    if not tokens or any(type(token) is not int or token < 0 for token in tokens):
        return False
    if output.get("input_ids") != case["input_ids"] or len(tokens) > case["max_new_tokens"]:
        return False
    if case.get("fixed_length") and len(tokens) != case["max_new_tokens"]:
        return False
    if protocol.get("correctness_mode", "strict") == "strict":
        return tokens == case["token_ids"]
    agreement = sum(a == b for a, b in zip(tokens, case["token_ids"])) / max(len(tokens), len(case["token_ids"]), 1)
    return agreement >= protocol.get("min_token_agreement", 0.9)


def numeric_report(actual, expected, protocol):
    if actual.get("indices") != expected.get("indices"):
        return {"passed": False, "error": "logit_indices_mismatch"}
    a, b = actual.get("values", []), expected.get("values", [])
    passed = compare_values(a, b, protocol.get("atol", 1e-4), protocol.get("rtol", 1e-4))
    if not a or len(a) != len(b) or not all(math.isfinite(float(v)) for v in [*a, *b]):
        return {"passed": False, "error": "empty_mismatched_or_nonfinite_logits"}
    differences = [abs(x - y) for x, y in zip(a, b)]
    rms = math.sqrt(sum(d * d for d in differences) / len(a))
    normalized = rms / max(math.sqrt(sum(x * x for x in b) / len(b)), 1e-6)
    return {"passed": passed and normalized <= protocol.get("max_normalized_rms", math.inf),
            "max_absolute_error": max(differences), "normalized_rms": normalized}


def evaluate_correctness(client, reference, protocol):
    checks = []
    for case in reference["cases"]:
        try:
            arguments = {"prompt": case["prompt"], "max_new_tokens": case["max_new_tokens"]}
            if case.get("fixed_length"):
                arguments["fixed_length"] = True
            output = client.generate(**arguments)
            numeric = client.logits(input_ids=case["input_ids"], indices=case["logits"]["indices"])
            numerical = numeric_report(numeric, case["logits"], protocol)
            passed = numerical["passed"] and compare_generation(output, case, protocol)
            if "continuation_logits" in case:
                continuation = client.logits(input_ids=case["input_ids"] + case["token_ids"][:2],
                                             indices=case["continuation_logits"]["indices"])
                passed = passed and numeric_report(continuation, case["continuation_logits"], protocol)["passed"]
            # Interleave another request, then repeat the original request to
            # expose request-state contamination in generated cache code.
            client.generate(prompt="request isolation sentinel", max_new_tokens=1)
            repeated = client.generate(**arguments)
            passed = passed and repeated["token_ids"] == output["token_ids"]
            if case.get("expected_contains"):
                passed = passed and case["expected_contains"] in output["text"]
            checks.append({"case": case["id"], "passed": bool(passed), "actual_tokens": output["token_ids"],
                           "fixed_length": bool(case.get("fixed_length")), "numerical": numerical})
        except Exception as error:
            checks.append({"case": case["id"], "passed": False, "error": str(error)})
    if protocol.get("concurrency", 1) > 1:
        try:
            cases = reference.get("benchmark_cases") or reference["cases"]
            with ThreadPoolExecutor(max_workers=protocol["concurrency"]) as pool:
                outputs = list(pool.map(lambda case: client.generate(
                    prompt=case["prompt"], max_new_tokens=case["max_new_tokens"],
                    **({"fixed_length": True} if case.get("fixed_length") else {})),
                    cases * protocol["concurrency"]))
            checks.append({"case": "concurrent_request_isolation", "passed": all(
                compare_generation(output, case, protocol)
                for output, case in zip(outputs, cases * protocol["concurrency"]))})
        except Exception as error:
            checks.append({"case": "concurrent_request_isolation", "passed": False, "error": str(error)})
    coverage = ["deterministic_generation", "tokenization", "sampled_prefill_logits", "sequential_request_isolation"]
    if any("continuation_logits" in case for case in reference["cases"]):
        coverage.append("sampled_continuation_logits")
    if reference.get("benchmark_cases"):
        coverage.append("fixed_length_benchmark_path")
    if protocol.get("concurrency", 1) > 1:
        coverage.append("concurrent_request_isolation")
    return {"passed": bool(checks) and all(c["passed"] for c in checks), "checks": checks,
            "coverage": coverage,
            "not_covered": ["full_logits", "per_layer_numerics", "multimodal", "long_context",
                            "long_duration_stability", "task_quality_benchmark"] + (
                                ["concurrency"] if protocol.get("concurrency", 1) == 1 else [])}
