"""Differential generation and sampled-logit checks with explicit coverage."""

import math
from concurrent.futures import ThreadPoolExecutor


def compare_values(actual, expected, atol, rtol, minimum=None, maximum=None):
    if not actual or len(actual) != len(expected):
        return False
    minimum = expected if minimum is None else minimum
    maximum = expected if maximum is None else maximum
    if len(minimum) != len(expected) or len(maximum) != len(expected):
        return False
    return all(math.isfinite(float(a)) and math.isfinite(float(b)) and
               math.isfinite(float(low)) and math.isfinite(float(high)) and
               low - (atol + rtol * max(abs(low), abs(b))) <= a <= high + (atol + rtol * max(abs(high), abs(b)))
               for a, b, low, high in zip(actual, expected, minimum, maximum))


def compare_generation(output, case, protocol):
    tokens = output.get("token_ids", [])
    if not isinstance(tokens, list) or not tokens or any(type(token) is not int or token < 0 for token in tokens):
        return False
    if output.get("input_ids") != case["input_ids"] or len(tokens) > case["max_new_tokens"]:
        return False
    if case.get("fixed_length") and len(tokens) != case["max_new_tokens"]:
        return False
    references = case.get("accepted_token_ids") or [case["token_ids"]]
    if protocol.get("correctness_mode", "strict") == "strict":
        return any(tokens == reference for reference in references)
    agreement = max(
        sum(a == b for a, b in zip(tokens, reference)) / max(len(tokens), len(reference), 1)
        for reference in references
    )
    return agreement >= protocol.get("min_token_agreement", 0.9)


def numeric_report(actual, expected, protocol):
    if actual.get("indices") != expected.get("indices"):
        return {"passed": False, "error": "logit_indices_mismatch"}
    a, b = actual.get("values", []), expected.get("values", [])
    if (not isinstance(a, list) or not isinstance(b, list) or not a or len(a) != len(b)
            or not all(type(v) in (int, float) and math.isfinite(v) for v in [*a, *b])):
        return {"passed": False, "error": "empty_mismatched_or_nonfinite_logits"}
    minimum, maximum = expected.get("min_values", b), expected.get("max_values", b)
    if (not isinstance(minimum, list) or not isinstance(maximum, list)
            or len(minimum) != len(b) or len(maximum) != len(b)
            or not all(type(v) in (int, float) and math.isfinite(v) for v in [*minimum, *maximum])
            or any(not low <= center <= high for low, center, high in zip(minimum, b, maximum))):
        return {"passed": False, "error": "invalid_reference_envelope"}
    # A noisy baseline must not grant an unbounded error allowance to candidates.
    reference_atol = protocol.get("reference_atol", protocol.get("atol", 1e-4))
    reference_rtol = protocol.get("reference_rtol", protocol.get("rtol", 1e-4))
    if any(max(center - low, high - center) > reference_atol + reference_rtol * abs(center)
           for low, center, high in zip(minimum, b, maximum)):
        return {"passed": False, "error": "reference_numeric_instability"}
    passed = compare_values(a, b, protocol.get("atol", 1e-4), protocol.get("rtol", 1e-4), minimum, maximum)
    differences = [abs(x - y) for x, y in zip(a, b)]
    rms = math.sqrt(sum(d * d for d in differences) / len(a))
    normalized = rms / max(math.sqrt(sum(x * x for x in b) / len(b)), 1e-6)
    envelope_excess = [max(minimum[i] - a[i], a[i] - maximum[i], 0)
                       for i in range(len(a))]
    return {"passed": passed and normalized <= protocol.get("max_normalized_rms", math.inf),
            "max_absolute_error": max(differences), "max_reference_envelope_excess": max(envelope_excess),
            "normalized_rms": normalized, "reference_repeat_count": expected.get("repeat_count", 1)}


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
            generation_passed = compare_generation(output, case, protocol)
            continuation_report = None
            passed = numerical["passed"] and generation_passed
            if "continuation_logits" in case:
                continuation = client.logits(input_ids=case["input_ids"] + case["token_ids"][:2],
                                             indices=case["continuation_logits"]["indices"])
                continuation_report = numeric_report(continuation, case["continuation_logits"], protocol)
                passed = passed and continuation_report["passed"]
            # Interleave another request, then repeat the original request to
            # expose request-state contamination in generated cache code.
            client.generate(prompt="request isolation sentinel", max_new_tokens=1)
            repeated = client.generate(**arguments)
            repeat_passed = compare_generation(repeated, case, protocol)
            passed = passed and repeat_passed
            if case.get("expected_contains"):
                passed = passed and case["expected_contains"] in output["text"]
            checks.append({"case": case["id"], "passed": bool(passed), "actual_tokens": output["token_ids"],
                           "repeat_tokens": repeated["token_ids"], "generation_passed": generation_passed,
                           "repeat_passed": repeat_passed, "fixed_length": bool(case.get("fixed_length")),
                           "numerical": numerical, "continuation_numerical": continuation_report})
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
    coverage = ["generation_reference_agreement", "tokenization", "sampled_prefill_logits", "sequential_request_isolation"]
    if any("continuation_logits" in case for case in reference["cases"]):
        coverage.append("sampled_continuation_logits")
    if reference.get("benchmark_cases"):
        coverage.append("fixed_length_benchmark_path")
    if protocol.get("concurrency", 1) > 1:
        coverage.append("concurrent_request_isolation")
    if reference.get("reference_repeats", 1) > 1:
        coverage.append("repeated_reference_stability")
    input_lengths = [len(case["input_ids"]) for case in reference["cases"]]
    failed = [check for check in checks if not check["passed"]]
    return {"passed": bool(checks) and not failed, "checks": checks,
            "coverage": coverage,
            "reference_schema_version": reference.get("schema_version", 2),
            "reference_role": reference.get("reference_role", "legacy_regression_baseline"),
            "failure_summary": {"failed_checks": len(failed),
                                "operational_errors": sum(bool(check.get("error")) for check in failed)},
            "tested_input_token_range": [min(input_lengths), max(input_lengths)] if input_lengths else None,
            "workload_profile_ids": sorted({c["profile_id"] for c in reference["cases"] if c.get("profile_id")}),
            "not_covered": ["full_logits", "per_layer_numerics", "multimodal", "unconfigured_context_lengths",
                            "long_duration_stability", "task_quality_benchmark"] + (
                                ["concurrency"] if protocol.get("concurrency", 1) == 1 else [])}


def evaluate_candidate_precheck(client, reference, protocol):
    """Cheaply reject request-state or concurrency breakage before the full suite."""
    cases = reference.get("benchmark_cases") or reference["cases"]
    case = min(cases, key=lambda item: (item.get("max_new_tokens", 1), len(item.get("input_ids", []))))
    arguments = {"prompt": case["prompt"], "max_new_tokens": case["max_new_tokens"]}
    if case.get("fixed_length"):
        arguments["fixed_length"] = True
    checks = []
    try:
        first = client.generate(**arguments)
        client.generate(prompt="request isolation sentinel", max_new_tokens=1)
        repeated = client.generate(**arguments)
        checks.append({"case": "sequential_request_isolation", "passed": bool(
            compare_generation(first, case, protocol) and compare_generation(repeated, case, protocol)),
            "actual_tokens": first.get("token_ids", []), "repeat_tokens": repeated.get("token_ids", [])})
    except Exception as error:
        checks.append({"case": "sequential_request_isolation", "passed": False, "error": str(error)})
    concurrency = max(1, min(int(protocol.get("concurrency", 1)), 2))
    if concurrency > 1:
        # Identical shortest inputs hide padding and cross-request state bugs.
        # Keep the probe bounded, but pair the shortest request with a distinct
        # frozen request of the most different input length when available.
        pool_cases = [*cases, *reference.get("cases", [])]
        distinct = [item for item in pool_cases if item["prompt"] != case["prompt"]]
        other = max(distinct, key=lambda item: abs(
            len(item.get("input_ids", [])) - len(case.get("input_ids", [])))) if distinct else case
        concurrent_cases = [case, other]
        def generate_case(item):
            kwargs = {"prompt": item["prompt"], "max_new_tokens": item["max_new_tokens"]}
            if item.get("fixed_length"):
                kwargs["fixed_length"] = True
            return client.generate(**kwargs)
        try:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                outputs = list(pool.map(generate_case, concurrent_cases))
            passed = all(compare_generation(output, expected, protocol)
                         for output, expected in zip(outputs, concurrent_cases))
            check = {"case": "concurrent_request_isolation", "passed": passed,
                     "input_lengths": [len(item.get("input_ids", [])) for item in concurrent_cases],
                     "observed_batch_sizes": [output.get("host_observation", {}).get("batch_size")
                                              for output in outputs],
                     "batching_modes": [output.get("host_observation", {}).get("batching_mode")
                                        for output in outputs]}
            if not passed:
                rows = []
                for output, case_reference in zip(outputs, concurrent_cases):
                    expected = case_reference.get("accepted_token_ids") or [case_reference.get("token_ids", [])]
                    tokens = output.get("token_ids", [])
                    matching = next((reference for reference in expected if tokens == reference), None)
                    reference = matching if matching is not None else expected[0]
                    mismatch = next((index for index, pair in enumerate(zip(tokens, reference))
                                     if pair[0] != pair[1]), None)
                    rows.append({"input_ids_match": output.get("input_ids") == case_reference.get("input_ids"),
                                 "output_length": len(tokens), "reference_length": len(reference),
                                 "first_mismatch_index": mismatch,
                                 "actual_prefix": tokens[:16], "reference_prefix": reference[:16]})
                check["output_differences"] = rows
            checks.append(check)
        except Exception as error:
            checks.append({"case": "concurrent_request_isolation", "passed": False, "error": str(error)})
    try:
        observation = client.profile(prompt=case["prompt"], max_new_tokens=min(case["max_new_tokens"], 2))
    except Exception as error:
        observation = None
        checks.append({"case": "runtime_effect_observation", "passed": False, "error": str(error)})
    return {"passed": bool(checks) and all(check["passed"] for check in checks),
            "checks": checks, "runtime_observation": observation,
            "scope": "shortest frozen case sequential repeat; up to two distinct input lengths concurrently; runtime-effect profile"}
