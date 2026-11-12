"""Capture a repeated, frozen regression reference before candidate edits."""

from vibeforge.storage.atomic import digest, read_json, write_json
import random
import math
import statistics
from vibeforge.contracts.workload import benchmark_cases as workload_cases
from vibeforge.evaluation.correctness import numeric_report


def _numeric_reference(samples, protocol):
    if not samples:
        raise ValueError("A numeric reference requires at least one sample")
    indices = samples[0].get("indices")
    values = [sample.get("values") for sample in samples]
    if (not isinstance(indices, list) or not indices or any(sample.get("indices") != indices for sample in samples)
            or any(not isinstance(row, list) or len(row) != len(indices) for row in values)
            or any(type(value) not in (int, float) or not math.isfinite(value) for row in values for value in row)):
        raise ValueError("Reference logits changed shape, indices, or contained non-finite values")
    columns = list(zip(*values))
    result = {
        "indices": indices,
        "values": [statistics.median(column) for column in columns],
        "min_values": [min(column) for column in columns],
        "max_values": [max(column) for column in columns],
        "repeat_count": len(samples),
    }
    if any(not numeric_report(sample, result, protocol)["passed"] for sample in samples):
        raise ValueError("Reference numeric instability exceeds the frozen baseline tolerance")
    return result


def capture_reference(client, cases, path, provenance, protocol=None, workload=None):
    protocol = protocol or {}
    vocab_size = provenance.get("vocab_size", 32)
    count = min(protocol.get("logit_samples", 64), vocab_size)
    indices = sorted(random.Random(protocol.get("seed", 17)).sample(range(vocab_size), count))
    repeats = protocol.get("reference_repeats", 3)
    min_consensus = protocol.get("min_reference_consensus", 2 / 3)
    if type(repeats) is not int or not 2 <= repeats <= 10:
        raise ValueError("reference_repeats must be an integer in [2, 10]")
    results = []
    benchmark_cases = workload_cases(workload) if workload else []
    if not cases:
        raise ValueError("A reference requires nonempty correctness cases")
    for case in [*cases, *benchmark_cases]:
        arguments = {"prompt": case["prompt"], "max_new_tokens": case.get("max_new_tokens", 4),
                     **({"fixed_length": True} if case.get("fixed_length") else {})}
        outputs = [client.generate(**arguments) for _ in range(repeats)]
        input_ids = outputs[0]["input_ids"]
        if (not isinstance(input_ids, list) or not input_ids
                or any(type(token) is not int or token < 0 for token in input_ids)):
            raise ValueError(f"Reference input IDs are invalid for {case['id']}")
        if any(output.get("input_ids") != input_ids for output in outputs):
            raise ValueError(f"Reference tokenization is unstable for {case['id']}")
        length = len(input_ids)
        if not case.get("min_input_tokens", 1) <= length <= case.get("max_input_tokens", float("inf")):
            raise ValueError(f"Actual tokenized input length {length} violates bounds for {case['id']}")
        variants = {}
        for output in outputs:
            tokens = output.get("token_ids")
            if (not isinstance(tokens, list) or not tokens or len(tokens) > arguments["max_new_tokens"]
                    or (case.get("fixed_length") and len(tokens) != arguments["max_new_tokens"])
                    or any(type(token) is not int or token < 0 for token in tokens)):
                raise ValueError(f"Reference generation returned invalid tokens for {case['id']}")
            variants[tuple(tokens)] = variants.get(tuple(tokens), 0) + 1
        ranked = sorted(variants.items(), key=lambda item: (-item[1], item[0]))
        canonical, canonical_count = ranked[0]
        consensus = canonical_count / repeats
        if protocol.get("correctness_mode", "strict") == "strict" and consensus < min_consensus:
            raise ValueError(
                f"Reference generation is unstable for {case['id']}: consensus {consensus:.3f} "
                f"is below {min_consensus:.3f}; use a deterministic backend or approximate mode with quality evaluation"
            )
        numeric = _numeric_reference([
            client.logits(input_ids=input_ids, indices=indices) for _ in range(repeats)
        ], protocol)
        continuation_input = input_ids + list(canonical[:2])
        continuation = _numeric_reference([
            client.logits(input_ids=continuation_input, indices=indices) for _ in range(repeats)
        ], protocol)
        results.append({"id": case["id"], "prompt": case["prompt"], "input_ids": input_ids,
                        "max_new_tokens": case.get("max_new_tokens", 4),
                        "token_ids": list(canonical),
                        "accepted_token_ids": [list(tokens) for tokens, _ in ranked],
                        "output_variants": [{"token_ids": list(tokens), "count": occurrences}
                                            for tokens, occurrences in ranked],
                        "reference_consensus": consensus,
                        "logits": numeric,
                        "continuation_logits": continuation,
                        "fixed_length": bool(case.get("fixed_length")),
                        "expected_contains": case.get("expected_contains")})
        results[-1].update(input_tokens=length, profile_id=case.get("profile_id"))
    result = {"schema_version": 3, "provenance": provenance, "cases": results,
              "benchmark_cases": [case for case in results if case["fixed_length"]], "vocab_size": vocab_size,
              "reference_repeats": repeats, "min_reference_consensus": min_consensus,
              "reference_role": "regression_baseline_not_ground_truth"}
    result["hash"] = digest(result)
    validate_reference(result, protocol, cases, workload)
    write_json(path, result)
    return result


def validate_reference(value, protocol=None, cases=None, workload=None):
    """Validate evidence structure and coverage, independently of its self-hash."""
    protocol = protocol or {}
    if value.get("schema_version") != 3:
        raise ValueError("New campaigns require a schema v3 repeated reference")
    repeats = value.get("reference_repeats")
    if type(repeats) is not int or not 2 <= repeats <= 10 or repeats < protocol.get("reference_repeats", 2):
        raise ValueError("Reference has insufficient repeated measurements")
    rows = value.get("cases")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Reference cases must be nonempty")
    identifiers = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or row["id"] in identifiers:
            raise ValueError("Reference case IDs must be unique strings")
        identifiers.add(row["id"])
        if (not isinstance(row.get("prompt"), str) or type(row.get("max_new_tokens")) is not int
                or row["max_new_tokens"] <= 0 or type(row.get("fixed_length")) is not bool):
            raise ValueError("Invalid reference generation contract")
        inputs = row.get("input_ids")
        if not isinstance(inputs, list) or not inputs or any(type(t) is not int or t < 0 for t in inputs):
            raise ValueError("Invalid reference input IDs")
        variants = row.get("output_variants")
        if not isinstance(variants, list) or not variants:
            raise ValueError("Reference must record its output variants")
        sequences, counts = [], []
        for variant in variants:
            tokens, count = variant.get("token_ids"), variant.get("count")
            if (not isinstance(tokens, list) or not tokens or tokens in sequences
                    or any(type(t) is not int or t < 0 for t in tokens)
                    or len(tokens) > row["max_new_tokens"]
                    or (row["fixed_length"] and len(tokens) != row["max_new_tokens"])
                    or type(count) is not int or count <= 0):
                raise ValueError("Invalid reference output variant")
            sequences.append(tokens)
            counts.append(count)
        consensus = max(counts) / repeats
        if (sum(counts) != repeats or row.get("accepted_token_ids") != sequences
                or row.get("token_ids") not in sequences
                or counts[sequences.index(row["token_ids"])] != max(counts)
                or row.get("reference_consensus") != consensus):
            raise ValueError("Reference output counts or consensus are inconsistent")
        if protocol.get("correctness_mode", "strict") == "strict" and consensus < protocol.get("min_reference_consensus", 2 / 3):
            raise ValueError("Reference generation is unstable under the frozen protocol")
        for name in ("logits", "continuation_logits"):
            numeric = row.get(name, {})
            indices, values = numeric.get("indices"), numeric.get("values")
            if (not isinstance(indices, list) or not indices or len(set(indices)) != len(indices)
                    or any(type(index) is not int or not 0 <= index < value["vocab_size"] for index in indices)
                    or not isinstance(values, list) or len(indices) != len(values)
                    or numeric.get("repeat_count") != repeats
                    or "min_values" not in numeric or "max_values" not in numeric):
                raise ValueError("Invalid repeated reference logits")
            for bound in ("min_values", "max_values"):
                report = numeric_report({"indices": indices, "values": numeric[bound]}, numeric, protocol)
                if not report["passed"]:
                    raise ValueError("Invalid or unstable reference logits: " + report.get("error", "numeric_tolerance"))
    # Benchmark rows must be the same records evaluated by the correctness gate.
    if value.get("benchmark_cases") != [row for row in rows if row["fixed_length"]]:
        raise ValueError("Reference benchmark rows do not match correctness cases")
    if cases is not None:
        required = [*cases, *(workload_cases(workload) if workload else [])]
        for required_case in required:
            match = next((row for row in rows if row["id"] == required_case["id"]), None)
            if (match is None or match["prompt"] != required_case["prompt"]
                    or match["max_new_tokens"] != required_case.get("max_new_tokens", 4)
                    or match["fixed_length"] != bool(required_case.get("fixed_length"))
                    or match.get("expected_contains") != required_case.get("expected_contains")
                    or match.get("profile_id") != required_case.get("profile_id")
                    or not required_case.get("min_input_tokens", 1) <= len(match["input_ids"]) <= required_case.get("max_input_tokens", math.inf)):
                raise ValueError("Reference does not cover configured case: " + required_case["id"])


def load_reference(path):
    value = read_json(path)
    if value["hash"] != digest({k: v for k, v in value.items() if k != "hash"}):
        raise ValueError("Frozen reference was modified")
    if value.get("schema_version") not in (2, 3):
        raise ValueError("Unsupported correctness reference schema")
    return value
