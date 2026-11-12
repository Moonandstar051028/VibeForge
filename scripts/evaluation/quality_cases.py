"""Trusted, backend-independent answer checking over a user-selected task dataset."""
import argparse
import json
import math
import re
from pathlib import Path
from urllib.request import Request, urlopen


def evaluate(url, cases, timeout):
    if not isinstance(cases, list) or not cases:
        raise ValueError("Quality cases must be a nonempty list")
    results, earned, total = [], 0.0, 0.0
    for case in cases:
        weight = case.get("weight", 1)
        if not isinstance(weight, (int, float)) or isinstance(weight, bool) or not math.isfinite(weight) or weight <= 0:
            raise ValueError("Case weights must be positive finite numbers")
        total += weight
        mode = case.get("mode", "chat")
        if mode not in ("chat", "generate"):
            raise ValueError("Case mode must be chat or generate")
        matcher = case.get("match", "exact")
        if matcher not in ("exact", "contains", "regex") or not isinstance(case.get("expected"), str):
            raise ValueError("Each case needs expected text and a supported matcher")
        expected = case["expected"]
        payload = {"max_new_tokens": case.get("max_new_tokens", 64)}
        if mode == "chat":
            payload["messages"] = [{"role": "user", "content": case["prompt"]}]
        else:
            payload["prompt"] = case["prompt"]
        try:
            request = Request(url.rstrip("/") + "/" + mode, json.dumps(payload).encode(),
                              headers={"Content-Type": "application/json"})
            with urlopen(request, timeout=timeout) as response:
                output = json.load(response)
            text = output["text"].strip()
            passed = (text == expected if matcher == "exact" else expected in text if matcher == "contains"
                      else re.search(expected, text) is not None)
            results.append({"id": case["id"], "passed": passed, "text": text})
        except Exception as error:
            passed = False
            results.append({"id": case["id"], "passed": False, "error": str(error)})
        if passed:
            earned += weight
    # Wrong answers contribute to accuracy; transport/response failures invalidate
    # the run even when the remaining answers would meet the configured floor.
    return {"passed": not any("error" in result for result in results),
            "metrics": {"accuracy": earned / total}, "cases": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--request-timeout", type=float, default=120)
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text())
    cases = json.loads(Path(args.cases).read_text())
    result = evaluate(request["engine_url"], cases, args.request_timeout)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
