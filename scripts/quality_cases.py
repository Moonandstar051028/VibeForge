"""Trusted, backend-independent answer checking over a user-selected task dataset."""
import argparse
import json
import math
import re
import ast
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen


def execute_function(text, case):
    blocks = re.findall(r'```(?:python|py)?\s*\n(.*?)```', text, re.S)
    for code in blocks or [text]:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        if any(isinstance(node, ast.FunctionDef) and node.name == case['entry_point'] for node in tree.body):
            break
    else:
        return False
    # Fixed, trusted assertions; generated implementations run in a disposable process.
    program = ('import resource\nresource.setrlimit(resource.RLIMIT_CPU, (2, 2))\n'
               'resource.setrlimit(resource.RLIMIT_AS, (536870912, 536870912))\n' + code + '\n' + case['tests'])
    with tempfile.TemporaryDirectory(prefix='vf-quality-code-') as folder:
        path = Path(folder) / 'answer.py'
        path.write_text(program)
        try:
            result = subprocess.run([sys.executable, '-I', str(path)], cwd=folder,
                                    env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''},
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=4)
        except subprocess.TimeoutExpired:
            return False
    return result.returncode == 0


def tool_task(url, case, timeout):
    def send(messages):
        payload = {'messages': messages, 'tools': case['tools'],
                   'max_tokens': 96, 'temperature': 0}
        request = Request(url.rstrip('/') + '/v1/chat/completions', json.dumps(payload).encode(),
                          headers={'Content-Type': 'application/json'})
        with urlopen(request, timeout=timeout) as response:
            return json.load(response)['choices'][0]['message']
    messages = [{'role': 'user', 'content': case['prompt']}]
    first = send(messages)
    calls = first.get('tool_calls', [])
    if len(calls) != 1:
        return False, first
    call = calls[0]
    try:
        arguments = json.loads(call['function']['arguments'])
    except (ValueError, KeyError, TypeError):
        return False, first
    # Exact schema/types, then execute the trusted local tool and return feedback.
    passed = (call['function']['name'] == 'add' and type(arguments) is dict and
              arguments == case['arguments'] and all(type(x) is int for x in arguments.values()))
    if not passed:
        return False, first
    messages += [first, {'role': 'tool', 'tool_call_id': call['id'],
                         'content': str(arguments['a'] + arguments['b'])}]
    final = send(messages)
    return final.get('content', '').strip() == case['expected'], {'first': first, 'final': final}


def evaluate(url, cases, timeout):
    if not isinstance(cases, list) or not cases:
        raise ValueError("Quality cases must be a nonempty list")
    results, earned, total = [], 0.0, 0.0
    for case_number, case in enumerate(cases, 1):
        print(f'quality {case_number}/{len(cases)}: {case.get("id")}', flush=True)
        weight = case.get("weight", 1)
        if not isinstance(weight, (int, float)) or isinstance(weight, bool) or not math.isfinite(weight) or weight <= 0:
            raise ValueError("Case weights must be positive finite numbers")
        total += weight
        mode = case.get("mode", "chat")
        if mode not in ("chat", "generate", "tool"):
            raise ValueError("Case mode must be chat, generate or tool")
        matcher = case.get("match", "exact")
        if matcher not in ("exact", "contains", "regex", "python") or not isinstance(case.get("expected"), str):
            raise ValueError("Each case needs expected text and a supported matcher")
        expected = case["expected"]
        payload = {"max_new_tokens": case.get("max_new_tokens", 64)}
        if mode == "chat":
            payload["messages"] = [{"role": "user", "content": case["prompt"]}]
        else:
            payload["prompt"] = case["prompt"]
        try:
            if mode == 'tool':
                passed, transcript = tool_task(url, case, timeout)
                results.append({'id': case['id'], 'category': case.get('category', 'core'),
                                'passed': passed, 'transcript': transcript})
                if passed:
                    earned += weight
                continue
            request = Request(url.rstrip("/") + "/" + mode, json.dumps(payload).encode(),
                              headers={"Content-Type": "application/json"})
            with urlopen(request, timeout=timeout) as response:
                output = json.load(response)
            text = output["text"].strip()
            passed = (execute_function(text, case) if matcher == 'python' else
                      text == expected if matcher == "exact" else expected in text if matcher == "contains"
                      else re.search(expected, text) is not None)
            results.append({"id": case["id"], "category": case.get('category', 'core'), "passed": passed, "text": text,
                            "output_tokens": len(output.get('token_ids', [])),
                            "finish_reason": output.get('finish_reason'),
                            "reached_output_limit": len(output.get('token_ids', [])) >= payload['max_new_tokens'],
                            "failure_kind": None if passed else ('execution_or_assertion' if matcher == 'python'
                                               else 'exact_format_or_answer' if matcher == 'exact' else 'answer_mismatch')})
        except Exception as error:
            passed = False
            results.append({"id": case["id"], "category": case.get('category', 'core'), "passed": False, "error": str(error)})
        if passed:
            earned += weight
    # Wrong answers contribute to accuracy; transport/response failures invalidate
    # the run even when the remaining answers would meet the configured floor.
    metrics = {'accuracy': earned / total}
    for category in {row['category'] for row in results}:
        subset = [row for row in results if row['category'] == category]
        metrics['accuracy_' + category] = sum(row['passed'] for row in subset) / len(subset)
    return {"passed": not any("error" in result for result in results),
            "metrics": metrics, "cases": results}


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
