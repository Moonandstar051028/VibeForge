"""Run a user-owned quality evaluator outside candidate code with a hard deadline."""

import math
import os
import sys
import hashlib
import json
import uuid
from pathlib import Path
from vibeforge.observability.logs import run_command
from vibeforge.storage.atomic import digest, read_json, write_json


def run_quality(client, settings, folder, context, baseline=None):
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    request_path, output_path = folder / "request.json", folder / "result.json"
    write_json(request_path, {"engine_url": client.url, **context})
    substitutions = {"url": client.url, "request": str(request_path), "output": str(output_path),
                     "python": sys.executable, "project": context["project"]}
    command = [arg.format_map(substitutions) for arg in settings["command"]]
    for path, expected in settings.get("file_hashes", {}).items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen quality input was modified: {path}")
    # A failed evaluator must never inherit a stale successful result.
    if output_path.exists():
        output_path.rename(folder / f"stale-result-{os.stat(output_path).st_mtime_ns}.json")
    env = dict(os.environ, VIBEFORGE_ENGINE_URL=client.url,
               VIBEFORGE_QUALITY_REQUEST=str(request_path), VIBEFORGE_QUALITY_OUTPUT=str(output_path),
               PYTHONDONTWRITEBYTECODE="1")
    env["PYTHONPATH"] = str(Path(context["project"]) / "src") + os.pathsep + env.get("PYTHONPATH", "")
    token = uuid.uuid4().hex
    # The wrapper owns the whole evaluator group, including descendants, on restart.
    wrapper = [sys.executable, "-m", "vibeforge.evaluation.quality", "--token", token,
               "--recipe", json.dumps(command)]
    record = run_command(command, folder, folder / "evaluator.log", settings["timeout_seconds"],
                         env=env, trace=folder / "commands.jsonl") if not context.get("process_record") else run_command(
                             wrapper, folder, folder / "evaluator.log", settings["timeout_seconds"], env=env,
                             trace=folder / "commands.jsonl", process_record=context["process_record"], token=token)
    errors, metrics = [], {}
    try:
        if record["exit_code"]:
            raise ValueError(f"Quality evaluator exited with {record['exit_code']}")
        result = read_json(output_path)
        if result.get("passed") is not True:
            errors.append("evaluator_failed")
        metrics = result.get("metrics", {})
        if not isinstance(metrics, dict):
            raise ValueError("Quality result metrics must be an object")
        if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) for v in metrics.values()):
            metrics = {}
            raise ValueError("Quality metrics must be finite numbers")
        for name, minimum in settings["min_scores"].items():
            score = metrics.get(name)
            if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(score):
                errors.append(f"missing_or_nonfinite_score:{name}")
            elif score < minimum:
                errors.append(f"below_quality_floor:{name}")
        for name, maximum_drop in settings.get("max_regressions", {}).items():
            score, initial = metrics.get(name), (baseline or {}).get("metrics", {}).get(name)
            if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(score):
                errors.append(f"missing_regression_score:{name}")
            if baseline is not None:
                if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v)
                       for v in (score, initial)):
                    errors.append(f"missing_regression_score:{name}")
                elif initial - score > maximum_drop:
                    errors.append(f"quality_regression:{name}")
    except (OSError, ValueError, TypeError, AttributeError) as error:
        errors.append(str(error))
    report = {"passed": not errors, "metrics": metrics, "errors": errors,
              "evaluator_hash": digest(settings), "binding": context, "process": record}
    write_json(folder / "report.json", report)
    return report


if __name__ == "__main__":
    import argparse
    import subprocess
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    parser.add_argument("--recipe", required=True)
    args = parser.parse_args()
    raise SystemExit(subprocess.run(json.loads(args.recipe), check=False).returncode)
