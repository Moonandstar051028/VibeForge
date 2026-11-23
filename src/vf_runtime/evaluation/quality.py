"""Run a user-owned quality evaluator outside candidate code with a hard deadline."""

import math
import os
import sys
import hashlib
import json
import uuid
from pathlib import Path
from vf_runtime.observability.logs import run_command
from vf_runtime.storage.atomic import digest, read_json, write_json


def run_quality(client, settings, folder, context, baseline=None):
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    repeats = settings.get("repeats", 1)
    if repeats > 1:
        single = {**settings, "repeats": 1, '_pair_single': False}
        runs, early_stop = [], None
        for index in range(1, repeats + 1):
            runs.append(run_quality(client, single, folder / "runs" / f"{index:04d}", context, baseline))
            if settings.get('paired') and baseline is not None and index < repeats:
                from vibeforge_v2.quality import paired_quality
                try:
                    partial = paired_quality(baseline, {'runs': runs}, settings['paired'])
                except (ValueError, KeyError, TypeError):
                    partial = None
                if partial is not None and not partial['passed']:
                    # Candidate task pass is AND over repeats, so additional
                    # repeats cannot remove an observed loss or lower its bound.
                    early_stop = {'reason': 'paired_failure_cannot_recover_with_more_repeats',
                                  'completed_repeats': len(runs), 'planned_repeats': repeats,
                                  'partial_paired': partial}
                    break
        names = set().union(*(run.get("metrics", {}) for run in runs))
        metrics = {name: min(run["metrics"][name] for run in runs)
                   for name in names if all(name in run.get("metrics", {}) for run in runs)}
        statistics = {name: {"minimum": min(run["metrics"][name] for run in runs),
                             "maximum": max(run["metrics"][name] for run in runs),
                             "mean": sum(run["metrics"][name] for run in runs) / len(runs)}
                      for name in metrics}
        errors = [f"run_{index}:{error}" for index, run in enumerate(runs, 1)
                  for error in run.get("errors", [])]
        report = {"passed": all(run["passed"] for run in runs) and not errors,
                  "metrics": metrics, "metric_statistics": statistics, "errors": errors,
                  "repeat_count": len(runs), "planned_repeat_count": repeats, "early_stop": early_stop,
                  "runs": runs, "evaluator_hash": digest(settings),
                  "binding": context}
        if settings.get('paired') and baseline is not None:
            from vibeforge_v2.quality import paired_quality
            try:
                paired = paired_quality(baseline, report, settings['paired'])
                report['paired'] = paired
                if not paired['passed']:
                    report['passed'] = False
                    report['errors'].append('paired_quality_not_established')
            except (ValueError, KeyError, TypeError) as error:
                report['passed'] = False
                report['errors'].append(str(error))
        write_json(folder / "report.json", report)
        return report
    request_path, output_path = folder / "request.json", folder / "result.json"
    write_json(request_path, {"engine_url": client.url, **context})
    substitutions = {"url": client.url, "request": str(request_path), "output": str(output_path),
                     "python": context.get("python", sys.executable), "project": context["project"]}
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
    wrapper = [sys.executable, "-m", "vf_runtime.evaluation.quality", "--token", token,
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
            metrics = {}
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
        # A category with n items can only move in steps of 1/n. A 5% limit on a
        # 12-item category therefore rejects any single flipped item, which is
        # stricter than the overall paired loss limit it is meant to refine.
        # Allow at least `category_item_tolerance` items per category; the
        # overall paired limit still bounds the total number of lost items.
        sizes = {}
        for row in result.get("cases", []) if isinstance(result.get("cases"), list) else []:
            category = row.get("category", "core")
            sizes[category] = sizes.get(category, 0) + 1
        item_tolerance = settings.get("category_item_tolerance", 1)
        regression_limits = {}
        for name, maximum_drop in settings.get("max_regressions", {}).items():
            size = len(result.get("cases", [])) if name == "accuracy" else sizes.get(name.removeprefix("accuracy_"))
            allowed = max(maximum_drop, item_tolerance / size) if size and item_tolerance else maximum_drop
            regression_limits[name] = {"configured": maximum_drop, "items": size, "allowed_drop": allowed}
            score, initial = metrics.get(name), (baseline or {}).get("metrics", {}).get(name)
            if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(score):
                errors.append(f"missing_regression_score:{name}")
            if baseline is not None:
                if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v)
                       for v in (score, initial)):
                    errors.append(f"missing_regression_score:{name}")
                elif initial - score > allowed + 1e-9:
                    errors.append(f"quality_regression:{name}")
    except (OSError, ValueError, TypeError, AttributeError) as error:
        errors.append(str(error))
    report = {"passed": not errors, "metrics": metrics, "errors": errors, "repeat_count": 1,
              "regression_limits": locals().get("regression_limits", {}),
              'cases': result.get('cases', []) if 'result' in locals() else [],
              "evaluator_hash": digest(settings), "binding": context, "process": record}
    if settings.get('paired') and baseline is not None and settings.get('_pair_single', True):
        # Aggregate repeats are paired above; a standalone single run is paired here.
        from vibeforge_v2.quality import paired_quality
        try:
            report['paired'] = paired_quality(baseline, report, settings['paired'])
            if not report['paired']['passed']:
                report['passed'] = False
                report['errors'].append('paired_quality_not_established')
        except (ValueError, KeyError, TypeError) as error:
            report['passed'] = False
            report['errors'].append(str(error))
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
