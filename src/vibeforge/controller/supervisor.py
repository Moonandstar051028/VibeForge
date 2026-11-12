"""Executable single-device optimization loop with durable, evidence-based decisions."""

import os
import random
import sys
import time
import uuid
import json
import hashlib
from copy import deepcopy
from pathlib import Path
from vibeforge.agent.analysis import analysis_attribution_error, write_analysis, performance_analysis_error
from vibeforge.agent.backend_policy import backend_only_change, runtime_configuration_only_change
from vibeforge.agent.context import create_context
from vibeforge.agent.candidate import candidate_gate, runtime_effect_fingerprint
from vibeforge.agent.memory import refresh_memory, reflection_error
from vibeforge.contracts.precision import precision_fields, validate_candidate_runtime
from vibeforge.controller.admission import decide
from vibeforge.controller.budgets import stop_reason
from vibeforge.controller.recovery import recover
from vibeforge.controller.state_machine import transition
from vibeforge.engine.builder import build
from vibeforge.engine.process import EngineProcess
from vibeforge.evaluation.correctness import evaluate_candidate_precheck, evaluate_correctness
from vibeforge.evaluation.performance import diagnose_concurrency, measure_suite, measurement_mode
from vibeforge.contracts.workload import workload_profiles, profiling_cases
from vibeforge.evaluation.statistics import paired_gain
from vibeforge.evaluation.reference import capture_reference, load_reference, validate_reference
from vibeforge.evaluation.quality import run_quality
from vibeforge.knowledge.extractor import extract_draft
from vibeforge.knowledge.registry import load_rules, materialize_evidence
from vibeforge.knowledge.retriever import retrieve
from vibeforge.observability.events import EventLog
from vibeforge.observability.logs import run_command
from vibeforge.observability.reports import write_campaign_report
from vibeforge.profiling.collector import collect
from vibeforge.storage.atomic import atomic_write, read_json, write_json, digest
from vibeforge.storage.database import Database
from vibeforge.storage.locks import exclusive_lock
from vibeforge.storage.paths import identifier
from vibeforge.versioning.artifacts import inventory, verify_release, source_hash
from vibeforge.versioning.candidate import create_workspace, write_diff
from vibeforge.versioning.promotion import begin_promotion, commit_promotion
from vibeforge.versioning.release import freeze
from vibeforge.versioning.rollback import cancel_promotion, rollback_completed


REJECTED_PATCH_REASONS = {
    "correctness_failed", "quality_failed", "performance_regression", "no_significant_gain",
}


def probe_prediction(improvement, expected):
    """Classify diagnostic direction without making an admission decision."""
    if not isinstance(improvement, (int, float)):
        return None, "inconclusive", "inconclusive"
    if improvement <= 0:
        return False, "refuted", "revise"
    if not isinstance(expected, (int, float)):
        return None, "inconclusive", "promising"
    observed = improvement >= expected
    return observed, "supported" if observed else "directionally_positive", "promising"


def remaining_seconds(deadline, phase):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError(f"{phase} deadline exceeded")
    return remaining


class Supervisor:
    def __init__(self, paths, campaign_id):
        self.paths, self.campaign_id = paths, identifier(campaign_id)
        self.database = Database(paths.data / "state" / "controller.sqlite3")
        self.directory = paths.campaign(campaign_id)
        self.events = EventLog(self.database, campaign_id, self.directory / "events")
        self.engine = None
        self.client = None

    def close(self):
        if self.engine:
            self.engine.stop()
        self.database.close()

    def create(self, config):
        if any(row["campaign_id"] == self.campaign_id for row in self.database.list_campaigns()):
            raise ValueError(f"Campaign already exists: {self.campaign_id}")
        self.directory.mkdir(parents=True, exist_ok=True)
        write_json(self.directory / "manifest.json", config)
        for section in ("model", "hardware", "workload", "evaluation", "agent", "precision", "engine",
                        "requirements", "optimization", "telemetry", "profiling", "probe", "refinement"):
            write_json(self.directory / "configs" / f"{section}.json", config[section])
        atomic_write(self.directory / "configs" / "agent-prompt.md",
                     (self.paths.project / config["agent"]["prompt"]).read_text())
        write_json(self.directory / "configs" / "cases.json",
                   read_json(self.paths.project / config["workload"]["cases"]))
        state = {"campaign_id": self.campaign_id, "state": "RECOVERING",
                 "stable_ref": None, "previous_stable_ref": None, "active_ref": None,
                 "completed_iterations": 0, "consecutive_failures": 0, "started_at": time.time(),
                 "consecutive_candidate_failures": 0,
                 "proposal_attempts": 0, "valid_candidates": 0, "completed_experiments": 0, "stalled_attempts": 0,
                 "profile_requests": 0, "probe_attempts": 0, "refinement_attempts": 0,
                 "mode": "toy_fixture" if config["model"].get("fixture") else "local_model",
                 "config_hash": config["config_hash"], "analysis_pending": False}
        with self.database.transaction():
            self.database.put(self.campaign_id, state)
            self.database.emit(self.campaign_id, "campaign_created", payload={"config_hash": config["config_hash"]})
        self.events.sync()
        return state

    def state(self, target, **changes):
        value = transition(self.database, self.campaign_id, target, **changes)
        self.events.sync()
        return value

    def start_engine(self, version, log, timeout_seconds=None):
        manifest = read_json(self.paths.release(version) / "manifest.json")
        verify_release(self.paths.release(version), manifest)
        if self.engine:
            self.engine.stop()
        startup_timeout = self.config.get("startup_timeout", 300)
        request_timeout = self.config.get("request_timeout", 120)
        deadline = None
        if timeout_seconds is not None:
            deadline = time.monotonic() + timeout_seconds
            startup_timeout = min(startup_timeout, timeout_seconds)
            request_timeout = min(request_timeout, timeout_seconds)
        self.engine = EngineProcess(self.paths, self.campaign_id, manifest["runtime_python"],
                                    startup_timeout, request_timeout,
                                    self.config.get("inference_pythonpath", ()))
        started = time.monotonic()
        try:
            client = self.engine.start(self.paths.release(version), log)
            health_client = client.bounded(deadline, "engine startup") if deadline else client
            write_json(Path(log).with_suffix(".runtime.json"), health_client.health())
        except Exception:
            self.engine.stop()
            self.client = None
            raise
        self.client = client
        self.database.update(self.campaign_id, {"active_ref": version}, "engine_started",
                             payload={"version": version, "startup_seconds": time.monotonic() - started})
        self.events.sync()
        return client

    def stop_engine(self):
        if self.engine:
            self.engine.stop()
        self.client = None
        self.database.update(self.campaign_id, {"active_ref": None}, "engine_stopped")
        self.events.sync()

    def worker_environment(self):
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join([str(self.paths.project / "src"),
                                                     *self.config.get("inference_pythonpath", ()),
                                                     environment.get("PYTHONPATH", "")])
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["QWEN_AGENT_MAX_LLM_CALL_PER_RUN"] = str(self.config["agent"].get("max_turns", 6))
        return environment

    def smoke(self, client, folder, stem="agent-smoke"):
        """Smoke tests are owned workers too: a stalled LLM call cannot hang recovery."""
        folder.mkdir(parents=True, exist_ok=True)
        context, result = folder / f"{stem}-context.json", folder / f"{stem}.json"
        write_json(context, {"agent": self.config["agent"], "model": self.config["model"]})
        token = uuid.uuid4().hex
        command = [sys.executable, "-u", "-m", "vibeforge.cli.main", "--project", str(self.paths.project),
                   "agent-worker", "--smoke", "--context", str(context), "--workspace", str(folder),
                   "--trace", str(folder / f"{stem}.jsonl"), "--result", str(result), "--url", client.url,
                   "--prompt", str(self.directory / "configs/agent-prompt.md"), "--token", token]
        record = run_command(command, folder, folder / f"{stem}.log",
                             timeout=min(self.config["budget"]["agent_seconds"], self.config["agent"].get("timeout_seconds", 180)),
                             env=self.worker_environment(), process_record=self.paths.data / "state" / f"{self.campaign_id}.agent.json",
                             token=token, trace=folder / "commands.jsonl")
        if record["exit_code"] or not read_json(result).get("passed"):
            raise RuntimeError(f"Agent readiness failed; see {folder / (stem + '.log')}")
        return read_json(result)

    def build_workspace(self, workspace, folder):
        engine = self.config["engine"]
        native = [name for name in inventory(workspace)
                  if Path(name).suffix in {".cu", ".cpp", ".c", ".h", ".hpp"}]
        if native and "native_kernel" not in self.config["optimization"]["allowed_surfaces"]:
            raise ValueError("Candidate native sources are outside the frozen optimization surfaces")
        if native and not engine["build_command"]:
            raise ValueError("Candidate native sources require the frozen build command")
        cache = self.paths.data / "cache" / "build"
        cache.mkdir(parents=True, exist_ok=True)
        replacements = {"workspace": str(workspace), "cache": str(cache), "python": self.config["runtime_python"]}
        recipe = [arg.format_map(replacements) for arg in engine["build_command"]]
        if recipe:
            token = uuid.uuid4().hex
            command = [self.config["runtime_python"], "-u", "-m", "vibeforge.cli.main", "--project", str(self.paths.project),
                       "build-worker", "--directory", str(workspace), "--cache", str(cache),
                       "--recipe", json.dumps(recipe), "--token", token, "--device", self.config["model"]["device"]]
            env = self.worker_environment()
            env.update(TMPDIR=str(cache), XDG_CACHE_HOME=str(cache), CUDA_CACHE_PATH=str(cache / "cuda"),
                       TRITON_CACHE_DIR=str(cache / "triton"), TORCH_EXTENSIONS_DIR=str(cache / "extensions"), HF_HUB_OFFLINE="1")
            record = run_command(command, workspace, folder / "logs/build.log", engine["build_timeout_seconds"],
                                 env=env, trace=folder / "commands.jsonl",
                                 process_record=self.paths.data / "state" / f"{self.campaign_id}.build.json", token=token)
            if record["exit_code"]:
                raise RuntimeError(f"Engine build failed; see {folder / 'logs/build.log'}")
        report = build(workspace)
        write_json(folder / "build.json", report)
        return report

    def quality(self, client, version, folder, baseline=None):
        settings = self.config["evaluation"].get("quality")
        if not settings:
            return {"passed": True, "enabled": False}
        manifest = read_json(self.paths.release(version) / "manifest.json")
        return run_quality(client, settings, folder, {
            "project": str(self.paths.project), "campaign_id": self.campaign_id, "version": version,
            "source_hash": manifest["source_hash"], "config_hash": self.config["config_hash"],
            "profile_hash": self.database.get(self.campaign_id).get("version_profiles", {}).get(version),
            "precision": read_json(self.paths.release(version) / "engine.json")["precision"],
            "process_record": str(self.paths.data / "state" / f"{self.campaign_id}.quality.json"),
        }, baseline=baseline)

    def agent_session(self, client, workspace, iteration, analysis_only=False, *,
                      session_folder=None, context_additions=None, phase="agent", timeout_seconds=None):
        folder = Path(session_folder) if session_folder else self.directory / "iterations" / f"{iteration:04d}"
        if analysis_only and session_folder is None:
            folder = self.directory / "final-analysis"
        folder.mkdir(parents=True, exist_ok=True)
        session_deadline = (time.monotonic() + timeout_seconds
                            if timeout_seconds is not None else None)
        if session_deadline is not None:
            client = client.bounded(session_deadline, f"{phase} session")
        telemetry_session_id = None
        telemetry = self.config["telemetry"]
        if telemetry["enabled"] and telemetry["capture_agent_requests"]:
            telemetry_session_id = uuid.uuid4().hex
            client.telemetry_start(
                session_id=telemetry_session_id,
                max_requests=telemetry["max_session_requests"],
                retention_seconds=telemetry["retention_seconds"],
            )
        context = create_context(self.database, self.paths, self.campaign_id, self.config, workspace, iteration, analysis_only)
        context["engine_url"] = client.url
        if timeout_seconds is not None:
            context["agent"] = deepcopy(context["agent"])
            context["agent"]["timeout_seconds"] = min(
                context["agent"].get("timeout_seconds", timeout_seconds), timeout_seconds)
        if analysis_only and context.get("previous") and session_folder is None:
            folder = self.directory / "final-analysis" / f"{context['previous']['iteration']:04d}"
            folder.mkdir(parents=True, exist_ok=True)
        if context_additions:
            context.update(deepcopy(context_additions))
        if telemetry_session_id:
            context["telemetry_session"] = {
                "session_id": telemetry_session_id,
                "evidence_path": str((folder / "agent-telemetry.json").relative_to(self.directory)),
                "admission": False,
            }
        rules = retrieve(([] if context["knowledge_scope"] == "current_campaign"
                          else load_rules(self.paths.project / "knowledge")),
                         {"profile_hash": context["profile_hash"], "precision": context["precision"],
                          "device": self.config["model"]["device"],
                          "model_type": context.get("model_features", {}).get("model_type"),
                          "transformers_version": context.get("environment", {}).get("inference", {}).get("packages", {}).get("transformers"),
                          "objective": self.config["requirements"]["objective"]})
        context["knowledge"] = materialize_evidence(
            rules, self.paths.project, self.directory)
        context_path, result_path = folder / "context.json", folder / "agent-result.json"
        write_json(context_path, context)
        token = uuid.uuid4().hex
        command = [sys.executable, "-u", "-m", "vibeforge.cli.main", "--project", str(self.paths.project),
                   "agent-worker", "--context", str(context_path), "--workspace", str(workspace),
                   "--trace", str(folder / "commands.jsonl"), "--result", str(result_path),
                   "--url", client.url, "--prompt", str(self.directory / "configs" / "agent-prompt.md"),
                   "--token", token]
        if analysis_only:
            command.append("--analysis-only")
        self.events.emit("agent_started", iteration_id=iteration,
                         payload={"analysis_only": analysis_only, "phase": phase})
        try:
            session_timeout = min(
                self.config["budget"]["agent_seconds"],
                self.config["agent"].get("timeout_seconds", self.config["budget"]["agent_seconds"]),
                (remaining_seconds(session_deadline, f"{phase} session")
                 if session_deadline is not None else float("inf")),
            )
            record = run_command(command, workspace, folder / "logs" / "agent.log",
                                 timeout=session_timeout,
                                 env=self.worker_environment(),
                                 trace=folder / "commands.jsonl",
                                 process_record=self.paths.data / "state" / f"{self.campaign_id}.agent.json", token=token,
                                 heartbeat=lambda: self.events.emit("agent_heartbeat", iteration_id=iteration))
        finally:
            if telemetry_session_id:
                try:
                    telemetry_report = client.telemetry_finish(telemetry_session_id)
                    write_json(folder / "agent-telemetry.json", telemetry_report)
                    self.events.emit("agent_telemetry_recorded", iteration_id=iteration, payload={
                        "path": str(folder / "agent-telemetry.json"),
                        "request_count": telemetry_report["request_count"],
                    })
                except Exception as error:
                    write_json(folder / "agent-telemetry-error.json", {
                        "session_id": telemetry_session_id, "error": str(error)})
        if record["exit_code"]:
            raise RuntimeError(f"Agent failed with exit code {record['exit_code']}; see {folder / 'logs/agent.log'}")
        result = read_json(result_path)
        if analysis_only and session_folder is None:
            write_json(self.directory / "final-analysis" / "agent-result.json", result)
        self.events.emit("agent_completed", iteration_id=iteration,
                         payload={"session_id": result["session_id"], "mode": result["mode"],
                                  "phase": phase})
        previous = context.get("previous")
        if previous and previous.get("analysis_pending") and result.get("previous_analysis"):
            attribution_error = analysis_attribution_error(previous, result["previous_analysis"])
            if attribution_error:
                raise RuntimeError(attribution_error)
            measurement_error = performance_analysis_error(previous, result["previous_analysis"], result.get("reflection"))
            if measurement_error:
                raise RuntimeError(measurement_error)
            error = reflection_error(previous, result.get("reflection"), str(self.directory))
            if error:
                raise RuntimeError(error)
            previous["analysis_pending"] = False
            previous["interpretation"] = result["previous_analysis"]
            previous["reflection"] = result.get("reflection")
            self.database.iteration(self.campaign_id, previous["iteration"], previous)
            write_analysis(self.directory / "iterations" / f"{previous['iteration']:04d}" / "analysis.md",
                           previous, result["previous_analysis"])
            pending = any(row.get("analysis_pending") for row in self.database.iterations(self.campaign_id))
            self.database.update(self.campaign_id, {"analysis_pending": pending}, "analysis_completed",
                                 iteration_id=previous["iteration"])
            self.events.sync()
            refresh_memory(self.database, self.campaign_id, self.directory)
        return result

    def observe_stable(self, client, stable):
        path = self.directory / "observations" / stable / "baseline.json"
        profile_hash = self.database.get(self.campaign_id).get("version_profiles", {}).get(stable)
        settings = self.config["profiling"]
        if not settings["enabled"]:
            record = {
                "schema_version": 4, "kind": "profiling_disabled", "enabled": False,
                "stable_ref": stable, "profile_hash": profile_hash,
                "source_hash": source_hash(self.paths.release(stable)),
                "workload": self.config["workload"], "observed_at": time.time(),
                "measurements": {}, "runtime": {},
                "limitations": [
                    "Diagnostic profiling is disabled by the frozen campaign configuration.",
                    "Evidence-bound candidate planning is unavailable until profiling is enabled in a new campaign.",
                ],
            }
            write_json(path, record)
            self.events.emit("profiling_skipped", payload={"stable_ref": stable, "path": str(path)})
            return record
        if path.is_file():
            record = read_json(path)
            if (0 <= time.time() - record.get("observed_at", 0) <= self.config["profiling"]["max_age_seconds"]
                    and record.get("schema_version", 1) >= 3 and record.get("profile_matrix", {}).get("schema_version") == 2
                    and record.get("profile_hash") == profile_hash and record.get("workload") == self.config["workload"]
                    and record.get("source_hash") == source_hash(self.paths.release(stable))):
                return record
        deadline = time.monotonic() + settings["timeout_seconds"]
        bounded_client = client.bounded(deadline, "stable profiling")
        cases = profiling_cases(
            self.config["workload"], settings["coverage"], settings["max_matrix_profiles"])
        if not cases:
            raise RuntimeError("The profiling matrix selected no frozen workload cases")
        expected_source = source_hash(self.paths.release(stable))
        records, entries = [], []
        for case in cases:
            profile, prompt_index = case["profile"], case["prompt_index"]
            remaining_seconds(deadline, "stable profiling")
            record = bounded_client.profile(
                prompt=case["prompt"],
                max_new_tokens=min(profile["max_new_tokens"], settings["max_new_tokens"], 128),
            )
            record.update(
                stable_ref=stable, profile_hash=profile_hash, workload=self.config["workload"],
                prompt_index=prompt_index, profile_id=profile["id"], observed_at=time.time(),
            )
            if record.get("source_hash") != expected_source:
                raise RuntimeError("Diagnostic profile source identity does not match the stable release")
            relative = f"observations/{stable}/matrix/{profile['id']}-{prompt_index}.json"
            write_json(self.directory / relative, record)
            records.append(record)
            normal = record.get("normal_request") or {}
            measurements = record.get("measurements") or {}
            entries.append({
                "profile_id": profile["id"], "prompt_index": prompt_index,
                "prompt_characters": len(case["prompt"]),
                "profile_max_new_tokens": record.get("profile_max_new_tokens"),
                "evidence_path": relative,
                "normal_request": {key: normal.get(key) for key in (
                    "backend_wall_seconds", "queue_seconds", "input_tokens", "output_tokens",
                    "output_tokens_per_second", "ttft_seconds", "tpot_seconds",
                )},
                "measurements": {key: value for key, value in measurements.items()
                                 if key.startswith("normal.")},
                "hotspots": {
                    "cpu": [row.get("name") for row in (record.get("top_cpu_ops") or [])[:3]],
                    "device": [row.get("name") for row in (record.get("top_device_ops") or [])[:3]],
                    "device_operators": [row.get("name") for row in
                                         (record.get("top_device_operator_ops") or [])[:3]],
                },
            })
        selected_id = self.config["workload"]["observation_profile"]
        primary = next((record for record in records
                        if record["profile_id"] == selected_id and record["prompt_index"] == 0), records[0])
        concurrent_profiles = [profile for profile in workload_profiles(self.config["workload"])
                               if profile.get("concurrency", 1) > 1]
        concurrency_diagnostics = {
            profile["id"]: diagnose_concurrency(bounded_client, profile) for profile in concurrent_profiles
        }
        primary["concurrency_diagnostics"] = concurrency_diagnostics
        for profile_id, diagnostic in concurrency_diagnostics.items():
            primary.setdefault("measurements", {})[
                f"diagnostic.concurrent.{profile_id}.throughput"] = {
                    "value": diagnostic["throughput"], "unit": "output_tokens_per_second",
                    "scope": "small frozen concurrent diagnostic; excluded from admission",
                }
        primary["profile_matrix"] = {
            "schema_version": 2, "coverage": settings["coverage"],
            "selected_cases": len(entries), "available_profiles": len(workload_profiles(self.config["workload"])),
            "entries": entries,
            "limitations": [
                "Each profile entry is an isolated request; concurrency_diagnostics contains the separate concurrent load.",
                "Profiler overhead excludes these timings from formal performance decisions.",
            ],
        }
        write_json(path, primary)
        self.events.emit("profile_observed", payload={
            "stable_ref": stable, "path": str(path), "matrix_cases": len(entries),
            "coverage": settings["coverage"],
        })
        return primary

    def complete_pending_analyses(self):
        """A planning barrier: no new candidate while a recorded result is unanalyzed."""
        state = self.database.get(self.campaign_id)
        pending = [row for row in self.database.iterations(self.campaign_id) if row.get("analysis_pending")]
        if not pending or state.get("stop_requested"):
            return
        version = state["stable_ref"]
        client = self.client
        if client is None or state.get("active_ref") != version:
            client = self.start_engine(version, self.directory / "final-analysis" / "engine.log")
        for _ in pending:
            if self.database.get(self.campaign_id).get("stop_requested"):
                return
            unfinished = [row for row in self.database.iterations(self.campaign_id) if row.get("analysis_pending")]
            before, target = len(unfinished), unfinished[0]["iteration"]
            attempts = self.config["budget"].get("max_analysis_attempts", 2)
            for attempt in range(attempts):
                if self.database.get(self.campaign_id).get("stop_requested"):
                    return
                try:
                    retry_folder = (self.directory / "final-analysis" / f"{target:04d}" / f"retry-{attempt}"
                                    if attempt else None)
                    self.agent_session(client, self.paths.release(version), state.get("proposal_attempts", state["completed_iterations"]) + 1,
                                       analysis_only=True, session_folder=retry_folder)
                    after = sum(bool(row.get("analysis_pending")) for row in self.database.iterations(self.campaign_id))
                    if after >= before:
                        raise RuntimeError("Analysis barrier made no progress; refusing to plan another candidate")
                    break
                except Exception as error:
                    self.events.emit("analysis_attempt_failed", iteration_id=target,
                                     payload={"attempt": attempt + 1, "error": str(error)})
                    if attempt + 1 >= attempts:
                        raise
                    try:
                        client.health()
                    except Exception:
                        client = self.start_engine(version, self.directory / "final-analysis" / f"retry-engine-{target}-{attempt}.log")

    def bootstrap(self):
        profile, preflight = collect(self.config)
        write_json(self.paths.data / "profiles" / f"{profile['profile_hash']}.json", profile)
        write_json(self.directory / "preflight.json", preflight)
        self.database.update(self.campaign_id, {"profile_hash": profile["profile_hash"]}, "profile_collected",
                             payload={"preflight": preflight})
        self.events.sync()
        if not preflight["passed"]:
            self.database.update(self.campaign_id, {"state": "BLOCKED", "stop_reason": ",".join(preflight["reasons"])},
                                 "deployment_blocked", payload=preflight)
            self.events.sync()
            return False
        workspace = self.paths.workspace(self.campaign_id, 0)
        if source_hash(self.config["engine"]["initial_source"]) != self.config["engine"]["initial_source_hash"]:
            raise ValueError("Human-supplied initial engine changed after configuration was frozen")
        engine_config = read_json(Path(self.config["engine"]["initial_source"]) / "engine.json")
        engine_config.update(precision_fields(self.config["precision"]))
        engine_config.update(device=self.config["model"]["device"], model_loader=self.config["engine"]["model_loader"])
        # The same engine serves both benchmark requests and Agent tool calls.
        # A larger Agent budget must not remain capped by the template default.
        engine_config["max_new_tokens"] = max(
            engine_config.get("max_new_tokens", 1024),
            max(p["max_new_tokens"] for p in workload_profiles(self.config["workload"])),
            self.config["agent"].get("max_new_tokens", 0),
        )
        engine_config["thinking"] = self.config["agent"].get("thinking", {"enabled": False})
        if not self.config["model"].get("fixture"):
            model = self.config["model"]
            engine_config.update(model_path=model["path"], max_context=model.get("max_context", 4096))
        if not workspace.exists():
            create_workspace(self.paths.project / self.config["engine_template"], workspace)
        write_json(workspace / "engine.json", engine_config)
        self.build_workspace(workspace, self.directory / "bootstrap")
        validate_candidate_runtime(engine_config, read_json(workspace / "engine.json"),
                                   {"optimization": {"allow_precision_changes": False}})
        release = freeze(workspace, self.paths, None, self.config["runtime_python"], self.config["config_hash"])
        self.database.update(self.campaign_id, {"version_profiles": {release["version_id"]: profile["profile_hash"]}}, "release_profile_bound")
        client = self.start_engine(release["version_id"], self.directory / "bootstrap" / "engine.log")
        self.smoke(client, self.directory / "bootstrap")
        cases = read_json(self.directory / "configs" / "cases.json")
        protocol = self.config["evaluation"]
        if protocol.get("reference_path"):
            raw = Path(protocol["reference_path"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != protocol["reference_file_hash"]:
                raise ValueError("Independent reference was modified")
            reference = load_reference(protocol["reference_path"])
            if (reference.get("schema_version") != 3
                    or reference.get("provenance", {}).get("oracle") != "independent"):
                raise ValueError("An external reference must use schema v3 and declare provenance.oracle=independent")
            validate_reference(reference, protocol, cases, self.config["workload"])
            write_json(self.directory / "reference.json", reference)
        else:
            vocab = profile["model"].get("text_config", {}).get("vocab_size", profile["model"].get("vocab_size", 32))
            reference = capture_reference(client, cases, self.directory / "reference.json",
                                      {"kind": "frozen_bootstrap_backend", "source_hash": release["source_hash"],
                                       "profile_hash": profile["profile_hash"], "environment": profile["environment"],
                                       "vocab_size": vocab, "oracle": "self_generated_regression_baseline"},
                                      protocol, self.config["workload"])
        report = evaluate_correctness(client, reference, self.config["evaluation"])
        initial_quality = self.quality(client, release["version_id"], self.directory / "bootstrap/quality")
        write_json(self.directory / "initial-quality.json", initial_quality)
        report["quality"] = initial_quality
        report["passed"] = report["passed"] and initial_quality["passed"]
        if initial_quality.get("enabled") is not False:
            report["coverage"].append("independent_task_quality")
            report["not_covered"] = [item for item in report["not_covered"] if item != "task_quality_benchmark"]
        write_json(self.directory / "bootstrap" / "correctness.json", report)
        if not report["passed"]:
            raise RuntimeError("Bootstrap correctness/quality gates failed; see bootstrap/correctness.json")
        operation = begin_promotion(self.database, self.campaign_id, release["version_id"])
        self.events.sync()
        self.state("COMMITTING")
        commit_promotion(self.database, self.campaign_id, operation)
        self.database.update(self.campaign_id, {"initial_ref": release["version_id"]}, "bootstrap_completed")
        self.events.sync()
        return True

    def activate(self, version, folder):
        operation = begin_promotion(self.database, self.campaign_id, version)
        self.events.sync()
        # Only the toy fixture supports deterministic crash injection.
        if self.config["model"].get("fixture") and os.environ.get("VIBEFORGE_FAILPOINT") == "after_promotion_intent":
            os._exit(86)
        try:
            client = self.start_engine(version, folder / "logs" / "activation.log")
            self.smoke(client, folder, "activation-smoke")
            self.state("COMMITTING")
            commit_promotion(self.database, self.campaign_id, operation)
            self.events.sync()
            if self.config["model"].get("fixture") and os.environ.get("VIBEFORGE_FAILPOINT") == "after_promotion_commit":
                os._exit(87)
            return client
        except Exception:
            self.restore("activation_failed", folder)
            raise

    def restore(self, reason, folder):
        target = cancel_promotion(self.database, self.campaign_id, reason)
        self.events.sync()
        if not target:
            self.database.update(self.campaign_id, {"state": "HALTED"}, "recovery_failed")
            raise RuntimeError("No committed stable release is available")
        client = self.start_engine(target, folder / "logs" / "restore.log")
        self.smoke(client, folder, "restore-smoke")
        rollback_completed(self.database, self.campaign_id, target)
        self.events.sync()
        return client

    def benchmark(self, stable, candidate, folder, attempt=0):
        baseline, challenger = [], []
        protocol, workload = self.config["evaluation"], self.config["workload"]
        for group in range(protocol["repeats"]):
            order = [(stable, baseline), (candidate, challenger)]
            if group % 2:
                order.reverse()
            for version, output in order:
                deadline = getattr(self, "evaluation_deadline", None)
                if ((deadline is not None and time.monotonic() > deadline) or
                        (deadline is None and time.time() - self.database.get(self.campaign_id)["started_at"] > self.config["budget"]["max_seconds"])):
                    raise TimeoutError("Campaign wall-clock budget reached during evaluation")
                client = self.start_engine(version, folder / "logs" / f"benchmark-{group}-{version}.log")
                for _ in range(protocol["warmup"]):
                    for profile in workload_profiles(workload):
                        for prompt in profile["prompts"]:
                            client.generate(prompt=prompt, max_new_tokens=profile["max_new_tokens"], fixed_length=True,
                                            observation_mode=measurement_mode(protocol))
                allowed = (self.engine.process.pid,) if self.engine.process else ()
                manifest = read_json(self.paths.release(version) / "manifest.json")
                source_bytes = sum((self.paths.release(version) / name).stat().st_size for name in manifest["files"])
                output.append(measure_suite(client, workload, folder / "benchmarks" / "samples.jsonl",
                                      version, group, protocol, allowed,
                                      reference=load_reference(self.directory / "reference.json"),
                                      startup_seconds=client.startup_seconds, source_bytes=source_bytes, attempt=attempt))
                self.events.emit("measurement_completed", iteration_id=folder.name,
                                 payload={"version": version, "group": group, "sample": output[-1]})
                self.stop_engine()
        return baseline, challenger

    def previously_rejected_patch(self, patch_text, iteration, parent, workspace):
        backend_change = backend_only_change(self.paths.release(parent), workspace)
        for row in self.database.iterations(self.campaign_id):
            if row["iteration"] >= iteration or row.get("parent") != parent:
                continue
            decision = row.get("decision") or {}
            if decision.get("outcome") != "reject" or decision.get("reason") not in REJECTED_PATCH_REASONS:
                continue
            previous_patch = (self.directory / "iterations" / f"{row['iteration']:04d}" / "change.patch")
            if not previous_patch.is_file():
                continue
            previous_text = previous_patch.read_text()
            if previous_text == patch_text:
                return row
            candidate = row.get("candidate")
            if backend_change and candidate and backend_change == backend_only_change(
                    self.paths.release(parent), self.paths.release(candidate)):
                return row
        return None

    def reject_before_candidate(self, number, folder, decision):
        write_json(folder / "decision.json", decision)
        self.database.iteration(self.campaign_id, number, {"decision": decision})
        self.events.emit("admission_decision", iteration_id=number, payload=decision)
        self.state("DECIDING")
        self.state("ACTIVATING")
        self.restore(decision["reason"], folder)

    def materialize_candidate(self, workspace, stable, number, folder, parent_config):
        """Build, preflight and freeze one immutable candidate revision."""
        folder.mkdir(parents=True, exist_ok=True)
        self.build_workspace(workspace, folder)
        precision = validate_candidate_runtime(
            parent_config, read_json(workspace / "engine.json"), self.config)
        candidate_config = deepcopy(self.config)
        candidate_config["precision"] = precision
        profile, preflight = collect(candidate_config)
        write_json(self.paths.data / "profiles" / f"{profile['profile_hash']}.json", profile)
        write_json(folder / "preflight.json", preflight)
        self.database.iteration(self.campaign_id, number, {
            "profile_hash": profile["profile_hash"], "precision": precision})
        if not preflight["passed"]:
            raise ValueError("Candidate deployment blocked: " + ",".join(preflight["reasons"]))
        release = freeze(
            workspace, self.paths, stable, self.config["runtime_python"],
            self.config["config_hash"], [str(folder)])
        version = release["version_id"]
        write_json(folder / "candidate.json", release)
        self.database.iteration(self.campaign_id, number, {"candidate": version})
        bindings = self.database.get(self.campaign_id).get("version_profiles", {})
        bindings[version] = profile["profile_hash"]
        self.database.update(
            self.campaign_id, {"version_profiles": bindings}, "release_profile_bound")
        return version, precision, release

    def quick_candidate_probe(self, client, version, plan, folder, deadline=None):
        """Collect non-admission correctness and mechanism feedback for revision."""
        deadline = deadline or (time.monotonic() + self.config["probe"]["timeout_seconds"])
        client = client.bounded(deadline, "candidate quick probe")
        reference = load_reference(self.directory / "reference.json")
        reference_info = plan.get("bottleneck_reference") or {}
        evidence_path = reference_info.get("evidence_path")
        before_record = (read_json(self.directory / evidence_path)
                         if evidence_path else read_json(folder.parent / "observation.json"))
        profile = next(
            item for item in workload_profiles(self.config["workload"])
            if item["id"] == before_record.get("profile_id", "default"))
        prompt_index = before_record.get("prompt_index", 0)
        probe_tokens = before_record.get(
            "profile_max_new_tokens",
            min(profile["max_new_tokens"], self.config["profiling"]["max_new_tokens"], 128))
        stable = self.database.get(self.campaign_id)["stable_ref"]
        stable_config = read_json(self.paths.release(stable) / "engine.json")
        candidate_config = read_json(self.paths.release(version) / "engine.json")
        scheduler_changed = (stable_config.get("request_scheduler", "serialized_lock") !=
                             candidate_config.get("request_scheduler", "serialized_lock"))
        concurrent_profiles = [item for item in workload_profiles(self.config["workload"])
                               if item.get("concurrency", 1) > 1]
        warmup_runs = 0
        precheck = evaluate_candidate_precheck(client, reference, self.config["evaluation"])
        if not precheck["passed"]:
            # Profiling the same broken candidate can throw and erase the
            # actionable correctness result, bypassing the refinement loop.
            report = {
                "schema_version": 1, "kind": "candidate_quick_probe", "candidate": version,
                "admission": False, "correctness_precheck": precheck,
                "mechanism": {"status": "inconclusive"}, "profile_reports": [],
                "profile_id": profile["id"], "prompt_index": prompt_index,
                "warmup_runs": warmup_runs, "feedback_state": "revise",
                "recommended_action": "revise_candidate", "needs_revision": True,
                "limitations": [
                    "Correctness precheck failed; no performance conclusion is available.",
                    "Inspect the recorded errors before revising; formal gates still own admission.",
                ],
            }
            write_json(folder / "probe.json", report)
            return report
        for _ in range(self.config["probe"]["warmup"]):
            remaining_seconds(deadline, "candidate quick probe")
            if scheduler_changed and concurrent_profiles:
                diagnose_concurrency(client, concurrent_profiles[0])
            else:
                client.generate(prompt=profile["prompts"][prompt_index],
                                max_new_tokens=probe_tokens, fixed_length=True)
            warmup_runs += 1
        reports = []
        for _ in range(self.config["probe"]["repeats"]):
            remaining_seconds(deadline, "candidate quick probe")
            report = client.profile(
                prompt=profile["prompts"][prompt_index],
                max_new_tokens=probe_tokens,
            )
            if report.get("source_hash") != source_hash(self.paths.release(version)):
                raise RuntimeError("Candidate probe source identity mismatch")
            reports.append(report)
        metric = reference_info.get("metric")
        before = reference_info.get("value")
        concurrency_report = None
        if scheduler_changed:
            if concurrent_profiles:
                target = concurrent_profiles[0]
                baseline = read_json(
                    self.directory / "observations" / stable / "baseline.json")
                concurrency_report = diagnose_concurrency(client, target)
                metric = f"diagnostic.concurrent.{target['id']}.throughput"
                before = ((baseline.get("concurrency_diagnostics") or {}).get(target["id"]) or {}).get("throughput")
        after_values = [
            report.get("measurements", {}).get(metric, {}).get("value") for report in reports
            if isinstance(report.get("measurements", {}).get(metric, {}).get("value"), (int, float))
        ] if metric else []
        after = (concurrency_report.get("throughput") if concurrency_report else
                 sum(after_values) / len(after_values) if after_values else None)
        change = (after / before - 1) * 100 if isinstance(before, (int, float)) and before > 0 and after is not None else None
        direction = "increase" if concurrency_report else plan.get("bottleneck_direction")
        improvement = (-change if direction == "decrease" else change) if change is not None else None
        expected = plan.get("expected_bottleneck_change_pct")
        prediction_observed, prediction_status, direction_feedback = probe_prediction(
            improvement, expected)
        feedback_state = "revise" if not precheck["passed"] else direction_feedback
        recommended_action = (
            "revise_candidate" if feedback_state == "revise" else "continue_formal_validation")
        report = {
            "schema_version": 1, "kind": "candidate_quick_probe", "candidate": version,
            "admission": False, "correctness_precheck": precheck,
            "mechanism": {
                "metric": metric, "before": before, "after": after,
                "observed_change_pct": change, "improvement_pct": improvement,
                "expected_improvement_pct": expected, "prediction_observed": prediction_observed,
                "status": prediction_status,
            },
            "profile_id": profile["id"], "prompt_index": prompt_index,
            "profile_reports": reports,
            "concurrency_report": concurrency_report,
            "warmup_runs": warmup_runs,
            "feedback_state": feedback_state,
            "recommended_action": recommended_action,
            "needs_revision": feedback_state == "revise",
            "limitations": [
                "Diagnostic profiler runs only and cannot accept or reject a candidate.",
                "Prediction magnitude is descriptive feedback; formal correctness and paired performance gates own admission.",
            ],
        }
        write_json(folder / "probe.json", report)
        return report

    def run_iteration(self, number):
        stable = self.database.get(self.campaign_id)["stable_ref"]
        folder = self.directory / "iterations" / f"{number:04d}"
        folder.mkdir(parents=True, exist_ok=True)
        workspace = create_workspace(self.paths.release(stable), self.paths.workspace(self.campaign_id, number))
        source_profile = self.database.get(self.campaign_id).get("version_profiles", {}).get(stable)
        self.database.iteration(self.campaign_id, number, {"parent": stable, "complete": False,
            "source_profile_hash": source_profile, "source_precision": read_json(workspace / "engine.json")["precision"]})
        decision = {"outcome": "reject", "reason": "unfinished"}
        failure = False
        try:
            client = self.start_engine(stable, folder / "logs" / "stable-agent.log")
            write_json(folder / "observation.json", self.observe_stable(client, stable))
            self.state("AGENT_RUNNING")
            agent_result = self.agent_session(client, workspace, number)
            atomic_write(folder / "hypothesis.md", agent_result["hypothesis"] + "\n")
            self.database.iteration(self.campaign_id, number, {"hypothesis": agent_result["hypothesis"],
                                    "experiment_plan": agent_result.get("experiment_plan")})
            change_patch = folder / "change.patch"
            write_diff(self.paths.release(stable), workspace, change_patch)
            parent_config = read_json(self.paths.release(stable) / "engine.json")
            try:
                child_config = read_json(workspace / "engine.json")
                if not isinstance(child_config, dict):
                    raise ValueError("engine.json must be an object")
                precision = validate_candidate_runtime(parent_config, child_config, self.config)
            except (ValueError, TypeError) as error:
                # Invalid generated configuration is a proposal failure, not
                # an engine/infrastructure outage consuming recovery attempts.
                decision = {"outcome": "reject", "reason": "invalid_candidate_runtime", "error": str(error)}
                write_json(folder / "candidate-gate.json", {"valid": False, **decision})
                self.reject_before_candidate(number, folder, decision)
                return
            self.stop_engine()
            if agent_result.get("submission_origin") == "controller_recovery":
                failure = True
                decision = {"outcome": "reject", "reason": "agent_protocol_error",
                            "error": agent_result.get("recovery_error")}
                self.reject_before_candidate(number, folder, decision)
                return
            if agent_result.get("blocked"):
                decision = {"outcome": "reject", "reason": "blocked_proposal", "blocker": agent_result.get("blocked_reason")}
                self.reject_before_candidate(number, folder, decision)
                return
            patch_text = change_patch.read_text()
            if not patch_text.strip():
                decision = {"outcome": "reject", "reason": "no_change"}
                self.reject_before_candidate(number, folder, decision)
                return
            gate_context = create_context(self.database, self.paths, self.campaign_id, self.config, workspace, number)
            # Additional diagnostics can only have been written through the host profile tool.
            gate_context["candidate_contract"]["observation_paths"].extend(
                str(path.relative_to(self.directory)) for path in (self.directory / "observations" / stable).glob(f"agent-{number}-*.json"))
            gate = candidate_gate(workspace, gate_context, agent_result.get("experiment_plan"))
            self.database.iteration(self.campaign_id, number, {"candidate_gate": gate})
            write_json(folder / "candidate-gate.json", gate)
            if not gate["valid"]:
                decision = {"outcome": "reject", **gate}
                self.reject_before_candidate(number, folder, decision)
                return
            # The gate binds Agent judgment to controller-owned evidence and
            # fills objective, direction, profile and runtime facts.  All
            # downstream stages must consume that canonical plan rather than
            # the Agent's intentionally concise submission.
            if gate.get("experiment_plan"):
                agent_result["experiment_plan"] = gate["experiment_plan"]
                write_json(folder / "agent-result.json", agent_result)
                self.database.iteration(
                    self.campaign_id, number,
                    {"experiment_plan": gate["experiment_plan"]},
                )
            previous = self.previously_rejected_patch(patch_text, number, stable, workspace)
            if previous:
                previous_decision = previous.get("decision") or {}
                decision = {
                    "outcome": "reject", "reason": "duplicate_change",
                    "previous_iteration": previous["iteration"],
                    "previous_reason": previous_decision.get("reason"),
                }
                self.reject_before_candidate(number, folder, decision)
                return
            state = self.database.get(self.campaign_id)
            experiment_id = state.get("valid_candidates", 0) + 1
            self.database.iteration(self.campaign_id, number, {"experiment_id": experiment_id})
            self.database.update(self.campaign_id, {"valid_candidates": experiment_id}, "candidate_validated", iteration_id=number)
            self.state("BUILDING")
            version, precision, release = self.materialize_candidate(
                workspace, stable, number, folder, parent_config)
            revision = 0
            probe_attempt = 0
            probe_report = None
            refinement_error = None
            while (self.config["probe"]["enabled"]
                   and probe_attempt < self.config["probe"]["max_attempts"]):
                self.state("PROBING")
                probe_folder = folder / "probe" / f"{probe_attempt:04d}"
                probe_folder.mkdir(parents=True, exist_ok=True)
                probe_deadline = time.monotonic() + self.config["probe"]["timeout_seconds"]
                try:
                    client = self.start_engine(
                        version, probe_folder / "engine.log",
                        timeout_seconds=remaining_seconds(probe_deadline, "candidate quick probe"))
                    probe_report = self.quick_candidate_probe(
                        client, version, agent_result["experiment_plan"], probe_folder,
                        deadline=probe_deadline)
                except Exception as error:
                    probe_report = {
                        "schema_version": 1, "kind": "candidate_quick_probe",
                        "candidate": version, "admission": False,
                        "feedback_state": "inconclusive",
                        "recommended_action": "continue_formal_validation",
                        "needs_revision": False,
                        "correctness_precheck": {"passed": None, "checks": []},
                        "mechanism": {"status": "inconclusive"},
                        "error": str(error),
                        "limitations": [
                            "The diagnostic probe did not complete; this cannot reject or accept the candidate.",
                            "Formal correctness and paired performance gates own admission.",
                        ],
                    }
                    write_json(probe_folder / "probe.json", probe_report)
                finally:
                    self.stop_engine()
                state = self.database.get(self.campaign_id)
                self.database.update(self.campaign_id, {
                    "probe_attempts": state.get("probe_attempts", 0) + 1,
                }, "candidate_probe_completed", iteration_id=number, payload={
                    "revision": revision, "candidate": version,
                    "feedback_state": probe_report["feedback_state"],
                    "needs_revision": probe_report["needs_revision"],
                    "correctness_passed": probe_report["correctness_precheck"]["passed"],
                })
                probe_attempt += 1
                if (not probe_report["needs_revision"] or not self.config["refinement"]["enabled"]
                        or revision >= self.config["refinement"]["max_revisions"]):
                    break
                revision += 1
                self.state("REFINING")
                revision_folder = folder / "refinement" / f"{revision:04d}"
                base_workspace = self.paths.workspace(self.campaign_id, number)
                revision_workspace = create_workspace(
                    workspace, base_workspace.parent / f"{base_workspace.name}-revision-{revision:04d}")
                refinement_deadline = time.monotonic() + self.config["refinement"]["timeout_seconds"]
                revision_result = {"blocked": True}
                try:
                    stable_client = self.start_engine(
                        stable, revision_folder / "stable-agent.log",
                        timeout_seconds=remaining_seconds(refinement_deadline, "candidate refinement"))
                    revision_result = self.agent_session(
                        stable_client, revision_workspace, number,
                        session_folder=revision_folder,
                        context_additions={
                            "candidate_feedback": probe_report,
                            "candidate_feedback_path": str(
                                (probe_folder / "probe.json").relative_to(self.directory)),
                            "refinement": {"revision": revision,
                                           "max_revisions": self.config["refinement"]["max_revisions"]},
                            "refinement_target": {
                                "candidate": version, "stable_parent": stable,
                                "hypothesis": agent_result.get("hypothesis"),
                                "experiment_plan": {key: agent_result["experiment_plan"].get(key) for key in (
                                    "measured_bottleneck", "cause", "change", "expected_change_pct",
                                    "bottleneck_reference", "validation", "history_refs")},
                                "source_patch": change_patch.read_text()[:12000],
                                "source_patch_truncated": len(change_patch.read_text()) > 12000,
                                "source_patch_evidence": str(change_patch.relative_to(self.directory)),
                                "scope": "This is the unpromoted candidate being revised, not the stable baseline.",
                            },
                            "instructions": (
                                "Revise refinement_target, using its hypothesis, actual patch and quick-probe feedback. "
                                "Keep the revision tied to this experiment; do not silently substitute an unrelated "
                                "optimization. If it cannot be repaired, submit blocked=true with the reason. "
                                "The next iteration can select another experiment. Preserve the frozen protocol."),
                        },
                        phase="refinement",
                        timeout_seconds=remaining_seconds(refinement_deadline, "candidate refinement"),
                    )
                except Exception as error:
                    refinement_error = str(error)
                    revision_result = {"blocked": True, "error": refinement_error}
                finally:
                    self.stop_engine()
                state = self.database.get(self.campaign_id)
                self.database.update(self.campaign_id, {
                    "refinement_attempts": state.get("refinement_attempts", 0) + 1,
                }, "candidate_refinement_completed", iteration_id=number,
                   payload={"revision": revision, "blocked": revision_result.get("blocked", False),
                            "error": refinement_error})
                if refinement_error:
                    break
                revision_patch = revision_folder / "change.patch"
                write_diff(self.paths.release(stable), revision_workspace, revision_patch)
                revision_context = create_context(
                    self.database, self.paths, self.campaign_id, self.config,
                    revision_workspace, number)
                revision_context["candidate_contract"]["observation_paths"].extend(
                    str(path.relative_to(self.directory)) for path in
                    (self.directory / "observations" / stable).glob(f"agent-{number}-*.json"))
                revision_gate = candidate_gate(
                    revision_workspace, revision_context, revision_result.get("experiment_plan"))
                write_json(revision_folder / "candidate-gate.json", revision_gate)
                if (revision_result.get("blocked") or not revision_patch.read_text().strip()
                        or not revision_gate["valid"]):
                    write_json(revision_folder / "outcome.json", {
                        "applied": False, "blocked": revision_result.get("blocked", False),
                        "gate": revision_gate,
                    })
                    break
                self.state("BUILDING")
                try:
                    revised_version, revised_precision, revised_release = self.materialize_candidate(
                        revision_workspace, stable, number, revision_folder, parent_config)
                except Exception as error:
                    refinement_error = str(error)
                    write_json(revision_folder / "outcome.json", {
                        "applied": False, "blocked": False, "gate": revision_gate,
                        "error": refinement_error,
                    })
                    original_profile = self.database.get(self.campaign_id).get(
                        "version_profiles", {}).get(version)
                    self.database.iteration(self.campaign_id, number, {
                        "candidate": version, "profile_hash": original_profile,
                        "precision": precision,
                    })
                    break
                workspace = revision_workspace
                agent_result = revision_result
                agent_result["experiment_plan"] = revision_gate["experiment_plan"]
                gate = revision_gate
                version, precision, release = revised_version, revised_precision, revised_release
                write_json(folder / "agent-result.json", agent_result)
                write_json(folder / "candidate-gate.json", gate)
                write_diff(self.paths.release(stable), workspace, change_patch)
                atomic_write(folder / "hypothesis.md", agent_result["hypothesis"] + "\n")
                self.database.iteration(self.campaign_id, number, {
                    "hypothesis": agent_result["hypothesis"],
                    "experiment_plan": agent_result["experiment_plan"],
                    "candidate_gate": gate,
                })
                write_json(folder / "candidate.json", release)
                write_json(revision_folder / "outcome.json", {
                    "applied": True, "candidate": version, "gate": gate})
            if probe_report:
                probe_summary = {
                    "schema_version": 1, "admission": False, "candidate": version,
                    "feedback_state": probe_report["feedback_state"],
                    "recommended_action": probe_report["recommended_action"],
                    "needs_revision": probe_report["needs_revision"],
                    "probe_evidence_path": str((probe_folder / "probe.json").relative_to(self.directory)),
                    "refinement_attempts": revision,
                }
                if refinement_error:
                    probe_summary["refinement_error"] = refinement_error
                write_json(folder / "probe-summary.json", probe_summary)
                self.database.iteration(self.campaign_id, number, {"probe_feedback": probe_summary})
            self.state("CANDIDATE_FROZEN")
            self.state("VALIDATING")
            client = self.start_engine(version, folder / "logs" / "correctness.log")
            reference = load_reference(self.directory / "reference.json")
            precheck = evaluate_candidate_precheck(client, reference, self.config["evaluation"])
            write_json(folder / "correctness" / "precheck.json", precheck)
            effect_fingerprint = None
            duplicate_effect = None
            observation = precheck.get("runtime_observation")
            effect_scope = runtime_configuration_only_change(self.paths.release(stable), workspace)
            if observation:
                if observation.get("source_hash") != source_hash(self.paths.release(version)):
                    raise RuntimeError("Candidate precheck source identity mismatch")
                effect_fingerprint = runtime_effect_fingerprint(observation)
                self.database.iteration(self.campaign_id, number, {
                    "effect_fingerprint": effect_fingerprint, "effect_scope": effect_scope})
                baseline_effect = runtime_effect_fingerprint(read_json(folder / "observation.json"))
                if effect_scope and effect_fingerprint and effect_fingerprint == baseline_effect:
                    duplicate_effect = {"reason": "ineffective_runtime_change", "matches": "stable_parent"}
                else:
                    for previous in gate_context["candidate_contract"].get("tested_candidates", []):
                        if (effect_scope and previous.get("effect_scope") and effect_fingerprint
                                and previous.get("effect_fingerprint") == effect_fingerprint):
                            duplicate_effect = {"reason": "duplicate_runtime_effect",
                                "previous_campaign": previous.get("campaign_id"),
                                "previous_iteration": previous.get("iteration")}
                            break
            if precheck["passed"] and not duplicate_effect:
                correctness = evaluate_correctness(client, reference, self.config["evaluation"])
            else:
                correctness = {"passed": False, "checks": precheck["checks"], "precheck": True,
                    "runtime_effect": duplicate_effect,
                    "failure_summary": {"failed_checks": sum(not check["passed"] for check in precheck["checks"]),
                                        "operational_errors": sum(bool(check.get("error")) for check in precheck["checks"])},
                    "coverage": ["candidate_runtime_precheck"], "not_covered": ["full_correctness_suite"]}
            if correctness["passed"]:
                quality = self.quality(client, version, folder / "quality", read_json(self.directory / "initial-quality.json"))
                correctness["quality"] = quality
                correctness["passed"] = quality["passed"]
                if quality.get("enabled") is not False:
                    correctness["coverage"].append("independent_task_quality")
                    correctness["not_covered"] = [item for item in correctness["not_covered"]
                                                   if item != "task_quality_benchmark"]
            write_json(folder / "correctness" / "report.json", correctness)
            candidate_observation = None
            if correctness["passed"]:
                plan = agent_result["experiment_plan"]
                reference_path = (plan.get("bottleneck_reference") or {}).get("evidence_path")
                before_observation = read_json(self.directory / reference_path) if reference_path else read_json(folder / "observation.json")
                selected = next(p for p in workload_profiles(self.config["workload"])
                                if p["id"] == before_observation.get("profile_id", "default"))
                candidate_observation = client.profile(prompt=selected["prompts"][before_observation.get("prompt_index", 0)],
                    max_new_tokens=before_observation["profile_max_new_tokens"])
                candidate_observation.update(profile_id=selected["id"], workload=self.config["workload"],
                                             prompt_index=before_observation.get("prompt_index", 0))
                if candidate_observation.get("source_hash") != source_hash(self.paths.release(version)):
                    raise RuntimeError("Candidate diagnostic source identity mismatch")
                write_json(folder / "candidate-observation.json", candidate_observation)
            self.stop_engine()
            if correctness["passed"]:
                self.state("BENCHMARKING")
                for attempt in range(self.config["evaluation"]["max_retests"] + 1):
                    baseline, challenger = self.benchmark(stable, version, folder, attempt)
                    result = decide(correctness, baseline, challenger, self.config["evaluation"])
                    decision = result.to_dict()
                    write_json(folder / "benchmarks" / f"summary-{attempt}.json",
                               {"baseline": baseline, "candidate": challenger, "decision": decision})
                    self.state("DECIDING")
                    if result.outcome != "invalid" and result.reason != "no_significant_gain":
                        break
                    if attempt < self.config["evaluation"]["max_retests"]:
                        self.state("BENCHMARKING")
                plan = agent_result["experiment_plan"]
                verification = {"metric": plan["metric"], "expected_change_pct": plan["expected_change_pct"],
                                "direction": plan["direction"], "status": "inconclusive"}
                reference = plan.get("bottleneck_reference") or {}
                if reference and candidate_observation:
                    key = reference["metric"]
                    after = candidate_observation.get("measurements", {}).get(key)
                    mechanism = {"metric": key, "before": reference["value"], "after": after,
                                 "expected_change_pct": plan["expected_bottleneck_change_pct"],
                                 "evidence_path": "candidate-observation.json",
                                 "status": "unavailable", "causal_proof": False,
                                 "scope": "single diagnostic comparison; descriptive only; formal admission uses paired unprofiled requests"}
                    if after and after.get("unit") == reference["unit"]:
                        change = (after["value"] / reference["value"] - 1) * 100
                        improvement = change if plan["bottleneck_direction"] == "increase" else -change
                        mechanism.update(observed_change_pct=change,
                                         status="consistent_with_prediction" if improvement >= plan["expected_bottleneck_change_pct"] else "prediction_not_observed")
                    verification["mechanism"] = mechanism
                if all(sample["valid"] for sample in baseline + challenger):
                    try:
                        interval = paired_gain([sample["metrics"][plan["metric"]] for sample in baseline],
                                               [sample["metrics"][plan["metric"]] for sample in challenger],
                                               self.config["evaluation"]["confidence"],
                                               direction="maximize" if plan["direction"] == "increase" else "minimize")
                        verification.update(interval)
                        threshold = plan["expected_change_pct"] / 100
                        verification["status"] = ("supported" if interval["lower"] >= threshold else
                                                  "refuted" if interval["upper"] < threshold else "inconclusive")
                    except (KeyError, TypeError, ValueError):
                        verification["unavailable_reason"] = "predicted_metric_unavailable"
                self.database.iteration(self.campaign_id, number, {"experiment_verification": verification})
                write_json(folder / "verification.json", verification)
            else:
                self.state("DECIDING")
                reason = ((duplicate_effect or {}).get("reason") or
                          ("quality_failed" if not correctness.get("quality", {"passed": True})["passed"]
                           else "correctness_failed"))
                decision = {"outcome": "reject", "reason": reason}
                if duplicate_effect:
                    decision.update(duplicate_effect)
            write_json(folder / "decision.json", decision)
            self.database.iteration(self.campaign_id, number, {"decision": decision})
            self.events.emit("admission_decision", iteration_id=number, payload=decision)
            if decision["outcome"] == "accept":
                self.activate(version, folder)
            else:
                self.state("ACTIVATING")
                self.restore(decision["reason"], folder)
        except KeyboardInterrupt:
            recorded = (self.database.iteration(self.campaign_id, number).get("decision") or {})
            if recorded.get("reason") and recorded.get("reason") != "unfinished":
                decision = recorded
                self.database.iteration(self.campaign_id, number, {"interrupted_after_decision": True})
            else:
                decision = {"outcome": "reject", "reason": "interrupted_iteration"}
                self.database.iteration(self.campaign_id, number, {"interrupted": True, "decision": decision})
                write_json(folder / "decision.json", decision)
            raise
        except Exception as error:
            failure = True
            decision = {"outcome": "reject", "reason": "execution_failed", "error": str(error)}
            self.database.iteration(self.campaign_id, number, {"error": str(error), "decision": decision})
            write_json(folder / "decision.json", decision)
            self.events.emit("iteration_failed", iteration_id=number, payload={"error": str(error)})
            try:
                self.restore(str(error), folder)
            except Exception as recovery_error:
                self.database.update(self.campaign_id, {"state": "HALTED", "stop_reason": str(recovery_error)},
                                     "recovery_failed")
                self.events.sync()
                raise
        finally:
            row = self.database.iteration(self.campaign_id, number,
                                          {"decision": decision, "complete": True, "analysis_pending": True})
            write_analysis(folder / "analysis.md", row, None)
            refresh_memory(self.database, self.campaign_id, self.directory)
            extract_draft(self.paths, self.campaign_id, row, row.get("source_profile_hash", self.database.get(self.campaign_id).get("profile_hash")),
                          {"precision": row.get("source_precision", self.config["precision"]), "device": self.config["model"]["device"],
                           "objective": self.config["requirements"]["objective"]})
            state = self.database.get(self.campaign_id)
            candidate_failure = decision.get("reason") in {
                "correctness_failed", "quality_failed", "ineffective_runtime_change",
                "duplicate_runtime_effect", "performance_regression", "no_significant_gain",
                "candidate_probe_failed",
            }
            self.database.update(self.campaign_id, {
                "completed_iterations": number, "analysis_pending": True,
                "proposal_attempts": number,
                "completed_experiments": state.get("completed_experiments", 0) + int(bool(row.get("experiment_id")) and not row.get("interrupted")),
                "stalled_attempts": 0 if row.get("experiment_id") else state.get("stalled_attempts", 0) + 1,
                "consecutive_failures": state.get("consecutive_failures", 0) + 1 if failure else 0,
                "consecutive_candidate_failures": (
                    state.get("consecutive_candidate_failures", 0) + 1 if candidate_failure else 0),
            }, "iteration_completed", iteration_id=number, payload={"decision": decision})
            self.events.sync()
            write_campaign_report(self.database, self.paths, self.campaign_id)

    def run(self, resume=False):
        self.config = read_json(self.directory / "manifest.json")
        if self.config["config_hash"] != digest({k: v for k, v in self.config.items() if k != "config_hash"}):
            raise ValueError("Frozen campaign configuration was modified")
        if self.config.get("schema_version") != 4:
            raise ValueError("This campaign uses an older loop contract; retain its evidence and create a schema v4 campaign")
        if self.config["evaluation"].get("measurement_protocol_version") != 2:
            raise ValueError("This campaign predates comparable formal timing; retain its evidence and create a new campaign")
        with exclusive_lock(self.paths.data / "state" / f"{self.campaign_id}.lock"):
            with exclusive_lock(self.paths.data / "state" / "device.lock"):
                try:
                    if resume:
                        if self.database.get(self.campaign_id).get("stop_requested"):
                            self.database.update(self.campaign_id,
                                                 {"stop_requested": False, "stop_reason": None},
                                                 "resume_requested")
                            self.events.sync()
                        recover(self.database, self.paths, self.campaign_id)
                        self.events.sync()
                        current_profile, _ = collect(self.config)
                        recorded = self.database.get(self.campaign_id).get("profile_hash")
                        if recorded and current_profile["profile_hash"] != recorded:
                            raise RuntimeError("Model, hardware, or software profile changed; create a new campaign")
                    state = self.database.get(self.campaign_id)
                    if not state.get("stable_ref"):
                        if not self.bootstrap():
                            return self.database.get(self.campaign_id)
                    else:
                        self.database.update(self.campaign_id, {"state": "READY", "stop_reason": None}, "stable_recovered")
                        self.events.sync()
                    while not stop_reason(self.database.get(self.campaign_id), self.config["budget"]):
                        self.complete_pending_analyses()
                        if stop_reason(self.database.get(self.campaign_id), self.config["budget"]):
                            break
                        number = self.database.get(self.campaign_id).get("proposal_attempts", self.database.get(self.campaign_id)["completed_iterations"]) + 1
                        self.run_iteration(number)
                    reason = stop_reason(self.database.get(self.campaign_id), self.config["budget"])
                    state = self.database.get(self.campaign_id)
                    if state.get("analysis_pending") and not state.get("stop_requested"):
                        self.complete_pending_analyses()
                    self.state("STOPPED", stop_reason=reason)
                    return self.database.get(self.campaign_id)
                except KeyboardInterrupt:
                    self.database.update(self.campaign_id, {"state": "STOPPED", "stop_requested": True,
                                         "stop_reason": "keyboard_interrupt", "active_ref": None}, "campaign_interrupted")
                    self.events.sync()
                    return self.database.get(self.campaign_id)
                except Exception as error:
                    self.database.update(self.campaign_id, {"state": "HALTED", "stop_reason": str(error)},
                                         "campaign_failed", payload={"error": str(error)})
                    self.events.sync()
                    raise
                finally:
                    if self.engine:
                        self.stop_engine()
                    write_campaign_report(self.database, self.paths, self.campaign_id)
