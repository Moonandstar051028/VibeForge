"""Executable single-device optimization loop with durable, evidence-based decisions."""

import os
import random
import sys
import time
import uuid
from pathlib import Path
from vibeforge.agent.analysis import write_analysis
from vibeforge.agent.context import create_context
from vibeforge.agent.runner import smoke_agent
from vibeforge.controller.admission import decide
from vibeforge.controller.budgets import stop_reason
from vibeforge.controller.recovery import recover
from vibeforge.controller.state_machine import transition
from vibeforge.engine.builder import build
from vibeforge.engine.process import EngineProcess
from vibeforge.evaluation.correctness import evaluate_correctness
from vibeforge.evaluation.performance import measure
from vibeforge.evaluation.reference import capture_reference, load_reference
from vibeforge.knowledge.extractor import extract_draft
from vibeforge.knowledge.registry import load_rules
from vibeforge.knowledge.retriever import retrieve
from vibeforge.observability.events import EventLog
from vibeforge.observability.logs import run_command
from vibeforge.observability.reports import write_campaign_report
from vibeforge.profiling.collector import collect
from vibeforge.storage.atomic import atomic_write, read_json, write_json, digest
from vibeforge.storage.database import Database
from vibeforge.storage.locks import exclusive_lock
from vibeforge.storage.paths import identifier
from vibeforge.versioning.artifacts import verify_release
from vibeforge.versioning.candidate import create_workspace, write_diff
from vibeforge.versioning.promotion import begin_promotion, commit_promotion
from vibeforge.versioning.release import freeze
from vibeforge.versioning.rollback import cancel_promotion, rollback_completed


class Supervisor:
    def __init__(self, paths, campaign_id):
        self.paths, self.campaign_id = paths, identifier(campaign_id)
        self.database = Database(paths.data / "state" / "controller.sqlite3")
        self.directory = paths.campaign(campaign_id)
        self.events = EventLog(self.database, campaign_id, self.directory / "events")
        self.engine = None

    def close(self):
        if self.engine:
            self.engine.stop()
        self.database.close()

    def create(self, config):
        if any(row["campaign_id"] == self.campaign_id for row in self.database.list_campaigns()):
            raise ValueError(f"Campaign already exists: {self.campaign_id}")
        self.directory.mkdir(parents=True, exist_ok=True)
        write_json(self.directory / "manifest.json", config)
        for section in ("model", "hardware", "workload", "evaluation", "agent"):
            write_json(self.directory / "configs" / f"{section}.json", config[section])
        atomic_write(self.directory / "configs" / "agent-prompt.md",
                     (self.paths.project / config["agent"]["prompt"]).read_text())
        write_json(self.directory / "configs" / "cases.json",
                   read_json(self.paths.project / config["workload"]["cases"]))
        state = {"campaign_id": self.campaign_id, "state": "RECOVERING",
                 "stable_ref": None, "previous_stable_ref": None, "active_ref": None,
                 "completed_iterations": 0, "consecutive_failures": 0, "started_at": time.time(),
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

    def start_engine(self, version, log):
        manifest = read_json(self.paths.release(version) / "manifest.json")
        verify_release(self.paths.release(version), manifest)
        if self.engine:
            self.engine.stop()
        self.engine = EngineProcess(self.paths, self.campaign_id, manifest["runtime_python"],
                                    self.config.get("startup_timeout", 300),
                                    self.config.get("request_timeout", 120),
                                    self.config.get("inference_pythonpath", ()))
        started = time.monotonic()
        client = self.engine.start(self.paths.release(version), log)
        self.database.update(self.campaign_id, {"active_ref": version}, "engine_started",
                             payload={"version": version, "startup_seconds": time.monotonic() - started})
        self.events.sync()
        return client

    def stop_engine(self):
        if self.engine:
            self.engine.stop()
        self.database.update(self.campaign_id, {"active_ref": None}, "engine_stopped")
        self.events.sync()

    def agent_session(self, client, workspace, iteration, analysis_only=False):
        folder = self.directory / "iterations" / f"{iteration:04d}"
        if analysis_only:
            folder = self.directory / "final-analysis"
        folder.mkdir(parents=True, exist_ok=True)
        context = create_context(self.database, self.paths, self.campaign_id, self.config, workspace, iteration)
        context["knowledge"] = retrieve(load_rules(self.paths.project / "knowledge"),
                                        {"profile_hash": self.database.get(self.campaign_id).get("profile_hash")})
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
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(self.paths.project / "src") + os.pathsep + environment.get("PYTHONPATH", "")
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        self.events.emit("agent_started", iteration_id=iteration, payload={"analysis_only": analysis_only})
        record = run_command(command, workspace, folder / "logs" / "agent.log",
                             timeout=self.config["budget"]["agent_seconds"], env=environment,
                             trace=folder / "commands.jsonl",
                             process_record=self.paths.data / "state" / f"{self.campaign_id}.agent.json", token=token,
                             heartbeat=lambda: self.events.emit("agent_heartbeat", iteration_id=iteration))
        if record["exit_code"]:
            raise RuntimeError(f"Agent failed with exit code {record['exit_code']}; see {folder / 'logs/agent.log'}")
        result = read_json(result_path)
        self.events.emit("agent_completed", iteration_id=iteration,
                         payload={"session_id": result["session_id"], "mode": result["mode"]})
        previous = context.get("previous")
        if previous and result.get("previous_analysis"):
            previous["analysis_pending"] = False
            self.database.iteration(self.campaign_id, previous["iteration"], previous)
            write_analysis(self.directory / "iterations" / f"{previous['iteration']:04d}" / "analysis.md",
                           previous, result["previous_analysis"])
            self.database.update(self.campaign_id, {"analysis_pending": False}, "analysis_completed",
                                 iteration_id=previous["iteration"])
            self.events.sync()
        return result

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
        if not workspace.exists():
            create_workspace(self.paths.project / self.config["engine_template"], workspace)
            engine_config = read_json(workspace / "engine.json")
            if not self.config["model"].get("fixture"):
                model = self.config["model"]
                engine_config.update(model_path=model["path"], dtype=model.get("dtype", "bfloat16"),
                                     device=model.get("device", "cuda"), quantization=model.get("quantization"),
                                     max_context=model.get("max_context", 4096))
            write_json(workspace / "engine.json", engine_config)
        build(workspace)
        release = freeze(workspace, self.paths, None, self.config["runtime_python"], self.config["config_hash"])
        client = self.start_engine(release["version_id"], self.directory / "bootstrap" / "engine.log")
        smoke = smoke_agent(client)
        write_json(self.directory / "bootstrap" / "agent-smoke.json", smoke)
        cases = read_json(self.directory / "configs" / "cases.json")
        reference = capture_reference(client, cases, self.directory / "reference.json",
                                      {"kind": "frozen_bootstrap_backend", "source_hash": release["source_hash"],
                                       "profile_hash": profile["profile_hash"], "environment": profile["environment"]})
        report = evaluate_correctness(client, reference, self.config["evaluation"])
        write_json(self.directory / "bootstrap" / "correctness.json", report)
        if not report["passed"]:
            raise RuntimeError("Bootstrap repeatability failed")
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
            write_json(folder / "activation-smoke.json", smoke_agent(client))
            self.state("COMMITTING")
            commit_promotion(self.database, self.campaign_id, operation)
            self.events.sync()
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
        smoke_agent(client)
        rollback_completed(self.database, self.campaign_id, target)
        self.events.sync()
        return client

    def benchmark(self, stable, candidate, folder):
        baseline, challenger = [], []
        protocol, workload = self.config["evaluation"], self.config["workload"]
        for group in range(protocol["repeats"]):
            order = [(stable, baseline), (candidate, challenger)]
            if group % 2:
                order.reverse()
            for version, output in order:
                if time.time() - self.database.get(self.campaign_id)["started_at"] > self.config["budget"]["max_seconds"]:
                    raise TimeoutError("Campaign wall-clock budget reached during evaluation")
                client = self.start_engine(version, folder / "logs" / f"benchmark-{group}-{version}.log")
                for _ in range(protocol["warmup"]):
                    for prompt in workload["prompts"]:
                        client.generate(prompt=prompt, max_new_tokens=workload["max_new_tokens"], fixed_length=True)
                allowed = (self.engine.process.pid,) if self.engine.process else ()
                output.append(measure(client, workload, folder / "benchmarks" / "samples.jsonl",
                                      version, group, protocol, allowed))
                self.events.emit("measurement_completed", iteration_id=folder.name,
                                 payload={"version": version, "group": group, "sample": output[-1]})
                self.stop_engine()
        return baseline, challenger

    def run_iteration(self, number):
        stable = self.database.get(self.campaign_id)["stable_ref"]
        folder = self.directory / "iterations" / f"{number:04d}"
        folder.mkdir(parents=True, exist_ok=True)
        workspace = create_workspace(self.paths.release(stable), self.paths.workspace(self.campaign_id, number))
        self.database.iteration(self.campaign_id, number, {"parent": stable, "complete": False})
        decision = {"outcome": "reject", "reason": "unfinished"}
        failure = False
        try:
            client = self.start_engine(stable, folder / "logs" / "stable-agent.log")
            self.state("AGENT_RUNNING")
            agent_result = self.agent_session(client, workspace, number)
            atomic_write(folder / "hypothesis.md", agent_result["hypothesis"] + "\n")
            self.database.iteration(self.campaign_id, number, {"hypothesis": agent_result["hypothesis"]})
            write_diff(self.paths.release(stable), workspace, folder / "change.patch")
            parent_config = read_json(self.paths.release(stable) / "engine.json")
            child_config = read_json(workspace / "engine.json")
            for key in ("backend", "model_path", "dtype", "device", "quantization", "max_context"):
                if parent_config.get(key) != child_config.get(key):
                    raise ValueError(f"Candidate changed fixed engine field: {key}")
            release = freeze(workspace, self.paths, stable, self.config["runtime_python"],
                             self.config["config_hash"], [str(folder)])
            version = release["version_id"]
            write_json(folder / "candidate.json", release)
            self.database.iteration(self.campaign_id, number, {"candidate": version})
            self.state("CANDIDATE_FROZEN")
            self.stop_engine()
            self.state("BUILDING")
            write_json(folder / "build.json", build(self.paths.release(version)))
            self.state("VALIDATING")
            client = self.start_engine(version, folder / "logs" / "correctness.log")
            correctness = evaluate_correctness(client, load_reference(self.directory / "reference.json"),
                                                self.config["evaluation"])
            write_json(folder / "correctness" / "report.json", correctness)
            self.stop_engine()
            if correctness["passed"]:
                self.state("BENCHMARKING")
                for attempt in range(self.config["evaluation"]["max_retests"] + 1):
                    baseline, challenger = self.benchmark(stable, version, folder)
                    result = decide(correctness, baseline, challenger, self.config["evaluation"])
                    decision = result.to_dict()
                    write_json(folder / "benchmarks" / f"summary-{attempt}.json",
                               {"baseline": baseline, "candidate": challenger, "decision": decision})
                    self.state("DECIDING")
                    if result.outcome != "invalid" and result.reason != "no_significant_gain":
                        break
                    if attempt < self.config["evaluation"]["max_retests"]:
                        self.state("BENCHMARKING")
            else:
                self.state("DECIDING")
                decision = {"outcome": "reject", "reason": "correctness_failed"}
            write_json(folder / "decision.json", decision)
            self.database.iteration(self.campaign_id, number, {"decision": decision})
            self.events.emit("admission_decision", iteration_id=number, payload=decision)
            if decision["outcome"] == "accept":
                self.activate(version, folder)
            else:
                self.state("ACTIVATING")
                self.restore(decision["reason"], folder)
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
            extract_draft(self.paths, self.campaign_id, row, self.database.get(self.campaign_id).get("profile_hash"))
            state = self.database.get(self.campaign_id)
            self.database.update(self.campaign_id, {
                "completed_iterations": number, "analysis_pending": True,
                "consecutive_failures": state.get("consecutive_failures", 0) + 1 if failure else 0,
            }, "iteration_completed", iteration_id=number, payload={"decision": decision})
            self.events.sync()
            write_campaign_report(self.database, self.paths, self.campaign_id)

    def run(self, resume=False):
        self.config = read_json(self.directory / "manifest.json")
        if self.config["config_hash"] != digest({k: v for k, v in self.config.items() if k != "config_hash"}):
            raise ValueError("Frozen campaign configuration was modified")
        with exclusive_lock(self.paths.data / "state" / f"{self.campaign_id}.lock"):
            with exclusive_lock(self.paths.data / "state" / "device.lock"):
                try:
                    if resume:
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
                        self.database.update(self.campaign_id, {"state": "READY"}, "stable_recovered")
                        self.events.sync()
                    while not stop_reason(self.database.get(self.campaign_id), self.config["budget"]):
                        number = self.database.get(self.campaign_id)["completed_iterations"] + 1
                        self.run_iteration(number)
                    reason = stop_reason(self.database.get(self.campaign_id), self.config["budget"])
                    state = self.database.get(self.campaign_id)
                    if state.get("analysis_pending"):
                        try:
                            version = state["stable_ref"]
                            client = self.start_engine(version, self.directory / "final-analysis" / "engine.log")
                            self.agent_session(client, self.paths.release(version),
                                               state["completed_iterations"] + 1, analysis_only=True)
                        except Exception as error:
                            self.events.emit("final_analysis_failed", payload={"error": str(error)})
                    self.state("STOPPED", stop_reason=reason)
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
