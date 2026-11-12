"""Bounded in-memory aggregation for live Agent inference requests."""

import math
import threading
import time
from collections import Counter


def _percentile(values, fraction):
    values = sorted(value for value in values if isinstance(value, (int, float)) and math.isfinite(value))
    if not values:
        return None
    index = min(len(values) - 1, max(0, math.ceil(fraction * len(values)) - 1))
    return values[index]


class SessionTelemetry:
    def __init__(self):
        self.lock = threading.Lock()
        self.sessions = {}

    def start(self, session_id, max_requests, retention_seconds):
        if not isinstance(session_id, str) or not session_id or len(session_id) > 128:
            raise ValueError("Invalid telemetry session ID")
        if type(max_requests) is not int or not 1 <= max_requests <= 4096:
            raise ValueError("Invalid telemetry request budget")
        now = time.time()
        with self.lock:
            self._prune(now)
            self.sessions[session_id] = {
                "started_at": now, "updated_at": now, "max_requests": max_requests,
                "retention_seconds": retention_seconds, "requests": [], "dropped_requests": 0,
            }
        return self.report(session_id)

    def record(self, observation):
        now = time.time()
        with self.lock:
            self._prune(now)
            for session in self.sessions.values():
                session["updated_at"] = now
                if len(session["requests"]) >= session["max_requests"]:
                    session["dropped_requests"] += 1
                    continue
                session["requests"].append({**observation, "completed_at": now})

    def report(self, session_id):
        with self.lock:
            session = self.sessions.get(session_id)
            if session is None:
                raise ValueError("Unknown or expired telemetry session")
            requests = list(session["requests"])
            metadata = {key: session[key] for key in (
                "started_at", "updated_at", "max_requests", "retention_seconds", "dropped_requests")}
        walls = [row.get("backend_wall_seconds") for row in requests]
        queues = [row.get("queue_seconds") for row in requests]
        ttfts = [row.get("ttft_seconds") for row in requests]
        tpots = [row.get("tpot_seconds") for row in requests]
        output_tokens = sum(row.get("output_tokens") or 0 for row in requests)
        total_wall = sum(value for value in walls if isinstance(value, (int, float)))
        phases = {}
        for row in requests:
            for key, value in (row.get("phases") or {}).items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    phases.setdefault(key, []).append(value)
        return {
            "schema_version": 1, "kind": "agent_inference_telemetry",
            "session_id": session_id, **metadata, "request_count": len(requests),
            "operations": dict(Counter(row.get("operation", "unknown") for row in requests)),
            "protocols": dict(Counter(row.get("protocol", "native") for row in requests)),
            "input_tokens": sum(row.get("input_tokens") or 0 for row in requests),
            "output_tokens": output_tokens, "backend_wall_seconds": total_wall,
            "aggregate_output_tokens_per_second": output_tokens / total_wall if total_wall > 0 else None,
            "latency_seconds": {
                "p50": _percentile(walls, 0.50), "p95": _percentile(walls, 0.95),
                "max": max((value for value in walls if isinstance(value, (int, float))), default=None),
            },
            "queue_seconds": {"p50": _percentile(queues, 0.50), "p95": _percentile(queues, 0.95)},
            "ttft_seconds": {"p50": _percentile(ttfts, 0.50), "p95": _percentile(ttfts, 0.95)},
            "tpot_seconds": {"p50": _percentile(tpots, 0.50), "p95": _percentile(tpots, 0.95)},
            "phase_seconds": {key: {"p50": _percentile(values, 0.50),
                                            "p95": _percentile(values, 0.95)}
                              for key, values in phases.items() if key.endswith("_seconds")},
            "latest_request": requests[-1] if requests else None,
            "admission_role": "diagnostic_only",
        }

    def finish(self, session_id):
        report = self.report(session_id)
        with self.lock:
            self.sessions.pop(session_id, None)
        report["finished_at"] = time.time()
        return report

    def _prune(self, now):
        expired = [session_id for session_id, value in self.sessions.items()
                   if now - value["updated_at"] > value["retention_seconds"]]
        for session_id in expired:
            self.sessions.pop(session_id, None)
