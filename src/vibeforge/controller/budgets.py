"""Persisted campaign limits with restart-independent elapsed time."""

import time


def stop_reason(state, budget):
    if state.get("stop_requested"):
        return "stop_requested"
    if state.get("completed_iterations", 0) >= budget["max_iterations"]:
        return "max_iterations"
    if time.time() - state["started_at"] >= budget["max_seconds"]:
        return "max_seconds"
    if state.get("consecutive_failures", 0) >= budget["max_failures"]:
        return "max_failures"
    return None
