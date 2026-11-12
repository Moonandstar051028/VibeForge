"""Persisted campaign limits with restart-independent elapsed time."""

import time


def stop_reason(state, budget):
    if state.get("stop_requested"):
        return "stop_requested"
    if state.get("completed_experiments", state.get("completed_iterations", 0)) >= budget["max_iterations"]:
        return "max_iterations"
    if state.get("proposal_attempts", 0) >= budget.get("max_proposals", budget["max_iterations"] * 3):
        return "max_proposals"
    if state.get("stalled_attempts", 0) >= budget.get("max_stalled_attempts", 3):
        return "max_stalled_attempts"
    if time.time() - state["started_at"] >= budget["max_seconds"]:
        return "max_seconds"
    if state.get("consecutive_failures", 0) >= budget["max_failures"]:
        return "max_failures"
    if state.get("consecutive_candidate_failures", 0) >= budget.get("max_candidate_failures", 5):
        return "max_candidate_failures"
    return None
