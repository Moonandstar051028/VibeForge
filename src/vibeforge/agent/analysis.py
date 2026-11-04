"""Traceable factual summaries; interpretation is explicitly distinguished."""

from vibeforge.storage.atomic import atomic_write, canonical_json


def factual_analysis(iteration, interpretation=None):
    return (
        f"# Iteration {iteration['iteration']}\n\n"
        "## Recorded facts\n\n"
        f"- Candidate: {iteration.get('candidate')}\n"
        f"- Decision: {canonical_json(iteration.get('decision'))}\n"
        f"- Error: {iteration.get('error')}\n\n"
        "## Interpretation\n\n"
        + (interpretation or "Analysis pending. No causal explanation has been established.")
        + "\n"
    )


def write_analysis(path, iteration, interpretation):
    atomic_write(path, factual_analysis(iteration, interpretation))
