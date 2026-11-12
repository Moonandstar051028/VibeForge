"""Reproducible summaries from committed state and per-iteration decisions."""

from collections import Counter
from vibeforge.storage.atomic import atomic_write, write_json, read_json


def write_campaign_report(database, paths, campaign_id):
    state = database.get(campaign_id)
    rows = database.iterations(campaign_id)
    accepted = [row for row in rows if (row.get("decision") or {}).get("outcome") == "accept"]
    measured = [row for row in rows if (row.get("experiment_verification") or {}).get("pairs", 0) > 0]
    valid = [row for row in rows if row.get("experiment_id")]
    metrics = {
        "attempts": len(rows), "valid_candidates": len(valid), "measured_candidates": len(measured),
        "accepted_candidates": len(accepted),
        "valid_candidate_rate": len(valid) / len(rows) if rows else None,
        "acceptance_rate": len(accepted) / len(rows) if rows else None,
        "reasons": dict(Counter((row.get("decision") or {}).get("reason", "unfinished") for row in rows)),
        "initial_ref": state.get("initial_ref"), "stable_ref": state.get("stable_ref"),
        "cumulative_comparison": None,
    }
    comparison_path = paths.campaign(campaign_id) / "cumulative-comparison.json"
    if comparison_path.is_file():
        comparison = read_json(comparison_path)
        if (comparison.get("stable_ref") == state.get("stable_ref")
                and comparison.get("initial_ref") == state.get("initial_ref")):
            metrics["cumulative_comparison"] = {key: comparison.get(key) for key in ("decision", "evidence_path")}
    write_json(paths.campaign(campaign_id) / "progress.json", metrics)
    text = (
        f"# Campaign {campaign_id}\n\n"
        f"- State: {state['state']}\n- Mode: {state.get('mode')}\n"
        f"- Stable version: {state.get('stable_ref')}\n- Stop reason: {state.get('stop_reason')}\n"
        f"- Profile: {state.get('profile_hash')}\n- Analysis pending: {state.get('analysis_pending')}\n\n"
        f"- Proposal attempts: {state.get('proposal_attempts', state.get('completed_iterations', 0))}\n"
        f"- Valid candidates: {state.get('valid_candidates', 'legacy: unavailable')}\n"
        f"- Completed experiments: {state.get('completed_experiments', 'legacy: unavailable')}\n\n"
        "| Attempt | Experiment | Candidate | Precision | Objective | Outcome | Reason | Measured gain |\n| --- | --- | --- | --- | --- | --- | --- | --- |\n"
    )
    for row in rows:
        decision = row.get("decision") or {}
        gain = decision.get("gain")
        precision = row.get("precision", {})
        label = f"{precision.get('compute_dtype', '?')}/{precision.get('quantization', '?')}"
        gain_label = f"{gain:.2%} [{decision['lower']:.2%}, {decision['upper']:.2%}]" if gain is not None else 'not measured'
        text += f"| {row['iteration']} | {row.get('experiment_id', '-')} | {row.get('candidate')} | {label} | {decision.get('objective', '?')} | {decision.get('outcome')} | {decision.get('reason', row.get('error'))} | {gain_label} |\n"
    text += f"\nAccepted candidates: {len(accepted)}; formally measured: {len(measured)}; valid candidates: {len(valid)}/{len(rows)} attempts.\n"
    if metrics["cumulative_comparison"]:
        comparison = metrics["cumulative_comparison"]
        decision = comparison["decision"]
        gain = decision.get("gain")
        if gain is not None:
            text += (f"\nFresh final-versus-initial gain: **{gain:.2%}**, interval "
                     f"[{decision['lower']:.2%}, {decision['upper']:.2%}]; {decision['reason']}. "
                     f"Evidence: `{comparison['evidence_path']}/report.json`.\n")
        else:
            text += f"\nFinal comparison: {decision['reason']}; see `{comparison['evidence_path']}/report.json`.\n"
    else:
        text += f"\nCumulative gain has not been remeasured. After stopping, run `vibeforge compare {campaign_id}`. Do not add per-round gains.\n"
    text += "\nEach decision links to its raw samples and logs in iterations/<number>/. Toy fixture performance is not Qwen model performance.\n"
    path = paths.campaign(campaign_id) / "summary.md"
    atomic_write(path, text)
    return path
