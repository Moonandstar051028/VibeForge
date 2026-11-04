"""Reproducible summaries from committed state and per-iteration decisions."""

from vibeforge.storage.atomic import atomic_write


def write_campaign_report(database, paths, campaign_id):
    state = database.get(campaign_id)
    rows = database.iterations(campaign_id)
    text = (
        f"# Campaign {campaign_id}\n\n"
        f"- State: {state['state']}\n- Mode: {state.get('mode')}\n"
        f"- Stable version: {state.get('stable_ref')}\n- Stop reason: {state.get('stop_reason')}\n"
        f"- Profile: {state.get('profile_hash')}\n- Analysis pending: {state.get('analysis_pending')}\n\n"
        "| Iteration | Candidate | Precision | Objective | Outcome | Reason | Measured gain |\n| --- | --- | --- | --- | --- | --- | --- |\n"
    )
    for row in rows:
        decision = row.get("decision") or {}
        gain = decision.get("gain")
        precision = row.get("precision", {})
        label = f"{precision.get('compute_dtype', '?')}/{precision.get('quantization', '?')}"
        text += f"| {row['iteration']} | {row.get('candidate')} | {label} | {decision.get('objective', '?')} | {decision.get('outcome')} | {decision.get('reason', row.get('error'))} | {gain if gain is not None else 'not measured'} |\n"
    text += "\nEach decision links to its raw samples and logs in iterations/<number>/. Toy fixture performance is not Qwen model performance.\n"
    path = paths.campaign(campaign_id) / "summary.md"
    atomic_write(path, text)
    return path
