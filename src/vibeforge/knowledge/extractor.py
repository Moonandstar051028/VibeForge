"""Create evidence-linked drafts; a successful iteration is not a general rule."""

from vibeforge.storage.atomic import write_json


def extract_draft(paths, campaign_id, iteration, profile_hash, conditions=None):
    value = {
        "rule_id": f"{campaign_id}-{iteration['iteration']:04d}", "status": "draft",
        "conditions": {"profile_hash": profile_hash, **(conditions or {})}, "hypothesis": iteration.get("hypothesis"),
        "implementation": {"candidate": iteration.get("candidate"), "target_precision": iteration.get("precision"),
                           "diff": f"iterations/{iteration['iteration']:04d}/change.patch"},
        "validation": iteration.get("decision"),
        "evidence": [{"campaign_id": campaign_id, "iteration": iteration["iteration"], "profile_hash": profile_hash,
                      "target_profile_hash": iteration.get("profile_hash")}],
        "limitations": ["single_iteration", "causal_attribution_unconfirmed", "no_transfer_claim"],
    }
    write_json(paths.campaign(campaign_id) / "knowledge-drafts" / f"{iteration['iteration']:04d}.json", value)
    return value
