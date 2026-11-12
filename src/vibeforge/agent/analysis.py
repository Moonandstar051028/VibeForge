"""Traceable factual summaries; interpretation is explicitly distinguished."""

import re

from vibeforge.storage.atomic import atomic_write, canonical_json


def is_placeholder_analysis(text):
    """Reject a placeholder answer, while allowing it to be quoted as evidence."""
    value = re.sub(r"\s+", " ", (text or "").strip()).casefold()
    value = re.sub(r"^iteration\s*#?\s*\d+\s*[:.-]\s*", "", value)
    value = value.strip(" .\"'")
    return not value or bool(re.fullmatch(
        r"(?:analysis pending[.;]?\s*)?(?:no causal explanation has been established[.;]?)?", value))


def performance_analysis_error(previous, interpretation, reflection=None):
    """Reject quantified performance conclusions for an experiment stopped before A/B.

    This catches unsupported comparative claims, not arbitrary semantic errors.
    Forecasts and correctness/logit numbers remain valid review evidence.
    """
    previous = previous or {}
    text = " ".join([interpretation or "", (reflection or {}).get("explanation", "")])
    verification = previous.get("experiment_verification") or {}
    if (isinstance(verification.get("gain"), (float, int))
            and isinstance(verification.get("pairs"), int) and verification["pairs"] > 0):
        denied = re.search(
            r"(?:no|without|not|was(?:n't| not)|unavailable|incomplete|absent|未|没有|并未|不可用|未完成|缺失|不存在)"
            r".{0,55}(?:formal (?:paired )?(?:comparison|benchmark)|paired (?:comparison|benchmark)|"
            r"performance (?:comparison|measurement)|measured throughput|正式(?:配对)?(?:比较|基准)|"
            r"配对(?:比较|基准)|性能(?:比较|测量)|吞吐(?:比较|测量))"
            r"|(?:formal (?:paired )?(?:comparison|benchmark)|paired (?:comparison|benchmark)|"
            r"正式(?:配对)?(?:比较|基准)|配对(?:比较|基准)).{0,55}"
            r"(?:unavailable|incomplete|absent|not (?:available|completed)|不可用|未完成|缺失|不存在)",
            text, re.I,
        )
        if denied:
            return (f"Iteration {previous.get('iteration')} has a completed formal paired comparison in "
                    f"experiment_verification ({verification['pairs']} pairs). Cite verification.json or the "
                    "benchmark summary; do not claim that candidate performance was unmeasured or that the paired benchmark was absent/incomplete.")
    reason = (previous.get("decision") or {}).get("reason")
    if reason not in {"correctness_failed", "quality_failed", "execution_failed", "interrupted_iteration",
                      "recovered_incomplete_iteration", "agent_protocol_error"}:
        return None
    metric = r"throughput|tokens?\s*/\s*(?:s|sec)|tokens?\s+per\s+second|tpot|ttft|latency|speedup|decode_forward_seconds|prefill_forward_seconds|吞吐|延迟|耗时|性能|速度"
    direction = r"decreas\w*|increas\w*|improv\w*|degrad\w*|regress\w*|reduc\w*|worse|faster|slower|gain|下降|提高|提升|降低|增加|减少|退化|加速"
    for sentence in re.split(r"(?<=[.!?。！？])\s+|\n", text):
        forecast = re.search(r"predict|expect|forecast|hypothesi|propos|预计|预测|假设|预期", sentence, re.I)
        actual = re.search(r"observed performance|candidate actually|actual measurements|measured (?:gain|regression)|candidate measured|实测(?:性能|提升)|实际(?:性能|提速)", sentence, re.I)
        if forecast and not actual:
            continue
        quantified = (rf"(?:{metric}).{{0,80}}?(?:{direction}).{{0,40}}?\d"
                      rf"|(?:{metric}).{{0,40}}?\d.{{0,40}}?(?:{direction})"
                      rf"|(?:{direction}).{{0,30}}?\d.{{0,30}}?(?:{metric})")
        candidate_measurement = re.search(
            rf"(?:candidate|候选).{{0,60}}?(?:{metric}).{{0,35}}?\d", sentence, re.I)
        if re.search(quantified, sentence, re.I) or candidate_measurement:
            return (f"Iteration {(previous or {}).get('iteration')} has no formal candidate performance comparison "
                    f"because {reason}. observation.json is stable-parent evidence, not candidate performance. "
                    "Remove candidate metric values and quantified gain/regression claims; normal_request and timing are both stable-parent runs (ordinary versus profiled). Retain correctness facts and label candidate performance unmeasured.")
    return None


def analysis_attribution_error(previous, interpretation):
    """Reject an analysis that explicitly names only other iterations."""
    target = (previous or {}).get("iteration")
    if target is None:
        return None
    mentions = {int(number) for match in re.findall(
        r"\biteration\s*[#(（]?\s*(\d+)\b|第\s*(\d+)\s*轮", interpretation or "", re.IGNORECASE
    ) for number in match if number}
    if mentions and target not in mentions:
        return (f"previous_analysis is assigned to iteration {target}, but refers only to iterations "
                f"{sorted(mentions)}. Analyze context.previous (iteration {target}) explicitly; "
                "other history entries are supporting evidence only.")
    if not mentions and (interpretation or "").strip().casefold() in {
        "no previous iteration.", "no previous iteration", "this is the initial iteration.",
    }:
        return f"Iteration {target} exists and requires analysis; this is not the initial iteration."
    return None


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
