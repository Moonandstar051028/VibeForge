from vibeforge.agent.analysis import analysis_attribution_error


def test_analysis_attribution_checks_named_iterations_without_blocking_comparisons():
    previous = {"iteration": 24}
    assert analysis_attribution_error(previous, "Iteration 26 was unfinished.")
    assert analysis_attribution_error(previous, "第 26 轮没有完成。")
    assert analysis_attribution_error(previous, "No previous iteration.")
    assert analysis_attribution_error(previous, "Iteration 24: no change; unlike iteration 26.") is None
    assert analysis_attribution_error(previous, "The previous iteration (24) had no change, unlike iteration 26.") is None
    assert analysis_attribution_error(previous, "Iteration （24） had no change.") is None
    assert analysis_attribution_error(previous, "Iteration (26) was unfinished.")
    assert analysis_attribution_error(previous, "No source change was evaluated.") is None
    assert analysis_attribution_error(None, "No previous iteration.") is None


def test_placeholder_check_allows_evidence_quotes_but_rejects_placeholder_answers():
    from vibeforge.agent.analysis import is_placeholder_analysis
    for text in ("Analysis pending.", "No causal explanation has been established.",
                 "Iteration 1: Analysis pending. No causal explanation has been established."):
        assert is_placeholder_analysis(text)
    assert not is_placeholder_analysis('Iteration 1: rejected for interrupted_iteration; analysis.md says "Analysis pending. No causal explanation has been established." No candidate or paired measurements exist.')


def test_no_formal_comparison_cannot_become_a_quantified_candidate_regression():
    from vibeforge.agent.analysis import performance_analysis_error
    previous = {"iteration": 3, "decision": {"reason": "correctness_failed"}}
    bad = "The observation data shows the candidate actually performed worse: throughput decreased from 5.096 to 4.709 tokens/sec."
    assert performance_analysis_error(previous, bad)
    assert performance_analysis_error(previous, "第3轮候选实测吞吐下降7.6%。")
    assert performance_analysis_error(previous, "No gain was established.", {"explanation": bad})
    for text in ["Iteration 3: correctness failed. No performance gain can be measured.",
                 "The hypothesis predicted a 10% throughput increase based on measured decode time of 5.166s.",
                 "Maximum logit error increased to 2.27; this is correctness evidence, not measured performance."]:
        assert performance_analysis_error(previous, text) is None
    assert performance_analysis_error({"decision": {"reason": "measured_gain", "gain": .1}}, bad) is None


def test_candidate_metric_label_cannot_be_added_to_a_stable_profiler_timing():
    from vibeforge.agent.analysis import performance_analysis_error
    previous = {"iteration": 3, "decision": {"reason": "correctness_failed"}}
    bad = "The observation.json contains stable parent metrics (throughput 5.096 tokens/sec) and candidate metrics (throughput 4.709 tokens/sec). However, formal comparison is unavailable."
    assert performance_analysis_error(previous, bad)
    assert performance_analysis_error(previous, "The predicted candidate throughput increase was 10%; no candidate performance measurement exists.") is None


def test_completed_pairing_cannot_be_described_as_absent():
    from vibeforge.agent.analysis import performance_analysis_error
    previous = {
        "iteration": 8,
        "decision": {"outcome": "invalid", "reason": "profile_regression_not_excluded:concurrent_2",
                     "gain": None},
        "experiment_verification": {"gain": 0.0017, "lower": -0.016, "upper": 0.020,
                                    "pairs": 5, "method": "paired_run_bootstrap"},
    }
    error = performance_analysis_error(
        previous, "Iteration 8: no formal paired benchmark comparison was completed; candidate performance was unmeasured.")
    assert error and "completed formal paired comparison" in error
    assert performance_analysis_error(
        previous, "Iteration 8: verification.json reports a paired gain of 0.17% with CI crossing zero.") is None
