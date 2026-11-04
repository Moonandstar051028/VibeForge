import pytest
from vibeforge.controller.admission import decide
from vibeforge.evaluation.statistics import paired_gain
from vibeforge.evaluation.correctness import compare_values


def rows(values, valid=True, latency=0.1):
    return [{"throughput": value, "valid": valid, "requests": [{"ttft_seconds": latency}]} for value in values]


@pytest.mark.parametrize("baseline,candidate,outcome,reason", [
    ([10, 10, 10], [15, 15, 15], "accept", "measured_gain"),
    ([10, 10, 10], [8, 8, 8], "reject", "performance_regression"),
    ([10, 10, 10], [10, 10, 10], "reject", "no_significant_gain"),
])
def test_admission_direction_and_threshold(baseline, candidate, outcome, reason):
    result = decide({"passed": True}, rows(baseline), rows(candidate), {"min_gain": 0.02})
    assert (result.outcome, result.reason) == (outcome, reason)


def test_incorrect_or_contaminated_candidate_never_promotes():
    assert decide({"passed": False}, [], [], {"min_gain": 0}).reason == "correctness_failed"
    assert decide({"passed": True}, rows([10, 10]), rows([20, 20], valid=False),
                  {"min_gain": 0}).outcome == "invalid"


def test_protection_metric_overrides_throughput():
    result = decide({"passed": True}, rows([10, 10]), rows([20, 20], latency=5),
                    {"min_gain": 0, "absolute_limits": {"ttft_seconds": 1}})
    assert result.reason == "protection_failed:ttft_seconds"


@pytest.mark.parametrize("baseline,candidate", [([1], [1]), ([1, 2], [1]), ([1, 0], [1, 2]),
                                              ([1, 2], [float("nan"), 2])])
def test_invalid_statistics_are_rejected(baseline, candidate):
    with pytest.raises(ValueError):
        paired_gain(baseline, candidate)


def test_numeric_comparison_rejects_nonfinite_values():
    assert compare_values([1.001], [1], 0.01, 0)
    assert not compare_values([float("nan")], [1], 1, 1)
    assert not compare_values([1, 2], [1], 0, 0)
