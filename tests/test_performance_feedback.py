import pytest

from vibeforge_v2.performance_feedback import compare_profiles


def test_feedback_distinguishes_aggregate_request_and_decode_token_rates():
    def profile(rate, latency, tpot):
        return {'throughput': rate, 'metrics': {'latency_mean_seconds': latency,
                'ttft_seconds_mean': .2, 'tpot_seconds_mean': tpot},
                'requests': [{'tokens': 20, 'latency_seconds': latency}]}
    baseline = [{'metrics': {'throughput': 40}, 'profile_records': {
        'single': profile(20, 1, .025), 'concurrent': profile(80, 2, .05)}}]
    candidate = [{'metrics': {'throughput': 60}, 'profile_records': {
        'single': profile(30, .8, .02), 'concurrent': profile(120, 1.5, .03)}}]
    report = compare_profiles(baseline, candidate)
    row = report['profiles']['concurrent']
    assert row['baseline']['output_tokens_per_second'] == 80
    assert row['baseline']['request_output_tokens_per_second_mean'] == 10
    assert row['baseline']['decode_tokens_per_second'] == 20
    assert row['relative_change']['output_tokens_per_second'] == pytest.approx(.5)
    assert row['relative_change']['latency_mean_seconds'] == pytest.approx(-.25)
    # Missing telemetry stays unknown rather than becoming a claimed measurement.
    missing = compare_profiles([{'throughput': 10}], [{'throughput': 12}])
    assert missing['profiles']['default']['baseline']['decode_tokens_per_second'] is None
    assert 'not pooled' in report['metric_definitions']['composite_throughput_score']
