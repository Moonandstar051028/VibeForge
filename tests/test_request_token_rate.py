"""Client-clock request speed must drive both admission and profile protection."""
from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace

import pytest

from vf_runtime.controller.admission import decide
from vf_runtime.evaluation import performance as module
from vibeforge_v2.config import ROOT, load


METRIC = 'request_output_tokens_per_second_mean'


def test_measure_averages_each_request_rate_not_aggregate_or_mean_latency(tmp_path, monkeypatch):
    # Two requests: 8 tokens / 2s and 8 tokens / 4s. Total window is 10s.
    clock = iter([0, 1, 3, 4, 8, 10])
    monkeypatch.setattr(module.time, 'perf_counter', lambda: next(clock))
    monkeypatch.setattr(module, 'sample', lambda: {})
    monkeypatch.setattr(module, 'compare_generation', lambda *args: True)
    monkeypatch.setattr(module, 'ResourceSampler', lambda *args: nullcontext(
        SimpleNamespace(peak=1, interval=.1, samples=1)))
    device = nullcontext()
    device.report = lambda: {'status': 'disabled', 'devices': []}
    monkeypatch.setattr(module, 'DeviceTelemetrySampler', lambda *args, **kwargs: device)
    monkeypatch.setattr(module, 'ThreadPoolExecutor', lambda **kwargs: nullcontext(
        SimpleNamespace(map=lambda function, jobs: map(function, jobs))))
    client = SimpleNamespace(generate=lambda **kwargs: {'input_ids': [1], 'token_ids': [2] * 8})
    workload = {'prompts': ['p'], 'max_new_tokens': 8, 'requests_per_prompt': 2}
    record = module.measure(client, workload, tmp_path/'samples.jsonl', 'baseline', 0,
        {'objective': METRIC, 'device_backend': 'cpu'},
        reference={'benchmark_cases': [{'prompt': 'p', 'max_new_tokens': 8}]})
    assert record['valid']
    assert record['metrics'][METRIC] == pytest.approx(3)
    assert record['throughput'] == pytest.approx(1.6)
    assert record['metrics'][METRIC] != pytest.approx(8 / 3)
    assert record['measurement_mode'] == 'wall'


def test_suite_uses_weighted_geometric_mean_of_request_speeds(tmp_path, monkeypatch):
    profiles = [{'id': 'short', 'weight': 1}, {'id': 'long', 'weight': 3}]
    def measure(client, profile, *args, **kwargs):
        rate = {'short': 4, 'long': 16}[profile['id']]
        return {'profile_id': profile['id'], 'metrics': {METRIC: rate, 'throughput': 100},
                'throughput': 100, 'valid': True, 'invalid_reasons': [], 'requests': []}
    monkeypatch.setattr(module, 'measure', measure)
    result = module.measure_suite(None, {'profiles': profiles}, tmp_path/'samples.jsonl',
                                  'baseline', 0, {'objective': METRIC})
    assert result['valid']
    assert result['metrics'][METRIC] == pytest.approx((4 * 16**3)**.25)
    assert result['metrics']['throughput'] == pytest.approx(100)


def suite(rate, throughput=10, profiles=None):
    profiles = profiles or {'short': rate, 'long': rate}
    return {'valid': True, 'metrics': {METRIC: rate, 'throughput': throughput},
            'profile_weights': {'short': 1, 'long': 1},
            'profile_records': {name: {'metrics': {METRIC: value}, 'throughput': throughput}
                                for name, value in profiles.items()}}


@pytest.mark.parametrize('baseline,candidate,outcome', [
    (suite(10), suite(8, throughput=100), 'reject'),
    (suite(10), suite(12, throughput=5), 'accept'),
    (suite(10), suite(10, throughput=100), 'reject'),
    (suite(10), suite(12, throughput=100, profiles={'short': 8, 'long': 18}), 'reject'),
])
def test_only_request_speed_drives_gain_and_profile_regression(baseline, candidate, outcome):
    result = decide({'passed': True}, [baseline]*5, [candidate]*5,
                    {'objective': METRIC, 'min_gain': .02, 'max_profile_regression': .03})
    assert result.outcome == outcome
    assert result.objective == METRIC


def test_missing_profile_request_speed_cannot_fall_back_to_throughput():
    candidate = deepcopy(suite(12, throughput=100))
    candidate['profile_records']['short']['metrics'].pop(METRIC)
    result = decide({'passed': True}, [suite(10)]*5, [candidate]*5,
                    {'objective': METRIC, 'min_gain': .02})
    assert result.outcome == 'invalid' and result.reason == 'profile_metric_unavailable:short'


def test_qwen38_configuration_fixes_request_speed_objective():
    config = load(ROOT/'configs/qwen3.8-27b-a800.json')
    assert config['evaluation']['objective'] == METRIC
    assert config['evaluation']['measurement_protocol_version'] == 3
    assert config['agent']['related_history'] is False


def test_real_cpu_pipeline_accepts_using_request_speed(tmp_path):
    from test_pipeline import fixture_connection, fixture_edit
    from vibeforge_v2.loop import Pipeline
    from vf_runtime.storage.atomic import read_json
    config = load(ROOT/'configs/toy.json')
    config['evaluation']['objective'] = METRIC
    config['workload']['profiles'] = [
        {'id': 'single', 'prompts': ['alpha'], 'max_new_tokens': 8,
         'concurrency': 1, 'requests_per_prompt': 2, 'weight': 1},
        {'id': 'queued', 'prompts': ['beta'], 'max_new_tokens': 8,
         'concurrency': 2, 'requests_per_prompt': 2, 'weight': 2}]
    config['workload']['observation_profile'] = 'single'
    output = tmp_path/'run'
    state = Pipeline(config, output, edit=fixture_edit, connect=fixture_connection).execute(rounds=1)
    decision = read_json(output/'rounds/0001/decision.json')
    assert decision['outcome'] == 'accept' and decision['objective'] == METRIC
    assert state['stable'] == 'rounds/0001/engine'
