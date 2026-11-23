import copy
import math

import pytest

from vf_runtime.controller.admission import decide
from vf_runtime.evaluation.correctness import compare_generation, evaluate_correctness
from vibeforge_v2.correctness import distribution_report
from vibeforge_v2.quality import paired_quality, loss_upper


def evidence():
    logits = {'indices': [0, 1], 'values': [1., 2.], 'repeat_count': 3}
    case = {'id': 'case', 'prompt': 'hello', 'input_ids': [1], 'token_ids': [2, 3],
            'max_new_tokens': 2, 'fixed_length': True, 'logits': logits, 'continuation_logits': logits}
    return {'cases': [case], 'vocab_size': 10}


class Client:
    def __init__(self, *, nan=False, corrupt=False):
        self.calls = 0
        self.nan, self.corrupt = nan, corrupt

    def generate(self, **kwargs):
        self.calls += 1
        return {'input_ids': [1], 'token_ids': [5, 7 if self.corrupt and self.calls > 2 else 6], 'text': 'answer'}

    def logits(self, **kwargs):
        return {'indices': [0, 1], 'values': [math.nan if self.nan else 3., 4.]}


def test_fidelity_drift_does_not_masquerade_as_request_corruption():
    protocol = {'correctness_mode': 'quality', 'atol': .01, 'rtol': .01, 'concurrency': 2}
    report = evaluate_correctness(Client(), evidence(), protocol)
    assert report['passed'] and report['fidelity_drift'] and report['requires_quality']
    assert not report['diagnostics'][0]['baseline_token_match']
    assert decide(report, [], [], {}).reason == 'independent_quality_unverified'


@pytest.mark.parametrize('settings', [{'nan': True}, {'corrupt': True}])
def test_nonfinite_logits_and_candidate_local_contamination_still_reject(settings):
    report = evaluate_correctness(Client(**settings), evidence(), {'correctness_mode': 'quality'})
    assert not report['passed']


def test_performance_contract_accepts_different_content_but_not_different_work():
    case = evidence()['cases'][0]
    output = Client().generate()
    assert compare_generation(output, case, {'correctness_mode': 'quality'})
    assert not compare_generation(output, case, {'correctness_mode': 'strict'})
    assert not compare_generation({**output, 'token_ids': [5]}, case, {'correctness_mode': 'quality'})
    assert not compare_generation({**output, 'input_ids': [9]}, case, {'correctness_mode': 'quality'})
    assert not compare_generation({**output, 'token_ids': [True, 2]}, case, {'correctness_mode': 'quality'})


def test_distribution_metrics_ignore_constant_logit_shift_and_detect_probability_change():
    a = {'indices': [0, 1], 'values': [1., 2.]}
    shifted = {'indices': [0, 1], 'values': [11., 12.]}
    assert distribution_report(shifted, a)['total_variation'] < 1e-12
    assert distribution_report({'indices': [0, 1], 'values': [2., 1.]}, a)['total_variation'] > .4


def task_report(n, bad=()):
    return {'cases': [{'id': str(i), 'passed': i not in bad} for i in range(n)]}


def test_paired_quality_counts_tasks_not_repeats_and_rejects_small_or_regressed_suite():
    settings = {'min_cases': 100, 'confidence': .95, 'max_loss_rate': .05}
    small = {'runs': [task_report(10)] * 3}
    assert not paired_quality(small, small, settings)['passed']
    large = {'runs': [task_report(150)] * 3}
    good = paired_quality(large, large, settings)
    assert good['passed'] and good['unique_tasks'] == 150
    assert good['loss_rate_upper'] == pytest.approx(1 - .05 ** (1 / 150))
    assert not paired_quality(large, {'runs': [task_report(150, range(8))] * 3}, settings)['passed']
    with pytest.raises(ValueError):
        paired_quality(task_report(150), task_report(149), settings)


def test_task_oracles_execute_saved_code_and_generated_suite_is_reproducible():
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('quality_script', root / 'scripts/quality_cases.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    case = {'entry_point': 'f', 'tests': 'assert f(2) == 4\nassert f(-2) == 4'}
    assert module.execute_function('def f(x):\n    return x*x', case)
    assert not module.execute_function('def f(x):\n    return x+2', case)


def test_real_pipeline_reaches_paired_quality_despite_baseline_numeric_drift(tmp_path):
    import importlib.util
    import json
    from pathlib import Path
    from vf_runtime.storage.atomic import write_json, read_json
    from vibeforge_v2.config import ROOT, load
    from vibeforge_v2.loop import Pipeline
    spec = importlib.util.spec_from_file_location('toy_quality_mode', ROOT / 'engines/toy/engine.py')
    toy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(toy)
    expected = toy.generate('quality-only', max_new_tokens=2)['token_ids']
    config = load(ROOT / 'configs/toy.json')
    config['evaluation'].update(correctness_mode='quality', distribution_diagnostics=False,
        quality={'command': ['{python}', '{project}/tests/quality_fixture.py', '--request', '{request}',
                             '--output', '{output}', '--expected', json.dumps(expected)],
                 'files': [], 'file_hashes': {}, 'repeats': 2, 'timeout_seconds': 10,
                 'min_scores': {'accuracy': 1}, 'max_regressions': {'accuracy': 0},
                 # Deliberately tiny controller fixture, not production statistical settings.
                 'paired': {'min_cases': 1, 'confidence': .95, 'max_loss_rate': .99}})
    def connection(url, config, workspace, folder):
        write_json(Path(folder) / 'connection.json', {'passed': True, 'fixture': True})
    def edit(url, config, workspace, folder, context):
        p = Path(workspace) / 'engine.py'
        p.write_text(p.read_text().replace('/ 997 for index in indices]', '/ 997 + 0.1 for index in indices]'))
    output = tmp_path / 'run'
    Pipeline(config, output, edit=edit, connect=connection).execute(rounds=1)
    report = read_json(output / 'rounds/0001/correctness.json')
    assert report['passed'] and report['fidelity_drift'] and report['quality_passed']
    quality = read_json(output / 'rounds/0001/quality/report.json')
    assert quality['passed'] and quality['paired']['unique_tasks'] == 1
    assert (output / 'rounds/0001/performance/samples.jsonl').exists()
