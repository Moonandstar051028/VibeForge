"""Substituting an attention backend cannot verify the requested backend."""
import json
from pathlib import Path

import pytest

from vibeforge_v2 import harness
from vibeforge_v2.library import verified


@pytest.mark.parametrize('overridden_side', [None, 'stable', 'candidate', 'both'])
@pytest.mark.parametrize('math_error', [False, True])
def test_attention_substitution_is_unverified_without_hiding_math_errors(
        tmp_path, monkeypatch, overridden_side, math_error):
    stable, candidate = tmp_path / 'stable', tmp_path / 'candidate'
    for side, directory in (('stable', stable), ('candidate', candidate)):
        directory.mkdir()
        attention = 'flash_attention_2' if overridden_side in (side, 'both') else 'sdpa'
        # A CPU-testable config change avoids the unrelated untraced-source case.
        (directory / 'engine.json').write_text(json.dumps({
            'attn_implementation': attention, 'num_threads': 8 if side == 'stable' else 4}))
        (directory / 'engine.py').write_text('def generate():\n    return 1\n')

    def cpu_result(directory, *args, **kwargs):
        side = 'candidate' if Path(directory) == candidate else 'stable'
        report = {'ok': True, 'logits': [1., 3. if math_error and side == 'candidate' else 2.],
                  'decode_seconds_per_token': .01, 'parameter_bytes': 1,
                  'generate': {'short': {'token_ids': [1]}}, 'replaced_forwards': {},
                  'executed_lines': [['engine.py', 2]]}
        if overridden_side in (side, 'both'):
            report['cpu_config_overrides'] = {'attn_implementation': {
                'requested': 'flash_attention_2', 'tested': 'sdpa'}}
        return report

    monkeypatch.setattr(harness, 'build_reduced_model', lambda *args: tmp_path)
    monkeypatch.setattr(harness, 'run_engine', cpu_result)
    report = harness.run(candidate, stable=stable, model_path='m', cache=tmp_path / 'cache')
    if math_error:
        assert report['status'] == 'FAIL' and not report['allow_gpu_evaluation']
        assert any('logits differ' in error for error in report['errors'])
    elif overridden_side:
        assert report['status'] == 'SKIP' and report['allow_gpu_evaluation']
        assert not report['passed'] and not verified(report)
    else:
        assert report['status'] == 'PASS' and report['passed']
    if overridden_side:
        text = harness.describe(report)
        assert 'flash_attention_2' in text and 'sdpa' in text
        assert 'Only the substituted CPU path was tested' in text
        assert 'require host GPU evaluation' in text
        expected_sides = {'stable', 'candidate'} if overridden_side == 'both' else {overridden_side}
        assert {item['side'] for item in report['facts']['cpu_config_overrides']} == expected_sides
    else:
        assert 'cpu_config_overrides' not in report['facts']
