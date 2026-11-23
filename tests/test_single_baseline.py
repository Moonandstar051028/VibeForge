from pathlib import Path

import pytest
from vibeforge_v2.config import ROOT, load
from vibeforge_v2.loop import Pipeline
from vf_runtime.storage.atomic import read_json
from vf_runtime.evaluation.reference import validate_reference
from test_pipeline import fixture_connection, fixture_edit


@pytest.mark.parametrize('candidate_repeats', [1, 3])
def test_one_pass_quality_preserves_configured_candidate_and_performance_repeats(tmp_path, candidate_repeats):
    import importlib.util
    import json
    spec = importlib.util.spec_from_file_location('single_baseline_toy', ROOT / 'engines/toy/engine.py')
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    expected = engine.generate('quality-only', max_new_tokens=2)['token_ids']
    config = load(ROOT / 'configs/toy.json')
    config['evaluation'].update(reference_repeats=1, min_reference_consensus=1.)
    config['evaluation']['quality'] = {
        'command': ['{python}', '{project}/tests/quality_fixture.py', '--request', '{request}', '--output', '{output}', '--expected', json.dumps(expected)],
        'file_hashes': {}, 'repeats': candidate_repeats, 'baseline_repeats': 1, 'timeout_seconds': 10,
        'min_scores': {'accuracy': 1}, 'max_regressions': {'accuracy': 0}}
    output = tmp_path / 'run'
    Pipeline(config, output, edit=fixture_edit, connect=fixture_connection).execute(rounds=1)
    reference = read_json(output / 'reference.json')
    assert reference['reference_repeats'] == 1 and not reference['repeat_variability_observed']
    validate_reference(reference, config['evaluation'])
    with pytest.raises(ValueError, match='insufficient repeated'):
        validate_reference(reference, {'reference_repeats': 3})
    initial = read_json(output / 'baseline-quality.json')
    assert 'runs' not in initial
    candidate = read_json(output / 'rounds/0001/quality/report.json')
    assert candidate['repeat_count'] == candidate_repeats
    if candidate_repeats == 1:
        assert 'runs' not in candidate
        assert (output / 'rounds/0001/quality/request.json').exists()
        assert not (output / 'rounds/0001/quality/runs').exists()
    else:
        assert len(candidate['runs']) == candidate_repeats
    performance = read_json(output / 'rounds/0001/performance/attempt-0.json')
    assert len(performance['baseline']) == config['evaluation']['repeats']
