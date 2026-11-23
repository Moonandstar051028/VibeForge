from types import SimpleNamespace
from copy import deepcopy

import pytest

from vibeforge_v2.harness import sampling_contract, compare
from test_agent_loop_fixes import _engine_report


def test_probe_observes_real_generation_and_restores_method():
    model = SimpleNamespace(generation_config=SimpleNamespace(do_sample=False), generate=lambda **kw: kw)
    original = model.generate
    engine = SimpleNamespace(MODEL=model)
    engine.generate = lambda prompt, max_new_tokens, generation_options: model.generate(
        do_sample=bool(generation_options and generation_options['temperature'] > 0))
    assert sampling_contract(engine) == {'default': [False], 'greedy': [False], 'sampled': [True]}
    assert model.generate is original
    def fail(*args, **kwargs):
        raise ValueError('request failed')
    engine.generate = fail
    with pytest.raises(ValueError):
        sampling_contract(engine)
    assert model.generate is original


def test_identical_logits_do_not_excuse_changed_sampling():
    stable = {dtype: {**_engine_report([10., -3., 2.]), 'sampling_contract':
                     {'default': [False], 'greedy': [False], 'sampled': [True]}}
              for dtype in ('float32', 'bfloat16')}
    assert not compare(stable, deepcopy(stable))[0]
    candidate = deepcopy(stable)
    for report in candidate.values():
        report['sampling_contract']['default'] = [True]
    errors, _, _ = compare(stable, candidate)
    assert len(errors) == 2 and all('sampling contract changed for default' in e for e in errors)


def test_old_harness_verdict_is_rechecked(tmp_path, monkeypatch):
    import json
    from vibeforge_v2 import harness, agent
    workspace = tmp_path / 'engine'
    workspace.mkdir()
    (workspace / 'engine.py').write_text('x = 1\n')
    calls = []
    monkeypatch.setattr(harness, 'run', lambda *a, **k: calls.append(1) or {
        'status': 'PASS', 'passed': True, 'errors': [], 'warnings': [], 'facts': {}, 'scope': 'test', 'wall_seconds': 0})
    config = {'agent': {}}
    folder = tmp_path / 'agent'
    report = agent.run_harness(config, workspace, folder, None)
    report.pop('check_version')
    (folder / 'harness.json').write_text(json.dumps(report))
    assert agent.run_harness(config, workspace, folder, None)['check_version'] == harness.CHECK_VERSION
    assert len(calls) == 2
