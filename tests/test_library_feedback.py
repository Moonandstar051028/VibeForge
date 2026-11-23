import json

import pytest

from vibeforge_v2.library import render, update_library, usable, verified


def row(outcome=None, reason=None, gain=None):
    return {'round': '0001', 'replaced': '请求路径上的源码改动', 'path': '/evidence/files',
            'decision_outcome': outcome, 'decision_reason': reason, 'gpu_gain': gain,
            'outcome': f'{outcome}/{reason}', 'fp32_error': 0.0, 'cpu_ratio': 1.0}


@pytest.mark.parametrize('outcome,reason,gain', [
    ('reject', 'no_significant_gain', .0047),
    ('reject', 'no_significant_gain', -.0034),
    ('reject', 'quality_failed', .15),
    (None, None, None),
])
def test_unaccepted_cpu_equivalent_candidate_is_evidence_not_recommendation(outcome, reason, gain):
    evidence = row(outcome, reason, gain)
    text = render([evidence])
    assert not usable(evidence)
    assert '未接受的历史实现' in text
    assert 'cp -r' not in text
    assert '已证实更慢' not in text
    assert '暂无通过正式准入' in text
    if reason:
        assert reason in text


def test_formal_acceptance_controls_reuse():
    accepted = row('accept', 'significant_gain', .12)
    assert usable(accepted)
    text = render([accepted])
    assert '通过了宿主正式准入' in text and 'cp -r' in text


@pytest.mark.parametrize('failed', [{'passed': False}, {'errors': ['sampling contract changed']}])
def test_other_harness_failures_do_not_become_verified_implementations(failed):
    report = {'passed': True, 'errors': [], 'facts': {
        'float32_logit_relative_error': 0., 'changed_lines_executed_on_request_path': [1]}}
    report.update(failed)
    assert not verified(report)


@pytest.mark.parametrize('shared_forward_fingerprint', [False, True])
def test_library_preserves_each_round_and_rebuilds_previously_deduplicated_files(tmp_path, shared_forward_fingerprint):
    run = tmp_path / 'run'
    stable = run / 'initial/engine'
    stable.mkdir(parents=True)
    (stable / 'engine.py').write_text('value = 0\n')
    for number, gain in [(1, .0047), (2, -.0034)]:
        folder = run / f'rounds/{number:04}'
        engine = folder / 'engine'
        engine.mkdir(parents=True)
        (engine / 'engine.py').write_text(f'value = {number}\n')
        (folder / 'parent.json').write_text(json.dumps({'source': 'initial/engine'}))
        (folder / 'decision.json').write_text(json.dumps({
            'outcome': 'reject', 'reason': 'no_significant_gain', 'gain': gain}))
        facts = {'float32_logit_relative_error': 0., 'changed_lines_executed_on_request_path': [1]}
        if shared_forward_fingerprint:
            facts['replaced_forwards'] = {'Moe': {'fingerprint': 'shared', 'original_fingerprint': 'original'}}
        # Old deduplication left harness.json but deleted files/ for later rounds.
        cached = run / f'library/{number:04}'
        cached.mkdir(parents=True)
        (cached / 'harness.json').write_text(json.dumps({'passed': True, 'errors': [], 'facts': facts}))
    for _ in range(2):
        rows = update_library(run, entries=[])
        assert [r['round'] for r in rows] == ['0001', '0002']
        assert [r['gpu_gain'] for r in rows] == [.0047, -.0034]
        assert not any(usable(r) for r in rows)
        for number in (1, 2):
            assert (run / f'library/{number:04}/files/engine.py').read_text() == f'value = {number}\n'
        text = render(rows)
        assert '0001' in text and '0002' in text and 'no_significant_gain' in text
        assert 'cp -r' not in text
