"""Keep coverage/activation evidence visible in the next round's feedback excerpt."""
import json

import pytest

from vibeforge_v2.feedback import write_feedback


@pytest.mark.parametrize('status', ['PASS', 'SKIP', 'FAIL'])
def test_feedback_preserves_coverage_scope_and_runtime_comparison(tmp_path, status):
    folder = tmp_path / '0002'
    (folder / 'agent').mkdir(parents=True)
    (folder / 'logs').mkdir()
    (folder / 'agent/harness.json').write_text(json.dumps({
        'status': status, 'passed': status == 'PASS', 'allow_gpu_evaluation': status != 'FAIL',
        'errors': ['import failed'] if status == 'FAIL' else [],
        'warnings': ['CPU excludes CUDA-only branches.'],
        'facts': {'changed_lines_executed_on_request_path': 0, 'unobserved_source_effect': True,
                  'replaced_forwards': {}}}))
    for side, modules in [('stable', {'OriginalAttention': 64}), ('candidate', {'CustomAttention': 64})]:
        (folder / f'logs/{side}.runtime.json').write_text(json.dumps({'runtime': {
            'module_classes': modules, 'attention_implementations': {'model': 'sdpa'}}}))
    write_feedback(folder, {'outcome': 'reject', 'reason': 'performance_regression'})
    text = (folder / 'outcome.md').read_text()
    assert 'changed_lines_executed_on_request_path' in text[:2000]
    assert 'CPU excludes CUDA-only branches.' in text[:2000]
    assert ('SKIP 不是通过' in text) == (status == 'SKIP')
    assert 'OriginalAttention' in text and 'CustomAttention' in text
    assert ('import failed' in text) == (status == 'FAIL')


def test_feedback_bounds_replacement_list_without_losing_count(tmp_path):
    (tmp_path / 'agent').mkdir()
    (tmp_path / 'agent/harness.json').write_text(json.dumps({
        'status': 'PASS', 'facts': {'replaced_forwards': {
            f'layer{i}': {'fingerprint': f'fp{i}'} for i in range(100)}}}))
    write_feedback(tmp_path, {'outcome': 'reject'})
    text = (tmp_path / 'outcome.md').read_text()
    assert "'count': 100" in text and 'layer0' in text and 'layer99' not in text


def test_feedback_does_not_invent_missing_cpu_coverage(tmp_path):
    write_feedback(tmp_path, {'outcome': 'reject', 'reason': 'build_failed'})
    text = (tmp_path / 'outcome.md').read_text()
    assert 'CPU 路径检查证据' not in text and 'CPU 观测事实' not in text
