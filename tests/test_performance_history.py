import json

from vibeforge_v2.feedback import write_feedback
from vibeforge_v2.experience import collect, render
from vibeforge_v2.files import write_history
from vibeforge_v2.performance_feedback import performance_summary


def put(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def attempt(folder, number=0, outcome='reject', value=7.12762, valid=True):
    decision = {'outcome': outcome, 'reason': 'profile_regression:agent_2k',
                'objective': 'request_output_tokens_per_second_mean', 'gain': None}
    put(folder / f'performance/attempt-{number}.json', {
        'decision': decision, 'baseline': [{'valid': valid}], 'candidate': [{'valid': valid}],
        'comparison': {'profiles': {'agent_2k': {
            'baseline': {'request_output_tokens_per_second_mean': 24.0228, 'output_tokens_per_second': 300},
            'candidate': {'request_output_tokens_per_second_mean': value, 'output_tokens_per_second': 600},
            'relative_change': {'request_output_tokens_per_second_mean': -.7033, 'output_tokens_per_second': 1},
        }}}})
    return decision


def test_measured_regression_survives_context_excerpt_with_verbose_other_evidence(tmp_path):
    folder = tmp_path / 'rounds/0009'
    decision = attempt(folder)
    put(folder / 'decision.json', decision)
    put(folder / 'logs/candidate.runtime.json', {'runtime': {'requested': {'long': 'x' * 4000}}})
    put(folder / 'quality/report.json', {'paired': {'diagnostics': 'y' * 4000}})
    write_feedback(folder, decision)
    excerpt = (folder / 'outcome.md').read_text()[:2500]
    assert '24.0228 → 7.12762' in excerpt
    assert '-70.33%' in excerpt
    assert 'performance/attempt-0.json' in excerpt
    assert '300 → 600' not in excerpt  # Concurrent throughput cannot replace the objective.
    assert '不是汇总收益或配对置信区间' in excerpt
    write_history(tmp_path)
    history = (tmp_path / 'history.md').read_text()
    assert '已采样，无汇总收益' in history
    assert '未测量' not in history


def test_missing_and_incomplete_samples_are_not_claimed_as_completed_measurements(tmp_path):
    folder = tmp_path / 'rounds/0001'
    decision = {'outcome': 'reject', 'reason': 'execution_failed', 'gain': None}
    put(folder / 'decision.json', decision)
    write_feedback(folder, decision)
    write_history(tmp_path)
    assert '正式性能未执行' in (folder / 'outcome.md').read_text()
    assert '未测量' in (tmp_path / 'history.md').read_text()
    samples = folder / 'performance/samples.jsonl'
    samples.parent.mkdir()
    samples.write_text('{"valid": true}\n')
    write_feedback(folder, decision)
    write_history(tmp_path)
    assert '无完成的配对评测报告' in (folder / 'outcome.md').read_text()
    assert '正式采样未完成，无汇总收益' in (tmp_path / 'history.md').read_text()


def test_latest_numeric_attempt_and_invalid_unknown_values_stay_explicit(tmp_path):
    attempt(tmp_path, number=9, value=99)
    decision = attempt(tmp_path, number=10, outcome='invalid', value=None, valid=False)
    text = '\n'.join(performance_summary(tmp_path, decision))
    assert 'attempt-10.json' in text
    assert 'invalid/' in text
    assert '24.0228 → 未知' in text
    assert '样本有效性未全部通过' in text
    assert '→ 99' not in text


def test_operator_interruption_does_not_hide_previous_measured_regression(tmp_path):
    measured = tmp_path / 'rounds/0010'
    decision = attempt(measured)
    put(measured / 'decision.json', decision)
    put(tmp_path / 'rounds/0011/decision.json', {
        'outcome': 'interrupted', 'reason': 'interrupted_during_round', 'stage': 'observations'})
    text = render(collect(tmp_path), tmp_path, compact=True)
    assert '0011' in text
    assert '0010' in text and '-70.33%' in text
    assert '稳定 24.0228 → 候选 7.12762' in text
    assert '非汇总收益' in text
    assert 'performance/attempt-0.json' in text
