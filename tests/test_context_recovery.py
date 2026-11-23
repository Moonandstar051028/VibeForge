import copy
import json

import pytest

from vibeforge_v2.context_recovery import fold_reads, observation, source_read
from vibeforge_v2.agent_context import NativeContext
from test_exact_context import Tokenizer


def steps(n=6, command='cat engine.py', source='stable', output='source code'):
    rows = []
    for i in range(n):
        record = {'command': command, 'output': output, 'candidate_check': {'source_hash': source},
                  'exit_code': 0, 'timed_out': False}
        rows.extend([{'role': 'assistant', 'content': '', 'tool_calls': [{'id': str(i), 'function': {
            'name': 'shell', 'arguments': json.dumps({'command': command})}}]},
            {'role': 'tool', 'tool_call_id': str(i), 'content': output + observation(record, i)}])
    return rows


def test_repeat_fold_keeps_latest_pair_and_raw_history():
    rows = steps()
    original = copy.deepcopy(rows)
    folded, events = fold_reads(rows)
    assert rows == original
    assert folded[:2] == rows[-2:] and len(folded) == 3
    assert events[0]['count'] == 6
    assert '连续 6 次' in folded[-1]['content']


@pytest.mark.parametrize('command', ['echo x > engine.py', 'cat x; rm y', 'sed -i s/a/b/ x',
                                      'python -c "open(\"x\",\"w\")"'])
def test_mutating_commands_are_not_folded(command):
    rows = steps(command=command)
    assert fold_reads(rows) == (rows, [])


def test_source_changes_outputs_and_new_reasoning_reset_repeats():
    for rows in (steps(3) + steps(3, source='edited'), steps(3) + steps(3, output='new'),
                 steps(3) + [{'role': 'user', 'content': 'new evidence'}] + steps(3)):
        assert fold_reads(rows) == (rows, [])
    rows = steps()
    for i in range(0, len(rows), 2):
        rows[i]['content'] = f'new reasoning {i}'
    assert fold_reads(rows) == (rows, [])


def test_real_source_inspection_command_and_side_effect_rejected():
    cmd = '/candidate/bin/python -c "import inspect; from accelerate.hooks import remove_hook_from_module; print(inspect.getsource(remove_hook_from_module))"'
    assert source_read(cmd)
    assert not source_read('python -c "import inspect; print(inspect.getsource(do_work()))"')


def test_native_qwen_result_ids_and_scoped_reads_are_folded():
    rows = steps(command="cd /candidate && sed -n '1,60p' engine.py")
    for row in rows:
        if row['role'] == 'tool':
            row['id'] = row.pop('tool_call_id')
            row['extra'] = {'function_id': row['id']}
    folded, events = fold_reads(rows)
    assert folded[:2] == rows[-2:] and events[0]['count'] == 6
    assert not source_read('cd /candidate && cat engine.py && rm engine.py')
    assert not source_read('cd $(touch x) && cat engine.py')


def test_observed_version_query_loop_with_output_wrapper():
    cmd = ('cd /candidate && /candidate/bin/python -c "\nimport torch, triton\n'
           "print('triton', triton.__version__, triton.__file__)\n\" 2>&1 | tail -3")
    assert source_read(cmd)
    rows = steps(command=cmd)
    folded, events = fold_reads(rows)
    assert len(folded) == 3 and events[0]['count'] == 6
    assert not source_read(cmd + ' > output.txt')
    assert not source_read('python -c "import os; print(os.remove(\'x\'))" 2>&1 | tail -3')
    assert not source_read('cat engine.py | bash | tail -3')


def test_observed_repeats_fold_history_without_claiming_no_side_effects():
    rows = steps(command="python3 - <<'EOF'\nimport json\nd=json.load(open('data.json'))\nfor k in d: print(k)\nEOF")
    assert fold_reads(rows)[1] == []
    original = copy.deepcopy(rows)
    folded, events = fold_reads(rows, observed_repeats=True)
    assert events[0]['count'] == 6 and folded[:2] == rows[-2:]
    assert rows == original and '不代表命令没有副作用' in folded[-1]['content']
    # Unknown commands are history compression only, never execution deduplication.
    mutating = steps(command='echo x >> scratch.log')
    assert fold_reads(mutating)[1] == []
    assert fold_reads(mutating, observed_repeats=True)[1][0]['count'] == 6
    changed = steps(3) + steps(3, source='changed')
    assert fold_reads(changed, observed_repeats=True) == (changed, [])


def test_structured_memory_retries_invalid_shape_and_enforces_length(tmp_path):
    good = json.dumps(dict.fromkeys(('hypothesis', 'evidence', 'ruled_out', 'blocker', 'next_step'), '未知'))
    calls = []
    def create(**request):
        calls.append(request)
        return {'choices': [{'finish_reason': 'stop', 'message': {'content': 'unstructured' if len(calls) == 1 else good}}]}
    manager = NativeContext(create, Tokenizer(), 12000, 100, tmp_path, structured_memory=True)
    assert manager.summarize([{'role': 'tool', 'content': 'evidence'}], {'model': 'test'}) == good
    assert len(calls) == 2 and calls[0]['messages'][1] == calls[1]['messages'][1]
    assert manager.memory_error('x' * 1025) == 'memory token limit exceeded'


def test_actual_outbound_request_is_folded_without_reexecution(tmp_path):
    sent = []
    manager = NativeContext(lambda **r: sent.append(r), Tokenizer(), 20000, 100, tmp_path,
                            fold_repeated_reads=True)
    rows = [{'role': 'user', 'content': 'task'}] + steps()
    manager(messages=rows, model='test', max_tokens=500, stream=True)
    assert len(sent) == 1 and len(sent[0]['messages']) == 4
    assert len(rows) == 13


def test_harness_unavailable_is_error_but_gpu_gate_remains_available(tmp_path, monkeypatch):
    from vibeforge_v2 import agent, harness
    def unavailable(*a, **k):
        raise RuntimeError('CPU fixture unavailable')
    monkeypatch.setattr(harness, 'run', unavailable)
    workspace = tmp_path / 'engine'
    workspace.mkdir()
    (workspace / 'engine.py').write_text('x = 1')
    report = agent.run_harness({'agent': {}}, workspace, tmp_path / 'agent', None)
    assert not report['passed'] and report['status'] == 'ERROR'
    assert report['allow_gpu_evaluation'] and 'HARNESS: ERROR' in report['text']


def test_skipped_numeric_comparison_never_reports_pass(tmp_path, monkeypatch):
    from vibeforge_v2 import harness
    for name in ('stable', 'candidate'):
        (tmp_path / name).mkdir()
    monkeypatch.setattr(harness, 'build_reduced_model', lambda *a: tmp_path)
    monkeypatch.setattr(harness, 'run_engine', lambda *a, **k: {'ok': False, 'error': 'GPU only'})
    report = harness.run(tmp_path / 'candidate', stable=tmp_path / 'stable',
                         model_path='unused', cache=tmp_path / 'cache')
    assert report['status'] == 'SKIP' and not report['passed']
    assert report['allow_gpu_evaluation'] and 'HARNESS: SKIP' in harness.describe(report)
