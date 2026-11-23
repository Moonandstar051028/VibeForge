import json
from pathlib import Path

import pytest

from vibeforge_v2.agent import Shell, edit_candidate
from vibeforge_v2.checks import CandidateCheck
from vibeforge_v2.config import ROOT, load
from vibeforge_v2.interface_contract import check_contract, signatures
from test_candidate_feedback import server


@pytest.mark.parametrize('candidate', [
    'def logits(input_ids): pass',
    'def logits(input_ids, indices): pass',
    'def logits(input_ids, indices=None, /): pass',
    'def logits(input_ids, indexes=None): pass',
    'def logits(input_ids, indices=None, *, required): pass',
    'async def logits(input_ids, indices=None): pass',
])
def test_rejects_broken_calls_without_importing(candidate):
    stable = signatures('def logits(input_ids, indices=None): pass')
    errors, _ = check_contract(stable, 'raise RuntimeError("must never execute")\n' + candidate)
    assert errors


@pytest.mark.parametrize('candidate', [
    'def logits(input_ids, indices=None, extra=None): pass',
    'def logits(input_ids, *args, **kwargs): pass',
])
def test_allows_compatible_extensions(candidate):
    assert not check_contract(signatures('def logits(input_ids, indices=None): pass'), candidate)[0]


def test_variadic_host_options_and_positional_only_calls():
    stable = signatures('def chat_completion(messages, tools=None, max_tokens=None, **kwargs): pass')
    assert check_contract(stable, 'def chat_completion(messages, tools=None, max_tokens=None): pass')[0]
    stable = signatures('def generate(prompt, /, *args, limit=1): pass')
    assert not check_contract(stable, 'def generate(prompt, /, *args, limit=1, **kwargs): pass')[0]
    assert check_contract(stable, 'def generate(prompt, /, limit=1): pass')[0]


def test_real_engine_indices_regression_is_reported_immediately_by_shell(tmp_path):
    engine = tmp_path / 'engine'
    engine.mkdir()
    original = (ROOT / 'engines/transformers/engine.py').read_text()
    (engine / 'engine.py').write_text(original)
    (engine / 'engine.json').write_text('{"backend":"transformers"}')
    check = CandidateCheck(engine, tmp_path / 'checks')
    shell = Shell(engine, tmp_path / 'agent', feedback=check)
    result = shell.call(json.dumps({'command': "sed -i 's/def logits(input_ids, indices=None):/def logits(input_ids):/' engine.py"}))
    assert 'incompatible public call logits' in result
    assert not shell.calls[-1]['candidate_check']['ready_for_host_evaluation']
    (engine / 'engine.py').write_text(original.replace('min(32, values.shape[-1])', 'min(31, values.shape[-1])'))
    assert check.inspect()['ready_for_host_evaluation']


def test_agent_repairs_interface_and_preserves_candidate_before_submission(tmp_path):
    engine = tmp_path / 'engine'
    engine.mkdir()
    (engine / 'engine.json').write_text('{"backend":"toy","max_context":128}')
    (engine / 'engine.py').write_text('def load(config): pass\ndef logits(input_ids, indices=None): return 1\n')
    def edit(old, new):
        command = f"sed -i 's/{old}/{new}/' engine.py"
        return '<tool_call>\n' + json.dumps({'name': 'shell', 'arguments': {'command': command}}) + '\n</tool_call>'
    replies = [edit('indices=None', 'indexes=None'), '候选已修改。',
               edit('indexes=None): return 1', 'indices=None): return 2'), '已局部修复并保留改动。']
    folder = tmp_path / 'round/agent'
    with server(replies) as (url, requests):
        result = edit_candidate(url, load(ROOT / 'configs/toy.json'), engine, folder, '修改候选并提交。')
    assert result['tool_calls'] == 2
    assert 'indices=None): return 2' in (engine / 'engine.py').read_text()
    assert any('incompatible public call logits' in (m.get('content') or '') for m in requests[2]['messages'])
    assert json.loads((folder / 'completion.json').read_text())['ready_for_host_evaluation']
