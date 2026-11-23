import json

import pytest

from vibeforge_v2.edit_tool import EditFile
from vibeforge_v2.agent import edit_candidate
from vibeforge_v2.checks import CandidateCheck
from vibeforge_v2.config import ROOT, load
from test_candidate_feedback import engine, server


@pytest.mark.parametrize('old,new', [('missing', 'new'), ('', 'new'), ('return 1', 'return 1'),
                                    ('return 1', 'return (')])
def test_invalid_edit_leaves_file_unchanged(tmp_path, old, new):
    workspace = engine(tmp_path)
    before = (workspace / 'engine.py').read_bytes()
    tool = EditFile(workspace, tmp_path / 'agent')
    result = json.loads(tool.call(json.dumps({'path': 'engine.py', 'old_text': old, 'new_text': new})))
    assert not result['passed']
    assert (workspace / 'engine.py').read_bytes() == before


def test_ambiguous_edit_and_symlink_escape_are_rejected(tmp_path):
    workspace = engine(tmp_path)
    (workspace / 'note.txt').write_text('same same')
    outside = tmp_path / 'outside.py'
    outside.write_text('return 1')
    (workspace / 'linked.py').symlink_to(outside)
    tool = EditFile(workspace, tmp_path / 'agent')
    for path, old in [('note.txt', 'same'), ('linked.py', 'return 1'), ('../outside.py', 'return 1')]:
        result = json.loads(tool.call(json.dumps({'path': path, 'old_text': old, 'new_text': 'return 2'})))
        assert not result['passed']
    assert outside.read_text() == 'return 1'
    assert (workspace / 'note.txt').read_text() == 'same same'


def test_absolute_candidate_path_is_accepted(tmp_path):
    workspace = engine(tmp_path)
    tool = EditFile(workspace, tmp_path / 'agent')
    result = json.loads(tool.call(json.dumps({'path': str(workspace / 'engine.py'),
                                              'old_text': 'return 1', 'new_text': 'return 2'})))
    assert result['passed'] and 'return 2' in (workspace / 'engine.py').read_text()


def test_official_agent_can_edit_and_submit_through_exact_tool(tmp_path):
    workspace = engine(tmp_path)
    config = load(ROOT / 'configs/toy.json')
    config['agent']['edit_tool'] = True
    call = '<tool_call>\n' + json.dumps({'name': 'edit_file', 'arguments': {
        'path': 'engine.py', 'old_text': 'return 1', 'new_text': 'return 2'}}) + '\n</tool_call>'
    with server([call, '实际改动已提交。']) as (url, requests):
        result = edit_candidate(url, config, workspace, tmp_path / 'round/agent', '修改源码并提交。')
    assert result['tool_calls'] == 1
    assert 'return 2' in (workspace / 'engine.py').read_text()
    assert json.loads((tmp_path / 'round/agent/completion.json').read_text())['ready_for_host_evaluation']
