import hashlib
import json
from pathlib import Path

import pytest

from vibeforge_v2.edit_tool import EditFile


@pytest.mark.parametrize('stale', [False, True])
def test_large_edit_keeps_full_audit_without_echoing_source(tmp_path, stale):
    candidate = tmp_path / 'candidate'
    candidate.mkdir()
    source = 'def load():\n    return ' + repr('large source ' * 4000) + '\n'
    path = candidate / 'engine.py'
    path.write_text(source)
    old = source.replace('return', 'yield', 1) if stale else source
    new = source.replace('large source', 'new source')
    args = {'path': 'engine.py', 'old_text': old, 'new_text': new}
    result = EditFile(candidate, tmp_path / 'agent').call(json.dumps(args))
    response = json.loads(result)
    assert len(result) < 2500
    assert response['changed'] is (not stale)
    assert path.read_text() == (source if stale else new)
    if stale:
        assert 'matched 0 times' in response['error']
        assert '2:     return' in response['match_context']['actual_source']
    assert response['submitted_text']['old_text']['sha256'] == hashlib.sha256(old.encode()).hexdigest()
    audit = json.loads(Path(response['full_arguments_log']).read_text())
    assert audit['arguments'] == args
    assert audit['changed'] == response['changed']
