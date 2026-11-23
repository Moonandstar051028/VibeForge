"""Malformed auxiliary files must not be reported as formatting-only edits."""
import json
from pathlib import Path

from vibeforge_v2.checks import CandidateCheck


def test_empty_auxiliary_json_reports_filename_and_can_recover(tmp_path):
    source=tmp_path/'engine'
    source.mkdir()
    (source/'engine.py').write_text('def generate(): return 1\n')
    (source/'engine.json').write_text('{"backend":"toy"}')
    check=CandidateCheck(source,tmp_path/'checks')
    (source/'engine.json').write_text('{"backend":"toy","optimization":true}')
    (source/'temp.json').write_text('')
    report=check.inspect()
    assert not report['static_passed']
    assert not report['ready_for_host_evaluation']
    assert any('temp.json' in error and 'Semantic inspection failed' in error for error in report['errors'])
    assert not any('Only formatting/comments' in warning for warning in report['warnings'])
    (source/'temp.json').unlink()
    recovered=check.inspect()
    assert recovered['static_passed'] and recovered['ready_for_host_evaluation']
    assert recovered['semantic_changed_files']==['engine.json']
