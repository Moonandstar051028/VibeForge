import json

from vibeforge_v2.edit_tool import EditFile


def edit(tmp_path, source, old):
    candidate = tmp_path / 'engine'
    candidate.mkdir()
    path = candidate / 'engine.py'
    path.write_text(source)
    tool = EditFile(candidate, tmp_path / 'agent')
    report = json.loads(tool.call(json.dumps({'path': str(path), 'old_text': old, 'new_text': 'pass\n'})))
    assert not report['passed'] and not report['changed']
    assert path.read_text() == source
    return report


def test_stale_long_edit_returns_real_disk_context_at_first_difference(tmp_path):
    source = 'def load(model):\n    result = model.load().eval()\n    return result\n'
    old = source.replace('.eval()', '')
    report = edit(tmp_path, source, old)
    context = report['match_context']
    assert context['source_line'] == context['supplied_line'] == 2
    assert '2:     result = model.load().eval()' in context['actual_source']
    assert 'No edit was applied' in context['guidance']


def test_diagnostic_anchor_is_not_chosen_when_starting_line_is_ambiguous(tmp_path):
    source = 'def first():\n    result = 1\n\ndef second():\n    result = 1\n'
    report = edit(tmp_path, source, '    result = 1\n    missing = 2\n')
    assert 'match_context' not in report


def test_ambiguous_exact_edit_reports_locations_without_applying(tmp_path):
    report = edit(tmp_path, 'first = 1\nsecond = 1\n', '= 1')
    assert report['matching_source_lines'] == [1, 2]
    assert 'match_context' not in report


def test_missing_starting_line_does_not_invent_match(tmp_path):
    report = edit(tmp_path, 'def original():\n    pass\n', 'def invented():\n    return 1\n')
    assert 'match_context' not in report


def test_context_is_bounded_for_long_source_lines(tmp_path):
    source = 'def load():\n    result = ' + repr('a' * 10000) + '\n    return result\n'
    report = edit(tmp_path, source, 'def load():\n    result = None\n')
    assert len(report['match_context']['actual_source']) < 1100
    assert len(report['match_context']['supplied_near_mismatch']) <= 240
