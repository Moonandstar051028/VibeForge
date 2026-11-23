from pathlib import Path

import pytest

from vibeforge_v2 import harness


def make_engine(path):
    path.mkdir(parents=True)
    (path / 'engine.py').write_text('# engine\n')
    (path / 'engine.json').write_text('{}')
    return path


def no_harness_calls(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Invalid paths must fail before model/cache work')
    monkeypatch.setattr(harness, 'run', forbidden)


def test_round_parent_default_reports_actual_child_without_starting_harness(tmp_path, monkeypatch, capsys):
    folder = tmp_path / 'run/rounds/0013'
    child = make_engine(folder / 'engine')
    stable = make_engine(tmp_path / 'run/current/engine')
    monkeypatch.chdir(folder)
    no_harness_calls(monkeypatch)
    with pytest.raises(SystemExit) as error:
        harness.main(['--stable', str(stable)])
    assert error.value.code == 2
    text = capsys.readouterr().err
    assert f'Not an engine directory: {folder}' in text
    assert f'--candidate {child}' in text
    assert 'missing engine.py, engine.json' in text
    assert 'No model was loaded and no cache was written' in text


def test_partial_candidate_reports_only_missing_file(tmp_path, monkeypatch, capsys):
    (tmp_path / 'engine.json').write_text('{}')
    no_harness_calls(monkeypatch)
    with pytest.raises(SystemExit) as error:
        harness.main(['--candidate', str(tmp_path)])
    assert error.value.code == 2
    text = capsys.readouterr().err
    assert 'missing engine.py.' in text
    assert 'The engine files are in' not in text


def test_bad_stable_also_fails_before_running(tmp_path, monkeypatch, capsys):
    candidate = make_engine(tmp_path / 'candidate')
    no_harness_calls(monkeypatch)
    with pytest.raises(SystemExit) as error:
        harness.main(['--candidate', str(candidate), '--stable', str(tmp_path / 'absent')])
    assert error.value.code == 2
    assert 'argument --stable' in capsys.readouterr().err


def test_valid_candidate_and_stable_keep_exact_selected_paths(tmp_path, monkeypatch):
    candidate = make_engine(tmp_path / 'candidate')
    stable = make_engine(tmp_path / 'stable')
    observed = []
    def fake(path, **kwargs):
        observed.append((path, kwargs))
        return {'status': 'PASS', 'passed': True, 'allow_gpu_evaluation': True,
                'errors': [], 'warnings': [], 'scope': 'test', 'wall_seconds': 0}
    monkeypatch.setattr(harness, 'run', fake)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as error:
        harness.main(['--candidate', 'candidate', '--stable', 'stable'])
    assert error.value.code == 0
    assert observed[0][0] == candidate
    assert observed[0][1]['stable'] == stable
