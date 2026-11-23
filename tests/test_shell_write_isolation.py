"""Real subprocess checks for the shared-dependency corruption found in round 7."""

import json
import os
import sys

from vibeforge_v2.agent import Shell
from vibeforge_v2.harness import run_engine


def test_candidate_can_edit_and_install_locally_but_not_follow_shared_symlinks(tmp_path):
    run = tmp_path / 'run'
    candidate = run / 'rounds/0001/engine'
    candidate.mkdir(parents=True)
    dependency = run / 'environments/rounds/0001'
    (dependency / 'bin').mkdir(parents=True)
    (dependency / 'bin/python').symlink_to(sys.executable)
    shared = tmp_path / 'shared-package.py'
    shared.write_text('original\n')
    (candidate / 'shared-link.py').symlink_to(shared)
    tool = Shell(candidate, run / 'rounds/0001/agent')
    command = '''import os, pathlib
pathlib.Path('local.py').write_text('candidate')
pathlib.Path(os.environ['TMPDIR'], 'scratch').write_text('temporary')
pathlib.Path(os.environ['PATH'].split(os.pathsep)[0]).parent.joinpath('installed.py').write_text('dependency')
for name in ['shared-link.py', '../agent/tools.jsonl']:
    try:
        pathlib.Path(name).write_text('corruption')
    except PermissionError:
        print('BLOCKED', name)
    else:
        raise AssertionError('Unconfined write: ' + name)
'''
    result = tool.call(json.dumps({'command': 'python -c ' + __import__('shlex').quote(command)}))
    assert 'exit_code: 0' in result and 'BLOCKED shared-link.py' in result
    assert shared.read_text() == 'original\n'
    assert (candidate / 'local.py').read_text() == 'candidate'
    assert (dependency / 'installed.py').read_text() == 'dependency'
    assert tool.python == str(dependency / 'bin/python')


def test_cpu_harness_candidate_import_cannot_modify_shared_files(tmp_path):
    candidate, work = tmp_path / 'candidate', tmp_path / 'work'
    candidate.mkdir()
    work.mkdir()
    outside = tmp_path / 'shared.py'
    outside.write_text('original')
    (candidate / 'engine.json').write_text('{}')
    (candidate / 'engine.py').write_text(
        'from pathlib import Path\nPath(' + repr(str(outside)) + ').write_text("corruption")\n')
    report = run_engine(candidate, tmp_path, 'float32', sys.executable, 30, work)
    assert not report['ok'] and report['stage'] == 'import engine.py'
    assert 'PermissionError' in report['error']
    assert outside.read_text() == 'original'
