import json
import pytest
from vibeforge_v2.agent import Shell
from vibeforge_v2.checks import CandidateCheck
from vf_runtime.engine.backend import load_backend


def test_swallowed_entry_relative_import_is_reported_before_model_load(tmp_path):
    engine = tmp_path / 'engine'
    engine.mkdir()
    (engine / 'engine.json').write_text('{"backend":"toy"}')
    source = ('def load(config):\n    pass\n'
              'def generate(): pass\ndef chat(): pass\ndef chat_completion(): pass\ndef logits(): pass\n')
    (engine / 'engine.py').write_text(source)
    checker = CandidateCheck(engine, tmp_path / 'checks')
    (engine / 'local_patch.py').write_text('def apply_patch(): pass\n')
    changed = source.replace('    pass', '    try:\n        from .local_patch import apply_patch\n'
                             '        apply_patch()\n    except ImportError:\n        pass', 1)
    (engine / 'engine.py').write_text(changed)
    module, _ = load_backend(engine)
    assert module.__package__ == ''  # Actual loader confirms the diagnosis.
    tool = Shell(engine, tmp_path / 'agent', feedback=checker)
    response = tool.call(json.dumps({'command': 'true'}))
    assert 'relative import' in response and 'Catching ImportError silently skips' in response
    assert not checker.inspect()['ready_for_host_evaluation']


def test_package_module_relative_import_is_not_misdiagnosed(tmp_path):
    engine = tmp_path / 'engine'
    engine.mkdir()
    (engine / 'engine.json').write_text('{"backend":"toy"}')
    (engine / 'engine.py').write_text('def load(config): pass\n')
    checker = CandidateCheck(engine, tmp_path / 'checks')
    package = engine / 'helpers'
    package.mkdir()
    (package / '__init__.py').write_text('from .implementation import apply_patch\n')
    (package / 'implementation.py').write_text('def apply_patch(): pass\n')
    assert checker.inspect()['static_passed']
