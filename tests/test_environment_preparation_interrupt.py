from pathlib import Path

import pytest

from vf_runtime.storage.atomic import read_json
from vibeforge_v2.config import ROOT, load
from vibeforge_v2.environment import ExperimentIntegrityError
from vibeforge_v2.loop import Pipeline


@pytest.mark.parametrize('failure', [KeyboardInterrupt, OSError])
def test_incomplete_candidate_preparation_is_not_input_corruption(tmp_path, monkeypatch, failure):
    pipeline = Pipeline(load(ROOT / 'configs/toy.json'), tmp_path / 'run')
    try:
        pipeline.initialize()
        envs = pipeline.evaluator.environments
        original_python = envs.python
        monkeypatch.setattr(envs, 'python', lambda source: str(tmp_path / 'missing/bin/python')
                            if Path(source).parent.parent.name == 'rounds' else original_python(source))
        def prepare(*args, **kwargs):
            raise failure('preparation interrupted or failed before interpreter creation')
        monkeypatch.setattr(envs, 'prepare', prepare)
        monkeypatch.setattr(pipeline.evaluator, 'start', lambda *args: pytest.fail('must not evaluate candidate'))
        if failure is KeyboardInterrupt:
            with pytest.raises(KeyboardInterrupt):
                pipeline.round(1)
            reason = 'interrupted_during_round'
        else:
            pipeline.round(1)
            reason = 'execution_failed'
        folder = pipeline.run / 'rounds/0001'
        assert read_json(folder / 'decision.json')['reason'] == reason
        assert read_json(folder / 'environment.json')['candidate_after']['status'] == 'not_prepared'
        assert read_json(pipeline.run / 'state.json')['stable'] == 'initial/engine'
    finally:
        pipeline.evaluator.stop()


@pytest.mark.parametrize('corruption', ['frozen_input', 'stable_dependencies', 'prepared_candidate'])
def test_interruption_still_rejects_real_input_or_prepared_environment_failures(
        tmp_path, monkeypatch, corruption):
    pipeline = Pipeline(load(ROOT / 'configs/toy.json'), tmp_path / 'run')
    try:
        pipeline.initialize()
        envs = pipeline.evaluator.environments
        original_python = envs.python
        monkeypatch.setattr(envs, 'python', lambda source: str(tmp_path / 'missing/bin/python')
                            if Path(source).parent.parent.name == 'rounds' else original_python(source))
        def prepare(*args, **kwargs):
            if corruption == 'prepared_candidate':
                return
            if corruption == 'frozen_input':
                path = pipeline.run / 'cases.json'
                path.write_text(path.read_text() + '\n')
            if corruption == 'stable_dependencies':
                def corrupt(*args):
                    raise ExperimentIntegrityError('sealed stable dependencies changed')
                monkeypatch.setattr(envs, 'verify', corrupt)
            raise KeyboardInterrupt()
        def start(*args):
            raise KeyboardInterrupt()
        monkeypatch.setattr(envs, 'prepare', prepare)
        monkeypatch.setattr(pipeline.evaluator, 'start', start)
        with pytest.raises(ExperimentIntegrityError):
            pipeline.round(1)
        assert read_json(pipeline.run / 'rounds/0001/decision.json')['reason'] == 'experiment_integrity_failed'
    finally:
        pipeline.evaluator.stop()
