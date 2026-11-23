"""Unlimited execution, bounded overrides and cleanup on detached process signals."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from vibeforge_v2.config import ROOT, load
from vibeforge_v2.loop import Pipeline
from vf_runtime.storage.atomic import read_json


def fake_pipeline(tmp_path, monkeypatch, *, stop_at, output=None):
    config=load(ROOT/'configs/toy.json')
    config['rounds']=1
    pipeline=Pipeline(config,output or tmp_path/'run')
    calls=[]
    def initialize():
        pipeline.run.mkdir()
        (pipeline.run/'rounds').mkdir()
        pipeline.save_state(stable='initial/engine',completed_rounds=0)
    def execute_round(number):
        calls.append(number)
        if number==stop_at:
            raise KeyboardInterrupt()
        pipeline.save_state(completed_rounds=number)
        return {'outcome':'reject','reason':'fixture'}
    monkeypatch.setattr(pipeline,'initialize',initialize)
    monkeypatch.setattr(pipeline,'round',execute_round)
    monkeypatch.setattr(pipeline.evaluator,'stop',lambda:None)
    return pipeline,calls


def test_continuous_exceeds_configured_limit_and_resume_inherits_mode(tmp_path,monkeypatch):
    pipeline,calls=fake_pipeline(tmp_path,monkeypatch,stop_at=5)
    with pytest.raises(KeyboardInterrupt):
        pipeline.execute(continuous=True)
    assert calls==[1,2,3,4,5]
    state=read_json(pipeline.run/'state.json')
    assert state['continuous'] and state['round_limit'] is None
    assert state['completed_rounds']==4 and state['status']=='interrupted'
    resumed,next_calls=fake_pipeline(tmp_path,monkeypatch,stop_at=7,output=pipeline.run)
    with pytest.raises(KeyboardInterrupt):
        resumed.execute(resume=True)
    assert next_calls==[5,6,7]
    assert read_json(pipeline.run/'state.json')['completed_rounds']==6


def test_explicit_finite_override_stops_continuous_resume(tmp_path,monkeypatch):
    pipeline,_=fake_pipeline(tmp_path,monkeypatch,stop_at=3)
    with pytest.raises(KeyboardInterrupt):
        pipeline.execute(continuous=True)
    resumed,calls=fake_pipeline(tmp_path,monkeypatch,stop_at=10,output=pipeline.run)
    result=resumed.execute(rounds=4,resume=True)
    assert calls==[3,4]
    assert not result['continuous'] and result['round_limit']==4 and result['status']=='completed'


def test_continuous_and_round_limit_are_exclusive_before_creating_output(tmp_path):
    pipeline=Pipeline(load(ROOT/'configs/toy.json'),tmp_path/'run')
    with pytest.raises(ValueError,match='mutually exclusive'):
        pipeline.execute(rounds=1,continuous=True)
    assert not pipeline.run.exists()


def test_cli_continuous_sigterm_records_interruption_and_cleans_engine(tmp_path):
    output=tmp_path/'run'
    env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':''}
    with (tmp_path/'run.log').open('w') as log:
        process=subprocess.Popen([sys.executable,'-m','vibeforge_v2','run','--config',str(ROOT/'configs/toy.json'),
                                  '--output',str(output),'--continuous'],cwd=ROOT,env=env,
                                  stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            deadline=time.monotonic()+30
            while time.monotonic()<deadline:
                if process.poll() is not None:
                    pytest.fail((tmp_path/'run.log').read_text()[-3000:])
                state=read_json(output/'state.json') if (output/'state.json').exists() else {}
                if state.get('current_round',0)>=4:
                    break
                time.sleep(.05)
            else:
                pytest.fail('Continuous CLI did not exceed the default three rounds')
            assert read_json(output/'config.json')['run_control']=={'continuous':True,'round_limit':None}
            process.send_signal(signal.SIGTERM)
            process.wait(timeout=15)
            assert read_json(output/'state.json')['status']=='interrupted'
            assert not (output/'state/active.engine.json').exists()
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
                process.wait(timeout=15)
