"""Regression checks for the fresh A800 campaign review."""
import json
import sys
from pathlib import Path

from vibeforge_v2.checks import CandidateCheck
from vf_runtime.storage.atomic import write_json


def test_malformed_historical_candidate_does_not_abort_next_completion(tmp_path):
    previous = tmp_path / 'rounds/0001/engine'
    current = tmp_path / 'rounds/0002/engine'
    for folder in (previous, current):
        folder.mkdir(parents=True)
        (folder / 'engine.json').write_text('{"backend":"toy"}')
        (folder / 'engine.py').write_text('def generate(): return 1\n')
    (previous / 'engine.py').write_text('def broken(:\n')
    write_json(previous.parent / 'decision.json', {'outcome':'reject', 'reason':'execution_failed', 'stage':'build'})
    checker = CandidateCheck(current, tmp_path / 'checks')
    assert checker.repeated_failure(sys.executable) is None


def test_incomplete_historical_environment_is_not_repeat_evidence(tmp_path):
    previous = tmp_path / 'rounds/0001/engine'
    current = tmp_path / 'rounds/0002/engine'
    for folder in (previous, current):
        folder.mkdir(parents=True)
        (folder / 'engine.json').write_text('{"backend":"toy"}')
        (folder / 'engine.py').write_text('def generate(): return 1\n')
    write_json(previous.parent / 'decision.json', {'outcome':'reject', 'reason':'execution_failed'})
    write_json(previous.parent / 'environment.json', {'candidate_after':{'error':'environment creation interrupted'}})
    checker = CandidateCheck(current, tmp_path / 'checks')
    assert checker.repeated_failure(sys.executable) is None


def test_related_history_filters_model_and_reports_changed_evaluator(tmp_path):
    from vibeforge_v2.related_history import write_related_history
    config = {'project': str(tmp_path), 'model': {'name':'30b', 'path':'/weights/30b'},
              'hardware': {'name_contains':'A800'}, 'engine': {'adapter':'transformers'},
              'precision': {'weight_dtype':'bfloat16'}, 'workload': {'profiles':[{'id':'short'}]},
              'evaluation': {'quality': {'file_hashes':{'new-evaluator':'new-hash'}}}}
    for name, model in [('prior','30b'), ('unrelated','7b')]:
        prior = tmp_path/'runs'/name
        write_json(prior/'config.json', {**config,'model':{**config['model'],'name':model},
                   'evaluation':{'quality':{'file_hashes':{'old-evaluator':'old-hash'}}}})
        folder=prior/'rounds/0001'
        write_json(folder/'decision.json', {'outcome':'reject','reason':'quality_failed'})
        (folder/'outcome.md').write_text('Actual quality rejection')
    output=tmp_path/'runs/current/related-history.md'
    text=write_related_history(config, output.parent, output)
    assert str(tmp_path/'runs/prior/rounds/0001') in text
    assert 'unrelated' not in text
    assert '质量评测输入/脚本哈希不同' in text
    assert '不得复用基线' in text
    # An incomplete archive must not break the live round.
    (tmp_path/'runs/broken').mkdir()
    (tmp_path/'runs/broken/config.json').write_text('{broken')
    assert write_related_history(config, output.parent, output) == text


def test_related_history_without_prior_runs_is_valid(tmp_path):
    from vibeforge_v2.related_history import write_related_history
    text = write_related_history({'project':str(tmp_path)}, tmp_path/'current', tmp_path/'summary.md')
    assert '未发现匹配配置' in text


def test_disabled_related_history_does_not_scan_old_runs(tmp_path, monkeypatch):
    from vibeforge_v2.related_history import write_related_history
    def forbidden_scan(*args, **kwargs):
        raise AssertionError('Disabled cross-run history must not scan any runs')
    monkeypatch.setattr(Path, 'glob', forbidden_scan)
    output = tmp_path / 'current/related-history.md'
    text = write_related_history({'project': str(tmp_path), 'agent': {'related_history': False}},
                                 tmp_path / 'current', output)
    assert '禁用跨运行历史' in text
    assert output.read_text() == text


def test_config_only_batching_claims_require_a_reader_and_active_scheduler(tmp_path):
    root=tmp_path/'engine'
    root.mkdir()
    (root/'engine.py').write_text('def generate(): return 1\n')
    write_json(root/'engine.json', {'backend':'toy','microbatch_max_size':4})
    checker=CandidateCheck(root,tmp_path/'checks')
    write_json(root/'engine.json', {'backend':'toy','microbatch_max_size':8,'continuous_batching':True})
    report=checker.inspect()
    assert not report['ready_for_host_evaluation']
    assert any('unsupported mechanisms' in e for e in report['errors'])
    assert any('request_scheduler is not "microbatch"' in e for e in report['errors'])
    write_json(root/'engine.json', {'backend':'toy','microbatch_max_size':8,'request_scheduler':'microbatch'})
    assert checker.inspect()['ready_for_host_evaluation']


def test_removing_unread_microbatch_keys_is_a_note_not_an_error(tmp_path):
    root=tmp_path/'engine'
    root.mkdir()
    (root/'engine.py').write_text('def generate(): return 1\n')
    write_json(root/'engine.json', {'backend':'toy','microbatch_max_size':4,'microbatch_wait_ms':4})
    checker=CandidateCheck(root,tmp_path/'checks')
    write_json(root/'engine.json', {'backend':'toy'})
    report=checker.inspect()
    assert report['static_passed'] and not report['errors']
    assert any('Removed unused keys' in w for w in report['warnings'])
    # An inert cleanup alone is not an effective runtime change, so it is not sent to evaluation.
    assert not report['ready_for_host_evaluation']
    write_json(root/'engine.json', {'backend':'toy','attn_implementation':'sdpa'})
    assert checker.inspect()['ready_for_host_evaluation']


def test_feedback_reports_actual_scheduler_not_agent_batching_claim(tmp_path):
    from vibeforge_v2.feedback import write_feedback
    write_json(tmp_path/'logs/candidate.runtime.json', {'runtime': {
        'request_scheduler':'serialized_lock','continuous_batching':False,'batching_mode':'none',
        'attention_implementations':{'model':'flash_attention_2'},
        'requested':{'microbatch_max_size':8}}})
    write_feedback(tmp_path, {'outcome':'reject','reason':'quality_failed'})
    text=(tmp_path/'outcome.md').read_text()
    assert 'serialized_lock' in text and "'continuous_batching': False" in text
    assert 'flash_attention_2' in text


def test_tool_case_without_diagnostics_does_not_erase_other_quality_evidence(tmp_path):
    from vibeforge_v2.feedback import write_feedback
    write_json(tmp_path/'quality/report.json', {'passed':False,'cases':[
        {'id':'chat','passed':False,'reached_output_limit':True,'failure_kind':'exact_format_or_answer'},
        {'id':'tool','passed':True,'transcript':{}}]})
    write_feedback(tmp_path, {'reason':'quality_failed'})
    text=(tmp_path/'outcome.md').read_text()
    assert "'reached_output_limit': 1" in text
    assert "'output_limit_unobserved_cases': 1" in text
    assert "'failed_exact_match': 1" in text


def test_quality_failure_still_measures_exploration_without_promotion(tmp_path, monkeypatch):
    from vibeforge_v2.config import ROOT, load
    from vibeforge_v2.loop import Pipeline
    from test_pipeline import fixture_edit, fixture_connection
    from vf_runtime.storage.atomic import read_json
    calls=[]
    pipeline=Pipeline(load(ROOT/'configs/toy.json'),tmp_path/'run',edit=fixture_edit,connect=fixture_connection)
    original=pipeline.evaluator.quality
    def quality(client,source,folder,baseline=None):
        if baseline is None:
            return original(client,source,folder,baseline)
        report={'passed':False,'metrics':{},'errors':['paired_quality_not_established']}
        write_json(Path(folder)/'report.json',report)
        return report
    def diagnose(stable,candidate,reference,folder):
        calls.append((stable,candidate))
        report={'admission':False,'valid':True,'gain':.5,'pairs':2,'quality_passed':False}
        write_json(Path(folder)/'report.json',report)
        return report
    monkeypatch.setattr(pipeline.evaluator,'quality',quality)
    monkeypatch.setattr(pipeline.evaluator,'diagnose_performance',diagnose)
    state=pipeline.execute(rounds=1)
    assert calls
    decision=read_json(tmp_path/'run/rounds/0001/decision.json')
    assert decision['outcome']=='reject' and decision['reason']=='quality_failed'
    assert decision['diagnostic_performance']['gain']==.5
    assert state['stable']=='initial/engine'
    assert not (tmp_path/'run/rounds/0001/activation').exists()
    assert '探索性性能' in (tmp_path/'run/rounds/0001/outcome.md').read_text()


def test_diagnostic_measurement_uses_ab_ba_and_retains_invalid_samples(tmp_path, monkeypatch):
    import vibeforge_v2.evaluate as module
    from vibeforge_v2.config import ROOT, load
    from types import SimpleNamespace
    a,b=tmp_path/'initial/engine',tmp_path/'rounds/0001/engine'
    for p in (a,b):
        p.mkdir(parents=True)
        (p/'engine.py').write_text('pass\n')
    evaluator=module.Evaluator(load(ROOT/'configs/toy.json'),tmp_path)
    evaluator.config['evaluation']['collect_token_timing'] = True
    order=[]
    evaluator.environments.verify=lambda source:None
    evaluator.engine.process=SimpleNamespace(pid=123)
    observations=[]
    def start(source,log):
        order.append(source)
        return SimpleNamespace(generate=lambda **kwargs:observations.append(kwargs),startup_seconds=1)
    evaluator.start=start
    evaluator.stop=lambda:None
    def measure(*args,**kwargs):
        return {'valid':len(order)!=3,'metrics':{'throughput':10},'invalid_reasons':['external_gpu_process'] if len(order)==3 else []}
    monkeypatch.setattr(module,'measure_suite',measure)
    report=evaluator.diagnose_performance(a,b,{},tmp_path/'diagnostic')
    assert order==[a,b,b,a]
    assert len(observations) == 8
    assert all(row['observation_mode'] == 'detailed' for row in observations)
    assert not report['admission'] and not report['valid']
    assert report['gain'] is None
    assert len(report['candidate'])==2


def test_native_implementation_change_is_not_mislabeled_as_config_only(tmp_path):
    root=tmp_path/'engine'
    root.mkdir()
    (root/'engine.py').write_text('def generate(): return 1\n')
    write_json(root/'engine.json', {'backend':'toy'})
    checker=CandidateCheck(root,tmp_path/'checks')
    (root/'scheduler.cpp').write_text('const char* option = "continuous_batching";\n')
    write_json(root/'engine.json', {'backend':'toy','continuous_batching':True})
    report=checker.inspect()
    assert report['ready_for_host_evaluation']
    assert report['warnings']  # Source changes are not proof of runtime activation.
