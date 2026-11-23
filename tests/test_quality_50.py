"""Compact quality coverage and explicit small-suite evidence policy."""
import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest

from vibeforge_v2.config import ROOT, load
from vibeforge_v2.quality import paired_quality


def test_compact_suite_is_reproducible_and_covers_all_task_families():
    spec=importlib.util.spec_from_file_location('quality_suite_builder',ROOT/'scripts/build_quality_suite.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cases=module.build_compact()
    assert cases==json.loads((ROOT/'benchmarks/task-quality-50.json').read_text())
    assert len(cases)==len({c['id'] for c in cases})==50
    assert Counter(c['category'] for c in cases)=={
        'core':10,'arithmetic':12,'string':6,'reading':6,'logic':6,'coding':6,'tool':4}
    for prefix in ('multiply_','expression_','divide_'):
        assert sum(c['id'].startswith(prefix) for c in cases)==4
    full={c['id']:c for c in module.build()}
    assert all(c==full[c['id']] for c in cases)


def report(bad=()):
    return {'cases':[{'id':str(i),'passed':i not in bad} for i in range(50)]}


def test_50_task_screen_is_explicitly_empirical_not_95_percent_proof():
    settings={'min_cases':50,'confidence':.95,'max_loss_rate':.05,'decision_mode':'observed_loss'}
    result=paired_quality(report(),report(),settings)
    assert result['passed']
    assert not result['confidence_bound_satisfied']
    assert result['loss_rate_upper']==pytest.approx(1-.05**(1/50))
    assert result['observed_loss_rate']==0
    assert 'not a confidence-bound' in result['evidence_scope']
    assert paired_quality(report(),report([0,1]),settings)['passed']
    assert not paired_quality(report(),report([0,1,2]),settings)['passed']
    assert not paired_quality(report(),report(),{**settings,'decision_mode':'confidence_bound'})['passed']


def test_compact_repeats_are_not_treated_as_150_unique_tasks():
    r={'runs':[report()]*3}
    settings={'min_cases':50,'decision_mode':'observed_loss','max_loss_rate':.05}
    assert paired_quality(r,r,settings)['unique_tasks']==50
    with pytest.raises(ValueError,match='decision_mode'):
        paired_quality(report(),report(),{**settings,'decision_mode':'unknown'})


@pytest.mark.parametrize('name',[
    'qwen3-coder-30b-a3b-a800.json','qwen3-coder-30b-a3b-a800-representative.json','qwen3-32b-a800.json'])
def test_a800_configs_use_frozen_compact_suite_and_compatible_policy(name):
    config=load(ROOT/'configs'/name)
    q=config['evaluation']['quality']
    assert q['paired']['min_cases']==50
    assert q['paired']['decision_mode']=='observed_loss'
    assert any(x.endswith('task-quality-50.json') for x in q['command'])
    assert any(x.endswith('task-quality-50.json') for x in q['file_hashes'])
    # README: every A800 config runs the 50-task suite once for baseline and candidate.
    assert q['repeats']==1 and q['baseline_repeats']==1
    assert q['category_item_tolerance']==1
