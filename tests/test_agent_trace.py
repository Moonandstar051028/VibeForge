"""Tracing must retain exact inputs and streaming errors without extra inference."""
from types import SimpleNamespace

import pytest

from vibeforge_v2.agent_trace import attach_trace
from vf_runtime.storage.atomic import read_json


def test_trace_keeps_pre_fold_input_actual_input_and_reasoning(tmp_path):
    calls = []
    chunks = [
        {'choices':[{'delta':{'reasoning_content':'Inspect the measured path first.'},'finish_reason':None}]},
        {'choices':[{'delta':{'tool_calls':[{'id':'c1','function':{'name':'shell','arguments':'{"command":"pwd"}'}}]},'finish_reason':'tool_calls'}]}]
    def create(**kwargs):
        calls.append(kwargs)
        return iter(chunks)
    llm = SimpleNamespace(_chat_complete_create=create)
    def chat(messages):
        return llm._chat_complete_create(messages=[{'role':'user','content':'folded'}], stream=True)
    llm.chat = chat
    attach_trace(llm, tmp_path)
    original = [{'role':'user','content':'full source'}]
    assert list(llm.chat(messages=original)) == chunks
    assert len(calls)==1 and original[0]['content']=='full source'
    target = tmp_path/'requests/00001'
    assert read_json(target/'before-fold.json')==original
    assert read_json(target/'request.json')['arguments']==calls[0]
    response = read_json(target/'response.json')
    assert response['reasoning_content']=='Inspect the measured path first.'
    assert response['tool_calls'][0]['id']=='c1' and response['status']=='completed'


def test_trace_retains_partial_reasoning_and_propagates_failure(tmp_path):
    def broken():
        yield {'choices':[{'delta':{'reasoning_content':'partial'},'finish_reason':None}]}
        raise RuntimeError('transport failure')
    llm = SimpleNamespace(_chat_complete_create=lambda **kwargs:broken())
    llm.chat = lambda messages:llm._chat_complete_create(messages=messages,stream=True)
    attach_trace(llm,tmp_path)
    with pytest.raises(RuntimeError,match='transport failure'):
        list(llm.chat(messages=[{'role':'user','content':'task'}]))
    response = read_json(tmp_path/'requests/00001/response.json')
    assert response['status']=='failed' and response['reasoning_content']=='partial'


def test_qwen_thinking_is_enabled_only_for_optimizer_requests():
    from vibeforge_v2.config import ROOT, load
    config = load(ROOT/'configs/qwen3.8-27b-a800.json')
    assert config['agent']['thinking']['enabled'] is True
    assert config['agent']['trace_requests'] is True
    assert config['engine']['thinking']['enabled'] is False


def test_context_anomaly_keeps_actual_omissions_and_reasoning(tmp_path, monkeypatch):
    import sys
    from vibeforge_v2.config import ROOT
    from vf_runtime.storage.atomic import write_json
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    from review_agent_context import review
    run=tmp_path/'run'
    trace=run/'rounds/0001/agent/requests/00001'
    write_json(trace/'request.json',{'arguments':{'messages':[{'role':'tool','content':'omit'}]}})
    write_json(trace/'before-fold.json',[{'role':'function','content':'original result'}])
    write_json(trace/'response.json',{'status':'completed','reasoning_content':'Check prior result.', 'finish_reasons':['length']})
    result={'state':{'current_round':1},'run':str(run),'updated_at':1,
            'agent':{'shell_calls':8,'duplicate_command_count':1,'context_recoveries':0,'failed_commands':[]},'alerts':[]}
    output=tmp_path/'monitor'
    review(result,output)
    assert result['agent']['omitted_tool_results']==1
    copies=list((output/'context-reviews').glob('*/response.json'))
    assert len(copies)==1 and read_json(copies[0])['reasoning_content']=='Check prior result.'
    review(result,output)
    assert len(list((output/'context-reviews').glob('*')))==1
