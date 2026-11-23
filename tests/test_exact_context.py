import copy
import json

import pytest

from vibeforge_v2.agent_context import NativeContext, count_request, groups


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return json.dumps([messages, kwargs], ensure_ascii=False)

    def __call__(self, text, **kwargs):
        return {'input_ids': list(text)}


def test_count_includes_reasoning_tools_and_normalizes_arguments():
    request = {'messages': [{'role': 'assistant', 'content': '', 'reasoning_content': 'thinking',
                'tool_calls': [{'function': {'name': 'shell', 'arguments': '{"command":"pwd"}'}}]}],
               'tools': [{'function': {'name': 'shell'}}]}
    original = copy.deepcopy(request)
    n = count_request(Tokenizer(), request)
    request['messages'][0]['reasoning_content'] += 'x' * 100
    assert count_request(Tokenizer(), request) == n + 100
    request['messages'][0]['reasoning_content'] = 'thinking'
    assert request == original
    del request['tools']
    assert count_request(Tokenizer(), request) < n


def test_compaction_preserves_task_latest_results_and_reuses_memory(tmp_path):
    calls = []
    def create(**request):
        calls.append(copy.deepcopy(request))
        assert count_request(Tokenizer(), request) + request['max_tokens'] + 100 <= 4500
        if not request['stream']:
            return {'choices': [{'finish_reason': 'stop', 'message': {'content': 'Measured 10 ms; file engine.py; next edit.'}}]}
        return 'response'
    manager = NativeContext(create, Tokenizer(), 4500, 100, tmp_path)
    messages = [{'role': 'system', 'content': 'rules'}, {'role': 'user', 'content': 'task'}]
    for i in range(10):
        messages += [{'role': 'assistant', 'content': '', 'tool_calls': [{'id': str(i), 'function': {
            'name': 'shell', 'arguments': '{"command":"pwd"}'}}]},
            {'role': 'tool', 'tool_call_id': str(i), 'content': str(i) + 'evidence' * 100}]
    original = copy.deepcopy(messages)
    request = {'messages': messages, 'model': 'test', 'stream': True, 'max_tokens': 500}
    assert manager(**request) == 'response'
    sent = calls[-1]['messages']
    assert sent[:2] == messages[:2] and sent[-4:] == messages[-4:]
    assert messages == original and manager.memory
    for group in groups(sent):
        if group[0].get('tool_calls'):
            assert group[1]['tool_call_id'] == group[0]['tool_calls'][0]['id']
    summaries = len([r for r in calls if not r['stream']])
    messages[2]['reasoning_content'] = 'historical reasoning changed by request policy'
    manager(**request)
    assert len([r for r in calls if not r['stream']]) == summaries


def test_oversized_latest_evidence_fails_without_silent_truncation(tmp_path):
    manager = NativeContext(lambda **k: pytest.fail('must not send'), Tokenizer(), 1000, 50, tmp_path)
    with pytest.raises(RuntimeError, match='No latest evidence'):
        manager(messages=[{'role': 'user', 'content': 'task'}, {'role': 'assistant', 'content': 'call'},
                          {'role': 'tool', 'content': 'x' * 2000}], max_tokens=200, model='x', stream=True)


def test_incomplete_summary_does_not_discard_history(tmp_path):
    manager = NativeContext(lambda **k: {'choices': [{'finish_reason': 'length', 'message': {'content': 'partial'}}]},
                            Tokenizer(), 4500, 100, tmp_path)
    messages = [{'role': 'user', 'content': 'task'}] + [
        {'role': 'assistant', 'content': 'x' * 1000} for _ in range(10)]
    with pytest.raises(RuntimeError, match='summary incomplete'):
        manager(messages=messages, max_tokens=500, model='x', stream=True)
    assert manager.covered == []


def test_official_tokenizer_matches_engine_chat_path():
    """Optional local integration: use the real engine template without loading weights."""
    import importlib.util
    from pathlib import Path
    from vibeforge_v2.agent_context import tokenizer_for
    from vibeforge_v2.config import ROOT
    model_path = '/home/yxy/weight/Qwen3.8-27B'
    if not Path(model_path).exists():
        pytest.skip('local serving tokenizer not installed')
    spec = importlib.util.spec_from_file_location('count_engine', ROOT / 'engines/transformers-qwen38/engine.py')
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    engine.TOKENIZER = tokenizer_for(model_path)
    engine.CONFIG = {'thinking': {'enabled': False}, 'max_new_tokens': 8192}
    messages = [{'role': 'system', 'content': 'Engineer'}, {'role': 'user', 'content': '检查引擎'},
                {'role': 'assistant', 'content': '', 'reasoning_content': '检查文件', 'tool_calls': [
                    {'id': '1', 'type': 'function', 'function': {'name': 'shell', 'arguments': '{"command":"pwd"}'}}]},
                {'role': 'tool', 'tool_call_id': '1', 'content': '/candidate'}]
    tools = [{'type': 'function', 'function': {'name': 'shell', 'parameters': {'type': 'object'}}}]
    actual = []
    def generate(prompt, **kwargs):
        actual.append(len(engine.TOKENIZER(prompt, return_token_type_ids=False)['input_ids']))
        return {'text': '</think>done', 'token_ids': [], 'input_ids': [], 'finish_reason': 'stop'}
    engine.generate = generate
    for thinking in (True, False):
        for choice in (None, 'required', 'none', {'type': 'function', 'function': {'name': 'shell'}}):
            request = {'messages': messages, 'tools': tools, 'tool_choice': choice,
                       'extra_body': {'chat_template_kwargs': {'enable_thinking': thinking}}}
            engine.chat_completion(copy.deepcopy(messages), tools=tools, tool_choice=choice,
                                   chat_template_kwargs={'enable_thinking': thinking})
            assert count_request(engine.TOKENIZER, request) == actual[-1]


def test_summary_retries_same_evidence_on_output_limit(tmp_path):
    requests = []
    def create(**request):
        requests.append(copy.deepcopy(request))
        return {'choices': [{'finish_reason': 'length' if len(requests) == 1 else 'stop',
                             'message': {'content': 'too long' if len(requests) == 1 else 'evidence; next edit'}}]}
    manager = NativeContext(create, Tokenizer(), 12000, 100, tmp_path)
    result = manager.summarize([{'role': 'tool', 'content': 'UNIQUE EVIDENCE'}], {'model': 'test'})
    assert result == 'evidence; next edit' and len(requests) == 2
    assert requests[0]['messages'][1] == requests[1]['messages'][1]


def test_summary_sees_old_plan_but_main_request_does_not_keep_old_thinking(tmp_path):
    calls = []
    def create(**request):
        calls.append(copy.deepcopy(request))
        if request['stream']:
            return 'response'
        return {'choices': [{'finish_reason': 'stop', 'message': {'content': 'Keep selected plan.'}}]}
    manager = NativeContext(create, Tokenizer(), 4500, 100, tmp_path)
    rows = [{'role': 'user', 'content': 'task'}]
    for i in range(8):
        rows.extend([{'role': 'assistant', 'content': '', 'reasoning_content': f'SELECTED PLAN {i}',
                      'tool_calls': [{'id': str(i), 'function': {'name': 'shell', 'arguments': '{}'}}]},
                     {'role': 'tool', 'tool_call_id': str(i), 'content': 'data' * 300}])
    manager(messages=rows, model='test', max_tokens=500, stream=True)
    summaries = [c for c in calls if not c['stream']]
    assert any('SELECTED PLAN 0' in c['messages'][1]['content'] for c in summaries)
    assert sum(bool(m.get('reasoning_content')) for m in calls[-1]['messages']) == 1
    assert rows[1]['reasoning_content'] == 'SELECTED PLAN 0'


def test_summary_receives_original_host_facts_not_just_search_results(tmp_path):
    calls = []
    def create(**request):
        calls.append(copy.deepcopy(request))
        return {'choices': [{'finish_reason': 'stop', 'message': {'content': 'Observed module is active.'}}]}
    manager = NativeContext(create, Tokenizer(), 12000, 100, tmp_path)
    manager.memory = 'Incorrect previous inference: no class name means inactive.'
    task = 'HOST OBSERVED: dynamic module forward ran; optimize the request path.'
    request = {'model': 'test', 'messages': [{'role': 'system', 'content': 'Keep interfaces'},
               {'role': 'user', 'content': task}, {'role': 'assistant', 'content': 'OLD HISTORY'}]}
    rows = [{'role': 'tool', 'content': 'grep returned no matching text'}]
    manager.summarize(rows, request)
    content = calls[0]['messages'][1]['content']
    assert task in content and 'Keep interfaces' in content
    assert 'Incorrect previous inference' in content and 'grep returned' in content
    assert 'OLD HISTORY' not in content
    assert count_request(Tokenizer(), calls[0]) + calls[0]['max_tokens'] + 100 <= 12000


def test_extractive_memory_never_generates_conclusions_and_archives_full_evidence(tmp_path):
    manager = NativeContext(lambda **k: pytest.fail('extractive memory must not call model'),
                            Tokenizer(), 12000, 100, tmp_path, memory_max_tokens=1200, extractive_memory=True)
    rows = [{'role': 'assistant', 'content': 'CLAIM: module inactive'},
            {'role': 'assistant', 'tool_calls': [{'id': 'a', 'function': {'name': 'shell', 'arguments': '{"command":"grep Class engine.py"}'}}]},
            {'role': 'tool', 'content': 'NO MATCH; full source=' + 'x' * 8000 + '; NO NEW FILE CREATED'}]
    memory = json.loads(manager.summarize(rows, {'model': 'test'}))
    assert len(json.dumps(memory, ensure_ascii=False)) <= 1200
    assert memory['excerpts'][-1]['kind'] == 'observed_tool_result'
    assert 'NO NEW FILE CREATED' in memory['excerpts'][-1]['content_excerpt']
    archive = json.loads((tmp_path/'context/extract-0001.json').read_text())
    assert archive['source_messages'] == rows
    assert 'optimized_moe.py' not in json.dumps(memory)


def test_extractive_memory_labels_agent_claims_and_retains_across_compactions(tmp_path):
    manager = NativeContext(lambda **k: pytest.fail('must not call model'), Tokenizer(), 12000, 100,
                            tmp_path, memory_max_tokens=2000, extractive_memory=True)
    manager.memory = manager.summarize([{'role':'assistant','content':'Hypothesis, not fact'}], {})
    manager.memory = manager.summarize([{'role':'tool','content':'Measured result'}], {})
    entries = json.loads(manager.memory)['excerpts']
    assert [x['kind'] for x in entries] == ['unverified_agent_statement','observed_tool_result']
