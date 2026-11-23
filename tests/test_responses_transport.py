"""Deterministic protocol tests; these do not count as model successes."""
import json
import pytest

from vf_runtime.engine.responses import response_events, response_object, responses_to_chat


def test_namespace_custom_tool_and_feedback_preserve_text():
    raw = '*** Begin Patch\n*** Add File: a.py\n+text = "slash \\\\ and quote \\""\n*** End Patch\n'
    request = {'instructions': 'system', 'input': [
        {'role': 'developer', 'content': [{'type': 'input_text', 'text': 'guide'}]},
        {'role': 'user', 'content': 'edit'},
        {'type': 'custom_tool_call', 'name': 'apply_patch', 'namespace': 'functions', 'call_id': 'c1', 'input': raw},
        {'type': 'custom_tool_call_output', 'call_id': 'c1', 'output': 'SyntaxError: real feedback'},
    ], 'tools': [{'type': 'namespace', 'name': 'functions', 'tools': [
        {'type': 'custom', 'name': 'apply_patch', 'description': 'Patch',
         'format': {'type': 'grammar', 'syntax': 'lark', 'definition': 'start: "patch"'}}]}]}
    chat = responses_to_chat(request)
    assert chat['messages'][0] == {'role': 'system', 'content': 'system\n\nguide'}
    call = chat['messages'][2]['tool_calls'][0]
    assert call['function']['name'] == 'functions__apply_patch'
    assert json.loads(call['function']['arguments'])['input'] == raw
    assert chat['messages'][3] == {'role': 'tool', 'tool_call_id': 'c1', 'content': 'SyntaxError: real feedback'}
    response = response_object(request, {'tool_calls': [{'name': 'functions__apply_patch',
        'arguments': json.dumps({'input': raw})}], 'finish_reason': 'stop'})
    output = response['output'][0]
    assert output['type'] == 'custom_tool_call' and output['input'] == raw
    assert output['namespace'] == 'functions' and output['name'] == 'apply_patch'
    events = list(response_events(response))
    assert events[-1]['type'] == 'response.completed'
    assert next(e for e in events if e['type'] == 'response.custom_tool_call_input.done')['input'] == raw
    assert [e['sequence_number'] for e in events] == list(range(len(events)))


@pytest.mark.parametrize('finish,status,event', [('length','incomplete','response.incomplete'),
    ('incomplete_thinking','failed','response.failed'), ('stop','completed','response.completed')])
def test_truncation_is_not_reported_as_success(finish, status, event):
    response = response_object({}, {'finish_reason': finish, 'usage': {
        'prompt_tokens': 10, 'completion_tokens': 20, 'completion_tokens_details': {'reasoning_tokens': 12}}})
    assert response['status'] == status
    assert response['usage']['output_tokens_details']['reasoning_tokens'] == 12
    assert list(response_events(response))[-1]['type'] == event


def test_reasoning_and_parallel_calls_roundtrip_as_one_assistant_turn():
    thought = 'Local fixture reasoning: quote " slash \\ newline\nend'
    request = {'tools': [{'type':'function','name':'shell','parameters':{'type':'object'}}]}
    result = {'reasoning_content': thought, 'text':'Checking files', 'tool_calls': [
        {'name':'shell','arguments':'{"cmd":"one"}'}, {'name':'shell','arguments':'{"cmd":"two"}'}]}
    response = response_object(request, result)
    assert response['output'][0]['type'] == 'reasoning'
    events = list(response_events(response))
    assert next(e['item'] for e in events if e['type']=='response.output_item.done')['content'][0]['text'] == thought
    history = [{'role':'user','content':'task'}, *response['output'],
        {'type':'function_call_output','call_id':response['output'][2]['call_id'],'output':'one output'},
        {'type':'function_call_output','call_id':response['output'][3]['call_id'],'output':'two output'}]
    messages = responses_to_chat({'input':history})['messages']
    assert len(messages) == 4
    assert messages[1]['reasoning_content'] == thought
    assert messages[1]['content'] == 'Checking files'
    assert len(messages[1]['tool_calls']) == 2
    assert messages[2]['content'] == 'one output'
    assert messages[3]['content'] == 'two output'
