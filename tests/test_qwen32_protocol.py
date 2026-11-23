"""Protocol fixtures only: no model inference or optimization-success claims."""
import copy
import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from vibeforge_v2.agent import _check_qwen_connection, exchange
from vibeforge_v2.config import ROOT, load


@pytest.fixture
def engine():
    spec = importlib.util.spec_from_file_location('qwen32_protocol_engine', ROOT / 'engines/transformers/engine.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.CONFIG = {'thinking': {'enabled': False}}
    module.TOKENIZER = SimpleNamespace(apply_chat_template=lambda *a, **k: 'prompt')
    return module


TOOLS = [{'type': 'function', 'function': {'name': 'shell', 'parameters': {
    'type': 'object', 'properties': {'command': {'type': 'string'}}, 'required': ['command']}}}]
GOOD = '<tool_call>{"name":"shell","arguments":{"command":"printf OK"}}</tool_call>'


@pytest.mark.parametrize('bad', [
    '<tool_call>{broken}</tool_call>',
    '<tool_call>{"name":"shell","arguments":',
    '<tool_call>{"name":"missing","arguments":{}}</tool_call>',
    *['<tool_call>' + json.dumps({'name': 'shell', 'arguments': args}) + '</tool_call>'
      for args in ('printf BAD', [], None, 3)],
])
def test_bad_batch_never_exposes_valid_prefix(engine, bad):
    # Include valid prefix/suffix to catch silently skipped malformed calls.
    for raw in (bad, GOOD + bad, bad + GOOD):
        engine.generate = lambda *a, **k: {'text': raw, 'token_ids': [1], 'input_ids': [2]}
        result = engine.chat_completion([{'role': 'user', 'content': 'inspect'}], tools=TOOLS)
        assert result['tool_calls'] == [] and result['text'] == raw
        assert result['tool_parse_error']


def test_unwrapped_json_batch_is_atomic(engine):
    raw = json.dumps({'tool_calls': [{'name': 'shell', 'arguments': {}},
                                    {'name': 'shell', 'arguments': '{'}]})
    engine.generate = lambda *a, **k: {'text': raw, 'token_ids': [1], 'input_ids': [2]}
    result = engine.chat_completion([{'role': 'user', 'content': 'inspect'}], tools=TOOLS)
    assert result['tool_calls'] == [] and result['text'] == raw
    assert result['tool_parse_error']


def test_tool_markers_inside_json_command_are_data(engine):
    command = "printf '<tool_call></tool_call>'"
    raw = '<tool_call>' + json.dumps({'name': 'shell', 'arguments': {'command': command}}) + '</tool_call>'
    assert engine.parse_tool_calls(raw, TOOLS) == [{'name': 'shell', 'arguments': {'command': command}}]


def test_parallel_valid_calls_and_caller_history_preserved(engine):
    engine.generate = lambda *a, **k: {'text': GOOD + GOOD, 'token_ids': [1], 'input_ids': [2]}
    messages = [{'role': 'assistant', 'content': None, 'tool_calls': [{
        'function': {'name': 'shell', 'arguments': '{"command":"pwd"}'}}]},
        {'role': 'tool', 'content': 'evidence'}]
    original = copy.deepcopy(messages)
    result = engine.chat_completion(messages, tools=TOOLS)
    assert len(result['tool_calls']) == 2
    assert messages == original
    assert all(isinstance(json.loads(c['arguments']), dict) for c in result['tool_calls'])


def test_real_agent_readiness_recovers_without_poisoning_next_request(engine, tmp_path):
    """Official Assistant -> HTTP -> engine parser -> shell -> next HTTP request."""
    requests, raw_outputs = [], []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests.append(request)
            if len(requests) == 1:
                raw = '<tool_call>{"name":"shell","arguments":"printf UNEXECUTED"}</tool_call>'
            elif len(requests) == 2:
                # Readiness embeds an exact nonce command in the initial user turn.
                command = request['messages'][1]['content'].split('\n')[1]
                raw = '<tool_call>' + json.dumps({'name': 'shell', 'arguments': {'command': command}}) + '</tool_call>'
            else:
                raw = 'ready'
            raw_outputs.append(raw)
            engine.generate = lambda *a, **k: {'text': raw, 'token_ids': [1], 'input_ids': [2]}
            result = engine.chat_completion(request['messages'], tools=request.get('tools'),
                                            **{k: v for k, v in request.items() if k in ('chat_template_kwargs',)})
            delta = {'content': result['text']}
            if result['tool_calls']:
                delta['tool_calls'] = [{'index': i, 'id': f'call_{i}', 'type': 'function',
                                       'function': c} for i, c in enumerate(result['tool_calls'])]
            body = ('data: ' + json.dumps({'id': 'fixture', 'object': 'chat.completion.chunk',
                    'created': 1, 'model': 'fixture', 'choices': [{'index': 0, 'delta': delta,
                    'finish_reason': None}]}) + '\n\ndata: [DONE]\n\n').encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    config = load(ROOT / 'configs/toy.json')
    config['agent'].update(use_raw_api=True, trace_requests=True)
    try:
        report = _check_qwen_connection(f'http://127.0.0.1:{server.server_port}', config, tmp_path, tmp_path / 'connection')
        assert report['passed'] and report['tool_calls'] == 1
        assert len(requests) == 3
        assert raw_outputs[0] in str(requests[1]['messages'])
        assert not any(m.get('tool_calls') for m in requests[1]['messages'])
        history = requests[2]['messages']
        assert any(m['role'] == 'tool' and report['challenge'] in m['content'] for m in history)
        for message in history:
            for call in message.get('tool_calls', []):
                assert isinstance(json.loads(call['function']['arguments']), dict)
        assert (tmp_path / 'connection/tool-repairs.jsonl').exists()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_repeated_invalid_output_ends_as_protocol_failure(tmp_path):
    class Bot:
        def run(self, messages):
            yield [{'role': 'assistant', 'content': '<tool_call>{broken}'}]
    messages = [{'role': 'user', 'content': 'original task'}]
    with pytest.raises(RuntimeError, match='unexecuted tool text'):
        exchange(Bot(), messages, tmp_path)
    assert len((tmp_path / 'tool-repairs.jsonl').read_text().splitlines()) == 4
    assert messages[0]['content'] == 'original task'
