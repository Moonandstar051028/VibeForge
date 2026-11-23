"""Exercise the official raw-API switch at the real HTTP boundary."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from vibeforge_v2.agent import Shell, assistant
from vibeforge_v2.config import ROOT, load


def test_native_tools_return_real_error_and_preserve_reasoning(tmp_path):
    captures = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            captures.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            if len(captures) == 1:
                delta = {'reasoning_content': 'Inspect the actual failure after running the command.',
                         'tool_calls': [{'index': 0, 'id': 'call_native', 'type': 'function',
                                         'function': {'name': 'shell', 'arguments': json.dumps({
                                             'command': 'printf NATIVE_REAL_ERROR >&2; exit 7'})}}]}
            else:
                delta = {'content': 'Observed the actual exit code 7.'}
            chunk = {'id': 'native-wire', 'object': 'chat.completion.chunk', 'created': 1,
                     'model': 'test', 'choices': [{'index': 0, 'delta': delta, 'finish_reason': None}]}
            body = ('data: ' + json.dumps(chunk) + '\n\ndata: [DONE]\n\n').encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    config = load(ROOT / 'configs/toy.json')
    config['agent'].update(use_raw_api=True, thinking={'enabled': False},
                           sampling={'temperature': .7, 'top_p': .8, 'top_k': 20,
                                     'repetition_penalty': 1.05})
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        tool = Shell(tmp_path, tmp_path / 'trace')
        bot = assistant(f'http://127.0.0.1:{server.server_port}', config, tool, 'Use the real shell.')
        for _ in bot.run(messages=[{'role': 'user', 'content': 'Run the diagnostic.'}]):
            pass
        assert len(captures) == 2
        assert len(tool.calls) == 1
        assert tool.calls[0]['exit_code'] == 7
        assert tool.calls[0]['output'] == 'NATIVE_REAL_ERROR'
        first = captures[0]
        assert first['tools'][0]['function']['name'] == 'shell'
        schema = first['tools'][0]['function']['parameters']
        assert schema['type'] == 'object' and schema['required'] == ['command']
        assert schema['properties']['command']['type'] == 'string'
        assert first['temperature'] == .7 and first['top_p'] == .8
        assert first['top_k'] == 20 and first['repetition_penalty'] == 1.05
        history = captures[1]['messages']
        assert any(row['role'] == 'tool' and 'exit_code: 7' in row['content']
                   and 'NATIVE_REAL_ERROR' in row['content'] for row in history)
        assert any(row['role'] == 'assistant' and row.get('tool_calls') for row in history)
        assert any('Inspect the actual failure' in row.get('reasoning_content', '') for row in history)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_non_thinking_model_explicit_sampling_is_not_discarded():
    import importlib.util
    from types import SimpleNamespace
    path = ROOT / 'engines/transformers/engine.py'
    spec = importlib.util.spec_from_file_location('sampling_engine', path)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    engine.TOKENIZER = SimpleNamespace(eos_token_id=2)
    engine.CONFIG = {'max_new_tokens': 8192, 'thinking': {'enabled': False,
                     'temperature': .7, 'top_p': .8, 'top_k': 20, 'repetition_penalty': 1.05}}
    enabled, sampling = engine._thinking_settings()
    assert not enabled
    options = engine._generation_options(100, False, sampling)
    assert options['do_sample'] and options['temperature'] == .7
    assert options['top_p'] == .8 and options['top_k'] == 20
    assert options['repetition_penalty'] == 1.05
    greedy = engine._generation_options(100, False, {'temperature': 0, 'repetition_penalty': 1.05})
    assert not greedy['do_sample'] and greedy['repetition_penalty'] == 1.05
    engine.CONFIG['thinking'] = {'enabled': False}
    assert engine._thinking_settings() == (False, {})


def test_malformed_model_call_is_visible_without_fabricated_execution(capsys):
    import importlib.util
    from types import SimpleNamespace
    spec = importlib.util.spec_from_file_location('raw_output_engine', ROOT / 'engines/transformers/engine.py')
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    engine.CONFIG = {'max_new_tokens': 100, 'thinking': {'enabled': False}}
    engine.TOKENIZER = SimpleNamespace(apply_chat_template=lambda *args, **kwargs: 'prompt')
    raw = '<tool_call>{"name":"shell","arguments":{"command":"unterminated'
    engine.generate = lambda *args, **kwargs: {'text': raw, 'token_ids': [1, 2],
                                              'input_ids': [3], 'finish_reason': 'length'}
    result = engine.chat_completion([{'role': 'user', 'content': 'edit'}], tools=[{
        'type': 'function', 'function': {'name': 'shell', 'parameters': {'type': 'object'}}}])
    assert result['text'] == raw and result['tool_calls'] == []
    assert result['finish_reason'] == 'length' and result['tool_parse_error']
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert next(row for row in events if row['event'] == 'tool_parse_error')['raw_text'] == raw
    code = '<tool_call>{"command":"print(\'</think>\')"}</tool_call>'
    assert engine.split_reasoning(code, [1, 2, 3], False) == ('', code, 0, 3, True)


def test_native_agent_can_correct_unexecuted_call_from_actual_feedback(tmp_path):
    from vibeforge_v2.agent import edit_candidate
    captures = []
    command = ('python -c \'import json; from pathlib import Path; p=Path("engine.json"); '
               'x=json.loads(p.read_text()); x["use_cache"]=False; p.write_text(json.dumps(x))\'')

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            captures.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            n = len(captures)
            if n == 1:
                delta = {'content': '<tool_call>{invalid model-generated call'}
            elif n == 2:
                delta = {'tool_calls': [{'index': 0, 'id': 'call_fixed', 'type': 'function',
                                         'function': {'name': 'shell', 'arguments': json.dumps({'command': command})}}]}
            else:
                delta = {'content': 'The actual candidate file has now changed.'}
            chunk = {'id': 'repair-wire', 'object': 'chat.completion.chunk', 'created': 1,
                     'model': 'test', 'choices': [{'index': 0, 'delta': delta, 'finish_reason': None}]}
            body = ('data: ' + json.dumps(chunk) + '\n\ndata: [DONE]\n\n').encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    workspace = tmp_path / 'candidate'
    workspace.mkdir()
    (workspace / 'engine.json').write_text('{"backend":"toy"}')
    (workspace / 'engine.py').write_text('def answer(): return 0\n')
    config = load(ROOT / 'configs/toy.json')
    config['agent'].update(use_raw_api=True, config_tool=False)
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = edit_candidate(f'http://127.0.0.1:{server.server_port}', config, workspace,
                                tmp_path / 'round/agent', 'Actually change the candidate.')
        assert len(captures) == 3 and result['tool_calls'] == 1
        assert json.loads((workspace / 'engine.json').read_text())['use_cache'] is False
        assert any('文本中的命令没有执行' in row.get('content', '') for row in captures[1]['messages'])
        assert any('invalid model-generated call' in row.get('content', '') for row in captures[1]['messages'])
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_string_tool_arguments_stay_visible_text_instead_of_poisoning_history(capsys):
    import importlib.util
    from types import SimpleNamespace
    for directory in ('transformers', 'transformers-qwen38'):
        spec = importlib.util.spec_from_file_location('invalid_arguments_engine', ROOT / 'engines' / directory / 'engine.py')
        engine = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(engine)
        engine.CONFIG = {'max_new_tokens': 100, 'thinking': {'enabled': False}}
        engine.TOKENIZER = SimpleNamespace(apply_chat_template=lambda *a, **k: 'prompt')
        for arguments in ('sed -n 1,20p engine.py', [], None, 3):
            raw = '<tool_call>' + json.dumps({'name': 'shell', 'arguments': arguments}) + '</tool_call>'
            engine.generate = lambda *a, **k: {'text': raw, 'token_ids': [1], 'input_ids': [2], 'finish_reason': 'stop'}
            result = engine.chat_completion([{'role': 'user', 'content': 'inspect'}], tools=[{
                'type': 'function', 'function': {'name': 'shell', 'parameters': {'type': 'object'}}}])
            assert result['tool_calls'] == [] and result['text'] == raw
            assert result['tool_parse_error']
