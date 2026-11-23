"""Actual official Assistant over HTTP, with real shell failures and edits."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from vibeforge_v2.agent import Shell, edit_candidate
from vibeforge_v2.config import ROOT, load


def test_shell_confines_writes_and_preserves_errors(tmp_path):
    workspace = tmp_path / 'candidate'
    workspace.mkdir()
    tool = Shell(workspace, tmp_path / 'trace')
    outside = tmp_path / 'outside.txt'
    result = tool.call(json.dumps({'command': f"printf outside > '{outside}'; printf REAL_FAILURE >&2; exit 7"}))
    assert not outside.exists()
    assert 'Permission denied' in result
    assert 'exit_code: 7' in result and 'REAL_FAILURE' in result
    assert tool.calls[0]['exit_code'] == 7


def test_shell_keeps_runtime_python_and_project_imports_from_candidate(tmp_path):
    import sys
    workspace = tmp_path / 'candidate'
    workspace.mkdir()
    tool = Shell(workspace, tmp_path / 'trace', python=sys.executable)
    result = tool.call(json.dumps({'command': 'python -c "import vf_runtime,sys; print(sys.executable)"'}))
    assert 'exit_code: 0' in result
    assert sys.executable in result


def test_official_framework_repairs_after_real_tool_error(tmp_path):
    requests = []
    replies = [
        '<tool_call>\n' + json.dumps({'name': 'shell', 'arguments': {
            'command': 'printf ACTUAL_ERROR_42 >&2; exit 7'}}) + '\n</tool_call>',
        '<tool_call>\n' + json.dumps({'name': 'shell', 'arguments': {
            'command': "printf 'def answer(): return 42\\n' > fixed.py"}}) + '\n</tool_call>',
        '检查实际错误后完成了修改。无需提交 JSON。',
    ]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests.append(request)
            index = len(requests) - 1
            chunk = {'id': f'test-{index}', 'object': 'chat.completion.chunk', 'created': 1,
                     'model': 'test', 'choices': [{'index': 0, 'delta': {'content': replies[index]},
                                                 'finish_reason': None}]}
            body = ('data: ' + json.dumps(chunk, ensure_ascii=False) + '\n\ndata: [DONE]\n\n').encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    workspace = tmp_path / 'candidate'
    workspace.mkdir()
    (workspace / 'engine.py').write_text('def answer(): return 0\n')
    (workspace / 'engine.json').write_text('{"backend":"toy"}')
    try:
        config = load(ROOT / 'configs/toy.json')
        config['agent']['trace_requests'] = True
        result = edit_candidate(f'http://127.0.0.1:{server.server_port}', config, workspace,
                                tmp_path / 'round/agent', 'Inspect feedback, implement, and explain normally.')
        assert result['tool_calls'] == 2
        assert 'return 42' in (workspace / 'fixed.py').read_text()
        assert len(requests) == 3
        traced = json.loads((tmp_path / 'round/agent/requests/00002/request.json').read_text())
        assert traced['arguments']['messages'] == requests[1]['messages']
        assert all(not r.get('tools') for r in requests)
        assert any('ACTUAL_ERROR_42' in (m.get('content') or '') and 'exit_code: 7' in (m.get('content') or '')
                   for m in requests[1]['messages'])
        assert '无需提交 JSON' in (tmp_path / 'round/analysis.md').read_text()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_partial_conversation_survives_model_failure(tmp_path):
    import pytest
    from vibeforge_v2.agent import exchange
    class BrokenModel:
        def run(self, messages):
            yield [{'role': 'assistant', 'content': '已观察瓶颈，正在验证。'}]
            raise RuntimeError('actual model failure')
    with pytest.raises(RuntimeError, match='actual model failure'):
        exchange(BrokenModel(), [{'role': 'user', 'content': 'task'}], tmp_path)
    saved = json.loads((tmp_path / 'conversation.json').read_text())
    assert saved[-1]['content'] == '已观察瓶颈，正在验证。'


def test_framework_tool_step_boundary_continues_without_a_v2_turn_cap(tmp_path):
    from vibeforge_v2.agent import exchange
    class Bot:
        calls = 0
        def run(self, messages):
            self.calls += 1
            if self.calls == 1:
                yield [{'role': 'assistant', 'function_call': {'name': 'shell', 'arguments': '{"command":"true"}'}},
                       {'role': 'function', 'name': 'shell', 'content': 'exit_code: 0'}]
            else:
                assert messages[-1]['role'] == 'function'
                yield [{'role': 'assistant', 'content': '完成实际修改。'}]
    bot = Bot()
    result = exchange(bot, [{'role': 'user', 'content': 'task'}], tmp_path)
    assert bot.calls == 2 and result[-1]['content'] == '完成实际修改。'


def test_qwen3_template_receives_tool_chain_reasoning_without_adapter_deletion():
    import importlib.util
    spec = importlib.util.spec_from_file_location('template', ROOT / 'engines/transformers/engine.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    received = []
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            received.extend(messages)
            return 'test prompt'
        def encode(self, text, **kwargs):
            return [1]
    module.TOKENIZER = Tokenizer()
    module.CONFIG = {'max_new_tokens': 32, 'thinking': {'enabled': False}}
    module.generate = lambda *args, **kwargs: {'text': 'ready', 'token_ids': [2], 'input_ids': [1]}
    messages = [{'role': 'user', 'content': 'task'},
                {'role': 'assistant', 'content': 'actual tool call', 'reasoning_content': 'old internal thinking'},
                {'role': 'user', 'content': '<tool_response>actual error</tool_response>'}]
    result = module.chat_completion(messages, max_tokens=32)
    assert received[1]['content'] == 'actual tool call'
    assert received[1]['reasoning_content'] == 'old internal thinking'
    assert 'actual error' in received[2]['content']
    assert messages[1]['reasoning_content'] == 'old internal thinking'
    assert result['text'] == 'ready'


def test_incomplete_thinking_is_not_silently_regenerated_without_thinking():
    import importlib.util
    spec = importlib.util.spec_from_file_location('thinking_template', ROOT / 'engines/transformers/engine.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            calls.append(kwargs['enable_thinking'])
            return 'prompt'

        def encode(self, text, **kwargs):
            return [99]

    module.TOKENIZER = Tokenizer()
    module.CONFIG = {'max_new_tokens': 32, 'thinking': {'enabled': True}}
    module.generate = lambda *args, **kwargs: {
        'text': 'Unfinished real reasoning', 'token_ids': [1, 2], 'input_ids': [3], 'finish_reason': 'length'}
    result = module.chat_completion([{'role': 'user', 'content': 'task'}], max_tokens=32)
    assert calls == [True]
    assert result['reasoning_content'] == 'Unfinished real reasoning'
    assert result['text'] == ''
    assert result['finish_reason'] == 'incomplete_thinking'
    assert result['thinking']['complete'] is False
    assert result['thinking']['recovered_without_thinking'] is False
