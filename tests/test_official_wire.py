"""Compare the project factory with official Assistant at the actual HTTP boundary."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from qwen_agent.agents import Assistant

from vibeforge_v2.agent import Shell, assistant
from vibeforge_v2.config import ROOT, load


@pytest.mark.parametrize('raw_thinking', [False, True])
def test_factory_matches_official_thinking_and_tool_history(tmp_path, raw_thinking):
    captures = []
    phantom = '<tool_call>' + json.dumps({'name': 'shell', 'arguments': {
        'command': 'printf PHANTOM_MUST_NOT_EXECUTE'}}) + '</tool_call>'
    thought = 'Consider this hypothetical call, but do not execute it: ' + phantom
    call = '<tool_call>' + json.dumps({'name': 'shell', 'arguments': {
        'command': 'printf REAL_ERROR_73 >&2; exit 7'}}) + '</tool_call>'

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            captures.append(request)
            content = call if len(captures) == 1 else 'Observed exit code 7; task complete.'
            # Qwen3's template already emits the opening <think> in the prompt;
            # raw servers can therefore return only the thought and closing tag.
            delta = {'content': thought + '</think>' + content} if raw_thinking else {
                'reasoning_content': thought, 'content': content}
            chunk = {'id': 'wire-test', 'object': 'chat.completion.chunk', 'created': 1,
                     'model': 'test', 'choices': [{'index': 0, 'delta': delta, 'finish_reason': None}]}
            body = ('data: ' + json.dumps(chunk) + '\n\ndata: [DONE]\n\n').encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    config = load(ROOT / 'configs/toy.json')
    config['agent']['thinking'] = {'enabled': True, 'temperature': .6, 'top_p': .95, 'top_k': 20}
    config['agent']['thought_in_content'] = raw_thinking
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{server.server_port}'
    requests, calls = [], []
    try:
        for kind in ('official', 'project'):
            captures.clear()
            workspace = tmp_path / kind
            workspace.mkdir()
            tool = Shell(workspace, tmp_path / (kind + '-trace'))
            if kind == 'official':
                settings = config['agent']
                generate = {'max_tokens': settings['max_new_tokens'],
                            'max_input_tokens': config['model']['max_context'] - settings['max_new_tokens'],
                            'request_timeout': settings['request_timeout'],
                            'temperature': .6, 'top_p': .95, 'top_k': 20,
                            'thought_in_content': raw_thinking,
                            'extra_body': {'min_p': 0, 'chat_template_kwargs': {'enable_thinking': True}}}
                bot = Assistant(llm={'model': config['model']['name'], 'model_server': url + '/v1',
                                     'api_key': 'EMPTY', 'generate_cfg': generate},
                                function_list=[tool], system_message='Use tools and preserve evidence.')
            else:
                bot = assistant(url, config, tool, 'Use tools and preserve evidence.')
            for _ in bot.run(messages=[{'role': 'user', 'content': 'Execute the real command.'}]):
                pass
            # The official framework generates a fresh seed when none is supplied.
            requests.append([{k: v for k, v in row.items() if k != 'seed'} for row in captures])
            calls.append(list(tool.calls))
        (tmp_path / 'wire.json').write_text(json.dumps(requests, ensure_ascii=False, indent=2))
        assert requests[0] == requests[1], str(tmp_path / 'wire.json')
        for rows in calls:
            assert len(rows) == 1
            assert rows[0]['exit_code'] == 7
            assert rows[0]['output'] == 'REAL_ERROR_73'
        history = requests[1][1]['messages']
        assert any('exit_code: 7' in m.get('content', '') and 'REAL_ERROR_73' in m.get('content', '')
                   for m in history)
        assert any(thought in (m.get('reasoning_content', '') + m.get('content', ''))
                   for m in history if m['role'] == 'assistant')
        assert all(not request.get('tools') for request in requests[1])
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
