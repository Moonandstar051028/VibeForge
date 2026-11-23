"""Independent disk errors and real official-Assistant recovery paths."""
import contextlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from vibeforge_v2.agent import Shell, edit_candidate
from vibeforge_v2.checks import CandidateCheck
from vibeforge_v2.config import ROOT, load
from qwen_agent.tools.base import ToolServiceError


def engine(tmp_path):
    workspace = tmp_path / 'engine'
    workspace.mkdir()
    (workspace / 'engine.py').write_text('def load(config): pass\ndef generate(): return 1\n')
    (workspace / 'engine.json').write_text('{"backend":"toy","max_context":128}')
    return workspace


@contextlib.contextmanager
def server(replies):
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            index = len(requests) - 1
            content = replies[index]
            if isinstance(content, tuple):
                body = json.dumps({'error': {'message': content[1], 'type': 'invalid_request_error',
                                            'code': 'context_length_exceeded'}}).encode()
                self.send_response(content[0])
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            event = {'id': str(index), 'object': 'chat.completion.chunk', 'created': 1, 'model': 'fixture',
                     'choices': [{'index': 0, 'delta': content if isinstance(content, dict) else {'content': content}, 'finish_reason': None}]}
            body = ('data: ' + json.dumps(event) + '\n\ndata: [DONE]\n\n').encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    http = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{http.server_port}', requests
    finally:
        http.shutdown()
        http.server_close()
        thread.join()


def test_command_success_does_not_hide_invalid_actual_configuration(tmp_path):
    workspace = engine(tmp_path)
    checker = CandidateCheck(workspace, tmp_path / 'checks')
    tool = Shell(workspace, tmp_path / 'agent', feedback=checker)
    result = tool.call(json.dumps({'command': "printf '\n{}' >> engine.json"}))
    assert tool.calls[-1]['exit_code'] == 0 and tool.calls[-1]['output'] == ''
    assert 'ERROR:' in result and 'Extra data' in result
    assert not tool.calls[-1]['candidate_check']['ready_for_host_evaluation']


def test_static_validation_checks_all_python_and_fixed_fields(tmp_path):
    workspace = engine(tmp_path)
    checker = CandidateCheck(workspace, tmp_path / 'checks')
    (workspace / 'helper.py').write_text('def broken(:\n')
    assert any('SyntaxError' in row for row in checker.inspect()['errors'])
    (workspace / 'helper.py').unlink()
    (workspace / 'engine.json').write_text('{"backend":"toy","max_context":256}')
    assert any('fixed model/deployment field changed: max_context' in row for row in checker.inspect()['errors'])
    (workspace / 'engine.json').write_text('{"backend":"toy","max_context":128,"microbatch_max_size":"x"}')
    assert any('positive integer' in row for row in checker.inspect()['errors'])
    (workspace / 'engine.json').write_text('{"backend":"toy","backend":"other"}')
    assert any('duplicate JSON key' in row for row in checker.inspect()['errors'])
    (workspace / 'engine.json').write_text('[]')
    assert not checker.inspect()['static_passed']


def test_suspicious_new_keys_are_visible_without_banning_extensions(tmp_path):
    workspace = engine(tmp_path)
    checker = CandidateCheck(workspace, tmp_path / 'checks')
    (workspace / 'engine.json').write_text('{"backend":"toy","max_context":128,"request_sheduler":"microbatch"}')
    report = checker.inspect()
    assert report['static_passed']
    assert any('request_sheduler' in row and 'spelling' in row for row in report['warnings'])


def test_shell_blocks_host_edits_before_integrity_is_lost(tmp_path):
    workspace = engine(tmp_path)
    protected = tmp_path / 'host'
    protected.mkdir()
    source = protected / 'runtime.py'
    source.write_text('state = False\n')
    checker = CandidateCheck(workspace, tmp_path / 'checks', protected=[protected])
    tool = Shell(workspace, tmp_path / 'agent', feedback=checker)
    result = tool.call(json.dumps({'command': f"sed -i 's/False/true/' {source}"}))
    assert source.read_text() == 'state = False\n'
    assert 'Permission denied' in result
    record = tool.calls[-1]
    assert record['exit_code'] != 0
    assert record['candidate_check']['protected_changed_files'] == []
    assert not record['candidate_check']['ready_for_host_evaluation']


def test_plan_only_answer_is_returned_to_official_agent_for_actual_edits(tmp_path):
    workspace = engine(tmp_path)
    call = '<tool_call>\n' + json.dumps({'name': 'shell', 'arguments': {
        'command': "python -c \"from pathlib import Path; p=Path('engine.py'); p.write_text(p.read_text().replace('return 1','return 2'))\""}}) + '\n</tool_call>'
    with server(['已完成修改。', call, '已实际写入并收到宿主静态检查通过。']) as (url, requests):
        result = edit_candidate(url, load(ROOT / 'configs/toy.json'), workspace,
                                tmp_path / 'round/agent', '实际修改引擎并核对文件。')
    assert result['tool_calls'] == 1 and len(requests) == 3
    assert 'return 2' in (workspace / 'engine.py').read_text()
    assert any('actual runtime changes: NONE' in (m.get('content') or '') for m in requests[1]['messages'])
    assert json.loads((tmp_path / 'round/agent/completion.json').read_text())['ready_for_host_evaluation']


def test_repeated_unchanged_answer_is_incomplete_instead_of_submission(tmp_path):
    workspace = engine(tmp_path)
    with server(['建议修改。', '已完成。']) as (url, requests):
        with pytest.raises(RuntimeError, match='same incomplete candidate'):
            edit_candidate(url, load(ROOT / 'configs/toy.json'), workspace,
                           tmp_path / 'round/agent', '实际修改引擎。')
    assert len(requests) == 2
    report = json.loads((tmp_path / 'round/agent/completion.json').read_text())
    assert not report['ready_for_host_evaluation']
    assert report['runtime_changed_files'] == []


@pytest.mark.parametrize('status,detail', [(400, 'Request exceeds configured context'), (400, 'invalid request')])
def test_service_rejection_keeps_history_without_budget_decay(tmp_path, status, detail):
    from qwen_agent.llm.base import ModelServiceError
    workspace = engine(tmp_path)
    call = '<tool_call>\n' + json.dumps({'name': 'shell', 'arguments': {'command': 'cat engine.json'}}) + '\n</tool_call>'
    folder = tmp_path / 'round/agent'
    config = load(ROOT / 'configs/toy.json')
    with server([call, (status, detail)]) as (url, requests):
        with pytest.raises(ModelServiceError):
            edit_candidate(url, config, workspace, folder, '实际修改引擎。')
    assert len(requests) == 2
    history = json.loads((folder / 'conversation.json').read_text())
    assert history[-1]['role'] == 'function' and 'max_context' in history[-1]['content']
    report = json.loads((folder / 'context-error.json').read_text())
    assert report['max_input_tokens'] == (config['model']['max_context'] - config['agent']['max_new_tokens']
                                          - config['agent']['context_reserve_tokens'])
    assert not (folder / 'context-restarts.json').exists()


@pytest.mark.parametrize('raw', [False, True])
def test_official_agent_folds_large_real_tool_output_at_http_boundary(tmp_path, raw):
    from vibeforge_v2.agent import Shell, assistant, exchange
    config = load(ROOT / 'configs/toy.json')
    config['model']['max_context'] = 4096
    config['agent'].update(max_new_tokens=512, context_reserve_tokens=512, use_raw_api=raw)
    arguments = {'command': "python -c \"print('START_TOOL_EVIDENCE' + ' middle output' * 6000 + 'LAST_TOOL_EVIDENCE')\""}
    call = ({'tool_calls': [{'index': 0, 'id': 'long_output_call', 'type': 'function',
              'function': {'name': 'shell', 'arguments': json.dumps(arguments)}}]} if raw else
            '<tool_call>\n' + json.dumps({'name': 'shell', 'arguments': arguments}) + '\n</tool_call>')
    folder = tmp_path / 'agent'
    with server([call, 'Actual output inspected.']) as (url, requests):
        tool = Shell(tmp_path, folder)
        bot = assistant(url, config, tool, 'Use tools to inspect evidence.')
        exchange(bot, [{'role': 'user', 'content': 'ORIGINAL_TASK_EVIDENCE: inspect the large tool response.'}], folder)
    assert len(requests) == 2
    sent = json.dumps(requests[1]['messages'])
    assert 'ORIGINAL_TASK_EVIDENCE' in sent and 'LAST_TOOL_EVIDENCE' in sent
    assert len(sent) < len(tool.calls[0]['output']) // 2
    assert ' middle output' * 6000 in (folder / 'tools.jsonl').read_text()
