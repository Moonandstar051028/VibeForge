import json
from pathlib import Path
import pytest

from vibeforge_v2.backends import AgentBackend, CodexBackend, QwenAgentBackend, get_backend
from vibeforge_v2.config_tool import SetEngineOption


def test_backend_selection_and_abstract_contract():
    with pytest.raises(TypeError):
        AgentBackend()
    assert isinstance(get_backend({}), QwenAgentBackend)
    assert isinstance(get_backend({'agent': {'backend': 'codex'}}), CodexBackend)
    with pytest.raises(ValueError):
        get_backend({'agent': {'backend': 'typo'}})


def test_config_tool_preserves_fields_and_serializes_quotes(tmp_path):
    path = tmp_path / 'engine.json'
    path.write_text('{"device":"cpu","precision":{"quantized":false},"count":4}')
    tool = SetEngineOption(tmp_path, tmp_path / 'logs')
    value = 'quote " and slash \\ and\nnewline'
    result = json.loads(tool.call({'name': 'note', 'value': value, 'value_type': 'string'}))
    assert result['passed']
    actual = json.loads(path.read_text())
    assert actual == {'device': 'cpu', 'precision': {'quantized': False}, 'count': 4, 'note': value}
    assert json.loads(tool.call({'name': 'active', 'value': 'false', 'value_type': 'boolean'}))['passed']
    assert json.loads(path.read_text())['active'] is False


@pytest.mark.parametrize('kind,value', [('boolean', 'False'), ('number', 'NaN'), ('number', 'true'), ('number', '{}')])
def test_config_tool_refuses_invalid_types_without_writing(tmp_path, kind, value):
    path = tmp_path / 'engine.json'; path.write_text('{"keep":4}')
    before = path.read_bytes()
    result = json.loads(SetEngineOption(tmp_path, tmp_path / 'logs').call(
        {'name': 'x', 'value': value, 'value_type': kind}))
    assert not result['passed'] and path.read_bytes() == before


def test_config_tool_does_not_guess_how_to_repair_multiple_objects(tmp_path):
    path = tmp_path / 'engine.json'; path.write_text('{"keep":4}\n{"x":1}\n')
    before = path.read_bytes()
    result = json.loads(SetEngineOption(tmp_path, tmp_path / 'logs').call(
        {'name': 'x', 'value': '2', 'value_type': 'number'}))
    assert not result['passed'] and path.read_bytes() == before


def test_config_tool_file_value_copies_exact_text_and_rejects_duplicates(tmp_path):
    path=tmp_path/'engine.json'; path.write_text('{"keep":4}')
    value='quotes " slash \\ literal\\n and real\r\nnewline\r\n'
    (tmp_path/'value.txt').write_bytes(value.encode('utf-8'))
    tool=SetEngineOption(tmp_path,tmp_path/'logs')
    assert json.loads(tool.call({'name':'note','value_file':'value.txt','value_type':'string'}))['passed']
    assert json.loads(path.read_text())['note']==value
    before=path.read_bytes()
    assert not json.loads(tool.call({'name':'note','value':'x','value_file':'value.txt','value_type':'string'}))['passed']
    assert path.read_bytes()==before
    path.write_text('{"keep":4,"keep":5}')
    before=path.read_bytes()
    assert not json.loads(tool.call({'name':'note','value':'x','value_type':'string'}))['passed']
    assert path.read_bytes()==before


def test_codex_events_are_host_json_not_model_submission(tmp_path):
    import sys
    executable = tmp_path / 'fake-codex'
    executable.write_text('#!' + sys.executable + '\nimport sys,json,pathlib\n'
        'a=sys.argv;sys.stdin.read();pathlib.Path(a[a.index("--output-last-message")+1]).write_text("done")\n'
        'print(json.dumps({"type":"item.completed","item":{"type":"command_execution","exit_code":0}}))\n'
        'print(json.dumps({"type":"turn.completed"}))\n')
    executable.chmod(0o755)
    config = {'runtime_python': sys.executable,
              'agent': {'codex': {'executable': str(executable), 'inference': 'user-config'}}}
    result = CodexBackend().run(config, tmp_path, tmp_path / 'logs', 'task')
    assert result['completed'] and result['tool_calls'] == 1 and result['analysis'] == 'done'
    assert '--output-schema' not in json.loads((tmp_path/'logs/command.json').read_text())['argv']


def test_codex_local_provider_pins_engine_and_isolates_user_config(tmp_path, monkeypatch):
    import sys
    executable = tmp_path / 'fake-codex'
    executable.write_text('#!' + sys.executable + '\nimport sys,json,pathlib,os\n'
        'a=sys.argv;sys.stdin.read();p=pathlib.Path(a[a.index("--output-last-message")+1])\n'
        'p.write_text("done");(p.parent/"child.json").write_text(json.dumps({"home":os.environ["CODEX_HOME"],'
        '"api_key_present":"OPENAI_API_KEY" in os.environ,"base_present":"OPENAI_BASE_URL" in os.environ}))\n'
        'print(json.dumps({"type":"turn.completed"}))\n')
    executable.chmod(0o755)
    monkeypatch.setenv('OPENAI_API_KEY', 'fixture-not-a-real-key')
    monkeypatch.setenv('OPENAI_BASE_URL', 'https://remote.invalid/v1')
    config = {'runtime_python': sys.executable, 'model': {'name': 'Qwen3-32B', 'max_context': 40960},
              'agent': {'codex': {'executable': str(executable)}}}
    CodexBackend().run(config, tmp_path, tmp_path/'logs', 'task', url='http://127.0.0.1:1234')
    record = json.loads((tmp_path/'logs/command.json').read_text())
    assert record['model'] == 'Qwen3-32B' and record['base_url'] == 'http://127.0.0.1:1234/v1'
    assert not record['uses_existing_user_config']
    argv = record['argv']
    assert argv[argv.index('--model')+1] == 'Qwen3-32B'
    assert 'model_provider="vibeforge_local"' in argv
    assert 'model_providers.vibeforge_local.wire_api="responses"' in argv
    assert 'model_providers.vibeforge_local.requires_openai_auth=false' in argv
    catalog = json.loads((tmp_path/'logs/model-catalog.json').read_text())['models'][0]
    assert catalog['slug'] == 'Qwen3-32B' and catalog['apply_patch_tool_type'] == 'freeform'
    assert catalog['context_window'] == 40960
    assert catalog['auto_compact_token_limit'] == 40960 - 8192
    assert 'web_search="disabled"' in argv
    child = json.loads((tmp_path/'logs/child.json').read_text())
    assert Path(child['home']) == (tmp_path/'logs/codex-home').resolve()
    assert not child['api_key_present'] and not child['base_present']


def test_codex_local_never_falls_back_without_engine_url(tmp_path):
    config = {'agent': {'codex': {'executable': '/must-not-be-started'}}}
    with pytest.raises(ValueError, match='no user-provider fallback'):
        CodexBackend().run(config, tmp_path, tmp_path/'logs', 'task')


def test_codex_rejects_misspelled_inference_mode(tmp_path):
    with pytest.raises(ValueError, match='inference must be'):
        CodexBackend().run({'agent': {'codex': {'inference': 'loacl'}}}, tmp_path, tmp_path/'logs', 'task')


@pytest.mark.parametrize('change,valid', [('none', False), ('syntax', False), ('config', True)])
def test_public_codex_edit_uses_actual_candidate_gate(tmp_path, monkeypatch, change, valid):
    from vibeforge_v2.agent import edit_candidate
    from vibeforge_v2.config import ROOT
    workspace = tmp_path / 'engine'; workspace.mkdir()
    (workspace/'engine.py').write_text('def load(config):pass\ndef generate():return 1\n')
    (workspace/'engine.json').write_text('{"backend":"toy","max_context":128}')
    def run(self, config, workspace, folder, prompt, *, url=None):
        if change == 'syntax':
            (workspace/'engine.py').write_text('def broken(:\n')
        elif change == 'config':
            (workspace/'engine.json').write_text('{"backend":"toy","max_context":128,"request_scheduler":"microbatch"}')
        return {'analysis': 'model says done', 'tool_calls': 1}
    monkeypatch.setattr(CodexBackend, 'run', run)
    config = {'project': str(ROOT), 'agent': {'backend': 'codex', 'prompt': 'prompts/engineer.md'}}
    if valid:
        assert edit_candidate('', config, workspace, tmp_path/'round/agent', 'task')['tool_calls'] == 1
    else:
        with pytest.raises(RuntimeError, match='incomplete or invalid'):
            edit_candidate('', config, workspace, tmp_path/'round/agent', 'task')
        assert '未完成有效候选' in (tmp_path/'round/analysis.md').read_text()


def test_official_qwen_helper_roundtrip_preserves_quoted_value(tmp_path):
    from test_candidate_feedback import server
    from vibeforge_v2.agent import assistant, exchange
    path = tmp_path/'engine.json'; path.write_text('{"keep":4}')
    value = 'a "quote"; C:\\tmp\nnext'
    arguments = {'name': 'note', 'value': value, 'value_type': 'string'}
    call = '<tool_call>\n' + json.dumps({'name': 'set_engine_option', 'arguments': arguments}) + '\n</tool_call>'
    config = {'model': {'name': 'fixture', 'max_context': 8192},
              'agent': {'max_new_tokens': 1024, 'request_timeout': 30, 'thinking': {'enabled': False}}}
    tool = SetEngineOption(tmp_path, tmp_path/'logs')
    with server([call, 'done']) as (url, requests):
        exchange(assistant(url, config, [tool], 'Use the tool'),
                 [{'role': 'user', 'content': 'task'}], tmp_path/'conversation')
    assert json.loads(path.read_text()) == {'keep': 4, 'note': value}
    assert tool.calls[0]['arguments'] == arguments
    assert any('passed' in str(m.get('content', '')) for m in requests[1]['messages'])
