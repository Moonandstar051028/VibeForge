"""Small agent boundary; candidate gates remain owned by the pipeline."""
from abc import ABC, abstractmethod
import json
import os
from pathlib import Path
import signal
import subprocess

from vf_runtime.storage.atomic import atomic_write, write_json
from .checks import CandidateCheck


class AgentBackend(ABC):
    @abstractmethod
    def check_connection(self, url, config, workspace, folder):
        """Prove a real tool action, not just an HTTP response."""

    @abstractmethod
    def edit_candidate(self, url, config, workspace, folder, context):
        """Edit files and explain; never submit a result JSON."""


class QwenAgentBackend(AgentBackend):
    def check_connection(self, *args):
        from .agent import _check_qwen_connection
        return _check_qwen_connection(*args)

    def edit_candidate(self, *args):
        from .agent import _edit_qwen_candidate
        return _edit_qwen_candidate(*args)


def local_codex_catalog(config):
    """Native Codex metadata: declare real transport/tool capabilities, not weights."""
    model = config['model']
    # Reserve the engine's actual maximum output inside its finite context window.
    available_input = model['max_context'] - config['agent'].get('max_new_tokens', 8192)
    if available_input <= 0:
        raise ValueError('Codex output must fit the local model context window')
    return {'models': [{
        'slug': model['name'], 'display_name': model['name'],
        'description': 'Local model served by the running VibeForge engine',
        'supported_reasoning_levels': [], 'shell_type': 'unified_exec',
        'visibility': 'list', 'supported_in_api': True, 'priority': 1,
        'support_verbosity': False, 'supports_reasoning_summary_parameter': False,
        'apply_patch_tool_type': 'freeform', 'input_modalities': ['text'],
        'truncation_policy': {'mode': 'bytes', 'limit': 10000},
        'experimental_supported_tools': [], 'context_window': model['max_context'],
        'auto_compact_token_limit': available_input,
        'base_instructions': (
            'You are a coding agent running in the official Codex CLI. Use the tools declared '
            'in this request to inspect and modify workspace files and verify results. Tool '
            'schemas and descriptions are authoritative. Implement the requested changes, then '
            'explain actual changes, checks and remaining limitations.'),
    }]}


class CodexBackend(AgentBackend):
    def run(self, config, workspace, folder, prompt, *, url=None):
        folder = Path(folder).resolve()
        folder.mkdir(parents=True, exist_ok=True)
        settings = config['agent'].get('codex', {})
        inference = settings.get('inference', 'local')
        overrides = []
        if inference == 'local':
            if not url or not url.startswith(('http://', 'https://')):
                raise ValueError('Local Codex requires the running engine URL; no user-provider fallback')
            model = config['model']['name']
            base_url = url.rstrip('/') + ('' if url.rstrip('/').endswith('/v1') else '/v1')
            catalog = folder / 'model-catalog.json'
            write_json(catalog, local_codex_catalog(config))
            values = {
                'model_provider': 'vibeforge_local',
                'model_catalog_json': str(catalog),
                'model_context_window': config['model']['max_context'],
                # This local inference service has no provider-hosted search endpoint.
                # Native shell/network access and file tools retain their permissions.
                'web_search': 'disabled',
                'model_providers.vibeforge_local.name': 'VibeForge local engine',
                'model_providers.vibeforge_local.base_url': base_url,
                'model_providers.vibeforge_local.wire_api': 'responses',
                'model_providers.vibeforge_local.requires_openai_auth': False,
                'model_providers.vibeforge_local.stream_idle_timeout_ms':
                    int(config['agent'].get('request_timeout', 1800) * 1000),
            }
            for name, value in values.items():
                overrides.extend(['-c', name + '=' + json.dumps(value)])
        elif inference == 'user-config':
            model, base_url = settings.get('model', 'gpt-6.1-sol'), None
        else:
            raise ValueError('agent.codex.inference must be local or user-config')
        command = [settings.get('executable', 'codex'), 'exec', *overrides, '--model', model,
                   '--dangerously-bypass-approvals-and-sandbox', '--skip-git-repo-check',
                   '--cd', str(Path(workspace).resolve()), '--json', '--color', 'never',
                   '--output-last-message', str(folder / 'final.txt'), '-']
        atomic_write(folder / 'task.md', prompt)
        # Record arguments only; never dump the user's config, environment or credentials.
        write_json(folder / 'command.json', {'argv': command, 'inference': inference,
                    'model': model, 'base_url': base_url,
                    'uses_existing_user_config': inference == 'user-config',
                    'target_engine_url_not_used_for_codex_inference': inference != 'local'})
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
        if inference == 'local':
            # Isolate inference routing; native Codex tools retain full permissions.
            home = folder / 'codex-home'
            home.mkdir(exist_ok=True)
            environment['CODEX_HOME'] = str(home)
            for key in ('OPENAI_API_KEY', 'OPENAI_BASE_URL', 'OPENAI_ORG_ID', 'OPENAI_PROJECT_ID'):
                environment.pop(key, None)
        environment['PATH'] = str(Path(config['runtime_python']).parent) + os.pathsep + environment.get('PATH', '')
        environment['PYTHONPATH'] = str(Path(__file__).resolve().parents[1]) + os.pathsep + environment.get('PYTHONPATH', '')
        with (folder / 'events.jsonl').open('w') as events, (folder / 'stderr.log').open('w') as errors:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=events, stderr=errors,
                                       text=True, env=environment, start_new_session=True)
            from vf_runtime.engine.process import process_identity
            write_json(folder / 'process.json', process_identity(process.pid))
            try:
                process.communicate(prompt)
            finally:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
        rows = []
        for line in (folder / 'events.jsonl').read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
        commands = [r['item'] for r in rows if r.get('type') == 'item.completed'
                    and r.get('item', {}).get('type') == 'command_execution']
        completed = process.returncode == 0 and any(r.get('type') == 'turn.completed' for r in rows)
        final = (folder / 'final.txt').read_text() if (folder / 'final.txt').exists() else ''
        result = {'completed': completed, 'exit_code': process.returncode, 'tool_calls': len(commands),
                  'completion': 'natural_language', 'analysis': final, 'backend': 'codex'}
        write_json(folder / 'execution.json', result)
        if not completed:
            raise RuntimeError(f'Codex did not complete a turn; see {folder}')
        return result

    def check_connection(self, url, config, workspace, folder):
        import uuid
        challenge = uuid.uuid4().hex
        proof = Path(folder).resolve() / 'tool-proof.txt'
        result = self.run(config, workspace, folder,
                 f"Use a shell command to write exactly {challenge} to {proof}. Read it back, then reply ready.", url=url)
        local = config['agent'].get('codex', {}).get('inference', 'local') == 'local'
        report = {'passed': proof.exists() and proof.read_text().strip() == challenge,
                  'framework': 'Codex CLI', 'inference': 'local engine' if local else 'existing user Codex provider',
                  'model': config['model']['name'] if local else config['agent']['codex'].get('model'),
                  'local_target_model_connection_tested': local, 'tool_calls': result.get('tool_calls')}
        write_json(Path(folder) / 'connection.json', report)
        if not report['passed']:
            raise RuntimeError('Codex did not execute the real file challenge')
        return report

    def edit_candidate(self, url, config, workspace, folder, context):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        system = (Path(config['project']) / config['agent']['prompt']).read_text()
        atomic_write(folder / 'prompt.md', system)
        atomic_write(folder / 'context.md', context)
        checker = CandidateCheck(workspace, folder / 'checks', protected=[Path(config['project']) / 'src'])
        result, report = None, None
        try:
            result = self.run(config, workspace, folder, system + '\n\n' + context, url=url)
            report = checker.inspect()
            write_json(folder / 'completion.json', report)
            if not report['ready_for_host_evaluation']:
                raise RuntimeError('Codex candidate is incomplete or invalid; see checks/latest.md')
            return result
        finally:
            final = result['analysis'] if result else ((folder/'final.txt').read_text() if (folder/'final.txt').exists() else '')
            ready = result is not None and report is not None and report['ready_for_host_evaluation']
            atomic_write(folder.parent / 'analysis.md', ('' if ready else 'Agent未完成有效候选；以下仅为模型说明。\n\n') + final)


def get_backend(config):
    name = config.get('agent', {}).get('backend', 'qwen')
    implementations = {'qwen': QwenAgentBackend, 'codex': CodexBackend}
    if name not in implementations:
        raise ValueError(f'Unknown agent backend: {name}')
    return implementations[name]()
