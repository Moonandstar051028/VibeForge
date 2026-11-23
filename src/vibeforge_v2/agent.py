"""Official Assistant with optional config helper; backend-neutral public entrypoints."""

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

from qwen_agent.agents import Assistant
from qwen_agent.llm.base import ModelServiceError
from qwen_agent.tools.base import BaseTool, ToolServiceError

from vf_runtime.observability.logs import append_record
from vf_runtime.storage.atomic import atomic_write, write_json
from vf_runtime.engine.command_sandbox import command_argv
from .checks import CandidateCheck
from .identity import current_banner


def recent_repeat(calls, window=8, threshold=5):
    """Flag a suspected local loop, never use lifetime rereads as a stopping rule.

    Source stability is only a proxy for progress: reasoning, verification and
    evidence gathering may advance without edits. The result is advisory only.
    Unknown source identity or an intervening source change resets the window.
    Keep command text exact; collapsing whitespace can change quoted shell code.
    """
    if not calls:
        return None
    current = calls[-1]
    source = current.get('candidate_check', {}).get('source_hash')
    if not source:
        return None

    def signature(row):
        return (row['command'], hashlib.sha256(row['output'].encode()).hexdigest(),
                row['exit_code'], row['timed_out'])

    key = signature(current)
    matches = []
    for index in range(len(calls) - 1, max(-1, len(calls) - window - 1), -1):
        row = calls[index]
        if row.get('candidate_check', {}).get('source_hash') != source:
            break
        if signature(row) == key:
            matches.append(index + 1)
    if len(matches) < threshold:
        return None
    return {'kind': 'suspected_no_progress_loop', 'action': 'warn_only',
            'window_calls': window, 'matching_calls': list(reversed(matches)),
            'source_hash': source,
            'scope': 'Recent identical results with unchanged candidate source; not proof of reasoning stagnation.'}


class Shell(BaseTool):
    name = 'shell'
    description = ('Execute Bash in the candidate engine. Read performance evidence, same-run history and engine/library sources; write only candidate files, '
                   'the candidate dependency environment and private scratch/cache files. '
                   'run checks or use Python. The working directory is the candidate engine. '
                   'Each call starts a fresh Bash process in that directory; cd, environment changes and '
                   'Python variables/models from previous commands do not persist. Files do persist. '
                   'Keep model loading and inspection in the same Python process. '
                   'Use foreground commands; stdout/stderr and the real exit code are returned.')
    parameters = {'type': 'object', 'properties': {
        'command': {'type': 'string', 'description': 'Bash command; scope searches to candidate or a known library path.'},
        'timeout_seconds': {'type': 'number', 'description': 'Default 60s; request up to the configured ceiling for builds or CPU checks.'}}, 'required': ['command']}

    def __init__(self, workspace, folder, python=None, feedback=None, timeout_seconds=900, output_chars=12000,
                 guard=None):
        super().__init__()
        self.workspace, self.folder = Path(workspace), Path(folder)
        self.python = python or sys.executable
        self.dependency_directory = None
        workspace_path = self.workspace.resolve()
        if workspace_path.parent.parent.name == 'rounds':
            dependency = workspace_path.parents[2] / 'environments/rounds' / workspace_path.parent.name
            if (dependency / 'bin/python').exists():
                self.python = str(dependency / 'bin/python')
                self.dependency_directory = dependency
        self.calls = []
        self.feedback = feedback
        self.timeout_seconds = timeout_seconds
        # Bound what one command adds to the conversation. The official framework
        # otherwise silently replaces older tool results with "omit" once the
        # window fills, and the Agent then re-reads the same files in a loop.
        self.output_chars = output_chars
        self.guard = guard  # SharedGuard: restores in-place edits of shared site-packages after each command

    def call(self, params, **kwargs):
        args = self._verify_json_format_args(params)
        timeout = args.get('timeout_seconds', min(60, self.timeout_seconds))
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout <= self.timeout_seconds:
            return f'Invalid timeout_seconds: must be positive and at most {self.timeout_seconds}.'
        sequence = len(self.calls) + 1
        log = self.folder / f'shell-{sequence:03d}.log'
        self.folder.mkdir(parents=True, exist_ok=True)
        environment = dict(os.environ)
        environment['PATH'] = str(Path(self.python).parent) + os.pathsep + environment.get('PATH', '')
        environment['PYTHONPATH'] = str(Path(__file__).resolve().parents[1]) + os.pathsep + environment.get('PYTHONPATH', '')
        environment['PYTHONDONTWRITEBYTECODE'] = '1'
        environment['CUDA_VISIBLE_DEVICES'] = ''
        if (Path(self.python).parent.parent / 'pyvenv.cfg').exists():
            environment['VIRTUAL_ENV'] = str(Path(self.python).parent.parent)
        environment['PIP_REQUIRE_VIRTUALENV'] = 'true'
        scratch = self.folder.resolve() / 'shell-scratch'
        temporary, cache = scratch / 'tmp', scratch / 'cache'
        for directory in (temporary, cache):
            directory.mkdir(parents=True, exist_ok=True)
        environment.update(TMPDIR=str(temporary), TMP=str(temporary), TEMP=str(temporary),
                           XDG_CACHE_HOME=str(cache), PIP_CACHE_DIR=str(cache / 'pip'),
                           TORCH_HOME=str(cache / 'torch'))
        writable = [self.workspace.resolve(), scratch]
        if self.dependency_directory is not None:
            writable.append(self.dependency_directory)
        workspace_path = self.workspace.resolve()
        if workspace_path.parent.parent.name == 'rounds':
            writable.append(workspace_path.parents[2] / 'cache/harness')
        command = command_argv(['bash', '-c', args['command']], writable)
        started = time.monotonic()
        timed_out = False
        with log.open('w') as stream:
            process = subprocess.Popen(command, cwd=self.workspace,
                                       stdout=stream, stderr=subprocess.STDOUT, env=environment, start_new_session=True)
            try:
                try:
                    code = process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    code = 124
            finally:
                # Clean only children owned by this command, including on interruption.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
        output = log.read_text(errors='replace')
        record = {'command': args['command'], 'exit_code': code, 'output': output,
                  'timed_out': timed_out, 'timeout_seconds': timeout,
                  'elapsed_seconds': time.monotonic() - started}
        record['write_directories'] = [str(p) for p in writable]
        if timed_out:
            record['output'] += f'\nHOST: command timed out after {timeout}s; process group killed.\n'
            output = record['output']
        guard_text = ''
        if self.guard is not None:
            shared = self.guard.check()
            if shared['changed'] or shared['added'] or shared['removed']:
                record['shared_dependency_guard'] = shared
                guard_text = '\n\n' + self.guard.describe(shared)
        if self.feedback is not None:
            record['candidate_check'] = self.feedback.inspect()
        self.calls.append(record)
        repeat = recent_repeat(self.calls)
        if repeat:
            record['repeat_detection'] = repeat
        append_record(self.folder / 'tools.jsonl', record)
        if record.get('candidate_check', {}).get('protected_changed_files'):
            # Official Assistant propagates ToolServiceError, preserving logs and
            # stopping evaluation of a run whose host source was altered.
            raise ToolServiceError(message='Host source changed outside candidate; see tools.jsonl and checks/latest.md')
        result = f'exit_code: {code}\n{clip_output(output, self.output_chars, log)}'
        if code != 0 and any(marker in output.lower() for marker in ('permission denied', 'read-only file system')):
            result += ('\n\nHOST PATH CONTEXT: this call started in ' + str(self.workspace.resolve())
                       + '. Writable roots: ' + ', '.join(str(path) for path in writable)
                       + '. Check the failed target against these roots. The round parent and host evidence '
                       'directories are read-only. Put WORK_LOG.md in the candidate engine directory: '
                       + str(self.workspace.resolve() / 'WORK_LOG.md')
                       + '. Each new shell call starts in the candidate engine; a previous cd does not persist.')
        if code == 1 and not output.strip() and 'grep' in args['command']:
            result += '\n(HOST: grep exit code 1 with no output means NO LINE MATCHED; it is not an error.)'
        if repeat:
            append_record(self.folder / 'repeat-events.jsonl', {'sequence': sequence, **repeat})
            result += (f'\n\nHOST NOTE: possible local loop: calls {repeat["matching_calls"]} within the latest '
                       f'{repeat["window_calls"]} calls returned identical output and exit status, with no observed '
                       'candidate source change between them. This is advisory; the round continues. '
                       'If rereading restores summarized details or verifies a specific hypothesis, continue. '
                       'Otherwise use the returned evidence or fix the reported error before retrying.')
        result += guard_text
        if self.feedback is not None:
            result += '\n\n' + self.feedback.describe(record['candidate_check'])
        if (self.feedback is not None and sequence % 8 == 0
                and not record['candidate_check'].get('runtime_changed_files')):
            result += ('\nHOST PROGRESS: 已执行 ' + str(sequence) + ' 次 Shell，但候选尚无运行时修改。'
                       '请依据现有证据选择一个最小可验证变化，先更新 WORK_LOG.md 中的假设与下一步，再实现并自检。'
                       '如果仍缺信息，明确列出阻碍这一项修改的具体问题，仅补查对应位置。'
                       '这不是调用上限，也不要求冒险修改或保证加速。')
        if len(self.calls) == 1:
            # Identity does not change during a round; repeating it on every call only consumes context.
            result += current_banner(self.workspace)
        from .context_recovery import observation
        result += observation(record, sequence)
        return result


def clip_output(output, limit, log):
    """Keep head and tail of long output; the full text stays in the per-command log file."""
    if not limit or len(output) <= limit:
        return output
    head, tail = output[:limit * 2 // 3], output[-(limit // 3):]
    omitted = len(output) - len(head) - len(tail)
    return (head + f'\n\n[HOST: {omitted} characters omitted from the middle to save context. Full output: {log}. '
            'Use grep -n, sed -n START,ENDp or python to read only the lines you need.]\n\n' + tail)


def assistant(url, config, tool, system, *, readiness=False):
    settings = config['agent']
    thinking = settings['thinking'] if not readiness else {'enabled': False}
    tokens = settings['max_new_tokens'] if not readiness else 256
    generate = {'max_tokens': tokens, 'max_input_tokens': config['model']['max_context'] - tokens
                - settings.get('context_reserve_tokens', 0),
                'request_timeout': settings['request_timeout'],
                'thought_in_content': settings.get('thought_in_content', False),
                'temperature': thinking.get('temperature', .6) if thinking['enabled'] else 0,
                'extra_body': {'chat_template_kwargs': {'enable_thinking': thinking['enabled']}}}
    if thinking['enabled']:
        generate.update(top_p=thinking.get('top_p', .95), top_k=thinking.get('top_k', 20))
        generate['extra_body']['min_p'] = thinking.get('min_p', 0)
    # This is the framework's own native-API switch, not a new tool protocol.
    generate['use_raw_api'] = settings.get('use_raw_api', False)
    if generate['use_raw_api']:
        # Native/raw mode bypasses Qwen-Agent's Nous preprocessing, which
        # normally consumes this framework-only option before the HTTP SDK.
        generate.pop('thought_in_content')
    if not readiness:
        for name, value in settings.get('sampling', {}).items():
            if name == 'min_p':
                generate['extra_body'][name] = value
            else:
                generate[name] = value
    bot = Assistant(function_list=list(tool) if isinstance(tool, (list, tuple)) else [tool], system_message=system, llm={
        'model': config['model']['name'], 'model_type': 'oai',
        'model_server': url.rstrip('/') + '/v1', 'api_key': 'EMPTY', 'generate_cfg': generate})
    if settings.get('history_reasoning', 'last') == 'last' and not settings.get('exact_context', False):
        limit_history_reasoning(bot.llm)
    if settings.get('trace_requests', False):
        from .agent_trace import attach_trace
        trace_tool = tool[0] if isinstance(tool, (list, tuple)) else tool
        attach_trace(bot.llm, trace_tool.folder)
    if settings.get('exact_context', False):
        from .agent_context import attach_context
        trace_tool = tool[0] if isinstance(tool, (list, tuple)) else tool
        attach_context(bot.llm, config, trace_tool.folder)
    return bot


def strip_old_reasoning(messages):
    """Keep reasoning only on the most recent assistant message; earlier thinking is dropped from the request.

    Qwen3's template keeps every thinking block after the last real user turn.
    An Agent session is one user turn, so dozens of thinking blocks accumulate,
    fill the window and the model starts copying its earlier thinking verbatim
    (observed: identical 4936-character thinking and the same command 4 times).
    The on-disk conversation keeps everything; only the outgoing request is trimmed.
    """
    roles = [message.get('role') if isinstance(message, dict) else message.role for message in messages]
    last = max((index for index, role in enumerate(roles) if role == 'assistant'), default=None)
    # Qwen-Agent stores one step as consecutive assistant messages (thinking, then the call): keep that whole step.
    keep = set()
    index = last
    while index is not None and index >= 0 and roles[index] == 'assistant':
        keep.add(index)
        index -= 1
    result = []
    for index, message in enumerate(messages):
        role = roles[index]
        reasoning = message.get('reasoning_content') if isinstance(message, dict) else getattr(message, 'reasoning_content', None)
        if role == 'assistant' and index not in keep and reasoning:
            if isinstance(message, dict):
                message = {**message, 'reasoning_content': ''}
            else:
                message = message.model_copy(update={'reasoning_content': ''})
        result.append(message)
    return result


def limit_history_reasoning(llm):
    original = llm.chat
    def chat(*args, **kwargs):
        if 'messages' in kwargs:
            kwargs['messages'] = strip_old_reasoning(kwargs['messages'])
        elif args:
            args = (strip_old_reasoning(args[0]),) + tuple(args[1:])
        return original(*args, **kwargs)
    llm.chat = chat


def exchange(bot, messages, folder):
    """Continue a framework run ending on a tool step; finish on normal assistant text."""
    write_json(Path(folder) / 'conversation.json', messages)
    tool_repairs = 0
    while True:
        response = []
        try:
            for response in bot.run(messages=messages):
                snapshot = [m.model_dump(exclude_none=True) if hasattr(m, 'model_dump') else m for m in response]
                write_json(Path(folder) / 'conversation.json', messages + snapshot)
        except ModelServiceError as error:
            # A backend mismatch must be diagnosed, never shrink the budget repeatedly.
            write_json(Path(folder) / 'context-error.json', {
                'error': str(error), 'max_input_tokens': bot.llm.generate_cfg.get('max_input_tokens'),
                'action': 'stopped; conversation retained; no geometric budget reduction'})
            raise
        response = [m.model_dump(exclude_none=True) if hasattr(m, 'model_dump') else m for m in response]
        messages.extend(response)
        write_json(Path(folder) / 'conversation.json', messages)
        if not response:
            raise RuntimeError('Qwen-Agent returned no response')
        if response[-1].get('role') != 'function' and not response[-1].get('function_call'):
            # Invalid native calls arrive as visible text, never executable history.
            # Repair here so startup/readiness and coding use the same protocol loop.
            content = str(response[-1].get('content', ''))
            if '<tool_call>' in content:
                tool_repairs += 1
                append_record(Path(folder) / 'tool-repairs.jsonl', {
                    'attempt': tool_repairs, 'raw_text': content,
                    'action': 'request_correction' if tool_repairs <= 3 else 'protocol_failure'})
                if tool_repairs > 3:
                    raise RuntimeError('Model repeatedly returned unexecuted tool text; see tool-repairs.jsonl')
                messages.append({'role': 'user', 'content':
                    '上一条 assistant 文本含有未执行的工具调用。文本中的命令没有执行；'
                    '请按当前工具 schema 重新调用，arguments 必须是 JSON 对象，'
                    '不能是字符串、数组或 null。保留原任务，依据实际工具结果继续。'})
                write_json(Path(folder) / 'conversation.json', messages)
                continue
            return messages


def _check_qwen_connection(url, config, workspace, folder):
    folder = Path(folder)
    challenge = 'VIBEFORGE_READY_' + uuid.uuid4().hex
    tool = Shell(workspace, folder, python=config['runtime_python'],
                 timeout_seconds=config['agent'].get('shell_timeout_seconds', 900))
    bot = assistant(url, config, tool, 'Verify the local model and shell tool connection.', readiness=True)
    exchange(bot, [{'role': 'user', 'content':
        f"Run this exact Bash command using shell:\nprintf '%s' '{challenge}'\n"
        'After seeing its output, reply ready.'}], folder)
    passed = any(row['exit_code'] == 0 and row['output'].strip() == challenge for row in tool.calls)
    report = {'passed': passed, 'challenge': challenge, 'tool_calls': len(tool.calls),
              'framework': 'official Qwen-Agent Assistant', 'model_server': url,
              'tool_parser': ('Qwen-Agent native API; server-side model-specific tool parsing'
                              if config['agent'].get('use_raw_api', False) else
                              'Qwen-Agent default; server-side tool parsing disabled')}
    write_json(folder / 'connection.json', report)
    if not passed:
        raise RuntimeError(f'Agent did not execute the real shell challenge; see {folder}')
    return report


def harness_enabled(config):
    return (config.get('agent', {}).get('harness', True) and not config.get('model', {}).get('fixture')
            and config.get('engine', {}).get('adapter') == 'transformers')


def run_harness(config, workspace, folder, source_hash):
    """CPU equivalence check of the real engine path; cached per candidate source hash."""
    from .harness import run, describe, CHECK_VERSION
    from vf_runtime.versioning.artifacts import source_hash as tree_hash
    source_hash = tree_hash(workspace)  # one identity for the agent gate and the pipeline gate
    folder = Path(folder)
    record = folder / 'harness.json'
    if record.exists():
        previous = json.loads(record.read_text())
        if (previous.get('source_hash') == source_hash and 'status' in previous
                and previous.get('check_version') == CHECK_VERSION):
            return previous
    try:
        report = run(workspace, python=config.get('runtime_python'),
                     timeout=config['agent'].get('harness_timeout_seconds', 420))
    except Exception as error:  # an unusable harness must not block evaluation; record why
        report = {'status': 'ERROR', 'passed': False, 'allow_gpu_evaluation': True, 'errors': [], 'warnings': [f'harness unavailable: {type(error).__name__}: {error}'],
                  'facts': {}, 'scope': 'not run', 'wall_seconds': 0}
    report.setdefault('status', 'PASS' if report['passed'] else 'FAIL')
    report.setdefault('allow_gpu_evaluation', report['passed'])
    report['source_hash'] = source_hash
    report['check_version'] = CHECK_VERSION
    workspace_path = Path(workspace).resolve()
    if report.get('passed') and workspace_path.parent.parent.name == 'rounds':
        from .library import known_slow
        prints = [info.get('fingerprint') for info in ((report.get('facts') or {}).get('replaced_forwards') or {}).values()]
        try:
            slow = known_slow(workspace_path.parents[2], prints)
        except Exception:
            slow = []
        if slow:
            row = slow[-1]
            report['passed'] = False
            report['status'] = 'FAIL'
            report['allow_gpu_evaluation'] = False
            report.setdefault('errors', []).append(
                f"The replacement forward is the same code as round {row['round']}'s implementation, which the host "
                f"already measured on the GPU: {row['gpu_gain']:.1%}. Resubmitting it cannot be faster. Keep it as a "
                'correctness reference and change the mechanism that makes it slow (see the bottleneck digest).')
    history = folder / 'harness-history.jsonl'
    if history.exists() and not report.get('passed'):
        lines = history.read_text().splitlines()
        previous = json.loads(lines[-1]) if lines else {}
        if previous.get('errors') == report.get('errors'):
            report.setdefault('warnings', []).append(
                'This result is IDENTICAL to the previous harness run although the source changed: your last edit did '
                'not touch the computation that is wrong. Print the .shape of every intermediate tensor in your '
                'replacement and of the same tensors in the original source on one small input, and compare them.')
    report['text'] = describe(report)
    write_json(record, report)
    append_record(history, {key: report[key] for key in ('source_hash', 'passed', 'errors', 'warnings', 'facts')})
    return report


def exchange_candidate(bot, messages, folder, checker, python=None, config=None, workspace=None):
    """Return only a changed, statically valid candidate; feed real errors back in prose."""
    returned_states = set()
    while True:
        messages = exchange(bot, messages, folder)
        report = checker.inspect()
        harness_text = ''
        if report['ready_for_host_evaluation'] and config is not None and harness_enabled(config):
            harness = run_harness(config, workspace, folder, report['source_hash'])
            harness_text = '\n\n' + harness['text']
            report['harness'] = {key: harness.get(key) for key in ('status', 'passed', 'allow_gpu_evaluation', 'errors', 'warnings', 'facts')}
            if not harness.get('allow_gpu_evaluation', harness['passed']):
                report['ready_for_host_evaluation'] = False
                report['errors'].append('CPU equivalence harness failed (see the HOST CPU EQUIVALENCE HARNESS block).')
        repeated = checker.repeated_failure(python or sys.executable) if report['ready_for_host_evaluation'] else None
        if repeated:
            report['ready_for_host_evaluation'] = False
            detail = repeated.get('detail') or repeated['decision'].get('reason')
            report['errors'].append('Same effective candidate source/configuration and dependencies already rejected in round '
                                    + repeated['round'] + ' (' + str(detail) + '); evidence: ' + repeated['feedback']
                                    + '. Resubmitting it cannot produce a different result. Change the mechanism itself, '
                                    'for example the engine.py code path that the bottleneck digest points at.')
            report['repeated_failure'] = repeated
        write_json(Path(folder) / 'completion.json', report)
        if report['ready_for_host_evaluation']:
            return messages
        if report['source_hash'] in returned_states:
            raise RuntimeError('Agent ended again with the same incomplete candidate; '
                               'no valid runtime changes. See agent/checks/latest.md')
        returned_states.add(report['source_hash'])
        unexecuted = ('\n上一条 assistant 文本含有工具标记，但没有通过框架执行的调用。'
                      '原始文本已保留在对话中；文本中的命令没有执行。请按当前工具定义重新发起合法调用，'
                      '不要把原始工具文本或解析失败当作成功操作。\n'
                      if '<tool_call>' in str(messages[-1].get('content', '')) else '')
        messages.append({'role': 'user', 'content': checker.describe(report) + harness_text + unexecuted + '\n\n'
            '若上面有 HARNESS 的 ERROR：这是宿主用缩小版真实结构模型在 CPU 上走你的 engine.py 真实路径得到的结果，'
            '报错和 traceback 是真实的。请据此修复，并可用 `python -m vibeforge_v2.harness` 复测（约 1 分钟）。\n'
            '候选尚未具备进入宿主评测的条件（宿主只评测有实际运行时效果的改动）。请按上面的 ERROR/NOTE 继续修改磁盘文件。'
            '宿主检查只做静态比较：它比较的是候选目录相对稳定版本的有效差异，不存在隐藏黑名单；'
            '删除未被读取的配置项、调整格式或恢复原样都不算改动。'
            '若所有配置项都已在经验台账中试过，请转向源码层改动（engine.py 中被瓶颈摘要指出的调用路径）。'
            '如确有无法解决的阻塞，请如实说明。'})


def _edit_qwen_candidate(url, config, workspace, folder, context):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    system = (Path(config['project']) / config['agent']['prompt']).read_text()
    atomic_write(folder / 'prompt.md', system)
    atomic_write(folder / 'context.md', context)
    protected = [Path(config['project']) / 'src']
    current = Path(workspace).resolve().parents[2] / 'current'
    if current.exists():
        protected += [current / 'manifest.json', current / 'README.md']
    history = []
    workspace_path = Path(workspace).resolve()
    if workspace_path.parent.parent.name == 'rounds':
        from .experience import collect
        try:
            history = collect(workspace_path.parents[2])
        except Exception:
            history = []
    checker = CandidateCheck(workspace, folder / 'checks', protected=protected, history=history)
    guard = config.get('_shared_guard')
    tool = Shell(workspace, folder, python=config['runtime_python'], feedback=checker,
                 timeout_seconds=config['agent'].get('shell_timeout_seconds', 900),
                 output_chars=config['agent'].get('shell_output_chars', 12000), guard=guard)
    from .config_tool import SetEngineOption
    helper = SetEngineOption(workspace, folder, feedback=checker)
    tools = [tool, helper] if config['agent'].get('config_tool', True) else [tool]
    if config['agent'].get('edit_tool', False):
        from .edit_tool import EditFile
        tools.append(EditFile(workspace, folder, feedback=checker))
    bot = assistant(url, config, tools, system)
    messages = [{'role': 'user', 'content': context}]
    completed = False
    try:
        messages = exchange_candidate(bot, messages, folder, checker, config['runtime_python'],
                                      config=config, workspace=workspace)
        completed = True
    finally:
        messages = json.loads((folder / 'conversation.json').read_text()) if (folder / 'conversation.json').exists() else messages
        texts = [str(m['content']) for m in messages
                 if m.get('role') == 'assistant' and not m.get('function_call') and m.get('content')]
        final = texts[-1] if texts else ''
        status = '' if completed else 'Agent 未完成有效候选；以下仅为最后一次模型说明，不能视为完成。\n\n'
        atomic_write(folder.parent / 'analysis.md', status + (final or 'Agent 未完成自然语言说明；参见对话与错误记录。\n'))
    return {'tool_calls': sum(len(item.calls) for item in tools), 'completion': 'natural_language', 'analysis': final}


def check_connection(url, config, workspace, folder):
    from .backends import get_backend
    return get_backend(config).check_connection(url, config, workspace, folder)


def edit_candidate(url, config, workspace, folder, context):
    from .backends import get_backend
    return get_backend(config).edit_candidate(url, config, workspace, folder, context)
