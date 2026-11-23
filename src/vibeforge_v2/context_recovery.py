"""Conservative history folding; never suppress execution or edit the raw trace."""
import ast
import hashlib
import json
import re
import shlex
from pathlib import Path

MARKER = '\nHOST_OBSERVATION_JSON: '


def source_read(command):
    """Recognize reads/source inspection/version queries, not arbitrary shell."""
    try:
        words = shlex.split(command)
    except ValueError:
        return False
    if not words:
        return False
    # Output-only wrappers used by native Agent environment queries. Do not
    # accept arbitrary pipelines or file redirects. Execution is never skipped.
    if len(words) > 4 and words[-3:-1] == ['|', 'tail'] and re.fullmatch(r'-\d+', words[-1]):
        words = words[:-3]
        if words and words[-1] == '2>&1':
            words = words[:-1]
        return source_read(shlex.join(words))
    # Common Agent form: cd /known/source && sed -n ... . No other shell chain.
    if len(words) >= 4 and words[0] == 'cd' and words[2] == '&&':
        if any(c in words[1] for c in ';|&<>`$\n'):
            return False
        return source_read(shlex.join(words[3:]))
    if words[0] in ('cat', 'head', 'tail'):
        return not any(c in command for c in ';|&<>`$\n') and len(words) > 1
    if words[0] == 'sed':
        return (len(words) >= 4 and words[1] == '-n'
                and re.fullmatch(r'\d+(,\d+)?p', words[2]) is not None
                and not any(c in command for c in ';|&<>`$\n'))
    code = None
    if Path(words[0]).name in ('python', 'python3'):
        if len(words) == 3 and words[1] == '-c':
            code = words[2]
        else:
            match = re.fullmatch(r"python3?\s+<<\s*['\"]?(\w+)['\"]?\s*\n(.*?)\n\1\s*", command, re.S)
            if match:
                code = match[2]
    if code is None:
        return False
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    # Imports plus print(inspect.getsource(symbol)); no assignment or invocation of symbol.
    allowed = (ast.Module, ast.Import, ast.ImportFrom, ast.alias, ast.Expr, ast.Call,
               ast.Name, ast.Attribute, ast.Load, ast.Constant)
    if any(not isinstance(node, allowed) for node in ast.walk(tree)):
        return False
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    return bool(calls) and all(
        (isinstance(node.func, ast.Name) and node.func.id == 'print') or
        (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
         and node.func.value.id == 'inspect' and node.func.attr == 'getsource') for node in calls)


def observation(record, sequence):
    return MARKER + json.dumps({
        'sequence': sequence, 'source': record.get('candidate_check', {}).get('source_hash'),
        'output': hashlib.sha256(record['output'].encode()).hexdigest(),
        'read': source_read(record['command']), 'ok': record['exit_code'] == 0 and not record['timed_out'],
    }, sort_keys=True)


def step_key(step, observed_repeats=False):
    if len(step) != 2 or step[1].get('role') != 'tool':
        return None
    calls = step[0].get('tool_calls', [])
    # Qwen native adapter uses `id`; OpenAI-style histories use `tool_call_id`.
    result_id = step[1].get('tool_call_id') or step[1].get('id') or step[1].get('extra', {}).get('function_id')
    if len(calls) != 1 or result_id != calls[0].get('id'):
        return None
    try:
        meta = json.loads(step[1]['content'].rsplit(MARKER, 1)[1])
        fn = calls[0]['function']
        args = json.loads(fn['arguments']) if isinstance(fn['arguments'], str) else fn['arguments']
        if fn['name'] != 'shell' or not meta['ok'] or not meta['source']:
            return None
        if not observed_repeats and (not meta['read'] or not source_read(args['command'])):
            return None
        # Preserve substantive text/reasoning: don't fold steps with new explanations.
        return (args['command'], meta['source'], meta['output'],
                step[0].get('content') or '', step[0].get('reasoning_content') or '')
    except (KeyError, IndexError, ValueError, TypeError):
        return None


def fold_reads(rows, threshold=5, recover=True, observed_repeats=False):
    from .agent_context import groups
    steps, result, events = groups(rows), [], []
    i = 0
    while i < len(steps):
        key = step_key(steps[i], observed_repeats)
        end = i + 1
        if key:
            while end < len(steps) and step_key(steps[end], observed_repeats) == key:
                end += 1
        count = end - i
        if key and count >= threshold:
            # Keep the latest complete invocation and its evidence, including the real call ID.
            result.extend(steps[end - 1])
            note = (
                f'HOST HISTORY: 连续 {count} 次相同命令返回相同结果，候选源码身份未变。'
                '上方保留最后一次完整调用与结果；原始调用均已执行并记录在 tools.jsonl。'
                '这仅折叠重复历史，不代表命令没有副作用；执行次数仍是上述次数，必要时核对文件和完整日志。')
            if recover:
                note += '请依据已有证据说明当前已确认内容、仍缺的具体信息和下一步；不要虚构结论或收益。'
            result.append({'role': 'user', 'content': note})
            events.append({'command': key[0], 'source': key[1], 'count': count})
        else:
            for step in steps[i:end]:
                result.extend(step)
        i = end
    return result, events
