"""Candidate-local exact replacements with explicit stale/ambiguous edit feedback."""
import ast
import hashlib
import json
from pathlib import Path

from qwen_agent.tools.base import BaseTool
from vf_runtime.storage.atomic import atomic_write
from vf_runtime.observability.logs import append_record


def unmatched_context(source, old):
    """Locate the first mismatch only when old_text has a unique exact starting line.

    Diagnostic only: never choose a fuzzy replacement or mutate the file.
    Excerpts are actual source text with bounded size and one-based line numbers.
    """
    offset = 0
    for line in old.splitlines(keepends=True):
        if line.strip():
            break
        offset += len(line)
    else:
        return None
    anchor = line.rstrip('\r\n')
    if len(anchor.strip()) < 8:
        return None
    starts, cursor = [], 0
    for actual in source.splitlines(keepends=True):
        if actual.rstrip('\r\n') == anchor:
            starts.append(cursor)
        cursor += len(actual)
    if len(starts) != 1:
        return None
    start, common = starts[0], 0
    for actual, supplied in zip(source[start:], old[offset:]):
        if actual != supplied:
            break
        common += 1
    position = start + common
    line_number = source.count('\n', 0, position) + 1
    supplied_position = offset + common
    lines = source.splitlines()
    first = max(0, line_number - 2)
    excerpt = '\n'.join(f'{i + 1}: {lines[i][:250]}' for i in range(first, min(len(lines), line_number + 2)))
    return {
        'source_line': line_number,
        'supplied_line': old.count('\n', 0, supplied_position) + 1,
        'actual_source': excerpt,
        'supplied_near_mismatch': old[max(0, supplied_position - 80):supplied_position + 160],
        'guidance': 'old_text does not match this disk content. Read this location and use a smaller exact unique block. No edit was applied.',
    }


class EditFile(BaseTool):
    name = 'edit_file'
    description = ('Replace one exact, unique text block in an existing candidate file. Read the relevant '
                   'source first, then supply old_text and new_text including indentation. No shell escaping '
                   'or line numbers. A stale/ambiguous match or invalid Python syntax leaves the file unchanged. '
                   'Use this for local edits; shell remains available for inspection, new modules and checks.')
    parameters = {'type': 'object', 'properties': {
        'path': {'type': 'string', 'description': 'Existing file inside the candidate directory; relative or absolute path.'},
        'old_text': {'type': 'string', 'description': 'Nonempty exact source block, occurring once.'},
        'new_text': {'type': 'string', 'description': 'Replacement block; empty string deletes the matched text.'}},
        'required': ['path', 'old_text', 'new_text']}

    def __init__(self, workspace, folder, feedback=None):
        super().__init__()
        self.workspace, self.folder = Path(workspace).resolve(), Path(folder)
        self.feedback, self.calls = feedback, []

    def call(self, params, **kwargs):
        args = self._verify_json_format_args(params)
        record = {'tool': self.name, 'arguments': args}
        try:
            if not all(isinstance(args.get(key), str) for key in ('path', 'old_text', 'new_text')):
                raise ValueError('path, old_text and new_text must be strings')
            relative = Path(args['path'])
            path = (self.workspace / relative).resolve()
            if not path.is_relative_to(self.workspace) or not path.is_file():
                raise ValueError('path must name an existing file inside the candidate directory')
            old, new = args['old_text'], args['new_text']
            if not old or old == new:
                raise ValueError('Supply nonempty old_text and a different new_text')
            before = path.read_text()
            count = before.count(old)
            if count != 1:
                if count == 0:
                    context = unmatched_context(before, old)
                    if context is not None:
                        record['match_context'] = context
                else:
                    positions, cursor = [], 0
                    while len(positions) < 3:
                        cursor = before.find(old, cursor)
                        if cursor < 0:
                            break
                        positions.append(before.count('\n', 0, cursor) + 1)
                        cursor += len(old)
                    record['matching_source_lines'] = positions
                raise ValueError(f'old_text matched {count} times; reread the relevant source and include unique context')
            after = before.replace(old, new, 1)
            if path.suffix == '.py':
                ast.parse(after, filename=str(relative))
            atomic_write(path, after)
            record.update(passed=True, changed=True)
        except (OSError, ValueError, TypeError, SyntaxError) as error:
            record.update(passed=False, changed=False, error=f'{type(error).__name__}: {error}')
        if self.feedback:
            record['candidate_check'] = self.feedback.inspect()
        self.calls.append(record)
        append_record(self.folder / 'tools.jsonl', record)
        # The full arguments remain in tools.jsonl and the assistant tool call.
        # Echoing old_text/new_text doubles large replacements in the context.
        response = {key: value for key, value in record.items() if key != 'arguments'}
        response['path'] = args.get('path')
        response['submitted_text'] = {
            key: {'characters': len(value), 'sha256': hashlib.sha256(value.encode()).hexdigest()}
            for key in ('old_text', 'new_text') if isinstance(value := args.get(key), str)}
        response['full_arguments_log'] = str(self.folder / 'tools.jsonl')
        result = json.dumps(response, ensure_ascii=False)
        if self.feedback:
            result += '\n' + self.feedback.describe(record['candidate_check'])
        return result
