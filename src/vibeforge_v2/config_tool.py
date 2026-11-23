"""Typed single-option edits; host code owns JSON quoting and serialization."""
import json
import math
from pathlib import Path

from qwen_agent.tools.base import BaseTool
from vf_runtime.storage.atomic import atomic_write
from vf_runtime.observability.logs import append_record
from .checks import unique_keys
from .identity import current_banner


class SetEngineOption(BaseTool):
    name = 'set_engine_option'
    description = ('Set one top-level option in the actual engine.json. Supply a name, a plain text value '
                   'and value_type (string, number, boolean, null). Supply either plain value or value_file '
                   '(UTF-8 file relative to candidate, or absolute path) to copy exact bytes without escaping. '
                   'Host reads and serializes the complete '
                   'JSON object and preserves other fields. No Bash quoting or complete JSON document needed. '
                   'Malformed existing files return an error; use shell to inspect/repair them.')
    parameters = {'type': 'object', 'properties': {
        'name': {'type': 'string', 'description': 'Top-level configuration key.'},
        'value': {'type': 'string', 'description': 'Plain text value, without JSON quotes; omit when using value_file.'},
        'value_file': {'type': 'string', 'description': 'Read complete UTF-8 text from this file; omit value. No trimming or unescaping.'},
        'value_type': {'type': 'string', 'description': 'string, number, boolean or null.'}},
        'required': ['name', 'value_type']}

    def __init__(self, workspace, folder, feedback=None):
        super().__init__()
        self.path, self.folder = Path(workspace) / 'engine.json', Path(folder)
        self.feedback, self.calls = feedback, []

    def call(self, params, **kwargs):
        args = self._verify_json_format_args(params)
        record = {'tool': self.name, 'arguments': args}
        try:
            if ('value' in args) == ('value_file' in args):
                raise ValueError('Supply exactly one of value and value_file')
            if 'value_file' in args:
                source = Path(args['value_file'])
                value = (source if source.is_absolute() else self.path.parent / source).read_bytes().decode('utf-8')
            else:
                value = args['value']
            kind = args['value_type']
            if not isinstance(value, str) or not isinstance(args['name'], str):
                raise ValueError('name and input value must be strings')
            if kind == 'string':
                pass
            elif kind == 'boolean' and value in ('true', 'false'):
                value = value == 'true'
            elif kind == 'null' and value in ('', 'null'):
                value = None
            elif kind == 'number':
                value = json.loads(value)
                if type(value) not in (int, float) or (type(value) is float and not math.isfinite(value)):
                    raise ValueError('Expected a finite JSON number')
            else:
                raise ValueError('Invalid value_type/value; boolean must be true or false')
            current = json.loads(self.path.read_text(), object_pairs_hook=unique_keys)
            if not isinstance(current, dict):
                raise ValueError('engine.json must contain one object')
            before = current.copy()
            current[args['name']] = value
            serialized = json.dumps(current, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
            atomic_write(self.path, serialized)
            actual = json.loads(self.path.read_text())
            record.update(passed=True, changed=json.dumps(current, sort_keys=True) != json.dumps(before, sort_keys=True), actual=actual)
        except (OSError, ValueError, TypeError, KeyError) as error:
            record.update(passed=False, error=f'{type(error).__name__}: {error}')
        if self.feedback:
            record['candidate_check'] = self.feedback.inspect()
        self.calls.append(record)
        append_record(self.folder / 'tools.jsonl', record)
        result = json.dumps(record, ensure_ascii=False)
        if self.feedback:
            result += '\n' + self.feedback.describe(record['candidate_check'])
        result += current_banner(self.path.parent)
        return result
