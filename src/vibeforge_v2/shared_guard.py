"""Real-time guard for the shared dependency environment (site-packages used by stable and candidates).

Candidate environments add the shared runtime's site-packages through a .pth
file, so the stable engine, the reference baseline, the quality evaluator and
every candidate import the very same transformers/torch files. In round 7 of
the revised run the Agent edited such a file in place with ``sed -i``. The
end-of-round dependency snapshot compares only distribution metadata and RECORD
contents, which in-place edits do not change; only an external watcher caught it.

This guard stats every file under the shared roots after each Agent command
(~0.25 s for ~32k files), restores modified/removed Python files from a
per-run backup, deletes added files, and reports the paths so the Agent gets a
precise error instead of the experiment being killed. Files without a backup
(binary libraries) are reported as unrecoverable so the host can stop safely.
"""
import json
import os
import shutil
import subprocess
from pathlib import Path

from vf_runtime.storage.atomic import digest


def shared_roots(python):
    """Resolved site-packages roots of the shared runtime Python (deduplicated, existing only)."""
    script = 'import site, json; print(json.dumps(site.getsitepackages()))'
    result = subprocess.run([str(python), '-c', script], capture_output=True, text=True, check=True, timeout=60)
    roots = []
    for path in json.loads(result.stdout):
        resolved = Path(path).resolve()
        if resolved.is_dir() and resolved not in roots:
            roots.append(resolved)
    return roots


def _snapshot(roots):
    state = {}
    for root in roots:
        for directory, folders, files in os.walk(root):
            folders[:] = [name for name in folders if name != '__pycache__']
            for name in files:
                if name.endswith(('.pyc', '.pyo')):
                    continue
                path = os.path.join(directory, name)
                try:
                    info = os.stat(path)
                except OSError:
                    continue
                state[path] = (info.st_size, info.st_mtime_ns)
    return state


class SharedGuard:
    def __init__(self, roots, backup_root):
        self.roots = [Path(root) for root in roots]
        self.state = _snapshot(self.roots)
        key = digest(sorted((path, size, mtime) for path, (size, mtime) in self.state.items()
                            if path.endswith('.py')))[:16]
        self.backup = Path(backup_root) / f'shared-py-{key}'
        if not (self.backup / 'ready').exists():
            staging = Path(str(self.backup) + f'.tmp-{os.getpid()}')
            shutil.rmtree(staging, ignore_errors=True)
            for path in self.state:
                if path.endswith('.py'):
                    target = staging / path.lstrip('/')
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, target)
            (staging / 'ready').write_text(json.dumps({'roots': [str(root) for root in self.roots]}))
            shutil.rmtree(self.backup, ignore_errors=True)
            staging.rename(self.backup)

    @classmethod
    def for_python(cls, python, backup_root):
        return cls(shared_roots(python), backup_root)

    def check(self, restore=True):
        """Return {'changed', 'added', 'removed', 'restored', 'unrecoverable'}; restore when possible."""
        current = _snapshot(self.roots)
        changed = sorted(path for path in current.keys() & self.state.keys() if current[path] != self.state[path])
        added = sorted(current.keys() - self.state.keys())
        removed = sorted(self.state.keys() - current.keys())
        report = {'changed': changed, 'added': added, 'removed': removed, 'restored': [], 'unrecoverable': []}
        if not restore:
            return report
        for path in changed + removed:
            backup = self.backup / path.lstrip('/')
            if backup.exists():
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(backup, path)  # copy2 also restores the original mtime
                os.utime(path, ns=(self.state[path][1], self.state[path][1]))
                report['restored'].append(path)
            else:
                report['unrecoverable'].append(path)
        for path in added:
            try:
                os.remove(path)
                report['restored'].append(path)
            except OSError:
                report['unrecoverable'].append(path)
        if report['restored']:
            refreshed = _snapshot(self.roots)
            for path in report['restored']:
                if path in self.state and refreshed.get(path) != self.state[path]:
                    report['unrecoverable'].append(path)
        return report

    @staticmethod
    def describe(report):
        touched = report['changed'] + report['added'] + report['removed']
        if not touched:
            return ''
        lines = ['HOST SHARED DEPENDENCY GUARD: ERROR - this command modified the shared environment that the '
                 'stable engine, baseline and evaluator also import:']
        lines += ['  ' + path for path in touched[:8]]
        if report['restored']:
            lines.append(f"The host restored {len(report['restored'])} file(s) to their original content. "
                         'Your change there has NO effect on the candidate.')
        if report['unrecoverable']:
            lines.append('Could not restore: ' + ', '.join(report['unrecoverable'][:5]))
        lines.append('To change library behaviour, put the code in the candidate directory and apply it from '
                     'engine.py, e.g. `from transformers.models.qwen3_moe import modeling_qwen3_moe as m; '
                     'm.Qwen3MoeSparseMoeBlock.forward = my_forward` (or a subclass). Install new packages only with '
                     'the candidate Python (`python -m pip install ...`), which writes to the candidate environment.')
        return '\n'.join(lines)
