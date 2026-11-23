"""Per-source dependency overlays; rejection keeps the stable Python environment."""
import json
import shutil
import subprocess
from pathlib import Path

from vf_runtime.storage.atomic import digest, read_json, write_json


class ExperimentIntegrityError(RuntimeError):
    """The experiment's independent inputs changed; do not run another round."""


def package_snapshot(python):
    script = '''import importlib.metadata as m,json,hashlib
rows = []
for d in m.distributions():
    rows.append({'name': d.metadata.get('Name', ''), 'version': d.version,
                 'metadata_hash': hashlib.sha256(((d.read_text('METADATA') or '') + (d.read_text('RECORD') or '')).encode()).hexdigest()})
print(json.dumps(sorted(rows,key=lambda x:(x['name'],x['version'],x['metadata_hash']))))'''
    result = subprocess.run([str(python), '-c', script], capture_output=True, text=True, check=True, timeout=60)
    packages = json.loads(result.stdout)
    return {'packages': packages, 'hash': digest(packages),
            'scope': 'installed distribution versions and metadata/RECORD identity; not a sandbox or hash of every library file'}


class Environments:
    def __init__(self, config, run):
        self.config, self.run = config, Path(run).resolve()
        self.enabled = config['agent'].get('isolate_dependencies', not config['model'].get('fixture', False))

    def directory(self, source):
        relative = Path(source).resolve().relative_to(self.run)
        return self.run / 'environments' / relative.parent

    def python(self, source):
        return str(self.directory(source) / 'bin/python') if self.enabled else self.config['runtime_python']

    def prepare(self, source, parent=None):
        if not self.enabled:
            return self.python(source)
        target = self.directory(source)
        if target.exists():
            return self.python(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        if parent is None:
            subprocess.run([self.config['runtime_python'], '-m', 'venv', str(target)],
                           check=True, capture_output=True, text=True, timeout=120)
            # Nesting a venv does not inherit its parent's site-packages. Add the
            # selected runtime's package roots without copying the model stack.
            query = subprocess.run([self.config['runtime_python'], '-c',
                'import site,json; print(json.dumps(site.getsitepackages()))'],
                check=True, capture_output=True, text=True, timeout=30)
            roots = json.loads(query.stdout)
            sites = list((target / 'lib').glob('python*/site-packages'))
            (sites[0] / 'vibeforge-runtime.pth').write_text(''.join('import site; site.addsitedir(' + repr(str(Path(p).resolve())) + ')\n' for p in roots))
        else:
            origin = self.directory(parent)
            shutil.copytree(origin, target, symlinks=True)
            for path in (target / 'bin').iterdir():
                if path.is_symlink() or not path.is_file():
                    continue
                try:
                    text = path.read_text()
                except UnicodeError:
                    continue
                path.write_text(text.replace(str(origin), str(target)))
        write_json(target / 'created.json', {'parent': str(parent) if parent else None,
                    'runtime_python': self.config['runtime_python'], 'python': self.python(source)})
        return self.python(source)

    def seal(self, source):
        snapshot = package_snapshot(self.python(source))
        folder = self.directory(source) if self.enabled else Path(source).parent
        write_json(folder / 'packages.json', snapshot)
        return snapshot

    def verify(self, source):
        if not self.enabled:
            return
        expected = read_json(self.directory(source) / 'packages.json')
        if package_snapshot(self.python(source))['hash'] != expected['hash']:
            raise ExperimentIntegrityError(f'Sealed dependency environment changed: {source}')
