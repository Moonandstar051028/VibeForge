"""Read-only candidate feedback. Never imports candidate code or executes a GPU check."""
import ast
import hashlib
import json
from pathlib import Path

from vf_runtime.engine.builder import build
from vf_runtime.storage.atomic import atomic_write, digest, write_json
from vf_runtime.versioning.artifacts import inventory
from .experience import MICROBATCH_KEYS, config_changes, effective_config, matching_rounds
from .interface_contract import signatures, check_contract


FIXED = ('backend', 'model_path', 'device', 'max_context', 'max_new_tokens',
         'precision', 'dtype', 'quantization', 'thinking')
RUNTIME_SUFFIXES = {'.py', '.json', '.c', '.cc', '.cpp', '.cu', '.cuh', '.h', '.hpp', '.rs'}


def protected_inventory(path):
    path = Path(path)
    if path.is_file():
        return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()}
    if not path.exists():
        raise FileNotFoundError(path)
    return inventory(path)


def _bound_names(node):
    names = set()
    for item in ast.walk(node):
        if isinstance(item, ast.Name) and isinstance(item.ctx, (ast.Store, ast.Del)):
            names.add(item.id)
        elif isinstance(item, (ast.Import, ast.ImportFrom)):
            names.update((alias.asname or alias.name).split('.')[0] for alias in item.names)
        elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(item.name)
        elif isinstance(item, ast.arg):
            names.add(item.arg)
        elif isinstance(item, ast.ExceptHandler) and item.name:
            names.add(item.name)
        elif isinstance(item, (ast.Global, ast.Nonlocal)):
            names.update(item.names)
    return names


def undefined_names(source):
    """Names read inside functions that no enclosing scope, the module or builtins ever binds.

    Deliberately conservative (no flow analysis): it only reports a name that is
    bound nowhere visible, e.g. ``torch`` imported as a local of ``load()`` but
    used in ``generate()``. Such code imports fine and fails on the first request.
    """
    import builtins
    tree = ast.parse(source)
    if any(isinstance(item, ast.ImportFrom) and any(alias.name == '*' for alias in item.names)
           for item in ast.walk(tree)):
        return []
    module = set(dir(builtins)) | {'__file__', '__name__', '__doc__', '__spec__', '__builtins__'}
    for item in tree.body:
        if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            module |= _bound_names(item)
        else:
            module.add(item.name)
    for item in ast.walk(tree):  # names assigned through `global` declarations
        if isinstance(item, ast.Global):
            module.update(item.names)
    problems = []
    def visit(function, visible):
        local = visible | _bound_names(function)
        for item in ast.walk(function):
            if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load) and item.id not in local:
                problems.append((function.name, item.id, item.lineno))
    def functions(node, visible):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                if isinstance(child, ast.Lambda):
                    continue
                visit(child, visible)
                functions(child, visible | _bound_names(child))
            elif isinstance(child, ast.ClassDef):
                functions(child, visible)
            else:
                functions(child, visible)
    functions(tree, module)
    seen, unique = set(), []
    for function, name, line in problems:
        if (function, name) not in seen:
            seen.add((function, name))
            unique.append(f'{name} (in {function}, line {line})')
    return unique


def top_level_symbols(source):
    return {item.name for item in ast.parse(source).body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}


def unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


class CandidateCheck:
    def __init__(self, workspace, folder, *, parent=None, protected=(), history=None):
        self.workspace, self.folder = Path(workspace), Path(folder)
        # Host-derived ledger entries (experience.collect); used only to annotate, never to reject.
        self.history = list(history or [])
        root = Path(parent) if parent is not None else self.workspace
        self.before = inventory(root)
        self.config = json.loads((root / 'engine.json').read_text(), object_pairs_hook=unique_keys)
        try:
            self.symbols_before = top_level_symbols((root / 'engine.py').read_text())
            self.interface_before = signatures((root / 'engine.py').read_text())
        except (OSError, SyntaxError, ValueError):
            self.symbols_before = set()
            self.interface_before = {}
        try:
            self.semantic_before = self.semantic(root)
        except (ValueError, SyntaxError):
            self.semantic_before = {}
        self.sequence = 0
        self.protected = {str(Path(path).resolve()): protected_inventory(path) for path in protected}

    @staticmethod
    def semantic(root):
        result = {}
        for name in inventory(root):
            path = Path(root) / name
            if path.suffix == '.py':
                result[name] = ast.dump(ast.parse(path.read_text(), filename=str(path)), include_attributes=False)
            elif path.suffix == '.json':
                try:
                    value = json.loads(path.read_text(), object_pairs_hook=unique_keys)
                except (ValueError, UnicodeError) as error:
                    raise ValueError(f'{name}: {type(error).__name__}: {error}') from error
                # Values that the unchanged scheduler never reads are not a runtime change.
                result[name] = effective_config(value) if name == 'engine.json' else value
            elif path.suffix in RUNTIME_SUFFIXES:
                result[name] = path.read_bytes().hex()
        return result

    def inspect(self):
        errors, warnings = [], []
        protected_changes = []
        for name, before in self.protected.items():
            try:
                after = protected_inventory(name)
                protected_changes.extend(str(Path(name) if Path(name).is_file() else Path(name) / relative)
                    for relative in before.keys() | after.keys() if before.get(relative) != after.get(relative))
            except Exception as error:
                protected_changes.append(f'{name}: {type(error).__name__}: {error}')
        if protected_changes:
            errors.append('Host source changed outside candidate; stop this experiment: ' + ', '.join(protected_changes))
        try:
            current = inventory(self.workspace)
            changed = sorted(name for name in current.keys() | self.before.keys()
                             if current.get(name) != self.before.get(name))
        except Exception as error:
            current, changed = {}, []
            errors.append(f'{type(error).__name__}: {error}')
        config = None
        try:
            config = json.loads((self.workspace / 'engine.json').read_text(), object_pairs_hook=unique_keys)
            if not isinstance(config, dict):
                raise ValueError('engine.json must contain exactly one JSON object')
            for key in FIXED:
                if self.config.get(key) != config.get(key):
                    errors.append(f'engine.json: fixed model/deployment field changed: {key}')
            for key in ('microbatch_max_size', 'max_context', 'max_new_tokens'):
                if key in config and (type(config[key]) is not int or config[key] < 1):
                    errors.append(f'engine.json: {key} must be a positive integer')
            if 'microbatch_wait_ms' in config:
                value = config['microbatch_wait_ms']
                if type(value) not in (int, float) or not 0 <= value < float('inf'):
                    errors.append('engine.json: microbatch_wait_ms must be a finite non-negative number')
        except Exception as error:
            errors.append(f'engine.json: {type(error).__name__}: {error}')
        try:
            build(self.workspace)
        except Exception as error:
            detail = f'{type(error).__name__}: {error}'
            if not any(detail in item for item in errors):
                errors.append(detail)
        python_sources = {}
        for path in sorted(self.workspace.rglob('*.py')):
            relative = str(path.relative_to(self.workspace))
            if relative in changed:
                try:
                    python_sources[relative] = path.read_text()
                except (OSError, UnicodeError):
                    pass
        if 'engine.py' in python_sources and self.symbols_before:
            try:
                tree = ast.parse(python_sources['engine.py'])
                relative_imports = [node for node in ast.walk(tree)
                                    if isinstance(node, ast.ImportFrom) and node.level]
                if relative_imports:
                    # backend.load_backend loads engine.py as candidate_engine, not a package.
                    # Explicit package setup may be intentional and needs runtime validation.
                    overridden = '__package__' in _bound_names(tree)
                    for node in relative_imports:
                        message = (f'engine.py:{node.lineno}: relative import `{ast.unparse(node)}` '
                                   'requires a parent package. The host loads engine.py as the standalone '
                                   'module candidate_engine, so its default __package__ is empty. '
                                   'Catching ImportError silently skips this code; repeated model loads cannot '
                                   'fix it. Use an import of the candidate module from its own directory, or '
                                   'explicitly establish a valid package context; preserve the optimization.')
                        (warnings if overridden else errors).append(message)
                interface_errors, interface_warnings = check_contract(self.interface_before, python_sources['engine.py'])
                errors.extend(interface_errors)
                warnings.extend(interface_warnings)
                defined = set()
                for text in python_sources.values():
                    defined |= top_level_symbols(text)
                missing = sorted(self.symbols_before - defined)
                if missing:
                    errors.append('engine.py no longer defines ' + ', '.join(missing) + ' (present in the stable '
                                  'version). Compare current/engine/engine.py and restore the missing helpers locally '
                                  'while preserving your optimization; removed helpers such as tool-call parsing or sampling '
                                  'break requests that the static check cannot exercise.')
            except SyntaxError:
                pass
        import re as _re
        own = str(self.workspace.resolve())
        for relative, text in python_sources.items():
            foreign = sorted({match for match in _re.findall(r'[\w./-]*/rounds/\d{4}/engine[\w./-]*', text)
                              if not match.startswith(own)})
            if foreign:
                errors.append(f'{relative}: refers to another round\'s directory ({", ".join(foreign[:3])}). The candidate '
                              'must be self-contained: code is loaded from outside it, so your edits here may be shadowed '
                              'and the host cannot freeze what runs. Use a path relative to this file, e.g. '
                              'os.path.dirname(os.path.abspath(__file__)).')
        for relative, text in python_sources.items():
            try:
                unresolved = undefined_names(text)
            except SyntaxError:
                continue
            if unresolved:
                errors.append(f'{relative}: names used but never bound in any visible scope (NameError on first call): '
                              + ', '.join(unresolved[:8]))
        runtime_changes = [name for name in changed if Path(name).suffix in RUNTIME_SUFFIXES]
        semantic_complete = True
        try:
            semantic = self.semantic(self.workspace)
            semantic_changes = sorted(name for name in semantic.keys() | self.semantic_before.keys()
                                      if semantic.get(name) != self.semantic_before.get(name))
        except Exception as error:
            semantic_complete = False
            semantic_changes = []
            errors.append(f'Semantic inspection failed: {type(error).__name__}: {error}. '
                          'Repair or remove the malformed auxiliary file before submitting; '
                          'an unreadable file is not evidence of formatting-only changes.')
        if isinstance(config, dict) and not errors:
            literals = set()
            for path in self.workspace.rglob('*.py'):
                literals.update(node.value for node in ast.walk(ast.parse(path.read_text()))
                                if isinstance(node, ast.Constant) and isinstance(node.value, str))
            unused = sorted(key for key in config.keys() - self.config.keys() if key not in literals)
            if unused:
                warnings.append('New configuration keys not referenced as literals in candidate Python: '
                                + ', '.join(unused) + '. Check spelling and the actual reader; '
                                'dynamic or external readers cannot be verified by this static check.')
            unchanged_source = not any(Path(name).suffix in RUNTIME_SUFFIXES - {'.json'}
                                       for name in semantic_changes)
            unsupported = [key for key in unused if key in
                           {'continuous_batching', 'paged_attention', 'paged_kv_cache'} and config[key]]
            if unsupported and unchanged_source:
                errors.append('Configuration-only change enables unsupported mechanisms with no candidate reader '
                              'or implementation change: ' + ', '.join(unsupported)
                              + '. Remove the inert flags or implement and check their actual execution path. '
                              'Static microbatching is not continuous batching.')
        if isinstance(config, dict) and config.get('request_scheduler') != 'microbatch':
            source_changed = any(Path(name).suffix in RUNTIME_SUFFIXES - {'.json'} for name in semantic_changes)
            # Adding or changing a value is suspicious; removing an unused value is a harmless cleanup.
            set_inert = [key for key in MICROBATCH_KEYS
                         if key in config and config.get(key) != self.config.get(key)]
            removed_inert = [key for key in MICROBATCH_KEYS if key in self.config and key not in config]
            if set_inert:
                message = (', '.join(set_inert) + ' changed while request_scheduler is not "microbatch"; the '
                           'current scheduler never reads these keys, so the change has no runtime effect. '
                           'Either also set request_scheduler="microbatch", or revert these keys.')
                (warnings if source_changed else errors).append(message)
            if removed_inert:
                warnings.append('Removed unused keys ' + ', '.join(removed_inert) + ' (not read by the current '
                                'scheduler). This cleanup is allowed but has no runtime effect by itself.')
        if isinstance(config, dict) and self.history:
            changes = config_changes(self.config, config)
            for entry, keys in matching_rounds(self.history, changes)[-6:]:
                warnings.append(f"HISTORY: {', '.join(keys)} was already submitted in round {entry['round']} -> "
                                f"{entry['outcome']}/{entry['reason']}: {entry['detail']}")
        if semantic_complete and runtime_changes and not semantic_changes:
            warnings.append('Only formatting/comments or unused configuration keys changed; there is no effective '
                            'runtime change for the host to evaluate.')
        if not runtime_changes:
            warnings.append('No runtime source/configuration changes relative to the parent candidate.')
        self.sequence += 1
        report = {'sequence': self.sequence, 'source_hash': digest(current), 'changed_files': changed,
                  'protected_changed_files': sorted(protected_changes),
                  'runtime_changed_files': runtime_changes, 'semantic_changed_files': semantic_changes, 'static_passed': not errors,
                  'ready_for_host_evaluation': not errors and bool(semantic_changes),
                  'errors': errors, 'warnings': warnings,
                  'scope': 'actual disk JSON, Python syntax, public call binding, fixed settings and diff only; '
                           'not proof of runtime activation, numerical correctness, task quality or speed'}
        write_json(self.folder / f'check-{self.sequence:03d}.json', report)
        atomic_write(self.folder / 'latest.md', self.describe(report))
        return report

    def repeated_failure(self, python):
        """Compare actual files and dependency identity, never the model's prose."""
        from .environment import package_snapshot
        run = self.workspace.parent.parent.parent
        if self.workspace.parent.parent.name != 'rounds':
            return None
        def identity(snapshot):
            return {(row['name'].lower().replace('_', '-'), row['version'], row['metadata_hash'])
                    for row in snapshot['packages']
                    if row['name'].lower() not in {'pip', 'setuptools', 'wheel'}}
        current, environment = self.semantic(self.workspace), None
        for folder in sorted((run / 'rounds').glob('*')):
            if folder >= self.workspace.parent or not (folder / 'decision.json').exists():
                continue
            try:
                decision = json.loads((folder / 'decision.json').read_text())
                if decision.get('outcome') != 'reject' or not (folder / 'engine').exists():
                    continue
                if self.semantic(folder / 'engine') != current:
                    continue
                evidence = folder / 'environment.json'
                if not evidence.exists():
                    continue
                snapshot = json.loads(evidence.read_text()).get('candidate_after')
                if not snapshot or not snapshot.get('packages'):
                    continue
                previous_environment = identity(snapshot)
            except (OSError, ValueError, SyntaxError, KeyError, TypeError, AttributeError):
                # A rejected build or partial crash archive is diagnostic history,
                # not a valid source/dependency identity to blacklist.
                continue
            if environment is None:
                environment = identity(package_snapshot(python))
            if previous_environment == environment:
                detail = next((entry['detail'] for entry in self.history if entry['round'] == folder.name), None)
                return {'round': folder.name, 'decision': decision, 'detail': detail,
                        'feedback': str(folder / 'outcome.md')}
        return None

    @staticmethod
    def describe(report):
        lines = ['HOST CANDIDATE CHECK (read-only, independent of command exit code)',
                 'static_check: ' + ('PASS' if report['static_passed'] else 'FAIL'),
                 'actual runtime changes: ' + (', '.join(report['runtime_changed_files']) or 'NONE')]
        lines += ['ERROR: ' + error for error in report['errors']]
        lines += ['NOTE: ' + warning for warning in report['warnings']]
        lines.append('Scope: syntax/public call binding/config/diff only. Runtime correctness, quality and speed remain untested.')
        return '\n'.join(lines)
