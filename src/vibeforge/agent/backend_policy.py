"""Validate attention backend choices against the model and experiment history."""

import ast
from pathlib import Path

from vibeforge.storage.atomic import read_json
from vibeforge.versioning.artifacts import inventory


TRANSFORMERS_ATTENTION_BACKENDS = {
    "eager", "sdpa", "flash_attention_2", "flash_attention_3", "flash_attention_4", "flex_attention",
    "paged|eager",
}
QWEN35_ATTENTION_BACKENDS = {
    "eager", "sdpa", "flash_attention_2", "flash_attention_3", "flash_attention_4", "paged|eager",
}
OPTIONAL_BACKEND_PACKAGES = {
    "flash_attention_2": "flash-attn",
    "flash_attention_3": "flash-attn",
    "flash_attention_4": "flash-attn",
}
TRANSFORMERS_CACHE_IMPLEMENTATIONS = {
    "static", "offloaded_static", "sliding_window", "hybrid", "hybrid_chunked",
    "offloaded_hybrid", "offloaded_hybrid_chunked", "dynamic", "offloaded", "quantized", "paged",
}


def _selected_string_configuration(workspace, key):
    selected = set()
    source_path = Path(workspace) / "engine.py"
    if source_path.is_file():
        tree = ast.parse(source_path.read_text(), filename=str(source_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg == key:
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                    selected.add(node.value.value)
            elif isinstance(node, ast.Dict):
                for dict_key, value in zip(node.keys, node.values):
                    if (isinstance(dict_key, ast.Constant) and dict_key.value == key
                            and isinstance(value, ast.Constant) and isinstance(value.value, str)):
                        selected.add(value.value)
            elif isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == key for target in node.targets):
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                    selected.add(node.value.value)
            elif isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Subscript)
                    and isinstance(target.slice, ast.Constant)
                    and target.slice.value == key for target in node.targets):
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                    selected.add(node.value.value)

    config_path = Path(workspace) / "engine.json"
    if config_path.is_file():
        def collect_config(value):
            if isinstance(value, dict):
                for config_key, item in value.items():
                    if config_key == key and isinstance(item, str):
                        selected.add(item)
                    collect_config(item)
            elif isinstance(value, list):
                for item in value:
                    collect_config(item)

        collect_config(read_json(config_path))
    return selected


def selected_attention_backends(workspace):
    return _selected_string_configuration(workspace, "attn_implementation")


def cache_implementation_issues(workspace):
    issues = []
    for implementation in sorted(_selected_string_configuration(workspace, "cache_implementation")):
        if implementation not in TRANSFORMERS_CACHE_IMPLEMENTATIONS:
            issues.append((implementation,
                           "is not a valid Transformers cache_implementation; attention backends such as "
                           "sdpa belong in attn_implementation"))
    return issues


def runtime_configuration_only_change(parent, candidate):
    """Return changed attention/cache selections when no other semantics changed."""
    keys = {"attn_implementation", "cache_implementation"}
    before, after = inventory(parent), inventory(candidate)
    changed = {name for name in before.keys() | after.keys() if before.get(name) != after.get(name)}
    if not changed or not changed <= {"engine.py", "engine.json"}:
        return None

    def strip_config(value):
        if isinstance(value, dict):
            return {key: strip_config(item) for key, item in value.items() if key not in keys}
        if isinstance(value, list):
            return [strip_config(item) for item in value]
        return value

    configs = [read_json(Path(root) / "engine.json") if (Path(root) / "engine.json").is_file() else {}
               for root in (parent, candidate)]
    if strip_config(configs[0]) != strip_config(configs[1]):
        return None

    class WithoutRuntimeConfiguration(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            node.keywords = [keyword for keyword in node.keywords if keyword.arg not in keys]
            return node

        def visit_Dict(self, node):
            self.generic_visit(node)
            pairs = [(key, value) for key, value in zip(node.keys, node.values)
                     if not (isinstance(key, ast.Constant) and key.value in keys)]
            node.keys = [key for key, _ in pairs]
            node.values = [value for _, value in pairs]
            return node

        def visit_Assign(self, node):
            self.generic_visit(node)
            if node.targets and all(isinstance(target, ast.Subscript)
                    and isinstance(target.slice, ast.Constant) and target.slice.value in keys
                    for target in node.targets):
                return None
            return node

    trees = [ast.parse((Path(root) / "engine.py").read_text()) for root in (parent, candidate)]
    normalized = [ast.dump(WithoutRuntimeConfiguration().visit(tree), include_attributes=False) for tree in trees]
    if normalized[0] != normalized[1]:
        return None
    selected = {key: sorted(_selected_string_configuration(candidate, key)) for key in keys}
    return selected if any(selected.values()) else None


def backend_only_change(parent, candidate):
    """Identify a backend-only edit, ignoring its keyword/dict spelling."""
    before, after = inventory(parent), inventory(candidate)
    changed = {name for name in before.keys() | after.keys() if before.get(name) != after.get(name)}
    if not changed or not changed <= {"engine.py", "engine.json"}:
        return None
    def without_config_backend(value):
        if isinstance(value, dict):
            return {key: without_config_backend(item) for key, item in value.items() if key != "attn_implementation"}
        if isinstance(value, list):
            return [without_config_backend(item) for item in value]
        return value
    configs = [read_json(Path(root) / "engine.json") if (Path(root) / "engine.json").is_file() else {} for root in (parent, candidate)]
    if without_config_backend(configs[0]) != without_config_backend(configs[1]):
        return None

    class WithoutBackend(ast.NodeTransformer):
        @staticmethod
        def _backend_target(target):
            return (isinstance(target, ast.Subscript)
                    and isinstance(target.slice, ast.Constant)
                    and target.slice.value == "attn_implementation")

        def visit_If(self, node):
            self.generic_visit(node)
            mentions_backend = any(
                isinstance(item, ast.Constant) and item.value == "attn_implementation"
                for item in ast.walk(node.test)
            )
            statements = [*node.body, *node.orelse]
            assignments = [item for item in statements if isinstance(item, ast.Assign)]
            if (mentions_backend and assignments and len(assignments) == len(statements)
                    and all(all(self._backend_target(target) for target in item.targets)
                            for item in assignments)):
                return None
            return node

        def visit_Call(self, node):
            self.generic_visit(node)
            node.keywords = [keyword for keyword in node.keywords if keyword.arg != "attn_implementation"]
            return node

        def visit_Dict(self, node):
            self.generic_visit(node)
            pairs = [(key, value) for key, value in zip(node.keys, node.values)
                     if not (isinstance(key, ast.Constant) and key.value == "attn_implementation")]
            node.keys = [key for key, _ in pairs]
            node.values = [value for _, value in pairs]
            return node

    trees = [ast.parse((Path(root) / "engine.py").read_text()) for root in (parent, candidate)]
    normalized = [ast.dump(WithoutBackend().visit(tree), include_attributes=False) for tree in trees]
    selected = selected_attention_backends(candidate)
    if selected and normalized[0] == normalized[1]:
        return selected
    return None


def attention_backend_family(backend):
    if backend == "paged|eager":
        return "eager"
    if backend in {"flash_attention_2", "flash_attention_3", "flash_attention_4"}:
        return "flash_attention"
    return backend


def rejected_attention_backend_families(history):
    rejected = set()
    for item in history:
        if (item.get("decision") or {}).get("outcome") != "reject":
            continue
        for backend in item.get("attention_backends", []):
            rejected.add(attention_backend_family(backend))
    return rejected


def attention_backend_issues(workspace, context):
    selected = selected_attention_backends(workspace)
    model_type = context.get("model_features", {}).get("model_type", "")
    supported = QWEN35_ATTENTION_BACKENDS if model_type.startswith("qwen3_5") else TRANSFORMERS_ATTENTION_BACKENDS
    packages = context.get("environment", {}).get("inference", {}).get("packages", {})
    installed = {name.casefold().replace("_", "-") for name, version in packages.items() if version}
    issues = []
    runtime = (context.get("performance_observation") or {}).get("runtime") or context.get("effective_runtime") or {}
    for backend in sorted(selected):
        if backend not in supported:
            issues.append((backend, f"is not supported by the {model_type or 'configured'} Transformers model"))
        elif backend.startswith("paged|"):
            issues.append((backend, "requires a paged-cache allocator and verified continuous-batching execution path; the current serialized generate adapter provides neither"))
        elif backend in OPTIONAL_BACKEND_PACKAGES:
            package = OPTIONAL_BACKEND_PACKAGES[backend]
            if package not in installed:
                issues.append((backend, f"requires {package}, which is absent from the captured inference environment"))
    contract = context.get("candidate_contract") or {}
    if contract.get("parent_source") and runtime:
        change = backend_only_change(contract["parent_source"], workspace)
        effective = {value for value in runtime.get("attention_implementations", {}).values() if isinstance(value, str) and value}
        # A mixed root/text configuration is not enough to prove a no-op:
        # selecting one value could still change the other component.
        if change and len(effective) == 1 and change == effective:
            issues.append((",".join(sorted(change)), "already effective in the loaded parent; this backend-only edit has no effective configuration change"))
    return issues
