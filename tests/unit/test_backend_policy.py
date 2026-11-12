from types import SimpleNamespace

from vibeforge.agent.backend_policy import (
    attention_backend_issues,
    backend_only_change,
    cache_implementation_issues,
    runtime_configuration_only_change,
)
from vibeforge.controller.supervisor import Supervisor


BASE = 'arguments = {"dtype": "bf16"}\nmodel = Model.from_pretrained(path, **arguments)\n'
DICT_SDPA = BASE.replace('{"dtype": "bf16"}', '{"dtype": "bf16", "attn_implementation": "sdpa"}')
KEYWORD_SDPA = BASE.replace('path, **arguments', 'path, attn_implementation="sdpa", **arguments')


def source(tmp_path, name, text):
    root = tmp_path / name
    root.mkdir()
    (root / "engine.py").write_text(text)
    return root


def test_backend_only_change_recognizes_dict_and_direct_keyword(tmp_path):
    parent = source(tmp_path, "parent", BASE)
    for name, text in (("dictionary", DICT_SDPA), ("keyword", KEYWORD_SDPA)):
        candidate = source(tmp_path, name, text)
        assert backend_only_change(parent, candidate) == {"sdpa"}
    eager = source(tmp_path, "eager", KEYWORD_SDPA.replace('"sdpa"', '"eager"'))
    assert backend_only_change(parent, eager) == {"eager"}


def test_backend_only_change_preserves_other_experiment_mechanisms(tmp_path):
    parent = source(tmp_path, "parent", BASE)
    extra_code = source(tmp_path, "extra-code", KEYWORD_SDPA + "torch.set_num_threads(1)\n")
    assert backend_only_change(parent, extra_code) is None
    extra_file = source(tmp_path, "extra-file", KEYWORD_SDPA)
    (extra_file / "engine.json").write_text('{"batch_size": 2}')
    assert backend_only_change(parent, extra_file) is None
    unchanged = source(tmp_path, "unchanged", BASE)
    assert backend_only_change(parent, unchanged) is None


def test_valid_name_does_not_bypass_cache_execution_requirements(tmp_path):
    candidate = source(tmp_path, "paged", KEYWORD_SDPA.replace('"sdpa"', '"paged|eager"'))
    issues = attention_backend_issues(candidate, {"model_features": {"model_type": "qwen3_5_text"}})
    assert issues and "paged-cache" in issues[0][1]


def test_absent_package_with_null_version_is_not_installed(tmp_path):
    candidate = source(tmp_path, "flash", KEYWORD_SDPA.replace('"sdpa"', '"flash_attention_2"'))
    context = {"environment": {"inference": {"packages": {"flash-attn": None}}}}
    assert "absent" in attention_backend_issues(candidate, context)[0][1]


def test_cache_implementation_rejects_attention_backend_name_in_all_config_forms(tmp_path):
    sources = (
        'options = {"cache_implementation": "sdpa"}\n',
        'options["cache_implementation"] = "sdpa"\n',
        'generate(cache_implementation="sdpa")\n',
    )
    for index, text in enumerate(sources):
        candidate = source(tmp_path, f"python-{index}", text)
        issues = cache_implementation_issues(candidate)
        assert issues and issues[0][0] == "sdpa" and "attn_implementation" in issues[0][1]
    configured = source(tmp_path, "json", "value = 1\n")
    (configured / "engine.json").write_text('{"nested":{"cache_implementation":"sdpa"}}')
    assert cache_implementation_issues(configured)


def test_known_transformers_cache_implementation_is_allowed(tmp_path):
    candidate = source(tmp_path, "valid-cache", 'options = {"cache_implementation": "static"}\n')
    assert cache_implementation_issues(candidate) == []


def test_runtime_configuration_scope_excludes_unrelated_code_changes(tmp_path):
    parent = source(tmp_path, "parent", 'options = {"use_cache": True}\n')
    cache = source(tmp_path, "cache", 'options = {"use_cache": True, "cache_implementation": "hybrid"}\n')
    assert runtime_configuration_only_change(parent, cache) == {
        "attn_implementation": [], "cache_implementation": ["hybrid"]}
    (cache / "engine.py").write_text(
        'options = {"use_cache": True, "cache_implementation": "hybrid"}\ntorch.compile(model)\n')
    assert runtime_configuration_only_change(parent, cache) is None


def test_explicit_default_backend_is_not_an_effective_optimization(tmp_path):
    parent = source(tmp_path, "parent", BASE)
    candidate = source(tmp_path, "candidate", DICT_SDPA)
    context = {"candidate_contract": {"parent_source": str(parent)},
               "effective_runtime": {"attention_implementations": {"model": "sdpa", "text": "sdpa"}}}
    assert "already effective" in attention_backend_issues(candidate, context)[0][1]


def test_explicit_backend_replacing_config_guard_is_still_backend_only(tmp_path):
    guarded = '''arguments = {"dtype": "bf16"}
if config.get("attn_implementation"):
    arguments["attn_implementation"] = config["attn_implementation"]
model = Model.from_pretrained(path, **arguments)
'''
    explicit = '''arguments = {"dtype": "bf16", "attn_implementation": "sdpa"}
model = Model.from_pretrained(path, **arguments)
'''
    parent = source(tmp_path, "parent", guarded)
    candidate = source(tmp_path, "candidate", explicit)
    context = {"candidate_contract": {"parent_source": str(parent)},
               "effective_runtime": {"attention_implementations": {"model": "sdpa", "text": "sdpa"}}}

    assert backend_only_change(parent, candidate) == {"sdpa"}
    assert "already effective" in attention_backend_issues(candidate, context)[0][1]

    (candidate / "engine.py").write_text(explicit + "torch.set_num_threads(1)\n")
    assert backend_only_change(parent, candidate) is None


def test_explicit_default_else_branch_is_backend_only(tmp_path):
    guarded = '''arguments = {"dtype": "bf16"}
if config.get("attn_implementation"):
    arguments["attn_implementation"] = config["attn_implementation"]
model = Model.from_pretrained(path, **arguments)
'''
    explicit_default = guarded.replace(
        '    arguments["attn_implementation"] = config["attn_implementation"]\n',
        '    arguments["attn_implementation"] = config["attn_implementation"]\nelse:\n'
        '    arguments["attn_implementation"] = "sdpa"\n')
    parent = source(tmp_path, "parent", guarded)
    candidate = source(tmp_path, "candidate", explicit_default)
    context = {"candidate_contract": {"parent_source": str(parent)},
               "effective_runtime": {"attention_implementations": {"model": "sdpa", "text": "sdpa"}}}

    assert backend_only_change(parent, candidate) == {"sdpa"}
    assert "already effective" in attention_backend_issues(candidate, context)[0][1]
    context["effective_runtime"]["attention_implementations"] = {"model": "eager", "text": "sdpa"}
    assert not attention_backend_issues(candidate, context)
    context["effective_runtime"]["attention_implementations"] = {"model": "eager"}
    assert not attention_backend_issues(candidate, context)
    (candidate / "engine.py").write_text(BASE)
    (parent / "engine.json").write_text('{}')
    (candidate / "engine.json").write_text('{"attn_implementation":"eager"}')
    assert "already effective" in attention_backend_issues(candidate, context)[0][1]


def test_rejected_backend_only_edit_is_deduplicated_across_spellings(tmp_path):
    parent = source(tmp_path, "parent", BASE)
    prior = source(tmp_path, "prior", DICT_SDPA)
    current = source(tmp_path, "current", KEYWORD_SDPA)
    folder = tmp_path / "iterations" / "0001"
    folder.mkdir(parents=True)
    (folder / "change.patch").write_text("previous dictionary patch")
    row = {"iteration": 1, "parent": "parent", "candidate": "prior",
           "decision": {"outcome": "reject", "reason": "no_significant_gain"}}
    controller = object.__new__(Supervisor)
    controller.campaign_id = "test"
    controller.directory = tmp_path
    controller.database = SimpleNamespace(iterations=lambda _: [row])
    releases = {"parent": parent, "prior": prior}
    controller.paths = SimpleNamespace(release=lambda ref: releases[ref])

    assert controller.previously_rejected_patch("new keyword patch", 2, "parent", current) == row
    (current / "engine.py").write_text(KEYWORD_SDPA.replace('"sdpa"', '"eager"'))
    assert controller.previously_rejected_patch("different backend patch", 2, "parent", current) is None
    (current / "engine.py").write_text(KEYWORD_SDPA + "torch.set_num_threads(1)\n")
    assert controller.previously_rejected_patch("new mechanism patch", 2, "parent", current) is None
