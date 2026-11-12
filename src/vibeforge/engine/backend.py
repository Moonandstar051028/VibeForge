"""Load one candidate backend from its frozen source directory."""

import importlib.util
from copy import deepcopy
from pathlib import Path
from vibeforge.storage.atomic import read_json


def load_backend(directory):
    directory = Path(directory)
    spec = importlib.util.spec_from_file_location("candidate_engine", directory / "engine.py")
    if spec is None or spec.loader is None:
        raise ValueError("Candidate must provide engine.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("load", "generate", "chat", "chat_completion", "logits"):
        if not callable(getattr(module, name, None)):
            raise ValueError(f"Missing backend callable: {name}")
    config = read_json(directory / "engine.json")
    # The candidate may keep or mutate its input; retain an independent contract.
    module.load(deepcopy(config))
    return module, config
