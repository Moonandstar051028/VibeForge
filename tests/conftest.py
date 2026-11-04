import sys
from pathlib import Path
import pytest
from vibeforge.contracts.campaign import load_config
from vibeforge.storage.atomic import digest
from vibeforge.storage.paths import Paths


@pytest.fixture
def project():
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def paths(project, tmp_path):
    return Paths.create(project, tmp_path / "var")


@pytest.fixture
def toy_config(project):
    config = load_config(project / "configs/campaigns/toy.json", project)
    config["runtime_python"] = sys.executable
    config["evaluation"].update(repeats=2, warmup=0)
    config["config_hash"] = digest({k: v for k, v in config.items() if k != "config_hash"})
    return config
