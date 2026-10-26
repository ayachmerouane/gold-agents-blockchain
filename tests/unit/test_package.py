"""Tests de fumée : le paquet et ses sous-modules s'importent."""

import importlib

import pytest

import gold_agents

SUBPACKAGES = ["data", "models", "agents", "simulation", "blockchain", "evaluation"]


def test_version() -> None:
    assert gold_agents.__version__


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name: str) -> None:
    importlib.import_module(f"gold_agents.{name}")
