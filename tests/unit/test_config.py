"""Tests de la configuration : le YAML du dépôt est valide et les erreurs sont détectées."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError
from typer.testing import CliRunner

from gold_agents.cli import app
from gold_agents.config import Config, load_config


def test_repo_config_is_valid(config_path: Path) -> None:
    cfg = load_config(config_path)
    assert cfg.simulation.initial_capital > 0
    assert cfg.risk.allow_short is False


def _raw(config_path: Path) -> dict:
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def test_unknown_key_rejected(config_path: Path) -> None:
    raw = _raw(config_path)
    raw["risk"]["stop_los"] = 0.05
    with pytest.raises(ValidationError):
        Config.model_validate(raw)


def test_overlapping_splits_rejected(config_path: Path) -> None:
    raw = _raw(config_path)
    raw["splits"]["val_start"] = "2018-06-01"
    with pytest.raises(ValidationError):
        Config.model_validate(raw)


def test_cli_check_config(config_path: Path) -> None:
    result = CliRunner().invoke(app, ["check-config", "--config", str(config_path)])
    assert result.exit_code == 0
    assert "Configuration valide" in result.output
