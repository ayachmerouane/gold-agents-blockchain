"""Fixtures partagées par tous les tests."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config_path() -> Path:
    return ROOT / "config" / "config.yaml"
