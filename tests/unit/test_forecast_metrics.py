"""Tests des métriques de prévision (§5.4), sur des valeurs calculées à la main."""

import math

import numpy as np
import pytest

from gold_agents.evaluation import forecast_metrics as fm


def test_mae_rmse_by_hand() -> None:
    y, p = np.array([0.01, -0.02, 0.03]), np.array([0.0, 0.0, 0.01])
    assert fm.mae(y, p) == pytest.approx((0.01 + 0.02 + 0.02) / 3)
    assert fm.rmse(y, p) == pytest.approx(math.sqrt((1e-4 + 4e-4 + 4e-4) / 3))


def test_directional_accuracy_ignores_zero_forecasts() -> None:
    y = np.array([0.01, -0.01, 0.02, -0.03])
    assert fm.directional_accuracy(y, np.array([1.0, 1.0, 0.0, -1.0])) == pytest.approx(2 / 3)
    assert math.isnan(fm.directional_accuracy(y, np.zeros(4)))


def test_r2_vs_zero() -> None:
    y = np.array([0.01, -0.02, 0.03])
    assert fm.r2_vs_zero(y, np.zeros(3)) == 0.0
    assert fm.r2_vs_zero(y, y) == 1.0
    assert fm.r2_vs_zero(y, -y) < 0


def test_diebold_mariano() -> None:
    rng = np.random.default_rng(0)
    y = rng.normal(0, 0.01, 500)
    good = y + rng.normal(0, 0.001, 500)
    test = fm.diebold_mariano(y, good, np.zeros(500))
    assert test.statistic < 0 and test.p_value < 0.01  # bien meilleur que B0
    same = fm.diebold_mariano(y, np.zeros(500), np.zeros(500))
    assert same.p_value == 1.0


def test_forecast_metrics_keys() -> None:
    y = np.array([0.01, -0.02, 0.03])
    out = fm.forecast_metrics(y, np.array([0.001, -0.001, 0.002]))
    assert set(out) == {"mae", "rmse", "direction", "r2_vs_zero", "dm_vs_zero", "dm_p_value"}
    assert out["direction"] == 1.0
