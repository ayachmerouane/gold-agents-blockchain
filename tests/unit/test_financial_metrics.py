"""Métriques financières sur des courbes dont on connaît la réponse (§10.2)."""

import math
from datetime import date

import pandas as pd
import pytest

from gold_agents.evaluation import financial_metrics as fm
from gold_agents.simulation.costs import CostModel
from gold_agents.simulation.engine import run_backtest
from gold_agents.simulation.strategies import SmaCrossover


def test_max_drawdown() -> None:
    values = pd.Series([100, 120, 90, 130, 65, 140])
    assert fm.max_drawdown(values) == pytest.approx(-0.5)
    assert fm.max_drawdown(pd.Series([1, 2, 3])) == 0


def test_cagr_doubles_in_two_years() -> None:
    dates = pd.Series([date(2020, 1, 1), date(2022, 1, 1)])
    assert fm.cagr(pd.Series([100.0, 200.0]), dates) == pytest.approx(math.sqrt(2) - 1, rel=1e-3)


def test_sharpe_and_sortino() -> None:
    flat = pd.Series([100.0] * 10)
    assert fm.sharpe(flat) == 0 and fm.sortino(flat) == 0
    values = pd.Series([100.0, 101.0, 100.5, 101.5, 101.0])
    r = values.pct_change().dropna()
    assert fm.sharpe(values) == pytest.approx(r.mean() / r.std() * math.sqrt(252))
    # Un taux sans risque plus élevé baisse le Sharpe.
    assert fm.sharpe(values, 0.05) < fm.sharpe(values)


def test_summarize_round_trips() -> None:
    closes = [100.0] * 30 + [110.0 + i for i in range(30)] + [150.0 - 2 * i for i in range(40)]
    dates = pd.bdate_range("2024-01-01", periods=len(closes), tz="UTC")
    frame = pd.DataFrame({"date": dates, "open": closes, "close": closes})
    res = run_backtest(frame, SmaCrossover(3, 10), CostModel(0.001), 1000, date(2024, 1, 1))
    m = fm.summarize(res)
    assert m["n_trades"] == 2 and m["n_round_trips"] == 1 and m["win_rate"] == 1.0
    assert m["final_value"] == pytest.approx(1000 + fm.round_trip_pnl(res)[0])
    assert m["total_fees"] > 0 and 0 < m["exposure"] < 1
    assert m["max_drawdown"] <= 0
