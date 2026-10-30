"""Phase 6 : simulateur rapide, reprise après coupe-circuit, grille et walk-forward."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from gold_agents.agents.execution import ExecutionAgent
from gold_agents.agents.forecast import ForecastAgent, ForecastTable
from gold_agents.agents.messages import MarketState, Signal
from gold_agents.agents.orchestrator import AgentSystem, run_agents
from gold_agents.agents.risk import RiskAgent
from gold_agents.agents.strategy import StrategyAgent
from gold_agents.config import Config, TuningGrid
from gold_agents.evaluation import tuning as tu
from gold_agents.models.walk_forward import before_test, fold_rows
from gold_agents.simulation.costs import CostModel
from gold_agents.simulation.engine import run_backtest
from gold_agents.simulation.fast import (
    AgentParams,
    oracle_positions,
    random_positions,
    simulate_agents,
    simulate_positions,
)
from gold_agents.simulation.strategies import Action, BuyAndHold

COSTS = CostModel(0.001, 0.0002)


def _market(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.015, n)))
    opens = close * np.exp(rng.normal(0, 0.004, n))
    dates = pd.bdate_range("2015-01-01", periods=n, tz="UTC")
    return pd.DataFrame({"date": dates, "open": opens, "close": close})


PARAMS = [
    AgentParams(0.0, 0.0, None, 0.05, 0.20),
    AgentParams(0.0005, 0.001, 0.5, 0.03, 0.10, 1.0, 10),
    AgentParams(0.0002, 0.0, 0.25, 0.08, 0.05, 0.5, 5),
    AgentParams(0.0, 0.0005, 1.0, 0.02, 0.08, 1.0, None),
]


@pytest.mark.parametrize("params", PARAMS)
@pytest.mark.parametrize("with_sigma", [True, False])
def test_fast_simulation_matches_the_agents(params: AgentParams, with_sigma: bool) -> None:
    frame = _market()
    rng = np.random.default_rng(1)
    r = rng.normal(0.0003, 0.002, len(frame))
    r[5:15] = np.nan  # jours sans prévision
    sigma = np.abs(rng.normal(0.001, 0.0005, len(frame))) if with_sigma else None
    ok = ~np.isnan(r)
    days = [d.date() for d in frame["date"]]
    table = ForecastTable(
        "m",
        [d for d, keep in zip(days, ok, strict=True) if keep],
        r[ok],
        None if sigma is None else sigma[ok],
    )
    system = AgentSystem(
        "agents",
        ForecastAgent(table),
        StrategyAgent(params.theta_buy, params.theta_sell, params.min_confidence),
        RiskAgent(
            params.max_position, params.stop_loss, params.max_drawdown, False, params.cooldown_days
        ),
        ExecutionAgent(COSTS),
    )
    full = run_agents(frame, system, 10_000.0, days[0], days[-1]).result
    fast = simulate_agents(
        frame["open"].to_numpy(), frame["close"].to_numpy(), r, sigma, params, COSTS, 10_000.0
    )
    np.testing.assert_allclose(fast.values, full.equity["value"].to_numpy(), rtol=1e-12)
    assert fast.n_trades == len(full.trades)
    # Sans σ̂, le filtre de confiance bloque tous les achats.
    assert (fast.n_trades > 0) == (with_sigma or params.min_confidence is None)
    exec_days = [days.index(t.exec_date) for t in full.trades]
    assert list(fast.trade_days) == exec_days


def test_fast_buy_and_hold_matches_the_engine() -> None:
    frame = _market()
    days = [d.date() for d in frame["date"]]
    full = run_backtest(frame, BuyAndHold(), COSTS, 10_000.0, days[0], days[-1])
    fast = simulate_positions(
        frame["open"].to_numpy(),
        frame["close"].to_numpy(),
        np.ones(len(frame), dtype=bool),
        COSTS,
        10_000.0,
    )
    np.testing.assert_allclose(fast.values, full.equity["value"].to_numpy(), rtol=1e-12)


def test_random_positions_make_exactly_the_same_number_of_trades() -> None:
    frame = _market(200)
    rng = np.random.default_rng(0)
    for n in (0, 1, 7, 30):
        want = random_positions(len(frame), n, rng)
        res = simulate_positions(
            frame["open"].to_numpy(), frame["close"].to_numpy(), want, COSTS, 1000.0
        )
        assert res.n_trades == n


def test_oracle_beats_buy_and_hold() -> None:
    frame = _market()
    o, c = frame["open"].to_numpy(), frame["close"].to_numpy()
    oracle = simulate_positions(o, c, oracle_positions(o), CostModel(), 1000.0)
    bh = simulate_positions(o, c, np.ones(len(o), dtype=bool), CostModel(), 1000.0)
    assert oracle.values[-1] > bh.values[-1]


# --- Reprise après le coupe-circuit ----------------------------------------------------


def _state(value: float, qty: float = 0.0) -> MarketState:
    return MarketState(date(2024, 1, 2), 100.0, value - qty * 100.0, qty, value, None)


def test_circuit_breaker_resumes_after_cooldown() -> None:
    risk = RiskAgent(max_drawdown=0.2, cooldown_days=3)
    buy = Signal(date(2024, 1, 2), Action.BUY, 1.0, "test")
    risk.act(buy, _state(1000.0, qty=10.0))
    assert risk.act(buy, _state(790.0, qty=7.9)).rule == "max_drawdown" and risk.halted
    for _ in range(3):  # trois jours en cash : toujours bloqué
        assert not risk.act(buy, _state(790.0)).executable
    order = risk.act(buy, _state(790.0))  # quatrième jour : reprise
    assert order.executable and order.action is Action.BUY and not risk.halted
    assert risk.peak_value == 790.0  # le plus haut repart du jour de la reprise


def test_circuit_breaker_without_cooldown_stays_in_cash() -> None:
    risk = RiskAgent(max_drawdown=0.2)
    buy = Signal(date(2024, 1, 2), Action.BUY, 1.0, "test")
    risk.act(buy, _state(1000.0, qty=10.0))
    risk.act(buy, _state(790.0, qty=7.9))
    assert all(not risk.act(buy, _state(790.0)).executable for _ in range(100))


# --- Grille et sélection ---------------------------------------------------------------

GRID = TuningGrid(
    theta_buy=[0.0, 0.0005],
    theta_sell=[0.0, 0.0005],
    min_confidence=[None, 0.5],
    stop_loss=[0.03, 0.05],
    max_drawdown=[0.15, 0.2],
    models=["ensemble", "lightgbm"],
)


def test_grid_has_no_confidence_filter_without_sigma() -> None:
    grid = tu.combos(GRID)
    assert len(grid) == 2 * 2 * 2 * 2 * 2 + 2 * 2 * 2 * 2
    assert {c.min_confidence for c in grid if c.model == "lightgbm"} == {None}


def _markets(n_folds: int = 3) -> list[tu.FoldMarket]:
    out = []
    for f in range(1, n_folds + 1):
        frame = _market(260, seed=f)
        rng = np.random.default_rng(10 + f)
        r = {m: rng.normal(0.0003, 0.002, len(frame)) for m in ("ensemble", "lightgbm")}
        sigma = {"ensemble": np.full(len(frame), 0.001), "lightgbm": None}
        out.append(
            tu.FoldMarket(
                f,
                [d.date() for d in frame["date"]],
                frame["open"].to_numpy(),
                frame["close"].to_numpy(),
                r,
                sigma,
            )
        )
    return out


def test_grid_selection_and_walk_forward_use_only_the_past() -> None:
    markets = _markets()
    sim = tu.Simulator(COSTS, 10_000.0, cooldown_days=20)
    table, bh = tu.run_grid(markets, tu.combos(GRID), sim)
    assert len(table) == len(tu.combos(GRID)) * 3 and len(bh) == 3
    summary = tu.summarize_grid(table, GRID, [1, 2, 3])
    # Tri : contrainte de perte max d'abord, puis score (le moins bon de soi et des voisins).
    assert summary["dd_ok"].is_monotonic_decreasing
    top = summary[summary["dd_ok"] == summary["dd_ok"].iloc[0]]
    assert top["score"].is_monotonic_decreasing
    assert (summary["score"] <= summary["sharpe"] + 1e-12).all()

    wf = tu.walk_forward(table, bh, GRID, markets, sim)
    assert list(wf["fold"]) == [2, 3]
    # La période 2 est jugée avec le réglage choisi sur la seule période 1.
    first = tu.best(tu.summarize_grid(table, GRID, [1]))
    assert wf.iloc[0]["model"] == first.model and wf.iloc[0]["theta_buy"] == first.theta_buy
    expected = sim.agents(markets[1], first).metrics()["sharpe"]
    assert wf.iloc[0]["sharpe"] == pytest.approx(expected)


def test_plateau_prefers_a_good_neighbourhood() -> None:
    markets = _markets(1)
    sim = tu.Simulator(COSTS, 10_000.0)
    table, _ = tu.run_grid(markets, tu.combos(GRID), sim)
    summary = tu.summarize_grid(table, GRID, [1])
    row = summary.iloc[0]
    same = summary[summary["model"] == row["model"]]
    # Sur une grille à deux valeurs par axe, tous les réglages sont voisins.
    assert row["plateau"] == pytest.approx(same["sharpe"].mean())


def test_costs_and_references() -> None:
    markets = _markets(2)
    sim = tu.Simulator(COSTS, 10_000.0)
    combo = tu.combos(GRID)[0]
    costs = tu.costs_sensitivity(markets, combo, sim, [0.0, 1.0, 5.0])
    by = costs.groupby("costs_factor")["total_return"].mean()
    assert by[0.0] >= by[1.0] >= by[5.0]
    refs, draws = tu.references(markets, combo, sim, runs=15, seed=0)
    assert len(refs) == 2 and len(draws) == 30
    assert (draws.groupby("fold")["n_trades"].max() == refs["agents_trades"].to_numpy()).all()
    assert (refs["oracle_sharpe"] > refs["agents_sharpe"]).all()


# --- Périodes du walk-forward -----------------------------------------------------------


def test_fold_rows_never_look_past_the_fold(cfg: Config, clean_series: pd.DataFrame) -> None:
    from gold_agents.data.prepare import prepare

    frame = before_test(prepare(clean_series, cfg).frame, cfg.splits.test_start)
    assert pd.to_datetime(frame["date"]).max().date() < cfg.splits.test_start
    start, end, w = date(2013, 1, 1), date(2014, 12, 31), cfg.features.window
    rows = fold_rows(frame, w, start, end, 250)
    days = pd.to_datetime(frame["date"]).dt.date
    # Apprentissage et arrêt anticipé : cible (le lendemain) avant la période.
    assert days.shift(-1)[rows.fit | rows.early_stop].max() < start
    assert rows.early_stop.sum() == 250
    gap = np.flatnonzero(rows.early_stop)[0] - np.flatnonzero(rows.fit)[-1]
    assert gap > w
    assert days[rows.predict].min() >= start and days[rows.predict].max() <= end


def test_tuning_folds_must_stay_before_the_test(cfg: Config) -> None:
    raw = cfg.model_dump(mode="json")
    raw["tuning"]["folds"][-1] = ["2019-01-01", "2022-06-30"]
    with pytest.raises(ValueError, match="test"):
        Config.model_validate(raw)
    raw["tuning"]["folds"][-1] = ["2018-06-01", "2021-12-31"]  # chevauche la précédente
    with pytest.raises(ValueError, match="chevauchement"):
        Config.model_validate(raw)
