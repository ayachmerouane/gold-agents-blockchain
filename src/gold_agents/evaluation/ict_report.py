"""Stratégie ICT : grille, walk-forward, frais, hasard et rapport (`gold-agents ict`).

Même protocole que la phase 6 : la période k est jugée avec le réglage choisi sur les
périodes 1 à k-1 seulement. Le test (2022+) n'est jamais lu : `frame` est coupé avant.
"""

from __future__ import annotations

import itertools
from dataclasses import asdict
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from gold_agents.config import Config, IctConfig
from gold_agents.evaluation.financial_metrics import summarize
from gold_agents.simulation.costs import CostModel
from gold_agents.simulation.engine import BacktestResult, run_backtest
from gold_agents.simulation.fast import random_positions, simulate_positions
from gold_agents.simulation.ict import IctParams, run_ict_backtest
from gold_agents.simulation.strategies import BuyAndHold, SmaCrossover

PARAM_COLUMNS = ["entry", "swing", "sweep_lookback", "discount", "min_reward_risk"]


def ict_combos(ict: IctConfig) -> list[IctParams]:
    g = ict.grid
    return [
        IctParams(e, s, lb, d, ict.max_hold_days, rr)
        for e, s, lb, d, rr in itertools.product(
            g.entry, g.swing, g.sweep_lookback, g.discount, g.min_reward_risk
        )
    ]


def _years(start: date, end: date) -> float:
    return max((end - start).days / 365.25, 1e-9)


def _row(result: BacktestResult, fold: int, start: date, end: date, rf: float) -> dict[str, Any]:
    s = summarize(result, rf)
    return {
        "fold": fold,
        "start": start,
        "end": end,
        "sharpe": s["sharpe"],
        "total_return": s["total_return"],
        "max_drawdown": s["max_drawdown"],
        "n_trades": s["n_trades"],
        "trades_per_year": s["n_trades"] / _years(start, end),
        "win_rate": s["win_rate"],
        "exposure": s["exposure"],
    }


def run_grid(
    frame: pd.DataFrame,
    combos: list[IctParams],
    folds: list[tuple[date, date]],
    costs: CostModel,
    capital: float,
    rf: float = 0.0,
) -> pd.DataFrame:
    """Chaque réglage sur chaque période (une ligne par réglage et par période)."""
    rows = []
    for p in combos:
        for k, (start, end) in enumerate(folds, start=1):
            res = run_ict_backtest(frame, p, costs, capital, start, end)
            rows.append({**_params(p), **_row(res, k, start, end, rf)})
    return pd.DataFrame(rows)


def references(
    frame: pd.DataFrame,
    folds: list[tuple[date, date]],
    costs: CostModel,
    capital: float,
    rf: float = 0.0,
) -> pd.DataFrame:
    """Buy & Hold et SMA 20/50 sur chaque période, avec le moteur commun."""
    rows = []
    for k, (start, end) in enumerate(folds, start=1):
        for strategy in (BuyAndHold(), SmaCrossover()):
            res = run_backtest(frame, strategy, costs, capital, start, end)
            rows.append({"strategy": strategy.name, **_row(res, k, start, end, rf)})
    return pd.DataFrame(rows)


def _params(p: IctParams) -> dict[str, Any]:
    d = asdict(p)
    return {c: d[c] for c in PARAM_COLUMNS}


def choose(
    grid: pd.DataFrame,
    folds: list[int],
    min_tpy: float,
    max_tpy: float | None,
    columns: list[str] = PARAM_COLUMNS,
) -> tuple[dict[str, Any], bool]:
    """Réglage au meilleur Sharpe moyen sur `folds`, de préférence dans les bornes de trades.

    Renvoie (réglage, True si le nombre de trades par an respecte les bornes partout).
    """
    part = grid[grid["fold"].isin(folds)]
    within = (part["trades_per_year"] >= min_tpy) & (
        part["trades_per_year"] <= (max_tpy if max_tpy is not None else np.inf)
    )
    stats = (
        part.assign(within=within)
        .groupby(columns, sort=True)
        .agg(sharpe=("sharpe", "mean"), within=("within", "all"))
        .reset_index()
    )
    pool = stats[stats["within"]] if stats["within"].any() else stats
    best = pool.sort_values("sharpe", ascending=False, kind="stable").iloc[0]
    return {c: best[c] for c in columns}, bool(best["within"])


def to_params(chosen: dict[str, Any], max_hold_days: int) -> IctParams:
    return IctParams(
        entry=chosen["entry"],
        swing=int(chosen["swing"]),
        sweep_lookback=int(chosen["sweep_lookback"]),
        discount=float(chosen["discount"]),
        max_hold_days=max_hold_days,
        min_reward_risk=float(chosen["min_reward_risk"]),
    )


def walk_forward(
    grid: pd.DataFrame,
    refs: pd.DataFrame,
    n_folds: int,
    min_tpy: float,
    max_tpy: float | None,
    columns: list[str] = PARAM_COLUMNS,
) -> pd.DataFrame:
    """Période k (k ≥ 2) jugée avec le réglage choisi sur les périodes 1 à k-1."""
    rows = []
    for k in range(2, n_folds + 1):
        chosen, within = choose(grid, list(range(1, k)), min_tpy, max_tpy, columns)
        mask = (grid["fold"] == k) & np.logical_and.reduce(
            [grid[c] == v for c, v in chosen.items()]
        )
        res = grid[mask].iloc[0]
        bh = refs[(refs["fold"] == k) & (refs["strategy"] == "buy_and_hold")].iloc[0]
        sma = refs[(refs["fold"] == k) & (refs["strategy"] == "sma_20_50")].iloc[0]
        rows.append(
            {
                "fold": k,
                "start": res["start"],
                "end": res["end"],
                "réglage choisi sur": f"périodes 1-{k - 1}",
                **chosen,
                "trades_ok_avant": within,
                **{
                    c: res[c]
                    for c in [
                        "sharpe",
                        "total_return",
                        "max_drawdown",
                        "n_trades",
                        "trades_per_year",
                        "win_rate",
                        "exposure",
                    ]
                },
                "bh_sharpe": bh["sharpe"],
                "bh_total_return": bh["total_return"],
                "bh_max_drawdown": bh["max_drawdown"],
                "sma_sharpe": sma["sharpe"],
                "sma_total_return": sma["total_return"],
            }
        )
    return pd.DataFrame(rows)


def costs_sensitivity(
    frame: pd.DataFrame,
    wf: pd.DataFrame,
    folds: list[tuple[date, date]],
    costs: CostModel,
    capital: float,
    factors: list[float],
    max_hold_days: int,
    rf: float = 0.0,
) -> pd.DataFrame:
    rows = []
    for _, w in wf.iterrows():
        k = int(w["fold"])
        start, end = folds[k - 1]
        p = to_params({c: w[c] for c in PARAM_COLUMNS}, max_hold_days)
        for f in factors:
            ict = summarize(run_ict_backtest(frame, p, costs.scaled(f), capital, start, end), rf)
            bh = summarize(
                run_backtest(frame, BuyAndHold(), costs.scaled(f), capital, start, end), rf
            )
            rows.append(
                {
                    "costs_factor": f,
                    "fold": k,
                    "sharpe": ict["sharpe"],
                    "total_return": ict["total_return"],
                    "bh_sharpe": bh["sharpe"],
                    "bh_total_return": bh["total_return"],
                }
            )
    return pd.DataFrame(rows)


def random_references(
    frame: pd.DataFrame,
    wf: pd.DataFrame,
    folds: list[tuple[date, date]],
    costs: CostModel,
    capital: float,
    runs: int,
    seed: int,
    rf: float = 0.0,
) -> pd.DataFrame:
    """Par période : part des stratégies au hasard (même nombre de trades) battues en Sharpe."""
    from gold_agents.simulation.fast import sharpe as fast_sharpe

    rng = np.random.default_rng(seed)
    days = frame["date"].map(lambda ts: pd.Timestamp(ts).date())
    rows = []
    for _, w in wf.iterrows():
        k = int(w["fold"])
        start, end = folds[k - 1]
        part = frame[(days >= start) & (days <= end)]
        opens = part["open"].to_numpy("float64")
        closes = part["close"].to_numpy("float64")
        n = int(w["n_trades"])
        s = np.asarray(
            [
                fast_sharpe(
                    simulate_positions(
                        opens, closes, random_positions(len(closes), n, rng), costs, capital
                    ).values,
                    rf,
                )
                for _ in range(runs)
            ]
        )
        rows.append(
            {
                "fold": k,
                "ict_sharpe": w["sharpe"],
                "n_trades": n,
                "random_sharpe_median": float(np.median(s)),
                "random_sharpe_p95": float(np.quantile(s, 0.95)),
                "beats_random": float(np.mean(w["sharpe"] > s)),
            }
        )
    return pd.DataFrame(rows)


def _pct(x: float) -> str:
    return f"{x * 100:+.1f} %".replace(".", ",")


def _num(x: float) -> str:
    return f"{x:.2f}".replace(".", ",")


def _setting(r: pd.Series | dict[str, Any]) -> str:
    return (
        f"{r['entry']}, swing {int(r['swing'])}, sweep ≤ {int(r['sweep_lookback'])} j, "
        f"discount {_num(float(r['discount']))}, R/R ≥ {_num(float(r['min_reward_risk']))}"
    )


def markdown_report(
    cfg: Config,
    n_combos: int,
    wf: pd.DataFrame,
    rand: pd.DataFrame,
    costs: pd.DataFrame,
    final: dict[str, Any],
    final_within: bool,
    grid: pd.DataFrame,
) -> str:
    lines = [
        "# Stratégie ICT : walk-forward 2011-2021",
        "",
        "Achat seulement. Chaque soir, un ordre conditionnel valable un jour est écrit avant "
        "l'ouverture ; le plus haut et le plus bas du lendemain disent s'il est exécuté, puis "
        "si le stop ou l'objectif est touché (pire cas si les deux le sont).",
        f"Grille : {n_combos} réglages. Période k jugée avec le réglage choisi sur les "
        "périodes 1 à k-1 (meilleur Sharpe moyen, nombre de trades par an dans les bornes "
        "de `tuning`). Le test 2022+ n'est jamais lu.",
        "",
        "## Walk-forward (hors échantillon)",
        "",
        "| Période | Réglage | Sharpe ICT | Sharpe B&H | Rendement ICT | Rendement B&H "
        "| Perte max ICT | Perte max B&H | Trades | Investi |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in wf.iterrows():
        lines.append(
            f"| {r['start']:%Y}-{r['end']:%Y} | {_setting(r)} | {_num(r['sharpe'])} "
            f"| {_num(r['bh_sharpe'])} | {_pct(r['total_return'])} | {_pct(r['bh_total_return'])} "
            f"| {_pct(r['max_drawdown'])} | {_pct(r['bh_max_drawdown'])} | {int(r['n_trades'])} "
            f"| {r['exposure'] * 100:.0f} % |"
        )
    lines += [
        "",
        f"Moyenne : Sharpe ICT {_num(wf['sharpe'].mean())} contre {_num(wf['bh_sharpe'].mean())} "
        f"pour Buy & Hold et {_num(wf['sma_sharpe'].mean())} pour SMA 20/50.",
        "",
        "## Hasard (même nombre de trades, exécution à l'ouverture)",
        "",
        "| Période | Trades | Sharpe ICT | Hasard médian | Hasard 95e centile | Hasard battu |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in rand.iterrows():
        lines.append(
            f"| {int(r['fold'])} | {int(r['n_trades'])} | {_num(r['ict_sharpe'])} "
            f"| {_num(r['random_sharpe_median'])} | {_num(r['random_sharpe_p95'])} "
            f"| {r['beats_random'] * 100:.0f} % |"
        )
    lines += [
        "",
        "## Sensibilité aux frais (Sharpe moyen sur le walk-forward)",
        "",
        "| Frais | ICT | Buy & Hold |",
        "|---|---|---|",
    ]
    for f, g in costs.groupby("costs_factor"):
        lines.append(f"| × {f:g} | {_num(g['sharpe'].mean())} | {_num(g['bh_sharpe'].mean())} |")
    best_all = grid.groupby(PARAM_COLUMNS)["sharpe"].mean().max()
    lines += [
        "",
        "## Réglage choisi sur les 5 périodes (pour un futur gel)",
        "",
        f"{_setting(final)}"
        + ("" if final_within else " (aucun réglage ne respecte les bornes de trades)"),
        "",
        f"Attention : {n_combos} réglages essayés. Le meilleur Sharpe moyen de la grille "
        f"({_num(best_all)}) est optimiste ; seul le walk-forward ci-dessus est hors "
        "échantillon.",
    ]
    return "\n".join(lines) + "\n"
