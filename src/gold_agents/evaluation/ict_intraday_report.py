"""Agent ICT intraday : grille, walk-forward, hasard, frais et rapport (`gold-agents ict-intraday`).

Même protocole que la phase 6 et que `ict_report` : la période k est jugée avec le réglage
choisi sur les périodes 1 à k-1. Les métriques sont calculées sur la valeur en fin de
journée (Sharpe annualisé sur des rendements journaliers).
"""

from __future__ import annotations

import itertools
from dataclasses import asdict
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from gold_agents.config import IctIntradayConfig
from gold_agents.evaluation.financial_metrics import summarize
from gold_agents.evaluation.ict_report import _row
from gold_agents.simulation.costs import CostModel
from gold_agents.simulation.engine import run_backtest
from gold_agents.simulation.fast import sharpe as fast_sharpe
from gold_agents.simulation.ict_intraday import (
    IntradayParams,
    daily_equity,
    in_kill_zone,
    run_intraday_backtest,
)
from gold_agents.simulation.strategies import BuyAndHold, SmaCrossover

COLUMNS = ["entry", "swing", "target", "min_reward_risk"]


def combos(cfg: IctIntradayConfig) -> list[IntradayParams]:
    g = cfg.grid
    zones = tuple((int(a), int(b)) for a, b in cfg.kill_zones)
    return [
        IntradayParams(e, s, t, rr, zones, cfg.flat_hour, cfg.max_hold_bars)
        for e, s, t, rr in itertools.product(g.entry, g.swing, g.target, g.min_reward_risk)
    ]


def _params(p: IntradayParams) -> dict[str, Any]:
    d = asdict(p)
    return {c: d[c] for c in COLUMNS}


def to_params(chosen: dict[str, Any], cfg: IctIntradayConfig) -> IntradayParams:
    zones = tuple((int(a), int(b)) for a, b in cfg.kill_zones)
    return IntradayParams(
        chosen["entry"],
        int(chosen["swing"]),
        chosen["target"],
        float(chosen["min_reward_risk"]),
        zones,
        cfg.flat_hour,
        cfg.max_hold_bars,
    )


def daily_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Bougies journalières (jour de cotation) pour Buy & Hold et SMA, datées à minuit UTC."""
    g = frame.groupby("trading_day", sort=True)
    out = pd.DataFrame(
        {
            "open": g["open"].first(),
            "high": g["high"].max(),
            "low": g["low"].min(),
            "close": g["close"].last(),
        }
    ).reset_index()
    out["date"] = pd.to_datetime(out.pop("trading_day")).dt.tz_localize("UTC")
    return out


def run_grid(
    frame: pd.DataFrame,
    params: list[IntradayParams],
    folds: list[tuple[date, date]],
    costs: CostModel,
    capital: float,
    rf: float = 0.0,
) -> pd.DataFrame:
    rows = []
    for p in params:
        for k, (start, end) in enumerate(folds, start=1):
            res = daily_equity(run_intraday_backtest(frame, p, costs, capital, start, end))
            rows.append({**_params(p), **_row(res, k, start, end, rf)})
    return pd.DataFrame(rows)


def references(
    daily: pd.DataFrame,
    folds: list[tuple[date, date]],
    costs: CostModel,
    capital: float,
    rf: float = 0.0,
) -> pd.DataFrame:
    rows = []
    for k, (start, end) in enumerate(folds, start=1):
        for strategy in (BuyAndHold(), SmaCrossover()):
            res = run_backtest(daily, strategy, costs, capital, start, end)
            rows.append({"strategy": strategy.name, **_row(res, k, start, end, rf)})
    return pd.DataFrame(rows)


def random_sharpes(
    frame: pd.DataFrame,
    start: date,
    end: date,
    n_round_trips: int,
    params: IntradayParams,
    costs: CostModel,
    runs: int,
    rng: np.random.Generator,
    rf: float = 0.0,
) -> np.ndarray:
    """Hasard comparable : même nombre d'allers-retours, achat à l'ouverture d'une barre de
    kill zone tirée au hasard, vente à la clôture de la séance, le même jour."""
    part = frame[(frame["trading_day"] >= start) & (frame["trading_day"] <= end)]
    days = np.asarray(sorted(part["trading_day"].unique()))
    hours = part["ny_hour"].to_numpy()
    kz = np.array([in_kill_zone(int(h), params.kill_zones) for h in hours])
    flat = hours == (params.flat_hour - 1) % 24
    by_day: dict[Any, tuple[np.ndarray, float | None]] = {}
    opens, closes, tdays = part["open"].to_numpy(), part["close"].to_numpy(), part["trading_day"]
    for day, idx in tdays.groupby(tdays).indices.items():
        idx = np.asarray(idx)
        exit_close = closes[idx[flat[idx]]][-1] if flat[idx].any() else None
        by_day[day] = (opens[idx[kz[idx]]], exit_close)
    eligible = [d for d in days if len(by_day[d][0]) and by_day[d][1] is not None]
    n = min(n_round_trips, len(eligible))
    pos = {d: i for i, d in enumerate(days)}
    out = []
    for _ in range(runs):
        r = np.zeros(len(days))
        for d in rng.choice(np.asarray(eligible, dtype=object), size=n, replace=False):
            entries, exit_close = by_day[d]
            buy = costs.buy_price(float(rng.choice(entries)))
            sell = costs.sell_price(float(exit_close))  # type: ignore[arg-type]
            r[pos[d]] = sell / buy * (1 - costs.fee_rate) / (1 + costs.fee_rate) - 1
        out.append(fast_sharpe(np.cumprod(np.r_[1.0, 1 + r]), rf))
    return np.asarray(out)


def random_references(
    frame: pd.DataFrame,
    wf: pd.DataFrame,
    folds: list[tuple[date, date]],
    cfg: IctIntradayConfig,
    costs: CostModel,
    runs: int,
    seed: int,
    rf: float = 0.0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _, w in wf.iterrows():
        k = int(w["fold"])
        start, end = folds[k - 1]
        p = to_params({c: w[c] for c in COLUMNS}, cfg)
        s = random_sharpes(frame, start, end, int(w["n_trades"]) // 2, p, costs, runs, rng, rf)
        rows.append(
            {
                "fold": k,
                "ict_sharpe": w["sharpe"],
                "n_trades": int(w["n_trades"]),
                "random_sharpe_median": float(np.median(s)),
                "random_sharpe_p95": float(np.quantile(s, 0.95)),
                "beats_random": float(np.mean(w["sharpe"] > s)),
            }
        )
    return pd.DataFrame(rows)


def costs_sensitivity(
    frame: pd.DataFrame,
    daily: pd.DataFrame,
    wf: pd.DataFrame,
    folds: list[tuple[date, date]],
    cfg: IctIntradayConfig,
    costs: CostModel,
    capital: float,
    factors: list[float],
    rf: float = 0.0,
) -> pd.DataFrame:
    rows = []
    for _, w in wf.iterrows():
        k = int(w["fold"])
        start, end = folds[k - 1]
        p = to_params({c: w[c] for c in COLUMNS}, cfg)
        for f in factors:
            res = run_intraday_backtest(frame, p, costs.scaled(f), capital, start, end)
            ict = summarize(daily_equity(res), rf)
            bh = summarize(
                run_backtest(daily, BuyAndHold(), costs.scaled(f), capital, start, end), rf
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


def _pct(x: float) -> str:
    return f"{x * 100:+.1f} %".replace(".", ",")


def _num(x: float) -> str:
    return f"{x:.2f}".replace(".", ",")


def _setting(r: pd.Series | dict[str, Any]) -> str:
    return (
        f"{r['entry']}, swing {int(r['swing'])}, objectif {r['target']}, "
        f"R/R ≥ {_num(float(r['min_reward_risk']))}"
    )


def markdown_report(
    n_combos: int,
    wf: pd.DataFrame,
    rand: pd.DataFrame,
    costs: pd.DataFrame,
    final: dict[str, Any],
    final_within: bool,
    grid: pd.DataFrame,
    data_info: str,
) -> str:
    lines = [
        "# Agent ICT intraday (barres 1 h) : walk-forward 2011-2021",
        "",
        data_info,
        "",
        "L'agent suit le marché barre par barre et décide pendant la séance : sweep du bas "
        "asiatique, changement de structure, entrée en kill zone (Londres, New York) par "
        "ordre valable une barre, sortie au stop, à l'objectif (pire cas) ou en fin de "
        "séance. Achat seulement, jamais de position la nuit.",
        f"Grille : {n_combos} réglages. Période k jugée avec le réglage choisi sur les "
        "périodes 1 à k-1. Le test 2022+ n'est jamais lu.",
        "",
        "## Walk-forward (hors échantillon)",
        "",
        "| Période | Réglage | Sharpe ICT | Sharpe B&H | Rendement ICT | Rendement B&H "
        "| Perte max ICT | Perte max B&H | Trades | Gagnants |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in wf.iterrows():
        win = "-" if pd.isna(r["win_rate"]) else f"{r['win_rate'] * 100:.0f} %"
        lines.append(
            f"| {r['start']:%Y}-{r['end']:%Y} | {_setting(r)} | {_num(r['sharpe'])} "
            f"| {_num(r['bh_sharpe'])} | {_pct(r['total_return'])} | {_pct(r['bh_total_return'])} "
            f"| {_pct(r['max_drawdown'])} | {_pct(r['bh_max_drawdown'])} | {int(r['n_trades'])} "
            f"| {win} |"
        )
    lines += [
        "",
        f"Moyenne : Sharpe ICT {_num(wf['sharpe'].mean())} contre {_num(wf['bh_sharpe'].mean())} "
        f"pour Buy & Hold et {_num(wf['sma_sharpe'].mean())} pour SMA 20/50.",
        "",
        "## Hasard (même nombre d'allers-retours, entrée en kill zone, sortie en fin de séance)",
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
    best_all = grid.groupby(COLUMNS)["sharpe"].mean().max()
    lines += [
        "",
        "## Réglage choisi sur les 5 périodes (candidat au gel)",
        "",
        _setting(final) + ("" if final_within else " (hors des bornes de trades par an)"),
        "",
        f"Attention : {n_combos} réglages essayés. Le meilleur Sharpe moyen de la grille "
        f"({_num(best_all)}) est optimiste ; seul le walk-forward est hors échantillon.",
    ]
    return "\n".join(lines) + "\n"
