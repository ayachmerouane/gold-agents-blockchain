"""Phase 6 : réglage des agents sur une grille grossière, en walk-forward (§4.3, §6.3, R4).

Pour chaque période du walk-forward et chaque réglage de la grille (modèle, θ_buy,
θ_sell, k, stop-loss, coupe-circuit), on simule les agents sur les prévisions hors
échantillon de la période (`models.walk_forward`) et on note Sharpe, rendement, perte
max et nombre de trades. Buy & Hold est simulé sur les mêmes jours.

Règle de choix, sur un ensemble de périodes :

1. **contrainte** : une perte max (drawdown) pas plus profonde que celle de Buy & Hold,
   dans le plus de périodes possible (toutes, si un réglage y arrive) ;
2. **critère** : le Sharpe moyen sur les périodes, et celui des **voisins** dans la grille
   (moyenne du réglage et de ceux à un cran de lui). Le score est le plus petit des deux :
   un réglage doit être bon lui-même et entouré de bons voisins. On préfère ainsi un
   plateau à un pic isolé, qui a toutes les chances d'être un coup de chance.

Walk-forward : la période k est jugée avec le réglage choisi sur les périodes 1 à k-1,
c'est-à-dire sans l'avoir vue. Le réglage final, celui qu'on fige, est choisi sur
toutes les périodes.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from gold_agents.config import TuningGrid
from gold_agents.simulation.costs import CostModel
from gold_agents.simulation.fast import (
    AgentParams,
    FastResult,
    oracle_positions,
    random_positions,
    simulate_agents,
    simulate_positions,
)

PARAMS = ("theta_buy", "theta_sell", "min_confidence", "stop_loss", "max_drawdown")
NO_SIGMA = ("lightgbm",)  # modèles sans incertitude : pas de filtre de confiance


@dataclass(frozen=True)
class Combo:
    model: str
    theta_buy: float
    theta_sell: float
    min_confidence: float | None
    stop_loss: float
    max_drawdown: float

    def params(self, max_position: float, cooldown_days: int | None) -> AgentParams:
        return AgentParams(
            self.theta_buy,
            self.theta_sell,
            self.min_confidence,
            self.stop_loss,
            self.max_drawdown,
            max_position,
            cooldown_days,
        )

    def describe(self) -> str:
        k = "sans filtre" if self.min_confidence is None else f"k {self.min_confidence:g}"
        return (
            f"{self.model}, θ_buy {self.theta_buy:.2%}, θ_sell {self.theta_sell:.2%}, {k}, "
            f"stop-loss {self.stop_loss:.0%}, coupe-circuit {self.max_drawdown:.0%}"
        )


def combos(grid: TuningGrid) -> list[Combo]:
    out = []
    for model in grid.models:
        ks = [None] if model in NO_SIGMA else grid.min_confidence
        for tb, ts, k, sl, dd in itertools.product(
            grid.theta_buy, grid.theta_sell, ks, grid.stop_loss, grid.max_drawdown
        ):
            out.append(Combo(model, tb, ts, k, sl, dd))
    return out


@dataclass(frozen=True)
class FoldMarket:
    """Les jours d'une période : prix et prévision de chaque modèle (NaN : pas d'avis)."""

    fold: int
    dates: list[date]
    opens: np.ndarray
    closes: np.ndarray
    r_hat: dict[str, np.ndarray]
    sigma_hat: dict[str, np.ndarray | None]


def fold_markets(
    frame: pd.DataFrame, forecasts: pd.DataFrame, folds: list[tuple[date, date]]
) -> list[FoldMarket]:
    """Découpe les prix et les prévisions walk-forward par période.

    La période simulée va du premier jour prédit au dernier jour de la période.
    """
    days = pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.date
    out = []
    for i, (_, end) in enumerate(folds, start=1):
        fc = forecasts[forecasts["fold"] == i]
        fc_days = pd.to_datetime(fc["date"]).dt.tz_localize(None).dt.date
        first = fc_days.min()
        part = frame[(days >= first) & (days <= end)]
        pdays = list(days[part.index])
        r_hat, sigma_hat = {}, {}
        for model, g in fc.groupby("model"):
            gd = pd.to_datetime(g["date"]).dt.tz_localize(None).dt.date
            r = pd.Series(g["r_hat"].to_numpy(), index=gd).reindex(pdays)
            s = pd.Series(g["sigma_hat"].to_numpy(), index=gd).reindex(pdays)
            r_hat[str(model)] = r.to_numpy(dtype="float64")
            sigma_hat[str(model)] = None if s.isna().all() else s.fillna(0).to_numpy("float64")
        out.append(
            FoldMarket(
                i,
                pdays,
                part["open"].to_numpy("float64"),
                part["close"].to_numpy("float64"),
                r_hat,
                sigma_hat,
            )
        )
    return out


@dataclass(frozen=True)
class Simulator:
    costs: CostModel
    capital: float
    max_position: float = 1.0
    cooldown_days: int | None = None
    risk_free_rate: float = 0.0

    def agents(self, m: FoldMarket, combo: Combo, costs_factor: float = 1.0) -> FastResult:
        return simulate_agents(
            m.opens,
            m.closes,
            m.r_hat[combo.model],
            m.sigma_hat[combo.model],
            combo.params(self.max_position, self.cooldown_days),
            self.costs.scaled(costs_factor),
            self.capital,
        )

    def positions(self, m: FoldMarket, want: np.ndarray, costs_factor: float = 1.0) -> FastResult:
        return simulate_positions(
            m.opens,
            m.closes,
            want,
            self.costs.scaled(costs_factor),
            self.capital,
            self.max_position,
        )

    def buy_and_hold(self, m: FoldMarket, costs_factor: float = 1.0) -> FastResult:
        return self.positions(m, np.ones(len(m.closes), dtype=bool), costs_factor)


# --- Grille --------------------------------------------------------------------------


def run_grid(
    markets: list[FoldMarket],
    grid: list[Combo],
    sim: Simulator,
    log: Callable[[str], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Une ligne par (réglage, période) ; et Buy & Hold par période."""
    rows = []
    bh_rows = []
    for m in markets:
        bh = sim.buy_and_hold(m).metrics(sim.risk_free_rate)
        bh_rows.append({"fold": m.fold, "start": m.dates[0], "end": m.dates[-1], **bh})
        for combo in grid:
            if combo.model not in m.r_hat:
                continue
            res = sim.agents(m, combo).metrics(sim.risk_free_rate)
            rows.append({**asdict(combo), "fold": m.fold, **res})
        if log is not None:
            log(f"  période {m.fold} : {len(grid)} réglages simulés")
    table = pd.DataFrame(rows)
    bh_table = pd.DataFrame(bh_rows)
    table = table.merge(
        bh_table[["fold", "worst_drawdown"]].rename(
            columns={"worst_drawdown": "bh_worst_drawdown"}
        ),
        on="fold",
    )
    table["dd_ok"] = table["worst_drawdown"] >= table["bh_worst_drawdown"] - 1e-12
    return table, bh_table


def _key(row: pd.Series | dict[str, Any]) -> tuple[Any, ...]:
    return (row["model"], *(row[p] for p in PARAMS))


def _neighbour_mean(summary: pd.DataFrame, grid: TuningGrid) -> pd.Series:
    """Sharpe moyen du réglage et de ses voisins (un cran au plus sur chaque paramètre)."""
    axes = {p: list(getattr(grid, p)) for p in PARAMS}
    if None not in axes["min_confidence"]:  # LightGBM, sans filtre, même hors grille
        axes["min_confidence"].insert(0, None)
    pos = {
        p: summary[p].map(lambda v, a=axes[p]: a.index(None if pd.isna(v) else v)) for p in PARAMS
    }
    coords = np.stack([pos[p].to_numpy() for p in PARAMS], axis=1)
    values = summary["sharpe"].to_numpy()
    models = summary["model"].to_numpy()
    out = np.empty(len(summary))
    for i in range(len(summary)):
        near = (models == models[i]) & (np.abs(coords - coords[i]).max(axis=1) <= 1)
        out[i] = values[near].mean()
    return pd.Series(out, index=summary.index)


def summarize_grid(table: pd.DataFrame, grid: TuningGrid, folds: list[int]) -> pd.DataFrame:
    """Résumé de chaque réglage sur `folds`, trié du meilleur au moins bon."""
    part = table[table["fold"].isin(folds)].copy()
    part["min_confidence"] = part["min_confidence"].astype("object")
    keys = ["model", *PARAMS]
    summary = (
        part.groupby(keys, dropna=False, sort=False)
        .agg(
            sharpe=("sharpe", "mean"),
            sharpe_min=("sharpe", "min"),
            total_return=("total_return", "mean"),
            worst_drawdown=("worst_drawdown", "min"),
            n_trades=("n_trades", "mean"),
            exposure=("exposure", "mean"),
            dd_ok=("dd_ok", "sum"),
        )
        .reset_index()
    )
    summary["min_confidence"] = summary["min_confidence"].map(
        lambda v: None if pd.isna(v) else float(v)
    )
    summary["plateau"] = _neighbour_mean(summary, grid)
    summary["score"] = np.minimum(summary["sharpe"], summary["plateau"])
    summary["n_folds"] = len(folds)
    return summary.sort_values(
        ["dd_ok", "score", "sharpe"], ascending=[False, False, False]
    ).reset_index(drop=True)


def best(summary: pd.DataFrame) -> Combo:
    row = summary.iloc[0]
    k = row["min_confidence"]
    return Combo(
        str(row["model"]),
        float(row["theta_buy"]),
        float(row["theta_sell"]),
        None if k is None or pd.isna(k) else float(k),
        float(row["stop_loss"]),
        float(row["max_drawdown"]),
    )


# --- Walk-forward, frais, références ------------------------------------------------


def walk_forward(
    table: pd.DataFrame,
    bh: pd.DataFrame,
    grid: TuningGrid,
    markets: list[FoldMarket],
    sim: Simulator,
) -> pd.DataFrame:
    """Période k jugée avec le réglage choisi sur les périodes 1 à k-1."""
    rows = []
    folds = [m.fold for m in markets]
    for m in markets[1:]:
        chosen = best(summarize_grid(table, grid, [f for f in folds if f < m.fold]))
        res = sim.agents(m, chosen).metrics(sim.risk_free_rate)
        b = bh[bh["fold"] == m.fold].iloc[0]
        rows.append(
            {
                "fold": m.fold,
                "start": m.dates[0],
                "end": m.dates[-1],
                "réglage choisi sur": f"périodes 1-{m.fold - 1}",
                **asdict(chosen),
                **res,
                "bh_sharpe": float(b["sharpe"]),
                "bh_total_return": float(b["total_return"]),
                "bh_worst_drawdown": float(b["worst_drawdown"]),
            }
        )
    return pd.DataFrame(rows)


def costs_sensitivity(
    markets: list[FoldMarket], combo: Combo, sim: Simulator, factors: list[float]
) -> pd.DataFrame:
    rows = []
    for factor in factors:
        for m in markets:
            agents = sim.agents(m, combo, factor).metrics(sim.risk_free_rate)
            bh = sim.buy_and_hold(m, factor).metrics(sim.risk_free_rate)
            rows.append(
                {
                    "costs_factor": factor,
                    "fold": m.fold,
                    "sharpe": agents["sharpe"],
                    "total_return": agents["total_return"],
                    "bh_sharpe": bh["sharpe"],
                    "bh_total_return": bh["total_return"],
                }
            )
    return pd.DataFrame(rows)


def references(
    markets: list[FoldMarket], combo: Combo, sim: Simulator, runs: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Par période : agents, hasard (même nombre de trades, `runs` tirages) et oracle."""
    rng = np.random.default_rng(seed)
    rows, draws = [], []
    for m in markets:
        agents = sim.agents(m, combo)
        a = agents.metrics(sim.risk_free_rate)
        sharpes = []
        for run in range(runs):
            want = random_positions(len(m.closes), agents.n_trades, rng)
            r = sim.positions(m, want).metrics(sim.risk_free_rate)
            sharpes.append(r["sharpe"])
            draws.append({"fold": m.fold, "run": run, **r})
        oracle = sim.positions(m, oracle_positions(m.opens)).metrics(sim.risk_free_rate)
        s = np.asarray(sharpes)
        rows.append(
            {
                "fold": m.fold,
                "agents_sharpe": a["sharpe"],
                "agents_trades": int(a["n_trades"]),
                "random_sharpe_median": float(np.median(s)),
                "random_sharpe_p95": float(np.quantile(s, 0.95)),
                # Part des tirages au hasard que les agents battent (en Sharpe).
                "beats_random": float(np.mean(a["sharpe"] > s)),
                "oracle_sharpe": oracle["sharpe"],
                "oracle_total_return": oracle["total_return"],
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(draws)
