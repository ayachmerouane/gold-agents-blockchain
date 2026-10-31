"""Phase 8 : évaluation finale, un seul run sur le test (§7, §12).

Rien n'est réglé ici. On reprend la procédure figée en phase 6, à l'identique :

- **prévisions** : LSTM, GRU, l'ensemble et LightGBM sont réentraînés sur tout ce qui
  précède le test (2004-2021), avec les hyperparamètres de la phase 4, comme pour chaque
  période du walk-forward de la phase 6 (`models.walk_forward`) ; la dernière année du
  passé sert à l'arrêt anticipé ; le test n'est lu que pour prédire, jour par jour ;
- **agents** : seuils et risque de `config.yaml`, vérifiés contre le bloc genèse des
  réglages figés (`reglages_figes.jsonl`) avant de lire le moindre prix du test ;
- **stratégies** : Buy & Hold, SMA 20/50, les cinq systèmes d'agents de la phase 5, une
  stratégie au hasard (même nombre de trades que le système retenu) et l'oracle, qui
  connaît le futur (plafond, pour illustration). Toutes passent par le même moteur.

Mesures : métriques financières (§7.4), précision des prévisions (§5.4), sensibilité aux
frais (× 0, 1, 2, 5), 200 tirages au hasard, rendements par année, et une alerte « trop
beau » (plus de 60 % de bonnes directions, Sharpe au-dessus de 3, agents meilleurs que
l'oracle) : c'est presque toujours une fuite du futur.

Une répétition générale sur la validation (`split="val"`) suit exactement le même code,
avec des modèles appris avant 2019 : elle teste la chaîne sans regarder le test.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gold_agents.agents.forecast import ForecastTable
from gold_agents.agents.orchestrator import run_agents
from gold_agents.blockchain.chain import Chain
from gold_agents.blockchain.signing import Signer
from gold_agents.blockchain.storage import read_jsonl
from gold_agents.config import Config
from gold_agents.evaluation import agents_report as ar
from gold_agents.evaluation import report
from gold_agents.evaluation.financial_metrics import drawdown, summarize
from gold_agents.evaluation.forecast_metrics import forecast_metrics
from gold_agents.evaluation.report import GRID, INK, MUTED, RED, Run, _money, _style
from gold_agents.models import walk_forward as wf
from gold_agents.simulation.costs import CostModel
from gold_agents.simulation.engine import BacktestResult, run_backtest
from gold_agents.simulation.fast import oracle_positions, random_positions, simulate_positions
from gold_agents.simulation.strategies import (
    HOLD,
    Action,
    BuyAndHold,
    Decision,
    SmaCrossover,
    Strategy,
)

LABELS = report.LABELS

# Bloc genèse des réglages figés le 2026-10-30 (`gold-agents freeze`, phase 6).
FROZEN_HEAD = "ca1c3fd2a834b035cdb1b8c55a1726faca8b89637276d7c1eb037c5a192d60c9"
PLAN_FILE = "plan.jsonl"
RESULTS_FILE = "resultats.jsonl"
SHARPE_ALERT = 3.0
MAIN = ar.MAIN
RANDOM, ORACLE = "aleatoire", "oracle"
REFERENCE_LABELS = {
    RANDOM: "Au hasard (même nb de trades)",
    ORACLE: "Oracle (connaît le futur)",
}
report.LABELS.update(REFERENCE_LABELS)
report.COLORS.update({RANDOM: "#9a9893", ORACLE: "#d6b4e8"})
MODEL_LABELS = {
    "lstm": "LSTM",
    "gru": "GRU",
    "ensemble": "Ensemble LSTM + GRU",
    "lightgbm": "LightGBM",
    "b0_zero": "B0 rendement nul",
    "always_up": "Toujours « hausse »",
}


# --- Contrôle des réglages figés -----------------------------------------------------


def current_settings(cfg: Config) -> dict[str, Any]:
    st, rk = cfg.strategy, cfg.risk
    return {
        "model": st.model,
        "theta_buy": st.theta_buy,
        "theta_sell": st.theta_sell,
        "min_confidence": st.min_confidence,
        "stop_loss": rk.stop_loss,
        "max_drawdown": rk.max_drawdown,
        "cooldown_days": rk.cooldown_days,
    }


def _config_diff(a: dict[str, Any], b: dict[str, Any], prefix: str = "") -> list[str]:
    out = []
    for key in sorted(set(a) | set(b)):
        name = f"{prefix}{key}"
        va, vb = a.get(key), b.get(key)
        if isinstance(va, dict) and isinstance(vb, dict):
            out += _config_diff(va, vb, f"{name}.")
        elif va != vb:
            out.append(f"{name} : config {va!r}, figé {vb!r}")
    return out


@dataclass(frozen=True)
class FrozenCheck:
    genesis: dict[str, Any]
    head: str
    problems: list[str]

    @property
    def ok(self) -> bool:
        return not self.problems


def check_frozen(
    cfg: Config, frozen_path: Path, expected_head: str | None, data_sha256: str
) -> FrozenCheck:
    """La configuration et les données sont-elles exactement celles qui ont été figées ?

    Les chemins (`data`) sont ignorés : ils s'écrivent différemment selon le système.
    """
    if not frozen_path.exists():
        return FrozenCheck({}, "", [f"{frozen_path} absent (lancer `gold-agents freeze`)"])
    chain = read_jsonl(frozen_path)
    problems = []
    verification = chain.verify(None)
    if not verification.valid:
        problems.append(f"{frozen_path} : {verification.describe()}")
    head = chain.head.hash
    if expected_head and head != expected_head:
        problems.append(f"bloc genèse {head[:12]}… au lieu de {expected_head[:12]}…")
    genesis = chain.blocks[0].events[0] if chain.blocks and chain.blocks[0].events else {}
    if genesis.get("kind") != "FROZEN_SETTINGS":
        problems.append("le premier bloc n'est pas un bloc de réglages figés")
        return FrozenCheck(genesis, head, problems)
    for key, (now, frozen) in {
        k: (v, genesis["settings"].get(k)) for k, v in current_settings(cfg).items()
    }.items():
        if now != frozen:
            problems.append(f"réglage {key} : config {now!r}, figé {frozen!r}")
    current = json.loads(cfg.model_dump_json())
    frozen_cfg = dict(genesis.get("config", {}))
    current.pop("data", None)
    frozen_cfg.pop("data", None)
    problems += _config_diff(current, frozen_cfg)
    if genesis.get("data_sha256") != data_sha256:
        problems.append(
            "fichier de features différent de celui du gel "
            f"({data_sha256[:12]}… au lieu de {str(genesis.get('data_sha256'))[:12]}…)"
        )
    return FrozenCheck(genesis, head, problems)


def check_procedure(
    tuning_meta_path: Path,
    net_meta: dict[str, dict[str, Any]],
    lgbm_params: dict[str, Any] | None,
) -> list[str]:
    """Les hyperparamètres des modèles sont-ils ceux du walk-forward de la phase 6 ?"""
    if not tuning_meta_path.exists():
        return [f"{tuning_meta_path} absent : hyperparamètres non comparés à la phase 6"]
    key = json.loads(tuning_meta_path.read_text(encoding="utf-8"))["key"]
    networks = {k: {"net": v["net"], "train": v["train"]} for k, v in net_meta.items()}
    out = []
    if key.get("networks") != json.loads(json.dumps(networks)):
        out.append("hyperparamètres LSTM/GRU différents de ceux de `tune`")
    if key.get("lightgbm") != json.loads(json.dumps(lgbm_params)):
        out.append("hyperparamètres LightGBM différents de ceux de `tune`")
    return out


# --- Période et prévisions -----------------------------------------------------------


def split_bounds(frame: pd.DataFrame, cfg: Config, split: str) -> tuple[date, date]:
    """Premier jour civil du split (config) et dernier jour présent dans les données."""
    days = pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.date
    part = days[frame["split"] == split]
    if part.empty:
        raise ValueError(f"aucune ligne {split} dans le fichier de features")
    start = cfg.splits.test_start if split == "test" else cfg.splits.val_start
    return start, part.iloc[-1]


def final_forecasts(
    frame: pd.DataFrame,
    columns: list[str],
    window: int,
    start: date,
    end: date,
    early_stop_days: int,
    net_meta: dict[str, dict[str, Any]],
    lgbm_params: dict[str, Any] | None,
    seeds: list[int],
    max_epochs: int,
    patience: int,
    mc_samples: int,
    threads: int = 0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Prévisions de [start, end] par des modèles appris sur tout ce qui précède `start`.

    C'est une période de plus du walk-forward de la phase 6, avec le même code.
    """
    rows = wf.fold_rows(frame, window, start, end, early_stop_days)
    data = wf.fold_inputs(frame, columns, window, rows)
    forecasts, info = wf.fold_forecasts(
        data, net_meta, lgbm_params, seeds, max_epochs, patience, mc_samples, threads
    )
    info = {"start": str(start), "end": str(end), "n_predicted": len(data.dates), **info}
    return forecasts, info


def tables_from(forecasts: pd.DataFrame) -> dict[str, ForecastTable]:
    tables = {}
    for model, g in forecasts.groupby("model", sort=False):
        days = [pd.Timestamp(d).date() for d in g["date"]]
        sigma = g["sigma_hat"].to_numpy("float64")
        tables[str(model)] = ForecastTable(
            "ensemble_lstm_gru" if model == "ensemble" else str(model),
            days,
            g["r_hat"].to_numpy("float64"),
            None if np.isnan(sigma).all() else np.nan_to_num(sigma),
        )
    return tables


# --- Stratégies de référence par positions (hasard, oracle) --------------------------


@dataclass
class PositionStrategy:
    """Être investi le lendemain ou non, donné à l'avance jour par jour."""

    name: str
    want: dict[date, bool]
    reason: str

    def decide(self, history: pd.DataFrame, qty: float) -> Decision:
        today = pd.Timestamp(history["date"].iloc[-1]).date()
        invest = self.want.get(today, False)
        if invest and qty == 0:
            return Decision(Action.BUY, f"{self.reason} : investi demain")
        if not invest and qty > 0:
            return Decision(Action.SELL, f"{self.reason} : en cash demain")
        return HOLD


@dataclass(frozen=True)
class Market:
    dates: list[date]
    opens: np.ndarray
    closes: np.ndarray


def market(frame: pd.DataFrame, start: date, end: date) -> Market:
    days = pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.date
    part = frame[(days >= start) & (days <= end)]
    return Market(
        list(days[part.index]), part["open"].to_numpy("float64"), part["close"].to_numpy("float64")
    )


# --- Runs ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Outcome:
    run: Run
    decisions: pd.DataFrame | None


def _genesis(
    cfg: Config, data_sha256: str, name: str, extra: dict[str, Any]
) -> list[dict[str, Any]]:
    return [
        {
            "type": "GENESIS",
            "strategy": name,
            "config": json.loads(cfg.model_dump_json()),
            "data_sha256": data_sha256,
            **extra,
        }
    ]


def run_all(
    cfg: Config,
    frame: pd.DataFrame,
    tables: dict[str, ForecastTable],
    start: date,
    end: date,
    split: str,
    data_sha256: str,
    plan_head: str,
    out_dir: Path,
    ledger_dir: Path,
    signer: Signer | None,
    costs_factor: float = 1.0,
    seed: int = 0,
    write: bool = True,
) -> list[Outcome]:
    """Toutes les stratégies, du premier jour prédit à `end`. `write` : registres + CSV."""
    sim = cfg.simulation
    costs = CostModel(sim.fee_rate, sim.slippage).scaled(costs_factor)
    first = min(t.dates[0] for t in tables.values())
    begin = max(start, first)
    mk = market(frame, begin, end)
    extra = {"period": split, "plan_head": plan_head, "costs_factor": costs_factor}
    outcomes: list[Outcome] = []

    def finish(result: BacktestResult, ledger: Chain | None) -> Run:
        if write and ledger is not None:
            return ar._finish(result, ledger, ledger_dir, out_dir, cfg)
        from gold_agents.blockchain.chain import Verification

        metrics = summarize(result, sim.risk_free_rate)
        return Run(result, metrics, Path(), Verification(True, 0))

    def chain(name: str, more: dict[str, Any] | None = None) -> Chain | None:
        if not write:
            return None
        return Chain.start(_genesis(cfg, data_sha256, name, {**extra, **(more or {})}), signer)

    strategies: list[Strategy] = [BuyAndHold(), SmaCrossover(20, 50)]
    for strategy in strategies:
        ledger = chain(strategy.name)
        result = run_backtest(
            frame, strategy, costs, sim.initial_capital, begin, end, cfg.risk.max_position, ledger
        )
        outcomes.append(Outcome(finish(result, ledger), None))

    for system in ar.build_systems(cfg, tables, costs):
        ledger = chain(system.name, {"agents": system.describe()})
        agent_run = run_agents(frame, system, sim.initial_capital, begin, end, ledger)
        run = finish(agent_run.result, ledger)
        if write:
            agent_run.decisions.to_csv(out_dir / f"{system.name}_decisions.csv", index=False)
        run.metrics["rules"] = ar._rule_counts(agent_run.result, agent_run.decisions)
        outcomes.append(Outcome(run, agent_run.decisions))

    main = next((o for o in outcomes if o.run.result.strategy == MAIN), outcomes[-1])
    n_trades = int(main.run.metrics["n_trades"])
    rng = np.random.default_rng(seed)
    want_random = random_positions(len(mk.closes), n_trades, rng)
    want_oracle = oracle_positions(mk.opens)
    refs = [
        (RANDOM, want_random, "hasard", {"n_trades": n_trades, "seed": seed}),
        (ORACLE, want_oracle, "oracle (connaît le futur, illustration)", {}),
    ]
    for name, want, reason, more in refs:
        positions = PositionStrategy(name, dict(zip(mk.dates, want.tolist(), strict=True)), reason)
        ledger = chain(name, more)
        result = run_backtest(
            frame, positions, costs, sim.initial_capital, begin, end, cfg.risk.max_position, ledger
        )
        outcomes.append(Outcome(finish(result, ledger), None))
    return outcomes


def costs_sensitivity(
    cfg: Config,
    frame: pd.DataFrame,
    tables: dict[str, ForecastTable],
    start: date,
    end: date,
    split: str,
    factors: list[float],
    seed: int,
) -> pd.DataFrame:
    rows = []
    for factor in factors:
        outcomes = run_all(
            cfg, frame, tables, start, end, split, "", "", Path(), Path(), None, factor, seed, False
        )
        for o in outcomes:
            m = o.run.metrics
            rows.append(
                {
                    "costs_factor": factor,
                    "strategy": m["strategy"],
                    "final_value": m["final_value"],
                    "net_profit": m["net_profit"],
                    "total_return": m["total_return"],
                    "sharpe": m["sharpe"],
                    "max_drawdown": m["max_drawdown"],
                    "n_trades": m["n_trades"],
                    "total_costs": m["total_costs"],
                }
            )
    return pd.DataFrame(rows)


def random_draws(
    cfg: Config, outcomes: list[Outcome], frame: pd.DataFrame, runs: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pour chaque système d'agents : `runs` stratégies au hasard avec autant de trades."""
    sim = cfg.simulation
    costs = CostModel(sim.fee_rate, sim.slippage)
    eq = outcomes[0].run.result.equity
    mk = market(frame, eq["date"].iloc[0], eq["date"].iloc[-1])
    rng = np.random.default_rng(seed + 1)
    rows, draws = [], []
    for o in outcomes:
        if o.decisions is None:
            continue
        m = o.run.metrics
        sharpes, returns = [], []
        for run in range(runs):
            want = random_positions(len(mk.closes), int(m["n_trades"]), rng)
            res = simulate_positions(
                mk.opens, mk.closes, want, costs, sim.initial_capital, cfg.risk.max_position
            ).metrics(sim.risk_free_rate)
            sharpes.append(res["sharpe"])
            returns.append(res["total_return"])
            draws.append({"strategy": m["strategy"], "run": run, **res})
        s = np.asarray(sharpes)
        rows.append(
            {
                "strategy": m["strategy"],
                "n_trades": int(m["n_trades"]),
                "sharpe": m["sharpe"],
                "random_sharpe_median": float(np.median(s)),
                "random_sharpe_p05": float(np.quantile(s, 0.05)),
                "random_sharpe_p95": float(np.quantile(s, 0.95)),
                "random_return_median": float(np.median(returns)),
                "beats_random": float(np.mean(m["sharpe"] > s)),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(draws)


def forecast_table(forecasts: pd.DataFrame) -> pd.DataFrame:
    """MAE, RMSE, bonne direction, gain sur B0 et Diebold-Mariano, par modèle."""
    rows = []
    first = forecasts[forecasts["model"] == forecasts["model"].iloc[0]]
    y_all = first["target_return"].to_numpy("float64")
    ok = ~np.isnan(y_all)
    y = y_all[ok]
    for model, g in forecasts.groupby("model", sort=False):
        pred = g["r_hat"].to_numpy("float64")[ok]
        rows.append({"model": str(model), **forecast_metrics(y, pred)})
    zero = forecast_metrics(y, np.zeros_like(y))
    rows.append({"model": "b0_zero", **zero})
    up = forecast_metrics(y, np.full_like(y, 1e-9))
    rows.append({"model": "always_up", **up})
    out = pd.DataFrame(rows)
    out["n_days"] = len(y)
    out["share_up"] = float(np.mean(y > 0))
    return out


def yearly_returns(outcomes: list[Outcome]) -> pd.DataFrame:
    """Rendement de chaque stratégie année civile par année civile."""
    out = {}
    for o in outcomes:
        eq = o.run.result.equity
        index = pd.DatetimeIndex(pd.to_datetime(eq["date"]))
        values = pd.Series(eq["value"].to_numpy(), index=index)
        first = float(values.iloc[0])
        year_end = values.groupby(index.year).last()
        start_values = year_end.shift(1).fillna(first)
        out[o.run.result.strategy] = year_end / start_values - 1
    return pd.DataFrame(out).T


def alerts(outcomes: list[Outcome], fc: pd.DataFrame, direction_alert: float) -> list[str]:
    """Résultats « trop beaux » : presque toujours une fuite du futur."""
    out = []
    for _, row in fc.iterrows():
        if row["model"] in ("b0_zero", "always_up"):  # références sans modèle
            continue
        d = row["direction"]
        if d == d and d > direction_alert:
            out.append(
                f"{MODEL_LABELS.get(row['model'], row['model'])} : {d:.1%} de bonnes directions "
                f"(> {direction_alert:.0%}), fuite du futur probable"
            )
    oracle = next((o for o in outcomes if o.run.result.strategy == ORACLE), None)
    for o in outcomes:
        m = o.run.metrics
        if m["strategy"] == ORACLE:
            continue
        if m["sharpe"] > SHARPE_ALERT:
            label = report.LABELS.get(m["strategy"], m["strategy"])
            out.append(f"{label} : Sharpe {m['sharpe']:.2f} (> {SHARPE_ALERT:g}), trop beau")
        if oracle is not None and m["final_value"] > oracle.run.metrics["final_value"] + 1e-6:
            label = report.LABELS.get(m["strategy"], m["strategy"])
            out.append(f"{label} bat l'oracle : impossible sans voir le futur")
    return out


# --- Tableau final -------------------------------------------------------------------


def final_table(outcomes: list[Outcome], refs: pd.DataFrame, costs: pd.DataFrame) -> pd.DataFrame:
    """Le tableau comparatif du rapport : une ligne par stratégie."""
    bh = outcomes[0].run.metrics
    beats = dict(zip(refs["strategy"], refs["beats_random"], strict=True)) if len(refs) else {}
    x2 = costs[costs["costs_factor"] == 2].set_index("strategy")["total_return"]
    rows = {}
    for o in outcomes:
        m = o.run.metrics
        name = m["strategy"]
        rows[report.LABELS.get(name, name)] = {
            "Valeur finale": _money(m["final_value"]),
            "Profit net": _money(m["net_profit"]),
            "Rendement": report._pct(m["total_return"]),
            "Par an": report._pct(m["cagr"]),
            "vs B&H": "-"
            if name == "buy_and_hold"
            else f"{(m['total_return'] - bh['total_return']) * 100:+.1f} pts",
            "Sharpe": f"{m['sharpe']:.2f}",
            "Sortino": f"{m['sortino']:.2f}",
            "Perte max": report._pct(m["max_drawdown"]),
            "Trades": str(m["n_trades"]),
            "Réussite": "-" if m["win_rate"] is None else f"{m['win_rate']:.0%}",
            "Frais": _money(m["total_costs"]),
            "Investi": f"{m['exposure']:.0%}",
            "Frais x2": report._pct(float(x2[name])) if name in x2 else "-",
            "Bat le hasard": f"{beats[name]:.0%}" if name in beats else "-",
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def costs_text(costs: pd.DataFrame) -> pd.DataFrame:
    table = costs.pivot(index="strategy", columns="costs_factor", values="total_return")
    order = list(dict.fromkeys(costs["strategy"]))
    table = table.loc[order]
    table.index = [report.LABELS.get(s, s) for s in table.index]
    table.columns = [f"frais x{c:g}" for c in table.columns]
    return table.map(lambda v: f"{v:+.1%}")


def forecast_text(fc: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "MAE (%)": (fc["mae"] * 100).map(lambda v: f"{v:.3f}"),
            "RMSE (%)": (fc["rmse"] * 100).map(lambda v: f"{v:.3f}"),
            "Bonne direction": fc["direction"].map(lambda v: "-" if v != v else f"{v:.1%}"),
            "Gain vs B0": fc["r2_vs_zero"].map(lambda v: f"{v:+.2%}"),
            "DM p-valeur": fc["dm_p_value"].map(lambda v: f"{v:.2f}"),
        }
    )
    out.index = [MODEL_LABELS.get(m, m) for m in fc["model"]]
    return out


def yearly_text(yearly: pd.DataFrame) -> pd.DataFrame:
    out = yearly.map(lambda v: f"{v:+.1%}")
    out.index = [report.LABELS.get(s, s) for s in out.index]
    return out


def markdown(table: pd.DataFrame) -> str:
    head = "| | " + " | ".join(map(str, table.columns)) + " |"
    sep = "|---|" + "---|" * len(table.columns)
    body = [f"| **{i}** | " + " | ".join(map(str, r)) + " |" for i, r in table.iterrows()]
    return "\n".join([head, sep, *body]) + "\n"


# --- Fichiers et registre des résultats ----------------------------------------------


def clean_json(data: Any) -> Any:
    """NaN et infinis -> None (le JSON canonique du registre les refuse)."""
    if isinstance(data, dict):
        return {str(k): clean_json(v) for k, v in data.items()}
    if isinstance(data, list | tuple):
        return [clean_json(v) for v in data]
    if isinstance(data, np.generic):
        data = data.item()
    if isinstance(data, float) and not math.isfinite(data):
        return None
    if isinstance(data, date):
        return data.isoformat()
    return data


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return clean_json(frame.to_dict(orient="records"))  # type: ignore[no-any-return]


def file_digests(paths: list[Path]) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


# --- Figures -------------------------------------------------------------------------


def final_figure(
    outcomes: list[Outcome], table: pd.DataFrame, cfg: Config, title: str, path: Path
) -> Path:
    """Valeur, perte max et tableau comparatif (l'oracle est seulement dans le tableau)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 16), dpi=120)
    grid = fig.add_gridspec(4, 1, height_ratios=[3.2, 2.0, 1.6, 3.0], hspace=0.38)
    ax_value, ax_price, ax_dd, ax_table = (fig.add_subplot(grid[i]) for i in range(4))
    for ax in (ax_value, ax_price, ax_dd):
        _style(ax)
    eq0 = outcomes[0].run.result.equity
    dates = pd.to_datetime(eq0["date"])
    sim = cfg.simulation
    fig.suptitle(
        f"{title}, du {dates.iloc[0]:%d/%m/%Y} au {dates.iloc[-1]:%d/%m/%Y}\n"
        f"Capital {_money(sim.initial_capital)}, frais {sim.fee_rate:.2%}, slippage "
        f"{sim.slippage:.2%}, réglages figés en phase 6 (θ_buy {cfg.strategy.theta_buy:g}, "
        f"k {cfg.strategy.min_confidence}, stop-loss {cfg.risk.stop_loss:.0%}, "
        f"coupe-circuit {cfg.risk.max_drawdown:.0%})",
        fontsize=13,
        color=INK,
    )
    ax_value.axhline(sim.initial_capital, color=MUTED, linewidth=1, linestyle="--")
    for o in outcomes:
        name = o.run.result.strategy
        if name == ORACLE:
            continue
        eq = o.run.result.equity
        strong = name in ("buy_and_hold", MAIN)
        ax_value.plot(
            pd.to_datetime(eq["date"]),
            eq["value"],
            color=report.COLORS.get(name, MUTED),
            linewidth=2.4 if strong else 1.2,
            linestyle=":" if name == RANDOM else "-",
            label=f"{report.LABELS.get(name, name)} : {_money(eq['value'].iloc[-1])}",
            zorder=3 if strong else 2,
        )
    ax_value.set_title(
        "Valeur du portefeuille (l'oracle, hors échelle, est dans le tableau)",
        loc="left",
        fontsize=11,
        color=INK,
    )
    ax_value.set_ylabel("USD", color=MUTED)
    ax_value.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=2)

    ax_price.plot(dates, eq0["close"], color=MUTED, linewidth=1.1, label="Clôture XAU/USD")
    main = next((o for o in outcomes if o.run.result.strategy == MAIN), None)
    if main is not None:
        eq = main.run.result.equity
        ax_price.fill_between(
            pd.to_datetime(eq["date"]),
            eq0["close"].min(),
            eq0["close"].max(),
            where=(eq["qty"] > 0).to_numpy(),
            color="#1baf7a",
            alpha=0.12,
            step="post",
            label="investi (Agents + LSTM/GRU, confiance)",
        )
        for t in main.run.result.trades:
            ax_price.scatter(
                pd.to_datetime([t.exec_date]),
                [t.market_open],
                marker="^" if t.side == "BUY" else "v",
                s=40,
                color="#1baf7a" if t.side == "BUY" else "white",
                edgecolor="#1baf7a" if t.side == "BUY" else RED,
                zorder=3,
            )
    ax_price.set_title(
        "Prix de l'or et ordres du système retenu", loc="left", fontsize=11, color=INK
    )
    ax_price.set_ylabel("USD / once", color=MUTED)
    ax_price.legend(frameon=False, fontsize=8.5, loc="upper left")

    for o in outcomes:
        name = o.run.result.strategy
        if name in ("buy_and_hold", MAIN, "agents_lightgbm", "sma_20_50"):
            eq = o.run.result.equity
            ax_dd.plot(
                pd.to_datetime(eq["date"]),
                drawdown(eq["value"]) * 100,
                color=report.COLORS.get(name, MUTED),
                linewidth=1.3,
                label=report.LABELS.get(name, name),
            )
    ax_dd.axhline(-cfg.risk.max_drawdown * 100, color=RED, linewidth=0.8, linestyle=":")
    ax_dd.set_title(
        "Baisse depuis le dernier plus haut (%, pointillés : coupe-circuit)",
        loc="left",
        fontsize=11,
        color=INK,
    )
    ax_dd.legend(frameon=False, fontsize=8.5, loc="lower left", ncol=4)

    ax_table.axis("off")
    shown = table[["Rendement", "vs B&H", "Sharpe", "Perte max", "Trades", "Frais", "Frais x2"]]
    cells = ax_table.table(
        cellText=shown.to_numpy().tolist(),
        rowLabels=list(shown.index),
        colLabels=list(shown.columns),
        cellLoc="center",
        rowLoc="left",
        loc="upper center",
    )
    cells.auto_set_font_size(False)
    cells.set_fontsize(9)
    cells.scale(1, 1.45)
    names = [o.run.result.strategy for o in outcomes]
    for (row, col), cell in cells.get_celld().items():
        cell.set_edgecolor(GRID)
        if row == 0:
            cell.get_text().set_color(MUTED)
        if col == -1 and row > 0:
            cell.get_text().set_fontweight("bold")
            cell.get_text().set_color(report.COLORS.get(names[row - 1], INK))
    ax_table.set_title(
        "Tableau comparatif (frais x2 : rendement avec frais et slippage doublés)",
        loc="left",
        fontsize=11,
        color=INK,
    )
    fig.subplots_adjust(top=0.93)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def robustness_figure(
    costs: pd.DataFrame,
    refs: pd.DataFrame,
    draws: pd.DataFrame,
    fc: pd.DataFrame,
    yearly: pd.DataFrame,
    direction_alert: float,
    title: str,
    path: Path,
) -> Path:
    """Frais x0/1/2/5, hasard, précision des prévisions et rendements par année."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=120)
    (ax_costs, ax_rand), (ax_fc, ax_year) = axes
    for ax in axes.flat:
        _style(ax)
    fig.suptitle(f"{title} : robustesse", fontsize=13, color=INK)

    for name, g in costs.groupby("strategy", sort=False):
        if name in (ORACLE, RANDOM):
            continue
        strong = name in ("buy_and_hold", MAIN)
        ax_costs.plot(
            g["costs_factor"],
            g["total_return"] * 100,
            marker="o",
            color=report.COLORS.get(str(name), MUTED),
            linewidth=2.2 if strong else 1.1,
            label=report.LABELS.get(str(name), str(name)),
        )
    ax_costs.set_xticks(sorted(costs["costs_factor"].unique()))
    ax_costs.set_xlabel("multiplicateur des frais et du slippage", color=MUTED)
    ax_costs.set_ylabel("rendement total (%)", color=MUTED)
    ax_costs.set_title("Sensibilité aux frais", loc="left", fontsize=11, color=INK)
    ax_costs.legend(frameon=False, fontsize=8)

    main = draws[draws["strategy"] == MAIN] if len(draws) else draws
    if len(main):
        ax_rand.hist(main["sharpe"], bins=30, color="#c9c7c1")
        row = refs[refs["strategy"] == MAIN].iloc[0]
        ax_rand.axvline(row["sharpe"], color="#1baf7a", linewidth=2.2)
        ax_rand.set_title(
            f"Agents + LSTM/GRU, confiance (vert) contre {len(main)} stratégies au hasard "
            f"\navec {int(row['n_trades'])} trades : meilleur que "
            f"{row['beats_random']:.0%} d'entre elles",
            loc="left",
            fontsize=10.5,
            color=INK,
        )
        ax_rand.set_xlabel("Sharpe", color=MUTED)

    models = fc[~fc["model"].isin(["b0_zero"])]
    labels = [MODEL_LABELS.get(m, m) for m in models["model"]]
    values = models["direction"].to_numpy() * 100
    bars = ax_fc.barh(labels, values, color="#7b5cd6")
    for bar, v in zip(bars, values, strict=True):
        ax_fc.text(v + 0.3, bar.get_y() + bar.get_height() / 2, f"{v:.1f} %", va="center")
    ax_fc.axvline(50, color=MUTED, linewidth=0.8, linestyle="--")
    ax_fc.axvline(direction_alert * 100, color=RED, linewidth=0.8, linestyle=":")
    ax_fc.set_xlim(40, max(direction_alert * 100 + 5, float(np.nanmax(values)) + 5))
    ax_fc.set_title(
        "Bonne direction des prévisions (pointillés rouges : alerte fuite)",
        loc="left",
        fontsize=11,
        color=INK,
    )

    shown = yearly.loc[[s for s in yearly.index if s != ORACLE]]
    years = list(shown.columns)
    width = 0.8 / max(len(shown), 1)
    for i, (label, row) in enumerate(shown.iterrows()):
        name = str(label)
        ax_year.bar(
            np.arange(len(years)) + i * width,
            row.to_numpy() * 100,
            width,
            color=report.COLORS.get(str(name), MUTED),
            label=report.LABELS.get(str(name), str(name)),
        )
    ax_year.set_xticks(np.arange(len(years)) + 0.4 - width / 2, [str(y) for y in years])
    ax_year.axhline(0, color=MUTED, linewidth=0.8)
    ax_year.set_title("Rendement par année civile (%)", loc="left", fontsize=11, color=INK)
    ax_year.legend(frameon=False, fontsize=7, ncol=2)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
