"""Phase 5 : les agents tradent dans le moteur de la phase 3 avec les modèles de la phase 4.

Sur une période (la validation par défaut), on fait passer par le même moteur, avec les
mêmes coûts :

- les références : Buy & Hold et SMA 20/50 ;
- le système multi-agents avec chaque modèle de prévision : LSTM, GRU, ensemble LSTM + GRU,
  ensemble avec le filtre de confiance MC Dropout, et LightGBM, le concurrent sans réseau.

Chaque simulation écrit son registre, relu et vérifié. `agents_figure` résume tout.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gold_agents.agents.execution import ExecutionAgent
from gold_agents.agents.forecast import ForecastAgent, ForecastTable, lightgbm_table, network_table
from gold_agents.agents.orchestrator import AgentSystem, run_agents
from gold_agents.agents.risk import RiskAgent
from gold_agents.agents.strategy import StrategyAgent
from gold_agents.blockchain.chain import Chain
from gold_agents.blockchain.storage import read_jsonl, write_jsonl
from gold_agents.config import Config
from gold_agents.evaluation import report
from gold_agents.evaluation.financial_metrics import drawdown, summarize
from gold_agents.evaluation.report import GRID, INK, MUTED, RED, Run, _money, _style
from gold_agents.models.dataset import ForecastData
from gold_agents.simulation.costs import CostModel
from gold_agents.simulation.engine import BacktestResult, run_backtest
from gold_agents.simulation.strategies import BuyAndHold, SmaCrossover, Strategy

MAIN = "agents_ensemble_mc"  # le système mis en avant sur le schéma
AGENT_LABELS = {
    "agents_lstm": "Agents + LSTM",
    "agents_gru": "Agents + GRU",
    "agents_ensemble": "Agents + LSTM/GRU",
    MAIN: "Agents + LSTM/GRU, confiance",
    "agents_lightgbm": "Agents + LightGBM",
}
AGENT_COLORS = {
    "agents_lstm": "#8fb7ea",
    "agents_gru": "#f2a684",
    "agents_ensemble": "#7b5cd6",
    MAIN: "#1baf7a",
    "agents_lightgbm": "#c9a227",
}
report.LABELS.update(AGENT_LABELS)
report.COLORS.update(AGENT_COLORS)
FORECAST_FILE = "summary_val.json"  # résultats de `gold-agents train` (phase 4)


@dataclass(frozen=True)
class AgentResult:
    run: Run
    decisions: pd.DataFrame | None  # None pour les références (pas d'agents)


def simulation_period(frame: pd.DataFrame, split: str) -> tuple[date, date]:
    """Premier et dernier jour d'exemple de `split` (les jours où les modèles prédisent)."""
    part = frame[frame["sample"] & (frame["split"] == split)]
    if part.empty:
        raise ValueError(f"aucun exemple {split} dans le fichier de features")
    return pd.Timestamp(part["date"].iloc[0]).date(), pd.Timestamp(part["date"].iloc[-1]).date()


# --- Prévisions --------------------------------------------------------------------


def forecast_tables(
    cfg: Config,
    data: ForecastData,
    split: str,
    models_dir: Path,
    forecast_dir: Path,
) -> dict[str, ForecastTable]:
    """Prévisions de chaque modèle sur `split`, avec les poids figés de la phase 4.

    Le MC Dropout (σ̂) n'est calculé que pour l'ensemble, le seul à s'en servir.
    LightGBM n'a pas de poids sauvegardés : on le réentraîne avec les réglages Optuna
    écrits par `train`, exactement comme en phase 4 (même train, même arrêt anticipé).
    """
    from gold_agents.models import baselines
    from gold_agents.models.predict import load_network

    part = data[split]
    dates = part.windows.dates
    nets = {kind: load_network(models_dir / kind) for kind in ("lstm", "gru")}
    for kind, net in nets.items():
        if net.meta.get("columns") != data.columns or net.meta.get("window") != data.window:
            raise ValueError(f"{kind} : features ou fenêtre différentes de l'entraînement")
    mc = cfg.models
    tables = {
        "lstm": network_table("lstm", [nets["lstm"]], part.x, dates, 0),
        "gru": network_table("gru", [nets["gru"]], part.x, dates, 0),
        "ensemble": network_table(
            "ensemble_lstm_gru", list(nets.values()), part.x, dates, mc.mc_samples, cfg.seed
        ),
    }
    summary = json.loads((forecast_dir / FORECAST_FILE).read_text(encoding="utf-8"))
    params = summary["settings"].get("b4")
    if params:
        train, val = data["train"], data["val"]
        models = [
            baselines.fit_lightgbm(
                params, train.tabular, train.y, val.tabular, val.y, seed, mc.threads
            )
            for seed in mc.seeds
        ]
        tables["lightgbm"] = lightgbm_table(models, part.tabular, dates)
    return tables


def diagnostics(tables: dict[str, ForecastTable], cfg: Config) -> pd.DataFrame:
    """Ordre de grandeur des prévisions, et part des jours où chaque règle se déclenche."""
    k, theta = cfg.strategy.min_confidence, cfg.strategy.theta_buy
    rows = {}
    for key, t in tables.items():
        r = t.r_hat
        q05, q95 = np.quantile(r, [0.05, 0.95])
        row = {
            "r̂ médian (%)": f"{np.median(r) * 100:+.3f}",
            "r̂ 5-95 % (%)": f"{q05 * 100:+.3f} / {q95 * 100:+.3f}",
            "r̂ > θ_buy": f"{np.mean(r > theta):.0%}",
            "r̂ < -θ_sell": f"{np.mean(r < -cfg.strategy.theta_sell):.0%}",
        }
        if t.sigma_hat is not None:
            z = r / np.where(t.sigma_hat > 0, t.sigma_hat, np.nan)
            row["σ̂ médian (%)"] = f"{np.median(t.sigma_hat) * 100:.3f}"
            row["z médian"] = f"{np.nanmedian(z):+.2f}"
            if k is not None:
                row[f"z > {k:g}"] = f"{np.nanmean(z > k):.0%}"
        rows[key] = row
    return pd.DataFrame.from_dict(rows, orient="index").fillna("-")


# --- Simulations -------------------------------------------------------------------


def build_systems(
    cfg: Config, tables: dict[str, ForecastTable], costs: CostModel
) -> list[AgentSystem]:
    st, rk = cfg.strategy, cfg.risk
    plan = [
        ("agents_lstm", "lstm", None),
        ("agents_gru", "gru", None),
        ("agents_ensemble", "ensemble", None),
        (MAIN, "ensemble", st.min_confidence),
        ("agents_lightgbm", "lightgbm", None),
    ]
    return [
        AgentSystem(
            name,
            ForecastAgent(tables[key]),
            StrategyAgent(st.theta_buy, st.theta_sell, k),
            RiskAgent(
                rk.max_position, rk.stop_loss, rk.max_drawdown, rk.allow_short, rk.cooldown_days
            ),
            ExecutionAgent(costs),
        )
        for name, key, k in plan
        if key in tables
    ]


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


def _finish(
    result: BacktestResult, ledger: Chain, ledger_dir: Path, out_dir: Path, cfg: Config
) -> Run:
    path = write_jsonl(ledger, ledger_dir / f"{result.strategy}.jsonl")
    verification = read_jsonl(path).verify()  # on vérifie le fichier écrit
    metrics = summarize(result, cfg.simulation.risk_free_rate)
    metrics["ledger"] = {
        "path": str(path),
        "valid": verification.valid,
        "blocks": verification.n_blocks,
        "head_hash": ledger.head.hash,
    }
    result.equity.to_csv(out_dir / f"{result.strategy}_equity.csv", index=False)
    result.trades_frame().to_csv(out_dir / f"{result.strategy}_trades.csv", index=False)
    return Run(result, metrics, path, verification)


def run_agent_backtests(
    cfg: Config,
    frame: pd.DataFrame,
    tables: dict[str, ForecastTable],
    split: str,
    data_sha256: str,
    out_dir: Path,
    ledger_dir: Path,
    costs_factor: float = 1.0,
) -> list[AgentResult]:
    start, end = simulation_period(frame, split)
    sim = cfg.simulation
    costs = CostModel(sim.fee_rate, sim.slippage).scaled(costs_factor)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[AgentResult] = []

    references: list[Strategy] = [BuyAndHold(), SmaCrossover(20, 50)]
    for strategy in references:
        ledger = Chain.start(_genesis(cfg, data_sha256, strategy.name, {"period": split}))
        result = run_backtest(
            frame, strategy, costs, sim.initial_capital, start, end, cfg.risk.max_position, ledger
        )
        results.append(AgentResult(_finish(result, ledger, ledger_dir, out_dir, cfg), None))

    for system in build_systems(cfg, tables, costs):
        extra = {"period": split, "agents": system.describe()}
        ledger = Chain.start(_genesis(cfg, data_sha256, system.name, extra))
        agent_run = run_agents(frame, system, sim.initial_capital, start, end, ledger)
        run = _finish(agent_run.result, ledger, ledger_dir, out_dir, cfg)
        agent_run.decisions.to_csv(out_dir / f"{system.name}_decisions.csv", index=False)
        run.metrics["rules"] = _rule_counts(agent_run.result, agent_run.decisions)
        results.append(AgentResult(run, agent_run.decisions))

    forecasts = pd.DataFrame({"date": next(iter(tables.values())).dates})
    for key, t in tables.items():
        forecasts[f"{key}_r_hat"] = t.r_hat
        if t.sigma_hat is not None:
            forecasts[f"{key}_sigma_hat"] = t.sigma_hat
    forecasts.to_csv(out_dir / "forecasts.csv", index=False)
    (out_dir / "metrics.json").write_text(
        json.dumps([r.run.metrics for r in results], indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return results


def _rule_counts(result: BacktestResult, decisions: pd.DataFrame) -> dict[str, int]:
    """Combien de trades chaque règle a déclenchés (signal, stop-loss, drawdown max)."""
    executed = {t.decision_date for t in result.trades}
    done = decisions[decisions["date"].isin(executed)]
    counts = done["rule"].value_counts().to_dict()
    counts["jours d'achat bloqués (incertain)"] = int(
        decisions["reason"].str.contains("incertain", regex=False).sum()
    )
    return {str(k): int(v) for k, v in counts.items()}


# --- Texte -------------------------------------------------------------------------


def comparison_table(results: list[AgentResult]) -> pd.DataFrame:
    """Mesures métier, avec l'écart de rendement et de Sharpe par rapport à Buy & Hold."""
    runs = [r.run for r in results]
    table = report.metrics_table(runs).drop(columns=["Registre"])
    bh = runs[0].metrics
    table["vs B&H"] = [
        "-"
        if r.metrics["strategy"] == "buy_and_hold"
        else f"{(r.metrics['total_return'] - bh['total_return']) * 100:+.1f} pts"
        for r in runs
    ]
    table["Réussite"] = [
        "-" if r.metrics["win_rate"] is None else f"{r.metrics['win_rate']:.0%}" for r in runs
    ]
    table["Registre"] = [
        f"{'valide' if r.verification.valid else 'INVALIDE'}, {r.verification.n_blocks} blocs"
        for r in runs
    ]
    return table


# --- Figure ------------------------------------------------------------------------


def agents_figure(
    results: list[AgentResult],
    tables: dict[str, ForecastTable],
    cfg: Config,
    split: str,
    path: Path,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [r.run for r in results]
    by_name = {r.run.result.strategy: r for r in results}
    main = by_name.get(MAIN)
    fig = plt.figure(figsize=(13, 17), dpi=120)
    grid = fig.add_gridspec(5, 1, height_ratios=[3.2, 2.4, 2.4, 1.4, 2.8], hspace=0.42)
    ax_value, ax_fc, ax_price, ax_dd, ax_table = (fig.add_subplot(grid[i]) for i in range(5))
    for ax in (ax_value, ax_fc, ax_price, ax_dd):
        _style(ax)

    eq0 = runs[0].result.equity
    dates = pd.to_datetime(eq0["date"])
    sim, st = cfg.simulation, cfg.strategy
    name_split = {"val": "la validation", "test": "le test", "train": "le train"}[split]
    fig.suptitle(
        f"Phase 5 : agents de trading sur {name_split}, du {dates.iloc[0]:%d/%m/%Y} au "
        f"{dates.iloc[-1]:%d/%m/%Y}\nCapital {_money(sim.initial_capital)}, frais "
        f"{sim.fee_rate:.2%}, slippage {sim.slippage:.2%}, exécution à l'ouverture du lendemain, "
        f"stop-loss {cfg.risk.stop_loss:.0%}, drawdown max {cfg.risk.max_drawdown:.0%}",
        fontsize=13,
        color=INK,
    )

    # 1. Valeur du portefeuille de chaque stratégie.
    ax_value.axhline(sim.initial_capital, color=MUTED, linewidth=1, linestyle="--")
    for run in runs:
        eq = run.result.equity
        name = run.result.strategy
        x = pd.to_datetime(eq["date"])
        strong = name in ("buy_and_hold", MAIN)
        ax_value.plot(
            x,
            eq["value"],
            color=report.COLORS.get(name, MUTED),
            linewidth=2.4 if strong else 1.3,
            label=f"{report.LABELS.get(name, name)} : {_money(eq['value'].iloc[-1])}",
            zorder=3 if strong else 2,
        )
    ax_value.set_title(
        "Valeur du portefeuille (pointillés : capital de départ)",
        loc="left",
        fontsize=11,
        color=INK,
    )
    ax_value.set_ylabel("USD", color=MUTED)
    ax_value.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=2)

    # 2. Prévision de l'ensemble, son incertitude MC Dropout et les jours « confiants ».
    ens = tables.get("ensemble")
    if ens is not None:
        days = pd.DatetimeIndex(pd.to_datetime(ens.dates))
        r = ens.r_hat * 100
        ax_fc.axhline(0, color=MUTED, linewidth=0.8)
        if ens.sigma_hat is not None:
            s = ens.sigma_hat * 100
            ax_fc.fill_between(
                x, r - s, r + s, color="#1baf7a", alpha=0.18, label="r̂ ± σ̂ (MC Dropout)"
            )
            k = st.min_confidence or 0.0
            sure = ens.r_hat > k * ens.sigma_hat
            ax_fc.scatter(
                days[sure],
                r[sure],
                s=8,
                color="#1baf7a",
                zorder=3,
                label=f"r̂ > {k:g} σ̂ : achat autorisé ({sure.mean():.0%} des jours)",
            )
        ax_fc.plot(days, r, color="#7b5cd6", linewidth=1.1, label="r̂ ensemble LSTM + GRU")
        if "lightgbm" in tables:
            lg = tables["lightgbm"]
            ax_fc.plot(
                pd.to_datetime(lg.dates),
                lg.r_hat * 100,
                color="#c9a227",
                linewidth=0.8,
                alpha=0.8,
                label="r̂ LightGBM",
            )
        if st.theta_buy:
            ax_fc.axhline(st.theta_buy * 100, color=RED, linewidth=0.8, linestyle=":")
        ax_fc.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=2)
    ax_fc.set_title(
        "Agent de prévision : rendement prévu pour le lendemain (%)",
        loc="left",
        fontsize=11,
        color=INK,
    )
    ax_fc.set_ylabel("%", color=MUTED)

    # 3. Prix, périodes investies et ordres du système principal.
    ax_price.plot(dates, eq0["close"], color=MUTED, linewidth=1.1, label="Clôture XAU/USD")
    if main is not None:
        eq = main.run.result.equity
        invested = (eq["qty"] > 0).to_numpy()
        ax_price.fill_between(
            pd.to_datetime(eq["date"]),
            eq0["close"].min(),
            eq0["close"].max(),
            where=invested,
            color="#1baf7a",
            alpha=0.10,
            step="post",
            label="investi (agents, confiance)",
        )
        style = {"signal": "#1baf7a", "stop_loss": RED, "max_drawdown": INK}
        names = {"signal": "prévision", "stop_loss": "stop-loss", "max_drawdown": "coupe-circuit"}
        rules = (
            dict(zip(main.decisions["date"], main.decisions["rule"], strict=True))
            if main.decisions is not None
            else {}
        )
        for t in main.run.result.trades:
            rule = rules.get(t.decision_date, "signal")
            ax_price.scatter(
                pd.to_datetime([t.exec_date]),
                [t.market_open],
                marker="^" if t.side == "BUY" else "v",
                s=55,
                color=style.get(rule, "#1baf7a") if t.side == "BUY" else "white",
                edgecolor=style.get(rule, "#1baf7a"),
                linewidth=1.4,
                zorder=3,
            )
        for rule, color in style.items():
            ax_price.scatter(
                [], [], marker="v", color="white", edgecolor=color, label=f"vente : {names[rule]}"
            )
        ax_price.scatter([], [], marker="^", color="#1baf7a", label="achat")
    ax_price.set_title(
        "Prix de l'or et ordres du système « Agents + LSTM/GRU, confiance »",
        loc="left",
        fontsize=11,
        color=INK,
    )
    ax_price.set_ylabel("USD / once", color=MUTED)
    ax_price.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(0, -0.1), ncol=6)

    # 4. Drawdown.
    for run in runs:
        eq = run.result.equity
        name = run.result.strategy
        if name in ("buy_and_hold", MAIN, "agents_lightgbm"):
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
    ax_dd.legend(frameon=False, fontsize=8.5, loc="lower left", ncol=3)

    # 5. Tableau.
    ax_table.axis("off")
    table = comparison_table(results).drop(columns=["Gain net", "Coûts"])
    cells = ax_table.table(
        cellText=table.to_numpy().tolist(),
        rowLabels=list(table.index),
        colLabels=list(table.columns),
        cellLoc="center",
        rowLoc="left",
        loc="upper center",
    )
    cells.auto_set_font_size(False)
    cells.set_fontsize(9)
    cells.scale(1, 1.45)
    for (row, col), cell in cells.get_celld().items():
        cell.set_edgecolor(GRID)
        if row == 0:
            cell.get_text().set_color(MUTED)
        if col == -1 and row > 0:
            cell.get_text().set_fontweight("bold")
            name = runs[row - 1].result.strategy
            cell.get_text().set_color(report.COLORS.get(name, INK))
    ax_table.set_title(
        "Mesures métier (Sharpe avec un taux sans risque de "
        f"{sim.risk_free_rate:.1%} ; « vs B&H » : écart de rendement total)",
        loc="left",
        fontsize=11,
        color=INK,
    )

    fig.subplots_adjust(top=0.93)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
