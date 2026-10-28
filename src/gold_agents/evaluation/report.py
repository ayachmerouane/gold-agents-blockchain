"""Backtests de référence (phase 3) : exécution, registre, fichiers de résultats et schéma.

`run_reference_backtests` fait passer chaque stratégie de référence (Buy & Hold, SMA 20/50)
par le même moteur sur la période de test, écrit un registre par stratégie, puis
`backtest_figure` résume tout dans une seule image.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from gold_agents.blockchain.chain import Chain, Verification, tamper_demo
from gold_agents.blockchain.storage import read_jsonl, write_jsonl
from gold_agents.config import Config
from gold_agents.evaluation.financial_metrics import drawdown, summarize
from gold_agents.simulation.costs import CostModel
from gold_agents.simulation.engine import BacktestResult, run_backtest
from gold_agents.simulation.strategies import BuyAndHold, SmaCrossover, Strategy

if TYPE_CHECKING:
    from matplotlib.axes import Axes

# Mêmes couleurs que les figures de l'EDA.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, RED = "#0b0b0b", "#52514e", "#e4e3df", "#c8333a"
COLORS = {"buy_and_hold": BLUE, "sma_20_50": ORANGE}
LABELS = {"buy_and_hold": "Buy & Hold", "sma_20_50": "SMA 20/50"}


@dataclass(frozen=True)
class Run:
    result: BacktestResult
    metrics: dict[str, Any]
    ledger_path: Path
    verification: Verification


def evaluation_period(frame: pd.DataFrame) -> tuple[date, date]:
    """Premier et dernier jour d'exemple du test (les mêmes que pour les modèles)."""
    test = frame[frame["sample"] & (frame["split"] == "test")]
    if test.empty:
        raise ValueError("aucun exemple de test dans le fichier de features")
    return pd.Timestamp(test["date"].iloc[0]).date(), pd.Timestamp(test["date"].iloc[-1]).date()


def genesis_events(cfg: Config, strategy: Strategy, data_sha256: str) -> list[dict[str, Any]]:
    """Bloc genèse : config complète et empreinte des données, fixées avant de simuler."""
    return [
        {
            "type": "GENESIS",
            "strategy": strategy.name,
            "config": json.loads(cfg.model_dump_json()),
            "data_sha256": data_sha256,
        }
    ]


def run_reference_backtests(
    cfg: Config,
    frame: pd.DataFrame,
    data_sha256: str,
    out_dir: Path,
    ledger_dir: Path,
    costs_factor: float = 1.0,
) -> list[Run]:
    start, end = evaluation_period(frame)
    sim = cfg.simulation
    costs = CostModel(sim.fee_rate, sim.slippage).scaled(costs_factor)
    out_dir.mkdir(parents=True, exist_ok=True)
    strategies: list[Strategy] = [BuyAndHold(), SmaCrossover(20, 50)]
    runs = []
    for strategy in strategies:
        ledger = Chain.start(genesis_events(cfg, strategy, data_sha256))
        result = run_backtest(
            frame,
            strategy,
            costs,
            sim.initial_capital,
            start,
            end,
            max_position=cfg.risk.max_position,
            ledger=ledger,
        )
        path = write_jsonl(ledger, ledger_dir / f"{strategy.name}.jsonl")
        # On relit le fichier écrit : c'est lui qu'on vérifie, pas l'objet en mémoire.
        verification = read_jsonl(path).verify()
        metrics = summarize(result, sim.risk_free_rate)
        metrics["ledger"] = {
            "path": str(path),
            "valid": verification.valid,
            "blocks": verification.n_blocks,
            "head_hash": ledger.head.hash,
        }
        result.equity.to_csv(out_dir / f"{strategy.name}_equity.csv", index=False)
        result.trades_frame().to_csv(out_dir / f"{strategy.name}_trades.csv", index=False)
        runs.append(Run(result, metrics, path, verification))
    (out_dir / "metrics.json").write_text(
        json.dumps([r.metrics for r in runs], indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return runs


# --- Texte ------------------------------------------------------------------


def _money(x: float) -> str:
    return f"{x:,.0f} $".replace(",", " ")


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{x:+.1%}"


def metrics_table(runs: list[Run]) -> pd.DataFrame:
    rows = {}
    for run in runs:
        m = run.metrics
        rows[LABELS.get(m["strategy"], m["strategy"])] = {
            "Valeur finale": _money(m["final_value"]),
            "Gain net": _money(m["net_profit"]),
            "Rendement": _pct(m["total_return"]),
            "Par an": _pct(m["cagr"]),
            "Perte max": _pct(m["max_drawdown"]),
            "Sharpe": f"{m['sharpe']:.2f}",
            "Trades": str(m["n_trades"]),
            "Coûts": _money(m["total_costs"]),
            "Investi": f"{m['exposure']:.0%}",
            "Registre": f"{'valide' if m['ledger']['valid'] else 'INVALIDE'}, "
            f"{m['ledger']['blocks']} blocs",
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def monthly_values(result: BacktestResult) -> pd.Series:
    """Valeur du portefeuille à chaque fin de mois (pour un résumé compact)."""
    index = pd.DatetimeIndex(pd.to_datetime(result.equity["date"]))
    values = pd.Series(result.equity["value"].to_numpy(), index=index)
    return values.groupby(index.to_period("M")).last()


def tamper_lines(run: Run) -> list[str]:
    chain = read_jsonl(run.ledger_path)
    return [
        f"  {scenario:32s} -> {'NON DÉTECTÉ' if v.valid else 'détecté'} ({v.describe()})"
        for scenario, v in tamper_demo(chain)
    ]


# --- Figure -----------------------------------------------------------------


def _style(ax: Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def backtest_figure(runs: list[Run], cfg: Config, path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 13), dpi=120)
    grid = fig.add_gridspec(4, 1, height_ratios=[3, 3, 1.6, 2.2], hspace=0.35)
    ax_price, ax_value, ax_dd, ax_table = (fig.add_subplot(grid[i]) for i in range(4))
    for ax in (ax_price, ax_value, ax_dd):
        _style(ax)

    first = runs[0].result.equity
    dates = pd.to_datetime(first["date"])
    sim = cfg.simulation
    fig.suptitle(
        f"Backtest sur la période de test, du {dates.iloc[0]:%d/%m/%Y} au {dates.iloc[-1]:%d/%m/%Y}"
        f"\nCapital {_money(sim.initial_capital)}, frais {sim.fee_rate:.2%}, "
        f"slippage {sim.slippage:.2%}, exécution à l'ouverture du lendemain",
        fontsize=13,
        color=INK,
    )

    # 1. Prix de l'or et points d'achat / de vente.
    ax_price.plot(dates, first["close"], color=MUTED, linewidth=1.1, label="Clôture XAU/USD")
    for run in runs:
        trades = run.result.trades
        name = run.result.strategy
        color = COLORS.get(name, AQUA)
        for side, marker in (("BUY", "^"), ("SELL", "v")):
            pts = [t for t in trades if t.side == side]
            if pts:
                ax_price.scatter(
                    pd.to_datetime([t.exec_date for t in pts]),
                    [t.market_open for t in pts],
                    marker=marker,
                    s=70 if side == "BUY" else 60,
                    color=color if side == "BUY" else "white",
                    edgecolor=color,
                    linewidth=1.5,
                    zorder=3,
                    label=f"{LABELS.get(name, name)} : {'achat' if side == 'BUY' else 'vente'}",
                )
    ax_price.set_title("Prix de l'or et ordres exécutés", loc="left", fontsize=11, color=INK)
    ax_price.set_ylabel("USD / once", color=MUTED)
    ax_price.legend(frameon=False, fontsize=8.5, loc="upper left")

    # 2. Valeur du portefeuille.
    ax_value.axhline(sim.initial_capital, color=MUTED, linewidth=1, linestyle="--")
    for run in runs:
        eq = run.result.equity
        name = run.result.strategy
        color = COLORS.get(name, AQUA)
        x = pd.to_datetime(eq["date"])
        ax_value.plot(x, eq["value"], color=color, linewidth=2, label=LABELS.get(name, name))
        ax_value.annotate(
            _money(eq["value"].iloc[-1]),
            (x.iloc[-1], eq["value"].iloc[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            color=color,
            fontsize=9,
            va="center",
        )
    ax_value.set_title(
        "Valeur du portefeuille (pointillés : capital de départ)",
        loc="left",
        fontsize=11,
        color=INK,
    )
    ax_value.set_ylabel("USD", color=MUTED)
    ax_value.legend(frameon=False, fontsize=9, loc="upper left")

    # 3. Baisse depuis le dernier plus haut.
    for run in runs:
        eq = run.result.equity
        name = run.result.strategy
        dd = drawdown(eq["value"]) * 100
        ax_dd.plot(pd.to_datetime(eq["date"]), dd, color=COLORS.get(name, AQUA), linewidth=1.3)
    ax_dd.set_title(
        "Baisse depuis le dernier plus haut (drawdown, %)", loc="left", fontsize=11, color=INK
    )
    ax_dd.set_ylabel("%", color=MUTED)

    # 4. Tableau des métriques.
    ax_table.axis("off")
    table = metrics_table(runs)
    cells = ax_table.table(
        cellText=table.to_numpy().tolist(),
        rowLabels=list(table.index),
        colLabels=list(table.columns),
        cellLoc="center",
        rowLoc="left",
        loc="upper center",
    )
    cells.auto_set_font_size(False)
    cells.set_fontsize(9.5)
    cells.scale(1, 1.5)
    for (r, c), cell in cells.get_celld().items():
        cell.set_edgecolor(GRID)
        if c == -1 and r > 0:
            cell.get_text().set_fontweight("bold")
        if r == 0:
            cell.get_text().set_color(MUTED)
        if c >= 0 and table.columns[c] == "Registre" and r > 0:
            ok = runs[r - 1].verification.valid
            cell.get_text().set_color(AQUA if ok else RED)
    ax_table.set_title(
        f"Résultats (Sharpe avec un taux sans risque de {sim.risk_free_rate:.1%} ; "
        "registre relu et vérifié bloc par bloc)",
        loc="left",
        fontsize=11,
        color=INK,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
