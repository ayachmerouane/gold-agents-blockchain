"""Phase 6 : tableaux texte et schéma du réglage des agents."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from gold_agents.evaluation.report import BLUE, GRID, INK, MUTED, RED, _style
from gold_agents.evaluation.tuning import Combo

if TYPE_CHECKING:
    from matplotlib.axes import Axes

GREEN, PURPLE = "#1baf7a", "#7b5cd6"
MODEL_LABELS = {"lstm": "LSTM", "gru": "GRU", "ensemble": "LSTM/GRU", "lightgbm": "LightGBM"}


def _k(v: float | None) -> str:
    return "-" if v is None or pd.isna(v) else f"{v:g}"


def summary_text(summary: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    top = summary.head(n)
    return pd.DataFrame(
        {
            "modèle": top["model"].map(MODEL_LABELS),
            "θ_buy": top["theta_buy"].map(lambda v: f"{v:.2%}"),
            "θ_sell": top["theta_sell"].map(lambda v: f"{v:.2%}"),
            "k": top["min_confidence"].map(_k),
            "stop": top["stop_loss"].map(lambda v: f"{v:.0%}"),
            "coupe": top["max_drawdown"].map(lambda v: f"{v:.0%}"),
            "Sharpe moy.": top["sharpe"].map(lambda v: f"{v:.2f}"),
            "Sharpe voisins": top["plateau"].map(lambda v: f"{v:.2f}"),
            "score": top["score"].map(lambda v: f"{v:.2f}"),
            "Sharpe min": top["sharpe_min"].map(lambda v: f"{v:.2f}"),
            "rend. moy.": top["total_return"].map(lambda v: f"{v:+.1%}"),
            "perte max": top["worst_drawdown"].map(lambda v: f"{v:.1%}"),
            "trades": top["n_trades"].map(lambda v: f"{v:.0f}"),
            "DD ≤ B&H": [f"{a}/{b}" for a, b in zip(top["dd_ok"], top["n_folds"], strict=True)],
        }
    )


def walk_forward_text(wf: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "période": [
                f"{a:%Y-%m} -> {b:%Y-%m}" for a, b in zip(wf["start"], wf["end"], strict=True)
            ],
            "réglé sur": wf["réglage choisi sur"],
            "modèle": wf["model"].map(MODEL_LABELS),
            "θ_buy/θ_sell": [
                f"{a:.2%}/{b:.2%}" for a, b in zip(wf["theta_buy"], wf["theta_sell"], strict=True)
            ],
            "k": wf["min_confidence"].map(_k),
            "stop/coupe": [
                f"{a:.0%}/{b:.0%}" for a, b in zip(wf["stop_loss"], wf["max_drawdown"], strict=True)
            ],
            "Sharpe": wf["sharpe"].map(lambda v: f"{v:.2f}"),
            "B&H": wf["bh_sharpe"].map(lambda v: f"{v:.2f}"),
            "rend.": wf["total_return"].map(lambda v: f"{v:+.1%}"),
            "rend. B&H": wf["bh_total_return"].map(lambda v: f"{v:+.1%}"),
            "perte max": wf["worst_drawdown"].map(lambda v: f"{v:.1%}"),
            "B&H ": wf["bh_worst_drawdown"].map(lambda v: f"{v:.1%}"),
            "trades": wf["n_trades"].map(lambda v: f"{v:.0f}"),
        }
    ).set_index("période")


def costs_text(costs: pd.DataFrame) -> pd.DataFrame:
    g = costs.groupby("costs_factor")
    return pd.DataFrame(
        {
            "Sharpe agents (moy.)": g["sharpe"].mean().map(lambda v: f"{v:.2f}"),
            "Sharpe B&H (moy.)": g["bh_sharpe"].mean().map(lambda v: f"{v:.2f}"),
            "rend. agents (moy.)": g["total_return"].mean().map(lambda v: f"{v:+.1%}"),
            "rend. B&H (moy.)": g["bh_total_return"].mean().map(lambda v: f"{v:+.1%}"),
            "périodes où agents > B&H": [
                f"{int((p['sharpe'] > p['bh_sharpe']).sum())}/{len(p)}" for _, p in g
            ],
        }
    ).rename(index=lambda f: f"frais x {f:g}")


def references_text(refs: pd.DataFrame, bh: pd.DataFrame) -> pd.DataFrame:
    merged = refs.merge(bh[["fold", "sharpe"]].rename(columns={"sharpe": "bh"}), on="fold")
    merged.index = pd.Index([f"période {f}" for f in merged["fold"]])
    return pd.DataFrame(
        {
            "agents": merged["agents_sharpe"].map(lambda v: f"{v:.2f}"),
            "trades": merged["agents_trades"],
            "hasard (médiane)": merged["random_sharpe_median"].map(lambda v: f"{v:.2f}"),
            "hasard (95e centile)": merged["random_sharpe_p95"].map(lambda v: f"{v:.2f}"),
            "agents battent le hasard": merged["beats_random"].map(lambda v: f"{v:.0%}"),
            "Buy & Hold": merged["bh"].map(lambda v: f"{v:.2f}"),
            "oracle": merged["oracle_sharpe"].map(lambda v: f"{v:.1f}"),
        }
    )


def yaml_snippet(combo: Combo, cooldown_days: int | None) -> str:
    k = "null" if combo.min_confidence is None else f"{combo.min_confidence:g}"
    return (
        "strategy:\n"
        f"  theta_buy: {combo.theta_buy:g}\n"
        f"  theta_sell: {combo.theta_sell:g}\n"
        f"  min_confidence: {k}\n"
        f"  model: {combo.model}\n"
        "risk:\n"
        f"  stop_loss: {combo.stop_loss:g}\n"
        f"  max_drawdown: {combo.max_drawdown:g}\n"
        f"  cooldown_days: {'null' if cooldown_days is None else cooldown_days}\n"
    )


# --- Figure ------------------------------------------------------------------------


def _axis(values: pd.Series, name: str) -> tuple[pd.Series, list[str]]:
    """Étiquettes d'un axe de carte de chaleur, dans l'ordre (sans filtre en premier)."""
    order = sorted(values.dropna().unique())
    if name == "min_confidence":
        labels = values.map(lambda v: "aucun" if pd.isna(v) else f"{v:g}")
        return labels, (["aucun"] if values.isna().any() else []) + [f"{v:g}" for v in order]
    return values.map(lambda v: f"{v:.2%}"), [f"{v:.2%}" for v in order]


def _label(value: float | None, name: str) -> str:
    if name == "min_confidence":
        return "aucun" if value is None else f"{value:g}"
    return f"{value:.2%}"


def _heatmap(
    ax: Axes, table: pd.DataFrame, rows: str, cols: str, chosen: Combo, title: str
) -> None:
    from matplotlib.patches import Rectangle

    r_lab, r_order = _axis(table[rows], rows)
    c_lab, c_order = _axis(table[cols], cols)
    frame = pd.DataFrame({"r": r_lab, "c": c_lab, "sharpe": table["sharpe"]})
    piv = frame.pivot_table(index="r", columns="c", values="sharpe", aggfunc="mean")
    piv = piv.reindex(index=r_order[::-1], columns=c_order)
    values = piv.to_numpy(dtype="float64")
    vmax = float(np.nanmax(np.abs(values))) if np.isfinite(values).any() else 1.0
    ax.imshow(values, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(piv.columns)), list(piv.columns), fontsize=8.5)
    ax.set_yticks(range(len(piv.index)), list(piv.index), fontsize=8.5)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]):
                ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=8.5)
    want_r = _label(getattr(chosen, rows), rows)
    want_c = _label(getattr(chosen, cols), cols)
    if want_r in piv.index and want_c in piv.columns:
        i, j = list(piv.index).index(want_r), list(piv.columns).index(want_c)
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=INK, linewidth=2.2))
    names = {"theta_buy": "θ_buy", "theta_sell": "θ_sell", "min_confidence": "confiance k"}
    ax.set_xlabel(names[cols], color=MUTED)
    ax.set_ylabel(names[rows], color=MUTED)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK)
    for spine in ax.spines.values():
        spine.set_color(GRID)


def tuning_figure(
    table: pd.DataFrame,
    bh: pd.DataFrame,
    wf: pd.DataFrame,
    costs: pd.DataFrame,
    refs: pd.DataFrame,
    draws: pd.DataFrame,
    chosen: Combo,
    path: Path,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 15), dpi=120)
    grid = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.42, wspace=0.25)
    ax_h1, ax_h2 = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    ax_wf, ax_dd = fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
    ax_costs, ax_rand = fig.add_subplot(grid[2, 0]), fig.add_subplot(grid[2, 1])
    fig.suptitle(
        "Phase 6 : réglage des agents en walk-forward (2011-2021, test non lu)\n"
        f"Réglage figé : {chosen.describe()}",
        fontsize=12.5,
        color=INK,
    )

    # 1-2. Cartes de chaleur autour du réglage choisi (Sharpe moyen sur les périodes).
    same = table["model"] == chosen.model
    same &= (table["stop_loss"] == chosen.stop_loss) & (
        table["max_drawdown"] == chosen.max_drawdown
    )
    t1 = table[same & (table["theta_sell"] == chosen.theta_sell)]
    _heatmap(ax_h1, t1, "theta_buy", "min_confidence", chosen, "Sharpe moyen : θ_buy × k")
    k_same = (
        table["min_confidence"].isna()
        if chosen.min_confidence is None
        else table["min_confidence"] == chosen.min_confidence
    )
    _heatmap(
        ax_h2,
        table[same & k_same],
        "theta_buy",
        "theta_sell",
        chosen,
        "Sharpe moyen : θ_buy × θ_sell",
    )

    # 3. Walk-forward : Sharpe hors échantillon contre Buy & Hold.
    for ax in (ax_wf, ax_dd, ax_costs, ax_rand):
        _style(ax)
    x = np.arange(len(wf))
    labels = [f"{a:%Y}-{b:%Y}" for a, b in zip(wf["start"], wf["end"], strict=True)]
    ax_wf.bar(x - 0.2, wf["sharpe"], 0.4, color=GREEN, label="agents (réglés sur le passé)")
    ax_wf.bar(x + 0.2, wf["bh_sharpe"], 0.4, color=BLUE, label="Buy & Hold")
    ax_wf.axhline(0, color=MUTED, linewidth=0.8)
    ax_wf.set_xticks(x, labels)
    ax_wf.set_title("Walk-forward : Sharpe de chaque période jugée", loc="left", fontsize=10.5)
    ax_wf.legend(frameon=False, fontsize=8.5)

    ax_dd.bar(x - 0.2, wf["worst_drawdown"] * 100, 0.4, color=GREEN, label="agents")
    ax_dd.bar(x + 0.2, wf["bh_worst_drawdown"] * 100, 0.4, color=BLUE, label="Buy & Hold")
    ax_dd.set_xticks(x, labels)
    ax_dd.set_ylabel("%", color=MUTED)
    ax_dd.set_title("Walk-forward : perte max (%)", loc="left", fontsize=10.5)
    ax_dd.legend(frameon=False, fontsize=8.5, loc="lower left")

    # 5. Sensibilité aux frais.
    g = costs.groupby("costs_factor")
    ax_costs.plot(
        g["sharpe"].mean().index,
        g["sharpe"].mean(),
        "o-",
        color=GREEN,
        label="agents (réglage figé)",
    )
    ax_costs.plot(
        g["bh_sharpe"].mean().index, g["bh_sharpe"].mean(), "o-", color=BLUE, label="Buy & Hold"
    )
    ax_costs.set_xlabel("frais et slippage multipliés par", color=MUTED)
    ax_costs.set_ylabel("Sharpe moyen", color=MUTED)
    ax_costs.set_title("Sensibilité aux frais (moyenne des périodes)", loc="left", fontsize=10.5)
    ax_costs.legend(frameon=False, fontsize=8.5)

    # 6. Hasard et oracle.
    folds = sorted(refs["fold"])
    data = [draws.loc[draws["fold"] == f, "sharpe"].to_numpy() for f in folds]
    ax_rand.boxplot(
        data,
        positions=range(len(folds)),
        widths=0.5,
        showfliers=False,
        medianprops={"color": MUTED},
        boxprops={"color": MUTED},
        whiskerprops={"color": MUTED},
        capprops={"color": MUTED},
    )
    ax_rand.scatter(
        range(len(folds)),
        refs["agents_sharpe"],
        color=GREEN,
        zorder=3,
        s=40,
        label="agents (réglage figé)",
    )
    ax_rand.scatter(
        range(len(folds)),
        bh.set_index("fold").loc[folds, "sharpe"],
        color=BLUE,
        marker="D",
        zorder=3,
        s=30,
        label="Buy & Hold",
    )
    for i, f in enumerate(folds):
        o = refs.loc[refs["fold"] == f, "oracle_sharpe"].iloc[0]
        ax_rand.annotate(
            f"oracle {o:.1f}",
            (i, ax_rand.get_ylim()[1]),
            ha="center",
            va="top",
            fontsize=8,
            color=RED,
        )
    ax_rand.set_xticks(range(len(folds)), [f"période {f}" for f in folds])
    ax_rand.set_title("Agents contre le hasard (même nombre de trades)", loc="left", fontsize=10.5)
    ax_rand.legend(frameon=False, fontsize=8.5, loc="lower left")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
