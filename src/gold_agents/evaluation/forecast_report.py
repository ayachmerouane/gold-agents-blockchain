"""Tableau et schéma de la phase 4 : chaque modèle comparé aux baselines (validation)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from gold_agents.evaluation.report import BLUE, GRID, INK, MUTED, ORANGE, RED, _style

if TYPE_CHECKING:
    from gold_agents.models.benchmark import Benchmark

AQUA = "#1baf7a"
NETWORK_COLORS = {"lstm": BLUE, "gru": ORANGE, "ensemble": AQUA}
BASELINE_COLOR = "#a9a7a1"


def _pm(mean: float, std: float, scale: float = 100, digits: int = 3) -> str:
    if math.isnan(mean):
        return "-"
    text = f"{mean * scale:.{digits}f}"
    return text if std == 0 else f"{text} ± {std * scale:.{digits}f}"


def _b0_rmse(summary: pd.DataFrame) -> float:
    return float(summary["rmse"].to_dict().get("b0", math.nan))


def summary_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Version lisible : erreurs en % de rendement, direction en %, gain sur B0 en %."""
    rmse0 = _b0_rmse(summary)
    rows = {}
    for key, r in summary.iterrows():
        rows[r["label"]] = {
            "MAE (%)": _pm(r["mae"], r["mae_std"]),
            "RMSE (%)": _pm(r["rmse"], r["rmse_std"]),
            "vs B0": "-" if key == "b0" else f"{(r['rmse'] / rmse0 - 1):+.2%}",
            "Direction": _pm(r["direction"], r["direction_std"], digits=1),
            "DM p": "-" if key == "b0" else f"{r['dm_p_value']:.2f}",
            "Graines": int(r["n_seeds"]),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def _color(key: str) -> str:
    return NETWORK_COLORS.get(key, BASELINE_COLOR)


def forecast_figure(bench: Benchmark, share_up: float, alert: float, path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = bench.summary()
    keys = list(summary.index)
    labels = list(summary["label"])
    has_curves = bool(bench.histories)
    fig = plt.figure(figsize=(12, 11 if has_curves else 7), dpi=120)
    grid = fig.add_gridspec(
        2 if has_curves else 1, 2, height_ratios=[3, 2] if has_curves else [1], hspace=0.4
    )
    ax_rmse, ax_dir = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    fig.suptitle(
        f"Prévision du rendement du lendemain, validation "
        f"({bench.dates[0]:%d/%m/%Y} au {bench.dates[-1]:%d/%m/%Y}, {len(bench.y_val)} jours)",
        fontsize=13,
        color=INK,
    )
    y = list(range(len(keys)))[::-1]
    colors = [_color(k) for k in keys]

    # 1. RMSE relative à B0 : à gauche de 0, le modèle fait mieux que « rendement nul ».
    _style(ax_rmse)
    rmse0 = _b0_rmse(summary)
    gain = (summary["rmse"] / rmse0 - 1) * 100
    err = summary["rmse_std"] / rmse0 * 100
    # B1 « rendement d'hier » est très loin derrière : l'axe est coupé pour garder
    # lisibles les petits écarts, et la valeur exacte reste écrite à côté de la barre.
    right = max(3.0, min(float((gain + err).max()), 10.0) + 1)
    left = min(-1.0, float((gain - err).min()) - 0.5)
    ax_rmse.barh(
        y,
        gain.clip(upper=right),
        xerr=err,
        color=colors,
        height=0.6,
        error_kw={"ecolor": INK, "lw": 1},
    )
    ax_rmse.axvline(0, color=INK, linewidth=1)
    ax_rmse.set_xlim(left, right + 2.5)
    ax_rmse.set_yticks(y, labels, color=INK, fontsize=9)
    ax_rmse.set_xlabel("Écart de RMSE avec B0 (%), négatif = mieux que B0", color=MUTED)
    ax_rmse.set_title("Erreur de prévision", color=INK, fontsize=11, loc="left")
    for yi, g, e in zip(y, gain, err, strict=True):
        x = min(max(g, 0) + e, right)
        text = f"{g:+.2f} %" + (" (axe coupé)" if g > right else "")
        ax_rmse.annotate(
            text,
            (x, yi),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=MUTED,
        )

    # 2. Précision directionnelle, avec le hasard (50 %), « toujours hausse » et l'alerte.
    _style(ax_dir)
    direction = summary["direction"].fillna(0) * 100
    ax_dir.barh(
        y,
        direction,
        xerr=summary["direction_std"].fillna(0) * 100,
        color=colors,
        height=0.6,
        error_kw={"ecolor": INK, "lw": 1},
    )
    ax_dir.axvline(50, color=INK, linewidth=1, label="Hasard (50 %)")
    ax_dir.axvline(
        share_up * 100,
        color=MUTED,
        linestyle="--",
        linewidth=1,
        label=f"Toujours « hausse » ({share_up:.1%})",
    )
    ax_dir.axvline(
        alert * 100, color=RED, linestyle=":", linewidth=1.2, label=f"Alerte fuite ({alert:.0%})"
    )
    ax_dir.set_xlim(40, max(65, float(direction.max()) + 3))
    ax_dir.set_yticks(y, ["" for _ in labels])
    ax_dir.set_xlabel("Bonnes directions (%)", color=MUTED)
    ax_dir.set_title("Hausse ou baisse : bon signe ?", color=INK, fontsize=11, loc="left")
    ax_dir.legend(loc="lower right", fontsize=8, frameon=False)
    spread = summary["direction_std"].fillna(0) * 100
    for yi, d, e, key in zip(y, direction, spread, keys, strict=True):
        text = "sans avis" if key == "b0" else f"{d:.1f} %"
        ax_dir.annotate(
            text,
            (max(d + e, 40), yi),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=MUTED,
        )

    # 3. Courbes d'apprentissage des réseaux (première graine).
    if has_curves:
        ax_curve = fig.add_subplot(grid[1, :])
        _style(ax_curve)
        for kind, history in bench.histories.items():
            epochs = range(1, len(history) + 1)
            color = NETWORK_COLORS[kind]
            ax_curve.plot(
                epochs,
                [h[0] for h in history],
                color=color,
                linewidth=1,
                alpha=0.5,
                label=f"{kind.upper()} entraînement",
            )
            ax_curve.plot(
                epochs,
                [h[1] for h in history],
                color=color,
                linewidth=2,
                label=f"{kind.upper()} validation",
            )
            best = min(range(len(history)), key=lambda i: history[i][1])
            ax_curve.plot(
                best + 1, history[best][1], "o", color=color, markersize=8, markeredgecolor="white"
            )
        ax_curve.axhline(1.0, color=GRID, linewidth=1)
        ax_curve.set_xlabel("Époque (point : poids gardés, arrêt anticipé ensuite)", color=MUTED)
        ax_curve.set_ylabel("MSE (cible réduite)", color=MUTED)
        ax_curve.set_title(
            "Apprentissage des réseaux (graine 0)", color=INK, fontsize=11, loc="left"
        )
        ax_curve.legend(fontsize=8, frameon=False, ncol=2)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path
