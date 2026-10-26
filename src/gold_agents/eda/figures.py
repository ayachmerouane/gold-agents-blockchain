"""Figures de l'EDA (matplotlib, PNG). Une fonction par figure, chacune renvoie son chemin."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from gold_agents.config import SplitsConfig
from gold_agents.eda.analysis import assign_split, rolling_volatility, train_part

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

# Palette catégorielle validée (ordre fixe) et encres neutres.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SPLIT_COLORS = {"train": BLUE, "val": ORANGE, "test": AQUA}


def _new(figsize: tuple[float, float] = (10, 4)) -> tuple[Figure, Axes]:
    import matplotlib

    matplotlib.use("Agg")  # pas d'écran nécessaire
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize, dpi=120)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    return fig, ax


def _save(fig: Figure, path: Path) -> Path:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _naive(index: pd.Index) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(index).tz_localize(None)


def _shade_splits(ax: Axes, splits: SplitsConfig, end: pd.Timestamp) -> None:
    spans = {
        "train": (None, pd.Timestamp(splits.train_end)),
        "val": (pd.Timestamp(splits.val_start), pd.Timestamp(splits.val_end)),
        "test": (pd.Timestamp(splits.test_start), end),
    }
    # matplotlib accepte les dates sur l'axe x, mais ses annotations de type attendent un float.
    for name, (start, stop) in spans.items():
        if start is not None:
            ax.axvspan(start, stop, color=SPLIT_COLORS[name], alpha=0.07, linewidth=0)  # type: ignore[arg-type]
        ax.axvline(stop, color=MUTED, linewidth=0.6, linestyle=":")  # type: ignore[arg-type]


def price_figure(
    kaggle: pd.DataFrame, update: pd.DataFrame, splits: SplitsConfig, path: Path
) -> Path:
    fig, ax = _new()
    k = kaggle.set_index("date")["close"]
    end = _naive(k.index).max()
    _shade_splits(ax, splits, end)
    for name, color in SPLIT_COLORS.items():
        part = k[[s == name for s in assign_split(k.index, splits)]]
        ax.plot(_naive(part.index), part.to_numpy(), color=color, linewidth=1.2, label=name)
    if not update.empty:
        u = update.set_index("date")["close"]
        ax.plot(_naive(u.index), u.to_numpy(), color=MUTED, linewidth=1.2, label="mise à jour API")
    from matplotlib.ticker import FuncFormatter

    ax.set_yscale("log")
    plain = FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", " "))
    ax.yaxis.set_major_formatter(plain)
    ax.yaxis.set_minor_formatter(plain)
    ax.set_title("Clôture XAU/USD (échelle log) et découpage train / val / test", color=INK)
    ax.set_ylabel("USD par once", color=MUTED)
    ax.legend(frameon=False, fontsize=9)
    return _save(fig, path)


def returns_figure(r: pd.Series, path: Path) -> Path:
    fig, ax = _new()
    ax.plot(_naive(r.index), r.to_numpy() * 100, color=BLUE, linewidth=0.5)
    ax.axhline(0, color=MUTED, linewidth=0.6)
    ax.set_title("Rendement log quotidien (%)", color=INK)
    ax.set_ylabel("%", color=MUTED)
    return _save(fig, path)


def distribution_figure(r: pd.Series, splits: SplitsConfig, path: Path) -> Path:
    """Histogramme des rendements du train face à la loi normale de même moyenne/écart-type."""
    x = train_part(r, splits) * 100
    fig, ax = _new((8, 4))
    ax.hist(x, bins=120, density=True, color=BLUE, alpha=0.8, edgecolor="white", linewidth=0.3)
    grid = np.linspace(x.min(), x.max(), 400)
    mu, sd = x.mean(), x.std()
    ax.plot(
        grid,
        np.exp(-((grid - mu) ** 2) / (2 * sd**2)) / (sd * np.sqrt(2 * np.pi)),
        color=ORANGE,
        linewidth=2,
        label="loi normale",
    )
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-4)
    ax.set_title("Distribution des rendements (train) : queues plus épaisses que la normale")
    ax.set_xlabel("rendement quotidien (%)", color=MUTED)
    ax.set_ylabel("densité (échelle log)", color=MUTED)
    ax.legend(frameon=False, fontsize=9)
    return _save(fig, path)


def volatility_figure(
    r: pd.Series, splits: SplitsConfig, window: int, thresholds: dict[str, float], path: Path
) -> Path:
    vol = rolling_volatility(r, window).dropna() * 100
    fig, ax = _new()
    _shade_splits(ax, splits, _naive(vol.index).max())
    ax.plot(_naive(vol.index), vol.to_numpy(), color=BLUE, linewidth=1)
    # Étiquette du seuil bas sous la ligne, du seuil haut au-dessus : elles ne se chevauchent pas.
    for label, value, dy in (
        ("basse / moyenne", "low_mid", -11),
        ("moyenne / haute", "mid_high", 3),
    ):
        ax.axhline(thresholds[value] * 100, color=ORANGE, linewidth=1, linestyle="--")
        ax.annotate(
            label,
            (_naive(vol.index).min(), thresholds[value] * 100),
            xytext=(4, dy),
            textcoords="offset points",
            fontsize=8,
            color=MUTED,
        )
    ax.set_title(f"Volatilité annualisée sur {window} jours ; seuils de régime issus du train")
    ax.set_ylabel("%", color=MUTED)
    return _save(fig, path)


def acf_figure(acf_report: dict[str, Any], path: Path) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), dpi=120, sharey=True)
    band = acf_report["band_95"]
    titles = {"returns": "ACF des rendements r", "abs_returns": "ACF de |r| (volatilité)"}
    for ax, (key, title) in zip(axes, titles.items(), strict=True):
        values = acf_report[key]["acf"]
        lags = np.arange(1, len(values) + 1)
        ax.bar(lags, values, color=BLUE, width=0.6)
        ax.axhspan(-band, band, color=MUTED, alpha=0.12, linewidth=0)
        ax.axhline(0, color=MUTED, linewidth=0.6)
        ax.set_title(title, color=INK, fontsize=10)
        ax.set_xlabel("retard (jours)", color=MUTED)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("autocorrélation (train)", color=MUTED)
    return _save(fig, path)
