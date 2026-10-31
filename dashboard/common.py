"""Outils partagés par les pages : chargement en cache, couleurs, style, mise en forme.

Rien n'est recalculé ici : on lit les registres de `outputs/ledger/<période>/` et, s'ils
existent, les résultats de `outputs/results/`. Le cache est invalidé quand un fichier
change (date de modification dans la clé). Aucun nom de fichier n'apparaît à l'écran.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from gold_agents.dashboard import data

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_KEY = ROOT / "ledger" / "public_key.pem"
MLFLOW_DB = ROOT / "mlruns" / "mlflow.db"

# Couleur fixe par stratégie (palette catégorielle validée, dans l'ordre) : une stratégie
# garde sa couleur quelle que soit la sélection. L'oracle, simple illustration, est gris.
COLORS = {
    "buy_and_hold": "#2a78d6",
    "sma_20_50": "#eb6834",
    "agents_lstm": "#1baf7a",
    "agents_gru": "#eda100",
    "agents_ensemble": "#e87ba4",
    "agents_ensemble_mc": "#008300",
    "agents_lightgbm": "#6250d6",
    "aleatoire": "#e34948",
    "oracle": "#8a8984",
}
BUY_COLOR, SELL_COLOR = "#0f8a3c", "#c62f2f"
DEFAULT_COMPARE = ("buy_and_hold", "sma_20_50", "agents_gru", "agents_ensemble_mc")


def color(name: str) -> str:
    return COLORS.get(name, "#52514e")


def file_stamp(paths: list[Path]) -> tuple[tuple[str, int], ...]:
    return tuple((p.as_posix(), p.stat().st_mtime_ns) for p in paths if p.is_file())


@dataclass
class Period:
    run: data.RunSet
    frames: dict[str, pd.DataFrame]
    trades: dict[str, pd.DataFrame]
    geneses: dict[str, dict[str, Any]]
    results: dict[str, Any]
    plan: dict[str, Any] | None
    close: pd.Series
    metrics: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def strategies(self) -> list[str]:
        return [s for s in self.run.strategies if not self.frames[s].empty]

    @property
    def main_system(self) -> str | None:
        main = (self.plan or {}).get("main_system")
        return main if main in self.frames else None

    def forecasters(self) -> list[str]:
        return [s for s in self.strategies if "r_hat" in self.frames[s]]


@st.cache_data(show_spinner="Lecture des registres…", max_entries=8)
def _load_period(ledger_dir: str, stamp: tuple[tuple[str, int], ...]) -> Period:
    root = ROOT
    run = next(r for r in data.run_sets(root) if r.ledger_dir.as_posix() == ledger_dir)
    frames, trades, geneses = {}, {}, {}
    for name in run.strategies:
        blocks = data.read_blocks(run.ledger(name))
        frames[name] = data.daily_frame(blocks)
        trades[name] = data.trades_frame(blocks)
        geneses[name] = data.genesis(blocks)
    plan_path = run.ledger_dir / "plan.jsonl"
    plan = data.genesis(data.read_blocks(plan_path)) if plan_path.is_file() else None
    results = data.results_events(run.ledger_dir)
    period = Period(run, frames, trades, geneses, results, plan, data.prices(frames))
    period.metrics = data.metrics_table(results.get("metrics"), frames, trades)
    return period


def run_sets() -> list[data.RunSet]:
    return data.run_sets(ROOT)


def current_period() -> Period | None:
    """La période choisie dans la barre latérale (voir app.py)."""
    sets = run_sets()
    if not sets:
        return None
    name = st.session_state.get("period", sets[0].name)
    run = next((r for r in sets if r.name == name), sets[0])
    files = sorted(run.ledger_dir.glob("*.jsonl"))
    return _load_period(run.ledger_dir.as_posix(), file_stamp(files))


def require_period() -> Period:
    period = current_period()
    if period is None:
        st.error(
            "Aucun résultat à afficher pour le moment. Lancez d'abord l'évaluation finale, "
            "puis rouvrez le tableau de bord."
        )
        st.stop()
    return period


GOLD = "#C9A227"
STYLE = """
<style>
.block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1280px;}
h1 {font-weight: 700; letter-spacing: -0.02em; margin-bottom: 0 !important;}
h2, h3 {font-weight: 600; letter-spacing: -0.01em;}
.ga-subtitle {color: rgba(128,128,128,0.95); font-size: 0.98rem; margin: 0.15rem 0 1.4rem;}
.ga-section {font-size: 0.78rem; font-weight: 600; letter-spacing: 0.08em;
             text-transform: uppercase; color: rgba(128,128,128,0.95); margin: 1.6rem 0 0.4rem;}
[data-testid="stMetricValue"] {font-size: 1.7rem; font-weight: 600;}
[data-testid="stMetricLabel"] p {font-size: 0.82rem; color: rgba(128,128,128,0.95);}
code {white-space: nowrap;}
footer {visibility: hidden;}
</style>
"""


def apply_style() -> None:
    st.markdown(STYLE, unsafe_allow_html=True)


def header(title: str, subtitle: str = "") -> None:
    """Titre de page et sous-titre discret."""
    st.title(title)
    if subtitle:
        st.markdown(f'<p class="ga-subtitle">{subtitle}</p>', unsafe_allow_html=True)


def section(title: str) -> None:
    """Petit intertitre en capitales, pour séparer les blocs d'une page."""
    st.markdown(f'<p class="ga-section">{title}</p>', unsafe_allow_html=True)


@st.cache_data(show_spinner="Vérification de la chaîne…", max_entries=64)
def verify_cached(path: str, stamp: tuple[tuple[str, int], ...]) -> dict[str, Any]:
    v = data.verify(Path(path), PUBLIC_KEY)
    return {
        "valid": v.valid,
        "n_blocks": v.n_blocks,
        "bad_index": v.bad_index,
        "reason": v.reason,
        "signatures": v.signatures,
        "text": v.describe(),
    }


def verify(path: Path) -> dict[str, Any]:
    return verify_cached(path.as_posix(), file_stamp([path, PUBLIC_KEY]))


@st.cache_data(show_spinner=False, max_entries=16)
def read_blocks_cached(path: str, stamp: tuple[tuple[str, int], ...]) -> list[dict[str, Any]]:
    return data.read_blocks(Path(path))


def blocks(path: Path) -> list[dict[str, Any]]:
    return read_blocks_cached(path.as_posix(), file_stamp([path]))


def strategy_picker(
    period: Period, key: str, label: str = "Stratégie", default: str | None = None
) -> str:
    options = period.strategies
    preferred = default or period.main_system or options[0]
    index = options.index(preferred) if preferred in options else 0
    return st.selectbox(label, options, index=index, format_func=data.label, key=key)


def date_range(period: Period, key: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    dates = period.frames[period.strategies[0]]["date"]
    lo, hi = dates.min().date(), dates.max().date()
    picked = st.slider("Période affichée", lo, hi, (lo, hi), format="DD/MM/YYYY", key=key)
    return pd.Timestamp(picked[0]), pd.Timestamp(picked[1])


def plot_layout(fig: Any, height: int = 420, y_title: str = "", y_format: str = "") -> Any:
    """Mise en page commune : survol unifié, légende au-dessus, grille discrète."""
    fig.update_layout(
        height=height,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        hovermode="x unified",
        title={"x": 0, "xanchor": "left", "y": 0.98, "yanchor": "top"},
        # Légende sous le graphique : elle ne chevauche jamais le titre.
        legend={"orientation": "h", "yanchor": "top", "y": -0.08, "x": 0},
    )
    fig.update_yaxes(title_text=y_title, tickformat=y_format, gridcolor="rgba(128,128,128,0.15)")
    fig.update_xaxes(gridcolor="rgba(128,128,128,0.08)")
    return fig


def metrics_display(table: pd.DataFrame) -> pd.DataFrame:
    """Tableau des métriques lisible : noms en français, pourcentages, montants."""
    out = pd.DataFrame(
        {
            "Stratégie": table["strategy"].map(data.label),
            "Valeur finale": table["final_value"].map(data.fmt_money),
            "Profit net": table["net_profit"].map(data.fmt_money),
            "Rendement": table["total_return"].map(data.fmt_pct),
            "Rendement annuel": table["cagr"].map(data.fmt_pct),
            "Sharpe": table["sharpe"].map(data.fmt_num),
            "Perte maximale": table["max_drawdown"].map(lambda x: data.fmt_pct(x, signed=False)),
            "Transactions": table["n_trades"].astype("Int64"),
            "Temps investi": table["exposure"].map(lambda x: f"{x * 100:.0f} %"),
            "Frais payés": table["total_costs"].map(data.fmt_money),
        }
    )
    return out
