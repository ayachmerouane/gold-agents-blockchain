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
    "buy_and_hold": "#3987e5",
    "sma_20_50": "#d95926",
    "agents_lstm": "#199e70",
    "agents_gru": "#D4AF37",
    "agents_ensemble": "#d55181",
    "agents_ensemble_mc": "#2fb344",
    "agents_lightgbm": "#9085e9",
    "aleatoire": "#e66767",
    "oracle": "#7c8599",
}
BUY_COLOR, SELL_COLOR = "#2fb344", "#e5484d"
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


# Couleurs lisibles en mode clair comme en mode sombre : le texte principal hérite du
# thème, les fonds et bordures sont des gris semi-transparents.
GOLD = "#C9A227"
MUTED, GRID, LINE = "#7C8599", "rgba(128,128,128,0.16)", "rgba(128,128,128,0.35)"
STYLE = """
<style>
.block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1320px;}
h1 {font-weight: 700; letter-spacing: -0.02em; margin-bottom: 0 !important;
    font-size: 2.1rem !important;}
h2, h3, h4 {font-weight: 600; letter-spacing: -0.01em;}
.ga-subtitle {color: #7C8599; font-size: 0.98rem; margin: 0.1rem 0 1.5rem;}
.ga-section {font-size: 0.76rem; font-weight: 700; letter-spacing: 0.12em;
             text-transform: uppercase; color: #B8901F; margin: 1.8rem 0 0.6rem;}
.ga-kpis {display: grid; gap: 14px; margin: 0.4rem 0 1rem;}
.ga-kpi {background: rgba(128,128,128,0.07); border: 1px solid rgba(128,128,128,0.22);
         border-radius: 12px; padding: 16px 18px;}
.ga-kpi .lbl {font-size: 0.74rem; letter-spacing: 0.06em; text-transform: uppercase;
              color: #7C8599;}
.ga-kpi .val {font-size: 1.6rem; font-weight: 700; margin-top: 4px;
              font-variant-numeric: tabular-nums;}
.ga-kpi .sub {font-size: 0.8rem; margin-top: 2px; color: #7C8599;}
.ga-kpi .up {color: #1f9d3a;} .ga-kpi .down {color: #d93f44;}
.ga-kpi.accent {border-color: rgba(201,162,39,0.65); background: rgba(201,162,39,0.08);}
.ga-kpi.accent .val {color: #C9A227;}
.ga-chain {display: flex; align-items: center; overflow-x: auto; padding: 6px 0 12px;}
.ga-block {min-width: 150px; background: rgba(128,128,128,0.07);
           border: 1px solid rgba(128,128,128,0.22); border-radius: 10px; padding: 10px 12px;}
.ga-block.current {border-color: #C9A227; box-shadow: 0 0 0 1px rgba(201,162,39,0.35);}
.ga-block .n {font-size: 0.72rem; color: #7C8599; letter-spacing: 0.05em;}
.ga-block .d {font-size: 0.95rem; font-weight: 600;}
.ga-block .e {font-size: 0.75rem; color: #7C8599; margin-top: 2px;}
.ga-block .ok {font-size: 0.72rem; color: #1f9d3a; margin-top: 4px;}
.ga-link {color: #C9A227; font-size: 1.1rem; padding: 0 6px;}
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
    """Mise en page commune. Les couleurs de texte viennent du thème Streamlit (clair ou
    sombre) ; fonds transparents et grille grise conviennent aux deux."""
    fig.update_layout(
        height=height,
        separators=", ",  # décimales à la virgule, milliers à l'espace
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 8, "r": 8, "t": 48, "b": 8},
        hovermode="x unified",
        title={"x": 0, "xanchor": "left", "y": 0.97, "yanchor": "top", "font": {"size": 15}},
        legend={"orientation": "h", "yanchor": "top", "y": -0.1, "x": 0},
    )
    fig.update_yaxes(title_text=y_title, tickformat=y_format, gridcolor=GRID, zeroline=False)
    fig.update_xaxes(gridcolor=GRID, showgrid=False, linecolor=LINE)
    return fig


def kpis(cards: list[tuple[str, str, str, str]], accent: int | None = None) -> None:
    """Rangée de cartes : (libellé, valeur, sous-texte, sens « up »/« down »/« »)."""
    html = []
    for i, (lbl, val, sub, tone) in enumerate(cards):
        cls = "ga-kpi accent" if i == accent else "ga-kpi"
        sub_html = f'<div class="sub {tone}">{sub}</div>' if sub else ""
        html.append(
            f'<div class="{cls}"><div class="lbl">{lbl}</div><div class="val">{val}</div>'
            f"{sub_html}</div>"
        )
    style = f"grid-template-columns: repeat({len(cards)}, minmax(0, 1fr));"
    st.markdown(
        f'<div class="ga-kpis" style="{style}">{"".join(html)}</div>', unsafe_allow_html=True
    )


def tone(x: float) -> str:
    return "up" if x > 0 else "down" if x < 0 else ""


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
