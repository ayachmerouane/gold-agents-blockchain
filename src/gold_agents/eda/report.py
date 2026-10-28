"""Exécute toute l'EDA et écrit `summary.json`, `rapport_eda.md` et les figures."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gold_agents.config import Config
from gold_agents.data.load import read_kaggle_csv, read_update_csv
from gold_agents.eda import analysis, figures


@dataclass(frozen=True)
class EdaResult:
    summary: dict[str, Any]
    report_path: Path
    summary_path: Path
    figure_paths: list[Path]


def analyze(
    kaggle: pd.DataFrame, update: pd.DataFrame, cfg: Config, outlier_threshold: float = 6.0
) -> dict[str, Any]:
    """Toutes les mesures de l'EDA, sous forme de dictionnaire sérialisable en JSON."""
    if not update.empty:
        update = update[update["date"] > kaggle["date"].max()].reset_index(drop=True)
    splits, window = cfg.splits, cfg.features.window
    r = analysis.log_returns(kaggle)
    r_train = analysis.train_part(r, splits)
    log_price_train = analysis.train_part(np.log(kaggle.set_index("date")["close"]), splits)
    return {
        "calendar": analysis.calendar_report(kaggle),
        "ohlc": analysis.ohlc_report(kaggle),
        "returns_train": analysis.returns_report(r_train),
        "returns_all": analysis.returns_report(r),
        "outliers": analysis.outliers_report(kaggle, splits, threshold=outlier_threshold),
        "volatility": analysis.volatility_regimes(r, splits, cfg.features.volatility_window),
        "autocorrelation_train": analysis.autocorrelation_report(r_train),
        "stationarity_train": analysis.stationarity_report(log_price_train, r_train),
        "split_preview": analysis.split_preview(kaggle, splits, window),
        "leakage_checks": analysis.leakage_checks(kaggle, update, splits, window),
        "source_junction": analysis.source_junction(kaggle, update, splits),
        "update_calendar": analysis.calendar_report(update) if len(update) > 1 else None,
        "update_ohlc": analysis.ohlc_report(update) if not update.empty else None,
    }


def _pct(x: float, digits: int = 2) -> str:
    return f"{x * 100:.{digits}f} %"


def _p(x: float) -> str:
    return "< 0.001" if x < 0.001 else f"{x:.3f}"


def _split_line(name: str, p: dict[str, Any], share: float) -> str:
    if not p["n_rows"]:
        return f"  {name:5s} vide"
    return (
        f"  {name:5s} {p['first_date']} -> {p['last_date']}  {p['n_rows']} lignes "
        f"({_pct(share, 0)}), utilisables {p['n_usable_after_purge']}, "
        f"vol {_pct(p['annualized_volatility'], 1)}, buy & hold {_pct(p['buy_and_hold_return'], 0)}"
    )


def summary_lines(s: dict[str, Any]) -> list[str]:
    """Résumé court (une quarantaine de lignes), affiché dans le terminal pour être collé."""
    cal, ohlc, ret = s["calendar"], s["ohlc"], s["returns_train"]
    out, vol, acf = s["outliers"], s["volatility"], s["autocorrelation_train"]
    sp, junction = s["split_preview"], s["source_junction"]
    counts = {k: v for k, v in ohlc.items() if not k.startswith(("volume", "violation"))}
    ohlc_issues = {k: v for k, v in counts.items() if v}
    gaps = [(g["after"], g["before"], g["missing_business_days"]) for g in cal["largest_gaps"][:5]]
    cuts = vol["thresholds_from_train"]
    lines = [
        "== RÉSUMÉ EDA XAU/USD (CSV Kaggle) ==",
        f"Lignes : {cal['n_rows']}  du {cal['first_date']} au {cal['last_date']}",
        f"Lignes par jour : {cal['rows_per_weekday']}",
        f"Jours ouvrés absents : {cal['missing_business_days']}  "
        f"(trous > 1 jour ouvré : {cal['gaps_over_1_business_day']})",
        f"Plus grands trous : {gaps}",
        f"Absents par année : {cal['missing_business_days_per_year']}",
        f"OHLC incohérents : {ohlc_issues or 'aucun'}  exemples : {ohlc['violation_examples']}",
        f"Volume : manquant {_pct(ohlc['volume_missing_share'], 1)}, "
        f"nul sur {ohlc['volume_zero_rows']} lignes",
        "-- Rendements log (train) --",
        f"Moyenne annualisée {_pct(ret['annualized_return'])}, volatilité annualisée "
        f"{_pct(ret['annualized_volatility'])}, jours positifs {_pct(ret['positive_share'], 1)}",
        f"Asymétrie {ret['skewness']:.2f}, kurtosis en excès {ret['excess_kurtosis']:.2f}, "
        f"Jarque-Bera p = {_p(ret['jarque_bera']['p_value'])}",
        f"Au-delà de 4 écarts-types : {_pct(ret['share_beyond_4_sigma'], 3)} "
        f"(loi normale : {_pct(ret['normal_share_beyond_4_sigma'], 3)})",
        f"Pire jour {ret['min']['date']} {_pct(ret['min']['value'])}, "
        f"meilleur {ret['max']['date']} {_pct(ret['max']['value'])}",
        f"-- Valeurs aberrantes (|z robuste| > {out['threshold']:g}) --",
        f"Nombre : {out['counts']}  allers-retours suspects : {out['spike_reversals'] or 'aucun'}",
        *(
            f"  {o['date']} [{o['split']}] {o['kind']:8s} {o['value'] * 100:+.2f} %  "
            f"z = {o['z']:+.1f}"
            for o in out["top"][:8]
        ),
        f"-- Volatilité {vol['window']} j : seuils du train {_pct(cuts['low_mid'], 1)} "
        f"/ {_pct(cuts['mid_high'], 1)} --",
        *(
            f"  {name} : " + ", ".join(f"{k} {_pct(v, 0)}" for k, v in shares.items())
            for name, shares in vol["regime_share_by_split"].items()
        ),
        f"Pic de volatilité : {vol['max_volatility']['date']} "
        f"{_pct(vol['max_volatility']['value'], 1)}",
        "-- Dépendance temporelle (train) --",
    ]
    for key, label in (("returns", "r"), ("abs_returns", "|r|")):
        first = [round(v, 3) for v in acf[key]["acf"][:3]]
        p_lb = acf[key]["ljung_box"][10]["p_value"]
        lines.append(f"ACF {label} lags 1-3 : {first}  Ljung-Box(10) p = {_p(p_lb)}")
    lines.append(f"(bande de confiance 95 % : ±{acf['band_95']:.3f})")
    for name, t in s["stationarity_train"].items():
        lines.append(
            f"{name} : ADF p = {_p(t['adf']['p_value'])}, KPSS p = {_p(t['kpss']['p_value'])}"
            f" -> {t['verdict']}"
        )
    lines.append(f"-- Découpage (W = {sp['window']}, purge de W jours en tête de val/test) --")
    lines += [_split_line(name, p, sp["share"][name]) for name, p in sp["splits"].items()]
    lines.append(f"Contrôles anti-fuite : {s['leakage_checks']}")
    if junction["update_rows"]:
        lines.append(
            f"Mise à jour API : {junction['update_rows']} lignes "
            f"{junction['update_first_date']} -> {junction['update_last_date']} "
            f"({', '.join(junction['update_sources'])}), raccord "
            f"{_pct(junction['junction_log_return'])} (z = {junction['junction_robust_z']:+.1f}), "
            f"{junction['business_days_between']} jour(s) ouvré(s) manquant(s) au raccord"
        )
    else:
        lines.append("Mise à jour API : aucun fichier (ou aucun jour après le CSV Kaggle)")
    return lines


_REPORT_INTRO = """# Rapport d'analyse exploratoire (EDA) : XAU/USD quotidien

Généré par `gold-agents eda`. Toutes les statistiques de distribution, d'autocorrélation et de
stationnarité, ainsi que les seuils (régimes de volatilité, valeurs aberrantes), sont calculés
sur le **train seulement** ; validation et test ne servent qu'à appliquer ces seuils.
Les jours ajoutés par l'API ne font partie d'aucun ensemble.

## Résumé

```
{summary}
```

## Figures

{figures}

## Données complètes

Toutes les valeurs : `summary.json` (même dossier).
"""


def write_report(
    kaggle: pd.DataFrame, update: pd.DataFrame, cfg: Config, out_dir: Path
) -> EdaResult:
    summary = analyze(kaggle, update, cfg)
    fig_dir = out_dir / "figures"
    later = update[update["date"] > kaggle["date"].max()] if not update.empty else update
    r = analysis.log_returns(kaggle)
    fig_paths = [
        figures.price_figure(kaggle, later, cfg.splits, fig_dir / "01_prix.png"),
        figures.returns_figure(r, fig_dir / "02_rendements.png"),
        figures.distribution_figure(r, cfg.splits, fig_dir / "03_distribution.png"),
        figures.volatility_figure(
            r,
            cfg.splits,
            cfg.features.volatility_window,
            summary["volatility"]["thresholds_from_train"],
            fig_dir / "04_volatilite.png",
        ),
        figures.acf_figure(summary["autocorrelation_train"], fig_dir / "05_autocorrelation.png"),
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    report_path = out_dir / "rapport_eda.md"
    report_path.write_text(
        _REPORT_INTRO.format(
            summary="\n".join(summary_lines(summary)),
            figures="\n\n".join(f"![{p.stem}](figures/{p.name})" for p in fig_paths),
        ),
        encoding="utf-8",
    )
    return EdaResult(summary, report_path, summary_path, fig_paths)


def run_eda(cfg: Config, out_dir: Path) -> EdaResult:
    """Lit les fichiers bruts désignés par la configuration et écrit le rapport."""
    kaggle = read_kaggle_csv(cfg.data.raw_path)
    path = cfg.data.update_path
    update = read_update_csv(path) if path.exists() else kaggle.iloc[0:0]
    return write_report(kaggle, update, cfg, out_dir)
