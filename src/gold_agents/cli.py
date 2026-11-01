"""Interface en ligne de commande (Typer) : download, eda, prepare, train, backtest, evaluate…"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

import typer

from gold_agents.config import DEFAULT_CONFIG_PATH, apply_settings, load_config
from gold_agents.data import clean as cleaning
from gold_agents.data import sources, update
from gold_agents.data.load import read_kaggle_csv, read_update_csv

if TYPE_CHECKING:
    from gold_agents.blockchain.signing import Signer
    from gold_agents.config import Config

app = typer.Typer(help="Investissement simulé sur l'or : DL, agents et blockchain.")

CONFIG_OPTION = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c", help="Fichier YAML.")


def _todo(step: str) -> None:
    typer.echo(f"Étape « {step} » pas encore implémentée.")
    raise typer.Exit(code=1)


@app.command()
def check_config(config: Path = CONFIG_OPTION) -> None:
    """Valide le fichier de configuration."""
    cfg = load_config(config)
    typer.echo(f"Configuration valide (graine {cfg.seed}, fenêtre {cfg.features.window}).")


class UpdateSource(StrEnum):
    twelvedata = "twelvedata"
    yfinance = "yfinance"


SOURCE_OPTION = typer.Option(UpdateSource.twelvedata, help="API de mise à jour.")


@app.command()
def download(
    config: Path = CONFIG_OPTION,
    dataset: str | None = typer.Option(
        None, help="Identifiant Kaggle owner/slug (défaut : dataset de la spec, puis repli)."
    ),
    force: bool = typer.Option(False, "--force", help="Écraser le CSV brut déjà présent."),
) -> None:
    """Télécharge le CSV Kaggle quotidien et note son empreinte SHA-256."""
    cfg = load_config(config)
    raw_path = cfg.data.raw_path
    if raw_path.exists() and not force:
        status = update.verify_checksum(raw_path, cfg.data.checksums_path)
        label = {True: "empreinte conforme", False: "EMPREINTE DIFFÉRENTE", None: "sans empreinte"}
        typer.echo(f"{raw_path} existe déjà ({label[status]}) ; --force pour le remplacer.")
        raise typer.Exit(code=0 if status is not False else 1)
    try:
        path, used = sources.download_kaggle(
            raw_path.parent,
            datasets=(dataset,) if dataset else sources.KAGGLE_DATASETS,
            filename=raw_path.name,
        )
    except sources.SourceError as exc:
        typer.echo(f"Échec du téléchargement : {exc}", err=True)
        raise typer.Exit(code=1) from exc
    digest = update.record_checksum(path, cfg.data.checksums_path)
    typer.echo(f"{path} téléchargé depuis kaggle.com/datasets/{used}.")
    typer.echo(f"SHA-256 {digest} noté dans {cfg.data.checksums_path}.")


@app.command("update-data")
def update_data(
    config: Path = CONFIG_OPTION,
    source: UpdateSource = SOURCE_OPTION,
    overlap_days: int = typer.Option(60, help="Jours de recouvrement pour le contrôle."),
    force: bool = typer.Option(False, "--force", help="Ajouter même si le contrôle échoue."),
) -> None:
    """Ajoute les jours manquants depuis la fin du CSV Kaggle (fichier séparé)."""
    _update(load_config(config), source, overlap_days, force)


def _update(cfg: Config, source: UpdateSource, overlap_days: int = 60, force: bool = False) -> None:
    """Ajoute les jours clôturés manquants (Twelve Data ou yfinance) ; Exit en cas d'échec."""
    if not cfg.data.raw_path.exists():
        typer.echo(f"{cfg.data.raw_path} absent : lancer d'abord `make download`.", err=True)
        raise typer.Exit(code=1)
    fetch = (
        sources.fetch_twelvedata if source is UpdateSource.twelvedata else sources.fetch_yfinance
    )
    try:
        result = update.update_series(
            cfg.data.raw_path,
            cfg.data.update_path,
            fetch,
            today=datetime.now(UTC).date(),
            overlap_days=overlap_days,
            force=force,
        )
    except (sources.SourceError, update.CoherenceError) as exc:
        typer.echo(f"Mise à jour refusée : {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if result.report is not None:
        typer.echo(f"Contrôle de cohérence ({source.value}) : {result.report.describe()}")
    typer.echo(
        f"{result.n_new_rows} jour(s) ajouté(s) après le {result.last_date_before} "
        f"dans {cfg.data.update_path}."
    )


EDA_OUT_OPTION = typer.Option(Path("outputs/eda"), "--out", help="Dossier du rapport.")


@app.command()
def eda(
    config: Path = CONFIG_OPTION,
    out_dir: Path = EDA_OUT_OPTION,
) -> None:
    """Analyse exploratoire : trous, aberrations, rendements, volatilité, splits."""
    from gold_agents.eda.report import run_eda, summary_lines

    cfg = load_config(config)
    if not cfg.data.raw_path.exists():
        typer.echo(f"{cfg.data.raw_path} absent : lancer d'abord `make download`.", err=True)
        raise typer.Exit(code=1)
    result = run_eda(cfg, out_dir)
    typer.echo("\n".join(summary_lines(result.summary)))
    typer.echo(f"\nRapport : {result.report_path}  (figures dans {out_dir / 'figures'})")


@app.command("fill-gaps")
def fill_gaps(
    config: Path = CONFIG_OPTION,
    force: bool = typer.Option(False, "--force", help="Écrire même si le contrôle échoue."),
) -> None:
    """Comble les trous du CSV Kaggle avec Twelve Data (fichier séparé, figé)."""
    cfg = load_config(config)
    if not cfg.data.raw_path.exists():
        typer.echo(f"{cfg.data.raw_path} absent : lancer d'abord `make download`.", err=True)
        raise typer.Exit(code=1)
    raw = read_kaggle_csv(cfg.data.raw_path)
    try:
        bars, fills = cleaning.fetch_gap_bars(raw, sources.fetch_twelvedata, force=force)
    except (sources.SourceError, cleaning.GapFillError) as exc:
        typer.echo(f"Comblement refusé : {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if not fills:
        typer.echo("Aucun trou anormal dans le CSV Kaggle : rien à combler.")
        return
    for fill in fills:
        g = fill.gap
        typer.echo(
            f"Trou {g.after} -> {g.before} ({g.missing} jours ouvrés) : "
            f"{fill.n_filled} jour(s) récupéré(s). Contrôle : {fill.report.describe()}"
        )
    path = cfg.data.gapfill_path
    cleaning.write_gapfill(bars, path)
    digest = update.record_checksum(path, cfg.data.checksums_path)
    typer.echo(f"{len(bars)} jour(s) écrit(s) dans {path} (SHA-256 {digest[:12]}…).")


@app.command("clean")
def clean_data(config: Path = CONFIG_OPTION) -> None:
    """Nettoie le CSV Kaggle (+ trous comblés) et écrit le dataset propre en Parquet."""
    cfg = load_config(config)
    if not cfg.data.raw_path.exists():
        typer.echo(f"{cfg.data.raw_path} absent : lancer d'abord `make download`.", err=True)
        raise typer.Exit(code=1)
    raw = read_kaggle_csv(cfg.data.raw_path)
    gap_path = cfg.data.gapfill_path
    gapfill = read_update_csv(gap_path) if gap_path.exists() else None
    if gapfill is None and cleaning.find_gaps(raw):
        typer.echo(f"{gap_path} absent : trous non comblés (`make fill-gaps` pour les combler).")
    result = cleaning.clean(raw, cfg.splits, gapfill)
    problems = cleaning.check_clean(result)
    if problems:
        typer.echo(f"Nettoyage incohérent : {problems}", err=True)
        raise typer.Exit(code=1)
    path = cleaning.write_clean(result, cfg.data.processed_dir, cfg.data.raw_path, gap_path)
    typer.echo(
        f"{len(result)} lignes, dont {int(result['filled'].sum())} comblée(s), "
        f"{int(result['ohlc_fixed'].sum())} bougie(s) corrigée(s), "
        f"{int((~result['valid_return']).sum())} rendement(s) invalide(s). Écrit : {path}"
    )


@app.command()
def prepare(config: Path = CONFIG_OPTION) -> None:
    """Features, cible et découpage train/val/test à partir du dataset propre."""
    import pandas as pd

    from gold_agents.data import prepare as preparing

    cfg = load_config(config)
    clean_path = cfg.data.processed_dir / "xau_1d_clean.parquet"
    if not clean_path.exists():
        typer.echo(f"{clean_path} absent : lancer d'abord `make clean`.", err=True)
        raise typer.Exit(code=1)
    result = preparing.prepare(pd.read_parquet(clean_path), cfg)
    problems = preparing.check_prepared(result, cfg.features.window)
    if problems:
        typer.echo(f"Préparation incohérente : {problems}", err=True)
        raise typer.Exit(code=1)
    path = preparing.write_prepared(result, cfg, clean_path)
    frame = result.frame
    typer.echo(f"{len(result.columns)} features, fenêtre W = {cfg.features.window} jours.")
    for name in ("train", "val", "test"):
        part = frame[frame["sample"] & (frame["split"] == name)]
        typer.echo(
            f"  {name:5s} : {len(part)} exemples, du {part['date'].min():%Y-%m-%d} "
            f"au {part['date'].max():%Y-%m-%d}, {part['target_up'].mean():.1%} de hausses"
        )
    typer.echo(f"Écrit : {path} (+ scaler.json ajusté sur le train)")


TRAIN_OUT_OPTION = typer.Option(
    Path("outputs/results/forecast"), "--out", help="Dossier des résultats et du schéma."
)
MODELS_DIR_OPTION = typer.Option(Path("outputs/models"), "--models-dir", help="Poids.")


@app.command()
def train(
    config: Path = CONFIG_OPTION,
    out_dir: Path = TRAIN_OUT_OPTION,
    models_dir: Path = MODELS_DIR_OPTION,
    models: str = typer.Option(
        "all", help="Modèles, séparés par des virgules : b0,b1,b1_mm20,b2,b3,b4,lstm,gru,ensemble."
    ),
    trials: int | None = typer.Option(None, help="Essais Optuna par réseau (défaut : config)."),
    quick: bool = typer.Option(
        False, "--quick", help="Essai rapide : 3 essais, 5 époques, 2 graines (vérifie la chaîne)."
    ),
    mlflow: bool = typer.Option(True, "--mlflow/--no-mlflow", help="Suivi des essais MLflow."),
) -> None:
    """Baselines B0-B4 puis LSTM et GRU : tableau comparatif sur la validation."""
    import time

    from gold_agents.data.prepare import FEATURES_FILE, load_prepared
    from gold_agents.data.update import sha256_file
    from gold_agents.evaluation.forecast_report import forecast_figure, summary_table
    from gold_agents.models import benchmark as bm
    from gold_agents.models.dataset import build_forecast_data

    chosen = bm.ALL_MODELS if models == "all" else tuple(m.strip() for m in models.split(","))
    unknown = set(chosen) - set(bm.ALL_MODELS)
    if unknown:
        typer.echo(f"Modèles inconnus : {sorted(unknown)}", err=True)
        raise typer.Exit(code=1)
    cfg = load_config(config)
    path = cfg.data.processed_dir / FEATURES_FILE
    if not path.exists():
        typer.echo(f"{path} absent : lancer d'abord `make data`.", err=True)
        raise typer.Exit(code=1)
    mc = cfg.models
    settings = bm.Settings(
        models=chosen,
        seeds=tuple(mc.seeds[:2] if quick else mc.seeds),
        trials=3 if quick else (trials or mc.optuna_trials),
        lgbm_trials=3 if quick else mc.lgbm_trials,
        max_epochs=5 if quick else mc.max_epochs,
        patience=mc.patience,
        mc_samples=mc.mc_samples,
        direction_alert=mc.direction_alert,
        threads=mc.threads,
        mlflow_uri=mc.mlflow_uri if mlflow else None,
    )
    frame, scaler = load_prepared(cfg.data.processed_dir)
    data = build_forecast_data(frame, scaler, cfg.features.window)
    typer.echo(
        f"{len(data.columns)} features, fenêtre {data.window} jours : "
        f"train {len(data['train'].y)}, validation {len(data['val'].y)} exemples "
        f"(le test n'est pas utilisé). Graines {list(settings.seeds)}."
    )
    start = time.perf_counter()
    bench = bm.run_benchmark(data, settings, models_dir, log=typer.echo)
    summary = bm.write_results(bench, out_dir, {"data_sha256": sha256_file(path), "quick": quick})
    share_up = float((data["val"].y > 0).mean())
    figure = forecast_figure(bench, share_up, mc.direction_alert, out_dir / "forecast_val.png")
    typer.echo(f"\nRésultats sur la validation ({time.perf_counter() - start:.0f} s au total) :")
    typer.echo(summary_table(summary).to_string())
    typer.echo(f"Toujours « hausse » : {share_up:.1%} de bonnes directions.")
    for alert in bench.alerts:
        typer.echo(f"ALERTE : {alert}")
    typer.echo(f"\nSchéma : {figure}\nRésultats : {out_dir}  Poids : {models_dir}")
    if mlflow:
        typer.echo(f"Suivi MLflow : uv run mlflow ui --backend-store-uri {mc.mlflow_uri}")


BACKTEST_OUT_OPTION = typer.Option(
    Path("outputs/results/backtest"), "--out", help="Dossier des résultats et du schéma."
)
LEDGER_DIR_OPTION = typer.Option(Path("outputs/ledger"), "--ledger-dir", help="Registres.")
PRIVATE_KEY_OPTION = typer.Option(
    Path("keys/ledger_ed25519.key"), "--key", help="Clé privée qui signe les registres."
)


def _ledger_signer(key: Path) -> Signer | None:
    """Charge la clé privée ; sans elle, les registres sont écrits non signés."""
    from gold_agents.blockchain.signing import fingerprint, load_signer

    if not key.exists():
        typer.echo(
            f"ATTENTION : {key} absente, registres NON signés "
            "(créer la clé avec `uv run gold-agents keygen`).",
            err=True,
        )
        return None
    signer = load_signer(key)
    typer.echo(f"Registres signés avec la clé {fingerprint(signer.public_key)}.")
    return signer


@app.command()
def backtest(
    config: Path = CONFIG_OPTION,
    out_dir: Path = BACKTEST_OUT_OPTION,
    ledger_dir: Path = LEDGER_DIR_OPTION,
    costs_factor: float = typer.Option(1.0, "--costs", help="Multiplie frais et slippage."),
    key: Path = PRIVATE_KEY_OPTION,
) -> None:
    """Backtests de référence (Buy & Hold, SMA 20/50) sur le test, registre et schéma."""
    import pandas as pd

    from gold_agents.data.prepare import FEATURES_FILE
    from gold_agents.data.update import sha256_file
    from gold_agents.evaluation import report

    cfg = load_config(config)
    path = cfg.data.processed_dir / FEATURES_FILE
    if not path.exists():
        typer.echo(f"{path} absent : lancer d'abord `make data`.", err=True)
        raise typer.Exit(code=1)
    frame = pd.read_parquet(path, columns=["date", "split", "sample", "open", "close"])
    signer = _ledger_signer(key)
    runs = report.run_reference_backtests(
        cfg, frame, sha256_file(path), out_dir, ledger_dir, costs_factor, signer
    )
    start, end = report.evaluation_period(frame)
    typer.echo(f"Période de test : du {start} au {end}, frais x {costs_factor:g}.\n")
    typer.echo(report.metrics_table(runs).T.to_string())
    for run in runs:
        typer.echo(f"\nValeur en fin de mois, {run.result.strategy} (k$) :")
        months = report.monthly_values(run.result)
        typer.echo(" ".join(f"{p:%y-%m}={v / 1000:.2f}" for p, v in months.items()))
        trades = run.result.trades
        if 0 < len(trades) <= 40:
            typer.echo(" ".join(f"{t.side[0]}{t.exec_date:%y-%m-%d}" for t in trades))
    typer.echo("\nTest de falsification sur le registre Buy & Hold :")
    public_key = signer.public_key if signer else None
    typer.echo("\n".join(report.tamper_lines(runs[0], public_key)))
    figure = report.backtest_figure(runs, cfg, out_dir / "backtest_test.png")
    typer.echo(f"\nSchéma : {figure}\nRésultats : {out_dir}  Registres : {ledger_dir}")
    if not all(run.verification.valid for run in runs):
        raise typer.Exit(code=1)


AGENTS_OUT_OPTION = typer.Option(
    None, "--out", help="Dossier des résultats (défaut : outputs/results/agents_<période>)."
)
FORECAST_DIR_OPTION = typer.Option(
    Path("outputs/results/forecast"), "--forecast-dir", help="Résultats de `train`."
)


class Period(StrEnum):
    val = "val"
    test = "test"


PERIOD_OPTION = typer.Option(Period.val, help="Période simulée (le test : phase 8).")


@app.command()
def simulate(
    config: Path = CONFIG_OPTION,
    period: Period = PERIOD_OPTION,
    out_dir: Path | None = AGENTS_OUT_OPTION,
    models_dir: Path = MODELS_DIR_OPTION,
    forecast_dir: Path = FORECAST_DIR_OPTION,
    ledger_dir: Path = LEDGER_DIR_OPTION,
    costs_factor: float = typer.Option(1.0, "--costs", help="Multiplie frais et slippage."),
    key: Path = PRIVATE_KEY_OPTION,
) -> None:
    """Agents (prévision, stratégie, risque, exécution) contre Buy & Hold : registre et schéma."""
    import time

    from gold_agents.data.prepare import FEATURES_FILE, load_prepared
    from gold_agents.data.update import sha256_file
    from gold_agents.evaluation import agents_report as ar
    from gold_agents.models.dataset import build_forecast_data

    cfg = load_config(config)
    path = cfg.data.processed_dir / FEATURES_FILE
    if not path.exists():
        typer.echo(f"{path} absent : lancer d'abord `uv run gold-agents prepare`.", err=True)
        raise typer.Exit(code=1)
    missing = [p for p in (models_dir / "lstm", models_dir / "gru") if not p.exists()]
    if missing or not (forecast_dir / ar.FORECAST_FILE).exists():
        typer.echo("Modèles de la phase 4 absents : lancer d'abord `uv run gold-agents train`.")
        raise typer.Exit(code=1)
    if period is Period.test:
        typer.echo("ATTENTION : le test est réservé à l'évaluation finale (phase 8).")
    split = period.value
    out = out_dir or Path(f"outputs/results/agents_{split}")
    ledgers = ledger_dir / f"agents_{split}"

    start_time = time.perf_counter()
    frame, scaler = load_prepared(cfg.data.processed_dir)
    data = build_forecast_data(frame, scaler, cfg.features.window)
    tables = ar.forecast_tables(cfg, data, split, models_dir, forecast_dir)
    start, end = ar.simulation_period(frame, split)
    st = cfg.strategy
    k = "sans filtre" if st.min_confidence is None else f"{st.min_confidence:g}"
    rule = "" if st.buy_rule == "absolute" else " (quantile des prévisions passées)"
    typer.echo(
        f"Période {split} : du {start} au {end}, frais x {costs_factor:g}. Stratégie : "
        f"θ_buy {st.theta_buy:g}{rule}, θ_sell {st.theta_sell:g}, confiance k = {k}."
    )
    typer.echo("\nPrévisions des modèles (agent de prévision) :")
    typer.echo(ar.diagnostics(tables, cfg).to_string())

    signer = _ledger_signer(key)
    results = ar.run_agent_backtests(
        cfg, frame, tables, split, sha256_file(path), out, ledgers, costs_factor, signer
    )
    typer.echo("\nRésultats :")
    typer.echo(ar.comparison_table(results).T.to_string())
    typer.echo("\nTrades par règle de l'agent de risque :")
    for r in results:
        if r.decisions is not None:
            rules = ", ".join(f"{k} {v}" for k, v in r.run.metrics["rules"].items())
            typer.echo(f"  {ar.AGENT_LABELS[r.run.result.strategy]:30s} {rules}")
    figure = ar.agents_figure(results, tables, cfg, split, out / f"agents_{split}.png")
    typer.echo(
        f"\nSchéma : {figure}\nRésultats : {out}  Registres : {ledgers}  "
        f"({time.perf_counter() - start_time:.0f} s)"
    )
    if not all(r.run.verification.valid for r in results):
        raise typer.Exit(code=1)


TUNE_OUT_OPTION = typer.Option(
    Path("outputs/results/tuning"), "--out", help="Dossier des résultats et du schéma."
)


@app.command()
def tune(
    config: Path = CONFIG_OPTION,
    out_dir: Path = TUNE_OUT_OPTION,
    models_dir: Path = MODELS_DIR_OPTION,
    forecast_dir: Path = FORECAST_DIR_OPTION,
    retrain: bool = typer.Option(
        False, "--retrain", help="Réentraîner les modèles même si les prévisions sont en cache."
    ),
    quick: bool = typer.Option(
        False, "--quick", help="Essai rapide : 1 graine, 5 époques (vérifie la chaîne)."
    ),
    apply: bool = typer.Option(
        False,
        "--apply",
        help="Écrire le réglage choisi dans config.yaml (sections strategy, risk).",
    ),
) -> None:
    """Phase 6 : réglage des agents sur une grille, en walk-forward (le test n'est pas lu)."""
    import hashlib
    import json
    import time

    import numpy as np
    import pandas as pd

    from gold_agents.agents.execution import ExecutionAgent
    from gold_agents.agents.forecast import ForecastAgent, ForecastTable
    from gold_agents.agents.orchestrator import AgentSystem, run_agents
    from gold_agents.agents.risk import RiskAgent
    from gold_agents.agents.strategy import StrategyAgent
    from gold_agents.data.prepare import FEATURES_FILE, load_prepared
    from gold_agents.data.update import sha256_file
    from gold_agents.evaluation import tuning as tu
    from gold_agents.evaluation import tuning_report as tr
    from gold_agents.evaluation.agents_report import FORECAST_FILE
    from gold_agents.models import walk_forward as wf
    from gold_agents.simulation.costs import CostModel

    cfg = load_config(config)
    if cfg.tuning is None:
        typer.echo("Section `tuning` absente de la configuration.", err=True)
        raise typer.Exit(code=1)
    tc = cfg.tuning
    path = cfg.data.processed_dir / FEATURES_FILE
    if not path.exists():
        typer.echo(f"{path} absent : lancer d'abord `uv run gold-agents prepare`.", err=True)
        raise typer.Exit(code=1)
    missing = [p for p in (models_dir / "lstm", models_dir / "gru") if not p.exists()]
    if missing or not (forecast_dir / FORECAST_FILE).exists():
        typer.echo("Modèles de la phase 4 absents : lancer d'abord `uv run gold-agents train`.")
        raise typer.Exit(code=1)
    start_time = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)

    frame, scaler = load_prepared(cfg.data.processed_dir)
    frame = wf.before_test(frame, cfg.splits.test_start)  # le test n'est jamais lu
    columns, window = list(scaler.columns), cfg.features.window
    net_meta = wf.read_network_meta(models_dir)
    lgbm = json.loads((forecast_dir / FORECAST_FILE).read_text(encoding="utf-8"))
    lgbm_params = lgbm["settings"].get("b4")
    mc = cfg.models
    seeds = tc.seeds[:1] if quick else tc.seeds
    epochs = 5 if quick else mc.max_epochs
    mc_samples = 10 if quick else mc.mc_samples
    key = {
        "data_sha256": sha256_file(path),
        "folds": [[str(a), str(b)] for a, b in tc.folds],
        "early_stop_days": tc.early_stop_days,
        "seeds": seeds,
        "max_epochs": epochs,
        "patience": mc.patience,
        "mc_samples": mc_samples,
        "networks": {k: {"net": v["net"], "train": v["train"]} for k, v in net_meta.items()},
        "lightgbm": lgbm_params,
    }
    cache, cache_meta = out_dir / "forecasts_walk_forward.parquet", out_dir / "forecasts_meta.json"
    cached = cache_meta.exists() and cache.exists()
    if cached and not retrain and json.loads(cache_meta.read_text("utf-8"))["key"] == key:
        forecasts = pd.read_parquet(cache)
        typer.echo(f"Prévisions walk-forward relues depuis {cache} (--retrain pour refaire).")
    else:
        typer.echo(
            f"Prévisions walk-forward : {len(tc.folds)} périodes, LSTM + GRU ({len(seeds)} "
            f"graine(s)) et LightGBM réentraînés sur le passé de chaque période."
        )
        forecasts, infos = wf.walk_forward_forecasts(
            frame,
            columns,
            window,
            list(tc.folds),
            tc.early_stop_days,
            net_meta,
            lgbm_params,
            seeds,
            epochs,
            mc.patience,
            mc_samples,
            mc.threads,
            log=typer.echo,
        )
        forecasts.to_parquet(cache, index=False)
        cache_meta.write_text(
            json.dumps({"key": key, "folds": infos}, indent=2, ensure_ascii=False), "utf-8"
        )
    diag = wf.forecast_diagnostics(forecasts)
    typer.echo("\nPrévisions hors échantillon (gain RMSE < 0 : mieux que le rendement nul) :")
    shown = diag.pivot(index="fold", columns="model", values=["direction", "gain RMSE vs B0"])
    typer.echo(shown.map(lambda v: f"{v:+.2%}" if abs(v) < 0.2 else f"{v:.1%}").to_string())
    share = diag.groupby("fold")["part de hausses"].first()
    typer.echo("Part de jours en hausse : " + "  ".join(f"{f}: {v:.1%}" for f, v in share.items()))

    sim_cfg = cfg.simulation
    st_cfg = cfg.strategy
    sim = tu.Simulator(
        CostModel(sim_cfg.fee_rate, sim_cfg.slippage),
        sim_cfg.initial_capital,
        cfg.risk.max_position,
        cfg.risk.cooldown_days,
        sim_cfg.risk_free_rate,
        tc.buy_rule,
        st_cfg.quantile_window,
        st_cfg.quantile_min_days,
    )
    limits = (tc.min_trades_per_year, tc.max_trades_per_year)
    markets = tu.fold_markets(frame, forecasts, list(tc.folds))
    grid = tu.combos(tc.grid)
    rule_text = (
        "θ_buy = rendement prévu minimal"
        if tc.buy_rule == "absolute"
        else f"θ_buy = quantile des {st_cfg.quantile_window} prévisions précédentes"
    )
    typer.echo(f"\nGrille : {len(grid)} réglages x {len(markets)} périodes ({rule_text}).")
    if tc.min_trades_per_year or tc.max_trades_per_year:
        typer.echo(
            f"Contrainte : entre {tc.min_trades_per_year:g} et "
            f"{tc.max_trades_per_year or '∞'} trades par an dans chaque période."
        )
    table, bh = tu.run_grid(markets, grid, sim)
    folds = [m.fold for m in markets]
    summary = tu.summarize_grid(table, tc.grid, folds, *limits)
    chosen = tu.best(summary)
    walk = tu.walk_forward(table, bh, tc.grid, markets, sim, *limits)
    costs = tu.costs_sensitivity(markets, chosen, sim, list(tc.costs_factors))
    refs, draws = tu.references(markets, chosen, sim, 20 if quick else tc.random_runs, cfg.seed)

    # Contrôle : le simulateur rapide et le vrai système d'agents donnent le même résultat.
    last = markets[-1]
    fc_days = [d for d, r in zip(last.dates, last.r_hat[chosen.model], strict=True) if r == r]
    ok = ~np.isnan(last.r_hat[chosen.model])
    sig = last.sigma_hat[chosen.model]
    table_fc = ForecastTable(
        chosen.model, fc_days, last.r_hat[chosen.model][ok], None if sig is None else sig[ok]
    )
    p = sim.params(chosen)
    system = AgentSystem(
        "agents_regles",
        ForecastAgent(table_fc),
        StrategyAgent(
            p.theta_buy,
            p.theta_sell,
            p.min_confidence,
            p.buy_rule,
            p.quantile_window,
            p.quantile_min_days,
        ),
        RiskAgent(p.max_position, p.stop_loss, p.max_drawdown, False, p.cooldown_days),
        ExecutionAgent(sim.costs),
    )
    full = run_agents(frame, system, sim.capital, last.dates[0], last.dates[-1])
    fast_value = float(sim.agents(last, chosen).values[-1])
    full_value = float(full.result.equity["value"].iloc[-1])
    same = abs(fast_value - full_value) < 1e-6 * sim.capital

    table.to_csv(out_dir / "grid.csv", index=False)
    summary.to_csv(out_dir / "grid_summary.csv", index=False)
    bh.to_csv(out_dir / "buy_and_hold.csv", index=False)
    walk.to_csv(out_dir / "walk_forward.csv", index=False)
    costs.to_csv(out_dir / "costs.csv", index=False)
    refs.to_csv(out_dir / "references.csv", index=False)
    diag.to_csv(out_dir / "forecast_diagnostics.csv", index=False)
    grid_sha = hashlib.sha256((out_dir / "grid.csv").read_bytes()).hexdigest()
    best_row = summary.iloc[0]
    settings = tr.chosen_settings(
        chosen,
        cfg.risk.cooldown_days,
        tc.buy_rule,
        st_cfg.quantile_window,
        st_cfg.quantile_min_days,
    )
    (out_dir / "chosen.json").write_text(
        json.dumps(
            {
                "chosen": settings,
                "trades_ok_folds": f"{int(best_row['trades_ok'])}/{len(folds)}",
                "trades_per_year": {
                    "min": tc.min_trades_per_year,
                    "max": tc.max_trades_per_year,
                },
                "mean_sharpe": float(best_row["sharpe"]),
                "plateau_sharpe": float(best_row["plateau"]),
                "score": float(best_row["score"]),
                "dd_ok_folds": f"{int(best_row['dd_ok'])}/{len(folds)}",
                "grid_sha256": grid_sha,
                "quick": quick,
            },
            indent=2,
            ensure_ascii=False,
        ),
        "utf-8",
    )
    figure = tr.tuning_figure(table, bh, walk, costs, refs, draws, chosen, out_dir / "tuning.png")

    pd.set_option("display.width", 200)
    typer.echo("\nBuy & Hold par période :")
    typer.echo(
        bh.assign(
            sharpe=bh["sharpe"].map(lambda v: f"{v:.2f}"),
            total_return=bh["total_return"].map(lambda v: f"{v:+.1%}"),
            worst_drawdown=bh["worst_drawdown"].map(lambda v: f"{v:.1%}"),
        )[["fold", "start", "end", "sharpe", "total_return", "worst_drawdown"]].to_string(
            index=False
        )
    )
    typer.echo("\n10 meilleurs réglages sur toutes les périodes :")
    typer.echo(tr.summary_text(summary).to_string(index=False))
    typer.echo("\nWalk-forward (chaque période jugée avec le réglage choisi avant elle) :")
    typer.echo(tr.walk_forward_text(walk).to_string())
    typer.echo("\nSensibilité aux frais (réglage figé) :")
    typer.echo(tr.costs_text(costs).to_string())
    typer.echo("\nRéférences (Sharpe, réglage figé) :")
    typer.echo(tr.references_text(refs, bh).to_string())
    typer.echo(
        f"\nContrôle moteur complet sur la dernière période : "
        f"{'identique' if same else 'DIFFÉRENT'} ({fast_value:.2f} / {full_value:.2f})"
    )
    typer.echo(f"\nRéglage choisi : {chosen.describe(tc.buy_rule)}")
    if apply:
        changed = apply_settings(config, settings)
        typer.echo(
            f"Écrit dans {config} : "
            + (", ".join(changed) if changed else "rien à changer (déjà à jour)")
            + ". Étape suivante : `uv run gold-agents freeze`."
        )
    else:
        typer.echo(
            f"À recopier dans {config} (ou relancer avec --apply) :\n"
            + tr.yaml_snippet(chosen, cfg.risk.cooldown_days, tc.buy_rule)
        )
    typer.echo(
        f"Schéma : {figure}\nRésultats : {out_dir}  ({time.perf_counter() - start_time:.0f} s)"
    )
    if not same:
        raise typer.Exit(code=1)


TUNING_DIR_OPTION = typer.Option(
    Path("outputs/results/tuning"), "--tuning-dir", help="Résultats de `tune`."
)


MOVES_OPTION = typer.Option(
    Path("ledger/moves.json"),
    "--moves",
    help="Journal des registres archivés (versionné) : l'ancrage reste vérifiable.",
)


@app.command()
def freeze(
    config: Path = CONFIG_OPTION,
    tuning_dir: Path = TUNING_DIR_OPTION,
    ledger_dir: Path = LEDGER_DIR_OPTION,
    key: Path = PRIVATE_KEY_OPTION,
    again: str = typer.Option(
        "", "--again", help="Refiger des réglages déjà figés : raison, inscrite dans le registre."
    ),
    moves_path: Path = MOVES_OPTION,
) -> None:
    """Fige les réglages : les compare à `tune`, puis les scelle dans un bloc genèse."""
    import hashlib
    import json

    from gold_agents.blockchain.archive import archive_file
    from gold_agents.blockchain.chain import Chain
    from gold_agents.blockchain.storage import read_jsonl, write_jsonl
    from gold_agents.data.prepare import FEATURES_FILE
    from gold_agents.data.update import sha256_file
    from gold_agents.evaluation import final as fe

    cfg = load_config(config)
    chosen_path = tuning_dir / "chosen.json"
    if not chosen_path.exists():
        typer.echo(f"{chosen_path} absent : lancer d'abord `uv run gold-agents tune`.", err=True)
        raise typer.Exit(code=1)
    chosen = json.loads(chosen_path.read_text(encoding="utf-8"))
    if chosen.get("quick"):
        typer.echo(
            "Les résultats viennent de `tune --quick` : relancer `tune` en entier.", err=True
        )
        raise typer.Exit(code=1)
    current = fe.current_settings(cfg)
    keys = set(current) | set(chosen["chosen"])
    diff = {
        k: (current.get(k), chosen["chosen"].get(k))
        for k in sorted(keys)
        if current.get(k) != chosen["chosen"].get(k)
    }
    if diff:
        for k, (a, b) in diff.items():
            typer.echo(f"  {k} : config {a!r}, tune {b!r}", err=True)
        typer.echo(
            "La configuration ne correspond pas au réglage choisi par `tune` "
            "(`uv run gold-agents tune --apply` l'écrit dans la configuration).",
            err=True,
        )
        raise typer.Exit(code=1)
    grid = tuning_dir / "grid.csv"
    if hashlib.sha256(grid.read_bytes()).hexdigest() != chosen["grid_sha256"]:
        typer.echo(f"{grid} a changé depuis `tune`.", err=True)
        raise typer.Exit(code=1)

    # Des réglages déjà figés ne sont jamais écrasés : archivés, et cités dans le nouveau.
    target = ledger_dir / "reglages_figes.jsonl"
    previous: list[dict[str, object]] = []
    version = 1
    if target.exists():
        if not again:
            typer.echo(
                f"{target} existe déjà : des réglages sont figés. Pour en figer de nouveaux, "
                '--again "raison" (l\'ancien registre est archivé et cité).',
                err=True,
            )
            raise typer.Exit(code=1)
        old = read_jsonl(target)
        old_genesis = old.blocks[0].events[0]
        version = int(old_genesis.get("version", 1)) + 1
        kept = archive_file(
            target, target.with_name(f"reglages_figes_v{version - 1}.jsonl"), moves_path
        )
        previous = [
            *old_genesis.get("previous_versions", []),
            {"file": kept.as_posix(), "head": old.head.hash, "settings": old_genesis["settings"]},
        ]
    events = [
        {
            "type": "GENESIS",
            "kind": "FROZEN_SETTINGS",
            "version": version,
            "previous_versions": previous,
            "reason_again": again or None,
            "frozen_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "settings": current,
            "config": json.loads(cfg.model_dump_json()),
            "config_sha256": sha256_file(config),
            "data_sha256": sha256_file(cfg.data.processed_dir / FEATURES_FILE),
            "tuning": {"grid_sha256": chosen["grid_sha256"], **chosen},
        }
    ]
    signer = _ledger_signer(key)
    path = write_jsonl(Chain.start(events, signer), target)
    chain = read_jsonl(path)
    verification = chain.verify(signer.public_key if signer else None)
    typer.echo(f"Réglages figés (version {version}) dans {path} : {verification.describe()}.")
    typer.echo(f"Empreinte du bloc genèse : {chain.head.hash}")
    if version > 1:
        typer.echo(
            "Pour lancer le test avec ces réglages :\n"
            f'  uv run gold-agents evaluate --frozen-head {chain.head.hash} --again "raison"'
        )


LEDGERS_ARGUMENT = typer.Argument(None, help="Fichiers .jsonl (défaut : outputs/ledger/).")
PUBLIC_KEY_OPTION = typer.Option(
    Path("ledger/public_key.pem"), "--public-key", help="Clé publique Ed25519 publiée."
)
ENV_OPTION = typer.Option(Path(".env"), "--env", help="Fichier .env du portefeuille de test.")
ANCHORS_OPTION = typer.Option(
    Path("ledger/anchors.json"), "--anchors", help="Journal des ancrages (versionné)."
)


def _git_state() -> dict[str, object]:
    import subprocess

    def git(*args: str) -> str:
        try:
            done = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
        except (OSError, subprocess.CalledProcessError):
            return ""
        return done.stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD") or None,
        "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
    }


def _anchor_ledgers(paths: list[Path], env: Path, anchors_path: Path, public_key: Path) -> str:
    """Vérifie puis ancre les registres sur Sepolia ; renvoie le lien Etherscan."""
    from gold_agents.blockchain import anchor as anchoring
    from gold_agents.blockchain.signing import load_public_key
    from gold_agents.blockchain.storage import read_jsonl

    key = load_public_key(public_key) if public_key.exists() else None
    heads = []
    for path in paths:
        chain = read_jsonl(path)
        result = chain.verify(key)
        if not result.valid:
            raise RuntimeError(f"{path} : {result.describe()}")
        heads.append(anchoring.LedgerHead(path.as_posix(), len(chain), chain.head.hash))
    values = anchoring.read_env(env)
    typer.echo(
        f"Ancrage sur Sepolia de {len(heads)} registre(s), empreinte "
        f"{anchoring.anchor_digest(heads)[:16]}… (jusqu'à quelques minutes)"
    )
    done = anchoring.send_anchor(heads, values["SEPOLIA_RPC_URL"], values["SEPOLIA_PRIVATE_KEY"])
    anchoring.append_anchor(done, anchors_path)
    url = anchoring.etherscan_url(done.tx_hash)
    typer.echo(f"Ancré dans le bloc {done.block_number} : {url}")
    return url


FINAL_OUT_OPTION = typer.Option(None, "--out", help="Dossier des résultats.")
FINAL_LEDGER_OPTION = typer.Option(None, "--ledger-dir", help="Dossier des registres.")
FROZEN_OPTION = typer.Option(
    Path("outputs/ledger/reglages_figes.jsonl"), "--frozen", help="Registre des réglages figés."
)


@app.command()
def evaluate(
    config: Path = CONFIG_OPTION,
    rehearsal: bool = typer.Option(
        False, "--rehearsal", help="Répétition sur la validation : le test n'est pas lu."
    ),
    quick: bool = typer.Option(
        False, "--quick", help="Avec --rehearsal : 1 graine, 5 époques (vérifie la chaîne)."
    ),
    models_dir: Path = MODELS_DIR_OPTION,
    forecast_dir: Path = FORECAST_DIR_OPTION,
    tuning_dir: Path = TUNING_DIR_OPTION,
    frozen: Path = FROZEN_OPTION,
    frozen_head: str = typer.Option(
        "", "--frozen-head", help="Hash attendu du bloc genèse figé (défaut : celui du 30/10)."
    ),
    out_dir: Path | None = FINAL_OUT_OPTION,
    ledger_dir: Path | None = FINAL_LEDGER_OPTION,
    key: Path = PRIVATE_KEY_OPTION,
    public_key: Path = PUBLIC_KEY_OPTION,
    env: Path = ENV_OPTION,
    anchors_path: Path = ANCHORS_OPTION,
    no_anchor: bool = typer.Option(False, "--no-anchor", help="Ne pas ancrer sur Sepolia."),
    again: str = typer.Option(
        "", "--again", help="Relancer un test déjà lancé : raison, inscrite dans le registre."
    ),
    moves_path: Path = MOVES_OPTION,
) -> None:
    """Phase 8 : un seul run sur le test, toutes les stratégies, registres signés et ancrés."""
    import json
    import time

    import pandas as pd

    from gold_agents.blockchain.chain import Chain
    from gold_agents.blockchain.storage import read_jsonl, write_jsonl
    from gold_agents.data.prepare import FEATURES_FILE, load_prepared
    from gold_agents.data.update import sha256_file
    from gold_agents.evaluation import final as fe
    from gold_agents.evaluation.agents_report import FORECAST_FILE

    cfg = load_config(config)
    split = "val" if rehearsal else "test"
    if quick and not rehearsal:
        typer.echo("--quick n'existe qu'avec --rehearsal : le test se lance en entier.", err=True)
        raise typer.Exit(code=1)
    out = out_dir or Path(f"outputs/results/final_{split}")
    ledgers = ledger_dir or Path(
        "outputs/ledger/test" if not rehearsal else "outputs/ledger/final_val"
    )
    anchor = not (no_anchor or rehearsal)
    title = (
        "Phase 8 : évaluation finale sur le test"
        if not rehearsal
        else ("Phase 8, répétition sur la validation")
    )

    # 1. Contrôles, avant de lire le moindre prix de la période évaluée.
    path = cfg.data.processed_dir / FEATURES_FILE
    if not path.exists():
        typer.echo(f"{path} absent : lancer d'abord `uv run gold-agents prepare`.", err=True)
        raise typer.Exit(code=1)
    missing = [p for p in (models_dir / "lstm", models_dir / "gru") if not p.exists()]
    if missing or not (forecast_dir / FORECAST_FILE).exists():
        typer.echo("Modèles de la phase 4 absents : lancer d'abord `uv run gold-agents train`.")
        raise typer.Exit(code=1)
    if cfg.tuning is None:
        typer.echo("Section `tuning` absente de la configuration.", err=True)
        raise typer.Exit(code=1)
    data_sha = sha256_file(path)
    check = fe.check_frozen(cfg, frozen, frozen_head or fe.FROZEN_HEAD, data_sha)
    from gold_agents.models import walk_forward as wf

    net_meta = wf.read_network_meta(models_dir)
    lgbm_params = json.loads((forecast_dir / FORECAST_FILE).read_text("utf-8"))["settings"].get(
        "b4"
    )
    procedure = fe.check_procedure(tuning_dir / "forecasts_meta.json", net_meta, lgbm_params)
    if check.ok:
        typer.echo(
            f"Réglages figés vérifiés : bloc genèse {check.head[:16]}…, config et données "
            "identiques au gel."
        )
    for problem in check.problems + procedure:
        typer.echo(f"  {'REFUS' if not rehearsal else 'attention'} : {problem}", err=True)
    blocking = check.problems + [p for p in procedure if "différents" in p]
    if blocking and not rehearsal:
        typer.echo("Le test n'est lancé qu'avec les réglages figés : rien n'a été lu.", err=True)
        raise typer.Exit(code=1)
    signer = _ledger_signer(key)
    if signer is None and not rehearsal:
        typer.echo("Le test exige des registres signés : `uv run gold-agents keygen`.", err=True)
        raise typer.Exit(code=1)
    if anchor:
        from gold_agents.blockchain import anchor as anchoring

        try:
            anchoring.read_env(env)
            import web3  # noqa: F401
        except KeyError as exc:
            typer.echo(f"{exc.args[0]} (ou --no-anchor).", err=True)
            raise typer.Exit(code=1) from exc
        except ImportError as exc:
            typer.echo("web3 absent : `uv sync --all-extras` (ou --no-anchor).", err=True)
            raise typer.Exit(code=1) from exc

    # 2. Un seul run : un plan déjà écrit veut dire que le test a déjà été lancé.
    # Une nouvelle tentative n'efface rien : les registres de la précédente sont archivés
    # dans `<dossier>_tentativeN/` (déplacements notés dans ledger/moves.json, l'ancrage
    # reste vérifiable) et cités dans le nouveau plan avec la raison.
    from gold_agents.blockchain.archive import archive_file

    plan_path = ledgers / fe.PLAN_FILE
    archived = sorted(
        ledgers.parent.glob(f"{ledgers.name}_tentative*"),
        key=lambda d: int(d.name.rsplit("tentative", 1)[1] or 0),
    )
    previous: list[dict[str, object]] = []
    for d in archived:
        if (d / fe.PLAN_FILE).exists():
            previous.append(
                {"ledgers": d.as_posix(), "head": read_jsonl(d / fe.PLAN_FILE).head.hash}
            )
    if (plan_path.exists() or previous) and not rehearsal:
        if not again:
            done = len(previous) + plan_path.exists()
            typer.echo(
                f"Le test a déjà été lancé ({done} tentative(s)). Le principe : un seul run. "
                'Pour relancer quand même, --again "raison" ; les registres de la tentative '
                "précédente sont archivés, et la raison est inscrite dans le nouveau plan.",
                err=True,
            )
            raise typer.Exit(code=1)
        if plan_path.exists():
            n = len(previous) + 1
            kept_dir = ledgers.with_name(f"{ledgers.name}_tentative{n}")
            head = read_jsonl(plan_path).head.hash
            for f in sorted(ledgers.glob("*.jsonl")):
                archive_file(f, kept_dir / f.name, moves_path)
            if out.is_dir():
                out.replace(out.with_name(f"{out.name}_tentative{n}"))
            previous.append({"ledgers": kept_dir.as_posix(), "head": head})
            typer.echo(f"Tentative {n} archivée dans {kept_dir} (rien n'est effacé).")

    start_time = time.perf_counter()
    frame, scaler = load_prepared(cfg.data.processed_dir)
    start, end = fe.split_bounds(frame, cfg, split)
    tc, mc = cfg.tuning, cfg.models
    seeds = tc.seeds[:1] if quick else list(tc.seeds)
    epochs = 5 if quick else mc.max_epochs
    mc_samples = 10 if quick else mc.mc_samples
    runs = 20 if quick else tc.random_runs
    factors = list(tc.costs_factors)
    plan = {
        "type": "GENESIS",
        "kind": "TEST_PLAN" if not rehearsal else "REHEARSAL_PLAN",
        "attempt": len(previous) + 1,
        "previous_attempts": previous,
        "reason_again": again or None,
        "planned_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "period": {"split": split, "start": str(start), "end": str(end)},
        "frozen_settings": {"file": frozen.as_posix(), "head": check.head, "ok": check.ok},
        "settings": fe.current_settings(cfg),
        "config": json.loads(cfg.model_dump_json()),
        "data_sha256": data_sha,
        "code": _git_state(),
        "procedure": {
            "forecasts": "modèles réentraînés sur tout ce qui précède le début de la période "
            "(walk-forward de la phase 6)",
            "seeds": seeds,
            "max_epochs": epochs,
            "patience": mc.patience,
            "mc_samples": mc_samples,
            "early_stop_days": tc.early_stop_days,
            "networks": {k: {"net": v["net"], "train": v["train"]} for k, v in net_meta.items()},
            "lightgbm": lgbm_params,
        },
        "strategies": [
            "buy_and_hold",
            "sma_20_50",
            "agents_lstm",
            "agents_gru",
            "agents_ensemble",
            fe.MAIN,
            "agents_lightgbm",
            fe.RANDOM,
            fe.ORACLE,
        ],
        "main_system": fe.MAIN,
        "costs_factors": factors,
        "random_runs": runs,
        "alerts": {"direction": mc.direction_alert, "sharpe": fe.SHARPE_ALERT},
        "anchored": anchor,
        "quick": quick,
    }
    plan_chain = Chain.start([fe.clean_json(plan)], signer)
    write_jsonl(plan_chain, plan_path)
    typer.echo(f"Plan écrit avant de lire la période : {plan_path} ({plan_chain.head.hash[:16]}…)")
    if anchor:
        try:
            _anchor_ledgers([plan_path, frozen], env, anchors_path, public_key)
        except Exception as exc:
            plan_path.unlink()  # rien n'a été lu : on peut réessayer proprement
            typer.echo(f"Ancrage du plan impossible, rien n'a été lu : {exc}", err=True)
            raise typer.Exit(code=1) from exc

    # 3. Prévisions : modèles appris sur tout ce qui précède la période.
    typer.echo(
        f"\nPériode {split} : du {start} au {end}. Prévisions : LSTM + GRU "
        f"({len(seeds)} graine(s)) et LightGBM réentraînés sur le passé (avant le {start})."
    )
    columns, window = list(scaler.columns), cfg.features.window
    if mc.threads:
        import torch

        torch.set_num_threads(mc.threads)
    forecasts, info = fe.final_forecasts(
        frame,
        columns,
        window,
        start,
        end,
        tc.early_stop_days,
        net_meta,
        lgbm_params,
        seeds,
        epochs,
        mc.patience,
        mc_samples,
        mc.threads,
    )
    typer.echo(
        f"  appris sur {info['n_fit']} jours (+{info['n_early_stop']} pour l'arrêt anticipé), "
        f"{info['n_predicted']} jours prédits ({time.perf_counter() - start_time:.0f} s)"
    )
    tables = fe.tables_from(forecasts)
    out.mkdir(parents=True, exist_ok=True)
    forecasts.to_csv(out / "forecasts.csv", index=False)

    # 4. Le run : toutes les stratégies, registres signés.
    outcomes = fe.run_all(
        cfg,
        frame,
        tables,
        start,
        end,
        split,
        data_sha,
        plan_chain.head.hash,
        out,
        ledgers,
        signer,
        1.0,
        cfg.seed,
    )
    costs = fe.costs_sensitivity(cfg, frame, tables, start, end, split, factors, cfg.seed)
    refs, draws = fe.random_draws(cfg, outcomes, frame, runs, cfg.seed)
    fc = fe.forecast_table(forecasts)
    yearly = fe.yearly_returns(outcomes)
    warnings = fe.alerts(outcomes, fc, mc.direction_alert)
    table = fe.final_table(outcomes, refs, costs)

    costs.to_csv(out / "costs.csv", index=False)
    refs.to_csv(out / "random_references.csv", index=False)
    draws.to_csv(out / "random_draws.csv", index=False)
    fc.to_csv(out / "forecast_metrics.csv", index=False)
    yearly.to_csv(out / "yearly_returns.csv")
    table.to_csv(out / "final_table.csv")
    (out / "final_table.md").write_text(
        f"# {title}\n\nDu {outcomes[0].run.metrics['start']} au "
        f"{outcomes[0].run.metrics['end']}, frais x1.\n\n"
        + fe.markdown(table)
        + "\n## Sensibilité aux frais (rendement total)\n\n"
        + fe.markdown(fe.costs_text(costs))
        + "\n## Précision des prévisions\n\n"
        + fe.markdown(fe.forecast_text(fc))
        + "\n## Rendement par année\n\n"
        + fe.markdown(fe.yearly_text(yearly))
        + "\n## Alertes « trop beau »\n\n"
        + ("\n".join(f"- {w}" for w in warnings) if warnings else "Aucune.")
        + "\n",
        encoding="utf-8",
    )
    metrics = [o.run.metrics for o in outcomes]
    (out / "metrics.json").write_text(
        json.dumps(
            fe.clean_json(
                {
                    "plan_head": plan_chain.head.hash,
                    "info": info,
                    "metrics": metrics,
                    "alerts": warnings,
                }
            ),
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )
    fig1 = fe.final_figure(outcomes, table, cfg, title, out / f"final_{split}.png")
    fig2 = fe.robustness_figure(
        costs, refs, draws, fc, yearly, mc.direction_alert, title, out / f"robustesse_{split}.png"
    )

    # 5. Registre des résultats : tableaux, alertes et empreinte de chaque fichier.
    files = sorted(p for p in out.iterdir() if p.is_file())
    strategy_ledgers = [o.run.ledger_path for o in outcomes]
    result_events = [
        {
            "type": "GENESIS",
            "kind": "TEST_RESULTS" if not rehearsal else "REHEARSAL_RESULTS",
            "plan_head": plan_chain.head.hash,
            "ledgers": [
                {"file": p.as_posix(), "head": read_jsonl(p).head.hash} for p in strategy_ledgers
            ],
            "files_sha256": fe.file_digests(files),
            "forecast_info": info,
        },
        {"type": "METRICS", "metrics": metrics},
        {"type": "COSTS_SENSITIVITY", "rows": fe.records(costs)},
        {"type": "RANDOM_REFERENCES", "rows": fe.records(refs)},
        {"type": "FORECAST_METRICS", "rows": fe.records(fc)},
        {"type": "ALERTS", "alerts": warnings},
    ]
    results_path = write_jsonl(
        Chain.start(fe.clean_json(json.loads(json.dumps(result_events, default=str))), signer),
        ledgers / fe.RESULTS_FILE,
    )

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    typer.echo("\nPrévisions sur la période (bonne direction > 60 % : alerte) :")
    typer.echo(fe.forecast_text(fc).to_string())
    typer.echo(f"Part de jours en hausse : {fc['share_up'].iloc[0]:.1%}")
    typer.echo("\nTableau comparatif (frais x1) :")
    typer.echo(table.T.to_string())
    typer.echo("\nSensibilité aux frais (rendement total) :")
    typer.echo(fe.costs_text(costs).to_string())
    typer.echo("\nRendement par année :")
    typer.echo(fe.yearly_text(yearly).to_string())
    typer.echo("\nTrades par règle de l'agent de risque :")
    for o in outcomes:
        if o.decisions is not None:
            rules = ", ".join(f"{k} {v}" for k, v in o.run.metrics["rules"].items())
            typer.echo(f"  {fe.LABELS.get(o.run.result.strategy, '')[:32]:32s} {rules}")
    typer.echo("\nAlertes « trop beau » :")
    typer.echo("\n".join(f"  ALERTE : {w}" for w in warnings) if warnings else "  aucune")

    all_ledgers = [plan_path, *strategy_ledgers, results_path]
    key_pub = signer.public_key if signer else None
    typer.echo("\nRegistres :")
    ok = True
    for p in all_ledgers:
        v = read_jsonl(p).verify(key_pub)
        ok &= v.valid
        typer.echo(f"  {p} : {v.describe()}")
    if anchor:
        try:
            _anchor_ledgers(all_ledgers, env, anchors_path, public_key)
        except Exception as exc:
            typer.echo(
                f"Ancrage final impossible ({exc}). Relancer : uv run gold-agents anchor "
                f"{' '.join(p.as_posix() for p in all_ledgers)}",
                err=True,
            )
            ok = False
    typer.echo(
        f"\nSchémas : {fig1}  {fig2}\nRésultats : {out}  Registres : {ledgers}  "
        f"({time.perf_counter() - start_time:.0f} s)"
    )
    if not ok:
        raise typer.Exit(code=1)


@app.command()
def keygen(
    key: Path = PRIVATE_KEY_OPTION,
    public_key: Path = PUBLIC_KEY_OPTION,
    force: bool = typer.Option(False, "--force", help="Remplacer une clé existante."),
) -> None:
    """Crée la paire de clés Ed25519 du registre (la clé privée reste sur ce PC)."""
    from gold_agents.blockchain.signing import fingerprint, generate_keypair

    try:
        signer = generate_keypair(key, public_key, overwrite=force)
    except FileExistsError as exc:
        typer.echo(f"{exc}. Utiliser --force pour la remplacer (registres à resigner).", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Clé privée : {key}  (NE JAMAIS la versionner ni la partager)")
    typer.echo(f"Clé publique : {public_key}  (à versionner : elle sert à vérifier)")
    typer.echo(f"Empreinte de la clé publique : {fingerprint(signer.public_key)}")


@app.command()
def verify(
    ledgers: list[Path] | None = LEDGERS_ARGUMENT,
    public_key: Path = PUBLIC_KEY_OPTION,
) -> None:
    """Vérifie les registres : liens, hash, Merkle, preuve de travail et signatures."""
    from gold_agents.blockchain.signing import fingerprint, load_public_key
    from gold_agents.blockchain.storage import read_jsonl

    paths = ledgers or sorted(Path("outputs/ledger").rglob("*.jsonl"))
    if not paths:
        typer.echo("Aucun registre trouvé : lancer d'abord `make backtest`.", err=True)
        raise typer.Exit(code=1)
    key = load_public_key(public_key) if public_key.exists() else None
    if key is not None:
        typer.echo(f"Clé publique {public_key} ({fingerprint(key)}).")
    else:
        typer.echo(f"{public_key} absente : signatures non vérifiées.")
    ok = True
    for path in paths:
        result = read_jsonl(path).verify(key)
        ok &= result.valid
        typer.echo(f"{path} : {result.describe()}")
    if not ok:
        raise typer.Exit(code=1)


@app.command()
def anchor(
    ledgers: list[Path] | None = LEDGERS_ARGUMENT,
    public_key: Path = PUBLIC_KEY_OPTION,
    env: Path = ENV_OPTION,
    anchors_path: Path = ANCHORS_OPTION,
) -> None:
    """Ancre le dernier hash des registres sur Sepolia (bonus, extra `anchor`)."""
    from gold_agents.blockchain import anchor as anchoring
    from gold_agents.blockchain.signing import load_public_key
    from gold_agents.blockchain.storage import read_jsonl

    paths = ledgers or sorted(Path("outputs/ledger").rglob("*.jsonl"))
    if not paths:
        typer.echo("Aucun registre trouvé.", err=True)
        raise typer.Exit(code=1)
    key = load_public_key(public_key) if public_key.exists() else None
    heads = []
    for path in paths:
        chain = read_jsonl(path)
        result = chain.verify(key)
        if not result.valid:  # on n'ancre jamais un registre invalide
            typer.echo(f"{path} : {result.describe()}", err=True)
            raise typer.Exit(code=1)
        heads.append(anchoring.LedgerHead(path.as_posix(), len(chain), chain.head.hash))
        typer.echo(f"{path} : {result.describe()}")
    try:
        values = anchoring.read_env(env)
    except KeyError as exc:
        typer.echo(f"{exc.args[0]}. Voir la section « Ancrage Sepolia » du README.", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"\nEmpreinte à ancrer : {anchoring.anchor_digest(heads)}")
    typer.echo("Envoi de la transaction sur Sepolia (jusqu'à quelques minutes)...")
    done = anchoring.send_anchor(heads, values["SEPOLIA_RPC_URL"], values["SEPOLIA_PRIVATE_KEY"])
    path = anchoring.append_anchor(done, anchors_path)
    typer.echo(f"Ancré dans le bloc {done.block_number} : {anchoring.etherscan_url(done.tx_hash)}")
    typer.echo(f"Ancrage ajouté à {path} (à versionner).")


# --- Simulation en direct (paper trading) ---------------------------------------------

LIVE_LEDGER_OPTION = typer.Option(
    Path("outputs/ledger/live"), "--ledger-dir", help="Dossier du registre en direct."
)
LIVE_MODELS_OPTION = typer.Option(
    Path("outputs/models/live"), "--live-models", help="Modèles figés de la simulation en direct."
)


def _rule_text(s: dict[str, object]) -> str:
    if s.get("buy_rule") == "quantile":
        return (
            f"achat si r̂ dépasse le quantile {s['theta_buy']:.0%} des "
            f"{s['quantile_window']} prévisions précédentes, vente si r̂ < -{s['theta_sell']:g}"
        )
    return f"achat si r̂ > {s['theta_buy']:g}, vente si r̂ < -{s['theta_sell']:g}"


def _live_signer(key: Path) -> Signer:
    signer = _ledger_signer(key)
    if signer is None:
        typer.echo("La simulation en direct exige un registre signé.", err=True)
        raise typer.Exit(code=1)
    return signer


def _check_anchor_env(env: Path) -> None:
    from gold_agents.blockchain import anchor as anchoring

    try:
        anchoring.read_env(env)
        import web3  # noqa: F401
    except KeyError as exc:
        typer.echo(f"{exc.args[0]} (ou --no-anchor).", err=True)
        raise typer.Exit(code=1) from exc
    except ImportError as exc:
        typer.echo("web3 absent : `uv sync --all-extras` (ou --no-anchor).", err=True)
        raise typer.Exit(code=1) from exc


def _live_report(reports: Sequence[object]) -> None:
    from gold_agents.live.session import DayReport

    labels = {"BUY": "ACHAT", "SELL": "VENTE", "HOLD": "attente"}
    for r in reports:
        assert isinstance(r, DayReport)
        done = f", exécuté : {r.trade.side} à {r.trade.price:,.2f}" if r.trade else ""
        seuil = "" if r.threshold is None else f" (seuil d'achat {r.threshold:+.3%})"
        r_hat = "pas de prévision" if r.r_hat is None else f"r̂ {r.r_hat:+.3%}{seuil}"
        action = labels.get(r.order.action.value, r.order.action.value)
        if not r.order.executable:
            action = "attente"
        typer.echo(
            f"  {r.day} {'à temps   ' if r.on_time else 'EN RETARD '} {r_hat} -> {action} "
            f"({r.order.rule}){done}. Valeur {r.value:,.0f} $, Buy & Hold {r.bh_value:,.0f} $"
        )


def _save_live(
    chain: object,
    path: Path,
    public_key: Path,
    anchor: bool,
    env: Path,
    anchors_path: Path,
) -> None:
    from gold_agents.blockchain.chain import Chain
    from gold_agents.blockchain.signing import load_public_key
    from gold_agents.blockchain.storage import read_jsonl, write_jsonl

    assert isinstance(chain, Chain)
    write_jsonl(chain, path)
    key = load_public_key(public_key) if public_key.exists() else None
    result = read_jsonl(path).verify(key)
    typer.echo(f"Registre {path} : {result.describe()}.")
    if not result.valid:
        raise typer.Exit(code=1)
    if not anchor:
        typer.echo("Pas d'ancrage Sepolia (--no-anchor).")
        return
    try:
        _anchor_ledgers([path], env, anchors_path, public_key)
    except Exception as exc:
        typer.echo(
            f"Ancrage impossible ({exc}). Le registre est écrit : relancer l'ancrage avec "
            f"`uv run gold-agents anchor {path.as_posix()}` avant la prochaine clôture.",
            err=True,
        )
        raise typer.Exit(code=1) from exc


@app.command("live-init")
def live_init(
    config: Path = CONFIG_OPTION,
    source: UpdateSource = SOURCE_OPTION,
    no_update: bool = typer.Option(False, "--no-update", help="Ne pas appeler l'API avant."),
    models_dir: Path = MODELS_DIR_OPTION,
    live_models: Path = LIVE_MODELS_OPTION,
    ledger_dir: Path = LIVE_LEDGER_OPTION,
    frozen: Path = FROZEN_OPTION,
    key: Path = PRIVATE_KEY_OPTION,
    public_key: Path = PUBLIC_KEY_OPTION,
    env: Path = ENV_OPTION,
    anchors_path: Path = ANCHORS_OPTION,
    no_anchor: bool = typer.Option(False, "--no-anchor", help="Ne pas ancrer sur Sepolia."),
    quick: bool = typer.Option(False, "--quick", help="1 graine, 5 époques (vérifie la chaîne)."),
    again: str = typer.Option(
        "", "--again", help="Repartir de zéro : raison, inscrite dans le nouveau registre."
    ),
) -> None:
    """Démarre la simulation en direct : modèles figés, plan signé et ancré, premier jour."""
    import time

    from gold_agents.blockchain.chain import Chain
    from gold_agents.blockchain.storage import read_jsonl
    from gold_agents.config import LiveConfig
    from gold_agents.evaluation.final import current_settings
    from gold_agents.live import session as ls
    from gold_agents.models import walk_forward as wf

    cfg = load_config(config)
    live = cfg.live or LiveConfig()
    if quick:
        live = live.model_copy(update={"seeds": live.seeds[:1]})
    path = ledger_dir / ls.LEDGER_FILE
    previous: list[dict[str, object]] = []
    if path.exists():
        if not again:
            typer.echo(
                f"La simulation en direct a déjà démarré ({path}). Chaque jour : "
                '`uv run gold-agents live`. Pour repartir de zéro : --again "raison".',
                err=True,
            )
            raise typer.Exit(code=1)
        old = read_jsonl(path)
        n = len(list(ledger_dir.glob("agents_live_tentative*.jsonl"))) + 1
        kept = path.with_name(f"agents_live_tentative{n}.jsonl")
        path.replace(kept)
        previous = [{"ledger": kept.as_posix(), "head": old.head.hash, "blocks": len(old)}]
    if not all((models_dir / k / "model.json").exists() for k in ls.KINDS):
        typer.echo("Hyperparamètres de la phase 4 absents : `uv run gold-agents train`.", err=True)
        raise typer.Exit(code=1)
    signer = _live_signer(key)
    anchor = not no_anchor
    if anchor:
        _check_anchor_env(env)
    # Une seule règle gelée : celle du dernier `freeze`, la même que pour le test.
    if not frozen.exists():
        typer.echo(f"{frozen} absent : `uv run gold-agents freeze` d'abord.", err=True)
        raise typer.Exit(code=1)
    frozen_chain = read_jsonl(frozen)
    frozen_head = frozen_chain.head.hash
    frozen_settings = dict(frozen_chain.blocks[0].events[0].get("settings", {}))
    diff = {
        k: (v, frozen_settings.get(k))
        for k, v in current_settings(cfg).items()
        if frozen_settings.get(k) != v
    }
    if diff:
        for k, (now, then) in diff.items():
            typer.echo(f"  REFUS : {k} vaut {now!r} dans la config, {then!r} au gel.", err=True)
        typer.echo("La règle de la config n'est pas celle du dernier gel : rien n'a démarré.")
        raise typer.Exit(code=1)

    if not no_update:
        _update(cfg, source)
    frame = ls.live_frame(cfg)
    last = frame["date"].iloc[-1].date()
    typer.echo(
        f"\nDonnées jusqu'au {last} ({len(frame)} jours). Entraînement des modèles figés : "
        f"LSTM + GRU, {len(live.seeds)} graine(s), sur tout ce qui précède."
    )
    start = time.perf_counter()
    if live_models.exists():
        import shutil

        shutil.rmtree(live_models)
    training = ls.train_models(
        frame,
        cfg,
        live,
        wf.read_network_meta(models_dir),
        live_models,
        5 if quick else cfg.models.max_epochs,
        cfg.models.patience,
        log=typer.echo,
    )
    typer.echo(f"Modèles figés dans {live_models} ({time.perf_counter() - start:.0f} s).")
    models = ls.load_models(live_models, cfg.features.window)
    now = datetime.now(UTC)
    event, state = ls.plan_event(
        cfg,
        live,
        {"file": frozen.as_posix(), "settings": frozen_settings},
        frame,
        models,
        live_models,
        training,
        _git_state(),
        frozen_head,
        now,
    )
    event.update(previous_attempts=previous, reason_again=again or None, quick=quick)
    chain = Chain.start([event, state.event()], signer)
    s = event["settings"]
    typer.echo(
        f"Plan scellé avec la règle du gel {frozen_head[:12]}… : {_rule_text(s)}, "
        f"stop-loss -{s['stop_loss']:.0%}, drawdown max -{s['max_drawdown']:.0%}, "
        f"capital {s['initial_capital']:,.0f} $."
    )
    reports = ls.advance(chain, frame, models, now)
    typer.echo("\nPremier jour :")
    _live_report(reports)
    _save_live(chain, path, public_key, anchor, env, anchors_path)
    typer.echo(
        "\nChaque jour de cotation, après minuit UTC (1 h ou 2 h du matin à Paris) : "
        "`uv run gold-agents live`. Tableau de bord : page « Simulation en direct »."
    )


@app.command()
def live(
    config: Path = CONFIG_OPTION,
    source: UpdateSource = SOURCE_OPTION,
    no_update: bool = typer.Option(False, "--no-update", help="Ne pas appeler l'API avant."),
    live_models: Path = LIVE_MODELS_OPTION,
    ledger_dir: Path = LIVE_LEDGER_OPTION,
    key: Path = PRIVATE_KEY_OPTION,
    public_key: Path = PUBLIC_KEY_OPTION,
    env: Path = ENV_OPTION,
    anchors_path: Path = ANCHORS_OPTION,
    no_anchor: bool = typer.Option(False, "--no-anchor", help="Ne pas ancrer sur Sepolia."),
) -> None:
    """Simulation en direct : ajoute les nouveaux jours de cotation au registre, signe, ancre."""
    from gold_agents.blockchain.signing import load_public_key
    from gold_agents.blockchain.storage import read_jsonl
    from gold_agents.live import session as ls

    cfg = load_config(config)
    path = ledger_dir / ls.LEDGER_FILE
    if not path.exists():
        typer.echo(f"{path} absent : lancer d'abord `uv run gold-agents live-init`.", err=True)
        raise typer.Exit(code=1)
    chain = read_jsonl(path)
    pub = load_public_key(public_key) if public_key.exists() else None
    check = chain.verify(pub)
    if not check.valid:
        typer.echo(f"Registre en direct refusé : {check.describe()}", err=True)
        raise typer.Exit(code=1)
    plan = ls.genesis(chain)
    problems = ls.check_models(live_models, plan["models"]["files"])
    if problems:
        for p in problems:
            typer.echo(f"  REFUS : {p}", err=True)
        typer.echo("Les modèles ne sont plus ceux scellés dans le plan : rien n'est décidé.")
        raise typer.Exit(code=1)
    chain.signer = _live_signer(key)
    anchor = not no_anchor
    if anchor:
        _check_anchor_env(env)
    if not no_update:
        _update(cfg, source)
    frame = ls.live_frame(cfg)
    for day in ls.revised_bars(chain, frame):
        typer.echo(
            f"  Note : la barre du {day} a été révisée par l'API ; le registre garde l'ancienne."
        )
    models = ls.load_models(live_models, int(plan["settings"]["window"]))
    before = len(chain)
    reports = ls.advance(chain, frame, models)
    state = ls.last_state(chain)
    if not reports:
        typer.echo(
            f"Aucun nouveau jour de cotation depuis le {state.last_date} (la barre du jour "
            "n'est prise qu'une fois close, après minuit UTC). Rien à ajouter."
        )
        return
    typer.echo(f"{len(reports)} jour(s) ajouté(s) (blocs {before} à {len(chain) - 1}) :")
    _live_report(reports)
    late = sum(not r.on_time for r in reports)
    if late:
        typer.echo(
            f"  {late} jour(s) rattrapé(s) : décisions marquées « en retard » dans le registre."
        )
    _save_live(chain, path, public_key, anchor, env, anchors_path)


if __name__ == "__main__":
    app()
