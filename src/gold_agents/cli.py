"""Interface en ligne de commande (Typer) : download, eda, prepare, train, backtest, evaluate…"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

import typer

from gold_agents.config import DEFAULT_CONFIG_PATH, load_config
from gold_agents.data import clean as cleaning
from gold_agents.data import sources, update
from gold_agents.data.load import read_kaggle_csv, read_update_csv

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
    cfg = load_config(config)
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


@app.command()
def backtest(
    config: Path = CONFIG_OPTION,
    out_dir: Path = BACKTEST_OUT_OPTION,
    ledger_dir: Path = LEDGER_DIR_OPTION,
    costs_factor: float = typer.Option(1.0, "--costs", help="Multiplie frais et slippage."),
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
    runs = report.run_reference_backtests(
        cfg, frame, sha256_file(path), out_dir, ledger_dir, costs_factor
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
    typer.echo("\n".join(report.tamper_lines(runs[0])))
    figure = report.backtest_figure(runs, cfg, out_dir / "backtest_test.png")
    typer.echo(f"\nSchéma : {figure}\nRésultats : {out_dir}  Registres : {ledger_dir}")
    if not all(run.verification.valid for run in runs):
        raise typer.Exit(code=1)


@app.command()
def evaluate(config: Path = CONFIG_OPTION) -> None:
    """Calcule les métriques et génère les figures."""
    _todo("evaluate")


LEDGERS_ARGUMENT = typer.Argument(None, help="Fichiers .jsonl (défaut : outputs/ledger/).")


@app.command()
def verify(ledgers: list[Path] | None = LEDGERS_ARGUMENT) -> None:
    """Vérifie l'intégrité des registres (hash de chaque bloc et liens entre blocs)."""
    from gold_agents.blockchain.storage import read_jsonl

    paths = ledgers or sorted(Path("outputs/ledger").glob("*.jsonl"))
    if not paths:
        typer.echo("Aucun registre trouvé : lancer d'abord `make backtest`.", err=True)
        raise typer.Exit(code=1)
    ok = True
    for path in paths:
        result = read_jsonl(path).verify()
        ok &= result.valid
        typer.echo(f"{path} : {result.describe()}")
    if not ok:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
