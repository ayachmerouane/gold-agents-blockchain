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
    """Prépare les données : nettoyage, features, splits, fenêtres."""
    _todo("prepare")


@app.command()
def train(config: Path = CONFIG_OPTION) -> None:
    """Entraîne les modèles."""
    _todo("train")


@app.command()
def backtest(config: Path = CONFIG_OPTION) -> None:
    """Lance la simulation multi-agents et écrit le registre."""
    _todo("backtest")


@app.command()
def evaluate(config: Path = CONFIG_OPTION) -> None:
    """Calcule les métriques et génère les figures."""
    _todo("evaluate")


@app.command()
def verify(ledger: Path = Path("outputs/ledger/chain.jsonl")) -> None:
    """Vérifie l'intégrité du registre."""
    _todo("verify")


if __name__ == "__main__":
    app()
