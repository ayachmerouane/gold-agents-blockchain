"""Interface en ligne de commande (Typer) : prepare, train, backtest, evaluate, verify."""

from __future__ import annotations

from pathlib import Path

import typer

from gold_agents.config import DEFAULT_CONFIG_PATH, load_config

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
