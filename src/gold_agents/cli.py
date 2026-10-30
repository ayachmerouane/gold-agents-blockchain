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
    typer.echo(
        f"Période {split} : du {start} au {end}, frais x {costs_factor:g}. Stratégie : "
        f"θ_buy {st.theta_buy:g}, θ_sell {st.theta_sell:g}, confiance k = {st.min_confidence:g}."
    )
    typer.echo("\nPrévisions des modèles (agent de prévision) :")
    typer.echo(ar.diagnostics(tables, cfg).to_string())

    results = ar.run_agent_backtests(
        cfg, frame, tables, split, sha256_file(path), out, ledgers, costs_factor
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
    sim = tu.Simulator(
        CostModel(sim_cfg.fee_rate, sim_cfg.slippage),
        sim_cfg.initial_capital,
        cfg.risk.max_position,
        cfg.risk.cooldown_days,
        sim_cfg.risk_free_rate,
    )
    markets = tu.fold_markets(frame, forecasts, list(tc.folds))
    grid = tu.combos(tc.grid)
    typer.echo(f"\nGrille : {len(grid)} réglages x {len(markets)} périodes.")
    table, bh = tu.run_grid(markets, grid, sim)
    folds = [m.fold for m in markets]
    summary = tu.summarize_grid(table, tc.grid, folds)
    chosen = tu.best(summary)
    walk = tu.walk_forward(table, bh, tc.grid, markets, sim)
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
    p = chosen.params(cfg.risk.max_position, cfg.risk.cooldown_days)
    system = AgentSystem(
        "agents_regles",
        ForecastAgent(table_fc),
        StrategyAgent(p.theta_buy, p.theta_sell, p.min_confidence),
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
    (out_dir / "chosen.json").write_text(
        json.dumps(
            {
                "chosen": {
                    "model": chosen.model,
                    "theta_buy": chosen.theta_buy,
                    "theta_sell": chosen.theta_sell,
                    "min_confidence": chosen.min_confidence,
                    "stop_loss": chosen.stop_loss,
                    "max_drawdown": chosen.max_drawdown,
                    "cooldown_days": cfg.risk.cooldown_days,
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
    typer.echo(f"\nRéglage choisi : {chosen.describe()}")
    typer.echo(
        "À recopier dans config/config.yaml :\n" + tr.yaml_snippet(chosen, cfg.risk.cooldown_days)
    )
    typer.echo(
        f"Schéma : {figure}\nRésultats : {out_dir}  ({time.perf_counter() - start_time:.0f} s)"
    )
    if not same:
        raise typer.Exit(code=1)


TUNING_DIR_OPTION = typer.Option(
    Path("outputs/results/tuning"), "--tuning-dir", help="Résultats de `tune`."
)


@app.command()
def freeze(
    config: Path = CONFIG_OPTION,
    tuning_dir: Path = TUNING_DIR_OPTION,
    ledger_dir: Path = LEDGER_DIR_OPTION,
) -> None:
    """Fige les réglages : les compare à `tune`, puis les scelle dans un bloc genèse."""
    import hashlib
    import json

    from gold_agents.blockchain.chain import Chain
    from gold_agents.blockchain.storage import read_jsonl, write_jsonl
    from gold_agents.data.prepare import FEATURES_FILE
    from gold_agents.data.update import sha256_file

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
    st, rk = cfg.strategy, cfg.risk
    current = {
        "model": st.model,
        "theta_buy": st.theta_buy,
        "theta_sell": st.theta_sell,
        "min_confidence": st.min_confidence,
        "stop_loss": rk.stop_loss,
        "max_drawdown": rk.max_drawdown,
        "cooldown_days": rk.cooldown_days,
    }
    diff = {k: (current[k], v) for k, v in chosen["chosen"].items() if current[k] != v}
    if diff:
        for k, (a, b) in diff.items():
            typer.echo(f"  {k} : config {a!r}, tune {b!r}", err=True)
        typer.echo("La configuration ne correspond pas au réglage choisi par `tune`.", err=True)
        raise typer.Exit(code=1)
    grid = tuning_dir / "grid.csv"
    if hashlib.sha256(grid.read_bytes()).hexdigest() != chosen["grid_sha256"]:
        typer.echo(f"{grid} a changé depuis `tune`.", err=True)
        raise typer.Exit(code=1)
    events = [
        {
            "type": "GENESIS",
            "kind": "FROZEN_SETTINGS",
            "frozen_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "settings": current,
            "config": json.loads(cfg.model_dump_json()),
            "config_sha256": sha256_file(config),
            "data_sha256": sha256_file(cfg.data.processed_dir / FEATURES_FILE),
            "tuning": {"grid_sha256": chosen["grid_sha256"], **chosen},
        }
    ]
    path = write_jsonl(Chain.start(events), ledger_dir / "reglages_figes.jsonl")
    chain = read_jsonl(path)
    typer.echo(f"Réglages figés dans {path} : {chain.verify().describe()}.")
    typer.echo(f"Empreinte du bloc genèse : {chain.head.hash}")


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
