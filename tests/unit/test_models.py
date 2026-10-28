"""Tests des modèles (§10.2) : forme des sorties, surapprentissage volontaire d'un mini-lot,
reproductibilité, baselines sans regard sur le futur, chaîne complète `train --quick`."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightgbm")
pytest.importorskip("statsmodels")

from typer.testing import CliRunner  # noqa: E402

from gold_agents.cli import app  # noqa: E402
from gold_agents.config import Config  # noqa: E402
from gold_agents.data.prepare import load_prepared, prepare, write_prepared  # noqa: E402
from gold_agents.models import baselines  # noqa: E402
from gold_agents.models.dataset import ForecastData, build_forecast_data  # noqa: E402
from gold_agents.models.predict import load_network  # noqa: E402
from gold_agents.models.recurrent import NetConfig  # noqa: E402
from gold_agents.models.train import NETWORKS, TrainConfig, build_network, fit, predict  # noqa: E402
from gold_agents.models.uncertainty import mc_dropout  # noqa: E402

N, W, F = 32, 30, 20


@pytest.fixture(scope="module")
def data(cfg: Config, clean_series: pd.DataFrame) -> ForecastData:
    prepared = prepare(clean_series, cfg)
    return build_forecast_data(prepared.frame, prepared.scaler, cfg.features.window)


def _batch(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return (
        rng.normal(size=(N, W, F)).astype("float32"),
        rng.normal(size=N).astype("float32"),
    )


# --- Réseaux -----------------------------------------------------------------------


@pytest.mark.parametrize("kind", sorted(NETWORKS))
@pytest.mark.parametrize("layers", [1, 2])
def test_network_output_shape(kind: str, layers: int) -> None:
    model = build_network(kind, NetConfig(F, hidden=16, layers=layers))
    x, _ = _batch()
    assert model(torch.from_numpy(x)).shape == (N,)
    assert predict(model, x).shape == (N,)


def test_gru_has_fewer_parameters_than_lstm() -> None:
    def count(kind: str) -> int:
        return sum(p.numel() for p in build_network(kind, NetConfig(F, 32)).parameters())

    assert count("gru") < count("lstm")


@pytest.mark.parametrize("kind", sorted(NETWORKS))
def test_overfits_a_mini_batch(kind: str) -> None:
    """Preuve que la boucle d'entraînement apprend : 32 exemples appris par cœur."""
    x, y = _batch()
    cfg = TrainConfig(
        lr=1e-2, batch_size=N, weight_decay=0, loss="mse", max_epochs=300, patience=300
    )
    result = fit(kind, NetConfig(F, hidden=32, dropout=0.0), cfg, x, y, x, y, seed=0)
    assert result.best_val_loss < 0.05 * float(np.mean(y**2))
    assert result.history[0][0] > 10 * result.best_val_loss


@pytest.mark.parametrize("kind", sorted(NETWORKS))
def test_same_seed_same_result(kind: str) -> None:
    x, y = _batch()
    xv, yv = _batch(1)
    cfg = TrainConfig(batch_size=8, max_epochs=3, patience=3)
    net = NetConfig(F, hidden=16)
    a = fit(kind, net, cfg, x, y, xv, yv, seed=3)
    b = fit(kind, net, cfg, x, y, xv, yv, seed=3)
    c = fit(kind, net, cfg, x, y, xv, yv, seed=4)
    assert np.array_equal(predict(a.model, xv), predict(b.model, xv))
    assert a.history == b.history
    assert not np.array_equal(predict(a.model, xv), predict(c.model, xv))


def test_early_stopping_keeps_best_epoch() -> None:
    x, y = _batch()
    xv, yv = _batch(1)  # validation sans lien avec le train : elle se dégrade vite
    cfg = TrainConfig(lr=1e-2, batch_size=N, weight_decay=0, max_epochs=200, patience=5)
    result = fit("gru", NetConfig(F, hidden=32, dropout=0.0), cfg, x, y, xv, yv, seed=0)
    assert len(result.history) == result.best_epoch + 5 < 200
    best = min(h[1] for h in result.history)
    assert result.best_val_loss == best
    assert float(np.mean((predict(result.model, xv) - yv) ** 2)) == pytest.approx(best, rel=1e-5)


def test_mc_dropout() -> None:
    x, _ = _batch()
    model = build_network("lstm", NetConfig(F, hidden=16, dropout=0.3))
    mean, std = mc_dropout(model, x, n_samples=20)
    assert mean.shape == std.shape == (N,)
    assert (std > 0).all()
    assert not model.training  # remis en mode évaluation
    still = build_network("lstm", NetConfig(F, hidden=16, dropout=0.0))
    assert np.allclose(mc_dropout(still, x, n_samples=5)[1], 0, atol=1e-6)


# --- Données et baselines -------------------------------------------------------------


def test_forecast_data_shapes(data: ForecastData) -> None:
    for name in ("train", "val", "test"):
        part = data[name]
        assert part.x.shape == (len(part.y), 30, len(data.columns))
        assert part.tabular.shape == (len(part.y), len(data.columns))
        assert (data.frame.loc[part.rows, "split"] == name).all()
    assert data["train"].rows.max() < data["val"].rows.min()


def test_simple_baselines(data: ForecastData) -> None:
    val = data["val"]
    assert (baselines.zero(5) == 0).all()
    assert np.allclose(baselines.last_return(data.frame, val.rows), data.frame["ret_1d"][val.rows])
    mm = baselines.mean_return(data.frame, val.rows, 20)
    t = int(val.rows[0])
    by_hand = np.log(data.frame["close"]).diff()[t - 19 : t + 1].mean()
    assert mm[0] == pytest.approx(by_hand)


def test_arima_does_not_look_ahead(data: ForecastData) -> None:
    """Remplacer le futur par du bruit ne change pas les prévisions déjà faites."""
    # Paramètres fixés (constante, AR(1), variance) : la série synthétique est une marche
    # aléatoire, l'ordre choisi par l'AIC y serait (0, 0, 0), une constante.
    arima = baselines.Arima((1, 0, 0), np.array([0.0, 0.3, 1e-4]), 0.0)
    rows = data["val"].rows
    pred = arima.predict(data.frame, rows)
    cut = int(rows[len(rows) // 2])
    noisy = data.frame.copy()
    noisy.loc[cut + 1 :, "ret_1d"] = 0.5
    pred2 = arima.predict(noisy, rows)
    before = rows <= cut
    assert np.allclose(pred[before], pred2[before])
    assert not np.allclose(pred[~before], pred2[~before])


def test_fit_arima_returns_best_aic(data: ForecastData) -> None:
    arima = baselines.fit_arima(data.frame, max_p=1, max_q=1)
    assert arima.order[1] == 0 and max(arima.order) <= 1
    pred = arima.predict(data.frame, data["val"].rows)
    assert pred.shape == data["val"].y.shape and np.isfinite(pred).all()


def test_ridge_picks_alpha_on_validation() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 3))
    y = x @ np.array([1.0, -2.0, 0.5]) + rng.normal(0, 0.1, 400)
    model, alpha = baselines.fit_ridge(x[:300], y[:300], x[300:], y[300:])
    assert alpha <= 1.0
    assert np.allclose(model.coef_, [1.0, -2.0, 0.5], atol=0.05)


# --- Chaîne complète ------------------------------------------------------------------


def test_train_cli_quick(tmp_path: Path, cfg: Config, clean_series: pd.DataFrame) -> None:
    raw = cfg.model_dump(mode="json")
    raw["data"]["processed_dir"] = str(tmp_path / "processed")
    raw["models"]["seeds"] = [0, 1]
    local = Config.model_validate(raw)
    clean_path = tmp_path / "clean.parquet"
    clean_series.to_parquet(clean_path)
    write_prepared(prepare(clean_series, local), local, clean_path)
    config_path = tmp_path / "config.yaml"
    import yaml

    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    out, models_dir = tmp_path / "results", tmp_path / "models"
    result = CliRunner().invoke(
        app,
        [
            "train",
            "-c",
            str(config_path),
            "--quick",
            "--no-mlflow",
            "--out",
            str(out),
            "--models-dir",
            str(models_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "B0 rendement nul" in result.output and "M2 GRU" in result.output
    summary = pd.read_csv(out / "summary_val.csv", index_col="model")
    assert list(summary.index) == [
        "b0",
        "b1",
        "b1_mm20",
        "b2",
        "b3",
        "b4",
        "lstm",
        "gru",
        "ensemble",
    ]
    assert (summary["rmse"] > 0).all()
    assert (out / "forecast_val.png").stat().st_size > 10_000

    # Les poids sauvegardés redonnent la prévision moyenne écrite dans les résultats.
    frame, scaler = load_prepared(local.data.processed_dir)
    data = build_forecast_data(frame, scaler, local.features.window)
    network = load_network(models_dir / "gru")
    preds = pd.read_parquet(out / "predictions_val.parquet")
    assert np.allclose(network.predict(data["val"].x), preds["gru"], atol=1e-7)


def test_train_cli_rejects_unknown_model(cfg: Config) -> None:
    result = CliRunner().invoke(app, ["train", "--models", "b0,transformer"])
    assert result.exit_code == 1
    assert "transformer" in result.output
