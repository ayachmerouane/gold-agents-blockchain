"""L'agent de prévision ne voit jamais le futur (§10.3) : changer les prix après le jour t
ne change aucune prévision faite jusqu'à t."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from gold_agents.agents.forecast import network_table  # noqa: E402
from gold_agents.config import Config  # noqa: E402
from gold_agents.data.prepare import prepare  # noqa: E402
from gold_agents.models.dataset import build_forecast_data  # noqa: E402
from gold_agents.models.predict import TrainedNetwork  # noqa: E402
from gold_agents.models.recurrent import NetConfig  # noqa: E402
from gold_agents.models.train import build_network  # noqa: E402


def test_forecasts_ignore_future_prices(cfg: Config, clean_series: pd.DataFrame) -> None:
    cut = pd.Timestamp("2020-06-30", tz="UTC")
    future = clean_series["date"] > cut
    shocked = clean_series.copy()
    rng = np.random.default_rng(3)
    factor = np.exp(np.cumsum(rng.normal(0, 0.03, int(future.sum()))))
    for col in ("open", "high", "low", "close"):
        shocked.loc[future, col] = shocked.loc[future, col] * factor

    torch.manual_seed(0)
    data = {}
    for name, series in (("orig", clean_series), ("shocked", shocked)):
        prepared = prepare(series, cfg)
        data[name] = build_forecast_data(prepared.frame, prepared.scaler, cfg.features.window)
    n_features = len(data["orig"].columns)
    model = build_network("gru", NetConfig(n_features, hidden=8))
    net = TrainedNetwork("gru", [model], 0.01, {})

    tables = {
        name: network_table("gru", [net], d["val"].x, d["val"].windows.dates, 0)
        for name, d in data.items()
    }
    # Le jour t = cut lui-même a une cible dans le futur, mais sa fenêtre s'arrête à t.
    before = np.array([d <= cut.date() for d in tables["orig"].dates])
    assert before.sum() > 100 and (~before).sum() > 100
    assert np.allclose(tables["orig"].r_hat[before], tables["shocked"].r_hat[before])
    assert not np.allclose(tables["orig"].r_hat[~before], tables["shocked"].r_hat[~before])


def test_mc_dropout_sigma(cfg: Config) -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 30, 5)).astype("float32")
    dates = pd.bdate_range("2020-01-01", periods=40)
    nets = [
        TrainedNetwork(kind, [build_network(kind, NetConfig(5, hidden=8, dropout=0.3))], 0.01, {})
        for kind in ("lstm", "gru")
    ]
    table = network_table("ens", nets, x, dates, mc_samples=20)
    assert table.sigma_hat is not None and (table.sigma_hat > 0).all()
    point = np.mean([n.predict(x) for n in nets], axis=0)
    assert np.allclose(table.r_hat, point)  # r̂ : prévision sans dropout, comme en phase 4
