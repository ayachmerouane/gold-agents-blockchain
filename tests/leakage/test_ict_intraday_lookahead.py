"""Anti-fuite : l'agent intraday jusqu'à la barre t ne dépend d'aucun prix après t."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from gold_agents.simulation.costs import CostModel
from gold_agents.simulation.ict_intraday import IntradayParams, run_intraday_backtest


@pytest.mark.parametrize("entry", ["limit_fvg", "stop_pdh"])
@pytest.mark.parametrize("cut", [1500, 3100])
def test_future_replaced_by_noise(hourly_series: pd.DataFrame, cut: int, entry: str) -> None:
    series = hourly_series
    rng = np.random.default_rng(cut)
    noisy = series.copy()
    after = noisy.index > cut
    base = rng.uniform(1000, 2000, after.sum())
    noisy.loc[after, "open"] = base
    noisy.loc[after, "close"] = base * rng.uniform(0.99, 1.01, after.sum())
    noisy.loc[after, "high"] = noisy.loc[after, ["open", "close"]].max(axis=1) * 1.002
    noisy.loc[after, "low"] = noisy.loc[after, ["open", "close"]].min(axis=1) * 0.998
    params = IntradayParams(entry=entry, min_reward_risk=0.5)  # type: ignore[arg-type]
    costs = CostModel(0.001, 0.0002)
    a = run_intraday_backtest(series, params, costs, 10_000, date(2019, 1, 7))
    b = run_intraday_backtest(noisy, params, costs, 10_000, date(2019, 1, 7))
    t = series["date"].iloc[cut]
    pd.testing.assert_frame_equal(a.equity[a.equity["date"] <= t], b.equity[b.equity["date"] <= t])
    assert [x for x in a.trades if x.exec_date <= t] == [x for x in b.trades if x.exec_date <= t]
