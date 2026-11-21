"""Anti-fuite (§10.3) : la stratégie ICT jusqu'au jour t ne dépend d'aucun prix après t."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from gold_agents.simulation.costs import CostModel
from gold_agents.simulation.ict import IctParams, run_ict_backtest

CUTS = [3200, 3900, 4600]


@pytest.mark.parametrize("entry", ["limit_fvg", "stop_pdh"])
@pytest.mark.parametrize("cut", CUTS)
def test_future_replaced_by_noise(clean_series: pd.DataFrame, cut: int, entry: str) -> None:
    rng = np.random.default_rng(cut)
    series = clean_series.reset_index(drop=True)
    noisy = series.copy()
    after = noisy.index > cut
    base = rng.uniform(100, 5000, after.sum())
    noisy.loc[after, "open"] = base
    noisy.loc[after, "close"] = base * rng.uniform(0.9, 1.1, after.sum())
    noisy.loc[after, "high"] = noisy.loc[after, ["open", "close"]].max(axis=1) * 1.01
    noisy.loc[after, "low"] = noisy.loc[after, ["open", "close"]].min(axis=1) * 0.99
    params = IctParams(entry=entry, swing=1, discount=1.0, min_reward_risk=0.5)  # type: ignore[arg-type]
    start = date(2015, 1, 1)
    a = run_ict_backtest(series, params, CostModel(0.001, 0.0002), 10_000, start)
    b = run_ict_backtest(noisy, params, CostModel(0.001, 0.0002), 10_000, start)
    cut_day = pd.Timestamp(series["date"].iloc[cut]).date()
    ea, eb = (r.equity[r.equity["date"] <= cut_day] for r in (a, b))
    pd.testing.assert_frame_equal(ea, eb)  # y compris l'ordre écrit le soir du jour t
    assert [t for t in a.trades if t.exec_date <= cut_day] == [
        t for t in b.trades if t.exec_date <= cut_day
    ]
    assert len(a.trades) > 0
