"""Agent ICT intraday : heure du serveur MT4, séances, setup de manuel, moteur."""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gold_agents.data.intraday import add_session_columns, read_mt4_intraday
from gold_agents.simulation.costs import CostModel
from gold_agents.simulation.ict_intraday import (
    IntradayParams,
    daily_equity,
    intraday_order,
    run_intraday_backtest,
)

NY = "America/New_York"

# Heure de New York -> (high, low, close) : soir (18-19 h), Asie (20-23 h), sweep à
# minuit sous le bas asiatique (100), MSS à 1 h au-dessus du sommet de 22 h, FVG à 2 h.
SETUP = [
    ("2024-03-04 18:00", 101.5, 100.5, 101.0),
    ("2024-03-04 19:00", 101.8, 100.8, 101.2),
    ("2024-03-04 20:00", 101.9, 100.2, 101.0),
    ("2024-03-04 21:00", 102.0, 100.0, 101.5),
    ("2024-03-04 22:00", 102.5, 101.0, 102.0),
    ("2024-03-04 23:00", 102.2, 100.6, 100.8),
    ("2024-03-05 00:00", 100.5, 99.0, 100.3),
    ("2024-03-05 01:00", 103.0, 100.2, 102.8),
    ("2024-03-05 02:00", 102.9, 101.0, 102.0),
]


def _frame(rows: list[tuple[str, float, float, float]]) -> pd.DataFrame:
    ts = pd.DatetimeIndex([r[0] for r in rows]).tz_localize(NY).tz_convert("UTC")
    h, lo, c = (np.array([r[i] for r in rows]) for i in (1, 2, 3))
    return add_session_columns(
        pd.DataFrame({"date": ts, "open": c, "high": h, "low": lo, "close": c})
    )


def test_server_time_ny_plus_7(tmp_path: Path) -> None:
    """« ny+7 » : 01:00 serveur = 18:00 New York la veille, été comme hiver."""
    csv = tmp_path / "XAU_1h_data.csv"
    csv.write_text(
        "Date;Open;High;Low;Close;Volume\n"
        "2024.01.15 01:00;2050;2052;2049;2051;10\n"
        "2024.07.15 01:00;2400;2402;2399;2401;10\n",
        encoding="utf-8",
    )
    df = read_mt4_intraday(csv, "ny+7")
    assert list(df["ny_hour"]) == [18, 18]
    assert list(df["date"].dt.hour) == [23, 22]  # UTC : hiver UTC-5, été UTC-4
    # La barre de 18 h ouvre le jour de cotation suivant.
    assert list(df["trading_day"]) == [date(2024, 1, 15), date(2024, 7, 15)]


def test_textbook_setup() -> None:
    df = _frame(SETUP)
    order = intraday_order(df, IntradayParams(min_reward_risk=0.5))
    assert order is not None
    assert (order.level, order.stop, order.target) == (101.0, 99.0, 102.5)
    # Objectif trop proche si l'on exige R/R ≥ 1.
    assert intraday_order(df, IntradayParams(min_reward_risk=1.0)) is None


def test_only_before_kill_zone_bars() -> None:
    """À 1 h, la barre suivante (2 h) est en kill zone mais le FVG n'existe pas encore ;
    sans kill zone à 3 h, rien n'est décidé à 2 h."""
    df = _frame(SETUP)
    assert intraday_order(df.iloc[:-1], IntradayParams(min_reward_risk=0.5)) is None
    params = IntradayParams(min_reward_risk=0.5, kill_zones=((7, 10),))
    assert intraday_order(df, params) is None


def test_swing_needs_confirmation() -> None:
    """Le sommet de 22 h n'est confirmé qu'à 23 h ; si le sweep a lieu à 23 h, il ne
    compte pas : pas de MSS reconnu."""
    rows = [r for r in SETUP if not r[0].endswith("23:00")]
    rows[5] = ("2024-03-04 23:00", 100.5, 99.0, 100.3)  # sweep dès 23 h (Asie)
    assert intraday_order(_frame(rows), IntradayParams(min_reward_risk=0.5)) is None


@pytest.mark.parametrize("entry", ["limit_fvg", "stop_pdh"])
def test_backtest_never_holds_overnight(hourly_series: pd.DataFrame, entry: str) -> None:
    df = hourly_series
    params = IntradayParams(entry=entry, min_reward_risk=0.5)  # type: ignore[arg-type]
    res = run_intraday_backtest(df, params, CostModel(0.001, 0.0002), 10_000, date(2019, 1, 7))
    assert len(res.trades) >= 2
    buys = [t for t in res.trades if t.side == "BUY"]
    sells = [t for t in res.trades if t.side == "SELL"]
    for b, s in zip(buys, sells, strict=False):
        assert s.exec_date >= b.exec_date
        assert (s.exec_date - b.exec_date) < pd.Timedelta(hours=24)
        assert b.exec_date > b.decision_date  # exécuté après la barre de décision
    eq = res.equity
    ny_hour = pd.to_datetime(eq["date"], utc=True).dt.tz_convert(NY).dt.hour
    assert (eq.loc[ny_hour.isin([17, 18, 19, 20]), "qty"] == 0).all()
    daily = daily_equity(res).equity
    assert len(daily) < len(eq) and daily["value"].iloc[-1] == eq["value"].iloc[-1]


def test_fixed_r_target() -> None:
    order = intraday_order(_frame(SETUP), IntradayParams(target="fixed_r", min_reward_risk=2.0))
    assert order is not None
    assert (order.level, order.stop, order.target) == (101.0, 99.0, 105.0)
