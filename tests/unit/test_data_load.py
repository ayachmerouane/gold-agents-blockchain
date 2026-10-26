"""Tests de lecture du CSV Kaggle (format MetaTrader 4) et de la série combinée."""

from pathlib import Path

import pandas as pd
import pytest

from gold_agents.data.load import COLUMNS, load_combined, read_kaggle_csv

MT4_CSV = (
    "Date;Open;High;Low;Close;Volume\n"
    "2004.06.14 00:00;384.3;386.0;383.5;385.1;5120\n"
    "2004.06.11 00:00;384.0;385.2;382.9;384.3;4980\n"
    "2004.06.15 00:00;385.1;385.9;384.0;384.6;5011\n"
)


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_reads_mt4_semicolon_csv(tmp_path: Path) -> None:
    df = read_kaggle_csv(_write(tmp_path / "XAU_1d_data.csv", MT4_CSV))
    assert list(df.columns) == COLUMNS
    assert df["date"].is_monotonic_increasing
    assert str(df["date"].dt.tz) == "UTC"
    assert df["date"].iloc[0] == pd.Timestamp("2004-06-11", tz="UTC")
    assert df["close"].tolist() == [384.3, 385.1, 384.6]
    assert (df["source"] == "kaggle").all()


def test_reads_comma_csv_with_bom_and_iso_dates(tmp_path: Path) -> None:
    text = "﻿Date,Open,High,Low,Close\n2024-01-02,2063.1,2078.0,2058.5,2073.4\n"
    df = read_kaggle_csv(_write(tmp_path / "x.csv", text))
    assert df["date"].iloc[0] == pd.Timestamp("2024-01-02", tz="UTC")
    assert df["volume"].isna().all()


def test_duplicate_dates_keep_last(tmp_path: Path) -> None:
    text = MT4_CSV + "2004.06.15 00:00;385.1;385.9;384.0;999.0;1\n"
    df = read_kaggle_csv(_write(tmp_path / "x.csv", text))
    assert len(df) == 3
    assert df["close"].iloc[-1] == 999.0


def test_unreadable_dates_raise(tmp_path: Path) -> None:
    text = "Date;Open;High;Low;Close\nhier;1;1;1;1\n"
    with pytest.raises(ValueError, match="dates illisibles"):
        read_kaggle_csv(_write(tmp_path / "x.csv", text))


def test_combined_keeps_kaggle_rows_and_appends_later_days(tmp_path: Path) -> None:
    raw = _write(tmp_path / "raw.csv", MT4_CSV)
    update = _write(
        tmp_path / "update.csv",
        "date,open,high,low,close,volume,source\n"
        "2004-06-15,1,1,1,1.0,,twelvedata\n"
        "2004-06-16,384.6,386.1,384.2,385.9,,twelvedata\n",
    )
    df = load_combined(raw, update)
    assert len(df) == 4
    assert df.set_index(df["date"].dt.strftime("%Y-%m-%d"))["source"].to_dict() == {
        "2004-06-11": "kaggle",
        "2004-06-14": "kaggle",
        "2004-06-15": "kaggle",
        "2004-06-16": "twelvedata",
    }


def test_combined_without_update_file(tmp_path: Path) -> None:
    raw = _write(tmp_path / "raw.csv", MT4_CSV)
    assert len(load_combined(raw, tmp_path / "absent.csv")) == 3
