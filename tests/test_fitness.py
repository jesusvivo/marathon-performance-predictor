"""Unit tests for CTL / ATL / TSB fitness state."""

from __future__ import annotations

import numpy as np
import pandas as pd

from marathon.features.fitness import _ewma, fitness_state


def test_ewma_matches_recursive_definition() -> None:
    load = pd.Series([100.0, 0.0, 50.0, 0.0, 80.0], index=pd.date_range("2024-01-01", periods=5))
    days = 7
    alpha = 1.0 - np.exp(-1.0 / days)
    expected = []
    prev: float | None = None
    for x in load:
        prev = x if prev is None else (1 - alpha) * prev + alpha * x
        expected.append(prev)
    assert np.allclose(_ewma(load, days).to_numpy(), expected)


def test_atl_decays_faster_than_ctl() -> None:
    # one big day then rest: the 7-day ATL drops below the 42-day CTL during recovery
    load = pd.Series([200.0] + [0.0] * 10, index=pd.date_range("2024-01-01", periods=11))
    fs = fitness_state(load)
    assert fs["atl"].iloc[5] < fs["ctl"].iloc[5]


def test_tsb_is_prior_day_ctl_minus_atl() -> None:
    load = pd.Series([100.0, 50.0, 20.0], index=pd.date_range("2024-01-01", periods=3))
    fs = fitness_state(load)
    assert pd.isna(fs["tsb"].iloc[0])  # no prior day exists
    assert fs["tsb"].iloc[1] == fs["ctl"].iloc[0] - fs["atl"].iloc[0]
    assert fs["tsb"].iloc[2] == fs["ctl"].iloc[1] - fs["atl"].iloc[1]


def test_columns_and_index_preserved() -> None:
    load = pd.Series(
        [10.0, 20.0, 30.0], index=pd.date_range("2024-01-01", periods=3), name="load"
    )
    fs = fitness_state(load)
    assert list(fs.columns) == ["load", "ctl", "atl", "tsb"]
    assert list(fs.index) == list(load.index)
