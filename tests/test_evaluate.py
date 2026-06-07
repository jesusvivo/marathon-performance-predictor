import numpy as np
import pandas as pd
import pytest

from marathon.model.curve import fit_riegel, predict_riegel
from marathon.model.evaluate import (
    _distance_to_column,
    evaluation_table,
    garmin_predictions,
    holdout_predict,
    mape,
    rmspe,
)


@pytest.mark.parametrize(
    ("actual", "predicted", "expected_mape", "expected_rmspe"),
    [
        ([100, 200], [110, 180], 0.10, 0.10),  # equal-size errors: metrics agree
        ([100, 100], [110, 130], 0.20, 0.2236),  # uneven errors: rmspe punishes the big miss
    ],
    ids=["equal_errors", "uneven_errors"],
)
def test_metrics(
    actual: list[float], predicted: list[float], expected_mape: float, expected_rmspe: float
) -> None:
    """MAPE and RMSPE agree when errors are equal-sized and diverge when they are not."""
    assert mape(actual, predicted) == pytest.approx(expected_mape, rel=1e-3)
    assert rmspe(actual, predicted) == pytest.approx(expected_rmspe, rel=1e-3)


def test_holdout_predict_ignores_future_efforts() -> None:
    """The prediction for a race must not change when a (future) effort after it is added.

    Two efforts precede the race so the fit has enough points; a third, impossibly fast
    effort sits after the race. If the holdout guard works, that future effort is filtered
    out and the prediction is identical with or without it.
    """
    races = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-06-01"]),
            "distance_km": [10.0],
            "actual_s": [2400.0],
        }
    )
    points = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01", "2025-02-01", "2025-09-01"]),
            "duration_s": [1200.0, 600.0, 300.0],
            "distance_km": [4.0, 2.5, 2.0],  # last one is impossibly fast, and in the future
            "speed_ms": [4000 / 1200, 2500 / 600, 2000 / 300],
        }
    )
    before_only = points[points["date"] < pd.Timestamp("2025-06-01")]

    with_future = holdout_predict(races, points, fit_riegel, predict_riegel)
    without_future = holdout_predict(races, before_only, fit_riegel, predict_riegel)

    assert not np.isnan(with_future.iloc[0])  # a real prediction was made
    assert with_future.iloc[0] == pytest.approx(without_future.iloc[0])  # future ignored


@pytest.mark.parametrize(
    ("distance_km", "expected"),
    [
        (10.0, "pred_10k_s"),
        (21.1, "pred_half_s"),
        (5.0, "pred_5k_s"),
        (42.2, "pred_marathon_s"),
        (20.2, None),  # a 20K sits between distances: no standard column
        (15.0, None),
    ],
)
def test_distance_to_column(distance_km: float, expected: str | None) -> None:
    """Standard distances map to their column; off-distance races map to None."""
    assert _distance_to_column(distance_km) == expected


def _garmin() -> pd.DataFrame:
    g = pd.DataFrame(
        {
            "pred_5k_s": [1300, 1290],
            "pred_10k_s": [2700, 2690],
            "pred_half_s": [5800, 5790],
            "pred_marathon_s": [12600, 12500],
        },
        index=pd.to_datetime(["2025-01-01", "2025-02-01"]),
    )
    g.index.name = "date"
    return g


def test_garmin_predictions_matches_distance_and_date() -> None:
    """Picks the right column for the distance and the right row via asof, NaN if no match."""
    races = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-02-01", "2025-01-15", "2025-02-01"]),
            "distance_km": [10.0, 21.1, 15.0],
            "actual_s": [2600.0, 5700.0, 4000.0],
        }
    )
    preds = garmin_predictions(races, _garmin())
    assert preds.iloc[0] == 2690  # 10K, exact date 2025-02-01
    assert preds.iloc[1] == 5800  # Half, asof falls back to the 2025-01-01 row
    assert np.isnan(preds.iloc[2])  # 15 km has no standard column


def test_evaluation_table_assembles_all_models() -> None:
    """The table carries actual plus all three model predictions, one row per race."""
    races = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-03-01"]),
            "distance_km": [10.0],
            "actual_s": [2600.0],
        }
    )
    points = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01", "2025-02-01"]),
            "duration_s": [1200.0, 600.0],
            "distance_km": [4.0, 2.5],
            "speed_ms": [4000 / 1200, 2500 / 600],
        }
    )
    table = evaluation_table(races, points, _garmin())
    assert list(table.columns) == [
        "date",
        "distance_km",
        "actual_s",
        "riegel_s",
        "cs_s",
        "garmin_s",
    ]
    assert len(table) == 1
    assert table["garmin_s"].iloc[0] == 2690  # 10K, asof 2025-02-01
    assert not np.isnan(table["riegel_s"].iloc[0])  # a real model prediction was made
