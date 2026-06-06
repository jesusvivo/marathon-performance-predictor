import pandas as pd
import pytest

from marathon.model.curve import fit_riegel, predict_riegel


def _points() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "distance_km": [5.0, 10.0, 21.097, 42.195],
            "duration_s": [1101.38, 2296.31, 5066.45, 10563.47]
        }
    )

def test_fit_riegel() -> None:
    points_df = _points()

    a, b = fit_riegel(points_df)

    assert a == pytest.approx(200, rel=1e-3)
    assert b == pytest.approx(1.06, rel=1e-3)

def test_predict_riegel() -> None:
    points_df = _points()

    marathon_distance = points_df['distance_km'].iloc[-1]
    marathon_duration = points_df['duration_s'].iloc[-1]
    a = 200
    b = 1.06

    assert marathon_duration == pytest.approx(predict_riegel(marathon_distance, a, b), rel= 1e-3)