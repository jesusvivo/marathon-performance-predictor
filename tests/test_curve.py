import pandas as pd
import pytest

from marathon.model.curve import fit_cs, fit_riegel, predict_cs, predict_riegel


def _riegel_points() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "distance_km": [5.0, 10.0, 21.097, 42.195],
            "duration_s": [1101.38, 2296.31, 5066.45, 10563.47]
        }
    )

def _cs_points() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "distance_km": [1.7, 3.2, 6.2, 12.2],
            "duration_s": [300, 600, 1200, 2400]
        }
    )

def test_fit_riegel() -> None:
    points_df = _riegel_points()

    a, b = fit_riegel(points_df)

    assert a == pytest.approx(200, rel=1e-3)
    assert b == pytest.approx(1.06, rel=1e-3)

def test_predict_riegel() -> None:
    points_df = _riegel_points()

    marathon_distance = points_df['distance_km'].iloc[-1]
    marathon_duration = points_df['duration_s'].iloc[-1]
    a = 200
    b = 1.06

    assert marathon_duration == pytest.approx(predict_riegel(marathon_distance, a, b),rel=1e-3)

def test_fit_cs() -> None:
    points_df = _cs_points()
    cs, d_prime = fit_cs(points_df)

    assert cs == pytest.approx(5, rel=1e-3)
    assert d_prime == pytest.approx(200, rel=1e-3)
    
def test_predict_cs() -> None:
    points_df = _cs_points()

    marathon_distance = points_df['distance_km'].iloc[-1]
    marathon_duration = points_df['duration_s'].iloc[-1]
    cs = 5
    d_prime = 200

    assert marathon_duration == pytest.approx(predict_cs(marathon_distance, cs, d_prime),rel=1e-3)

