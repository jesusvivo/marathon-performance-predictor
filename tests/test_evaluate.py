import pytest

from marathon.model.evaluate import mape, rmspe


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
