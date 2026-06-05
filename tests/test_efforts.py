"""Unit tests for velocity-duration effort points and the best-effort frontier."""

from __future__ import annotations

import pandas as pd

from marathon.features.efforts import effort_points, velocity_duration_frontier


def _fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "discipline": ["run", "run", "run", "run", "bike"],
            "is_parent": [False, False, False, False, False],
            "start_time_local": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
            ),
            "distance_km": [1.0, 2.0, 3.0, 1.5, 40.0],
            "duration_s": [100.0, 200.0, 300.0, 150.0, 3600.0],
            "avg_speed_ms": [5.0, 4.0, 3.0, 3.5, 8.0],
            "is_race": [True, False, False, False, False],
        }
    )


def test_effort_points_keeps_runs_and_sorts() -> None:
    pts = effort_points(_fixture())
    assert len(pts) == 4  # bike excluded
    assert list(pts["duration_s"]) == [100.0, 150.0, 200.0, 300.0]
    assert int(pts["is_race"].sum()) == 1


def test_frontier_drops_dominated_efforts() -> None:
    fr = velocity_duration_frontier(effort_points(_fixture()))
    # the 150 s / 3.5 m/s effort is beaten by the 200 s / 4.0 m/s effort, so it is dropped
    assert list(fr["duration_s"]) == [100.0, 200.0, 300.0]
    assert list(fr["speed_ms"]) == [5.0, 4.0, 3.0]  # monotonic decreasing
