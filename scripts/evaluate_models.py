"""Evaluate the Riegel and Critical Speed models against Garmin's race predictions.

Usage:
    uv run python scripts/evaluate_models.py
"""

from __future__ import annotations

import glob

from marathon.features.efforts import effort_points
from marathon.features.garmin import load_race_predictions
from marathon.model.evaluate import evaluation_table, mape, race_results, rmspe
from marathon.parse.activities import load_export

FITNESS_GLOB = "data/DI_CONNECT/DI-Connect-Fitness/*_summarizedActivities.json"
METRICS_DIR = "data/DI_CONNECT/DI-Connect-Metrics"
MODELS = ["riegel_s", "cs_s", "garmin_s"]


def main() -> None:
    activities = load_export(glob.glob(FITNESS_GLOB)[0])
    points = effort_points(activities)
    garmin = load_race_predictions(METRICS_DIR)
    races = race_results(activities)

    table = evaluation_table(races, points, garmin)
    print(table.to_string(index=False))

    print("\nper-model error (races where the model made a prediction):")
    for col in MODELS:
        sub = table.dropna(subset=[col])
        mape_val = mape(sub["actual_s"], sub[col])
        rmspe_val = rmspe(sub["actual_s"], sub[col])
        print(f"  {col:9s}  MAPE {mape_val:.3f}  RMSPE {rmspe_val:.3f}  (n={len(sub)})")


if __name__ == "__main__":
    main()
