"""Validate the recomputed CTL/ATL against Garmin's own training-load series.

Rebuilds the daily fitness state from the Garmin export and reports its Pearson
correlation with Garmin's published acute/chronic load, plus a window sweep used to
calibrate CTL_DAYS / ATL_DAYS. Correlation (not equality) is the metric because Garmin's
load is in different units; see docs/fitness-model.md.

Usage:
    uv run python scripts/validate_fitness.py
"""

from __future__ import annotations

import glob

from marathon.features.fitness import ATL_DAYS, CTL_DAYS, fitness_state
from marathon.features.garmin import correlation_report, load_reference
from marathon.features.load import daily_load
from marathon.parse import load_export

FITNESS_GLOB = "data/DI_CONNECT/DI-Connect-Fitness/*_summarizedActivities.json"
METRICS_DIR = "data/DI_CONNECT/DI-Connect-Metrics"
CTL_SWEEP = [21, 28, 35, 42, 49]
ATL_SWEEP = [5, 7, 10, 14]


def main() -> None:
    activities = load_export(glob.glob(FITNESS_GLOB)[0])
    load = daily_load(activities)
    ref = load_reference(METRICS_DIR)

    report = correlation_report(fitness_state(load), ref)
    print(f"overlapping days: {int(report['n_days'])}")
    print(f"defaults (CTL={CTL_DAYS}d, ATL={ATL_DAYS}d):")
    print(f"  CTL vs Garmin chronic: r = {report['ctl_vs_chronic']:.3f}")
    print(f"  ATL vs Garmin acute  : r = {report['atl_vs_acute']:.3f}")

    print("\nCTL window sweep (vs Garmin chronic):")
    for days in CTL_SWEEP:
        r = correlation_report(fitness_state(load, ctl_days=days), ref)["ctl_vs_chronic"]
        print(f"  CTL_DAYS={days:>2}  r = {r:.3f}")

    print("\nATL window sweep (vs Garmin acute):")
    for days in ATL_SWEEP:
        r = correlation_report(fitness_state(load, atl_days=days), ref)["atl_vs_acute"]
        print(f"  ATL_DAYS={days:>2}  r = {r:.3f}")


if __name__ == "__main__":
    main()
