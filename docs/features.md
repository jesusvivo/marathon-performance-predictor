# Daily feature matrix

`features/build.py::daily_features` produces one row per calendar day, from the first to the last
activity. It is the offline table the model trains on and the feature store serves. Build it with
`uv run python scripts/build_features.py` (writes `artifacts/daily_features.parquet`).

## Fitness state (`features/fitness.py`)

- `load`: total multisport training load that day (0 on rest days).
- `ctl`, `atl`, `tsb`: chronic load (fitness), acute load (fatigue), and form. See
  [fitness-model.md](fitness-model.md) for the physiology and Garmin calibration.

## Running load and form (`features/running.py`)

- `run_km`: running distance that day.
- `volume_7d_km`, `volume_28d_km`: trailing 7- and 28-day running distance (rolling sums).
- `avg_pace_s_per_km_28d`: distance-weighted average running pace over the trailing 28 days
  (total time / total distance), in seconds per km.
- `efficiency_28d`: efficiency factor (speed per heartbeat, `avg_speed / avg_hr`) averaged over the
  runs in the trailing 28 days; rises as aerobic fitness improves at a given heart rate.
- `days_since_long_run`: days since the last run of at least 15 km (NaN before the first one).
- `vo2max`: most recent running VO2max, carried forward.

## Wellness (`features/wellness.py`)

- `sleep_hours`: deep + light + rem sleep.
- `sleep_score`: Garmin overall sleep score.
- `readiness_score`: Garmin training-readiness (0-100).
- `recovery_time_h`: recommended recovery hours.
- `hrv_weekly_avg`: 7-day average heart-rate variability.

Slowly sampled signals (VO2max) are forward-filled. Wellness is left as-is (NaN where a day has no
record), so the model sees genuine gaps rather than fabricated values.

## Effort anchors (`features/efforts.py`)

Separate from the daily matrix. `effort_points` mines every running effort as a velocity-duration
point (duration, distance, speed, race flag). `velocity_duration_frontier` keeps the best-effort
upper envelope (each kept effort is faster than every longer one): the monotonic speed-vs-duration
curve the model fits. With whole-activity average speeds the frontier is sparse (~5 points); the
per-split data (`splitSummaries`) is the documented way to densify it later.

## Baseline (`features/garmin.py`)

`load_race_predictions` loads Garmin's own daily 5K / 10K / Half / Marathon predictions (seconds),
the baseline the model must beat. `load_reference` and `correlation_report` validate our recomputed
CTL/ATL against Garmin's load series (see [fitness-model.md](fitness-model.md)).
