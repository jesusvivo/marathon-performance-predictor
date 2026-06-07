# Marathon Performance Predictor

A personal multi-distance race predictor (5K / 10K / Half / Marathon) built on my own Garmin
training data and served end-to-end: BentoML + ONNX behind a Feast feature store, deployed to
GCP Cloud Run with scale-to-zero. The headline output is a marathon finish-time projection with
an uncertainty band, tracked across an 8-month training block and validated against my actual
race in February 2027.

## Why this project

My existing portfolio projects all stop at the notebook. This one is the deployment piece: a
live service with a feature store, online and batch inference, monitoring, and a reproducible
ingestion path for new training data.

## Dataset

The training data is my own Garmin Connect GDPR export: a multisport activity history (running,
cycling, swimming, plus strength and other cardio) starting in **May 2024**. The history depth
matters more than the raw count: CTL/ATL fitness state is an exponentially-weighted load, so the
first ~42 days are a warm-up before the long-term (42-day) average is trustworthy. Only **running**
efforts are prediction labels; the other disciplines feed systemic fatigue.

The set grows with every export, so the live counts (activities ingested, date span, weeks to race)
are not pinned here. They are computed by the pipeline (`python -m marathon.parse`) and surfaced on
the monitoring dashboard, where the countdown to the **Feb 2027 debut marathon** is the through-line.

## Requirements clarification (decided before building)

The scope was pinned up front rather than assumed:

- **Target**: multi-distance race-time prediction with uncertainty, from a personalized
  critical-speed / Riegel velocity-duration curve conditioned on fitness state. Chosen over a
  per-run duration model (low stakes), readiness (weak validation), and injury risk (noisy labels).
- **Fitness state is multisport**: load is computed across running, cycling, and swimming so the
  model sees true systemic fatigue, not running-only load. Labels remain running-only.
- **Evaluation**: per-distance MAPE / RMSPE, never an absolute-minutes average across distances.
- **Serving / features / deploy**: BentoML + ONNX, Feast (offline Parquet, online SQLite), GCP
  Cloud Run scale-to-zero. See the FinOps note below.
- **Baseline to beat**: Garmin's own daily race predictions. **Capstone**: the real Feb 2027 race.

## Architecture

Code layout as it stands:

```
src/marathon/
  parse/         # Garmin summarizedActivities.json -> tidy multisport activity frame
  features/
    load.py      # per-activity training load -> gap-free daily load series
    fitness.py   # daily load -> CTL (fitness) / ATL (fatigue) / TSB (form)
    running.py   # running daily volume, rolling load, pace, efficiency, long-run recency, VO2max
    wellness.py  # sleep and training-readiness daily features
    efforts.py   # velocity-duration effort points + best-effort frontier (model anchors)
    garmin.py    # Garmin's acute/chronic load (validation ref) + race predictions (baseline)
    build.py     # assemble the daily feature matrix from the above
scripts/
  build_features.py     # write daily_features.parquet + effort_points.parquet
  validate_fitness.py   # correlate recomputed CTL/ATL against Garmin's series
docs/
  fitness-model.md      # the sports science: training load, CTL/ATL/TSB, calibration
  features.md           # every column in the daily feature matrix
```

The feature pipeline is a chain of pure functions: `load_export` -> `daily_load` / `running_features`
/ `daily_wellness` -> `daily_features`, each independently tested. The serving layers (ONNX export,
Feast feature store, BentoML service, Cloud Run deploy) follow the build phases in the project plan.

See [docs/features.md](docs/features.md) for the feature columns and
[docs/fitness-model.md](docs/fitness-model.md) for the physiology and Garmin calibration.

## FinOps

Cloud Run scale-to-zero with an immutable SQLite online store baked into the container image,
chosen over App Runner + managed Redis (~$300/yr fixed) for a single-user, sporadically-queried
service. Ingestion runs offline and rebuilds the image, so the live runtime stays read-only and
idles at ~$0.

## Experiment log

- **Phase 1 (scaffold)**: uv project pinned to Python 3.11 (Feast/BentoML compatibility), ruff +
  mypy (strict) + pytest, `src/marathon` layout, staged dependency groups (dev / serving / feast).
- **Phase 2 (parser)**: `marathon.parse` turns `summarizedActivities.json` into a tidy 35-column
  multisport frame, units normalized (distance cm, duration/HR-zones ms, speed (m/s)/10, elevation
  and stride cm), discipline-tagged, deduped by `activityId` (latest start wins). Verified on the
  full export: 581 activities (272 run / 139 strength / 71 bike / 52 swim / 32 cardio), 6 races,
  May 2024 to Jun 2026. `python -m marathon.parse <export.json> <out.parquet>`. 11 unit tests on
  hand-checked fixtures.
- **Phase 3 (features)**: daily multisport training load (`features/load`) and the
  CTL/ATL/TSB fitness state (`features/fitness`) via impulse-response EWMA. Calibrated CTL against
  Garmin's own chronic load by window sweep (`scripts/validate_fitness.py`): a 28-day CTL matches
  Garmin's documented chronic window and lifts correlation from 0.870 (42d) to 0.899; ATL held at
  7 days (r 0.829). Decision rule revised down from the original r >= 0.95 to ~0.90, since Garmin's
  internal per-activity load is not fully observable; documented in docs/fitness-model.md.
  Added running features (`features/running`: rolling 7/28d volume, 28d pace, efficiency factor,
  days-since-long-run, VO2max), wellness (`features/wellness`: sleep, readiness, HRV), velocity-
  duration effort anchors + best-effort frontier (`features/efforts`), and the Garmin race-prediction
  baseline (`features/garmin`). `features/build` assembles a 754-day x 16 feature matrix
  (`scripts/build_features.py`); columns documented in docs/features.md. 32 unit tests.
- **Phase 4 (baseline + eval harness)**: two velocity-duration race-time curves, Riegel (log-log
  least-squares, T = a*D^b) and Critical Speed (linear, distance = D' + CS*time), each with fit +
  predict (`model/curve`). Eval harness (`model/evaluate`): per-distance MAPE/RMSPE, time-ordered
  holdout (each race predicted only from efforts strictly before it, leakage-tested), and Garmin's
  daily race predictions as the baseline. On the 4 real races (`scripts/evaluate_models.py`): Riegel
  MAPE 0.113, CS 0.117, Garmin 0.034. Garmin beats both simple models; both run ~11% slow because the
  frontier is fit on whole-activity average speeds (mostly easy pace), so the curve sits too slow.
  Riegel edges CS, so it is the baseline to beat. Garmin's MAPE is over 3 races (the 20K has no
  standard Garmin distance); the marathon stays unvalidated until the Feb 2027 race. 47 unit tests.
