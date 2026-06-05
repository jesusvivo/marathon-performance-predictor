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

(To be documented as the layers land.)

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
