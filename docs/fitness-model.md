# Fitness model: training load and CTL / ATL / TSB

How the feature pipeline turns raw activities into a daily "fitness state". The physiology
here is the standard impulse-response model used by TrainingPeaks and Garmin; this doc records
what we implement and how it was calibrated against Garmin's own numbers.

## Training load (per activity)

Each activity carries Garmin's `activityTrainingLoad`: a single number for how much stress the
session imposed, derived from the session's EPOC (excess post-exercise oxygen consumption), so
it is already comparable across running, cycling, and swimming. It is present and positive for
every activity in the export, so no fallback metric (e.g. TRIMP from heart-rate zones) is needed.

We keep load from **all** disciplines, not just running, because fatigue is systemic: a hard bike
or swim leaves you tired for a run. The two `multi_sport` parent records are excluded, because
their child legs are already counted and would otherwise double up.

## Daily load series

Per-activity loads are summed by **local calendar day** (the day you experienced the session, not
the UTC instant), then placed on a gap-free daily calendar where **rest days are explicit zeros**.
The zeros matter: the smoothing below decays toward zero on days off, so a missing row would wrongly
hold fitness flat through a rest week.

## CTL, ATL, TSB

Three exponentially weighted moving averages (EWMAs) of the daily load series:

- **CTL (Chronic Training Load) = fitness.** A slow, long-memory average. Rises with sustained
  training; the durable base you build over months.
- **ATL (Acute Training Load) = fatigue.** A fast, short-memory average. Spikes after hard days
  and decays quickly with rest.
- **TSB (Training Stress Balance) = form / freshness = CTL - ATL**, read from the **previous day**
  (today's freshness reflects training through yesterday). Positive TSB = rested and race-ready,
  negative TSB = carrying fatigue from a heavy block.

The EWMA uses the impulse-response weighting `alpha = 1 - exp(-1 / days)` with `adjust=False`, so
each day is exactly `alpha * load_today + (1 - alpha) * value_yesterday`, seeded with the first
day's load. This is the physiology convention; pandas' `halflife` and `span` options are different
weightings and would not match Garmin.

The first ~ (CTL window) days are a warm-up: the long average has not yet "filled", so early CTL
values understate true fitness. The data starts in May 2024, which sets that warm-up floor.

A related quantity Garmin reports is the **ACWR (acute:chronic workload ratio)**, ATL / CTL: a
ramp-rate / injury-risk heuristic (spiking acute load well above chronic is the classic overtraining
signal). We do not model injury risk, but it is the same two series under the hood.

## Calibration against Garmin

Garmin's export includes its own daily `dailyTrainingLoadAcute` (ATL-like) and
`dailyTrainingLoadChronic` (CTL-like) in `MetricsAcuteTrainingLoad_*.json`. We validate our
recompute against it with **Pearson correlation** on overlapping days (`correlation_report` in
`features/garmin.py`). Correlation, not equality, because Garmin's load is in different units
(it reads like a weighted sum, ours is a weighted average) and **correlation is invariant to that
linear rescale**, so it measures whether the *shape* of the fitness curve agrees. This also means
the sum-vs-mean choice does not affect correlation; only the time-constant (window) does.

Window sweep over 754 overlapping days (correlation vs Garmin's series):

| ATL window | r vs acute | | CTL window | r vs chronic |
|---|---|---|---|---|
| 5  | 0.802 | | 21 | 0.902 |
| **7**  | **0.829** | | **28** | **0.899** |
| 10 | 0.833 | | 35 | 0.886 |
| 14 | 0.818 | | 42 | 0.870 |

Findings:

- **ATL = 7 days** holds up (flat 7-10, matching Garmin's documented acute window).
- **Garmin's "chronic" tracks a ~21-28 day memory, not the TrainingPeaks-classic 42.** A 28-day
  CTL (Garmin's own documented chronic window) lifts correlation from 0.870 to 0.899 and is the
  principled pick; 21 is marginally higher (0.902) but is fitting noise rather than a documented model.
- Correlation tops out near **0.90**, not perfect, because Garmin's internal per-activity load and
  multisport weighting are not fully visible to us. We accept ~0.90 with the documented windows as
  "tracks Garmin's fitness signal", rather than chasing the last points by overfitting windows.

## References

- TrainingPeaks, "Applying the Numbers: Coggan's CTL/ATL/TSB" (impulse-response / Performance
  Manager Chart).
- Banister, E.W., impulse-response model of training effect.
- Garmin, Acute Load / Chronic Load / Acute:Chronic Workload Ratio documentation.
