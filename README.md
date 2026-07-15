# ParkMetrics hit detection

A two-stage on-device system for detecting pipe hits (and eventually other
actions) in real time on a phone strapped to a snowboarder or skier.

- **Stage 1 - trigger** (always on, essentially free): a simple threshold
  check (`g_mag >= 4.0`) on every raw accelerometer sample. When it fires,
  hand a window of surrounding samples to stage 2.
- **Stage 2 - classifier** (rare, only fires when triggered): a small ML
  model looks at aggregated features from the window around the trigger and
  decides whether it was actually a hit, returning the impact timestamp and
  `elapsed_sec` if confirmed.

Full background (data format, labeling methodology, design principles,
open questions) is in [`project_context.md`](project_context.md). This
README covers the toolchain and the current model results.

## Toolchain

| Script | Purpose |
|---|---|
| `hit_detection_pipeline.py` | Builds the labeled per-sample training dataset from raw movement folders + the shared tags file. |
| `build_windowed_features.py` | Turns the per-sample dataset into one row per trigger candidate, writing three feature-set folders (`minimal_5`, `balanced_8`, `full_10`). |
| `train_and_evaluate.py` | Trains `logreg` / `gbm` / `mlp` on each feature set, GroupKFold-CV'd by athlete, plus one held-out-athlete score. Writes `results.json`. |

```
python hit_detection_pipeline.py --root movement_data -t all_tags.csv -o training_dataset.csv
python build_windowed_features.py training_dataset.csv --output-dir windowed_datasets/
python train_and_evaluate.py --datasets-dir windowed_datasets/ --output-json results.json
```

`windowed_datasets/` in this repo holds the three feature-set folders
(`minimal_5`, `balanced_8`, `full_10`), each with `train.csv`/`test.csv`.
Held-out test athlete across all three is Nick Geiser (fewest positive
events); the other three athletes (Cael McCarthy, Jeffery Brown, Piper
Arnold) are cross-validated 3-way.

## Results

Current dataset: 341 trigger candidates, 4 athletes. From `results.json`
(average precision, the right metric for this class imbalance):

| Feature set | Model | Avg CV AP | Held-out AP (Nick Geiser) | Precision @ recall ≥ 0.90 |
|---|---|---|---|---|
| **minimal_5** | **logreg** | **0.732** | **0.861** | **0.74** |
| minimal_5 | gbm | 0.636 | 0.784 | 0.71 |
| minimal_5 | mlp | 0.359 | 0.620 | 0.55 |
| balanced_8 | logreg | 0.719 | 0.851 | 0.65 |
| balanced_8 | gbm | 0.631 | 0.833 | 0.74 |
| balanced_8 | mlp | 0.310 | 0.668 | 0.55 |
| full_10 | logreg | 0.702 | 0.848 | 0.65 |
| full_10 | gbm | 0.659 | 0.854 | 0.74 |
| full_10 | mlp | 0.699 | 0.836 | 0.74 |

### Best model and feature set: `minimal_5` + `logreg`

It has both the highest cross-validated AP (0.732) and the highest
held-out AP (0.861) of any combination tested, using only 5 features
(`peak_g_mag`, `std_g_mag`, `peak_hp8_vertical`, `peak_jerk_mag`,
`max_elevated_dur`). More features (`balanced_8`, `full_10`) don't buy
accuracy here — logreg's AP actually *drops* slightly as features are
added, which suggests the extra features are adding noise rather than
signal at this dataset size.

This result also happens to align with the cheapest deployment path:
logistic regression on 5 features is ~5 multiplications + a sigmoid,
compilable inline in Swift/Kotlin with no ML runtime dependency (see
next step 2(b) below). `gbm` is close behind and remains a reasonable
fallback if a nonlinear boundary turns out to matter once more data
comes in.

**Caveat:** 341 candidates from 4 athletes is a small dataset, and CV
folds on Piper Arnold underperform every other fold across all three
feature sets and all three models (AP 0.49-0.66 vs 0.52-0.86 elsewhere).
This ranking is indicative, not final — treat it as the current best
evidence, not a locked-in decision, until more sessions/athletes are
collected.

## How the `minimal_5` features are calculated

All 5 features are aggregates over a **±500ms window of raw samples
around the peak-`g_mag` sample** of one trigger candidate (see
`extract_window_features` in `build_windowed_features.py`, `WINDOW_MS_DEFAULT
= 500`). "Peak sample" = the sample with the highest `g_mag` within the
contiguous run of samples that were `>= 4.0g` (see the app flow below for
where that run comes from). Per-sample quantities referenced below are
computed in `hit_detection_pipeline.py`, in real time on a causal
(forward-only, past-samples-only) basis so they're reproducible on-device.

| Feature | Definition |
|---|---|
| `peak_g_mag` | `max(g_mag)` over the window. `g_mag` = total accelerometer magnitude in g (`sqrt(ax² + ay² + az²)`), a per-sample input column. |
| `std_g_mag` | `std(g_mag)` over the window (population standard deviation of the same samples used above). |
| `peak_hp8_vertical` | `max(abs(vertical_accel_g_hp8hz_causal))` over the window. `vertical_accel_g` is the component of acceleration along the device's inferred vertical/gravity axis, in g. `_hp8hz_causal` is a 4th-order Butterworth high-pass filter at an 8Hz cutoff, applied causally (`scipy.signal.lfilter`, forward-only — not `filtfilt`, which needs future samples and can't run live). See `add_highpass_columns`. |
| `peak_jerk_mag` | `max(jerk_mag)` over the window. `jerk_mag[i] = sqrt(jerk_ax[i]² + jerk_ay[i]² + jerk_az[i]²)`, where each axis's jerk is a causal backward difference against real elapsed time: `jerk_ax[i] = (ax[i] - ax[i-1]) / (t[i] - t[i-1])`. First sample of a session is set to 0 (no prior point). See `add_jerk_columns`. |
| `max_elevated_dur` | `max(elevated_duration_sec)` over the window. Per sample: `baseline[i]` = causal trailing rolling mean of `g_mag` over the prior 2000ms; `is_elevated[i]` = `g_mag[i] > 2.0 * baseline[i]`; `elevated_duration_sec[i]` = seconds since the current unbroken elevated run started if `is_elevated[i]` is true, else 0 (resets the instant `g_mag` drops back under 2x baseline). See `add_elevated_duration_feature`. |

All of these are deliberately **causal-only** (no `filtfilt`, no
centered/future-looking windows) so that what the model trains on is
exactly what the phone can compute while the stream is still arriving —
see the design principles in `project_context.md`.

## End-to-end app flow

This is how the trigger + classifier participate together on-device, and
why the classifier only ever needs to be trained on samples that already
crossed the trigger threshold:

1. **Stage 1 trigger runs on every incoming sample, always.** Cheap O(1)
   check: `g_mag >= 4.0`. Nothing else happens unless this fires.
2. **Hysteresis groups a burst of consecutive above-threshold samples
   into one trigger candidate**, so one physical impact's rebound
   bounces don't get treated as several separate hits. This is the
   level-plus-duration re-arm scheme specified below ("Trigger hysteresis:
   implementation spec"), not a simple fixed-count/fixed-time gap.
3. **The peak sample within that group** (highest `g_mag`) becomes the
   anchor. The app waits ~300-500ms after the trigger fires so the full
   rebound has landed in the buffer before extracting features.
4. **Stage 2 extracts the 5 `minimal_5` features** from the ±500ms window
   of raw samples around that peak (defined above).
5. **The classifier scores that one feature vector** and returns a hit
   probability:
   ```
   z = -3.439686
       + ( 0.297501 * peak_g_mag)
       + (-0.292572 * std_g_mag)
       + ( 0.120875 * peak_hp8_vertical)
       + (-0.000000 * peak_jerk_mag)
       + (26.518687 * max_elevated_dur)
   probability = 1 / (1 + exp(-z))
   ```
   (Raw-feature-space logistic regression, StandardScaler already folded
   into the coefficients — no separate scaling step needed at inference.
   Fit on all 3 training athletes; see `train_and_evaluate.py`.)
6. **If the probability clears the decision threshold**, the candidate is
   confirmed as a hit and the app reports `impact_timestamp_iso8601` and
   `impact_elapsed_sec` (both taken from the anchor/peak sample) plus
   `impact_source_sample_idx` (its raw index in `motion.csv`, in case a
   later stage needs to reach back into the raw window). If not, the
   candidate is discarded and stage 1 keeps running, unaffected.

**Why training only ever sees trigger candidates:** the classifier's job
is never "is this arbitrary sample a hit" — it's "given that the trigger
already fired, was this really a hit or a false alarm." That's why
`build_windowed_features.py` builds its training rows from exactly the
same `g_mag >= 4.0` condition the on-device trigger uses (see
`TRIGGER_THRESHOLD_DEFAULT` in that file) — every row the model is trained
or evaluated on is a real trigger firing, matching what the phone will
actually hand to the classifier at runtime. A negative row is a false
trigger, not "calm riding data" the model would never see live.

## Trigger hysteresis: implementation spec (Option A)

This is the exact re-arm logic the app should implement, and the one
`build_windowed_features.py` needs to be updated to match (currently it
uses a simpler raw-sample-count gap; this spec is the target, level +
duration based, behavior described in `project_context.md`).

### Constants

| Name | Value | Meaning |
|---|---|---|
| `TRIGGER_THRESHOLD_G` | 4.0 | `g_mag` at/above this fires the trigger. |
| `REARM_THRESHOLD_G` | 2.5 | `g_mag` must drop below this before the trigger can arm again. |
| `REARM_DURATION_MS` | 300 | How long `g_mag` must stay continuously below `REARM_THRESHOLD_G` before re-arming. |
| `FEATURE_WINDOW_MS` | 500 | Half-width of the feature-extraction window around the peak sample (±500ms, i.e. 1 second total). |
| `MAX_SUPPRESSION_MS` | 3000 (proposed, optional) | Safety valve — see note below. Not in the original docs. Confirmed acceptable to skip entirely for jump detection specifically: two distinct jumps can't occur within a few hundred ms of each other, so unbounded suppression until the level-based re-arm condition is met carries no real risk of swallowing a genuine second hit. Keep it only if this trigger logic ever gets reused for something with faster repeat events. |

### State machine

Two states: **ARMED** and **SUPPRESSED**. Runs on every incoming sample,
same cost as the plain threshold check (no extra per-sample work beyond a
couple of comparisons and, while suppressed, tracking a running max).

```
state = ARMED
candidate = null          # { peak_g_mag, peak_timestamp, peak_sample_idx }
below_rearm_since = null  # timestamp when g_mag most recently dropped below REARM_THRESHOLD_G

on each incoming sample (t, g_mag, sample_idx):
    buffer_sample(t, g_mag, sample_idx, ...)   # always append to the raw ring buffer

    if state == ARMED:
        if g_mag >= TRIGGER_THRESHOLD_G:
            state = SUPPRESSED
            candidate = { peak_g_mag: g_mag, peak_timestamp: t, peak_sample_idx: sample_idx }
            below_rearm_since = null

    elif state == SUPPRESSED:
        # keep tracking the true peak — the initial trigger sample is not
        # necessarily the hardest hit; a rebound bounce can be bigger.
        if g_mag > candidate.peak_g_mag:
            candidate = { peak_g_mag: g_mag, peak_timestamp: t, peak_sample_idx: sample_idx }

        if g_mag < REARM_THRESHOLD_G:
            if below_rearm_since is null:
                below_rearm_since = t
            elif (t - below_rearm_since) >= REARM_DURATION_MS:
                finalize_candidate(candidate)   # see "Feature extraction timing" below
                state = ARMED
                candidate = null
                below_rearm_since = null
        else:
            below_rearm_since = null   # level came back up; reset the countdown

        # safety valve: don't let one long high-energy stretch (e.g.
        # sustained rough terrain) suppress the trigger indefinitely
        if (t - candidate.peak_timestamp) >= MAX_SUPPRESSION_MS:
            finalize_candidate(candidate)
            state = ARMED
            candidate = null
            below_rearm_since = null
```

### Feature extraction timing

`finalize_candidate` doesn't have to run the classifier immediately — it
just means the candidate's peak is now known for certain (nothing later
can retroactively beat it, since we've re-armed). The classifier still
needs `FEATURE_WINDOW_MS` (500ms) of buffered samples *after*
`candidate.peak_timestamp` to build the full ±500ms window, which by
construction is already true by the time `finalize_candidate` runs in
virtually all real cases (re-arming requires 300ms below 2.5g, which
almost always lands after the peak + 500ms mark) — but guard it
explicitly rather than assume:

```
def finalize_candidate(candidate):
    wait_until(now >= candidate.peak_timestamp + FEATURE_WINDOW_MS)
    window = buffer.slice(candidate.peak_timestamp - FEATURE_WINDOW_MS,
                           candidate.peak_timestamp + FEATURE_WINDOW_MS)
    features = extract_minimal_5(window, candidate)   # the 5 features defined above
    probability = classify(features)                   # the logistic regression equation above
    if probability >= DECISION_THRESHOLD:
        report_hit(candidate.peak_timestamp, candidate.peak_elapsed_sec, candidate.peak_sample_idx)
```

### Practical notes for the app implementation

- **Ring buffer sizing:** must hold at least `MAX_SUPPRESSION_MS +
  FEATURE_WINDOW_MS` of raw samples (~3.5s at the values above) so the
  window slice is always available when `finalize_candidate` runs.
- **Detection latency:** a confirmed hit is reported at least
  `REARM_DURATION_MS + FEATURE_WINDOW_MS` (~800ms) after the physical
  impact, worst case up to `MAX_SUPPRESSION_MS + FEATURE_WINDOW_MS` for a
  long rebound. Budget for this if the app surfaces hits live (e.g. a
  toast/haptic) rather than only in a post-session summary.
- **`DECISION_THRESHOLD`:** default 0.5, but `results.json`'s
  `recall_tuned_threshold` gives a lower, higher-recall cutoff
  (precision 0.74 at recall ≥ 0.90 for `minimal_5` + `logreg`) if missed
  hits are worse than false positives for this app.
- This state machine is what `build_windowed_features.py`'s
  `find_trigger_groups` should be rewritten to match, so the trigger
  candidates the model is trained on are produced by the identical logic
  the app runs live — ideally both load `TRIGGER_THRESHOLD_G`,
  `REARM_THRESHOLD_G`, `REARM_DURATION_MS`, and `FEATURE_WINDOW_MS` from
  one shared config file rather than each hand-coding their own copies.

## Session load metric (proposed, not yet implemented)

Separate from hit detection: a running total of physical load accumulated
over a whole session, counted only while the athlete is going downhill
(excluding lift rides, walking, standing around). This hasn't been built
or validated against data yet — it's a design proposal for the app to
implement, laid out here for reference.

### Recommended metric: accumulated jerk-based load

```
session_load += jerk_mag[i] * dt[i]     # only while downhill_gate[i] is true
```

Where `jerk_mag` is the same quantity already computed in
`hit_detection_pipeline.py` (`add_jerk_columns`): `sqrt(jerk_ax² + jerk_ay²
+ jerk_az²)`, the causal rate-of-change of raw acceleration. `dt[i]` is the
real elapsed time since the previous sample, in seconds — multiplying by
`dt` (rather than just summing samples) makes the total independent of the
device's native sampling rate (100-150Hz, varies by phone), so two
identical runs recorded on different phones score the same.

**Why jerk rather than raw `g_mag` or gyroscope magnitude:** `g_mag` alone
conflates "held at 2g through one long carved turn" with "one sharp
jolt" — same average, very different physical load. Jerk (the *rate of
change* of acceleration) is closer to what actually stresses the body:
impacts, chatter, sudden edge sets. This is the same style of metric
sports-science IMU systems use for athlete workload monitoring (e.g.
Catapult's "PlayerLoad": accumulated `sqrt(Δax² + Δay² + Δaz²)`).

### Optional second metric: rotational load

If rotation-heavy stress (spins, rapid edge-to-edge angular velocity)
matters separately from translational/impact stress, track it as its own
number rather than merging it into `session_load`:

```
rotational_load += gyro_mag[i] * dt[i]     # only while downhill_gate[i] is true
```

`gyro_mag` = magnitude of the raw gyroscope vector (not yet a column in
this pipeline; would need to be added analogous to `g_mag`). Accelerometer
and gyroscope jerk/rate quantities represent physiologically different
kinds of load, so keeping them as two separate totals (rather than one
combined score) preserves that distinction rather than muddying it.

### The "only downhill" gate

No existing signal in this pipeline is validated for this yet. Candidate
approaches, roughly in order of expected reliability:

1. **`altitude.csv`** (currently unused everywhere else in this pipeline)
   — gate on a smoothed/trailing altitude trend being negative
   (descending) over some window (e.g. 5-10s), to avoid noise flipping the
   gate on small bumps or brief stops.
2. **The phone's own `run_state`** signal. This was dropped from the *hit
   classifier's training features* (see `DROP_FROM_MOTION_COLUMNS` in
   `hit_detection_pipeline.py`) because training the classifier on it
   would just teach it to imitate the phone's existing deterministic
   model — that objection doesn't apply here, since this is an
   operational gate, not a training feature. If `run_state` already
   reliably flags "on a downhill run" vs. lift/flat, it's likely the
   simplest signal to reuse.
3. A speed-based proxy, if GPS/speed data is available in a form not yet
   reflected in this repo.

Before committing to one of these, it's worth checking a sample session's
`altitude.csv` and `run_state` values to see which is actually reliable —
neither has been validated against ground truth for this purpose yet.

## Open questions / next steps

1. Get more sessions - 341 candidates isn't enough to draw model
   selection conclusions with confidence.
2. Port the chosen model on-device. Two paths: (a) CoreML/TFLite export
   for gbm/mlp, or (b) compile logreg to inline Swift/Kotlin, no ML
   runtime needed - the path the current results favor.
3. Write a shared config file (trigger threshold, hysteresis, window
   size, feature list) that both the offline pipeline and the on-device
   code load from, so they can't silently drift.
4. Investigate adaptive trigger thresholds (e.g. 3x recent baseline)
   once there's more data, since baseline `g_mag` isn't a fixed 1.0.
5. Investigate why Piper Arnold's folds underperform.
6. Build the post-classification stage (rotation amount, jump
   magnitude) - likely a regression model on a longer window.

See `project_context.md` for full detail on all of the above.
