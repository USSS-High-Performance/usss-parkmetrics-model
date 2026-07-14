# ParkMetrics hit detection: project context

Comprehensive brief on everything decided and built so far. Written to
be dropped into Claude Code as the starting context for the repo.


## What we're building

A two-stage on-device system for detecting pipe hits (and eventually
other actions) in real time on a phone strapped to a snowboarder or
skier. The two stages exist because a full ML model can't run 100+
times per second on a phone battery, but a cheap threshold can.

Stage 1 - trigger (always on, essentially free):
  A simple threshold check on every sample of the raw accelerometer
  stream. When it fires, hand a window of surrounding samples to
  stage 2. Otherwise do nothing.

Stage 2 - classifier (rare, only fires when triggered):
  A small ML model that looks at aggregated features from the ~1 second
  window around the trigger firing and decides whether it was actually
  a hit. Returns the impact timestamp and elapsed_sec if confirmed.

Later stages (not yet built): given a confirmed hit, estimate rotation
amount, jump magnitude, etc. Different problem, likely different model.


## Data

Motion sensor data comes from a phone strapped to the athlete. Native
sampling rate is 100-150Hz (varies by device). Ground truth impact tags
are collected separately in a shared file with athlete name and full
millisecond-precision UTC timestamps.

Session data lives in per-session folders, each containing:
  motion.csv      - the primary ML input (accel, gyro, quaternion, etc.)
  events.csv      - device-generated diagnostics, not used for labeling
  altitude.csv    - not used
  metadata.json   - athlete_name, session_id, motion_sampling_hz

Ground truth tags live in ONE separate shared file (columns:
session_id, athlete, action, iso_time). Not in per-folder events.csv.

Data quality issue: session_id in the tags file is unusable (Excel
collapsed it to scientific notation, every row shows 2.03E+13).
Matching relies on athlete name + tag time falling within a given
motion file's own recorded span.


## The offline pipeline (already built)

Turns raw motion folders + shared tags file into one combined labeled
training dataset. See `hit_detection_pipeline.py`.

Steps:
  1. For each folder, load motion.csv, dedupe millisecond-collision
     timestamps by adding microseconds, drop `run_state`, `jump_state`,
     `rail_state` (deterministic model outputs we don't want to train
     on), and `session_id`.
  2. Filter shared tags file to this folder's athlete + time range.
  3. Cluster the tags into "attentive runs" (consecutive tags within
     60 seconds count as one run, padded 3 seconds each side). This
     defines when the annotator was demonstrably watching.
  4. Find every candidate g_mag peak in the whole session (scipy
     find_peaks, height >= 2.0g, min spacing 300ms).
  5. Match each tag to a candidate peak within +/- 2 seconds, resolved
     by peak magnitude first, time proximity as tiebreaker. This was
     an important fix: the tagger's reaction time introduces up to
     1.5s of lag, and true impacts are physically enormous compared
     to anything else nearby, so magnitude is a more reliable signal
     of which peak is the real event.
  6. Three-way label every sample:
       event    - the single matched-peak sample per tag (one row)
       negative - inside an attentive run span, not the peak
       excluded - outside any attentive run span (unknown ground truth,
                  dropped from training rather than mislabeled negative)
  7. Feature engineering: rolling RMS/SMA at 150/400ms windows,
     elevated_duration (how long g_mag has stayed >2x its baseline),
     high-pass filters at 2Hz and 8Hz (both zero-phase and causal
     versions, causal is the deployment-honest one), jerk features.
  8. Combined across folders, robust-scaled columns added, written out
     as one CSV.

The combined training dataset that came out of this: 43,657 rows,
120 event rows, 43,537 negative rows, 4 athletes, 7 sessions, ~72
columns before feature selection.


## Where we are now: model building

### Stage 1 trigger

Decided threshold: `g_mag >= 4.0`.

Per-sample stats:
  - Recall: 100% (all 120 events fire the trigger)
  - Precision: 4.74% (2411 false positive rows out of 2531 firings)
  - Trigger rate within attentive runs: 5.80%

Per-distinct-trigger-candidate stats (after hysteresis collapses
adjacent firings from a single physical impact into one):
  - 341 distinct candidates across all sessions
  - 127 real hits, 214 false positives
  - Candidate-level precision: 37.24% (this is the number that
    matters for the classifier's input)

Deployment specifics:
  - Trigger runs on every sample (O(1) check, effectively free).
  - Hysteresis: once triggered, ignore new triggers until g_mag drops
    below ~2.5g for at least 300ms. Prevents one physical hit's
    rebound bounces from firing 3-4 classifications for the same event.
  - Trigger fires the classifier after a ~300-500ms delay so the full
    rebound has landed inside the buffer window before feature
    extraction runs.

### Stage 2 classifier

Insight that changed everything: framing the problem as per-sample
classification (score every 10ms of data) gave mediocre results
(cross-athlete AP 0.18-0.39). Framing it correctly as "given a
trigger candidate, is this a real hit?" gave much better results
(cross-athlete AP 0.60-0.85) and matches what the phone actually does.

For each trigger candidate, the classifier sees ONE fixed-size
feature vector aggregated over the +/- 500ms window around the peak
g_mag sample. It returns a probability. If confirmed, we report:
  - impact_timestamp_iso8601 (peak sample's original timestamp)
  - impact_elapsed_sec (peak sample's elapsed_sec, session-anchored)
  - impact_source_sample_idx (raw sample index in motion.csv, useful
    if a later tool wants to reach back into the raw data)

### Feature engineering for the classifier

Feature importance from the per-sample RF, ordered:
  1. peak_g_mag        (0.20)
  2. is_elevated       (0.19)
  3. planar_accel_g    (0.08)  ← highly correlated with g_mag, r=0.87, drop
  4. jerk_mag          (0.075)
  5. elevated_duration (0.065)
  ...

For the windowed classifier, aggregate features from the +/- 500ms
window rather than raw values. Three progressively larger sets:

  minimal_5:  peak_g_mag, std_g_mag, peak_hp8_vertical, peak_jerk_mag,
              max_elevated_dur
  balanced_8: minimal_5 + peak_sma150, peak_rms150, peak_hp8_planar
  full_10:    balanced_8 + pitch_range, rot_x_range

All three built by `build_windowed_features.py`. Held-out test athlete
(the one with fewest positive events) is Nick Geiser. Remaining three
athletes make up the training set, cross-validated 3-way by athlete.

### Model architectures to test

logreg: L2 logistic regression, class-weighted. The floor. Should be
  essentially free on-device (5-10 multiplications + sigmoid). If a
  tree ensemble doesn't beat this by a meaningful margin, use this.

gbm: GradientBoostingClassifier, 50 trees, max_depth=3, lr=0.1.
  Recommended production model. Exports cleanly to CoreML or as
  compiled decision code, sub-millisecond inference. Handles the
  class imbalance and nonlinearity that logreg can't.

mlp: MLPClassifier, one hidden layer, 12 units, early stopping.
  Similar accuracy ceiling to gbm but more overfitting risk given
  only 341 training candidates.

All three include a StandardScaler upstream (essential for logreg/mlp,
harmless for gbm).

### Preliminary results (with the current 341-candidate dataset)

Held-out AP on Nick Geiser, best per feature set:
  minimal_5:  logreg 0.86, gbm 0.78, mlp 0.62
  balanced_8: logreg ~0.83, gbm ~0.83, mlp ~0.84
  full_10:    logreg 0.85, gbm 0.85, mlp 0.84

Real caveat: 341 candidates from 4 athletes is a small dataset and
this ranking may not be stable. Numbers are indicative, not
definitive. More sessions/athletes will likely reshuffle things.

Also: cross-validation folds on Piper Arnold routinely underperform
other folds (AP ~0.55-0.66 vs 0.70-0.80 for others). Worth checking
whether Piper's tagging pattern differs from the others once you have
more data.


## Files that make up the toolchain

### hit_detection_pipeline.py
Builds the labeled per-sample training dataset from movement folders +
shared tags file. Multi-folder, discovers sessions under a root
directory. Full details in its module docstring.

Usage:
    python hit_detection_pipeline.py --root movement_data \
        -t all_tags.csv -o training_dataset.csv

### build_windowed_features.py
Turns the per-sample training dataset into windowed trigger candidates,
writes three feature-set folders (minimal_5, balanced_8, full_10) each
with train.csv and test.csv, plus an all_candidates.csv reference.

Usage:
    python build_windowed_features.py training_dataset.csv \
        --output-dir windowed_datasets/

### train_and_evaluate.py
Trains logreg, gbm, mlp on each feature set, GroupKFold CV over
training athletes + one final score on held-out athlete. Writes
results.json.

Usage:
    python train_and_evaluate.py --datasets-dir windowed_datasets/ \
        --output-json results.json


## Open questions / next steps

1. Get more sessions. 341 candidates is not enough to draw model
   selection conclusions with confidence.

2. Once a model is chosen: port it to on-device. Two paths worth
   considering: (a) CoreML/TFLite export for gbm/mlp, or (b) compile
   logreg or a small tree to inline Swift/Kotlin, no ML runtime
   needed. Path (b) is more work but keeps the app dependency-free.

3. Match the offline pipeline to what the phone will actually see.
   The classifier is being trained on causal high-pass filters, which
   is correct. But the trigger and hysteresis behavior needs to match
   too: same 4.0g threshold, same 300ms drop-to-2.5g before re-arming.
   Worth writing a shared config file (trigger threshold, hysteresis,
   window size, feature list) that both the offline pipeline and the
   on-device code load from, so they can't silently drift.

4. Baseline drift: baseline g_mag isn't a fixed 1.0. It's higher
   during high-frequency riding (~1.3g median), lower during still
   moments. On-device, the trigger threshold is currently absolute
   (4.0g). Once we have more data, worth checking whether an adaptive
   threshold (e.g. 3x recent baseline) recalls just as well while
   firing less often during high-energy riding stretches.

5. Post-classification stage (rotation amount, jump magnitude): not
   built yet, likely wants a different model (regression, not
   classification) on possibly a longer window (the whole airtime,
   not just the landing spike). A CNN on raw window samples starts
   making sense here once there's enough data.


## Design principles applied throughout

- Keep the label as a single timestamp per event, not a widened
  window. Window width becomes a downstream decision per model.
- Three-way labels (event / negative / excluded) beat two-way,
  because "we don't know" is a real state that must not get called
  negative.
- Train on the causal versions of filtered/derivative features,
  match what the phone can actually compute in real time. Zero-phase
  filters are for offline exploration and comparison, not deployment.
- GroupKFold by athlete for every cross-validation, not random split.
  A jump signature from one athlete appearing in both train and test
  would let the model memorize movement style rather than learn
  what a hit actually looks like.
- Drop the phone's own existing deterministic model outputs
  (`run_state`, `jump_state`, `rail_state`) from training data.
  Training on them teaches a new model to imitate the old one's
  mistakes.
- Robust scaling (median/IQR), not mean/std. Impact spikes are
  extreme legitimate outliers; a mean/std scaler would have its
  scale distorted by them.
