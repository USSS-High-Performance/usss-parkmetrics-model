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
