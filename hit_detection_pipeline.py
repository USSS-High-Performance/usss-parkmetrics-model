"""
Hit detection labeling and feature engineering pipeline, multi folder version.

Each "movement folder" is a self-contained session export containing:
    motion.csv     - accelerometer/gyroscope stream, the primary ML input
    events.csv     - device generated events (not used by this pipeline)
    altitude.csv   - not used by this pipeline
    metadata.json  - athlete_name, session_id, motion_sampling_hz, etc.

Ground truth tags come from a SEPARATE shared file (not events.csv),
covering many athletes and sessions at once, with columns including:
    session_id, athlete, action, iso_time

One known data quality issue in that tags file is handled explicitly:
session_id is not usable. In the source file every row shows the
identical value 2.03E+13, which is Excel's scientific notation having
collapsed a much longer real session_id down to 3 significant figures.
Matching therefore relies only on athlete name plus whether a tag's time
falls within a given motion file's own recorded span. iso_time itself
carries full millisecond precision with a "Z" (UTC) suffix and parses
directly, matching motion.csv's own UTC timestamps with no conversion.

A folder with no tags matching its athlete/time span contributes nothing
and is skipped, with a printed note, rather than treated as an error.

Usage:
    from hit_detection_pipeline import build_dataset
    df = build_dataset(
        ["/path/to/session_1", "/path/to/session_2"],
        tags_path="/path/to/verified_tags.csv",
    )
    df.to_csv("training_dataset.csv", index=False)

Or from the command line:
    python hit_detection_pipeline.py /path/to/session_1 /path/to/session_2 \\
        -t /path/to/verified_tags.csv -o out.csv
"""

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt, lfilter

# ---------------------------------------------------------------------------
# Tunable parameters
# ---------------------------------------------------------------------------

RUN_CLUSTER_GAP_SEC = 60.0        # tags farther apart than this start a new "attentive run"
RUN_CLUSTER_PAD_SEC = 3.0         # extend each run's attentive span this much before/after its first/last label

PEAK_MIN_HEIGHT_G = 2.0           # candidate peaks in g_mag must exceed this to even be considered
PEAK_MIN_SPACING_SEC = 0.3        # candidate peaks must be at least this far apart (dedupe noisy double counts)

# The tags file has full millisecond precision, but the tagger's own
# reaction time still lags the true impact, measured at up to about 1.5
# seconds in either direction against a sample session. This is the
# search window around each tag's exact timestamp; the magnitude-priority
# matching below picks the largest unclaimed spike inside it, so nearby
# back to back hits still resolve to the correct one each.
POINT_LABEL_SEARCH_BEFORE_SEC = 2.0
POINT_LABEL_SEARCH_AFTER_SEC = 2.0

HIGHPASS_CUTOFFS_HZ = [2.0, 8.0]     # mild and aggressive high pass variants
HIGHPASS_ORDER = 4
FILTER_COLUMNS = ["ax", "ay", "az", "vertical_accel_g", "planar_accel_g"]

ROLLING_WINDOWS_MS = [150, 400]      # short window ~= impact + rebound, longer window for broader context
BASELINE_WINDOW_MS = 2000            # trailing window used to establish "normal" g_mag level
ELEVATED_MULTIPLIER = 2.0            # how far above baseline counts as "elevated" for the duration feature

JERK_SOURCE_COLUMNS = ["ax", "ay", "az"]

DROP_FROM_MOTION_COLUMNS = ["run_state", "jump_state", "rail_state", "session_id"]
# run_state / jump_state / rail_state are the phone's own existing
# deterministic model output, not ground truth; training on them would
# just teach a new model to imitate the old one's mistakes, not to detect
# hits independently. session_id is dropped from the output dataset too:
# it's a per-session identifier, not a feature, and source_folder (added
# later per row) already identifies which session a row came from, so
# nothing is lost by dropping it here.

# Columns that get a scaled (robust z-score) version added at the end, once
# all folders are combined. Robust scaling (median / IQR) is used instead of
# mean/std because impact spikes are extreme, legitimate outliers, exactly
# the samples we care most about, and a mean/std scaler would have its scale
# distorted by the very events we want to keep sharp.
SCALE_CANDIDATE_COLUMNS = [
    "g_mag", "ax", "ay", "az", "vertical_accel_g", "planar_accel_g",
    "jerk_ax", "jerk_ay", "jerk_az", "jerk_mag",
]


# ---------------------------------------------------------------------------
# Folder loading
# ---------------------------------------------------------------------------

def load_metadata(folder):
    with open(Path(folder) / "metadata.json") as f:
        return json.load(f)


def discover_session_folders(root):
    """
    Scans immediate subdirectories of root and returns the ones that
    actually look like a session folder (contain motion.csv). Anything
    else under root, stray files, folders missing motion.csv, is skipped
    silently. Does not search deeper than one level.
    """
    root = Path(root)
    folders = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "motion.csv").exists():
            folders.append(str(child))
    return folders


def load_motion(folder):
    df = pd.read_csv(Path(folder) / "motion.csv")
    df["ts"] = pd.to_datetime(df["timestamp_iso8601"])
    df = df.sort_values("ts").reset_index(drop=True)

    # Millisecond-resolution timestamps occasionally repeat for consecutive
    # samples. Nudge exact duplicates by a microsecond per repeat so every
    # row has a distinct, strictly increasing timestamp. Needed because
    # derivative features (jerk) and time-based rolling windows both divide
    # by elapsed time, and a zero delta breaks that.
    dup_mask = df["ts"].duplicated(keep=False)
    if dup_mask.any():
        counters = df.groupby("ts").cumcount()
        df.loc[dup_mask, "ts"] = df.loc[dup_mask, "ts"] + pd.to_timedelta(
            counters[dup_mask], unit="us"
        )

    df = df.drop(columns=[c for c in DROP_FROM_MOTION_COLUMNS if c in df.columns])

    return df


def load_tags_file(tags_path):
    """
    Loads the shared, verified tags file (one file, covers many athletes
    and many sessions/dates). This is the ONLY source of ground truth
    tags; each folder's own events.csv is not used for labeling.

    Known data quality issue in this file, handled here:

    - session_id is not usable. Every row shows the identical value
      2.03E+13, which is Excel's scientific notation having collapsed a
      much longer real session_id down to 3 significant figures. There's
      no way to recover the original value, so matching does not use
      session_id at all, only athlete name plus whether the tag's time
      falls within a given motion file's own recorded span.

    iso_time carries full millisecond precision with an explicit "Z"
    (UTC) suffix, so it parses directly and is already timezone-aware,
    matching motion.csv's own UTC timestamps with no conversion needed.
    """
    tags = pd.read_csv(tags_path)
    tags["ts"] = pd.to_datetime(tags["iso_time"], errors="coerce")

    bad = tags["ts"].isna()
    if bad.any():
        print(f"  [warning] tags file: {bad.sum()} row(s) had an unparseable iso_time and were dropped.")
        tags = tags[~bad].copy()

    return tags


def select_tags_for_folder(tags_df, athlete_name, motion_df):
    """
    Filters the shared tags file down to the rows relevant to one
    folder: matching athlete name, and falling within that motion
    file's own recorded time span (widened by the same search buffers
    used for matching, so a genuine tag near the very start or end of
    a session isn't excluded just for landing close to the edge).
    """
    t_min = motion_df["ts"].min() - pd.Timedelta(seconds=POINT_LABEL_SEARCH_BEFORE_SEC)
    t_max = motion_df["ts"].max() + pd.Timedelta(seconds=POINT_LABEL_SEARCH_AFTER_SEC)

    mine = tags_df[
        (tags_df["athlete"] == athlete_name) &
        (tags_df["ts"] >= t_min) &
        (tags_df["ts"] <= t_max)
    ].sort_values("ts").reset_index(drop=True)

    return mine


def build_label_windows(folder_tags):
    """
    Converts the filtered tag rows for one folder into label windows
    (label_type, search_start, search_end), the same structure the
    rest of the pipeline (clustering, peak matching) expects.

    Every row here is a point tag (this tags file has no start/end
    range columns), so each gets the asymmetric minute-precision search
    window described above, centered on its recorded minute.
    """
    if len(folder_tags) == 0:
        return pd.DataFrame(columns=["label_type", "search_start", "search_end", "notes"])

    records = []
    for _, row in folder_tags.iterrows():
        records.append({
            "label_type": row["action"],
            "search_start": row["ts"] - pd.Timedelta(seconds=POINT_LABEL_SEARCH_BEFORE_SEC),
            "search_end": row["ts"] + pd.Timedelta(seconds=POINT_LABEL_SEARCH_AFTER_SEC),
            "notes": "",
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Run clustering ("was the annotator plausibly watching this stretch")
# ---------------------------------------------------------------------------

def cluster_labels(label_windows_df, gap_sec=RUN_CLUSTER_GAP_SEC, pad_sec=RUN_CLUSTER_PAD_SEC):
    """
    Groups labels into runs based on gaps between consecutive labels
    (using each label window's midpoint). Returns a list of (start, end)
    attentive spans, padded slightly. Same logic regardless of whether
    a label came from a point tag or an explicit start/end window.
    """
    if len(label_windows_df) == 0:
        return []

    midpoints = (
        label_windows_df["search_start"]
        + (label_windows_df["search_end"] - label_windows_df["search_start"]) / 2
    ).sort_values().tolist()

    clusters = [[midpoints[0], midpoints[0]]]
    for t in midpoints[1:]:
        gap = (t - clusters[-1][1]).total_seconds()
        if gap <= gap_sec:
            clusters[-1][1] = t
        else:
            clusters.append([t, t])

    padded = [
        (start - pd.Timedelta(seconds=pad_sec), end + pd.Timedelta(seconds=pad_sec))
        for start, end in clusters
    ]
    return padded


# ---------------------------------------------------------------------------
# Candidate peak detection and label-to-peak matching
# ---------------------------------------------------------------------------

def find_candidate_peaks(motion_df, height=PEAK_MIN_HEIGHT_G, min_spacing_sec=PEAK_MIN_SPACING_SEC):
    g = motion_df["g_mag"].values
    median_dt = motion_df["ts"].diff().dt.total_seconds().median()
    min_distance_samples = max(1, int(min_spacing_sec / median_dt))

    peak_idx, _ = find_peaks(g, height=height, distance=min_distance_samples)
    candidates = motion_df.iloc[peak_idx][["ts", "g_mag"]].copy()
    candidates["motion_idx"] = peak_idx
    return candidates.reset_index(drop=True)


def match_labels_to_peaks(label_windows_df, candidates):
    """
    Assignment between labels and candidate peaks, prioritized by peak
    magnitude within each label's search window (not by time proximity
    to the window's edges or midpoint). A true impact is physically
    enormous compared to anything else nearby, so the largest candidate
    inside the trusted window is the far more reliable signal of which
    one is the real event. Each candidate peak can only be claimed once.
    """
    pairs = []
    for li, lrow in label_windows_df.iterrows():
        window = candidates[
            (candidates["ts"] >= lrow["search_start"]) & (candidates["ts"] <= lrow["search_end"])
        ]
        center = lrow["search_start"] + (lrow["search_end"] - lrow["search_start"]) / 2
        for ci, crow in window.iterrows():
            offset = abs((crow["ts"] - center).total_seconds())
            pairs.append((crow["g_mag"], offset, li, ci))

    pairs.sort(key=lambda x: (-x[0], x[1]))

    used_labels, used_candidates = set(), set()
    matches = []
    for g_mag, offset, li, ci in pairs:
        if li in used_labels or ci in used_candidates:
            continue
        used_labels.add(li)
        used_candidates.add(ci)
        matches.append({
            "label_index": li,
            "label_type": label_windows_df.loc[li, "label_type"],
            "matched_motion_idx": int(candidates.loc[ci, "motion_idx"]),
            "matched_peak_time": candidates.loc[ci, "ts"],
            "matched_peak_g_mag": candidates.loc[ci, "g_mag"],
        })

    unmatched = label_windows_df.loc[~label_windows_df.index.isin(used_labels)]
    return pd.DataFrame(matches), unmatched


# ---------------------------------------------------------------------------
# Labeling
# ---------------------------------------------------------------------------

def label_motion(motion_df, matches_df, clusters):
    """
    Three way label:
      event      - the single matched-peak sample, one row per label
      negative   - inside an attentive run span, not the peak itself
      excluded   - outside any attentive run span (unknown whether the
                   annotator was watching; not safe to call negative)
    """
    labels = np.full(len(motion_df), "excluded", dtype=object)

    for start, end in clusters:
        mask = (motion_df["ts"] >= start) & (motion_df["ts"] <= end)
        labels[mask.values] = "negative"

    label_type_col = np.full(len(motion_df), "", dtype=object)
    for _, row in matches_df.iterrows():
        labels[row["matched_motion_idx"]] = "event"
        label_type_col[row["matched_motion_idx"]] = row["label_type"]

    out = motion_df.copy()
    out["Tag"] = labels
    out["label_type"] = label_type_col
    return out


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _causal_rolling_mean(df, value_series, window_ms):
    """Time-based trailing (causal) rolling mean, robust to variable or
    slightly irregular sampling rates since it windows on real elapsed
    time rather than a fixed sample count."""
    tmp = pd.DataFrame({"ts": df["ts"], "v": value_series})
    return tmp.rolling(window=f"{window_ms}ms", on="ts")["v"].mean().values


def add_rolling_features(motion_df, windows_ms=ROLLING_WINDOWS_MS):
    """
    Adds, for each window size:
      rms_gmag_{w}ms  - rolling RMS of g_mag, the standard HAR magnitude feature
      sma_{w}ms       - signal magnitude area: rolling mean of |ax|+|ay|+|az|

    Both are causal (trailing) windows only, so they're directly portable
    to a live phone app with no future-data problem.
    """
    out = motion_df.copy()
    abs_sum = out["ax"].abs() + out["ay"].abs() + out["az"].abs()

    for w in windows_ms:
        out[f"rms_gmag_{w}ms"] = np.sqrt(_causal_rolling_mean(out, out["g_mag"] ** 2, w))
        out[f"sma_{w}ms"] = _causal_rolling_mean(out, abs_sum, w)

    return out


def add_elevated_duration_feature(motion_df, baseline_window_ms=BASELINE_WINDOW_MS, multiplier=ELEVATED_MULTIPLIER):
    """
    Generalizes "peak width / rise time" into a continuous, causal feature:
    how many seconds the signal has stayed above (multiplier x its own
    recent baseline), as of this exact sample. Resets to 0 the moment the
    signal drops back down. This is computable in real time on a phone
    (only ever needs past samples) and captures the same "how sustained is
    this elevation" idea that a fixed peak-width measurement was getting at,
    without committing to one fixed width up front.
    """
    out = motion_df.copy()
    baseline = _causal_rolling_mean(out, out["g_mag"], baseline_window_ms)
    elevated = out["g_mag"].values > (multiplier * baseline)

    shifted = np.empty_like(elevated)
    shifted[0] = elevated[0]
    shifted[1:] = elevated[:-1]
    run_id = (elevated != shifted).cumsum()
    run_id[0] = 0

    tmp = pd.DataFrame({"ts": out["ts"], "run_id": run_id})
    run_start_ts = tmp.groupby("run_id")["ts"].transform("first")

    duration = (out["ts"] - run_start_ts).dt.total_seconds()
    out["elevated_duration_sec"] = np.where(elevated, duration, 0.0)
    out["is_elevated"] = elevated
    return out


def add_highpass_columns(motion_df, columns=FILTER_COLUMNS, cutoffs=HIGHPASS_CUTOFFS_HZ, order=HIGHPASS_ORDER):
    """
    Adds two versions of each high pass filter, so a real choice can be
    made later instead of it being baked in silently:

      _zerophase  - filtfilt, zero phase distortion, but uses future
                    samples. Fine for offline exploration and comparing
                    cutoffs, NOT reproducible on a live phone stream.
      _causal     - lfilter, forward only, exactly what a phone can
                    compute in real time. Has some phase lag, which is
                    the honest tradeoff for being deployable.

    Whichever cutoff ends up mattering for the model, train on the
    _causal version if the deployed app will filter live, so training
    and inference see the same signal. The _zerophase columns are for
    comparison during feature selection, not necessarily for the final
    model.
    """
    median_dt = motion_df["ts"].diff().dt.total_seconds().median()
    fs = 1.0 / median_dt
    nyquist = fs / 2.0

    out = motion_df.copy()
    for cutoff in cutoffs:
        b, a = butter(order, cutoff / nyquist, btype="highpass")
        for col in columns:
            out[f"{col}_hp{cutoff:g}hz_zerophase"] = filtfilt(b, a, out[col].values)
            out[f"{col}_hp{cutoff:g}hz_causal"] = lfilter(b, a, out[col].values)
    return out


def add_jerk_columns(motion_df, columns=JERK_SOURCE_COLUMNS):
    """
    Jerk = rate of change of acceleration. Uses a backward (causal)
    difference against actual elapsed time, so this is directly usable
    on a live stream, not just offline: jerk[i] = (x[i]-x[i-1])/(t[i]-t[i-1]).
    The first sample in the file has no prior point, so it's set to 0.
    """
    out = motion_df.copy()
    dt = out["ts"].diff().dt.total_seconds().values

    jerk_components = []
    for col in columns:
        dx = out[col].diff().values
        jerk_col = f"jerk_{col}"
        out[jerk_col] = dx / dt
        out.loc[out.index[0], jerk_col] = 0.0
        jerk_components.append(out[jerk_col].values)

    out["jerk_mag"] = np.sqrt(np.sum(np.square(jerk_components), axis=0))
    return out


def add_robust_scaled_columns(df, columns=SCALE_CANDIDATE_COLUMNS):
    """
    Adds a robust z-score (median / IQR, rather than mean / std) for each
    listed column, fit across the FULL combined dataset (all folders),
    not per session. Suffix: _rscaled.

    Worth flagging plainly: a random forest doesn't need scaled features,
    and correlation/VIF based multicollinearity checks aren't affected by
    scaling either, so nothing here is required for the analysis you
    described. These are provided in case a different model (logistic
    regression, SVM, a neural net) enters the picture later, so that step
    doesn't need to be redone from scratch. Median/IQR was used instead of
    mean/std specifically because impact spikes are extreme, legitimate
    outliers, exactly the samples that matter most, and a mean/std scaler
    would have its scale dominated (and therefore distorted for every other
    row) by those same rare events.
    """
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            continue
        median = out[col].median()
        q75, q25 = out[col].quantile(0.75), out[col].quantile(0.25)
        iqr = q75 - q25
        if iqr == 0 or pd.isna(iqr):
            out[f"{col}_rscaled"] = 0.0
        else:
            out[f"{col}_rscaled"] = (out[col] - median) / iqr
    return out


# ---------------------------------------------------------------------------
# Per-folder processing
# ---------------------------------------------------------------------------

def process_folder(folder, tags_df, drop_excluded=True):
    folder = Path(folder)
    metadata = load_metadata(folder)

    # session_id is dropped from the working dataframe (see
    # DROP_FROM_MOTION_COLUMNS), so check it against metadata here first,
    # from a cheap read of just that one column, before it's gone.
    raw_session_id = pd.read_csv(Path(folder) / "motion.csv", usecols=["session_id"])["session_id"].iloc[0]
    if metadata.get("session_id") != raw_session_id:
        print(f"  [warning] metadata session_id does not match motion.csv session_id in {folder}")

    motion = load_motion(folder)

    athlete_name = metadata.get("athlete_name", motion["athlete_name"].iloc[0])
    if metadata.get("athlete_name") != motion["athlete_name"].iloc[0]:
        print(f"  [warning] metadata athlete_name '{metadata.get('athlete_name')}' "
              f"does not match motion.csv athlete_name '{motion['athlete_name'].iloc[0]}' in {folder}")

    folder_tags = select_tags_for_folder(tags_df, athlete_name, motion)
    label_windows = build_label_windows(folder_tags)

    if len(label_windows) == 0:
        print(f"  [skipped] {folder.name}: no tags found for '{athlete_name}' whose time "
              f"falls within this session's span, nothing usable to contribute.")
        return None

    clusters = cluster_labels(label_windows)
    candidates = find_candidate_peaks(motion)
    matches, unmatched = match_labels_to_peaks(label_windows, candidates)

    if len(unmatched) > 0:
        print(f"  [warning] {folder.name}: {len(unmatched)} tag(s) could not be matched "
              f"to any candidate peak within their search window. Check PEAK_MIN_HEIGHT_G, "
              f"or whether the tag's minute genuinely falls outside this session.")

    labeled = label_motion(motion, matches, clusters)

    if drop_excluded:
        labeled = labeled[labeled["Tag"] != "excluded"].reset_index(drop=True)

    if len(labeled) == 0:
        print(f"  [skipped] {folder.name}: tags matched but produced no rows after "
              f"dropping excluded samples (check run clustering settings).")
        return None

    featured = add_rolling_features(labeled)
    featured = add_elevated_duration_feature(featured)
    featured = add_highpass_columns(featured)
    featured = add_jerk_columns(featured)

    featured["source_folder"] = str(folder)
    featured["motion_sampling_hz_metadata"] = metadata.get("motion_sampling_hz")

    print(f"  [ok] {folder.name}: {len(matches)} tag(s) matched, "
          f"{len(unmatched)} unmatched, {len(clusters)} run cluster(s), "
          f"{(featured['Tag']=='event').sum()} event rows, "
          f"{(featured['Tag']=='negative').sum()} negative rows kept.")

    return featured


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def build_dataset(folders=None, tags_path=None, root=None, drop_excluded=True, add_scaled_columns=True, output_path=None):
    """
    Processes a set of movement folders and returns one combined,
    labeled, feature-engineered DataFrame across all of them.

    folders           : explicit list of folder paths. Use this OR root,
                         not both.
    root              : a parent folder containing one subfolder per
                         session (each with motion.csv, events.csv,
                         altitude.csv, metadata.json). Every immediate
                         subfolder that contains a motion.csv is picked
                         up automatically; anything else under root is
                         ignored. Use this OR folders, not both.
    tags_path         : path to the single shared, verified tags CSV
                         (columns include: session_id, athlete, action,
                         iso_time). This is the only source of ground
                         truth labels; each folder's own events.csv is
                         not used.
    drop_excluded     : drop samples outside any attentive run span
                         (default True; see label_motion for what this means)
    add_scaled_columns: add robust-scaled (_rscaled) versions of the main
                         numeric feature columns, fit across the combined
                         dataset, not required for a random forest or for
                         correlation-based multicollinearity checks, but
                         provided for other model types later
    output_path       : if given, also writes the result to this CSV path
    """
    if root is not None and folders is not None:
        raise ValueError("Pass either folders or root, not both.")
    if root is not None:
        folders = discover_session_folders(root)
        print(f"Found {len(folders)} session folder(s) under {root}: "
              f"{[Path(f).name for f in folders]}")
    if not folders:
        raise ValueError("No session folders to process. Pass folders=[...] or root='...'.")

    tags_df = load_tags_file(tags_path)
    print(f"Loaded {len(tags_df)} tags from {tags_path} "
          f"({tags_df['athlete'].nunique()} athlete(s), "
          f"{tags_df['ts'].min()} to {tags_df['ts'].max()}).")

    frames = []
    print(f"\nProcessing {len(folders)} folder(s)...")
    for folder in folders:
        result = process_folder(folder, tags_df, drop_excluded=drop_excluded)
        if result is not None:
            frames.append(result)

    if not frames:
        raise ValueError("No folder produced any usable rows, nothing to return.")

    combined = pd.concat(frames, ignore_index=True)

    if add_scaled_columns:
        combined = add_robust_scaled_columns(combined)

    print(f"\nCombined dataset: {len(combined)} rows from {len(frames)} folder(s).")
    print(combined["Tag"].value_counts())
    print(combined.groupby("athlete_name")["Tag"].value_counts())

    if output_path:
        combined.to_csv(output_path, index=False)
        print(f"\nWritten to {output_path}")

    return combined


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a combined, labeled training dataset from movement folders.")
    parser.add_argument("folders", nargs="*", help="One or more movement folder paths (omit if using --root)")
    parser.add_argument("--root", default=None, help="Parent folder containing one subfolder per session; scanned automatically instead of listing folders")
    parser.add_argument("-t", "--tags", required=True, help="Path to the shared verified tags CSV")
    parser.add_argument("-o", "--output", default="training_dataset.csv", help="Output CSV path")
    parser.add_argument("--keep-excluded", action="store_true", help="Keep excluded (unlabeled-context) rows")
    parser.add_argument("--no-scaling", action="store_true", help="Skip adding robust-scaled columns")
    args = parser.parse_args()

    if args.root and args.folders:
        parser.error("Pass either explicit folders or --root, not both.")

    build_dataset(
        folders=args.folders or None,
        root=args.root,
        tags_path=args.tags,
        drop_excluded=not args.keep_excluded,
        add_scaled_columns=not args.no_scaling,
        output_path=args.output,
    )
