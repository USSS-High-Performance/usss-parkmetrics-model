"""
Windowed feature extractor.

Turns the labeled per-sample training dataset (produced by
hit_detection_pipeline.py) into one row per distinct trigger candidate,
which is what the on-device classifier will actually see: a fixed-size
feature vector aggregated over the window around each trigger firing.

Anchor point for each candidate: the peak-g_mag sample within the group
of consecutive samples that stayed above the trigger threshold. That's
the "moment of impact" the classifier should report if it confirms the
hit, both a timestamp (impact_timestamp_iso8601) and a per-session
elapsed-time offset (impact_elapsed_sec) are attached to every row.

Also stored: the raw file index of the peak sample in the source
motion.csv, so a downstream tool can reach back into the raw data if
it needs to (for example, to grab the full raw window for a CNN model
later).

Usage:
    python build_windowed_features.py training_dataset.csv \\
        --output-dir windowed_datasets/ \\
        --trigger-threshold 4.0 \\
        --window-ms 500 \\
        --hysteresis-gap-samples 30
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


TRIGGER_THRESHOLD_DEFAULT = 4.0        # g_mag >= this fires the trigger
WINDOW_MS_DEFAULT = 500                 # +/- ms around the peak sample to aggregate over
HYSTERESIS_GAP_SAMPLES_DEFAULT = 30     # trigger firings closer than this collapse into one candidate

# Feature sets, ordered by importance from the per-sample RF (dropping
# collinear/redundant columns as we go up in count). Nested to keep the
# selection choices explicit rather than magic-numbered elsewhere.
FEATURE_SETS = {
    "minimal_5": [
        "peak_g_mag",
        "std_g_mag",
        "peak_hp8_vertical",
        "peak_jerk_mag",
        "max_elevated_dur",
    ],
    "balanced_8": [
        "peak_g_mag",
        "std_g_mag",
        "peak_sma150",
        "peak_hp8_vertical",
        "peak_rms150",
        "peak_jerk_mag",
        "max_elevated_dur",
        "peak_hp8_planar",
    ],
    "full_10": [
        "peak_g_mag",
        "std_g_mag",
        "peak_sma150",
        "peak_hp8_vertical",
        "peak_rms150",
        "peak_jerk_mag",
        "max_elevated_dur",
        "peak_hp8_planar",
        "pitch_range",
        "rot_x_range",
    ],
}


def load_and_prepare(path):
    df = pd.read_csv(path)
    df["ts"] = pd.to_datetime(df["timestamp_iso8601"])
    df = df.sort_values(["source_folder", "ts"]).reset_index(drop=True)
    return df


def find_trigger_groups(session_df, threshold, gap_samples):
    """
    Within one session, find every stretch of consecutive samples where
    g_mag >= threshold and treat each stretch as a single trigger
    candidate. A gap of more than gap_samples raw samples starts a new
    candidate (the hysteresis: one physical hit + its rebound don't
    count as two distinct triggers).
    """
    fires = session_df.index[session_df["g_mag"] >= threshold].tolist()
    if not fires:
        return []

    groups, prev = [], -10 ** 9
    for i in fires:
        if i - prev > gap_samples:
            groups.append([i])
        else:
            groups[-1].append(i)
        prev = i
    return groups


def extract_window_features(session_df, group_indices, window_ms):
    """
    For one trigger candidate (a group of consecutive-above-threshold
    sample indices in the session), find the peak-g_mag sample as the
    anchor, then aggregate a fixed set of features from the +/- window
    of raw samples around it.
    """
    # peak sample within the trigger group
    peak_idx = max(group_indices, key=lambda k: session_df["g_mag"].iloc[k])
    peak_row = session_df.iloc[peak_idx]
    peak_ts = peak_row["ts"]

    # +/- window of raw samples around the peak, on real time
    lo = peak_ts - pd.Timedelta(milliseconds=window_ms)
    hi = peak_ts + pd.Timedelta(milliseconds=window_ms)
    w = session_df[(session_df["ts"] >= lo) & (session_df["ts"] <= hi)]

    # positive if ANY row in the window is a labeled event
    is_event = bool((w["Tag"] == "event").any())

    features = {
        # aggregated features (the ones the classifier will actually see)
        "peak_g_mag": float(w["g_mag"].max()),
        "mean_g_mag": float(w["g_mag"].mean()),
        "std_g_mag": float(w["g_mag"].std()),
        "peak_planar": float(w["planar_accel_g"].max()),
        "peak_vertical_abs": float(w["vertical_accel_g"].abs().max()),
        "peak_jerk_mag": float(w["jerk_mag"].max()),
        "peak_rms150": float(w["rms_gmag_150ms"].max()),
        "peak_sma150": float(w["sma_150ms"].max()),
        "max_elevated_dur": float(w["elevated_duration_sec"].max()),
        "n_samples_above_2g": int((w["g_mag"] > 2).sum()),
        "n_samples_above_4g": int((w["g_mag"] > 4).sum()),
        "rot_x_range": float(w["rot_x"].max() - w["rot_x"].min()),
        "rot_y_range": float(w["rot_y"].max() - w["rot_y"].min()),
        "rot_z_range": float(w["rot_z"].max() - w["rot_z"].min()),
        "pitch_range": float(w["pitch_rad"].max() - w["pitch_rad"].min()),
        "roll_range": float(w["roll_rad"].max() - w["roll_rad"].min()),
        "peak_hp8_vertical": float(w["vertical_accel_g_hp8hz_causal"].abs().max()),
        "peak_hp8_planar": float(w["planar_accel_g_hp8hz_causal"].abs().max()),
        # label
        "is_event": is_event,
        # metadata for locating the impact when the classifier confirms a hit
        "impact_timestamp_iso8601": peak_row["timestamp_iso8601"],
        "impact_elapsed_sec": float(peak_row["elapsed_sec"]),
        "impact_source_sample_idx": int(peak_idx),
        "source_folder": peak_row["source_folder"],
        "athlete_name": peak_row["athlete_name"],
    }
    return features


def build_windowed(df, threshold, window_ms, gap_samples):
    records = []
    for folder, session in df.groupby("source_folder"):
        session = session.sort_values("ts").reset_index(drop=True)
        for group in find_trigger_groups(session, threshold, gap_samples):
            records.append(extract_window_features(session, group, window_ms))
    return pd.DataFrame(records)


def split_grouped(df, test_athletes):
    """
    Athlete-based train/test split, since the model must generalize
    across athletes. Test set is any candidate whose source athlete is
    in test_athletes. Everything else is training.
    """
    test = df[df["athlete_name"].isin(test_athletes)].copy()
    train = df[~df["athlete_name"].isin(test_athletes)].copy()
    return train, test


def write_feature_set(candidates_df, feature_cols, output_dir, name):
    """
    For a given feature set, write train.csv and test.csv containing
    only those features plus the label and locating metadata.
    """
    keep_cols = feature_cols + [
        "is_event",
        "impact_timestamp_iso8601",
        "impact_elapsed_sec",
        "impact_source_sample_idx",
        "source_folder",
        "athlete_name",
    ]
    subset = candidates_df[keep_cols].copy()

    # Deterministic, reproducible held-out choice: use the athlete with
    # the smallest positive count in this dataset as the held-out test
    # athlete. That way the test set is guaranteed to contain at least
    # some positives from an unseen person, which is what we care about.
    per_athlete_pos = (
        subset[subset["is_event"]].groupby("athlete_name").size().sort_values()
    )
    if len(per_athlete_pos) < 2:
        raise ValueError(
            "Need at least 2 athletes with positive events for a train/test split."
        )
    held_out = [per_athlete_pos.index[0]]

    train, test = split_grouped(subset, held_out)

    out_dir = Path(output_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(out_dir / "train.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)

    print(
        f"  [{name}]  features={len(feature_cols)}  "
        f"train={len(train)} (pos={train['is_event'].sum()})  "
        f"test={len(test)} (pos={test['is_event'].sum()})  "
        f"held_out_athlete={held_out[0]}"
    )
    return {"name": name, "train": len(train), "test": len(test), "held_out": held_out[0]}


def main():
    parser = argparse.ArgumentParser(description="Build windowed-feature training datasets.")
    parser.add_argument("input_csv", help="Path to the per-sample training_dataset.csv")
    parser.add_argument("--output-dir", default="windowed_datasets", help="Where to write the feature-set folders")
    parser.add_argument("--trigger-threshold", type=float, default=TRIGGER_THRESHOLD_DEFAULT)
    parser.add_argument("--window-ms", type=float, default=WINDOW_MS_DEFAULT)
    parser.add_argument("--hysteresis-gap-samples", type=int, default=HYSTERESIS_GAP_SAMPLES_DEFAULT)
    args = parser.parse_args()

    print(f"Loading {args.input_csv}...")
    df = load_and_prepare(args.input_csv)
    print(f"  {len(df)} samples across {df['source_folder'].nunique()} session(s)")

    print(
        f"\nBuilding trigger candidates: g_mag >= {args.trigger_threshold}, "
        f"+/- {args.window_ms}ms window, hysteresis gap = {args.hysteresis_gap_samples} samples"
    )
    candidates = build_windowed(df, args.trigger_threshold, args.window_ms, args.hysteresis_gap_samples)
    print(
        f"  {len(candidates)} trigger candidates  "
        f"(pos={candidates['is_event'].sum()}, neg={(~candidates['is_event']).sum()})"
    )
    print(f"  per athlete:\n{candidates.groupby(['athlete_name', 'is_event']).size()}")

    print("\nWriting feature-set folders:")
    for name, feats in FEATURE_SETS.items():
        write_feature_set(candidates, feats, args.output_dir, name)

    # Also write the full unfiltered candidate table for reference / future models
    (Path(args.output_dir) / "all_candidates.csv").parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(Path(args.output_dir) / "all_candidates.csv", index=False)
    print(f"\nAlso wrote {args.output_dir}/all_candidates.csv (all features, all candidates).")


if __name__ == "__main__":
    main()
