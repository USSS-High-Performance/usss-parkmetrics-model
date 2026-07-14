"""
Post-trigger classifier training and evaluation pipeline.

For each feature set (minimal_5, balanced_8, full_10):
  1. Cross-validates three model architectures on the training athletes
     using GroupKFold, so every reported number reflects performance on
     an athlete the model didn't see during training.
  2. Fits each model on all training athletes and evaluates once on the
     held-out test athlete.

Model architectures:
  - logreg: L2 logistic regression (baseline, essentially free on device)
  - gbm:    LightGBM, 50 shallow trees (the recommended production model)
  - mlp:    Small MLP with one hidden layer, 12 units

Metrics reported (per fold and on held-out):
  - AP (average precision, the right metric for imbalanced binary)
  - ROC AUC (secondary)
  - Precision/recall/F1 at the default 0.5 threshold, plus at whatever
    threshold hits recall >= 0.90 on the fold, since keeping false
    negatives low matters more than false positives for this app.

Usage:
    python train_and_evaluate.py --datasets-dir windowed_datasets/
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TARGET_COL = "is_event"
GROUP_COL = "athlete_name"
METADATA_COLS = {
    TARGET_COL,
    GROUP_COL,
    "impact_timestamp_iso8601",
    "impact_elapsed_sec",
    "impact_source_sample_idx",
    "source_folder",
}
CV_FOLDS = 3          # 3-way GroupKFold over the training athletes (there are 3 of them)
RECALL_TARGET = 0.90  # find a threshold that hits at least this recall


def load_split(dataset_dir):
    train = pd.read_csv(dataset_dir / "train.csv")
    test = pd.read_csv(dataset_dir / "test.csv")

    feature_cols = [c for c in train.columns if c not in METADATA_COLS]
    X_train = train[feature_cols].astype(float).values
    y_train = train[TARGET_COL].astype(int).values
    groups_train = train[GROUP_COL].values

    X_test = test[feature_cols].astype(float).values
    y_test = test[TARGET_COL].astype(int).values

    return {
        "feature_cols": feature_cols,
        "X_train": X_train, "y_train": y_train, "groups_train": groups_train,
        "X_test": X_test, "y_test": y_test,
        "train_df": train, "test_df": test,
    }


def build_model(kind):
    """
    All three models include a StandardScaler upstream. That matters
    for logistic regression and MLP; for GBM it's harmless (trees are
    scale-invariant) and keeps the pipeline shape uniform so downstream
    code can treat them the same.
    """
    if kind == "logreg":
        clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    elif kind == "gbm":
        clf = GradientBoostingClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
    elif kind == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=(12,), max_iter=1000, random_state=42,
            early_stopping=True, validation_fraction=0.15,
        )
    else:
        raise ValueError(f"unknown model kind: {kind}")
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def threshold_for_recall(y_true, y_score, target_recall):
    """
    Find the highest probability threshold that still hits target_recall.
    Higher threshold = fewer false positives, so this returns the
    strictest possible cutoff that still catches (target_recall x 100)% of hits.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    # precision_recall_curve returns thresholds of length len(precisions)-1;
    # align by walking from the highest threshold downward
    for p, r, t in zip(precisions[:-1][::-1], recalls[:-1][::-1], thresholds[::-1]):
        if r >= target_recall:
            return float(t), float(p), float(r)
    return 0.0, float(precisions[-1]), float(recalls[-1])


def score_predictions(y_true, y_score, threshold=0.5):
    y_pred = (y_score >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    return {
        "threshold": float(threshold),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0,
    }


def cross_validate(X, y, groups, kind, n_splits):
    """
    GroupKFold on the training athletes. Each fold trains on all but one
    training athlete and scores on the held-out one.
    """
    gkf = GroupKFold(n_splits=n_splits)
    fold_results = []
    for fold, (tr, va) in enumerate(gkf.split(X, y, groups)):
        held = np.unique(groups[va])[0]
        model = build_model(kind)
        model.fit(X[tr], y[tr])
        y_score = model.predict_proba(X[va])[:, 1]

        ap = float(average_precision_score(y[va], y_score)) if y[va].sum() > 0 else float("nan")
        try:
            auc = float(roc_auc_score(y[va], y_score))
        except ValueError:
            auc = float("nan")

        default = score_predictions(y[va], y_score, 0.5)
        rec_t, rec_prec, rec_rec = threshold_for_recall(y[va], y_score, RECALL_TARGET)
        recall_tuned = score_predictions(y[va], y_score, rec_t)

        fold_results.append({
            "fold": fold, "held_out_athlete": held,
            "n_train": int(len(tr)), "n_val": int(len(va)),
            "n_val_positive": int(y[va].sum()),
            "ap": ap, "roc_auc": auc,
            "default_threshold": default,
            "recall_tuned_threshold": recall_tuned,
        })
    return fold_results


def evaluate_on_test(X_train, y_train, X_test, y_test, kind):
    """Final fit on all training data + score once on the held-out test athlete."""
    model = build_model(kind)
    model.fit(X_train, y_train)
    y_score = model.predict_proba(X_test)[:, 1]

    ap = float(average_precision_score(y_test, y_score)) if y_test.sum() > 0 else float("nan")
    try:
        auc = float(roc_auc_score(y_test, y_score))
    except ValueError:
        auc = float("nan")
    default = score_predictions(y_test, y_score, 0.5)
    rec_t, _, _ = threshold_for_recall(y_test, y_score, RECALL_TARGET)
    recall_tuned = score_predictions(y_test, y_score, rec_t)

    return {
        "ap": ap, "roc_auc": auc,
        "default_threshold": default,
        "recall_tuned_threshold": recall_tuned,
        "n_test": int(len(y_test)), "n_test_positive": int(y_test.sum()),
    }


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate post-trigger classifiers.")
    parser.add_argument("--datasets-dir", default="windowed_datasets", help="Directory produced by build_windowed_features.py")
    parser.add_argument("--output-json", default="results.json", help="Where to write the full results report")
    parser.add_argument("--models", nargs="+", default=["logreg", "gbm", "mlp"], help="Which model kinds to train")
    args = parser.parse_args()

    root = Path(args.datasets_dir)
    feature_set_dirs = sorted([p for p in root.iterdir() if p.is_dir()])

    all_results = {}
    for fs_dir in feature_set_dirs:
        fs_name = fs_dir.name
        print(f"\n=== feature set: {fs_name} ===")
        data = load_split(fs_dir)
        print(f"  features ({len(data['feature_cols'])}): {data['feature_cols']}")
        print(f"  train: {len(data['y_train'])} rows, {data['y_train'].sum()} positive")
        print(f"  test:  {len(data['y_test'])} rows, {data['y_test'].sum()} positive")

        fs_results = {"feature_cols": data["feature_cols"], "models": {}}
        for kind in args.models:
            print(f"\n  --- model: {kind} ---")
            cv = cross_validate(
                data["X_train"], data["y_train"], data["groups_train"], kind, CV_FOLDS
            )
            for r in cv:
                print(
                    f"    CV fold {r['fold']} (held out {r['held_out_athlete']}): "
                    f"AP={r['ap']:.3f}, ROC-AUC={r['roc_auc']:.3f}, "
                    f"@0.5: P={r['default_threshold']['precision']:.2f}/R={r['default_threshold']['recall']:.2f}, "
                    f"@recall>={RECALL_TARGET}: P={r['recall_tuned_threshold']['precision']:.2f}"
                )
            test = evaluate_on_test(
                data["X_train"], data["y_train"], data["X_test"], data["y_test"], kind
            )
            print(
                f"    HELD-OUT TEST: AP={test['ap']:.3f}, ROC-AUC={test['roc_auc']:.3f}, "
                f"@0.5: P={test['default_threshold']['precision']:.2f}/R={test['default_threshold']['recall']:.2f}, "
                f"@recall>={RECALL_TARGET}: P={test['recall_tuned_threshold']['precision']:.2f}"
            )
            fs_results["models"][kind] = {"cv": cv, "held_out_test": test}
        all_results[fs_name] = fs_results

    with open(args.output_json, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nFull results written to {args.output_json}")


if __name__ == "__main__":
    main()
