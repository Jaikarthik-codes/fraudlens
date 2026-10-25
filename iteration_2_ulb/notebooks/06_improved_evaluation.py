"""
06 - Leakage-free re-evaluation of the Random Forest.

Run from the notebooks/ folder (needs ../data/raw/creditcard.csv):
    python 06_improved_evaluation.py              # full run
    python 06_improved_evaluation.py --skip-cv    # skip the slow cross-validation

This script is analysis only. It does NOT overwrite anything in ../models, so
the deployed model and website are unaffected. It repeats the Random Forest
experiment with four fixes to the original pipeline:

  1. The scaler is fitted on the training data only (02_preprocess.py fits it
     on the whole dataset, so test-set statistics leak into the scaling).
  2. PR-AUC (average precision) is reported next to ROC-AUC. With 0.17% fraud,
     ROC-AUC looks excellent for almost any model.
  3. The decision threshold is chosen on a validation split taken from the
     training data, then reported once on the test set (never tuned on it).
  4. Probabilities are calibrated (isotonic) because SMOTE inflates them, and
     5-fold cross-validation gives a mean +/- std instead of a single split.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--data", default=str(HERE.parent / "data" / "raw" / "creditcard.csv"))
parser.add_argument("--out", default=str(HERE.parent / "reports" / "06_improved_evaluation.json"))
parser.add_argument("--trees", type=int, default=100, help="Random Forest size (100 matches the deployed model)")
parser.add_argument("--skip-cv", action="store_true", help="skip 5-fold cross-validation")
args = parser.parse_args()

SEED = 42


def new_forest():
    return RandomForestClassifier(n_estimators=args.trees, random_state=SEED, n_jobs=-1)


def scale(frame, scaler):
    """Replace raw Amount/Time with scaled versions, in the deployed column order."""
    out = frame.drop(columns=["Amount", "Time"]).copy()
    out[["scaled_Amount", "scaled_Time"]] = scaler.transform(frame[["Amount", "Time"]])
    return out


def at_threshold(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    return {
        "threshold": round(float(threshold), 4),
        "precision": round(precision_score(y_true, pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, pred, zero_division=0), 4),
        "false_positives": int(((pred == 1) & (y_true == 0)).sum()),
        "false_negatives": int(((pred == 0) & (y_true == 1)).sum()),
    }


def ranking(y_true, proba):
    return {
        "roc_auc": round(roc_auc_score(y_true, proba), 4),
        "pr_auc": round(average_precision_score(y_true, proba), 4),
    }


def show(title, metrics):
    print(f"  {title:<34}" + "  ".join(f"{k}={v}" for k, v in metrics.items()))


# ---------------------------------------------------------------- data + split
df = pd.read_csv(args.data)
X = df.drop(columns=["Class"])
y = df["Class"]
print(f"{len(df):,} rows, {int(y.sum())} frauds ({y.mean():.2%})")

# Same split as the original notebooks (random_state=42, stratified, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
y_test_arr = y_test.to_numpy()

# Fix 1: scaler fitted on training data only
scaler = StandardScaler().fit(X_train[["Amount", "Time"]])
X_train_s, X_test_s = scale(X_train, scaler), scale(X_test, scaler)

report = {"rows": int(len(df)), "frauds": int(y.sum()), "trees": args.trees}

# ----------------------------------------------- 1. baseline (same as deployed)
print("\n1. Baseline: scaler on train only, SMOTE, Random Forest, threshold 0.5")
X_sm, y_sm = SMOTE(random_state=SEED).fit_resample(X_train_s, y_train)
baseline = new_forest().fit(X_sm, y_sm)
p_test = baseline.predict_proba(X_test_s)[:, 1]
report["baseline"] = {**at_threshold(y_test_arr, p_test, 0.5), **ranking(y_test_arr, p_test)}
show("test set", report["baseline"])
print("  (NOTES.md reports precision 0.82, recall 0.82, ROC-AUC 0.9688 with the leaky scaler)")

# --------------------------------------- 2. threshold chosen on a validation set
print("\n2. Threshold chosen on a validation split (20% of the training data)")
X_core, X_val, y_core, y_val = train_test_split(
    X_train_s, y_train, test_size=0.2, random_state=SEED, stratify=y_train
)
X_core_sm, y_core_sm = SMOTE(random_state=SEED).fit_resample(X_core, y_core)
forest = new_forest().fit(X_core_sm, y_core_sm)

p_val = forest.predict_proba(X_val)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_val, p_val)
f1_curve = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
best_threshold = float(thresholds[np.argmax(f1_curve)])

p_test_core = forest.predict_proba(X_test_s)[:, 1]
report["threshold_tuned"] = {
    **at_threshold(y_test_arr, p_test_core, best_threshold),
    **ranking(y_test_arr, p_test_core),
}
show("test, threshold from validation", report["threshold_tuned"])
print("  threshold sweep on the test set (for understanding only, not used to choose):")
report["threshold_sweep_test"] = [at_threshold(y_test_arr, p_test_core, t) for t in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)]
for row in report["threshold_sweep_test"]:
    show(f"threshold {row['threshold']}", {k: v for k, v in row.items() if k != "threshold"})

# ------------------------------------------------------------- 3. calibration
print("\n3. Calibration (isotonic, fitted on the validation split)")
calibrated = CalibratedClassifierCV(FrozenEstimator(forest), method="isotonic").fit(X_val, y_val)
p_cal = calibrated.predict_proba(X_test_s)[:, 1]
actual_rate = float(y_test_arr.mean())
report["calibration"] = {
    "actual_fraud_rate": round(actual_rate, 5),
    "mean_score_before": round(float(p_test_core.mean()), 5),
    "mean_score_after": round(float(p_cal.mean()), 5),
    "brier_before": round(brier_score_loss(y_test_arr, p_test_core), 6),
    "brier_after": round(brier_score_loss(y_test_arr, p_cal), 6),
}
show("calibration", report["calibration"])
print("  A well-calibrated model's mean score is close to the actual fraud rate.")

# ----------------------------------------------------------- 4. cross-validation
if args.skip_cv:
    print("\n4. Cross-validation skipped (--skip-cv)")
else:
    print("\n4. 5-fold stratified cross-validation on the training data")
    # Scaling and SMOTE live inside the pipeline, so each fold fits them on its own training part only
    pipeline = Pipeline([
        ("scale", ColumnTransformer([("amount_time", StandardScaler(), ["Amount", "Time"])], remainder="passthrough")),
        ("smote", SMOTE(random_state=SEED)),
        ("forest", new_forest()),
    ])
    scores = cross_validate(
        pipeline, X_train, y_train,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
        scoring={"precision": "precision", "recall": "recall", "f1": "f1",
                 "roc_auc": "roc_auc", "pr_auc": "average_precision"},
    )
    report["cross_validation"] = {
        name: {"mean": round(float(scores[f"test_{name}"].mean()), 4), "std": round(float(scores[f"test_{name}"].std()), 4)}
        for name in ("precision", "recall", "f1", "roc_auc", "pr_auc")
    }
    for name, value in report["cross_validation"].items():
        print(f"  {name:<10} {value['mean']:.4f} +/- {value['std']:.4f}")

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2))
print(f"\nSaved {out}")
