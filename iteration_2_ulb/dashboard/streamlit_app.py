"""
FraudLens dashboard, built with Streamlit (pure Python).

Run from the repository root:
    pip install -r iteration_2_ulb/dashboard/requirements.txt
    streamlit run iteration_2_ulb/dashboard/streamlit_app.py

Why it works this way: the model's inputs are 28 anonymised PCA numbers, so
nobody can type a transaction in by hand. Instead the dashboard runs the
trained models on REAL transactions from the test set (never seen during
training) whose true answer is known, and shows what was caught, what was
missed and what was wrongly flagged. You can also upload your own CSV.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent  # the iteration_2_ulb folder

# Reuse the exact scoring code the FastAPI backend uses (backend/scoring.py)
sys.path.append(str(ROOT / "backend"))
from scoring import INPUT_COLUMNS, MODEL_NAMES, fraud_scores  # noqa: E402

SAMPLE_FILE = ROOT / "data" / "demo" / "test_sample.csv"  # made by notebooks/07_make_demo_sample.py
PRESETS_FILE = ROOT / "backend" / "sample_transactions.json"  # 5 fallback transactions

# Every way the project can decide "fraud": three single models, two ensembles
DETECTORS = {
    **MODEL_NAMES,
    "hard_voting": "Hard voting (2 of 3)",
    "soft_voting": "Soft voting (average)",
}

# Results on the full test set (56,962 transactions, 98 frauds), from NOTES.md
OFFICIAL_RESULTS = pd.DataFrame(
    {
        "Precision": [0.82, 0.11, 0.05, 0.14, 0.44],
        "Recall": [0.82, 0.90, 0.91, 0.90, 0.88],
        "F1": [0.82, 0.19, 0.10, 0.24, 0.59],
        "False alarms": [17, 746, 1598, 545, 110],
        "AUC-ROC": ["0.9688", "0.9807", "0.9747", "-", "-"],
    },
    index=list(DETECTORS.values()),
)


# ------------------------------------------------------------------ helpers

@st.cache_data
def load_bundled():
    """The 500-row test sample if it has been made, otherwise the 5 presets."""
    if SAMPLE_FILE.exists():
        return pd.read_csv(SAMPLE_FILE), True
    presets = json.loads(PRESETS_FILE.read_text())
    table = pd.DataFrame([p["data"] for p in presets])
    table.insert(0, "transaction_id", [p["label"] for p in presets])
    table["Class"] = 1  # all five are real fraud cases from the ULB data
    return table, False


@st.cache_data
def score(transactions):
    """Fraud probability from every model (cached so it only runs once per batch)."""
    return fraud_scores(transactions)


def decide(scores, threshold):
    """True/False "is fraud" for every detector, at the chosen threshold."""
    flags = pd.DataFrame(index=scores.index)
    for name in MODEL_NAMES:
        # "score above the threshold" is exactly what scikit-learn's predict() does at 0.5
        flags[name] = scores[name] > threshold
    flags["hard_voting"] = flags[list(MODEL_NAMES)].sum(axis=1) >= 2
    flags["soft_voting"] = scores["soft_voting"] > threshold
    return flags


def result(flagged, actual):
    """Name the four possible outcomes of one decision."""
    if actual == 1:
        return "Caught" if flagged else "Missed"
    return "False alarm" if flagged else "Cleared"


def summary(flagged, actual):
    """Counts and metrics for one detector on this batch."""
    caught = int((flagged & (actual == 1)).sum())
    missed = int((~flagged & (actual == 1)).sum())
    false_alarms = int((flagged & (actual == 0)).sum())
    cleared = int((~flagged & (actual == 0)).sum())
    precision = caught / (caught + false_alarms) if caught + false_alarms else 0.0
    recall = caught / (caught + missed) if caught + missed else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "Flagged": caught + false_alarms, "Caught": caught, "Missed": missed,
        "False alarms": false_alarms, "Cleared": cleared,
        "Precision": round(precision, 2), "Recall": round(recall, 2), "F1": round(f1, 2),
    }


# ------------------------------------------------------------------ sidebar

st.set_page_config(page_title="FraudLens", page_icon="🔍", layout="wide")

st.sidebar.header("Settings")
source = st.sidebar.radio("Transactions to check", ["Test-set sample", "Upload a CSV"])

if source == "Upload a CSV":
    uploaded = st.sidebar.file_uploader("CSV with V1-V28, Amount, Time (Class optional)", type="csv")
    if uploaded is None:
        st.title("🔍 FraudLens")
        st.info("Upload a CSV in the sidebar. It needs the columns V1-V28, Amount and Time, "
                "like creditcard.csv. If it has a Class column (1 = fraud), results are checked against it.")
        st.stop()
    transactions = pd.read_csv(uploaded)
    missing = [c for c in INPUT_COLUMNS if c not in transactions.columns]
    if missing:
        st.error(f"The CSV is missing these columns: {', '.join(missing)}")
        st.stop()
    if "transaction_id" not in transactions.columns:
        transactions.insert(0, "transaction_id", range(len(transactions)))
    have_sample = True
else:
    transactions, have_sample = load_bundled()

has_labels = "Class" in transactions.columns
actual = transactions["Class"] if has_labels else None

detector = st.sidebar.selectbox(
    "Decide fraud with", list(DETECTORS), format_func=DETECTORS.get,
    help="Random Forest is the final model. The others are shown for comparison.",
)
threshold = st.sidebar.slider(
    "Fraud threshold", 0.05, 0.95, 0.50, 0.05,
    help="A transaction is flagged when its fraud score is above this. "
         "Lower = catch more fraud but more false alarms. The project uses 0.50.",
)

scores = score(transactions)
flags = decide(scores, threshold)
flagged = flags[detector]

# Hard voting has no single probability, so its "score" is the share of models voting fraud
votes = flags[list(MODEL_NAMES)].sum(axis=1)
detector_score = votes / 3 if detector == "hard_voting" else scores[detector]

# ------------------------------------------------------------------ header

st.title("🔍 FraudLens")
st.markdown(
    "Credit-card fraud detection with machine learning. Three models were trained on "
    "**227,845 real transactions** (only 0.17% fraud). This dashboard runs them on "
    "transactions they **have never seen** and checks every answer against the truth."
)
if source == "Test-set sample" and not have_sample:
    st.warning("Showing only the 5 built-in fraud examples. Run `notebooks/07_make_demo_sample.py` "
               "once to load 500 real test-set transactions.")
elif source == "Test-set sample":
    n_fraud = int(actual.sum())
    st.caption(
        f"{len(transactions)} real transactions from the test set, {n_fraud} of them fraud "
        f"({n_fraud / len(transactions):.0%}). Real traffic is only 0.17% fraud, so this sample is "
        "enriched with frauds to give you something to look at. Precision here looks better than in "
        "real life; the full test-set results are in the **Model comparison** tab."
    )

monitor_tab, inspect_tab, compare_tab, about_tab = st.tabs(
    ["📋 Transaction monitor", "🔎 Inspect a transaction", "📊 Model comparison", "ℹ️ How it works"]
)

# ------------------------------------------------------------------ 1. monitor

with monitor_tab:
    cols = st.columns(5)
    cols[0].metric("Transactions checked", len(transactions))
    cols[1].metric("Flagged as fraud", int(flagged.sum()))
    if has_labels:
        s = summary(flagged, actual)
        cols[2].metric("Frauds caught", f"{s['Caught']} of {s['Caught'] + s['Missed']}")
        cols[3].metric("Frauds missed", s["Missed"])
        cols[4].metric("False alarms", s["False alarms"])

    table = pd.DataFrame({
        "Transaction": transactions["transaction_id"],
        "Amount ($)": transactions["Amount"],
        "Hours since start": (transactions["Time"] / 3600).round(1),
        "Fraud score": detector_score,
        "Verdict": flagged.map({True: "🚨 Fraud", False: "✅ Legitimate"}),
    })
    if has_labels:
        table["Actually"] = actual.map({1: "Fraud", 0: "Legitimate"})
        table["Result"] = [result(f, a) for f, a in zip(flagged, actual)]

    left, right = st.columns([3, 1])
    with left:
        options = ["All", "Flagged as fraud"] + (["Caught", "Missed", "False alarm"] if has_labels else [])
        show = st.radio("Show", options, horizontal=True)
    if show == "Flagged as fraud":
        table = table[flagged]
    elif show != "All":
        table = table[table["Result"] == show]

    st.dataframe(
        table.sort_values("Fraud score", ascending=False),
        hide_index=True,
        width="stretch",
        column_config={
            "Fraud score": st.column_config.ProgressColumn(
                "Fraud score", min_value=0.0, max_value=1.0, format="percent"),
            "Amount ($)": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    with right:
        st.download_button("Download results (CSV)", table.to_csv(index=False),
                           "fraudlens_results.csv", "text/csv")

    if has_labels:
        st.subheader("Confusion matrix")
        st.dataframe(pd.DataFrame(
            [[s["Caught"], s["Missed"]], [s["False alarms"], s["Cleared"]]],
            index=["Actually fraud", "Actually legitimate"],
            columns=["Flagged as fraud", "Not flagged"],
        ))

# ------------------------------------------------------------------ 2. inspect

with inspect_tab:
    # Most suspicious first
    order = detector_score.sort_values(ascending=False).index
    choice = st.selectbox(
        "Pick a transaction",
        order,
        format_func=lambda i: (f"#{transactions.at[i, 'transaction_id']} · "
                               f"${transactions.at[i, 'Amount']:.2f} · score {detector_score[i]:.0%}"),
    )

    is_fraud = bool(flagged[choice])
    truth = None if not has_labels else ("fraud" if actual[choice] == 1 else "legitimate")

    cols = st.columns(3)
    cols[0].metric("Amount", f"${transactions.at[choice, 'Amount']:.2f}")
    cols[1].metric("Hours since first transaction", f"{transactions.at[choice, 'Time'] / 3600:.1f}")
    cols[2].metric("Actually", truth.capitalize() if truth else "Unknown")

    verdict = f"{DETECTORS[detector]} says **{'FRAUD' if is_fraud else 'LEGITIMATE'}** (score {detector_score[choice]:.0%}, threshold {threshold:.0%})."
    if truth:
        correct = is_fraud == (truth == "fraud")
        verdict += " ✅ Correct." if correct else f" ❌ Wrong: it was really {truth}."
    (st.error if is_fraud else st.success)(verdict)

    st.markdown("**What each model thinks** (fraud probability)")
    st.bar_chart(pd.DataFrame(
        {"Fraud score": [scores.at[choice, m] for m in [*MODEL_NAMES, "soft_voting"]]},
        index=[*MODEL_NAMES.values(), "Soft voting (average)"],
    ), horizontal=True, sort=False)
    st.caption(f"Hard voting: {int(votes[choice])} of 3 models flag this transaction "
               f"→ {'fraud' if flags.at[choice, 'hard_voting'] else 'legitimate'}.")

    with st.expander("The 30 numbers the model actually sees"):
        st.write("V1-V28 are anonymised PCA features (the bank hid the real columns for privacy). "
                 "Amount and Time are scaled before they reach the model.")
        st.dataframe(transactions.loc[[choice], INPUT_COLUMNS], hide_index=True)

# ------------------------------------------------------------------ 3. compare

with compare_tab:
    if has_labels:
        st.subheader(f"On these {len(transactions)} transactions (threshold {threshold:.0%})")
        batch = pd.DataFrame({DETECTORS[d]: summary(flags[d], actual) for d in DETECTORS}).T
        st.dataframe(batch[["Flagged", "Caught", "Missed", "False alarms", "Precision", "Recall", "F1"]])
        st.bar_chart(batch[["Precision", "Recall"]].astype(float), stack=False, horizontal=True, sort=False)
        st.caption("Move the threshold in the sidebar and watch the trade-off: a lower threshold "
                   "catches more fraud but raises more false alarms.")
    st.subheader("Official results on the full test set (56,962 transactions, 98 frauds)")
    st.dataframe(OFFICIAL_RESULTS)
    st.caption("Random Forest was chosen as the final model because it has the best balance "
               "(F1 0.82). Gradient Boosting and AdaBoost catch slightly more fraud but raise "
               "hundreds of false alarms. Combining the models did not beat Random Forest alone.")

# ------------------------------------------------------------------ 4. about

with about_tab:
    st.markdown("""
**The data.** 284,807 European credit-card transactions from 2013, of which only 492 (0.17%)
are fraud. For privacy the bank turned the original columns into 28 anonymous PCA features
(V1-V28); only Amount and Time are left as they were.

**Training, step by step**
1. Scale Amount and Time so they are on the same scale as the PCA features.
2. Split 80% for training and 20% for testing, keeping 0.17% fraud in both.
3. Balance the training set with **SMOTE**, which creates synthetic fraud examples
   (the test set is left untouched so it reflects reality).
4. Train Random Forest, Gradient Boosting and AdaBoost, and try two ensembles.
5. Evaluate on the untouched test set with precision, recall and F1 (accuracy is
   useless here: always saying "legitimate" is 99.83% accurate).

**Why there is no "type in a transaction" form.** Nobody can make up 28 PCA values, so
the dashboard uses real test-set transactions the models never saw, where the right
answer is known.

**What the fraud score means.** It is the model's probability of fraud. Because the model
was trained on SMOTE-balanced data, the scores run higher than the real fraud rate, so
treat them as a ranking, not an exact probability.

**Limitations.** One train/test split, a fixed 0.5 threshold, data from 2013, and PCA
features that cannot be explained to a customer. See `VIVA_GUIDE.md` in the repository.
""")
