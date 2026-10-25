# 🔍 FraudLens

**A machine-learning system that spots fraudulent credit-card transactions.**

Out of every 1,000 card payments, fewer than 2 are fraud. FraudLens learns what those
few look like from 284,807 real transactions, and flags new ones that look suspicious.

> **Result:** the final model catches **82%** of fraud, and **82%** of the transactions it
> flags really are fraud. It was tested on 56,962 transactions it had never seen.

---

## Why this is hard

Fraud is extremely rare: only **0.17%** of transactions in the data are fraud.

That breaks the obvious way of judging a model. A "model" that simply says
*"everything is legitimate"* is **99.83% accurate**, and catches zero fraud. So instead of
accuracy, FraudLens is judged on two questions:

| Question | Name | FraudLens |
|---|---|---|
| Of all the real fraud, how much did we catch? | **Recall** | 0.82 (80 of 98) |
| Of everything we flagged, how much was really fraud? | **Precision** | 0.82 (80 of 97) |

---

## How it works

```
 284,807 real transactions
          │
          ▼
 1. Clean up      Scale Amount and Time so all inputs are on a similar scale
          │
          ▼
 2. Split         80% to learn from, 20% kept aside as an unseen test
          │
          ▼
 3. Balance       SMOTE creates synthetic fraud examples, so the model
                  sees enough fraud to learn from (training data only)
          │
          ▼
 4. Train         Random Forest, Gradient Boosting and AdaBoost
          │
          ▼
 5. Test          Check every model on the unseen 20%
          │
          ▼
 6. Use           Dashboard and API run the best model (Random Forest)
```

**About the data.** The [ULB credit-card dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
holds two days of European card payments from 2013. For privacy, the bank turned the original
details into 28 anonymous numbers (`V1`–`V28`, made with PCA). Only `Amount` and `Time` are left
as they were.

---

## Results

Tested on 56,962 transactions the models never saw (98 of them fraud):

| Model | Catches (recall) | Correct when it flags (precision) | F1 | False alarms |
|---|---|---|---|---|
| **Random Forest** ✅ | 0.82 | 0.82 | **0.82** | 17 |
| Gradient Boosting | 0.90 | 0.11 | 0.19 | 746 |
| AdaBoost | 0.91 | 0.05 | 0.10 | 1,598 |
| Hard voting (2 of 3 agree) | 0.90 | 0.14 | 0.24 | 545 |
| Soft voting (average score) | 0.88 | 0.44 | 0.59 | 110 |

**Random Forest wins** because it balances both questions. The other two catch slightly more
fraud but raise hundreds of false alarms, which would mean blocking many honest customers.
Combining the models didn't beat Random Forest alone, because the other two make the same kind
of mistake.

---

## The dashboard

The model's inputs are 28 anonymous numbers, so nobody can type a transaction in by hand.
Instead the dashboard runs the models on **real transactions from the test set**, where the
right answer is known, and shows exactly what happened:

- **Transaction monitor:** every transaction with its fraud score, the verdict, and whether
  it was really fraud: ✅ caught, ❌ missed, ⚠️ false alarm.
- **Inspect a transaction:** what each model thinks about one transaction.
- **Model comparison:** all models side by side, with a threshold slider to see the trade-off
  between catching fraud and raising false alarms.
- **Upload a CSV:** check your own file in the same format as `creditcard.csv`.

It is built with [Streamlit](https://streamlit.io), so the whole project, model and website,
is plain Python.

---

## Run it yourself

You need Python 3.11 or newer.

```bash
# Dashboard (opens in your browser)
pip install -r iteration_2_ulb/dashboard/requirements.txt
streamlit run iteration_2_ulb/dashboard/streamlit_app.py
```

```bash
# API (optional): send a transaction as JSON, get a verdict back
cd iteration_2_ulb/backend
pip install -r requirements.txt
uvicorn app:app --port 5000        # interactive docs at http://127.0.0.1:5000/docs
```

To retrain the models or rebuild the demo sample, download `creditcard.csv` from
[Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) into
`iteration_2_ulb/data/raw/` and run the scripts in `iteration_2_ulb/notebooks/` in order.

---

## What's in this repository

```
FraudLens/
├── iteration_1_fraud_data/     First attempt (a different dataset). A documented dead end:
│                               recall got stuck at 53%, so I switched datasets.
└── iteration_2_ulb/            The final system
    ├── notebooks/              01–05 explore, prepare, train, combine, evaluate
    │                           06 stricter re-evaluation, 07 makes the dashboard sample
    ├── models/                 The trained models and scaler (.pkl files)
    ├── backend/                FastAPI service + shared scoring code + tests
    ├── dashboard/              Streamlit website + tests
    └── data/                   Data goes here (creditcard.csv is too big for GitHub)
```

**Further reading:**
- [`PROJECT_GUIDE.md`](PROJECT_GUIDE.md): how every part works, and a log of all changes.
- [`VIVA_GUIDE.md`](VIVA_GUIDE.md): how to present and defend the project.
- [`iteration_2_ulb/NOTES.md`](iteration_2_ulb/NOTES.md): the reasoning behind each step.

---

## Limitations

- Scores are a ranking, not exact probabilities (SMOTE inflates them).
- The decision threshold is the default 0.5, not tuned to the cost of fraud.
- Evaluated on one train/test split. `notebooks/06_improved_evaluation.py` does a stricter check.
- The data is from 2013, and fraud patterns change over time.
- The anonymous PCA inputs mean the model can't explain *why* a payment looks suspicious.

## Built with

Python · pandas · NumPy · scikit-learn · imbalanced-learn · Streamlit · FastAPI · pytest · GitHub Actions
