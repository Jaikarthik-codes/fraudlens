# FraudLens — Viva Guide

How to explain this project from the first idea to the last prediction, and how to answer the questions an examiner is likely to ask. Read it with `PROJECT_GUIDE.md` (what the code does and what changed) and the two `NOTES.md` files (your own reasoning).

**Golden rule: only claim what you can explain.** If you don't know, say "I didn't test that, but I would check it by…". That scores better than a confident wrong answer. Section 9 lists the weak points so you can raise them yourself before an examiner does.

---

## 1. The three versions of your pitch

### 30 seconds
"FraudLens detects fraudulent credit-card transactions. The hard part is that only 0.17% of transactions are fraud, so accuracy is meaningless. I used SMOTE to rebalance the training data, trained three models, and found that a Random Forest gave the best balance: it catches 82% of fraud, and 82% of what it flags really is fraud. I deployed it as a live web app: a React page talks to a FastAPI service that runs the model."

### 3 minutes
1. **Problem.** Fraud is rare and costly. A model that says "legitimate" every time is 99.83% accurate and useless.
2. **First attempt, and why I dropped it.** My first dataset (e-commerce fraud) hit a wall: recall stuck at 0.53 whatever I tried, because one feature, time between signup and purchase, did almost all the work. I switched datasets because the evidence said the information wasn't there, not because I guessed.
3. **Second dataset.** The ULB credit-card data: 284,807 transactions, 492 fraud.
4. **Method.** Scale two columns, split 80/20 keeping the fraud ratio, apply SMOTE to the training set only, train Random Forest, Gradient Boosting and AdaBoost, and evaluate on the untouched test set.
5. **Result.** Random Forest: precision 0.82, recall 0.82, F1 0.82. The boosting models had higher recall but terrible precision. The ensemble didn't beat Random Forest.
6. **System.** A FastAPI backend loads the saved model and scaler. A React frontend lets you pick a transaction and see the verdict. Backend on Render, frontend on Vercel.
7. **Honest limits.** Scores aren't calibrated probabilities, the threshold is the default 0.5, and I evaluated on one split.

### 10 minutes
Follow section 2 in order and use the demo script in section 8.

---

## 2. The story from beginning to end

Say it in this order. Each step is one reason leading to the next.

### Step 1 — The problem
- Credit-card fraud is rare (here 0.17%) and missing one costs real money; a false alarm annoys a customer.
- So the model must catch most fraud (**recall**) without drowning people in false alarms (**precision**).
- **Accuracy is the wrong metric.** Always predicting "legitimate" scores 99.83% and catches zero fraud.

### Step 2 — Iteration 1: the e-commerce dataset (the documented dead end)
- `Fraud_Data.csv`: 151,112 rows, 9.4% fraud.
- Preprocessing: dates converted, `get_dummies` for text categories (browser, source, sex) because they have no order, dropped IDs and raw timestamps.
- Engineered `time_to_purchase` (seconds between signup and purchase): fraudsters buy almost immediately.
- Result: recall 0.53, precision 0.97–1.00. The model caught only the obvious cases.
- Feature importance explained why: `time_to_purchase` 77.5%, `purchase_value` 12.4%, `age` 9.2%, everything else under 0.1%.
- I tried to fix it: `class_weight='balanced'` (recall 0.530 → 0.540, but false alarms jumped from 40 to 461, so I reverted), plus `purchase_hour` and `high_value` features (recall 0.527, no change).
- **Conclusion:** the signal needed to find the other 47% of fraud isn't in this dataset's columns. Switching datasets was a decision based on evidence.
- The ensemble gave identical results to single models because all three relied on the same feature and made the same errors.

### Step 3 — Iteration 2: the ULB dataset
- 284,807 transactions × 31 columns: `Time`, `V1`–`V28`, `Amount`, `Class`. No missing values.
- `V1`–`V28` are PCA outputs (anonymised for privacy), so I can't say what each one means.
- `Time` is seconds since the first transaction in the dataset (two days of data), `Amount` is the transaction value.
- 284,315 legitimate vs 492 fraud (0.17%).

### Step 4 — Preprocessing
- **Scale `Amount` and `Time`** with `StandardScaler` (mean 0, standard deviation 1). `V1`–`V28` already come from PCA, so they're on comparable scales.
- **Split 80/20** with `random_state=42` (same split every run) and `stratify=y` (both sets keep 0.17% fraud): 227,845 train (394 fraud), 56,962 test (98 fraud).
- **SMOTE on the training set only**: 394 → 227,451 fraud, matching the legitimate count.
  - After the split, because the test set must look like reality. SMOTE before splitting would put synthetic fraud in the test set and inflate every score.
  - Better than copying fraud rows: copies add no new information; SMOTE makes new, plausible points.

### Step 5 — Three models, same data
All trained on the same SMOTE data (100 trees each, `random_state=42`), tested on the original test set:

| Model | Precision | Recall | F1 | AUC-ROC | False alarms |
|---|---|---|---|---|---|
| **Random Forest** | 0.82 | 0.82 | 0.82 | 0.9688 | 17 |
| Gradient Boosting | 0.11 | 0.90 | 0.19 | 0.9807 | 746 |
| AdaBoost | 0.05 | 0.91 | 0.10 | 0.9747 | 1,598 |

- Random Forest won on balance. Its confusion matrix: 56,847 true negatives, **17** false positives, **18** missed frauds, **80** caught frauds.
- Goal was recall above 0.80, up from 0.53 in Iteration 1. Achieved.

### Step 6 — Ensembles
- **Hard voting:** each model votes; fraud if at least 2 of 3 agree. Precision 0.14, recall 0.90, F1 0.24 (545 false alarms).
- **Soft voting:** average the three fraud probabilities; fraud if the average is at least 0.5. Precision 0.44, recall 0.88, F1 0.59 (110 false alarms).
- Neither beat Random Forest alone. Soft beat hard because Random Forest's lower confidence pulled the average down, while in hard voting two over-eager models simply outvote it.

### Step 7 — Saving and serving the model
- Models and the scaler are saved with `joblib` into `models/`.
- The scaler is saved too because **new transactions must be scaled exactly as the training data was**. Otherwise `Amount` would mean something different to the model.

### Step 8 — Backend (FastAPI)
When the page sends a transaction to `/predict`:
1. **Validate**: 30 required numbers (`V1`–`V28`, `Amount`, `Time`), otherwise a clear 422 error.
2. **Scale** `Amount` and `Time` with the saved scaler.
3. **Arrange** the 30 values in the exact column order the model was trained on.
4. **Predict**: `predict_proba` gives the fraud probability; the class is fraud when it is 0.5 or more.
5. **Return** `{"prediction": "fraud", "confidence": 0.97}`.

`/compare` runs all three models plus both ensembles. `/health` is a simple "I'm alive" check.

### Step 9 — Frontend (React)
- Pick one of five real transactions (taken from the test set, per your notes), optionally edit the Amount, click **Predict** or **Compare all models**.
- It shows the verdict and a "fraud score" bar.
- On page load it wakes the sleeping free-tier server.

### Step 10 — Deployment
- Backend on **Render** (free tier: sleeps when idle, first request can take 50+ seconds). Run by Uvicorn.
- Frontend on **Vercel** (auto-deploys when `main` changes).
- GitHub Actions runs the tests and the build on every pull request.

---

## 3. Key concepts in plain words

| Term | One-line explanation | Fraud example |
|---|---|---|
| Class imbalance | One class is far rarer than the other | 0.17% fraud |
| Precision | Of everything flagged as fraud, how much really was | 80 caught / (80 + 17 false alarms) = 0.82 |
| Recall | Of all real fraud, how much was caught | 80 / (80 + 18 missed) = 0.82 |
| F1 | A single score balancing precision and recall | 0.82 |
| Confusion matrix | The four outcomes: true/false positive/negative | RF: 56,847 / 17 / 18 / 80 |
| False negative | Fraud the model missed | Money lost |
| False positive | A legitimate transaction flagged | Annoyed customer |
| ROC-AUC | How well the model *ranks* fraud above legitimate across all thresholds (1.0 = perfect) | 0.9688 |
| PR-AUC | Like ROC-AUC, but focused on the rare class, so much stricter | Not yet computed on the real data |
| Threshold | The score above which we say "fraud" | 0.5 here |
| SMOTE | Creates synthetic fraud points between real ones, using nearest neighbours | 394 → 227,451 |
| Stratify | Keep the class ratio the same in train and test | 0.17% in both |
| Data leakage | Information from the test set sneaking into training | See section 9 |
| Calibration | Whether a "0.8" score really means 80% chance | Not calibrated here |
| PCA | Mixes the original columns into new anonymous ones | `V1`–`V28` |
| Ensemble | Combining several models | Hard and soft voting |

**How SMOTE works:** pick a fraud point, find its 5 nearest fraud neighbours, pick one, and create a new point at a random spot on the line between them.

**How the three models differ:**
- *Random Forest*: 100 trees built **independently**, each on a random sample; they vote. Mistakes average out.
- *Gradient Boosting*: 100 trees built **one after another**, each fixing the previous one's errors.
- *AdaBoost*: also sequential; it **increases the weight** of examples earlier trees got wrong.

---

## 4. Why each decision was made

| Decision | Reason | Possible follow-up |
|---|---|---|
| Precision/recall/F1 instead of accuracy | Accuracy is 99.83% for a useless model | "Which matters more?" Recall (missed fraud costs money), but precision matters too (blocking customers). |
| Switched datasets | Evidence: feature importance and a recall ceiling | "Couldn't you have engineered more features?" I tried two more and class weights; nothing moved. |
| `get_dummies` for categories | No natural order; numbering them would imply Safari > Chrome | |
| `StandardScaler` on `Amount` and `Time` | **SMOTE uses distances between points**, so a column with huge values would dominate. Tree models themselves don't need scaling. | "Random Forest doesn't need scaling, why did you?" Because of SMOTE. |
| Stratified split | Keeps 0.17% fraud in both sets | Without it, the test set might get too few frauds. |
| `random_state=42` | Reproducible results | |
| SMOTE after the split, train only | Test set must stay real | |
| Random Forest as the final model | Best F1 and balance | "Chosen on the test set?" Yes, see section 9. |
| Saved the scaler | Same scaling for new data | |
| FastAPI instead of Flask | Built-in validation, automatic docs at `/docs`, type hints; same output (checked) | "Why change a working app?" Cleaner input checking and testing, no change in behaviour. |
| Render + Vercel | Free, simple, auto-deploy | |

---

## 5. Likely questions and good answers

### About the data and problem
**Q: What do V1–V28 mean?**
They're the output of PCA applied to the original card data, anonymised for privacy. I can't interpret them individually. That limits explainability, but the model only needs the numbers.

**Q: Why is the data so imbalanced, and why is that a problem?**
Real fraud is rare. A model trained on it learns to say "legitimate" almost always, because that's right 99.83% of the time. We have to make fraud count more during training (SMOTE) and judge it with precision and recall.

**Q: What does `Time` mean?**
Seconds since the first transaction in the dataset, not time of day. So it's of limited use for a new transaction, and a real system would use something like hour of day.

### About preprocessing
**Q: Why SMOTE and not just duplicate the fraud rows?**
Duplicates add no new information; the model just memorises 394 points. SMOTE creates new points between real ones, so the model sees a wider fraud region.

**Q: Why did you apply SMOTE after splitting?**
So the test set stays realistic. If I oversampled first, synthetic fraud would be in the test set, and the scores would look better than real-world performance.

**Q: Did you try anything other than SMOTE?**
In Iteration 1 I tried `class_weight='balanced'` (it raised false alarms from 40 to 461 for almost no recall gain). On the ULB data I only tried SMOTE; comparing it with class weights, undersampling or SMOTE variants is future work.

### About the models
**Q: Why did Gradient Boosting and AdaBoost have such low precision?**
My explanation in the notes: they're sequential, so each tree reacts to the previous trees' mistakes, and on a training set that is half synthetic fraud they became too eager to flag fraud. Random Forest's independent trees average that out. **Be careful:** this is my hypothesis, and I didn't run an experiment that isolates it. One piece of evidence points the same way: their AUC-ROC is the highest (0.98 and 0.97), so they *rank* fraud well, and their problem is mostly where the 0.5 cut-off falls on over-confident scores. Choosing a better threshold would test this.

**Q: If AUC-ROC is higher for Gradient Boosting, why is Random Forest "better"?**
AUC-ROC measures ranking across all thresholds. I judged the models at the 0.5 threshold, where precision and recall matter in practice. With so few frauds, ROC-AUC also looks flattering for almost any model, which is why I'd add PR-AUC.

**Q: Why didn't the ensemble beat Random Forest?**
Ensembles help when models make *different* errors. Gradient Boosting and AdaBoost make the same error (too many false alarms), so combining them spreads it rather than cancels it.

**Q: How do hard and soft voting differ, and why did soft win?**
Hard: majority of yes/no votes. Soft: average of probabilities. In soft voting Random Forest's confident "legitimate" pulls the average down. In hard voting, two over-eager models simply outvote it.

**Q: What is overfitting and did you check?**
When a model memorises training data and fails on new data. I evaluated on a held-out test set (never seen in training or SMOTE), and the results are good there. I didn't use cross-validation, which would be a stronger check.

### About evaluation
**Q: Walk me through your confusion matrix.**
Out of 56,962 test transactions: 56,847 legitimate correctly passed, 17 legitimate wrongly flagged, 18 frauds missed, 80 frauds caught. Precision is 80 of 97 flagged = 0.82; recall is 80 of 98 = 0.82.

**Q: Is 82% recall good enough?**
It meets my goal of above 80%, up from 53%, but 18 frauds still slip through. In practice the threshold would be set from the cost of a missed fraud versus a false alarm.

**Q: Why not use accuracy?**
Because the model "legitimate always" scores 99.83%.

### About the system
**Q: What happens when I click Predict?**
Use the 5 steps in section 2, Step 8. Mention scaling and column order, because they're the easy places to go wrong.

**Q: Why does the first request take so long?**
Render's free tier puts the server to sleep. The page now pings `/health` when it opens to wake it early.

**Q: What does the "fraud score" mean?**
The Random Forest's probability that the transaction is fraud. It isn't a calibrated real-world probability (see section 9).

**Q: Why FastAPI?**
Automatic validation (bad input gets a clear error instead of a crash), automatic docs at `/docs`, and clean type hints. I checked that the old Flask version and the new FastAPI version give identical predictions on the same five transactions.

**Q: How did you test it?**
15 automated tests: valid prediction, bad input, the five demo transactions keeping their verdicts, the compare endpoint, and a health check. A GitHub Actions workflow runs them and the frontend build on every pull request.

**Q: What would you change for production?**
Retrain regularly (fraud patterns drift), pick the threshold from business cost, calibrate probabilities, add explanations (for example SHAP) so analysts see *why* a transaction was flagged, monitor performance, add authentication and rate limiting, restrict CORS, and send flagged transactions to human review rather than blocking automatically.

---

## 6. Draw this on the board

**Pipeline**
```
creditcard.csv ─▶ scale Amount, Time ─▶ split 80/20 (stratified)
                                             │
                          train 80% ──▶ SMOTE ──▶ RF / GB / AdaBoost ──▶ save .pkl
                          test 20% ──────────────────────────────────▶ evaluate (real, imbalanced)
```

**Live request**
```
React page ──JSON──▶ FastAPI ──validate──▶ scale Amount,Time ──▶ Random Forest ──▶ {"prediction","confidence"}
(Vercel)             (Render)              (scaler.pkl)           (rf_model.pkl)
```

---

## 7. Know who did what

Examiners often ask "which part did you write?" or ask you to open a file and explain it. Be straightforward:

| Part | Honest description |
|---|---|
| Both iterations: exploration, preprocessing, training, ensembles, evaluation, reasoning in the `NOTES.md` files | Yours. Know every line. |
| React UI (original version) | Your `NOTES.md` says it was built by your **mentor**. Say so plainly, and add what you did: connected it to the backend, set up the Vite proxy, and deployed it. |
| Flask → FastAPI port, validation, tests, `/compare`, editable Amount, wake-up ping, CI workflow, `06_improved_evaluation.py`, the guides | Written with an AI assistant (Claude) during a review of the repo. If asked, say that. Before the viva, read each of these files until you can explain every function and why it exists. |

Saying "I used an AI assistant for the migration and tests, and I reviewed and can explain all of it" is a better answer than being caught out. Whatever you present as yours, make sure it is.

---

## 8. Live demo script (3–4 minutes)

**Before the viva:** open the live site **at least 2 minutes early** so the Render server wakes up. Have the local version ready as a backup (`uvicorn app:app --port 5000` and `npm run dev`), plus screenshots in case the internet fails.

1. Open the page. Point at the status line: "Server ready". Explain the wake-up ping.
2. Select **Identity Theft Purchase — $239.94** → **Predict**. Result: fraud, score about 97%. "A clear fraud case; the model is confident."
3. Click **Compare all models**. Show the table: all three say fraud, hard voting 3 of 3, soft voting about 84%.
4. Change **Amount** to `10` and predict again (score drops to about 93%). "Same pattern, different amount: the V-features do most of the work."
5. Select **Suspicious Foreign Transfer — $239.93** → **Predict**. Result: **legitimate**, score about 8%. "This is one of the frauds my model misses; recall is 0.82, not 1.0. That's honest behaviour, not a bug."
6. Click **Compare all models** on the same transaction. Gradient Boosting (74%) and AdaBoost (53%) flag it; hard voting catches it; soft voting (45%) misses it. "This shows the trade-off in my ensemble experiment: the boosters catch more fraud but raise many more false alarms across the whole dataset."
7. Optional: open `/docs` on the API, show the request format and try one request there.

**Likely demo questions:** "What if I send bad data?" (show a 422 in `/docs`). "Why does this transaction score only 51%?" (the first preset is borderline: Random Forest gives it 0.51, just over 0.5).

---

## 9. Weak points: raise these yourself

Owning limitations shows understanding. Each has a short answer.

1. **Scaler fitted before the split (mild data leakage).** The scaler learned its mean and spread from *all* rows, including the test set, so a little test-set information leaks into training. The effect is tiny with 284,807 rows, but the correct way is to split first and fit on the training part only. `06_improved_evaluation.py` does it that way.
2. **Model chosen on the test set.** I compared models and picked Random Forest using test-set scores, with no separate validation set, so the reported scores are slightly optimistic. The `06` script uses a validation split for the threshold.
3. **Default threshold of 0.5.** It wasn't tuned. Better: choose it from a precision-recall curve or from the cost of missed fraud versus a false alarm.
4. **Scores aren't real probabilities.** SMOTE trained the model on 50% fraud, so scores run higher than the real 0.17% rate. Fix: calibration.
5. **One split, 98 frauds in the test set.** Each missed fraud moves recall by about 1 point. Cross-validation would give a mean ± standard deviation.
6. **Only SMOTE tried on the ULB data.** Class weights, undersampling and other methods weren't compared there.
7. **Old data.** Two days of European transactions from 2013. Fraud patterns change.
8. **Explainability.** Because of PCA I can't tell a user why a transaction was flagged.
9. **Demo is five fixed transactions** (plus an editable amount), not free-form input; nobody can type 28 PCA values.
10. **Deployment is a demo.** Free tier cold starts, open CORS by default, no authentication.

If asked "what would you do next?", the answer is the `06` script's agenda: fix the leakage, report PR-AUC, tune the threshold on validation data, calibrate, and cross-validate. Then compare XGBoost and other imbalance methods.

---

## 10. Numbers cheat sheet

| Fact | Value |
|---|---|
| ULB transactions | 284,807 (492 fraud, 0.17%) |
| Iteration 1 dataset | 151,112 rows, 9.4% fraud |
| Iteration 1 recall ceiling | 0.53 |
| Iteration 1 top feature | `time_to_purchase`, 77.5% (68.2% after more features) |
| Train / test | 227,845 / 56,962 |
| Frauds in train / test | 394 / 98 |
| After SMOTE | 227,451 fraud + 227,451 legitimate |
| Random Forest | Precision 0.82, recall 0.82, F1 0.82, AUC 0.9688 |
| RF confusion matrix | TN 56,847 · FP 17 · FN 18 · TP 80 |
| Gradient Boosting | 0.11 / 0.90 / 0.19, AUC 0.9807, 746 false alarms |
| AdaBoost | 0.05 / 0.91 / 0.10, AUC 0.9747, 1,598 false alarms |
| Hard voting | 0.14 / 0.90 / 0.24, 545 false alarms |
| Soft voting | 0.44 / 0.88 / 0.59, 110 false alarms |
| Demo results | 4 of 5 presets flagged as fraud by Random Forest |
| Trees / seed | 100 each / `random_state=42` |

---

## 11. Last-minute checklist

- [ ] Read `NOTES.md` (both), `PROJECT_GUIDE.md` and this guide once more.
- [ ] Run `06_improved_evaluation.py` on your machine and write the real numbers here, so you can say "I re-evaluated more rigorously and got…".
- [ ] Confirm Render's start command is `uvicorn app:app --host 0.0.0.0 --port $PORT` and the live site works.
- [ ] Open the live site 2+ minutes before you present.
- [ ] Be able to open `app.py`, `02_preprocess.py` and `03_train_models.py` and explain each line.
- [ ] Practise the 30-second, 3-minute and 10-minute versions out loud.
