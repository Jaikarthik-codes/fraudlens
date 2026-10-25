# FraudLens — Project Guide

A plain-language guide to what FraudLens does, how it works, what has changed since the project began, and what could be done next. Written so you can explain and defend every part of it.

---

## 1. What it does

FraudLens takes a credit-card transaction (30 numbers: `V1`–`V28`, `Amount`, `Time`) and says whether it looks **fraud** or **legitimate**, along with a score from 0 to 1 for how strongly the model leans toward fraud.

Users pick one of five real fraud cases from the test set in a React web page. The page sends the numbers to a Python API, and the API runs a trained Random Forest and returns the answer.

```
React page (Vercel)  ──POST /predict (JSON)──▶  FastAPI (Render)  ──▶  scaler.pkl + rf_model.pkl
        ◀────────── {"prediction": "fraud", "confidence": 0.93} ──────────────┘
```

The repository has two iterations:

| Folder | Dataset | Outcome |
|---|---|---|
| `iteration_1_fraud_data/` | `Fraud_Data.csv` (e-commerce, 9.4% fraud) | Documented dead end. Recall stuck at 0.53 because `time_to_purchase` carried 68–77% of the signal and no other feature added much. |
| `iteration_2_ulb/` | ULB Credit Card Fraud (284,807 rows, 0.17% fraud) | The final system. Random Forest reached recall 0.82. |

---

## 2. How it works

### 2.1 Machine-learning pipeline (`iteration_2_ulb/notebooks/`)

Run the scripts in order from the `notebooks/` folder. They need `data/raw/creditcard.csv` (download from Kaggle; it is too big for git).

| Script | What it does |
|---|---|
| `01_explore.py` | Looks at shape, missing values and class balance (284,315 legitimate vs 492 fraud). |
| `02_preprocess.py` | Scales `Amount` and `Time` with `StandardScaler` (the V columns are already PCA-scaled). Drops the originals and adds `scaled_Amount`, `scaled_Time`. Splits 80/20 with `stratify` so both sets keep the 0.17% fraud rate. Applies **SMOTE to the training set only**. Saves `creditcard_cleaned.csv` and `models/scaler.pkl`. |
| `03_train_models.py` | Trains Random Forest, Gradient Boosting and AdaBoost on the SMOTE data and tests them on the untouched test set. |
| `04_ensemble.py` | Tries hard voting and soft voting across the three models, then saves the models to `models/`. |
| `05_evaluate.py` | Confusion matrices, AUC-ROC and feature importance. |
| `06_improved_evaluation.py` | **Added in review.** A stricter re-evaluation of the Random Forest (see below). Analysis only: it never overwrites `models/`, so the live site is unaffected. |

**What `06_improved_evaluation.py` adds.** Run it locally with `python 06_improved_evaluation.py` (add `--skip-cv` to skip the slow part). It writes `reports/06_improved_evaluation.json`.

1. *Scaler fitted on the training data only.* `02_preprocess.py` fits the scaler on the whole dataset before splitting, so the test set's mean and spread leak into the scaling. The effect is tiny but it is the classic data-leakage question. Here the split comes first.
2. *PR-AUC next to ROC-AUC.* With 0.17% fraud, ROC-AUC looks excellent for almost any model (all three of yours are above 0.96, including the one with 0.05 precision). PR-AUC (average precision) is much harsher and shows real differences.
3. *Threshold chosen on a validation split.* Part of the training data is held back, the F1-best threshold is found there, and it is reported once on the test set. The test set never picks the threshold. A sweep from 0.1 to 0.9 is printed to show the precision/recall trade-off.
4. *Calibration.* SMOTE trains the model on 50% fraud, so its scores run far above the real 0.17%. Isotonic calibration pulls the average score back toward the true rate. The script prints the actual rate, the average score before and after, and the Brier score.
5. *5-fold cross-validation.* Reports mean ± standard deviation for precision, recall, F1, ROC-AUC and PR-AUC. Scaling and SMOTE sit inside the pipeline, so each fold fits them on its own training part only.

I could only test this script on synthetic data (the Kaggle file is not in the review environment), so it is proven to run but **its real numbers do not exist yet**. Run it on your machine and record the results here before you quote them.

Key decisions and why (all in `iteration_2_ulb/NOTES.md`):

- **SMOTE after the split.** The test set must look like reality (0.17% fraud). Oversampling before splitting would put synthetic fraud into the test set and inflate the scores.
- **Accuracy is ignored.** Predicting "legitimate" every time scores 99.83%. Precision, recall and F1 are used instead.
- **Random Forest won.** Trees are built independently and vote, so aggression from SMOTE gets averaged out. Gradient Boosting and AdaBoost build trees one after another, each correcting the last, and amplified the synthetic patterns. They got high recall (0.90–0.91) but very poor precision (0.11 and 0.05).
- **The ensemble did not help.** Gradient Boosting and AdaBoost make the same kind of error, so combining them spreads the problem. Soft voting (F1 0.59) beat hard voting (0.24), but neither beat Random Forest alone (0.82).

Final results on the test set (56,962 rows, 98 frauds):

| Model | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|
| **Random Forest** | 0.82 | 0.82 | 0.82 | 0.9688 |
| Gradient Boosting | 0.11 | 0.90 | 0.19 | 0.9807 |
| AdaBoost | 0.05 | 0.91 | 0.10 | 0.9747 |

Random Forest confusion matrix: 56,847 true negatives, 17 false positives, 18 false negatives, 80 true positives.

### 2.2 Backend (`iteration_2_ulb/backend/app.py`)

A FastAPI app with three routes:

- `POST /predict` — body is a JSON object with 30 numeric fields (`V1`…`V28`, `Amount`, `Time`).
  1. Pydantic checks the body. Missing or non-numeric fields get a **422** response that names them.
  2. `scaler.transform` scales raw `Amount` and `Time` exactly as in training.
  3. The values are arranged into a one-row table using the column names and order the model remembers from training (`V1`…`V28`, `scaled_Amount`, `scaled_Time`). Because the columns are named, scikit-learn itself raises an error if one is ever missing or misnamed.
  4. `rf_model.predict` gives the class and `predict_proba` gives the fraud probability.
  5. Returns `{"prediction": "fraud" | "legitimate", "confidence": 0.xxxx}`.
- `POST /compare` — same input. Scores the transaction with Random Forest, Gradient Boosting and AdaBoost, then applies the two ensemble rules from `04_ensemble.py`: **hard voting** (fraud if at least 2 of 3 models say fraud) and **soft voting** (average the three fraud probabilities, fraud at 0.5 or above). Uses the `gb_model.pkl` and `ada_model.pkl` files you already had.
- `GET /health` — returns `{"status": "ok"}` (for uptime checks and for waking Render).

FastAPI also generates interactive docs at `/docs`. CORS defaults to open (`*`) so the Vercel site can call the API; setting the `CORS_ORIGINS` environment variable on Render (comma-separated list of sites) restricts it.

### 2.3 Frontend (`iteration_2_ulb/frontend/`)

Vite + React + TypeScript. `src/App.tsx` holds five preset real transactions, a dropdown, an editable Amount box, a **Predict** button and a **Compare all models** button.

- **Predict** shows the verdict and a "Fraud score" bar (the Random Forest's fraud probability).
- **Compare all models** shows a table: each model's verdict and score, plus the hard- and soft-voting results.
- **Editable Amount.** The box starts at the preset's amount. Change it to see how the same transaction scores at a different amount. Negative or empty values disable the buttons.
- **Server wake-up.** When the page opens it quietly calls `/health`. The free Render server sleeps when idle, so this wakes it before the user clicks. A status line shows "Waking up the server…", "Server ready" or "Server unreachable".

The API address comes from `VITE_API_URL`:

- Development: empty, so `/predict`, `/compare` and `/health` go through the Vite proxy (`vite.config.ts`) to a local backend on port 5000.
- Production build: `.env.production` points to the Render URL.

### 2.4 Deployment

- Backend on Render (free tier; the first request after idle can take 50+ seconds).
- Frontend on Vercel (auto-deploys on push to `main`).

---

## 3. History from the beginning

From `git log`, oldest first:

| Date | What happened |
|---|---|
| 2026-07-01 | Initial commit: Iteration 1 exploration and preprocessing. `.gitignore` added, IDE files removed. |
| 2026-07-03 | Iteration 1 feature engineering, three models, ensemble, evaluation. Repo reorganised into `iteration_1_fraud_data/` and `iteration_2_ulb/`. ULB exploration and preprocessing added. |
| 2026-07-04 | Iteration 2 model training: Random Forest recall 0.82. |
| 2026-07-05 | Flask app with `/predict`; React frontend connected; scaler saved; `gunicorn` added; API URL pointed at Render; notes and README completed. |
| 2026-07-06 – 07-15 | Deployment notes, capitalisation fixes, README edits. |
| 2026-07-26 – 08-07 | README edits; "FraudLens complete – React frontend connected to Flask backend". |
| 2026-09-10 | README changed from the local URL to the live site. |

### Changes made in this review session

| # | Change | Where | Why |
|---|---|---|---|
| 1 | Removed the merge-conflict markers (`<<<<<<<`, `=======`, `>>>>>>>`) | `.gitignore` | Left over from a merge; the rules below them could be misread. Both halves kept. |
| 2 | One scaler fitted on `[Amount, Time]`, now saved by the pipeline | `notebooks/02_preprocess.py` | Before, `scaler.pkl` came only from `RoughWork.py`. The old code reused one scaler, fitting it on Amount and then on Time. The numbers are the same (each column is standardised on its own), but now the tracked pipeline produces the exact file the API loads. Models were **not** retrained. |
| 3 | **Flask → FastAPI** | `backend/app.py`, `requirements.txt` | Requested. See section 4. |
| 4 | Model paths resolved from `app.py`'s own location | `backend/app.py` | Previously `'../models/...'` only worked when started from inside `backend/`. |
| 5 | Pydantic request validation | `backend/app.py` | Missing/bad fields used to give a 500 `KeyError`. Now a clear 422 naming the fields. |
| 6 | Added `GET /health` | `backend/app.py` | For uptime checks. |
| 7 | Port read from the `PORT` environment variable (default 10000) | `backend/app.py` | Hosting platforms set the port themselves. |
| 8 | Removed the old HTML form route `/`, the `presets` dictionary, and `templates/index.html` | `backend/` | They were left over from before React: the form posted form-data to `/predict`, which has accepted only JSON since the React switch. Nothing used them. |
| 9 | Removed unused variables (`amount`, `time`) and a double call to `scaler.transform` | `backend/app.py` | Cleanup only; same result. |
| 10 | `scikit-learn==1.9.0` pinned; `gunicorn` and `flask-cors` removed; `fastapi`, `uvicorn` added; `requirements-dev.txt` added (`pytest`, `httpx`) | `backend/` | The pickles were saved with scikit-learn 1.9.0 and may not load on other versions. |
| 11 | API URL read from `VITE_API_URL` instead of being hard-coded | `frontend/src/App.tsx`, `frontend/.env.production` | Local development was silently calling the live Render server. |
| 12 | Added tests | `backend/test_app.py` | Covers `/health`, a valid prediction, missing fields, non-numeric fields and non-JSON bodies. |
| 13 | README fixes | `README.md` | Added `data/` to the structure, finished the cut-off "Stacks Used" list, documented the 422 behaviour, `/docs`, run commands, and the confidence caveat. |

### Second round of changes (after the FastAPI switch)

| # | Change | Where | Why |
|---|---|---|---|
| 14 | New `POST /compare` endpoint | `backend/app.py` | Shows all three models and both ensembles for one transaction, using model files you already had. Turns your ensemble experiment into something visible on the live site. |
| 15 | Features built as a named table, in the order the model remembers | `backend/app.py` | Removes the "column order matters" fragility and the scikit-learn warnings. Verified: results are identical to before. |
| 16 | `CORS_ORIGINS` environment variable (default stays `*`) | `backend/app.py` | Lets you lock the API to your Vercel site without a code change. Off by default so nothing breaks. |
| 17 | Server wake-up ping and status line | `frontend/src/App.tsx` | Render's free tier sleeps; the page now wakes it on load and tells the user what is happening. |
| 18 | Editable Amount box | `frontend/src/App.tsx` | Lets you demonstrate how the score changes with the amount. |
| 19 | "Compare all models" button and table | `frontend/src/App.tsx`, `index.css` | The front end for `/compare`. |
| 20 | "Confidence" relabelled **"Fraud score"** | `frontend/src/App.tsx` | The number is the *fraud* probability. For a legitimate result it read like "Legitimate, confidence 8%", which looks wrong. |
| 21 | Vite proxy now covers `/compare` and `/health` | `frontend/vite.config.ts` | Found during browser testing: only `/predict` was proxied, so the new routes failed in local development. |
| 22 | Regression tests for the five demo transactions, `/compare` tests, and `sample_transactions.json` | `backend/test_app.py`, `backend/sample_transactions.json` | 15 tests in total. The five expected verdicts match your NOTES.md (4 of 5 flagged). |
| 23 | New evaluation script | `notebooks/06_improved_evaluation.py` | See section 2.1. Analysis only; nothing deployed changes. |
| 24 | GitHub Actions workflow | `.github/workflows/ci.yml` | Runs the backend tests and the frontend build on every pull request and every push to `main`. |

**Not changed:** the trained models, the datasets, the results, the notebooks other than `02_preprocess.py` (changed) and the new `06` script, and the two `NOTES.md` files (they remain your original reasoning).

---

## 4. Flask → FastAPI: what changed and what to know

| | Flask (before) | FastAPI (now) |
|---|---|---|
| Server | `gunicorn app:app` (WSGI) | `uvicorn app:app` (ASGI) |
| Input checking | Hand-written, none at first | Declared once as a Pydantic model; automatic |
| Bad input | 500 (then 400 in an intermediate edit) | 422 with details |
| CORS | `flask-cors` | `CORSMiddleware` (built in) |
| API docs | None | Automatic at `/docs` |
| Predict handler | Synchronous | Still plain `def`, which FastAPI runs in a thread, so slow model calls don't block other requests |

The request and response JSON are unchanged, so the React page works without modification. This was checked, not assumed: your original Flask `app.py` (from git history) and the new FastAPI app were run on the same five demo transactions and returned identical predictions and scores.

### ⚠️ Required action before deploying to Render

The Render service's **Start Command** is set in the Render dashboard (it is not in the repo) and is currently something like `gunicorn app:app`. That will **not** work with FastAPI. Change it to:

```
uvicorn app:app --host 0.0.0.0 --port $PORT
```

Also confirm the build still runs `pip install -r requirements.txt` from the `backend` directory. Until the start command is changed, do not merge to the branch Render deploys from, or the live API will go down.

---

## 5. Current verification status

**Checked in the review environment (Python 3.13, scikit-learn 1.9.0, Node 22):**

- Backend: all 15 `pytest` tests pass.
- Original Flask app vs new FastAPI app: identical output on all five demo transactions.
- Frontend: `npm run build` (TypeScript check plus production build) succeeds.
- Whole site in a real browser (Chromium driven by Playwright, local API plus Vite dev server): the status line reaches "Server ready", Predict and Compare both work, negative and empty amounts disable the buttons, an edited amount changes the score (239.94 → 97.0% fraud; 10 → 93.0%), and there were no console errors.
- `06_improved_evaluation.py` runs end to end, but only on synthetic data.

**Not checked:**

- The live Render and Vercel deployments. The Render start command change (section 4) is still yours to do.
- The GitHub Actions workflow, which only runs once it is on GitHub. Look at the "Checks" tab of the pull request.
- The real numbers from `06_improved_evaluation.py`, which need `creditcard.csv`.

To repeat the checks yourself:

```bash
cd iteration_2_ulb/backend
pip install -r requirements-dev.txt
pytest                                  # 15 tests
uvicorn app:app --port 5000             # API docs at http://127.0.0.1:5000/docs

cd ../frontend
npm install && npm run build            # type-checks and builds
npm run dev                             # open the printed address (backend must be running)

cd ../notebooks                         # needs data/raw/creditcard.csv
python 06_improved_evaluation.py --skip-cv
```

---

## 6. Known limitations (and honest answers)

These are good points to raise yourself in a viva.

1. **`confidence` is not a true probability.** The Random Forest was trained on SMOTE-balanced data (50% fraud), so its scores are shifted upward compared with the real 0.17% fraud rate. Treat it as a ranking score. `06_improved_evaluation.py` measures this and shows calibration as the fix, but the deployed model is still the uncalibrated one.
2. **The 0.5 threshold is arbitrary.** AUC-ROC is above 0.96 for all three models, so the ranking is good; the poor precision of Gradient Boosting and AdaBoost is mostly a threshold/SMOTE issue, not a weak model. The `06` script's threshold sweep tests this claim; run it before you rely on it.
3. **Only five demo transactions.** The V1–V28 columns are PCA outputs, so users can't type meaningful values. Demos use real test-set rows.
4. **Evaluated on one split.** 98 frauds in the test set means each missed fraud moves recall by about one percentage point. Cross-validation would give a steadier estimate (the `06` script does this).
5. **The ULB data is two days of European transactions from 2013.** Fraud patterns drift; the model would need retraining on current data.
6. **`Time` is seconds since the first transaction in the dataset**, not time of day, so scaling it is of limited meaning for new transactions.
7. **Open CORS and no authentication.** Fine for a demo. CORS can now be restricted with the `CORS_ORIGINS` variable, but it defaults to open, and there is still no API key.
8. **Free-tier cold starts.** The first request after idle can take 50+ seconds.
9. **Model files are committed to git** (rf_model.pkl is about 16 MB) and are tied to scikit-learn 1.9.0.
10. **`RoughWork.py` files are still tracked in git.** The `.gitignore` rule `notebooks/RoughWork.py` only matches a top-level `notebooks/` folder, so it doesn't ignore the ones inside the iteration folders. They are harmless; `iteration_2_ulb/notebooks/RoughWork.py` is the original scaler script that `02_preprocess.py` now replaces.
11. **The Vercel URL differs between files.** The README says `fraudlens-two.vercel.app` and `NOTES.md` says `fraudlens-eight.vercel.app`. Check which is live and make them agree.
12. **Iteration 1's data CSVs are tracked in git**, although the README says large data is not. Decide whether to untrack them.
13. **The training pipeline still has the scaler leakage.** `02_preprocess.py` fits the scaler on all rows before splitting, and the deployed model was trained that way. The effect is tiny, and the `06` script measures a leakage-free version, but the deployed files were not regenerated.

---

## 7. What more can be done

Ordered roughly by value for a project defence. Things marked done were added in this review.

**Model quality** (needs `creditcard.csv`, so run locally)
- **Run `06_improved_evaluation.py` and record the real numbers** in this guide. That is the single most useful next step.
- If the leakage-free, threshold-tuned and calibrated results look better, retrain with them: split first, fit the scaler on training data only, save one `Pipeline` (scaler plus model) so the API cannot use a mismatched pair, and save the chosen threshold and a `metrics.json` next to the model.
- Compare SMOTE with `class_weight`, undersampling, or SMOTE variants (Borderline-SMOTE, ADASYN); try XGBoost/LightGBM.
- Report cost-based metrics (a missed fraud costs more than a false alarm).
- Add SHAP or feature-importance output to the API to explain *why* a transaction was flagged.

**Backend**
- Done: health check, validation, model comparison endpoint, configurable CORS, tests, CI.
- Batch endpoint (`POST /predict/batch`) and CSV upload.
- Request logging and a stored history of predictions.
- An API key or rate limit.
- A Dockerfile for repeatable deployment.
- Load models with FastAPI's `lifespan` handler and expose the model version in `/health`.

**Frontend**
- Done: editable amount, model comparison view, server wake-up status, "Fraud score" label.
- Show the top contributing features for each prediction (needs the SHAP item above).
- A threshold slider showing the precision/recall trade-off (needs the real numbers from the `06` script).
- Add `eslint` and component tests to `package.json`.

**Repository**
- Move the pickles to Git LFS or a release asset; add a script that runs the notebooks in order.
- Fix the `.gitignore` for `RoughWork.py`, unify the Vercel URL, and add a LICENSE.
