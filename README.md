# FraudLens
Multimodel fraud detection system for credit card transactions.
Built from scratch to understand every decision deeply enough 
to defend it in a technical interview.

## Project Structure
FraudLens/
├── iteration_1_fraud_data/   ← Documented dead end (Fraud_Data.csv)
└── iteration_2_ulb/          ← Final system (ULB Credit Card dataset)
    ├── backend/              ← Flask API + trained models
    ├── frontend/             ← React app (Vite + TypeScript)
    ├── notebooks/            ← Training and evaluation scripts
    └── models/               ← Saved .pkl model files

## Iteration 1 — Fraud_Data.csv (Documented Dead End)
E-commerce fraud dataset with interpretable features.
Engineered: time_to_purchase, purchase_hour, high_value.

Finding: time_to_purchase dominated at 68% feature importance
even after full feature engineering. Recall ceiling at 0.53.
Switched datasets based on evidence from feature importance 
analysis, not assumption.

## Iteration 2 — ULB Credit Card Fraud Dataset
284,807 transactions, V1-V28 PCA features + Amount + Time.
Fraud rate: 0.17% — handled with SMOTE after train/test split.

Results:
Model              Precision  Recall   F1
Random Forest        0.82      0.82    0.82  ← winner
Gradient Boosting    0.11      0.90    0.19
AdaBoost             0.05      0.91    0.10

Random Forest won on F1 score. Gradient Boosting and AdaBoost 
had higher recall but catastrophic precision due to sequential 
tree architecture amplifying SMOTE synthetic patterns.

Ensemble results:
Hard Voting          0.14      0.90    0.24
Soft Voting          0.44      0.88    0.59

Ensemble did not outperform Random Forest individually because 
all three models make similar errors on this dataset.

## Flask API
Accepts POST request with raw V1-V28, Amount, Time values.
Scales Amount and Time using StandardScaler.
Returns prediction (fraud/legitimate) and confidence score.

## React Frontend
Dark theme UI with FraudLens branding.
Preset transactions dropdown using real fraud cases from test set.
Displays prediction result with confidence percentage bar.

## How to Run

### Backend
cd iteration_2_ulb/backend
pip install -r requirements.txt
python app.py

### Frontend
cd iteration_2_ulb/frontend
npm install
npm run dev

Open http://localhost:5173

## Data
Download creditcard.csv from Kaggle:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
Place in iteration_2_ulb/data/raw/

## Stack
Python, pandas, NumPy, scikit-learn, imbalanced-learn, 
Flask, flask-cors, React, TypeScript, Vite

Update README: add project overview