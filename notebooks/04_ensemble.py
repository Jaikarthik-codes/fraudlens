import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load data
df = pd.read_csv("../data/processed/Fraud_Data_Cleaned.csv")
X = df.drop(columns=["class"])
Y = df["class"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

# Train all three Models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
ada_model = AdaBoostClassifier(n_estimators=100, random_state=42)

rf_model.fit(X_train, Y_train)
gb_model.fit(X_train, Y_train)
ada_model.fit(X_train, Y_train)

# Hard voting - majority wins
rf_preds = rf_model.predict(X_test)
gb_preds = gb_model.predict(X_test)
ada_preds = ada_model.predict(X_test)

hard_vote_preds = ((rf_preds + gb_preds + ada_preds) >= 2).astype(int)

print("=== Hard Voting Ensemble ===")
print(confusion_matrix(Y_test, hard_vote_preds))
print(classification_report(Y_test, hard_vote_preds))

# Soft voting - average the probabilities
rf_proba = rf_model.predict_proba(X_test)[:, 1]
gb_proba = gb_model.predict_proba(X_test)[:, 1]
ada_proba = ada_model.predict_proba(X_test)[:, 1]

avg_proba = (rf_proba + gb_proba + ada_proba) / 3
soft_vote_preds = (avg_proba >= 0.5).astype(int)

print("=== Soft Voting Ensemble ===")
print(confusion_matrix(Y_test, soft_vote_preds))
print(classification_report(Y_test, soft_vote_preds))