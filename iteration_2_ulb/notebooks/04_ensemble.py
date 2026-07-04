import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

df  = pd.read_csv("../data/processed/creditcard_cleaned.csv")

x = df.drop(["Class"], axis=1)
y = df["Class"]
X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size = 0.2, random_state = 42, stratify=y)

smote = SMOTE(random_state=42)

X_train_sm, Y_train_sm = smote.fit_resample(X_train, Y_train)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
ada_model = AdaBoostClassifier(n_estimators=100, random_state=42)

rf_model.fit(X_train_sm, Y_train_sm)
gb_model.fit(X_train_sm, Y_train_sm)
ada_model.fit(X_train_sm, Y_train_sm)

rf_preds = rf_model.predict(X_test)
gb_preds = gb_model.predict(X_test)
ada_preds = ada_model.predict(X_test)

hard_vote_preds = ((rf_preds + gb_preds + ada_preds) >= 2).astype(int)

print("=== Hard Voting Ensemble ===")
print(classification_report(Y_test, hard_vote_preds))
print(confusion_matrix(Y_test, hard_vote_preds))

rf_proba = rf_model.predict_proba(X_test)[:, 1]
gb_proba = gb_model.predict_proba(X_test)[:, 1]
ada_proba = ada_model.predict_proba(X_test)[:, 1]

avg_proba = (rf_proba + gb_proba + ada_proba) / 3
soft_vote_preds = (avg_proba >= 0.5).astype(int)

print("=== Soft Voting Ensemble ===")
print(confusion_matrix(Y_test, soft_vote_preds))
print(classification_report(Y_test, soft_vote_preds))

import joblib

joblib.dump(rf_model, '../models/rf_model.pkl')
joblib.dump(gb_model, '../models/gb_model.pkl')
joblib.dump(ada_model, '../models/ada_model.pkl')

print("Models saved successfully")

