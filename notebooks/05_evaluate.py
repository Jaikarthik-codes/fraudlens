import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load data
df = pd.read_csv('../data/processed/Fraud_Data_cleaned.csv')
X = df.drop(columns=['class'])
Y = df['class']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

# Train models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
ada_model = AdaBoostClassifier(n_estimators=100, random_state=42)

rf_model.fit(X_train, Y_train)
gb_model.fit(X_train, Y_train)
ada_model.fit(X_train, Y_train)

# Get predictions
rf_preds = rf_model.predict(X_test)
gb_preds = gb_model.predict(X_test)
ada_preds = ada_model.predict(X_test)

feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns)
print(feature_importance.sort_values(ascending=False))

# Summary comparison table
results = {
    'Model': ['Random Forest', 'Gradient Boosting', 'AdaBoost'],
    'Precision': [
        classification_report(Y_test, rf_preds, output_dict=True)['1']['precision'],
        classification_report(Y_test, gb_preds, output_dict=True)['1']['precision'],
        classification_report(Y_test, ada_preds, output_dict=True)['1']['precision']
    ],
    'Recall': [
        classification_report(Y_test, rf_preds, output_dict=True)['1']['recall'],
        classification_report(Y_test, gb_preds, output_dict=True)['1']['recall'],
        classification_report(Y_test, ada_preds, output_dict=True)['1']['recall']
    ],
    'F1': [
        classification_report(Y_test, rf_preds, output_dict=True)['1']['f1-score'],
        classification_report(Y_test, gb_preds, output_dict=True)['1']['f1-score'],
        classification_report(Y_test, ada_preds, output_dict=True)['1']['f1-score']
    ]
}

df_results = pd.DataFrame(results)
print("=== Model Comparison (Fraud Class Only) ===")
print(df_results.to_string(index=False))