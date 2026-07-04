import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

df = pd.read_csv("../data/processed/creditcard_cleaned.csv")

x = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, Y_train, Y_test = train_test_split(x, y, train_size=0.8, random_state=42, stratify=y)

smote = SMOTE(random_state=42)
X_train_sm, Y_train_sm = smote.fit_resample(X_train, Y_train)
print(Y_train_sm.value_counts())

# Step 4: Train Random Forest
rf_model = RandomForestClassifier(n_estimators = 100, random_state=42)
rf_model.fit(X_train_sm, Y_train_sm)
rf_peds = rf_model.predict(X_test)
print("=== Random Forest ===")
print(confusion_matrix(Y_test, rf_peds))
print(classification_report(Y_test, rf_peds))

# Step 5: Train Gradient Boosting
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train_sm, Y_train_sm)
gb_preds = gb_model.predict(X_test)
print("=== Gradient Boosting ===")
print(confusion_matrix(Y_test, gb_preds))
print(classification_report(Y_test, gb_preds))

# Step 6: Train AdaBoost
ada_model = AdaBoostClassifier(n_estimators=100, random_state=42)
ada_model.fit(X_train_sm, Y_train_sm)
ada_preds = ada_model.predict(X_test)
print("=== AdaBoost ===")
print(confusion_matrix(Y_test, ada_preds))
print(classification_report(Y_test, ada_preds))