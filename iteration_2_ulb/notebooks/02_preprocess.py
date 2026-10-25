import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Load the cleaned dataset
df = pd.read_csv("../data/raw/creditcard.csv")

# Scale Amount and Time since V1-V28 are already scaled from PCA
# One scaler fitted on both columns, so the API can reuse it as-is
scaler = StandardScaler()
scaled = scaler.fit_transform(df[["Amount", "Time"]])
df = df.drop(["Amount", "Time"], axis=1)

df["scaled_Amount"] = scaled[:, 0]
df["scaled_Time"] = scaled[:, 1]

# Save the scaler so the Flask API scales new transactions the same way
joblib.dump(scaler, "../models/scaler.pkl")

# Split into train and test sets before applying SMOTE
x = df.drop(columns = ["Class"])
y = df["Class"]

X_train, X_test ,Y_train, Y_test = train_test_split(x, y, random_state=42, stratify=y, test_size=0.2)
print(X_train.shape)
print(X_test.shape)

# Apply SMOTE to training data only to handle class imbalance
smote = SMOTE(random_state=42)
X_train_sm, Y_train_sm = smote.fit_resample(X_train, Y_train)
print(Y_train_sm.value_counts())

# Save cleaned data
df.to_csv("../data/processed/creditcard_cleaned.csv", index = False)
print("Saved Successfully")