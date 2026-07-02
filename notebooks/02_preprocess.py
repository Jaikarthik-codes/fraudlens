import pandas as pd

# Load raw data
df = pd.read_csv('../data/raw/Fraud_Data.csv')

# Step 1: Convert timestamp strings to actual datetime objects
df["signup_time"] = pd.to_datetime(df["signup_time"])
df["purchase_time"] = pd.to_datetime(df["purchase_time"])

# New feature 1: hour of purchase
df['purchase_hour'] = df['purchase_time'].dt.hour

# New feature 2: high value purchase flag
df['high_value'] = (df['purchase_value'] > df['purchase_value'].quantile(0.75)).astype(int)

# Step 2: Engineer new feature - seconds between signup and purchase
df["time_to_purchase"] = (df["purchase_time"] - df["signup_time"]).dt.total_seconds()

# Step 3: Encode categorical columns (convert text categories to numbers)
df = pd.get_dummies(df, columns=["source", "browser", "sex"])

# Step 4: Drop columns we no longer need
df = df.drop(columns=["user_id", "device_id", "ip_address", "signup_time", "purchase_time"])

# Step 5: Check the result
print(df.head())
print(df.shape)
print(df.dtypes)

# Step 6: Save cleaned data to processed folder
df.to_csv("../data/processed/Fraud_Data_Cleaned.csv", index = False)
print("Saved Successfully")