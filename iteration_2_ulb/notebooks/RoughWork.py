import pandas as pd

df = pd.read_csv('../data/processed/creditcard_cleaned.csv')

fraud_samples = df[df['Class'] == 1].head(3)
legit_samples = df[df['Class'] == 0].head(3)

samples = pd.concat([fraud_samples, legit_samples])
print(samples.to_dict('records'))