"""
07 - Make the demo sample used by the Streamlit dashboard.

Run once from the notebooks/ folder (needs ../data/raw/creditcard.csv):
    python 07_make_demo_sample.py

It writes ../data/demo/test_sample.csv (about 300 KB, small enough for GitHub):
every fraud from the TEST set (98 rows) plus random legitimate TEST
transactions, 500 rows in total, shuffled. Commit that file so the
dashboard can show real transactions the models have never seen.

Note: real traffic is 0.17% fraud; this sample is about 20% fraud so the
demo has enough frauds to look at. Precision measured on the sample is
therefore higher than in real life. The real test-set numbers are in NOTES.md.
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent

parser = argparse.ArgumentParser()
parser.add_argument("--data", default=str(HERE.parent / "data" / "raw" / "creditcard.csv"))
parser.add_argument("--out", default=str(HERE.parent / "data" / "demo" / "test_sample.csv"))
parser.add_argument("--rows", type=int, default=500, help="total rows in the sample")
args = parser.parse_args()

df = pd.read_csv(args.data)

# Keep the row number from creditcard.csv so any transaction can be traced back
df.insert(0, "transaction_id", df.index)

# Same split as notebooks 02-05 (random_state=42, stratified, 20% test),
# so every row in the sample comes from the test set the models never saw
_, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["Class"])

frauds = test[test["Class"] == 1]
legit = test[test["Class"] == 0].sample(n=args.rows - len(frauds), random_state=42)
sample = pd.concat([frauds, legit]).sample(frac=1, random_state=42)  # shuffle

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
sample.to_csv(out, index=False)
print(f"Saved {len(sample)} test transactions ({len(frauds)} fraud, {len(legit)} legitimate) to {out}")
