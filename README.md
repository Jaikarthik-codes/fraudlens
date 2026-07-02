# Fraud Detection System

A multi-model fraud detection system built from scratch for 
e-commerce transactions. Built with the explicit goal of being 
able to defend every decision in a technical interview.

## Iteration 1 — Fraud_Data.csv (Documented Dead End)

Built complete pipeline on e-commerce fraud dataset with 
interpretable features (signup time, purchase time, purchase value, 
browser, age).

Engineered features: time_to_purchase, purchase_hour, high_value.

Finding: Feature importance analysis showed time_to_purchase 
dominated at 68.2% of predictive signal even after full feature 
engineering. Recall ceiling at 0.527 confirmed. Dataset does not 
contain enough signal for meaningful recall improvement.

Decision: Switched to ULB dataset based on evidence from feature 
importance analysis, not assumption.

## Iteration 2 — ULB Credit Card Fraud Dataset (In Progress)

284,807 transactions, V1-V28 PCA features + Amount + Time.
Target: recall above 0.80.
Models: Random Forest, Gradient Boosting, AdaBoost + Ensemble.

## Stack
- Python, pandas, NumPy, scikit-learn
- Flask (backend)
- HTML/CSS (frontend)