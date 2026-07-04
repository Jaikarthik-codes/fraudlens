## Step 1 - Data Exploration (ULB Dataset)

Dataset: creditcard.csv
Shape: 284,807 rows x 31 columns: Time, V1-V28 (PCA transformed), Amount, Class
No missing values in any column.

Class distribution:
- Not fraud (0): 284,315 = 99.83%
- Fraud (1): 492 = 0.17%

Far more imbalanced than Iteration 1 (9.4% fraud vs 0.17% fraud).
Dumb model accuracy would be 99.83% — even more misleading than before.

## Step 2 - Preprocessing (ULB Dataset)

Columns scaled: Amount and Time
Why: V1-V28 are already scaled from PCA transformation.
Amount and Time are raw values — different scales would bias the model.
Used StandardScaler: transforms values to mean=0, standard deviation=1.

Split: 80% training (227,845 rows) / 20% testing (56,962 rows)
stratify=Y preserves 0.17% fraud ratio in both sets.

SMOTE applied to training data only:
Before: Fraud = 394, Not fraud = 227,451
After: Fraud = 227,451, Not fraud = 227,451 (perfectly balanced)

Why SMOTE after split and not before:
Test data must reflect reality (0.17% fraud).
If we SMOTEd before splitting, test data would contain fake synthetic 
fraud cases — evaluation numbers would not reflect real performance.

Why SMOTE instead of just copying fraud cases:
Copying the same 394 cases adds no new information.
SMOTE creates realistic synthetic cases between existing fraud cases,
giving the model genuinely new patterns to learn from.

## Step 3 - Model Training (ULB Dataset)

Trained on: X_train_sm, Y_train_sm (SMOTE balanced data)
Tested on: X_test, Y_test (original imbalanced data - reality)

Results:
Model              Precision  Recall   F1    False Alarms
Random Forest        0.82      0.82    0.82      17
Gradient Boosting    0.11      0.90    0.19     746
AdaBoost             0.05      0.91    0.10    1598

Winner: Random Forest — best balance across all three metrics.

Why GB and AdaBoost have poor precision:
SMOTE created 227,451 synthetic fraud cases. Sequential models
(GB and AdaBoost) amplified sensitivity to these synthetic patterns
aggressively. Each tree built on previous mistakes, compounding
the aggression. Result: high recall but catastrophic precision.

Why Random Forest handled SMOTE better:
100 trees built independently. Majority voting prevents any single
aggressive tree from dominating. Averaging effect naturally balances
precision and recall.

How the three models differ internally:
- Random Forest: 100 trees built in parallel, majority vote
- Gradient Boosting: 100 trees sequential, each fixes previous mistakes
- AdaBoost: 100 trees sequential, increases weight of misclassified cases

Compared to Iteration 1:
Random Forest recall: 0.53 → 0.82 (dataset switch confirmed correct)
Goal was recall above 0.80 — achieved with Random Forest.

## How the three models differ internally

Random Forest:
- Builds 100 trees independently and in parallel
- All 100 trees vote equally on final prediction
- Majority wins — bad trees get outvoted
- Safe with SMOTE because aggression gets averaged out

Gradient Boosting:
- Builds 100 trees sequentially
- Each tree focuses on mistakes the previous tree made
- Powerful but dangerous with SMOTE — early trees overcorrect 
  on synthetic fraud patterns, later trees amplify that overcorrection

AdaBoost:
- Also, sequential like Gradient Boosting
- Instead of focusing on errors, increases the weight of 
  misclassified cases so next tree pays more attention to them
- Same SMOTE problem as Gradient Boosting, even more extreme

Key insight:
Same SMOTE data fed to all three models.
Different internal architecture produced completely different 
precision results. Architecture matters as much as data.

## Step 4 - Ensemble Logic (ULB Dataset)

Results:
Model               Precision   Recall    F1    False Alarms
Random Forest         0.82       0.82     0.82      17
Gradient Boosting     0.11       0.90     0.19     746
AdaBoost              0.05       0.91     0.10    1598
Hard Voting           0.14       0.90     0.24     545
Soft Voting           0.44       0.88     0.59     110

Winner: Random Forest individually — ensemble made things worse here.

Why ensemble didn't improve over Random Forest:
Ensemble helps when models make different kinds of errors.
GB and AdaBoost both make the same error — too aggressive with 
fraud flagging due to SMOTE amplification in sequential trees.
Combining them just spreads that aggression into the ensemble.

Why soft voting beat hard voting:
Hard voting counts yes/no — two aggressive models outvote RF.
Soft voting averages confidence levels — RF's lower confidence 
pulls the average down, reducing false alarms from 545 to 110.