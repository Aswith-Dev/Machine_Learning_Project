# Paper Methodology + Holdout Extension

The implementation follows the supplied paper for the core replication and adds an independent held-out test evaluation.

## Paper-derived core workflow

- Diabetes 130-US Hospitals dataset.
- Binary target: `<30` = 1; `>30` and `NO` = 0.
- Missing-value handling and feature engineering described in the paper.
- StandardScaler for numerical features.
- OneHotEncoder(handle_unknown="ignore") for categorical features.
- Cost-sensitive learning for class imbalance.
- Logistic Regression, Random Forest, XGBoost, LightGBM, and stacking comparison.
- XGBoost nested CV: 5 outer folds / 3 inner folds.
- XGBoost tuning by ROC-AUC.
- Sigmoid probability calibration.
- Bootstrap robustness analysis.
- Decision Curve Analysis and Clinical Impact Curve.
- SHAP TreeExplainer and SHAP stability.

## Holdout extension

Before any model-development operation, the raw encounter-level dataset is split into:

```text
80% TRAIN
20% TEST
```

Only TRAIN enters the paper-style nested CV and all training-side analysis.

TEST is locked until the final evaluation script is run.

## Final training sequence

```text
RAW DATA
   |
   +---- 80% TRAIN ------------------------------+
   |                                             |
   |     Nested CV (5 outer x 3 inner)           |
   |              |                              |
   |        Best XGB parameters                  |
   |              |                              |
   |      Training-only OOF predictions          |
   |              |                              |
   |     Sigmoid calibration                     |
   |              |                              |
   |   Best threshold by training OOF F1         |
   |              |                              |
   |      Final XGB fit on all TRAIN             |
   |              |                              |
   +--------------+------------------------------+
                  |
                  v
        FINAL MODEL + CALIBRATOR
                  |
                  v
          +-------+-------+
          |               |
          | 20% TEST      |
          |               |
          +-------+-------+
                  |
                  v
       Probability + threshold
       classification report
       confusion matrix
       confidence per record
       record-level SHAP
```

The paper itself reports the nested 5x3 CV design and the need to separate tuning from evaluation. The independent test set is an additional evaluation layer requested for this project.
