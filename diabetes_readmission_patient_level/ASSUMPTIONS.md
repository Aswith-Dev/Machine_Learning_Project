# Assumption Register — Train/Test Extension

## Original replication assumptions

1. XGBoost grid candidate values are not published in the supplied article.
   Implemented grid: `n_estimators=[100,200,300]`, `max_depth=[3,5,7]`, `learning_rate=[0.01,0.05,0.10]`.

2. The paper says the final feature-engineered dataset has 55 predictors, but the supplied main article does not expose the complete Supplementary Table S1. Only explicitly described feature transformations are implemented.

3. Diagnosis grouping uses standard ICD-9 ranges matching the category labels stated in the paper.

4. The paper names aggregate concepts in the SHAP discussion without publishing all formulas. The formulas implemented in `src/features.py` remain documented replication assumptions.

5. The paper does not report the authors' random seed. Seed 42 is used for deterministic replication.

## New train/test extension assumptions

6. A single **stratified 80/20 encounter-level split** is created before preprocessing. The test set is never passed into nested CV, calibration, threshold selection, bootstrap analysis, clinical utility analysis, or model fitting.

7. Patient-level grouping is not enforced. The paper's replication notes already identify this as a limitation. Therefore, the independent holdout is an encounter-level holdout rather than a patient-level holdout.

8. Final XGBoost hyperparameters are selected by the mode across the five outer-fold best parameter sets, as in the original project, because the paper does not specify how the five configurations should be aggregated.

9. Final sigmoid calibration is fitted using **training-only out-of-fold XGBoost probabilities**, then applied to the XGBoost model retrained on all training records. This avoids fitting a calibrator on the same in-sample predictions used to train the final XGBoost.

10. The final operating threshold is selected from **training-only out-of-fold calibrated probabilities**.

11. The default threshold objective is maximum minority-class F1 over 0.01–0.30 in 0.01 increments. The paper's 10% threshold is retained only as a reference/tie-break point, not as the final threshold.

12. Per-record confidence is defined as the probability assigned to the predicted class:
    - predicted positive: `confidence = P(readmission <30)`
    - predicted negative: `confidence = 1 - P(readmission <30)`

13. SHAP is calculated from the final fitted XGBoost model. The finalized threshold determines the record's high-risk/low-risk classification, but SHAP values themselves remain continuous model contributions.

14. Test labels are used only after model, calibration, and threshold finalization. They are not used for model selection.
