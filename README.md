## Machine Learning
#Diabetes Readmission — Patient-Level Nested CV + Individual Prediction Reliability Extension

This project starts from the supplied replication and adds a patient-level holdout, patient-level nested cross-validation, and a second-stage correctness calibrator for individual prediction reliability.

The paper uses stratified 5-fold outer CV with 3-fold inner CV for XGBoost tuning and keeps preprocessing inside the pipeline to avoid leakage.

## Key change requested

The raw dataset is first split once into:

- **80% TRAIN** — the only data used for preprocessing/model development, nested CV, hyperparameter selection, calibration, threshold selection, bootstrap analysis, clinical utility analysis, and training-set SHAP stability.
- **20% TEST** — completely untouched until `src/test_model.py` is executed after the training workflow finishes.

The paper itself reports nested CV and pipeline-based leakage prevention, while its final model is retrained on the full modeling dataset. This project deliberately extends that design by preserving an independent holdout for the final test evaluation. See the paper's Methods, especially Sections 3.2.1–3.2.4. 

## Recommended execution order

### 1. Run the complete training workflow

```bash
python -m experiments.run_all
```

This performs, in order:

1. Patient-level stratified train/test split (all encounters of a patient stay together).
2. Train-only preprocessing/export.
3. 5x3 patient-level nested CV using StratifiedGroupKFold in both outer and inner loops.
4. Baseline model comparison.
5. Final XGBoost training using the selected hyperparameters.
6. Training-only patient-level out-of-fold sigmoid outcome calibration.
7. Training-only best-threshold selection.
8. Bootstrap analysis using training OOF predictions.
9. DCA/CIC/PR analysis using training OOF predictions.
10. Training-set SHAP global/stability analysis.
11. Training-only second-stage correctness calibration and reliability analysis.

**The test set is not evaluated by this command.**

### 2. Evaluate the held-out test set

Only after the training workflow is complete:

```bash
python -m src.test_model
```

This produces:

- test ROC-AUC
- test PR-AUC
- test Brier score
- accuracy, precision, recall, F1
- sensitivity, specificity, NPV
- confusion matrix CSV + PNG
- classification report CSV + JSON
- per-record predicted probability
- per-record predicted class
- per-record confidence
- finalized threshold
- per-record SHAP values
- per-record top SHAP contributors
- global test SHAP importance
- test SHAP summary plot

## Confidence definition

For every held-out test record:

```text
if predicted_class == 1:
    confidence = P(readmission < 30 days)
else:
    confidence = 1 - P(readmission < 30 days)
```

Thus confidence is always the model probability assigned to the class actually predicted.

Example:

```text
probability = 0.82
threshold = 0.17
prediction = 1
confidence = 0.82
```

or:

```text
probability = 0.06
threshold = 0.17
prediction = 0
confidence = 0.94
```

## Best-threshold selection

The original paper reports a predefined 10% operating threshold. This extension instead selects a **best threshold using training-only out-of-fold calibrated probabilities**.

Default selection criterion:

```text
maximize minority-class F1
```

Search range:

```text
0.01 to 0.30 in steps of 0.01
```

The selected threshold is saved to:

```text
results/tables/final_threshold.csv
```

The test set never participates in threshold selection.

## SHAP and the finalized threshold

SHAP itself is a continuous feature-contribution analysis and does not require a classification threshold. For the final test evaluation, each record receives:

- calibrated probability
- final threshold
- predicted class
- confidence
- SHAP contribution for every transformed feature
- top 10 SHAP contributors with direction (`increases_risk` / `decreases_risk`)

The threshold is therefore used to interpret the SHAP explanation in the context of the final high-risk/low-risk decision, while the SHAP values remain the model's continuous feature contributions.

## Important leakage protections

1. The train/test split occurs **before feature engineering**.
2. The test set is not passed into nested CV.
3. XGBoost hyperparameter tuning is performed only inside the training nested CV.
4. The final XGBoost is fit only on TRAIN.
5. Sigmoid calibration is fitted using training-only out-of-fold predictions.
6. The operating threshold is selected using training-only out-of-fold calibrated predictions.
7. Test predictions are generated only after the model, calibrator, and threshold are finalized.
8. Test labels are used only for final evaluation metrics.

## Output locations

```text
data/
├── raw/
├── splits/
│   ├── train.csv
│   ├── test.csv
│   └── split_manifest.json
└── processed/
    └── train_final_dataset.csv

results/
├── figures/
└── tables/
```

## Main new/modified files

### New

```text
src/data_split.py
src/threshold_selection.py
src/test_model.py
```

### Modified

```text
config/config.json
src/calibration.py
src/explainability.py
src/run_replication.py
experiments/run_all.py
README.md
ASSUMPTIONS.md
```

## Reproducibility

The split uses:

```text
test_size = 0.20
random_state = 42
stratify = readmitted
```

The project uses an encounter-level stratified split. Patient-level grouping is not enforced, matching the limitation already documented for the original replication.

## Paper vs extension

The supplied paper reports a 5-fold outer / 3-fold inner nested CV design, sigmoid calibration, bootstrap robustness, clinical utility analysis, and SHAP explainability. The paper also reports a predefined 10% threshold. The independent test set, training-only threshold optimization, per-record confidence, and record-level test SHAP output are extensions requested for this project rather than claims about the paper's exact original implementation.
