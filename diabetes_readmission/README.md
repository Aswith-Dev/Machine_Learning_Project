# Exact-Methodology Replication: 30-Day Hospital Readmission in Patients with Diabetes

Replication project for:

**Salim, S.S.; Ibrahim, A.A. (2026). A Machine Learning Approach for Predicting 30-Day Hospital Readmission in Patients with Diabetes. Healthcare, 14(9), 1185. DOI: 10.3390/healthcare14091185.**

## Purpose

This repository implements the methodology described in the supplied published paper and uses the supplied Diabetes 130-US Hospitals dataset.

The objective is **replication, not improvement**. No SMOTE/ADASYN/ENN, alternative model, feature-selection optimizer, patient-level grouping, or novel method is added.

## Paper methodology reproduced

1. Diabetes 130-US Hospitals dataset: 101,766 encounters.
2. Binary target:
   - `<30` -> 1
   - `>30` and `NO` -> 0
3. Remove `encounter_id`, `patient_nbr`, `weight`, and `payer_code`.
4. Treat missing categorical values as an explicit `Unknown` category.
5. Convert age intervals to numerical midpoints.
6. Group diagnosis codes into clinically meaningful ICD-9 categories and create diagnosis-presence flags.
7. Consolidate admission type, discharge disposition, and admission source into grouped categories.
8. Keep the ten most frequent medical specialties and group the remainder as `Other`.
9. Standardize numerical variables with `StandardScaler`.
10. One-hot encode categorical variables with unknown categories ignored.
11. Handle imbalance by cost-sensitive learning:
    - XGBoost: `scale_pos_weight = negatives / positives`
    - Logistic Regression / Random Forest / LightGBM: `class_weight="balanced"`
12. Nested stratified CV:
    - outer: 5 folds
    - inner: 3 folds
    - GridSearchCV scoring: ROC-AUC
    - XGBoost tuning only for `n_estimators`, `max_depth`, `learning_rate`
13. Compare Logistic Regression, Random Forest, XGBoost, LightGBM, and stacking.
14. Retrain the selected XGBoost on the full dataset.
15. Platt/sigmoid calibration.
16. Brier score and calibrated AUC.
17. 1,000 bootstrap resamples for AUC, Brier score, and risk-ratio confidence intervals.
18. Decision Curve Analysis from 1% to 30%.
19. Clinical Impact Curve from 1% to 30%.
20. Threshold metrics at 5%, 10%, 15%, 20%, 25%, 30%.
21. Tertile risk stratification.
22. SHAP TreeExplainer global importance.
23. 100-bootstrap SHAP top-10 stability analysis.
24. Precision-recall curve.

## Important reproducibility note

The paper does **not** publish the exact numeric GridSearchCV candidate values for the three XGBoost hyperparameters. Therefore the project uses the smallest conventional grid consistent with the reported methodology:

```text
n_estimators = [100, 200, 300]
max_depth    = [3, 5, 7]
learning_rate = [0.01, 0.05, 0.10]
```

This is explicitly an **assumption**, not a claimed fact from the paper.

The paper also references Supplementary Table S1 for detailed feature engineering, but the supplied PDF does not contain S1. The implementation therefore follows every feature-engineering operation explicitly stated in the paper and documents the few formula-level assumptions in `config/config.json` and `src/features.py`.

The paper states that the final modeling dataset contains 55 predictors, but it does not provide enough information in the supplied main article to unambiguously reconstruct every one of those 55 fields. The implementation preserves all explicitly described transformations and records the resulting feature count rather than silently fabricating additional predictors.

## Expected paper-reported reference results

These are **reference values reported by the paper**, not hard-coded project outputs:

| Model / metric | Paper value |
|---|---:|
| Logistic Regression ROC-AUC | 0.657 |
| Random Forest ROC-AUC | 0.650 |
| XGBoost ROC-AUC | 0.664 |
| LightGBM ROC-AUC | 0.660 |
| Stacking ROC-AUC | 0.665 |
| Nested CV XGBoost ROC-AUC | 0.664 |
| Full-data post-calibration AUC | 0.688 |
| Brier score | 0.094 |
| PR-AUC | 0.215 |
| F1 | 0.27 |
| Sensitivity at 10% | 0.723 |
| Specificity at 10% | 0.530 |
| PPV at 10% | 0.162 |
| NPV at 10% | 0.938 |

The paper itself notes that its DCA used full-dataset predictions and that patient-level splits were not enforced.

## Project structure

```text
diabetes_readmission_replication/
├── README.md
├── config/
│   └── config.json
├── data/
│   ├── raw/
│   │   ├── diabetic_data.csv
│   │   └── IDS_mapping.csv
│   └── processed/
├── src/
│   ├── __init__.py
│   ├── features.py
│   ├── pipeline.py
│   ├── models.py
│   ├── nested_cv.py
│   ├── calibration.py
│   ├── bootstrap_analysis.py
│   ├── clinical_utility.py
│   ├── explainability.py
│   └── run_replication.py
├── experiments/
│   └── run_all.py
├── evaluation/
├── results/
│   ├── figures/
│   └── tables/
├── logs/
├── tests/
│   └── test_features.py
└── references/
    └── paper.pdf
```

## Environment

The paper reports Python 3.12.7, XGBoost 3.1.2, LightGBM 4.6.0, scikit-learn 1.8.0, and SHAP 0.50.0.

A compatible environment can be created with:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install:

```bash
pip install -r requirements.txt
```

## Run order

### 1. Verify the supplied data

```bash
python -m pytest -q
```

### 2. Build the final modeling dataset

```bash
python -m src.run_replication --stage preprocess
```

This creates:

```text
data/processed/final_dataset.csv
```

### 3. Run nested cross-validation and model comparison

```bash
python -m src.run_replication --stage nested_cv
```

This performs the paper's 5x3 nested evaluation.

### 4. Train the final XGBoost and calibration analysis

```bash
python -m src.run_replication --stage final_model
```

### 5. Bootstrap robustness

```bash
python -m src.run_replication --stage bootstrap
```

### 6. Clinical utility

```bash
python -m src.run_replication --stage clinical
```

### 7. SHAP explainability and stability

```bash
python -m src.run_replication --stage shap
```

### 8. Run everything

```bash
python -m experiments.run_all
```

## Computational cost

The complete replication is computationally heavy because the paper requires:

- 5 outer folds
- 3 inner folds
- 27 XGBoost GridSearch combinations per outer fold
- multiple baseline models
- 1,000 bootstrap resamples
- 100 SHAP stability bootstrap fits

The implementation uses all available CPU cores where the paper's methodology permits it, but it does not alter the experimental design.

## Output files

Results are written to:

```text
results/tables/
results/figures/
logs/
data/processed/
```

Important outputs include:

- nested CV model comparison
- fold-level XGBoost metrics
- best parameters by outer fold
- calibrated predictions
- bootstrap confidence intervals
- threshold metrics
- risk-stratification table
- DCA curve
- clinical-impact curve
- PR curve
- SHAP summary plot
- SHAP stability table

## No fabricated results

The project never inserts the paper's reported metrics into the generated results. The reported values are stored only as reference values. Running the project produces actual values from the supplied data.

Differences from the paper may occur because:
- the exact XGBoost GridSearch ranges are not published;
- some feature formulas are not specified in the main paper;
- software versions or numerical libraries may differ;
- the authors' exact random seeds are not reported;
- the paper's DCA is described as using full-dataset predictions;
- patient-level grouping was not enforced.

## Source

The supplied paper is the primary source of truth. The official article and supplementary-material listing are available from MDPI.

DOI: 10.3390/healthcare14091185
