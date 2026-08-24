from src.run_replication import (
    stage_split,
    stage_preprocess,
    stage_nested,
    stage_final,
    stage_bootstrap,
    stage_clinical,
    stage_shap,
)


if __name__ == "__main__":
    # Everything here uses TRAIN ONLY. The held-out TEST set is evaluated
    # separately by src/test_model.py after this workflow completes.
    stage_split()
    stage_preprocess()
    stage_nested()
    stage_final()
    stage_bootstrap()
    stage_clinical()
    stage_shap()
    print("\nTraining workflow complete. TEST has not been evaluated.")
    print("Run: python -m src.test_model")
