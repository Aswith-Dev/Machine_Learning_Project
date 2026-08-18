from src.run_replication import (
    stage_preprocess, stage_nested, stage_final, stage_bootstrap,
    stage_clinical, stage_shap
)

if __name__ == "__main__":
    stage_preprocess()
    stage_nested()
    stage_final()
    stage_bootstrap()
    stage_clinical()
    stage_shap()
