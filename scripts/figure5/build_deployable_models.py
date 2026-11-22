from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT / "external_data" / "source_tree"
RUN = ROOT / "outputs" / "Food3D_Figure3_Figure5_Update_20261015"
FROZEN = RUN / "05_FIGURE5_FROZEN_CURRENT_DOMAIN_20261015"
REBUILT = ROOT / "outputs" / "Figure5_FROZEN_RESULT_REBUILT_20261015"
MASTER = RUN / "00_FROZEN_SOURCE" / "Food3D_590_merged_dataset.xlsx"
CROSS_SUMMARY = RUN / "04_FIGURE5_ALL590_REANALYSIS" / "Figure5_ALL590_REANALYSIS_SUMMARY.json"
OUT = ROOT / "outputs" / "Figure5_DEPLOYABLE_MODELS_20261015"
MODELS = OUT / "Models"
EXAMPLES = OUT / "Examples"
REPORTS = OUT / "Reports"
for folder in (MODELS, EXAMPLES, REPORTS):
    folder.mkdir(parents=True, exist_ok=True)

PROCESS = ["Print_temp_C", "Model_height_mm", "Print_speed_mm_s", "Nozzle_mm", "Nozzle_height_mm"]
RHEO = ["Gp_mean_Pa", "Gpp_mean_Pa", "eta_10_Pa_s", "eta_100_Pa_s", "Selected_tau0", "Selected_K", "Selected_n"]
FEATURES = PROCESS + [f"log10_{x}" for x in RHEO[:-1]] + ["Selected_n"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def feature_matrix(d: pd.DataFrame) -> pd.DataFrame:
    x = d[PROCESS].copy().astype(float)
    for col in RHEO[:-1]:
        x[f"log10_{col}"] = np.log10(pd.to_numeric(d[col], errors="raise").clip(lower=0) + 1e-8)
    x["Selected_n"] = pd.to_numeric(d["Selected_n"], errors="raise")
    return x[FEATURES]


def recommend(bundle: dict, rheology: dict, top_k: int = 10) -> pd.DataFrame:
    grid = bundle["candidate_process_grid"].copy()
    for key in RHEO:
        grid[key] = float(rheology[key])
    x = feature_matrix(grid)
    quality = np.clip(bundle["quality_regressor"].predict(x), 0, 100)
    clf = bundle["feasibility_classifier"]
    p_ext = clf.predict_proba(x)[:, list(clf.classes_).index(1)]
    grid["P_extrudable"] = p_ext
    grid["Predicted_integrated_outcome"] = quality
    grid["Feasibility_pass"] = grid["P_extrudable"] >= 0.50
    grid = grid.sort_values(
        ["Feasibility_pass", "Predicted_integrated_outcome", "P_extrudable"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    grid["Recommendation_rank"] = np.arange(1, len(grid) + 1)
    return grid.head(top_k)


def main():
    quality = joblib.load(FROZEN / "Figure5_CURRENT_DOMAIN_rheology_process_model.joblib")
    feasibility = joblib.load(REBUILT / "SourceData" / "Figure5e_deployment_feasibility_classifier.joblib")
    summary = json.loads((FROZEN / "Figure5_CURRENT_DOMAIN_SUMMARY.json").read_text(encoding="utf-8"))
    cross = json.loads(CROSS_SUMMARY.read_text(encoding="utf-8"))
    d = pd.read_excel(MASTER, sheet_name="MERGED_590")

    levels = {col: sorted(float(v) for v in pd.unique(d[col])) for col in PROCESS}
    candidate_grid = pd.MultiIndex.from_product([levels[col] for col in PROCESS], names=PROCESS).to_frame(index=False)
    assert len(candidate_grid) == 243
    support = pd.DataFrame({
        "rheology_feature": RHEO,
        "minimum": [float(d[c].min()) for c in RHEO],
        "maximum": [float(d[c].max()) for c in RHEO],
        "median": [float(d[c].median()) for c in RHEO],
    })
    state_library = d[["Rheology_ID", *RHEO]].drop_duplicates().reset_index(drop=True)

    forward = {
        "bundle_name": "Figure5 rheology-plus-process to feasibility and integrated print outcome",
        "version": "2026-10-15",
        "quality_regressor": quality,
        "feasibility_classifier": feasibility,
        "input_rheology": RHEO,
        "input_process": PROCESS,
        "ordered_model_features": FEATURES,
        "quality_target": "Integrated print outcome: D1-D5-derived score for extrudable attempts and protocol-defined zero for observed non-extrudable attempts",
        "quality_output": "Predicted_integrated_outcome, scale 0-100",
        "feasibility_output": "P_extrudable, model-estimated probability",
        "quality_validation": summary["selected_models"]["Rheology + process"],
        "feasibility_development_validation": cross["extrusion_failure_classifier_selected"]["Rheology + process"],
        "material_identity_used": False,
        "quality_train_n": summary["train_n"],
        "quality_test_n": summary["test_n"],
        "candidate_group_overlap": summary["candidate_group_overlap"],
        "warning": "Feasibility probabilities are screening outputs and are not prospective calibration evidence.",
    }
    forward_path = MODELS / "Figure5_FORWARD_rheology_process_to_accuracy.joblib"
    joblib.dump(forward, forward_path)

    recommender = {
        **forward,
        "bundle_name": "Figure5 rheology to ranked process candidates",
        "candidate_process_grid": candidate_grid,
        "candidate_count": len(candidate_grid),
        "feasibility_threshold": 0.50,
        "ranking_rule": "Filter P_extrudable >= 0.50, then rank Predicted_integrated_outcome descending; probability is only a tie-breaker",
        "rheology_training_support": support,
        "measured_rheology_state_library": state_library,
        "recommendation_scope": "Enumeration over observed discrete process levels only",
        "not_a_direct_inverse_regression": True,
        "reason": "Multiple process settings can be compatible with one rheological state; ranking the forward-model landscape preserves this multiplicity.",
    }
    recommender_path = MODELS / "Figure5_RECOMMENDER_rheology_to_process_topk.joblib"
    joblib.dump(recommender, recommender_path)

    candidate_grid.to_csv(OUT / "CANDIDATE_PROCESS_GRID_243.csv", index=False, encoding="utf-8-sig")
    support.to_csv(OUT / "RHEOLOGY_TRAINING_SUPPORT.csv", index=False, encoding="utf-8-sig")
    state_library.to_csv(OUT / "MEASURED_RHEOLOGY_STATE_LIBRARY.csv", index=False, encoding="utf-8-sig")

    example_id = "RHEO_M02_14_80_25"
    state = state_library.loc[state_library.Rheology_ID.eq(example_id)]
    if state.empty:
        state = state_library.iloc[[0]]
        example_id = str(state.iloc[0].Rheology_ID)
    rheology = state.iloc[0][RHEO].to_dict()
    top = recommend(recommender, rheology, top_k=10)
    top.insert(0, "Rheology_ID", example_id)
    top.to_csv(EXAMPLES / "example_rheology_to_top10_processes.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([rheology]).to_csv(EXAMPLES / "example_rheology_input.csv", index=False, encoding="utf-8-sig")

    inference_source = ROOT / "outputs" / "Figure5_DEPLOYABLE_MODELS_20261015" / "predict_process_from_rheology.py"
    summary_out = {
        "forward_model": str(forward_path.relative_to(OUT)),
        "recommender_model": str(recommender_path.relative_to(OUT)),
        "candidate_count": len(candidate_grid),
        "quality_validation": {
            "grouped_CV_R2": summary["selected_models"]["Rheology + process"]["train_group_cv_R2"],
            "independent_candidate_test_R2": summary["selected_models"]["Rheology + process"]["test_R2"],
            "test_RMSE": summary["selected_models"]["Rheology + process"]["test_RMSE"],
            "test_MAE": summary["selected_models"]["Rheology + process"]["test_MAE"],
        },
        "example_rheology_id": example_id,
        "example_top1": top.iloc[0].to_dict(),
        "source_hashes": {
            "master_590": sha256(MASTER),
            "frozen_quality_model": sha256(FROZEN / "Figure5_CURRENT_DOMAIN_rheology_process_model.joblib"),
            "feasibility_model": sha256(REBUILT / "SourceData" / "Figure5e_deployment_feasibility_classifier.joblib"),
        },
    }
    (REPORTS / "MODEL_SUMMARY.json").write_text(json.dumps(summary_out, ensure_ascii=False, indent=2), encoding="utf-8")
    readme = f"""# Figure 5 deployable models\n\nThis is the two-model package represented by Figure 5.\n\n## Model 1: rheology + process to print outcome\n\n`Models/Figure5_FORWARD_rheology_process_to_accuracy.joblib` accepts seven rheological descriptors and five candidate process variables. It outputs model-estimated extrusion probability and the integrated print outcome. The frozen quality result is grouped-CV R2 = {summary_out['quality_validation']['grouped_CV_R2']:.3f} and independent candidate-test R2 = {summary_out['quality_validation']['independent_candidate_test_R2']:.3f}.\n\n## Model 2: rheology to process recommendation\n\n`Models/Figure5_RECOMMENDER_rheology_to_process_topk.joblib` evaluates all 243 observed discrete process combinations, filters candidates at `P(extrudable) >= 0.50`, and ranks screen-passing candidates by predicted integrated outcome. Probability and OQI are not multiplied.\n\n## Inputs\n\n- Gp_mean_Pa\n- Gpp_mean_Pa\n- eta_10_Pa_s\n- eta_100_Pa_s\n- Selected_tau0\n- Selected_K\n- Selected_n\n\n## Outputs\n\n- Print temperature\n- Build height\n- Print speed\n- Nozzle diameter\n- Nozzle gap\n- P(extrudable)\n- Predicted integrated outcome\n- Feasibility flag and recommendation rank\n\nUse `predict_process_from_rheology.py` for direct inference. Recommendations are limited to the discrete process levels observed during model development and require experimental confirmation.\n"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")

    # The inference script is created separately and must be present before the final manifest.
    if not inference_source.exists():
        raise FileNotFoundError(inference_source)
    files = [p for p in OUT.rglob("*") if p.is_file() and p.name != "SHA256_MANIFEST.csv"]
    pd.DataFrame([{"file": str(p.relative_to(OUT)), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(files)]).to_csv(OUT / "SHA256_MANIFEST.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(summary_out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
