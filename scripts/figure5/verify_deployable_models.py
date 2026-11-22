from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS = REPO_ROOT / "models"
VALIDATION = REPO_ROOT / "validation" / "figure5" / "candidate_test_121.csv"
EXAMPLE = REPO_ROOT / "examples" / "example_rheology_input.csv"
EXPECTED = REPO_ROOT / "examples" / "expected_top10_processes.csv"
REPORT = REPO_ROOT / "reports" / "figure5_inference_validation.json"


def import_inference():
    path = Path(__file__).with_name("predict_process_from_rheology.py")
    spec = importlib.util.spec_from_file_location("figure5_inference", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    inference = import_inference()
    forward = joblib.load(MODELS / "Figure5_FORWARD_rheology_process_to_accuracy.joblib")
    recommender = joblib.load(MODELS / "Figure5_RECOMMENDER_rheology_to_process_topk.joblib")

    test = pd.read_csv(VALIDATION, encoding="utf-8-sig")
    predicted = np.clip(forward["quality_regressor"].predict(inference.feature_matrix(test)), 0, 100)
    frozen = test["pred_rheology_process"].to_numpy(float)
    target = test["OQI_current_domain_strict"].to_numpy(float)
    model = forward["quality_regressor"].named_steps["model"]

    rheology = pd.read_csv(EXAMPLE, encoding="utf-8-sig").iloc[0].to_dict()
    actual_example = inference.recommend(recommender, rheology, top_k=10)
    expected_example = pd.read_csv(EXPECTED, encoding="utf-8-sig")
    compare_cols = [*inference.PROCESS, "P_extrudable", "Predicted_integrated_outcome", "Recommendation_rank"]
    example_delta = float(
        np.max(np.abs(actual_example[compare_cols].to_numpy(float) - expected_example[compare_cols].to_numpy(float)))
    )

    grid = recommender["candidate_process_grid"]
    threshold = float(recommender.get("feasibility_threshold", 0.50))
    ranking_rule = str(recommender.get("ranking_rule", ""))
    report = {
        "verdict": "PASS",
        "validation_rows": int(len(test)),
        "max_abs_delta_vs_packaged_frozen_predictions": float(np.max(np.abs(predicted - frozen))),
        "fixed_candidate_test_R2": float(r2_score(target, predicted)),
        "fixed_candidate_test_RMSE": float(mean_squared_error(target, predicted) ** 0.5),
        "fixed_candidate_test_MAE": float(mean_absolute_error(target, predicted)),
        "quality_estimator": type(model).__name__,
        "quality_n_estimators": int(model.n_estimators),
        "quality_random_state": int(model.random_state),
        "candidate_grid_rows": int(len(grid)),
        "candidate_grid_unique_rows": int(len(grid.drop_duplicates())),
        "feasibility_threshold": threshold,
        "ranking_rule_metadata": ranking_rule,
        "example_max_abs_delta": example_delta,
        "scientific_rule_check": "filter P(extrudable) >= 0.50, then rank by predicted OQI; no multiplication",
    }
    failures = []
    if report["max_abs_delta_vs_packaged_frozen_predictions"] > 1e-10:
        failures.append("frozen predictions changed")
    if model.n_estimators != 400 or model.random_state != 20260902:
        failures.append("quality estimator contract changed")
    if len(grid) != 243 or len(grid.drop_duplicates()) != 243:
        failures.append("candidate grid is not 243 unique rows")
    if threshold != 0.50:
        failures.append("feasibility threshold changed")
    if example_delta > 1e-10:
        failures.append("example inference changed")
    if failures:
        report["verdict"] = "FAIL"
        report["failures"] = failures

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["verdict"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
