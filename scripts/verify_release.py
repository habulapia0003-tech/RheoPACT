from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT = REPO_ROOT / "reports" / "release_validation.json"


def syntax_check() -> list[str]:
    checked = []
    for path in sorted((REPO_ROOT / "scripts").rglob("*.py")):
        compile(path.read_bytes(), str(path), "exec")
        checked.append(str(path.relative_to(REPO_ROOT)))
    return checked


def figure3_check() -> dict:
    bundle = joblib.load(REPO_ROOT / "models" / "Figure3_HistGradientBoosting_D1D5_to_sensory.joblib")
    data = pd.read_csv(REPO_ROOT / "validation" / "figure3" / "fixed_test_99.csv", encoding="utf-8-sig")
    features = [
        "D1_global_geometry",
        "D2_structural_completeness",
        "D3_surface_deviation",
        "D4_polarity_tolerance",
        "D5_spatial_heterogeneity",
    ]
    predicted = bundle.predict(data[features]) if hasattr(bundle, "predict") else bundle["model"].predict(data[features])
    frozen = data["prediction"].to_numpy(float)
    target = data["Sensory_mean_new"].to_numpy(float)
    estimator = bundle.named_steps["model"] if hasattr(bundle, "named_steps") else bundle["model"].named_steps["model"]
    result = {
        "rows": int(len(data)),
        "max_abs_delta_vs_packaged_frozen_predictions": float(np.max(np.abs(predicted - frozen))),
        "fixed_test_R2": float(r2_score(target, predicted)),
        "fixed_test_RMSE": float(mean_squared_error(target, predicted) ** 0.5),
        "fixed_test_MAE": float(mean_absolute_error(target, predicted)),
        "estimator": type(estimator).__name__,
        "random_state": int(estimator.random_state),
    }
    if len(data) != 99 or result["max_abs_delta_vs_packaged_frozen_predictions"] > 1e-10:
        raise RuntimeError("Figure 3 frozen-prediction validation failed")
    return result


def bundled_research_data_check() -> dict:
    frozen = REPO_ROOT / "data" / "frozen_tables"
    state = REPO_ROOT / "data" / "state_space"

    merged_path = frozen / "Food3D_590_merged_dataset.xlsx"
    sensory_path = frozen / "Food3D_590_visual_ratings_50_assessors.xlsx"
    figure6_path = REPO_ROOT / "data/figure6/Figure6_12_print_inputs.csv"
    figure6_d1d5_path = REPO_ROOT / "data/figure6/Figure6_paired_results.csv"
    for required in (merged_path, sensory_path, figure6_path, figure6_d1d5_path):
        if not required.is_file():
            raise RuntimeError(f"Missing bundled frozen table: {required.relative_to(REPO_ROOT)}")

    merged = pd.read_excel(merged_path, sheet_name="MERGED_590")
    rheology = pd.read_excel(merged_path, sheet_name="RHEOLOGY_540")
    sensory = pd.read_excel(sensory_path, sheet_name="Visual ratings")
    measured = pd.read_csv(state / "Fig4a_measured_conditions_540.csv", encoding="utf-8-sig")
    projection = pd.read_csv(state / "reduced_order_states_131072_projection.csv", encoding="utf-8-sig")
    pointcloud_manifest = pd.read_csv(
        REPO_ROOT / "data_manifests" / "pointcloud_590_manifest.csv", encoding="utf-8-sig"
    )
    comsol_manifest = pd.read_csv(
        REPO_ROOT / "data_manifests" / "comsol_required_model_manifest.csv", encoding="utf-8-sig"
    )
    cad_files = sorted((REPO_ROOT / "data" / "reference_geometry").glob("*.[sS][tT][lL]"))

    evaluator_columns = [column for column in sensory.columns if str(column).removeprefix("Evaluator_").isdigit()]
    checks = {
        "merged_database_rows": int(len(merged)),
        "rheology_workbook_rows": int(len(rheology)),
        "sensory_rows": int(len(sensory)),
        "sensory_evaluator_columns": int(len(evaluator_columns)),
        "measured_rheology_rows": int(len(measured)),
        "reduced_state_projection_rows": int(len(projection)),
        "pointcloud_manifest_rows": int(len(pointcloud_manifest)),
        "reference_cad_files": int(len(cad_files)),
        "comsol_manifest_rows": int(len(comsol_manifest)),
        "canonical_figure6_source_record_present": (
            REPO_ROOT / "data/figure6/README.md"
        ).is_file(),
    }
    expected = {
        "merged_database_rows": 590,
        "rheology_workbook_rows": 540,
        "sensory_rows": 590,
        "sensory_evaluator_columns": 50,
        "measured_rheology_rows": 540,
        "reduced_state_projection_rows": 131072,
        "pointcloud_manifest_rows": 590,
        "reference_cad_files": 11,
    }
    for key, value in expected.items():
        if checks[key] != value:
            raise RuntimeError(f"Bundled research-data check failed: {key}={checks[key]}, expected {value}")
    if checks["comsol_manifest_rows"] < 1 or not checks["canonical_figure6_source_record_present"]:
        raise RuntimeError("COMSOL manifest or canonical Figure 6 source record is missing")
    return checks


def main() -> None:
    syntax = syntax_check()
    research_data = bundled_research_data_check()
    figure3 = figure3_check()
    figure5_script = REPO_ROOT / "scripts" / "figure5" / "verify_deployable_models.py"
    subprocess.run([sys.executable, str(figure5_script)], cwd=REPO_ROOT, check=True)
    figure5 = json.loads((REPO_ROOT / "reports" / "figure5_inference_validation.json").read_text(encoding="utf-8"))
    report = {
        "verdict": "PASS",
        "repository_root": str(REPO_ROOT),
        "python": sys.version,
        "syntax_checked_python_files": len(syntax),
        "syntax_checked_files": syntax,
        "bundled_research_data": research_data,
        "figure3_frozen_inference": figure3,
        "figure5_frozen_inference": figure5,
        "actually_executed": [
            "syntax compilation of every packaged Python file",
            "row/count and presence checks for bundled 590/540/131072-projection, sensory, CAD, point-cloud manifest, COMSOL manifest, and canonical Figure 6 sources",
            "Figure 3 inference on the packaged 99-row fixed-test snapshot",
            "Figure 5 inference on the packaged 121-row candidate-test snapshot",
            "Figure 5 243-candidate example recommendation",
        ],
        "external_full_scale_workflows": [
            "model retraining or hyperparameter search",
            "full point-cloud batches",
            "the complete 131072-state computation",
            "COMSOL builds or solves",
            "full main/supplementary figure regeneration",
        ],
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
