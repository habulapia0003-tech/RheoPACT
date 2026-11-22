from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import joblib
import pandas as pd


OUT = Path(__file__).resolve().parent
MODELS = OUT / "Models"
EXAMPLES = OUT / "Examples"
REPORTS = OUT / "Reports"
CURRENT = MODELS / "Figure5_RECOMMENDER_rheology_to_process_topk.joblib"
LEGACY = MODELS / "Figure5_RECOMMENDER_rheology_to_process_topk_LEGACY_PRODUCT_RULE.joblib"
THRESHOLD = 0.50


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def import_inference():
    path = OUT / "predict_process_from_rheology.py"
    spec = importlib.util.spec_from_file_location("figure5_inference_revised", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    if not LEGACY.exists():
        shutil.copy2(CURRENT, LEGACY)
    legacy_hash = sha256(LEGACY)
    bundle = joblib.load(LEGACY)
    bundle["version"] = "2026-10-30-ranking-rule-revision"
    bundle["feasibility_threshold"] = THRESHOLD
    bundle["ranking_rule"] = (
        "Candidates with P_extrudable >= 0.50 pass the feasibility screen; "
        "screen-passing candidates are ranked by Predicted_integrated_outcome descending, "
        "with P_extrudable used only as the final tie-breaker."
    )
    bundle["ranking_revision"] = (
        "No estimator was refitted. The former product rule was retired because the integrated-outcome "
        "target already assigns protocol-defined zero to observed non-extrudable attempts."
    )
    bundle["legacy_recommender_sha256"] = legacy_hash
    joblib.dump(bundle, CURRENT)

    inference = import_inference()
    example = pd.read_csv(EXAMPLES / "example_rheology_input.csv", encoding="utf-8-sig").iloc[0].to_dict()
    top10 = inference.recommend(bundle, example, top_k=10)
    summary_path = REPORTS / "MODEL_SUMMARY.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    example_id = summary["example_rheology_id"]
    top10.insert(0, "Rheology_ID", example_id)
    top10.to_csv(EXAMPLES / "example_rheology_to_top10_processes.csv", index=False, encoding="utf-8-sig")
    top10.head(5).drop(columns=["Rheology_ID"]).to_csv(EXAMPLES / "CLI_example_top5.csv", index=False, encoding="utf-8-sig")

    summary["ranking_revision"] = {
        "date": "2026-10-30",
        "estimators_refitted": False,
        "feasibility_threshold": THRESHOLD,
        "primary_rank": "Predicted_integrated_outcome descending among feasibility-passing candidates",
        "probability_role": "separate feasibility screen and reported auxiliary output",
        "legacy_recommender_sha256": legacy_hash,
        "current_recommender_sha256": sha256(CURRENT),
    }
    summary["example_top1"] = top10.iloc[0].to_dict()
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        "status": "RANKING_RULE_REVISED",
        "estimators_refitted": False,
        "quality_model_unchanged": True,
        "feasibility_model_unchanged": True,
        "old_rule": "P_extrudable * Predicted_integrated_outcome",
        "new_rule": bundle["ranking_rule"],
        "threshold": THRESHOLD,
        "legacy_recommender": str(LEGACY),
        "legacy_sha256": legacy_hash,
        "current_recommender": str(CURRENT),
        "current_sha256": sha256(CURRENT),
    }
    (REPORTS / "RANKING_RULE_REVISION_20261030.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
