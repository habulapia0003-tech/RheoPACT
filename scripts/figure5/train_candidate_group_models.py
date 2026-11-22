from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedGroupKFold, cross_val_predict


REPO_ROOT = Path(__file__).resolve().parents[2]
MASTER = REPO_ROOT / "data" / "frozen_tables" / "Food3D_590_merged_dataset.xlsx"
NEW518 = REPO_ROOT / "data" / "frozen_tables" / "Food3D_D1D5_NewSensory_518_extrudable_training.csv"
OUT = REPO_ROOT / "working" / "figure5_training"
SEED = 20260902


def import_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    shared = import_module(REPO_ROOT / "scripts" / "figure3" / "run_update_analysis.py", "shared_update_current")
    all590 = import_module(Path(__file__).with_name("reanalyze_all590_cross_state.py"), "all590_tools")
    d = pd.read_excel(MASTER, sheet_name="MERGED_590")
    new = pd.read_csv(NEW518, encoding="utf-8-sig")
    d["Extrusion_success"] = (~d["No_extrusion_zero_rule"].astype(bool)).astype(int)
    d = d.merge(new[["Attempt_ID", "Sensory_mean_new"]], on="Attempt_ID", how="left", validate="one_to_one")
    process = shared.PROCESS
    d["Candidate_group_ID"] = d[["Rheology_ID", *process]].astype(str).agg("|".join, axis=1)
    y_fail = 1 - d["Extrusion_success"]

    # Same rheological state may appear in both partitions, but the exact
    # rheology-process candidate (including replicates) may not.
    cv_outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    cand = []
    rate = float(y_fail.mean())
    for fold, (tr, te) in enumerate(cv_outer.split(d, y_fail, groups=d["Candidate_group_ID"]), start=1):
        balance = abs(len(te)/len(d)-0.20) + abs(float(y_fail.iloc[te].mean())-rate)
        cand.append((balance, fold, tr, te))
    cand.sort(key=lambda z: (z[0], z[1]))
    _, selected_fold, tr_idx, te_idx = cand[0]
    d["Candidate_split"] = "train"
    d.loc[d.index[te_idx], "Candidate_split"] = "test"
    train = d.Candidate_split.eq("train"); test = d.Candidate_split.eq("test")
    assert not (set(d.loc[train, "Candidate_group_ID"]) & set(d.loc[test, "Candidate_group_ID"]))

    # Split-specific D1-D5 objective target.
    success = d.Extrusion_success.eq(1)
    success_train = train & success; success_test = test & success
    d["OQI_current_domain_strict"] = 0.0
    oof = cross_val_predict(
        shared.objective_evaluator(), d.loc[success_train, shared.DIMS], d.loc[success_train, "Sensory_mean_new"],
        groups=d.loc[success_train, "Candidate_group_ID"], cv=GroupKFold(n_splits=5), n_jobs=-1,
    )
    evaluator = shared.objective_evaluator().fit(d.loc[success_train, shared.DIMS], d.loc[success_train, "Sensory_mean_new"])
    d.loc[success_train, "OQI_current_domain_strict"] = np.clip(oof, 0, 100)
    d.loc[success_test, "OQI_current_domain_strict"] = np.clip(evaluator.predict(d.loc[success_test, shared.DIMS]), 0, 100)
    d[["Attempt_ID", "Sample_ID_norm", "Rheology_ID", "Candidate_group_ID", "Candidate_split", "Extrusion_success", "OQI_current_domain_strict"]].to_csv(
        OUT / "Figure5_CURRENT_DOMAIN_strict_objective_target_590.csv", index=False, encoding="utf-8-sig"
    )

    X = shared.add_features(d)
    sets = {"Process only": process, "Rheology + process": list(X.columns)}
    rows, selected, predictions, fitted_models = [], {}, {}, {}
    for label, cols in sets.items():
        models = []
        for name, (estimator, grid) in shared.model_library().items():
            search = GridSearchCV(
                estimator, grid if grid else [{}],
                scoring={"rmse":"neg_root_mean_squared_error", "r2":"r2", "mae":"neg_mean_absolute_error"},
                refit="rmse", cv=GroupKFold(n_splits=5), n_jobs=-1, error_score="raise",
            )
            search.fit(X.loc[train, cols], d.loc[train, "OQI_current_domain_strict"], groups=d.loc[train, "Candidate_group_ID"])
            idx = search.best_index_
            pred = np.clip(search.best_estimator_.predict(X.loc[test, cols]), 0, 100)
            row = {
                "feature_set": label, "model": name,
                "train_group_cv_RMSE": float(-search.cv_results_["mean_test_rmse"][idx]),
                "train_group_cv_R2": float(search.cv_results_["mean_test_r2"][idx]),
                "train_group_cv_MAE": float(-search.cv_results_["mean_test_mae"][idx]),
                **{f"test_{k}": v for k,v in shared.metrics(d.loc[test, "OQI_current_domain_strict"], pred).items()},
                "best_params": json.dumps(search.best_params_, sort_keys=True),
            }
            models.append((row, search.best_estimator_, pred))
            print(f"{label} | {name} | CV RMSE={row['train_group_cv_RMSE']:.3f} | test R2={row['test_R2']:.3f}", flush=True)
        models.sort(key=lambda z:(z[0]["train_group_cv_RMSE"], z[0]["model"]))
        for rank, (row, _, _) in enumerate(models, start=1):
            row["cv_rank"] = rank; row["selected_by_training_cv"] = rank == 1; rows.append(row)
        selected[label] = models[0][0]; fitted_models[label] = models[0][1]; predictions[label] = models[0][2]
    pd.DataFrame(rows).to_csv(OUT / "Figure5_CURRENT_DOMAIN_candidate_group_model_comparison.csv", index=False, encoding="utf-8-sig")
    pred = d.loc[test, ["Attempt_ID", "Rheology_ID", "Candidate_group_ID", "Extrusion_success", "OQI_current_domain_strict"]].copy()
    pred["pred_process_only"] = predictions["Process only"]
    pred["pred_rheology_process"] = predictions["Rheology + process"]
    pred.to_csv(OUT / "Figure5_CURRENT_DOMAIN_test_predictions.csv", index=False, encoding="utf-8-sig")
    joblib.dump(fitted_models["Process only"], OUT / "Figure5_CURRENT_DOMAIN_process_only_model.joblib")
    joblib.dump(fitted_models["Rheology + process"], OUT / "Figure5_CURRENT_DOMAIN_rheology_process_model.joblib")
    d[["Attempt_ID", "Rheology_ID", "Candidate_group_ID", "Candidate_split", "Extrusion_success"]].to_csv(
        OUT / "Figure5_CURRENT_DOMAIN_candidate_group_split.csv", index=False, encoding="utf-8-sig"
    )
    summary = {
        "scope": "current-domain process selection for measured rheological states",
        "all_attempts": 590,
        "train_n": int(train.sum()), "test_n": int(test.sum()),
        "candidate_group_overlap": 0,
        "rheology_state_overlap_allowed_by_design": len(set(d.loc[train,"Rheology_ID"]) & set(d.loc[test,"Rheology_ID"])),
        "selected_fold": int(selected_fold),
        "selected_models": selected,
        "cross_state_external_validation": "reported separately in Figure5_ALL590_REANALYSIS_SUMMARY.json",
    }
    (OUT / "Figure5_CURRENT_DOMAIN_SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
