from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT / "external_data" / "source_tree"
RUN = ROOT / "outputs" / "Food3D_Figure3_Figure5_Update_20261015"
MASTER = RUN / "00_FROZEN_SOURCE" / "Food3D_590_merged_dataset.xlsx"
OQI = RUN / "01_DERIVED_DATA" / "Food3D_590_D1D5_objective_quality_OQI.csv"
NEW518 = RUN / "00_FROZEN_SOURCE" / "Food3D_D1D5_NewSensory_518_extrudable_training.csv"
OUT = RUN / "04_FIGURE5_ALL590_REANALYSIS"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20260902


def load_shared():
    path = RUN / "run_update_analysis.py"
    spec = importlib.util.spec_from_file_location("shared_update", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def classifier_library():
    def scaled(model):
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", model)])

    def unscaled(model):
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])

    return {
        "Dummy_prior": (unscaled(DummyClassifier(strategy="prior", random_state=SEED)), {}),
        "Logistic": (scaled(LogisticRegression(max_iter=20000, class_weight="balanced", random_state=SEED)), {"model__C": [0.01, 0.1, 1.0, 10.0]}),
        "SVC_RBF": (scaled(SVC(probability=True, class_weight="balanced", random_state=SEED)), {"model__C": [0.1, 1.0, 10.0, 100.0], "model__gamma": ["scale", 0.01, 0.1]}),
        "RandomForest": (unscaled(RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=SEED, n_jobs=1)), {"model__min_samples_leaf": [1, 2, 4, 8], "model__max_features": ["sqrt", 1.0]}),
        "ExtraTrees": (unscaled(ExtraTreesClassifier(n_estimators=500, class_weight="balanced", random_state=SEED, n_jobs=1)), {"model__min_samples_leaf": [1, 2, 4, 8], "model__max_features": ["sqrt", 1.0]}),
        "HistGradientBoosting": (unscaled(HistGradientBoostingClassifier(max_iter=300, class_weight="balanced", random_state=SEED)), {"model__learning_rate": [0.03, 0.1], "model__max_leaf_nodes": [7, 15, 31], "model__l2_regularization": [0.0, 1.0]}),
    }


def class_metrics(y, prob):
    y = np.asarray(y, dtype=int)
    prob = np.asarray(prob, dtype=float)
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "PR_AUC_failure": float(average_precision_score(y, prob)),
        "ROC_AUC": float(roc_auc_score(y, prob)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "failure_recall": float(recall_score(y, pred, pos_label=1)),
        "specificity_success": float(tn / (tn + fp)),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def main():
    shared = load_shared()
    master = pd.read_excel(MASTER, sheet_name="MERGED_590")
    oqi = pd.read_csv(OQI, encoding="utf-8-sig")
    new = pd.read_csv(NEW518, encoding="utf-8-sig")
    keep = ["Attempt_ID", "Extrusion_success", "OQI_D1D5_OOF", "OQI_value_origin"]
    d = master.merge(oqi[keep], on="Attempt_ID", validate="one_to_one")
    d = d.merge(new[["Attempt_ID", "Sensory_mean_new"]], on="Attempt_ID", how="left", validate="one_to_one")
    assert len(d) == 590 and d["Attempt_ID"].nunique() == 590
    assert d["Extrusion_success"].value_counts().to_dict() == {1: 518, 0: 72}

    # Figure 5 uses a rheological-condition-exclusive outer split. Select one
    # deterministic StratifiedGroupKFold partition by row-count and failure-rate
    # balance only; no outcome prediction metric is consulted.
    y_fail = 1 - d["Extrusion_success"].astype(int)
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    candidates = []
    global_failure_rate = float(y_fail.mean())
    for fold, (tr_idx, te_idx) in enumerate(outer.split(d, y_fail, groups=d["Rheology_ID"]), start=1):
        score = abs(len(te_idx) / len(d) - 0.20) + abs(float(y_fail.iloc[te_idx].mean()) - global_failure_rate)
        candidates.append((score, fold, tr_idx, te_idx))
    candidates.sort(key=lambda z: (z[0], z[1]))
    _, selected_fold, train_idx, test_idx = candidates[0]
    d["Figure5_split"] = "train"
    d.loc[d.index[test_idx], "Figure5_split"] = "test"
    train = d["Figure5_split"].eq("train")
    test = d["Figure5_split"].eq("test")
    assert not (set(d.loc[train, "Rheology_ID"]) & set(d.loc[test, "Rheology_ID"]))
    split_audit = pd.DataFrame([
        {
            "fold": fold,
            "balance_score": score,
            "test_n": len(te_idx),
            "test_failure_n": int(y_fail.iloc[te_idx].sum()),
            "test_failure_rate": float(y_fail.iloc[te_idx].mean()),
            "selected": fold == selected_fold,
        }
        for score, fold, _, te_idx in candidates
    ]).sort_values("fold")
    split_audit.to_csv(OUT / "Figure5_rheology_group_exclusive_split_audit.csv", index=False)
    d[["Attempt_ID", "Rheology_ID", "Experiment_group", "Figure5_split", "Extrusion_success"]].to_csv(
        OUT / "Figure5_rheology_group_exclusive_split_assignment.csv", index=False, encoding="utf-8-sig"
    )

    # Rebuild the D1-D5 objective target inside the Figure 5 outer split.
    successful = d["Extrusion_success"].eq(1)
    success_train = train & successful
    success_test = test & successful
    d["OQI_D1D5_STRICT_FIG5_GROUP_EXCLUSIVE"] = 0.0
    train_oof = np.clip(cross_val_predict(
        shared.objective_evaluator(), d.loc[success_train, shared.DIMS], d.loc[success_train, "Sensory_mean_new"],
        groups=d.loc[success_train, "Rheology_ID"], cv=GroupKFold(n_splits=5), n_jobs=-1,
    ), 0, 100)
    evaluator = shared.objective_evaluator().fit(
        d.loc[success_train, shared.DIMS], d.loc[success_train, "Sensory_mean_new"]
    )
    test_oqi = np.clip(evaluator.predict(d.loc[success_test, shared.DIMS]), 0, 100)
    d.loc[success_train, "OQI_D1D5_STRICT_FIG5_GROUP_EXCLUSIVE"] = train_oof
    d.loc[success_test, "OQI_D1D5_STRICT_FIG5_GROUP_EXCLUSIVE"] = test_oqi
    assert d["OQI_D1D5_STRICT_FIG5_GROUP_EXCLUSIVE"].notna().all()
    assert (d.loc[~successful, "OQI_D1D5_STRICT_FIG5_GROUP_EXCLUSIVE"] == 0).all()
    d[["Attempt_ID", "Sample_ID_norm", "Rheology_ID", "Figure5_split", "Extrusion_success", "OQI_D1D5_STRICT_FIG5_GROUP_EXCLUSIVE"]].to_csv(
        OUT / "Figure5_ALL590_strict_objective_target.csv", index=False, encoding="utf-8-sig"
    )

    X = shared.add_features(d)
    feature_sets = {"Process only": shared.PROCESS, "Rheology + process": list(X.columns)}

    # Main Figure 5 integrated-outcome regression: all 590 attempts, including 72 observed failures.
    reg_rows, reg_selected, reg_predictions = [], {}, {}
    for feature_set, cols in feature_sets.items():
        candidates = []
        for name, (estimator, grid) in shared.model_library().items():
            search = GridSearchCV(
                estimator, grid if grid else [{}],
                scoring={"rmse": "neg_root_mean_squared_error", "r2": "r2", "mae": "neg_mean_absolute_error"},
                refit="rmse", cv=GroupKFold(n_splits=5), n_jobs=-1, error_score="raise",
            )
            search.fit(X.loc[train, cols], d.loc[train, "OQI_D1D5_STRICT_FIG5_GROUP_EXCLUSIVE"], groups=d.loc[train, "Rheology_ID"])
            idx = search.best_index_
            pred = np.clip(search.best_estimator_.predict(X.loc[test, cols]), 0, 100)
            row = {
                "feature_set": feature_set, "model": name,
                "train_group_cv_RMSE": float(-search.cv_results_["mean_test_rmse"][idx]),
                "train_group_cv_R2": float(search.cv_results_["mean_test_r2"][idx]),
                "train_group_cv_MAE": float(-search.cv_results_["mean_test_mae"][idx]),
                **{f"test_{k}": v for k, v in shared.metrics(d.loc[test, "OQI_D1D5_STRICT_FIG5_GROUP_EXCLUSIVE"], pred).items()},
                "best_params": json.dumps(search.best_params_, ensure_ascii=False, sort_keys=True),
            }
            candidates.append((row, search.best_estimator_, pred))
            print(f"REG | {feature_set} | {name} | CV RMSE={row['train_group_cv_RMSE']:.3f} | test R2={row['test_R2']:.3f}", flush=True)
        candidates.sort(key=lambda z: (z[0]["train_group_cv_RMSE"], z[0]["model"]))
        for rank, (row, _, _) in enumerate(candidates, start=1):
            row["cv_rank"] = rank; row["selected_by_training_cv"] = rank == 1
            reg_rows.append(row)
        reg_selected[feature_set] = candidates[0][0]
        reg_predictions[feature_set] = candidates[0][2]
    pd.DataFrame(reg_rows).to_csv(OUT / "Figure5_ALL590_integrated_OQI_model_comparison.csv", index=False, encoding="utf-8-sig")
    reg_pred = d.loc[test, ["Attempt_ID", "Sample_ID_norm", "Experiment_group", "Rheology_ID", "Extrusion_success", "OQI_D1D5_STRICT_FIG5_GROUP_EXCLUSIVE"]].copy()
    reg_pred["pred_process_only"] = reg_predictions["Process only"]
    reg_pred["pred_rheology_process"] = reg_predictions["Rheology + process"]
    reg_pred.to_csv(OUT / "Figure5_ALL590_fixed_test_predictions_120.csv", index=False, encoding="utf-8-sig")

    reg_ci = {}
    for label, col in [("Process only", "pred_process_only"), ("Rheology + process", "pred_rheology_process")]:
        boot = shared.bootstrap_by_group(reg_pred["OQI_D1D5_STRICT_FIG5_GROUP_EXCLUSIVE"], reg_pred[col], reg_pred["Rheology_ID"])
        boot.to_csv(OUT / f"Figure5_ALL590_bootstrap_{label.lower().replace(' ', '_').replace('+', 'plus')}.csv", index=False)
        reg_ci[label] = {m: [float(boot[m].quantile(0.025)), float(boot[m].quantile(0.975))] for m in ["R2", "RMSE", "MAE"]}

    # Stage-1 extrusion feasibility, also all 590 attempts. Positive class = observed non-extrudable failure.
    cls_rows, cls_selected, cls_predictions = [], {}, {}
    for feature_set, cols in feature_sets.items():
        candidates = []
        for name, (estimator, grid) in classifier_library().items():
            search = GridSearchCV(
                estimator, grid if grid else [{}],
                scoring={"prauc": "average_precision", "rocauc": "roc_auc", "balacc": "balanced_accuracy"},
                refit="prauc", cv=StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED), n_jobs=-1, error_score="raise",
            )
            search.fit(X.loc[train, cols], y_fail.loc[train], groups=d.loc[train, "Rheology_ID"])
            idx = search.best_index_
            prob = search.best_estimator_.predict_proba(X.loc[test, cols])[:, 1]
            row = {
                "feature_set": feature_set, "model": name,
                "train_group_cv_PR_AUC_failure": float(search.cv_results_["mean_test_prauc"][idx]),
                "train_group_cv_ROC_AUC": float(search.cv_results_["mean_test_rocauc"][idx]),
                "train_group_cv_balanced_accuracy": float(search.cv_results_["mean_test_balacc"][idx]),
                **{f"test_{k}": v for k, v in class_metrics(y_fail.loc[test], prob).items()},
                "best_params": json.dumps(search.best_params_, ensure_ascii=False, sort_keys=True),
            }
            candidates.append((row, search.best_estimator_, prob))
            print(f"CLS | {feature_set} | {name} | CV PR-AUC={row['train_group_cv_PR_AUC_failure']:.3f} | test PR-AUC={row['test_PR_AUC_failure']:.3f}", flush=True)
        candidates.sort(key=lambda z: (-z[0]["train_group_cv_PR_AUC_failure"], z[0]["model"]))
        for rank, (row, _, _) in enumerate(candidates, start=1):
            row["cv_rank"] = rank; row["selected_by_training_cv"] = rank == 1
            cls_rows.append(row)
        cls_selected[feature_set] = candidates[0][0]
        cls_predictions[feature_set] = candidates[0][2]
    pd.DataFrame(cls_rows).to_csv(OUT / "Figure5_ALL590_extrusion_classifier_comparison.csv", index=False, encoding="utf-8-sig")
    cls_pred = d.loc[test, ["Attempt_ID", "Sample_ID_norm", "Experiment_group", "Rheology_ID", "Extrusion_success"]].copy()
    cls_pred["Observed_failure"] = y_fail.loc[test].to_numpy()
    cls_pred["P_failure_process_only"] = cls_predictions["Process only"]
    cls_pred["P_failure_rheology_process"] = cls_predictions["Rheology + process"]
    cls_pred.to_csv(OUT / "Figure5_ALL590_failure_test_predictions_120.csv", index=False, encoding="utf-8-sig")

    summary = {
        "status": "REANALYZED_ALL_590",
        "primary_figure5_population": {"all_attempts": 590, "train": int(train.sum()), "test": int(test.sum()), "successful": 518, "non_extrudable": 72},
        "outer_split": {"method": "StratifiedGroupKFold partition chosen by size and failure-rate balance only", "group": "Rheology_ID", "selected_fold": int(selected_fold), "rheology_group_overlap": 0},
        "primary_integrated_outcome": "OQI_D1D5_STRICT_FIG5_GROUP_EXCLUSIVE; successful prints are D1-D5 model-derived inside the outer split, non-extrudable attempts are observed protocol-defined zeros",
        "integrated_OQI_regression_selected": reg_selected,
        "integrated_OQI_group_bootstrap_95CI": reg_ci,
        "extrusion_failure_classifier_selected": cls_selected,
        "conditional_518_analysis_role": "supplementary sensitivity analysis only",
        "material_identity_used": False,
    }
    (OUT / "Figure5_ALL590_REANALYSIS_SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
