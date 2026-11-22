from __future__ import annotations

import hashlib
import json
import platform
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_predict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN = REPO_ROOT / "working" / "figure3_analysis"
FREEZE = REPO_ROOT / "data" / "frozen_tables"
DERIVED = RUN / "01_DERIVED_DATA"
RESULTS = RUN / "02_MODEL_RESULTS"
for folder in (DERIVED, RESULTS):
    folder.mkdir(parents=True, exist_ok=True)

MASTER = FREEZE / "Food3D_590_merged_dataset.xlsx"
NEW518 = FREEZE / "Food3D_D1D5_NewSensory_518_extrudable_training.csv"
SPLIT = FREEZE / "split_assignment_80_20.csv"

SEED = 20260902
DIMS = [
    "D1_global_geometry",
    "D2_structural_completeness",
    "D3_surface_deviation",
    "D4_polarity_tolerance",
    "D5_spatial_heterogeneity",
]
PROCESS = ["Print_temp_C", "Model_height_mm", "Print_speed_mm_s", "Nozzle_mm", "Nozzle_height_mm"]
RHEO_RAW = ["Gp_mean_Pa", "Gpp_mean_Pa", "eta_10_Pa_s", "eta_100_Pa_s", "Selected_tau0", "Selected_K", "Selected_n"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def metrics(y, p):
    return {
        "R2": float(r2_score(y, p)),
        "RMSE": float(mean_squared_error(y, p) ** 0.5),
        "MAE": float(mean_absolute_error(y, p)),
    }


def objective_evaluator():
    # Prespecified from the training-only selection in the corrected 518-print campaign.
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingRegressor(
            random_state=SEED,
            max_iter=300,
            learning_rate=0.1,
            max_leaf_nodes=7,
            l2_regularization=1.0,
        )),
    ])


def add_features(d: pd.DataFrame) -> pd.DataFrame:
    x = d[PROCESS].copy().astype(float)
    for col in RHEO_RAW[:-1]:
        x[f"log10_{col}"] = np.log10(pd.to_numeric(d[col], errors="coerce").clip(lower=0) + 1e-8)
    x["Selected_n"] = pd.to_numeric(d["Selected_n"], errors="coerce")
    if x.isna().any().any():
        raise ValueError(f"Missing predictors: {x.isna().sum()[x.isna().sum() > 0].to_dict()}")
    return x


def scaled(model):
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", model)])


def unscaled(model):
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])


def model_library():
    return {
        "Dummy_mean": (unscaled(DummyRegressor()), {}),
        "Linear": (scaled(LinearRegression()), {}),
        "Ridge": (scaled(Ridge()), {"model__alpha": [0.1, 1.0, 10.0, 100.0]}),
        "ElasticNet": (scaled(ElasticNet(max_iter=50000, random_state=SEED)), {"model__alpha": [0.001, 0.01, 0.1], "model__l1_ratio": [0.2, 0.5, 0.8]}),
        "KNN": (scaled(KNeighborsRegressor()), {"model__n_neighbors": [5, 9, 15, 25], "model__weights": ["uniform", "distance"]}),
        "SVR_RBF": (scaled(SVR()), {"model__C": [1.0, 10.0, 100.0], "model__gamma": ["scale", 0.01, 0.1], "model__epsilon": [0.1, 1.0, 5.0]}),
        "RandomForest": (unscaled(RandomForestRegressor(n_estimators=400, random_state=SEED, n_jobs=1)), {"model__min_samples_leaf": [1, 2, 4], "model__max_features": ["sqrt", 1.0]}),
        "ExtraTrees": (unscaled(ExtraTreesRegressor(n_estimators=400, random_state=SEED, n_jobs=1)), {"model__min_samples_leaf": [1, 2, 4], "model__max_features": ["sqrt", 1.0]}),
        "GradientBoosting": (unscaled(GradientBoostingRegressor(random_state=SEED)), {"model__n_estimators": [100, 300], "model__learning_rate": [0.03, 0.1], "model__max_depth": [1, 2, 3]}),
        "HistGradientBoosting": (unscaled(HistGradientBoostingRegressor(random_state=SEED, max_iter=300)), {"model__learning_rate": [0.03, 0.1], "model__max_leaf_nodes": [7, 15, 31], "model__l2_regularization": [0.0, 1.0]}),
        "AdaBoost": (unscaled(AdaBoostRegressor(random_state=SEED)), {"model__n_estimators": [100, 300], "model__learning_rate": [0.03, 0.1, 0.5]}),
    }


def bootstrap_by_group(y, p, groups, n_boot=2000):
    rng = np.random.default_rng(SEED)
    unique = np.asarray(pd.unique(groups))
    rows = []
    y = np.asarray(y); p = np.asarray(p); groups = np.asarray(groups)
    for i in range(n_boot):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == g) for g in sampled])
        if np.unique(y[idx]).size < 2:
            continue
        rows.append({"bootstrap": i, **metrics(y[idx], p[idx])})
    return pd.DataFrame(rows)


def main():
    raw = pd.read_excel(MASTER, sheet_name="MERGED_590")
    new = pd.read_csv(NEW518, encoding="utf-8-sig")
    split = pd.read_csv(SPLIT, encoding="utf-8-sig")[["Attempt_ID", "split"]]
    assert len(raw) == 590 and raw["Attempt_ID"].nunique() == 590
    assert len(new) == 518 and new["Attempt_ID"].nunique() == 518
    raw = raw.drop(columns=["split"], errors="ignore").merge(split, on="Attempt_ID", validate="one_to_one")
    raw["Extrusion_success"] = (~raw["No_extrusion_zero_rule"].astype(bool)).astype(int)
    assert raw["Extrusion_success"].value_counts().to_dict() == {1: 518, 0: 72}
    sensory = new[["Attempt_ID", "Sensory_mean_new", "Sensory_SD_new", "Sensory_n_new"]]
    d = raw.merge(sensory, on="Attempt_ID", how="left", validate="one_to_one")
    success = d["Extrusion_success"].eq(1)
    assert d.loc[success, "Sensory_mean_new"].notna().all()
    assert d.loc[~success, "Sensory_mean_new"].isna().all()

    # Global OOF values provide one leakage-controlled D1-D5-derived objective score for every successful print.
    ds = d.loc[success].copy().reset_index(drop=True)
    cv_all = GroupKFold(n_splits=5)
    oof_all = cross_val_predict(
        objective_evaluator(), ds[DIMS], ds["Sensory_mean_new"],
        groups=ds["Experiment_group"], cv=cv_all, n_jobs=-1,
    )
    ds["OQI_D1D5_OOF"] = np.clip(oof_all, 0, 100)
    ds["OQI_D1D5_residual_to_sensory"] = ds["Sensory_mean_new"] - ds["OQI_D1D5_OOF"]
    global_oof_metrics = metrics(ds["Sensory_mean_new"], ds["OQI_D1D5_OOF"])

    # Strict target generation for Figure 5: training scores are inner-OOF; test scores are generated
    # by an evaluator fitted only on the fixed training partition.
    train = ds["split"].eq("train")
    test = ds["split"].eq("test")
    assert (int(train.sum()), int(test.sum())) == (419, 99)
    inner = GroupKFold(n_splits=5)
    oqi_train = np.clip(cross_val_predict(
        objective_evaluator(), ds.loc[train, DIMS], ds.loc[train, "Sensory_mean_new"],
        groups=ds.loc[train, "Experiment_group"], cv=inner, n_jobs=-1,
    ), 0, 100)
    evaluator = objective_evaluator().fit(ds.loc[train, DIMS], ds.loc[train, "Sensory_mean_new"])
    oqi_test = np.clip(evaluator.predict(ds.loc[test, DIMS]), 0, 100)
    ds["OQI_D1D5_STRICT_FIG5"] = np.nan
    ds.loc[train, "OQI_D1D5_STRICT_FIG5"] = oqi_train
    ds.loc[test, "OQI_D1D5_STRICT_FIG5"] = oqi_test
    strict_test_evaluator_metrics = metrics(ds.loc[test, "Sensory_mean_new"], oqi_test)

    # Export 590 rows: failure states are protocol-defined zeros, not model predictions.
    out590 = d[["Attempt_ID", "Sample_ID", "Sample_ID_norm", "Material_code", "Experiment_group", "Rheology_ID", "split", "Extrusion_success", *DIMS]].copy()
    out590 = out590.merge(ds[["Attempt_ID", "Sensory_mean_new", "Sensory_SD_new", "OQI_D1D5_OOF", "OQI_D1D5_STRICT_FIG5"]], on="Attempt_ID", how="left", validate="one_to_one")
    failure = out590["Extrusion_success"].eq(0)
    out590.loc[failure, ["OQI_D1D5_OOF", "OQI_D1D5_STRICT_FIG5"]] = 0.0
    out590["OQI_value_origin"] = np.where(failure, "PROTOCOL_DEFINED_NON_EXTRUDABLE_ZERO", "D1D5_MODEL_DERIVED")
    out590.to_csv(DERIVED / "Food3D_590_D1D5_objective_quality_OQI.csv", index=False, encoding="utf-8-sig")
    ds.to_csv(DERIVED / "Food3D_518_D1D5_OQI_extrudable.csv", index=False, encoding="utf-8-sig")

    # Fair Figure 5 comparison using only successful prints and strict D1-D5-derived targets.
    X = add_features(ds)
    feature_sets = {"Process only": PROCESS, "Rheology + process": list(X.columns)}
    rows, winners, pred_cols = [], {}, {}
    for feature_set, cols in feature_sets.items():
        candidates = []
        for name, (estimator, grid) in model_library().items():
            search = GridSearchCV(
                estimator, grid if grid else [{}],
                scoring={"rmse": "neg_root_mean_squared_error", "r2": "r2", "mae": "neg_mean_absolute_error"},
                refit="rmse", cv=GroupKFold(n_splits=5), n_jobs=-1, error_score="raise",
            )
            search.fit(X.loc[train, cols], ds.loc[train, "OQI_D1D5_STRICT_FIG5"], groups=ds.loc[train, "Experiment_group"])
            idx = search.best_index_
            pred = np.clip(search.best_estimator_.predict(X.loc[test, cols]), 0, 100)
            row = {
                "feature_set": feature_set,
                "model": name,
                "train_group_cv_RMSE": float(-search.cv_results_["mean_test_rmse"][idx]),
                "train_group_cv_R2": float(search.cv_results_["mean_test_r2"][idx]),
                "train_group_cv_MAE": float(-search.cv_results_["mean_test_mae"][idx]),
                **{f"test_{k}": v for k, v in metrics(ds.loc[test, "OQI_D1D5_STRICT_FIG5"], pred).items()},
                "best_params": json.dumps(search.best_params_, ensure_ascii=False, sort_keys=True),
            }
            candidates.append((row, search.best_estimator_, pred))
            print(f"{feature_set} | {name} | CV RMSE={row['train_group_cv_RMSE']:.3f} | test R2={row['test_R2']:.3f}", flush=True)
        candidates.sort(key=lambda z: (z[0]["train_group_cv_RMSE"], z[0]["model"]))
        for rank, (row, _, _) in enumerate(candidates, start=1):
            row["cv_rank"] = rank
            row["selected_by_training_cv"] = rank == 1
            rows.append(row)
        winners[feature_set] = candidates[0][0]
        pred_cols[feature_set] = candidates[0][2]

    comparison = pd.DataFrame(rows)
    comparison.to_csv(RESULTS / "Figure5_model_comparison_D1D5_OQI.csv", index=False, encoding="utf-8-sig")
    pred = ds.loc[test, ["Attempt_ID", "Sample_ID_norm", "Experiment_group", "Rheology_ID", "OQI_D1D5_STRICT_FIG5", "Sensory_mean_new"]].copy()
    pred["pred_process_only"] = pred_cols["Process only"]
    pred["pred_rheology_process"] = pred_cols["Rheology + process"]
    pred.to_csv(RESULTS / "Figure5_fixed_test_predictions_99_D1D5_OQI.csv", index=False, encoding="utf-8-sig")

    ci_summary = {}
    for key, col in [("Process only", "pred_process_only"), ("Rheology + process", "pred_rheology_process")]:
        boot = bootstrap_by_group(pred["OQI_D1D5_STRICT_FIG5"], pred[col], pred["Experiment_group"])
        safe = key.lower().replace(" ", "_").replace("+", "plus")
        boot.to_csv(RESULTS / f"Figure5_bootstrap_{safe}.csv", index=False)
        ci_summary[key] = {m: [float(boot[m].quantile(0.025)), float(boot[m].quantile(0.975))] for m in ["R2", "RMSE", "MAE"]}

    # Provenance manifest for frozen inputs.
    manifest = []
    for path in sorted(FREEZE.iterdir()):
        if path.is_file():
            manifest.append({"file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    pd.DataFrame(manifest).to_csv(RESULTS / "input_checksums.csv", index=False, encoding="utf-8-sig")

    summary = {
        "status": "ANALYSIS_COMPLETE_FIGURES_PENDING",
        "analysis_population": {"all": 590, "extrudable": 518, "non_extrudable": 72, "train_extrudable": 419, "test_extrudable": 99},
        "objective_score_definition": "D1-D5 to 50-rater mean through prespecified HistGradientBoosting; global values are GroupKFold OOF; Figure 5 uses train-inner-OOF and train-only-to-test predictions",
        "global_518_oof_D1D5_to_sensory": global_oof_metrics,
        "fixed_99_test_D1D5_evaluator": strict_test_evaluator_metrics,
        "figure5_selected": winners,
        "figure5_group_bootstrap_95CI": ci_summary,
        "failure_state_handling": "72 non-extrudable rows assigned protocol-defined OQI=0 only in the 590-row export; excluded from quality regression",
        "material_identity_used": False,
        "software": {"python": platform.python_version(), "sklearn": sklearn.__version__},
    }
    (RESULTS / "UPDATE_ANALYSIS_SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
