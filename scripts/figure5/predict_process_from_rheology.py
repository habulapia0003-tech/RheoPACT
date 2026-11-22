from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROCESS = ["Print_temp_C", "Model_height_mm", "Print_speed_mm_s", "Nozzle_mm", "Nozzle_height_mm"]
RHEO = ["Gp_mean_Pa", "Gpp_mean_Pa", "eta_10_Pa_s", "eta_100_Pa_s", "Selected_tau0", "Selected_K", "Selected_n"]
FEATURES = PROCESS + [f"log10_{x}" for x in RHEO[:-1]] + ["Selected_n"]
DEFAULT_FEASIBILITY_THRESHOLD = 0.50
REPO_ROOT = Path(__file__).resolve().parents[2]


def feature_matrix(d: pd.DataFrame) -> pd.DataFrame:
    x = d[PROCESS].copy().astype(float)
    for col in RHEO[:-1]:
        x[f"log10_{col}"] = np.log10(pd.to_numeric(d[col], errors="raise").clip(lower=0) + 1e-8)
    x["Selected_n"] = pd.to_numeric(d["Selected_n"], errors="raise")
    return x[FEATURES]


def support_check(bundle: dict, rheology: dict) -> tuple[bool, str]:
    support = bundle["rheology_training_support"].set_index("rheology_feature")
    outside = []
    for key, value in rheology.items():
        lo, hi = float(support.loc[key, "minimum"]), float(support.loc[key, "maximum"])
        if float(value) < lo or float(value) > hi:
            outside.append(f"{key}={value:g} outside [{lo:g}, {hi:g}]")
    return not outside, "; ".join(outside) if outside else "All seven rheological inputs are within their marginal training ranges."


def recommend(bundle: dict, rheology: dict, top_k: int = 10) -> pd.DataFrame:
    missing = [key for key in RHEO if key not in rheology]
    if missing:
        raise ValueError(f"Missing rheology inputs: {missing}")
    grid = bundle["candidate_process_grid"].copy()
    for key in RHEO:
        grid[key] = float(rheology[key])
    x = feature_matrix(grid)
    quality = np.clip(bundle["quality_regressor"].predict(x), 0, 100)
    clf = bundle["feasibility_classifier"]
    p_ext = clf.predict_proba(x)[:, list(clf.classes_).index(1)]
    grid["P_extrudable"] = p_ext
    grid["Predicted_integrated_outcome"] = quality
    threshold = float(bundle.get("feasibility_threshold", DEFAULT_FEASIBILITY_THRESHOLD))
    grid["Feasibility_pass"] = grid["P_extrudable"] >= threshold
    grid = grid.sort_values(
        ["Feasibility_pass", "Predicted_integrated_outcome", "P_extrudable"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    grid["Recommendation_rank"] = np.arange(1, len(grid) + 1)
    inside, message = support_check(bundle, rheology)
    grid["All_inputs_within_marginal_training_ranges"] = inside
    grid["Support_message"] = message
    return grid.head(top_k)


def main():
    parser = argparse.ArgumentParser(description="Rank Food3D printing process candidates from seven rheological inputs.")
    parser.add_argument("--model", type=Path, default=REPO_ROOT / "models" / "Figure5_RECOMMENDER_rheology_to_process_topk.joblib")
    parser.add_argument("--gp", type=float, required=True, help="Mean storage modulus G' in Pa")
    parser.add_argument("--gpp", type=float, required=True, help="Mean loss modulus G'' in Pa")
    parser.add_argument("--eta10", type=float, required=True, help="Viscosity at 10 s^-1 in Pa s")
    parser.add_argument("--eta100", type=float, required=True, help="Viscosity at 100 s^-1 in Pa s")
    parser.add_argument("--tau0", type=float, required=True, help="Yield stress in Pa")
    parser.add_argument("--K", type=float, required=True, help="Consistency coefficient in Pa s^n")
    parser.add_argument("--n", type=float, required=True, help="Flow behaviour index")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("recommended_processes.csv"))
    args = parser.parse_args()
    bundle = joblib.load(args.model)
    rheology = {"Gp_mean_Pa": args.gp, "Gpp_mean_Pa": args.gpp, "eta_10_Pa_s": args.eta10, "eta_100_Pa_s": args.eta100, "Selected_tau0": args.tau0, "Selected_K": args.K, "Selected_n": args.n}
    result = recommend(bundle, rheology, top_k=args.top_k)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(result.to_string(index=False))
    print(f"\nSaved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
