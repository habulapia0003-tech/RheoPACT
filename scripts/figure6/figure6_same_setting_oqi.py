"""Evaluate and plot Figure 6e at the six recorded adjusted settings."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import tifffile


DOMAINS = [f"D{i}" for i in range(1, 6)]
SCORE_INPUTS = ["D1_global_geometry", "D2_structural_completeness", "D3_surface_deviation",
                "D4_polarity_tolerance", "D5_spatial_heterogeneity"]
PROCESS = ["Print_temp_C", "Model_height_mm", "Print_speed_mm_s", "Nozzle_mm", "Nozzle_height_mm"]
RHEO = ["Gp_mean_Pa", "Gpp_mean_Pa", "eta_10_Pa_s", "eta_100_Pa_s", "Selected_tau0", "Selected_K", "Selected_n"]
PREDICTED = "Predicted_Figure5_OQI"
PRINT_DERIVED = "Print_derived_Figure5_OQI"
ERROR = "Prediction_minus_print_derived_OQI"


def features(prints):
    x = prints[PROCESS].astype(float).copy()
    for column in RHEO[:-1]:
        x[f"log10_{column}"] = np.log10(prints[column].clip(lower=0) + 1e-8)
    x["Selected_n"] = prints.Selected_n
    return x


def comparison(prints, recommender, scorer):
    adjusted = prints.loc[prints.Condition.eq("Adjusted")].copy().reset_index(drop=True)
    assert len(adjusted) == adjusted.Material_ID.nunique() == 6
    regressor = recommender["quality_regressor"]
    classifier = recommender["feasibility_classifier"]
    adjusted[PREDICTED] = np.clip(regressor.predict(features(adjusted)), 0, 100)
    adjusted[PRINT_DERIVED] = np.clip(scorer.predict(adjusted[DOMAINS].set_axis(SCORE_INPUTS, axis=1)), 0, 100)
    adjusted[ERROR] = adjusted[PREDICTED] - adjusted[PRINT_DERIVED]
    adjusted["Absolute_error"] = adjusted[ERROR].abs()
    ranks = []
    for row in adjusted.itertuples(index=False):
        values = row._asdict()
        grid = recommender["candidate_process_grid"].assign(**{name: values[name] for name in RHEO})
        x = features(grid)
        grid["P"] = classifier.predict_proba(x)[:, list(classifier.classes_).index(1)]
        grid["Feasible"] = grid.P.ge(recommender["feasibility_threshold"])
        grid["Quality"] = np.clip(regressor.predict(x), 0, 100)
        ranked = grid.sort_values(["Feasible", "Quality", "P"], ascending=False).reset_index(drop=True)
        match = np.isclose(ranked[PROCESS], [values[name] for name in PROCESS], rtol=0, atol=1e-10).all(axis=1)
        assert match.sum() == 1 and len(ranked) == 243
        ranks.append(int(np.flatnonzero(match)[0] + 1))
    adjusted["Rank_among_243_candidates"] = ranks
    return adjusted[["Material_ID", "Material", *RHEO, *PROCESS, *DOMAINS, PREDICTED, PRINT_DERIVED, ERROR,
                     "Absolute_error", "Rank_among_243_candidates"]]


def metrics(data):
    return {
        "n": len(data),
        "MAE": float(data.Absolute_error.mean()),
        "RMSE": float(np.sqrt(np.mean(data[ERROR]**2))),
        "mean_signed_error": float(data[ERROR].mean()),
        "max_absolute_error": float(data.Absolute_error.max()),
        "minimum_tested_rank": int(data.Rank_among_243_candidates.min()),
        "maximum_tested_rank": int(data.Rank_among_243_candidates.max()),
    }


def draw_panel(data, folder: Path):
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 7, "pdf.fonttype": 42,
                        "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False,
                        "axes.linewidth": .7})
    blue, orange, navy = "#2F6F9F", "#E88932", "#173F5F"
    fig, ax = plt.subplots(figsize=(3.55, 2.56))
    fig.subplots_adjust(left=.275, right=.805, bottom=.26, top=.69)
    fig.text(.025, .965, "e", va="top", size=9, weight="bold")
    fig.text(.095, .965, "Predicted and print-derived OQI\nat the tested settings",
             va="top", size=8, weight="bold", color=navy, linespacing=1.03)
    y = np.arange(len(data))
    ax.hlines(y, data[PREDICTED], data[PRINT_DERIVED], color="#B6C6D1", lw=1.15)
    ax.scatter(data[PREDICTED], y, s=24, color=blue, marker="D", ec="white", lw=.45, zorder=3)
    ax.scatter(data[PRINT_DERIVED], y, s=27, color=orange, ec="white", lw=.45, zorder=4)
    lo = min(58, 5 * np.floor(min(data[PREDICTED].min(), data[PRINT_DERIVED].min()) / 5) - 1)
    hi = max(85.5, 5 * np.ceil(max(data[PREDICTED].max(), data[PRINT_DERIVED].max()) / 5) + .5)
    ax.set(xlim=(lo, hi), xticks=np.arange(5 * np.ceil(lo / 5), hi, 10 if hi - lo > 35 else 5), ylim=(5.5, -.5))
    ax.set_xlabel("Objective quality index (OQI)", fontweight="bold", labelpad=3)
    ax.set_yticks(y, data.Material, fontsize=6.7)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.tick_params(axis="x", length=2.4, pad=2)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color="#E4EAEE", lw=.55)
    ax.set_axisbelow(True)
    ax.legend(handles=[Line2D([], [], marker="D", color=blue, lw=0, ms=4, label="Predicted"),
                       Line2D([], [], marker="o", color=orange, lw=0, ms=4.4, label="Print-derived")],
              loc="lower left", bbox_to_anchor=(0, 1.09), frameon=False, ncol=2, fontsize=6.6,
              handletextpad=.24, handlelength=.8, columnspacing=.9, borderaxespad=0)
    ax.plot([1.05, 1.05], [-.5, 5.5], transform=ax.get_yaxis_transform(), color="#DDE5EB", lw=.7, clip_on=False)
    for i, value in enumerate(data[ERROR]):
        ax.text(1.20, i, f"{value:.2f}".replace("-", "−"), transform=ax.get_yaxis_transform(),
                va="center", ha="center", size=6.5, color=navy, weight="bold")
    ax.text(1.20, 1.08, "Error", transform=ax.transAxes, ha="center", weight="bold", size=6.7, color=navy)
    result = metrics(data)
    fig.text(.275, .067, f"n = 6     MAE = {result['MAE']:.2f}     RMSE = {result['RMSE']:.2f}",
             size=6.6, weight="bold", color=navy)
    fig.set_dpi(600)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(folder / f"Figure6e.{ext}", dpi=600, facecolor="white")
    fig.canvas.draw()
    tifffile.imwrite(folder / "Figure6e.tiff", np.asarray(fig.canvas.buffer_rgba())[:, :, :3],
                     photometric="rgb", resolution=(600, 600), resolutionunit="INCH", compression="deflate")
    plt.close(fig)

