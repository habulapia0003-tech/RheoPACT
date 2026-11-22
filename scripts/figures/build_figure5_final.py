from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.stats import wilcoxon


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT / "external_data" / "source_tree"
RUN = ROOT / "Figure5_NC_REFERENCE_MODE_TNR_v4_20261017"
SRC = RUN / "00_FROZEN_INPUTS"
DATA = RUN / "01_SOURCE_DATA"
PANELS = RUN / "02_INDIVIDUAL_PANELS"
MAIN = RUN / "03_MAIN_FIGURE"
QC = RUN / "04_QC"
for folder in (SRC, DATA, PANELS, MAIN, QC):
    folder.mkdir(parents=True, exist_ok=True)

FORMAL = ROOT / "Food3D_Figures2-6_FORMAL_FROZEN_20261016" / "Figure5"
ANALYSIS = FORMAL / "Frozen_Current_Domain_Analysis"
ACTIVE = FORMAL / "Figure_and_Panels"
OOF_DIR = ROOT / "Figure5_GROUPED_OOF_REPRODUCTION_20261017"
OOF_CSV = OOF_DIR / "Figure5b_grouped_oof_predictions_469.csv"
OOF_REPORT = OOF_DIR / "OOF_REPRODUCTION_REPORT.json"
NETWORK_CSV = ROOT / "outputs" / "Figure5_REDESIGN_OLDMODE_R0823_20261011" / "source_data" / "Figure5c_permutation_importance.csv"
RULE_ALIGNED = ROOT / "Figure5_RULE_ALIGNED_20261030" / "SourceData"
RANK_CSV = RULE_ALIGNED / "Figure5d_candidate_ranking_detail.csv"
SUMMARY_JSON = ANALYSIS / "Figure5_CURRENT_DOMAIN_SUMMARY.json"
TARGET_CSV = ANALYSIS / "Figure5_CURRENT_DOMAIN_strict_objective_target_590.csv"
LAND_CSV = RULE_ALIGNED / "Figure5e_candidate_landscape.csv"
TOP_CSV = RULE_ALIGNED / "Figure5e_top3_candidates.csv"

BLUE = "#3277A8"
PALE_BLUE = "#DCEAF3"
GREEN = "#2D8B68"
PALE_GREEN = "#E4F1EC"
ORANGE = "#EE741C"
PALE_ORANGE = "#FBE9DC"
NAVY = "#173F5F"
GREY = "#687781"
LIGHT = "#E3E9ED"
INK = "#20272E"
PURPLE = "#7A68A5"


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


base = import_module(ROOT / "build_figure5_nc_reference_layout_v3_20261017.py", "figure5_v3_base")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def save_all(fig, stem: Path):
    base.save_all(fig, stem, dpi=600)


def draw_b(container, oof: pd.DataFrame, summary: dict, letter=True):
    gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=container, wspace=0.22)
    axes = [plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1])]
    if letter:
        base.panel_label(axes[0], "b", -0.17, 1.20)
    axes[0].text(0.0, 1.20, "Rheology improves grouped quality estimation",
                 transform=axes[0].transAxes, fontsize=8.0, fontweight="bold", ha="left")
    axes[0].text(1.05, 1.085,
                 "Five-fold candidate-group CV; n = 469, including 57 observed non-extrusions",
                 transform=axes[0].transAxes, fontsize=5.5, color=GREY, ha="center")
    configs = [
        ("Process only", "pred_process_only_grouped_oof", PALE_BLUE, NAVY),
        ("Rheology + process", "pred_rheology_process_grouped_oof", BLUE, BLUE),
    ]
    for ax, (label, col, color, title_color) in zip(axes, configs):
        m = summary["selected_models"][label]
        ax.scatter(oof["OQI_current_domain_strict"], oof[col], s=9.5, c=color,
                   alpha=0.62 if label == "Process only" else 0.72,
                   edgecolors="white", linewidths=0.18, rasterized=True)
        ax.plot([0, 100], [0, 100], "--", color=GREY, lw=0.8)
        ax.set_xlim(-3, 103); ax.set_ylim(-3, 103)
        ax.set_title(label, color=title_color, fontweight="bold", pad=3)
        ax.set_xlabel("Observed integrated outcome")
        ax.grid(True, color=LIGHT, lw=0.45)
        ax.text(0.045, 0.955,
                f"Mean CV $R^2$ = {m['train_group_cv_R2']:.3f}\n"
                f"RMSE = {m['train_group_cv_RMSE']:.2f}\n"
                f"MAE = {m['train_group_cv_MAE']:.2f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=5.55,
                fontweight="bold", bbox=dict(boxstyle="round,pad=.18", fc="white", ec="none", alpha=.88))
        base.clean(ax)
    axes[0].set_ylabel("Out-of-fold prediction", labelpad=1.5)
    return axes


def draw_c(ax, imp: pd.DataFrame, letter=True):
    ax.set_axis_off()
    if letter:
        base.panel_label(ax, "c", -0.10, 1.07)
    ax.text(0.0, 1.055, "Rheological contributions are phenotype specific",
            transform=ax.transAxes, fontsize=7.6, fontweight="bold", ha="left")
    labels = {
        "log10_Gp_mean_Pa": "$G'$",
        "log10_Gpp_mean_Pa": "$G''$",
        "log10_eta_10_Pa_s": "$\\eta_{10}$",
        "log10_eta_100_Pa_s": "$\\eta_{100}$",
        "log10_Selected_tau0": "$\\tau_0$",
        "log10_Selected_K": "$K$",
        "Selected_n": "$n$",
    }
    sub = imp[imp["feature"].isin(labels)].copy()
    sub["positive"] = sub["importance_R2_drop"].clip(lower=0)
    sub["weight"] = sub.groupby("phenotype")["positive"].transform(lambda s: s / s.sum())
    features = list(labels)
    y_left = np.linspace(0.78, 0.20, len(features))
    y_right = {"D2": 0.70, "D4": 0.48, "D5": 0.26}
    colors = {"D2": BLUE, "D4": GREEN, "D5": ORANGE}
    fills = {"D2": "#EAF2F8", "D4": "#EAF6F2", "D5": "#FFF1E7"}
    ax.text(0.18, 0.88, "Rheological descriptors", transform=ax.transAxes, ha="center",
            fontsize=6.2, fontweight="bold")
    ax.text(0.82, 0.88, "Digital phenotypes", transform=ax.transAxes, ha="center",
            fontsize=6.2, fontweight="bold")
    for feat, y in zip(features, y_left):
        p = FancyBboxPatch((0.08, y-0.032), 0.20, 0.064, transform=ax.transAxes,
                           boxstyle="round,pad=.007,rounding_size=.018", fc="#F3F0FA",
                           ec=PURPLE, lw=.75)
        ax.add_patch(p)
        ax.text(0.18, y, labels[feat], transform=ax.transAxes, ha="center", va="center",
                fontsize=6.1, fontweight="bold")
    names = {"D2": "D2  Structural completeness", "D4": "D4  Polarity and tolerance",
             "D5": "D5  Spatial heterogeneity"}
    for ph, y in y_right.items():
        p = FancyBboxPatch((0.70, y-0.055), 0.27, 0.11, transform=ax.transAxes,
                           boxstyle="round,pad=.007,rounding_size=.018", fc=fills[ph],
                           ec=colors[ph], lw=.8)
        ax.add_patch(p)
        ax.text(0.835, y, names[ph], transform=ax.transAxes, ha="center", va="center",
                fontsize=5.6, fontweight="bold")
    for feat, yl in zip(features, y_left):
        for ph, yr in y_right.items():
            value = float(sub.loc[(sub.feature == feat) & (sub.phenotype == ph), "weight"].iloc[0])
            ax.add_patch(FancyArrowPatch((0.285, yl), (0.695, yr), transform=ax.transAxes,
                                         arrowstyle="-", lw=0.35 + 8.5*value,
                                         alpha=0.18 + 0.55*min(value/0.35, 1), color=colors[ph]))
    ax.text(0.50, 0.08, "Held-out permutation contribution", transform=ax.transAxes,
            ha="center", fontsize=5.5, fontweight="bold")
    ax.plot([0.37, 0.43], [0.045, 0.045], transform=ax.transAxes, color=GREY, lw=1.2)
    ax.plot([0.52, 0.58], [0.045, 0.045], transform=ax.transAxes, color=GREY, lw=4.0)
    ax.text(0.40, 0.012, "Lower", transform=ax.transAxes, ha="center", fontsize=5.0, color=GREY)
    ax.text(0.55, 0.012, "Higher", transform=ax.transAxes, ha="center", fontsize=5.0, color=GREY)
    ax.text(0.99, 0.012, "Predictive, not causal", transform=ax.transAxes,
            ha="right", fontsize=5.0, color=GREY)


def retrieval_source(rank: pd.DataFrame):
    rows = []
    rng = np.random.default_rng(20260902)
    for method in ["process", "combined"]:
        g = rank[rank.method.eq(method)].copy().reset_index(drop=True)
        for k in [1, 2, 3, 4]:
            hit = (g["chosen_rank"].to_numpy() <= k).astype(float)
            boots = np.array([hit[rng.integers(0, len(hit), len(hit))].mean() for _ in range(5000)]) * 100
            rows.append({"method": method, "top_k": k, "retrieval_pct": 100*hit.mean(),
                         "low": np.quantile(boots, .025), "high": np.quantile(boots, .975), "n": len(hit)})
    return pd.DataFrame(rows)


def draw_d(container, rank: pd.DataFrame, letter=True):
    gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=container, wspace=0.34, width_ratios=[1.2, .8])
    ax1 = plt.subplot(gs[0, 0]); ax2 = plt.subplot(gs[0, 1])
    if letter:
        base.panel_label(ax1, "d", -0.18, 1.16)
    ax1.text(0.0, 1.16, "Rheology preserves a compact quality-ranked shortlist",
             transform=ax1.transAxes, fontsize=7.4, fontweight="bold", ha="left")
    ret = retrieval_source(rank)
    styles = {"process": ("Process only", "#B8CDDB"), "combined": ("Rheology + process", BLUE)}
    for method, (label, color) in styles.items():
        g = ret[ret.method.eq(method)]
        ax1.fill_between(g.top_k, g.low, g.high, color=color, alpha=.18, linewidth=0)
        ax1.plot(g.top_k, g.retrieval_pct, "o-", color=color, lw=1.25, ms=3.2, label=label)
        for x, y in zip(g.top_k, g.retrieval_pct):
            ax1.text(x, y+1.5, f"{y:.1f}", ha="center", fontsize=5.0,
                     color=NAVY if method == "combined" else GREY)
    ax1.set_xticks([1, 2, 3, 4], ["Top-1", "Top-2", "Top-3", "Top-4"])
    ax1.set_ylim(40, 104)
    ax1.set_ylabel("Retrieval (%)", labelpad=1.0)
    ax1.set_xlabel("Review budget")
    ax1.grid(axis="y", color=LIGHT, lw=.45)
    ax1.legend(loc="lower right", frameon=False, fontsize=5.0)
    base.clean(ax1)

    pivot = rank.pivot(index="Rheology_ID", columns="method", values="chosen_rank").dropna()
    bp = ax2.boxplot([pivot["process"], pivot["combined"]], positions=[0, 1], widths=.52,
                     patch_artist=True, showfliers=True,
                     boxprops=dict(linewidth=.75), medianprops=dict(color=NAVY, linewidth=1.0),
                     whiskerprops=dict(linewidth=.7), capprops=dict(linewidth=.7),
                     flierprops=dict(marker="o", markersize=2, markerfacecolor=BLUE, markeredgecolor="none", alpha=.65))
    for patch, color in zip(bp["boxes"], ["#C9D8E3", PALE_GREEN]):
        patch.set_facecolor(color)
        patch.set_edgecolor(GREY)
    pval = wilcoxon(pivot["combined"], pivot["process"], alternative="two-sided").pvalue
    ax2.set_xticks([0, 1], ["Process\nonly", "Rheology-\ninformed"])
    ax2.set_ylabel("Rank of best observed process")
    ax2.set_ylim(4.35, .65)
    ax2.grid(axis="y", color=LIGHT, lw=.45)
    ax2.text(.5, 1.03, f"paired Wilcoxon $P$ = {pval:.3f}", transform=ax2.transAxes,
             ha="center", fontsize=5.2)
    ax2.text(.5, .03, f"n = {len(pivot)} states; lower is better", transform=ax2.transAxes,
             ha="center", fontsize=5.0, color=GREY)
    base.clean(ax2)
    return ret, pval


def make_panels(counts, oof, summary, imp, rank, grid, top):
    fig, ax = plt.subplots(figsize=(7.2, 2.55)); base.draw_a(ax, counts)
    fig.subplots_adjust(left=.045, right=.995, top=.90, bottom=.04)
    save_all(fig, PANELS / "Figure5a_two_stage_framework_TNR"); plt.close(fig)

    fig = plt.figure(figsize=(7.2, 3.55)); gs = GridSpec(1, 1, figure=fig)
    draw_b(gs[0], oof, summary); fig.subplots_adjust(left=.085, right=.99, top=.82, bottom=.15)
    save_all(fig, PANELS / "Figure5b_grouped_oof_prediction_TNR"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.8)); draw_c(ax, imp)
    fig.subplots_adjust(left=.05, right=.99, top=.90, bottom=.08)
    save_all(fig, PANELS / "Figure5c_rheology_phenotype_network_TNR"); plt.close(fig)

    fig = plt.figure(figsize=(7.2, 3.65)); gs = GridSpec(1, 1, figure=fig)
    ret, pval = draw_d(gs[0], rank); fig.subplots_adjust(left=.09, right=.99, top=.84, bottom=.17)
    save_all(fig, PANELS / "Figure5d_candidate_retrieval_TNR"); plt.close(fig)

    fig = plt.figure(figsize=(7.2, 3.55)); gs = GridSpec(1, 1, figure=fig)
    base.draw_e(gs[0], grid, top); fig.subplots_adjust(left=.065, right=.995, top=.94, bottom=.19)
    save_all(fig, PANELS / "Figure5e_candidate_landscape_TNR"); plt.close(fig)
    return ret, pval


def make_composite(counts, oof, summary, imp, rank, grid, top):
    fig = plt.figure(figsize=(7.2047, 7.75))
    outer = GridSpec(3, 12, figure=fig, height_ratios=[1.08, 1.12, 1.50], hspace=.56, wspace=.70)
    base.draw_a(fig.add_subplot(outer[0, :6]), counts)
    draw_b(outer[0, 6:], oof, summary)
    draw_c(fig.add_subplot(outer[1, :6]), imp)
    draw_d(outer[1, 6:], rank)
    base.draw_e(outer[2, :], grid, top)
    fig.subplots_adjust(left=.055, right=.993, top=.985, bottom=.07)
    save_all(fig, MAIN / "Figure5_NC_reference_mode_TNR_v4_20261017")
    plt.close(fig)


def main():
    summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    oof_report = json.loads(OOF_REPORT.read_text(encoding="utf-8"))
    oof = pd.read_csv(OOF_CSV)
    target = pd.read_csv(TARGET_CSV)
    imp = pd.read_csv(NETWORK_CSV)
    rank = pd.read_csv(RANK_CSV)
    grid = pd.read_csv(LAND_CSV)
    top = pd.read_csv(TOP_CSV)
    assert len(oof) == 469 and int((oof.Extrusion_success == 0).sum()) == 57
    assert abs(summary["selected_models"]["Rheology + process"]["train_group_cv_R2"] - .8103420072184087) < 1e-12
    assert len(target) == 590 and int((target.Extrusion_success == 0).sum()) == 72
    assert rank.Rheology_ID.nunique() == 139
    assert set(rank.method) == {"process", "combined"}
    counts = {"all": 590, "success": 518, "failure": 72}

    for path in [SUMMARY_JSON, TARGET_CSV, OOF_CSV, OOF_REPORT, NETWORK_CSV, RANK_CSV, LAND_CSV, TOP_CSV]:
        shutil.copy2(path, SRC / path.name)
        shutil.copy2(path, DATA / path.name)
    ret, pval = make_panels(counts, oof, summary, imp, rank, grid, top)
    ret.to_csv(DATA / "Figure5d_retrieval_curve_current.csv", index=False, encoding="utf-8-sig")
    make_composite(counts, oof, summary, imp, rank, grid, top)

    report = {
        "status": "BUILT_PENDING_FINAL_QC",
        "layout": "user-provided reference mode: a workflow; b dual grouped-OOF scatter; c contribution network; d retrieval plus rank; e candidate landscape",
        "n_total": 590,
        "grouped_cv_n": 469,
        "grouped_cv_non_extrusions": 57,
        "primary_grouped_cv_R2": summary["selected_models"]["Rheology + process"]["train_group_cv_R2"],
        "independent_test_R2_displayed": False,
        "independent_test_result_preserved_in_frozen_analysis": True,
        "ranking_states": 139,
        "ranking_rule": "maximum grouped out-of-fold predicted integrated OQI; no probability multiplication",
        "paired_rank_p": float(pval),
        "font": "Times New Roman",
        "formats": ["SVG", "PDF", "PNG 600 dpi", "TIFF 600 dpi"],
        "no_model_reselection": True,
    }
    (QC / "BUILD_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    manifest = []
    for p in sorted(RUN.rglob("*")):
        if p.is_file() and p.parent != QC:
            manifest.append({"relative_path": str(p.relative_to(RUN)), "bytes": p.stat().st_size, "sha256": sha256(p)})
    pd.DataFrame(manifest).to_csv(QC / "SHA256_MANIFEST.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
