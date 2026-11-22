from __future__ import annotations

import hashlib
import json
import shutil
from io import BytesIO
from pathlib import Path

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import tifffile
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image
from scipy.spatial import ConvexHull
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT / "external_data" / "source_tree"
FROZEN = ROOT / "Food3D_Figures2-6_FORMAL_FROZEN_20261016" / "Figure3"
RUN = ROOT / "Figure3_NC_REFERENCE_LAYOUT_TNR_20261017"
SRC = RUN / "00_FROZEN_INPUTS"
DATA_OUT = RUN / "01_SOURCE_DATA"
PANELS = RUN / "02_INDIVIDUAL_PANELS"
MAIN = RUN / "03_MAIN_FIGURE"
QC = RUN / "04_QC"
for folder in (SRC, DATA_OUT, PANELS, MAIN, QC):
    folder.mkdir(parents=True, exist_ok=True)

DATA = FROZEN / "00_FROZEN_SOURCE" / "Food3D_D1D5_NewSensory_518_extrudable_training.csv"
MODEL = FROZEN / "00_FROZEN_SOURCE" / "selected_extrudable_only_model.joblib"
SOURCE_OLD = FROZEN / "03_FIGURE3_REBUILT" / "source_data"
PCA_CSV = SOURCE_OLD / "Figure3b_PCA_scores_518.csv"
PCA_VAR_CSV = SOURCE_OLD / "Figure3b_PCA_variance.csv"
ASSOC_CSV = SOURCE_OLD / "Figure3c_associations_518.csv"
BETA_CSV = SOURCE_OLD / "Figure3c_standardized_coefficients_518.csv"
PRED_CSV = SOURCE_OLD / "Figure3d_fixed_test_predictions_99.csv"
IMP_CSV = SOURCE_OLD / "Figure3e_permutation_importance_99.csv"

FEATURES = [
    "D1_global_geometry", "D2_structural_completeness", "D3_surface_deviation",
    "D4_polarity_tolerance", "D5_spatial_heterogeneity",
]
DIMS = ["D1", "D2", "D3", "D4", "D5"]
TARGET = "Sensory_mean_new"
SEED = 20260902

INK = "#20272E"
GRAY = "#65717A"
GRID = "#DDE4E8"
NAVY = "#285B7E"
BLUE = "#7098B5"
PALE_BLUE = "#E8F0F5"
GREEN = "#4E826A"
PALE_GREEN = "#EAF2ED"
PURPLE = "#6B5B83"
PALE_PURPLE = "#F1EEF5"
ORANGE = "#D36A38"
PALE_ORANGE = "#FAEEE8"
RED = "#B65A51"

CMAP_BLUE = LinearSegmentedColormap.from_list("phenotype_blue", ["#F2F6F8", "#AFC7D7", "#315F80"])
CMAP_GREEN = LinearSegmentedColormap.from_list("sensory_green", ["#F3F7F4", "#A7C5B2", "#3D7157"])
CMAP_DIVERGE = LinearSegmentedColormap.from_list("shap_div", ["#3F7EA7", "#EFEFEF", "#C76554"])

mpl.rcParams.update({
    "font.family": "Times New Roman",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
    "font.size": 7.2,
    "axes.labelsize": 7.3,
    "axes.titlesize": 8.2,
    "axes.titleweight": "bold",
    "xtick.labelsize": 6.6,
    "ytick.labelsize": 6.6,
    "legend.fontsize": 6.3,
    "axes.linewidth": 0.7,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def save_all(fig: plt.Figure, stem: Path, dpi: int = 600) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight", pad_inches=0.035)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.035)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.035)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.035)
    buf.seek(0)
    with Image.open(buf) as im:
        arr = np.asarray(im.convert("RGB").copy())
    tifffile.imwrite(
        stem.with_suffix(".tiff"), arr, photometric="rgb",
        resolution=(dpi, dpi), resolutionunit="INCH", compression="deflate"
    )


def panel_letter(ax, letter: str) -> None:
    ax.text(-0.085, 1.06, letter, transform=ax.transAxes, fontsize=11.5,
            fontweight="bold", va="top", ha="left", clip_on=False)


def clean(ax, grid: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(color=GRID, linewidth=0.55, zorder=0)


def draw_table(ax, x, y, w, h, rows, cols, values, cmap, vmin, vmax, title, value_cols=True):
    ax.text(x, y+h+0.035, title, transform=ax.transAxes, fontsize=7.8,
            fontweight="bold", ha="left", va="bottom")
    nr, nc = len(rows)+1, len(cols)
    cw, rh = w/nc, h/nr
    norm = Normalize(vmin=vmin, vmax=vmax)
    for j, col in enumerate(cols):
        ax.add_patch(Rectangle((x+j*cw, y+h-rh), cw, rh, transform=ax.transAxes,
                               fc="#E7EDF1", ec="white", lw=0.7))
        ax.text(x+(j+0.5)*cw, y+h-rh/2, col, transform=ax.transAxes,
                ha="center", va="center", fontsize=6.5, fontweight="bold")
    for i, row in enumerate(rows):
        yy = y+h-(i+2)*rh
        for j in range(nc):
            val = values[i][j]
            if j == 0 or not value_cols:
                fc = "#F5F7F8"
                tc = INK
            else:
                fc = cmap(norm(float(val)))
                tc = "white" if norm(float(val)) > 0.58 else INK
            ax.add_patch(Rectangle((x+j*cw, yy), cw, rh, transform=ax.transAxes,
                                   fc=fc, ec="white", lw=0.7))
            text = str(val) if j == 0 else f"{float(val):.1f}"
            ax.text(x+(j+0.5)*cw, yy+rh/2, text, transform=ax.transAxes,
                    ha="center", va="center", fontsize=6.1, color=tc)


def draw_a(ax, d):
    ax.set_axis_off()
    panel_letter(ax, "a")
    ax.set_title("Digital phenotype–perception dataset construction", loc="left", pad=3)
    chosen_ids = ["ms1", "ms2", "ms3", "sp57", "sp58", "sp59"]
    ex = d.set_index("Sample_ID_norm").loc[chosen_ids]
    rows = ["ms1", "ms2", "ms3", "…", "sp57", "sp58", "sp59"]
    pvals = []
    svals = []
    for sid in ["ms1", "ms2", "ms3"]:
        r = ex.loc[sid]
        pvals.append([sid] + [r[f] for f in FEATURES])
        svals.append([sid, r[TARGET]])
    pvals.append(["…", np.nan, np.nan, np.nan, np.nan, np.nan])
    svals.append(["…", np.nan])
    for sid in ["sp57", "sp58", "sp59"]:
        r = ex.loc[sid]
        pvals.append([sid] + [r[f] for f in FEATURES])
        svals.append([sid, r[TARGET]])

    # Draw values manually so the ellipsis row stays neutral.
    draw_table(ax, 0.015, 0.20, 0.40, 0.60, rows, ["Sample ID"]+DIMS,
               pvals, CMAP_BLUE, 0, 100, "Digital phenotype table")
    draw_table(ax, 0.465, 0.20, 0.17, 0.60, rows, ["Sample ID", "Visual score"],
               svals, CMAP_GREEN, 0, 100, "Sensory evaluation")
    # cover the ellipsis cells because NaN color mapping is intentionally avoided
    for x, w, nc in [(0.015, 0.40, 6), (0.465, 0.17, 2)]:
        rh = 0.60/8
        yy = 0.20+0.60-(3+2)*rh
        for j in range(nc):
            ax.add_patch(Rectangle((x+j*w/nc, yy), w/nc, rh, transform=ax.transAxes,
                                   fc="#FFFFFF", ec="white", lw=0.7, zorder=5))
        ax.text(x+w/2, yy+rh/2, "…", transform=ax.transAxes, ha="center", va="center",
                fontsize=7.2, zorder=6)

    ax.text(0.430, 0.50, "Matched by\nSample ID", transform=ax.transAxes,
            ha="center", va="center", fontsize=6.4, color=GRAY)
    ax.add_patch(FancyArrowPatch((0.635, 0.50), (0.685, 0.50), transform=ax.transAxes,
                                 arrowstyle="-|>", mutation_scale=10, lw=1.1, color=GRAY))
    box = FancyBboxPatch((0.695, 0.24), 0.285, 0.54, transform=ax.transAxes,
                         boxstyle="round,pad=0.012,rounding_size=0.015",
                         fc="#FBFAFC", ec=PURPLE, lw=1.2)
    ax.add_patch(box)
    ax.text(0.837, 0.72, "Phenotype–perception dataset", transform=ax.transAxes,
            ha="center", va="center", fontsize=8.0, fontweight="bold", color=PURPLE)
    ax.text(0.837, 0.655, "518 matched successfully extruded prints", transform=ax.transAxes,
            ha="center", fontsize=6.7)
    lines = [
        "D1  Global geometric fidelity", "D2  Structural completeness",
        "D3  Surface-deviation magnitude", "D4  Deviation polarity and tolerance",
        "D5  Spatial heterogeneity", "+  Mean visual quality score",
    ]
    for i, t in enumerate(lines):
        ax.text(0.720, 0.58-i*0.055, t, transform=ax.transAxes, ha="left",
                fontsize=6.25, fontweight="bold" if t.startswith("+") else "normal")
    ax.add_patch(FancyArrowPatch((0.837, 0.235), (0.837, 0.16), transform=ax.transAxes,
                                 arrowstyle="->", mutation_scale=9, lw=0.9, color=PURPLE))
    for x, label in [(0.705, "Association analysis"), (0.845, "Prediction modelling")]:
        p = FancyBboxPatch((x, 0.055), 0.13, 0.095, transform=ax.transAxes,
                           boxstyle="round,pad=0.006,rounding_size=0.012",
                           fc="white", ec=PURPLE, lw=1.0)
        ax.add_patch(p)
        ax.text(x+0.065, 0.102, label, transform=ax.transAxes, ha="center", va="center",
                fontsize=6.1, fontweight="bold")
    ax.text(0.695, 0.005, "72 non-extrudable attempts are retained for feasibility analysis but excluded from perception regression.",
            transform=ax.transAxes, ha="left", fontsize=5.7, color=ORANGE)


def draw_b(ax, pca, var):
    panel_letter(ax, "b")
    clean(ax, grid=True)
    pts = pca[["PC1", "PC2"]].to_numpy()
    hull = ConvexHull(pts)
    poly = pts[hull.vertices]
    ax.fill(poly[:, 0], poly[:, 1], color=BLUE, alpha=0.09, zorder=0)
    ax.plot(np.r_[poly[:, 0], poly[0, 0]], np.r_[poly[:, 1], poly[0, 1]], color=BLUE, lw=0.8, alpha=0.5)
    sc = ax.scatter(pca.PC1, pca.PC2, c=pca[TARGET], s=15, cmap=CMAP_BLUE,
                    vmin=10, vmax=95, alpha=0.82, edgecolors="white", linewidths=0.18, zorder=2)
    ax.set_title("Digital phenotype landscape organizes perceived quality", loc="left", pad=4)
    ax.set_xlabel(f"PC1 ({100*var.iloc[0].explained_variance_ratio:.1f}%)")
    ax.set_ylabel(f"PC2 ({100*var.iloc[1].explained_variance_ratio:.1f}%)")
    cb = plt.colorbar(sc, ax=ax, fraction=0.047, pad=0.025)
    cb.set_label("Mean visual score")
    ax.text(0.03, 0.04, "n = 518 successful prints", transform=ax.transAxes, fontsize=6.2, color=GRAY)


def draw_c(container, assoc, betas, letter=True):
    gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=container, width_ratios=[1.05, 1.25], wspace=0.48)
    axh = plt.subplot(gs[0, 0])
    axb = plt.subplot(gs[0, 1])
    if letter:
        panel_letter(axh, "c")
    axh.set_title("Digital phenotype determinants of perceived quality", loc="left", pad=4)
    order = DIMS
    aa = assoc.set_index("dimension").loc[order]
    mat = np.c_[aa["marginal_spearman_rho"], aa["partial_spearman_rho"]]
    im = axh.imshow(mat, cmap=CMAP_BLUE, vmin=-0.1, vmax=0.8, aspect="auto")
    for i in range(5):
        for j in range(2):
            v = mat[i, j]
            axh.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                     color="white" if v > 0.45 else INK, fontweight="bold" if abs(v) > 0.5 else "normal")
    axh.set_yticks(range(5), order)
    axh.set_xticks([0, 1], ["Spearman ρ", "Partial ρ"], rotation=25, ha="right")
    axh.tick_params(length=0)
    for s in axh.spines.values():
        s.set_linewidth(0.6)

    bb = betas.set_index("dimension").loc[order, "standardized_beta"]
    idx = np.argsort(np.abs(bb.to_numpy()))
    vals = bb.to_numpy()[idx]
    labs = np.array(order)[idx]
    colors = [BLUE if v >= 0 else ORANGE for v in vals]
    axb.barh(np.arange(5), vals, color=colors, height=0.58)
    axb.axvline(0, color=GRAY, lw=0.7)
    axb.set_yticks(range(5), labs)
    axb.set_xlabel("Standardized coefficient, β")
    axb.set_title("Standardized partial coefficients", loc="left", pad=4)
    clean(axb, grid=False)
    lim = max(0.7, np.max(np.abs(vals))*1.22)
    axb.set_xlim(-lim, lim)
    for i, v in enumerate(vals):
        axb.text(v + (0.025 if v >= 0 else -0.025), i, f"{v:+.2f}", va="center",
                 ha="left" if v >= 0 else "right", fontsize=6.2)
    return axh, axb


def draw_d(container, pred, metrics, letter=True):
    gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=container, width_ratios=[1.55, 0.62], wspace=0.36)
    ax = plt.subplot(gs[0, 0])
    axr = plt.subplot(gs[0, 1])
    if letter:
        panel_letter(ax, "d")
    clean(ax, grid=True)
    y, p = pred[TARGET].to_numpy(), pred["prediction"].to_numpy()
    ax.scatter(y, p, s=22, facecolors="white", edgecolors=NAVY, linewidths=0.85, alpha=0.9, zorder=2)
    lo, hi = min(y.min(), p.min())-3, max(y.max(), p.max())+3
    ax.plot([lo, hi], [lo, hi], ls="--", lw=0.9, color=GRAY)
    ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="Observed mean visual score", ylabel="Predicted mean visual score")
    ax.set_title("Digital phenotypes predict human-perceived quality", loc="left", pad=4)
    ax.text(0.04, 0.96,
            f"Independent test n = 99\n$R^2$ = {metrics['R2']:.3f}\nRMSE = {metrics['RMSE']:.2f}\nMAE = {metrics['MAE']:.2f}",
            transform=ax.transAxes, va="top", fontsize=6.7, fontweight="bold")
    residual = y-p
    # Horizontal residual density with deterministic Gaussian kernels.
    yy = np.linspace(residual.min()-2, residual.max()+2, 250)
    bw = max(1.8, 1.06*np.std(residual)*len(residual)**(-1/5))
    den = np.exp(-0.5*((yy[:, None]-residual[None, :])/bw)**2).sum(axis=1)/(len(residual)*bw*np.sqrt(2*np.pi))
    axr.fill_betweenx(yy, 0, den, color=PALE_BLUE, edgecolor=NAVY, linewidth=0.8)
    axr.axhline(0, color=GRAY, lw=0.7)
    axr.set_xlabel("Density")
    axr.set_ylabel("Residual (observed − predicted)")
    axr.set_title("Residual distribution", loc="left", pad=4)
    axr.text(0.98, 0.96, f"mean = {residual.mean():+.2f}\nSD = {residual.std(ddof=1):.2f}",
             transform=axr.transAxes, ha="right", va="top", fontsize=6.0)
    clean(axr, grid=False)
    return ax, axr


def draw_e(container, shap_values, X, importance, letter=True):
    gs = GridSpecFromSubplotSpec(1, 3, subplot_spec=container, width_ratios=[1.55, 0.82, 1.15], wspace=0.48)
    ax = plt.subplot(gs[0, 0])
    axb = plt.subplot(gs[0, 1])
    axi = plt.subplot(gs[0, 2])
    if letter:
        panel_letter(ax, "e")
    ax.set_title("Interpretable phenotype contributions to perception", loc="left", pad=4)
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    rng = np.random.default_rng(SEED)
    for yi, j in enumerate(order):
        vals = shap_values[:, j]
        feat = X.iloc[:, j].to_numpy()
        ranks = pd.Series(feat).rank(pct=True).to_numpy()
        jitter = rng.normal(0, 0.085, size=len(vals))
        ax.scatter(vals, yi+jitter, c=ranks, cmap=CMAP_DIVERGE, vmin=0, vmax=1,
                   s=9, alpha=0.63, edgecolors="none")
    ax.axvline(0, color=GRAY, lw=0.7, ls="--")
    ax.set_yticks(range(5), np.array(DIMS)[order])
    ax.invert_yaxis()
    ax.set_xlabel("SHAP value (score units)")
    clean(ax, grid=False)

    vals = mean_abs[order][::-1]
    labs = np.array(DIMS)[order][::-1]
    axb.barh(np.arange(5), vals, color=BLUE, height=0.58)
    axb.set_yticks(range(5), labs)
    axb.set_xlabel("Mean |SHAP value|")
    axb.set_title("Global importance", loc="left", pad=4)
    clean(axb, grid=False)

    axi.set_axis_off()
    box = FancyBboxPatch((0.02, 0.08), 0.96, 0.84, transform=axi.transAxes,
                         boxstyle="round,pad=0.018,rounding_size=0.02",
                         fc="#F7F8F8", ec="#AAB6BD", lw=0.8)
    axi.add_patch(box)
    axi.text(0.08, 0.82, "Interpretation", transform=axi.transAxes,
             fontsize=7.2, fontweight="bold")
    top = np.array(DIMS)[order][:3]
    lines = [
        "Predictive, not causal.",
        f"• {top[0]} ranked first globally.",
        f"• {top[1]} and {top[2]} added",
        "  complementary information.",
        "• Colour: phenotype score.",
    ]
    for i, t in enumerate(lines):
        axi.text(0.08, 0.69-i*0.115, t, transform=axi.transAxes,
                 fontsize=5.75, color=GRAY if i == 0 else INK)
    grad = np.linspace(0, 1, 256).reshape(1, -1)
    iax = axi.inset_axes([0.18, 0.105, 0.68, 0.055])
    iax.imshow(grad, aspect="auto", cmap=CMAP_DIVERGE)
    iax.set_xticks([0, 255], ["Low", "High"])
    iax.set_yticks([])
    iax.set_xlabel("Phenotype score", labelpad=-1)
    for s in iax.spines.values():
        s.set_linewidth(0.6)
    return ax, axb, axi


def make_panel_a(d):
    fig, ax = plt.subplots(figsize=(7.20, 2.55))
    draw_a(ax, d)
    fig.subplots_adjust(left=0.035, right=0.995, top=0.90, bottom=0.04)
    save_all(fig, PANELS / "Figure3a_dataset_construction_TNR")
    plt.close(fig)


def make_panel_b(pca, var):
    fig, ax = plt.subplots(figsize=(3.85, 3.25))
    draw_b(ax, pca, var)
    fig.tight_layout(pad=0.8)
    save_all(fig, PANELS / "Figure3b_phenotype_landscape_TNR")
    plt.close(fig)


def make_panel_c(assoc, betas):
    fig = plt.figure(figsize=(5.25, 3.05))
    gs = GridSpec(1, 1, figure=fig)
    draw_c(gs[0], assoc, betas)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.88, bottom=0.20)
    save_all(fig, PANELS / "Figure3c_association_partial_coefficients_TNR")
    plt.close(fig)


def make_panel_d(pred, metrics):
    fig = plt.figure(figsize=(4.75, 3.15))
    gs = GridSpec(1, 1, figure=fig)
    draw_d(gs[0], pred, metrics)
    fig.subplots_adjust(left=0.10, right=0.985, top=0.88, bottom=0.18)
    save_all(fig, PANELS / "Figure3d_independent_test_residual_TNR")
    plt.close(fig)


def make_panel_e(sv, X, importance):
    fig = plt.figure(figsize=(7.20, 3.15))
    gs = GridSpec(1, 1, figure=fig)
    es, eb, ei = draw_e(gs[0], sv, X, importance)
    es.set_title("Interpretable phenotype contributions\nto perception", loc="left", pad=4, fontsize=7.8)
    eb.set_title("Global\nimportance", loc="left", pad=4, fontsize=7.8)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.20)
    save_all(fig, PANELS / "Figure3e_shap_interpretation_TNR")
    plt.close(fig)


def make_composite(d, pca, var, assoc, betas, pred, metrics, sv, X, importance):
    # 183 mm wide; larger height deliberately preserves text at final publication size.
    fig = plt.figure(figsize=(7.2047, 7.75))
    outer = GridSpec(3, 14, figure=fig, height_ratios=[1.12, 1.00, 1.04], hspace=0.60, wspace=0.82)
    axa = fig.add_subplot(outer[0, :])
    draw_a(axa, d)
    axb = fig.add_subplot(outer[1, :6])
    draw_b(axb, pca, var)
    axb.set_title("Digital phenotype landscape organizes\nperceived quality", loc="left", pad=3, fontsize=7.0)
    ch, cb = draw_c(outer[1, 7:], assoc, betas)
    ch.set_title("Phenotype determinants\nof perceived quality", loc="left", pad=3, fontsize=6.9)
    cb.set_title("Standardized partial\ncoefficients", loc="left", pad=3, fontsize=6.9)
    ds, dr = draw_d(outer[2, :6], pred, metrics)
    ds.set_title("Digital phenotypes predict held-out\nhuman-perceived quality", loc="left", pad=3, fontsize=6.9)
    dr.set_title("Residuals", loc="left", pad=3, fontsize=6.9)
    es, eb, ei = draw_e(outer[2, 7:], sv, X, importance)
    es.set_title("Model-based phenotype\ncontributions", loc="left", pad=3, fontsize=6.9)
    eb.set_title("Global\nimportance", loc="left", pad=3, fontsize=6.9)
    fig.subplots_adjust(left=0.055, right=0.992, top=0.986, bottom=0.06)
    save_all(fig, MAIN / "Figure3_NC_phenotype_perception_TNR_20261017")
    plt.close(fig)


def main():
    d = pd.read_csv(DATA, encoding="utf-8-sig")
    assert len(d) == 518
    assert d["Extrusion_success_new"].eq(1).all()
    assert d[FEATURES+[TARGET]].notna().all().all()
    assert d["split"].value_counts().to_dict() == {"train": 419, "test": 99}
    for p in (DATA, MODEL, PCA_CSV, PCA_VAR_CSV, ASSOC_CSV, BETA_CSV, PRED_CSV, IMP_CSV):
        shutil.copy2(p, SRC / p.name)

    pca = pd.read_csv(PCA_CSV, encoding="utf-8-sig")
    var = pd.read_csv(PCA_VAR_CSV, encoding="utf-8-sig")
    assoc = pd.read_csv(ASSOC_CSV, encoding="utf-8-sig")
    betas = pd.read_csv(BETA_CSV, encoding="utf-8-sig")
    pred = pd.read_csv(PRED_CSV, encoding="utf-8-sig")
    importance = pd.read_csv(IMP_CSV, encoding="utf-8-sig")
    metrics = {
        "R2": float(r2_score(pred[TARGET], pred["prediction"])),
        "RMSE": float(mean_squared_error(pred[TARGET], pred["prediction"])**0.5),
        "MAE": float(mean_absolute_error(pred[TARGET], pred["prediction"])),
    }
    assert abs(metrics["R2"] - 0.81747867) < 1e-6

    # Genuine model-agnostic permutation SHAP values for the frozen HGB pipeline.
    model = joblib.load(MODEL)
    train = d.loc[d["split"].eq("train"), FEATURES]
    test = d.loc[d["split"].eq("test"), FEATURES]
    background = shap.sample(train, 50, random_state=SEED)
    explainer = shap.Explainer(model.predict, background, algorithm="permutation")
    explanation = explainer(test, max_evals=31, silent=True)
    shap_values = np.asarray(explanation.values)
    assert shap_values.shape == (99, 5)
    shap_df = pd.DataFrame(shap_values, columns=[f"SHAP_{x}" for x in DIMS])
    shap_df.insert(0, "Attempt_ID", d.loc[d["split"].eq("test"), "Attempt_ID"].to_numpy())
    for j, dim in enumerate(DIMS):
        shap_df[f"VALUE_{dim}"] = test.iloc[:, j].to_numpy()
    shap_df.to_csv(DATA_OUT / "Figure3e_model_agnostic_SHAP_fixed_test_99.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"dimension": DIMS, "mean_abs_SHAP": np.abs(shap_values).mean(axis=0)}).sort_values(
        "mean_abs_SHAP", ascending=False
    ).to_csv(DATA_OUT / "Figure3e_SHAP_global_importance.csv", index=False, encoding="utf-8-sig")

    examples = d.set_index("Sample_ID_norm").loc[["ms1", "ms2", "ms3", "sp57", "sp58", "sp59"], FEATURES+[TARGET]]
    examples.to_csv(DATA_OUT / "Figure3a_display_rows.csv", encoding="utf-8-sig")
    for p in (PCA_CSV, PCA_VAR_CSV, ASSOC_CSV, BETA_CSV, PRED_CSV, IMP_CSV):
        shutil.copy2(p, DATA_OUT / p.name)

    make_panel_a(d)
    make_panel_b(pca, var)
    make_panel_c(assoc, betas)
    make_panel_d(pred, metrics)
    make_panel_e(shap_values, test, importance)
    make_composite(d, pca, var, assoc, betas, pred, metrics, shap_values, test, importance)

    manifest = []
    for p in sorted(RUN.rglob("*")):
        if p.is_file() and p.parent != QC:
            manifest.append({"relative_path": str(p.relative_to(RUN)), "bytes": p.stat().st_size, "sha256": sha256(p)})
    pd.DataFrame(manifest).to_csv(QC / "SHA256_MANIFEST.csv", index=False, encoding="utf-8-sig")
    qc = {
        "status": "BUILT_PENDING_VISUAL_QC",
        "source_population": 518,
        "training_n": 419,
        "independent_test_n": 99,
        "excluded_non_extrudable_n": 72,
        "model": "HistGradientBoostingRegressor",
        "fixed_test_metrics": metrics,
        "interpretation": "model-agnostic permutation SHAP on frozen independent test set",
        "font_requested": "Times New Roman",
        "figure_width_mm": 183.0,
        "exports": ["SVG", "PDF", "PNG 600 dpi", "TIFF 600 dpi"],
    }
    (QC / "BUILD_QC.json").write_text(json.dumps(qc, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
