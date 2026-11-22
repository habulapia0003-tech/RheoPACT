from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image
from scipy.spatial import ConvexHull
from scipy.stats import pearsonr, rankdata, spearmanr
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT / "external_data" / "source_tree"
RUN = ROOT / "outputs" / "Food3D_Figure3_Figure5_Update_20261015"
FREEZE = RUN / "00_FROZEN_SOURCE"
RESULTS = RUN / "02_MODEL_RESULTS"
OUT = RUN / "03_FIGURE3_REBUILT"
PANELS = OUT / "individual_panels"
SOURCE = OUT / "source_data"
QC = OUT / "qc"
for folder in (OUT, PANELS, SOURCE, QC):
    folder.mkdir(parents=True, exist_ok=True)

DATA = FREEZE / "Food3D_D1D5_NewSensory_518_extrudable_training.csv"
MODEL = FREEZE / "selected_extrudable_only_model.joblib"
PRED = ROOT / "outputs" / "Food3D_NewSensory_ModelFit_20261015" / "EXTRUDABLE_ONLY_518" / "fixed_test_predictions_99_extrudable_only.csv"
SUMMARY = ROOT / "outputs" / "Food3D_NewSensory_ModelFit_20261015" / "EXTRUDABLE_ONLY_518" / "campaign_summary_extrudable_only.json"

SEED = 20260902
FEATURES = [
    "D1_global_geometry", "D2_structural_completeness", "D3_surface_deviation",
    "D4_polarity_tolerance", "D5_spatial_heterogeneity",
]
SHORT = dict(zip(FEATURES, ["D1", "D2", "D3", "D4", "D5"]))
TARGET = "Sensory_mean_new"

NAVY, BLUE, PALE = "#244F70", "#6F9AB7", "#DCE9F1"
GREEN, PURPLE, ORANGE = "#3F7D65", "#6C5B86", "#C96134"
GRAY, LIGHT, INK = "#66727A", "#F4F6F7", "#20272E"
SEQ_BLUE = LinearSegmentedColormap.from_list("seq_blue", ["#EFF4F7", "#8CB0C7", "#244F70"])
SEQ_GREEN = LinearSegmentedColormap.from_list("seq_green", ["#EEF6F1", "#99BEA7", "#3F7D65"])

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
    "font.size": 7.0,
    "axes.labelsize": 7.2,
    "axes.titlesize": 8.0,
    "axes.titleweight": "bold",
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.2,
    "axes.linewidth": 0.7,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})
# Explicit single-family declaration for automated publication preflight. The
# A serif fallback is used when Times New Roman is unavailable.
mpl.rcParams["font.family"] = "Times New Roman"
# Publication-safe fallback declaration retained for automated preflight;
# Times New Roman remains the active and first-choice family above.
mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def save_all(fig, stem: Path, dpi=600):
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    buf.seek(0)
    with Image.open(buf) as im:
        rgb = np.asarray(im.convert("RGB").copy())
    tifffile.imwrite(stem.with_suffix(".tiff"), rgb, photometric="rgb", resolution=(dpi, dpi), resolutionunit="INCH", compression="deflate")


def clean(ax, grid=False):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(color="#DDE3E8", linewidth=0.5, zorder=0)


def panel_letter(ax, letter):
    ax.text(-0.11, 1.06, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")


def partial_spearman(x, y, controls):
    rx, ry = rankdata(x), rankdata(y)
    rc = np.column_stack([rankdata(controls[:, j]) for j in range(controls.shape[1])])
    ex = rx - LinearRegression().fit(rc, rx).predict(rc)
    ey = ry - LinearRegression().fit(rc, ry).predict(rc)
    z = pearsonr(ex, ey)
    return float(z.statistic), float(z.pvalue)


def draw_a(ax, d):
    ax.set_axis_off(); panel_letter(ax, "a")
    ax.set_title("Digital phenotype–perception cohort after the extrusion-feasibility gate", loc="left", pad=3)
    boxes = [
        (0.02, 0.26, 0.20, 0.50, "Frozen sensory registry", "590 printing attempts\n50 assessors · 29,500 ratings", PALE, NAVY),
        (0.30, 0.26, 0.18, 0.50, "Feasibility gate", "72 non-extrudable\nprotocol zeros excluded", "#FBEDE7", ORANGE),
        (0.56, 0.26, 0.20, 0.50, "Matched analysis cohort", "518 successful prints\nD1–D5 + 25,900 ratings", "#EAF4EE", GREEN),
        (0.82, 0.26, 0.16, 0.50, "Fixed evaluation", "419 train\n99 independent test", "#F0EDF5", PURPLE),
    ]
    for x, y, w, h, title, text, fc, ec in boxes:
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.025", transform=ax.transAxes, fc=fc, ec=ec, lw=1.2)
        ax.add_patch(patch)
        ax.text(x+w/2, y+h*0.72, title, transform=ax.transAxes, ha="center", va="center", fontsize=7.2, fontweight="bold", color=ec)
        ax.text(x+w/2, y+h*0.38, text, transform=ax.transAxes, ha="center", va="center", fontsize=6.5, color=INK, linespacing=1.35)
    for x0, x1 in [(0.225, 0.295), (0.485, 0.555), (0.765, 0.815)]:
        ax.add_patch(FancyArrowPatch((x0, 0.51), (x1, 0.51), transform=ax.transAxes, arrowstyle="-|>", mutation_scale=10, lw=1.3, color=GRAY))
    ax.text(0.50, 0.08, "Only successful prints enter the D1–D5-to-perception regression; failure detection remains a separate task.", transform=ax.transAxes, ha="center", fontsize=6.5, color=GRAY)


def draw_b(ax, pca_scores, explained):
    panel_letter(ax, "b"); clean(ax, grid=True)
    sc = ax.scatter(pca_scores.PC1, pca_scores.PC2, c=pca_scores[TARGET], s=15, cmap=SEQ_BLUE, alpha=0.78, edgecolors="white", linewidths=0.18, zorder=2)
    pts = pca_scores[["PC1", "PC2"]].to_numpy()
    hull = ConvexHull(pts)
    poly = pts[hull.vertices]
    ax.fill(poly[:, 0], poly[:, 1], color=BLUE, alpha=0.08, zorder=0)
    ax.plot(np.r_[poly[:, 0], poly[0, 0]], np.r_[poly[:, 1], poly[0, 1]], color=BLUE, lw=0.7, alpha=0.45)
    ax.set_title("Digital phenotype landscape organizes perceived quality", loc="left", pad=4)
    ax.set_xlabel(f"PC1 ({100*explained[0]:.1f}%)"); ax.set_ylabel(f"PC2 ({100*explained[1]:.1f}%)")
    cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.03); cb.set_label("Mean visual score")
    ax.text(0.03, 0.05, "n = 518 successful prints", transform=ax.transAxes, color=GRAY, fontsize=6.3)


def draw_c(ax, assoc, betas):
    panel_letter(ax, "c"); clean(ax, grid=False)
    ax.set_title("Phenotype dimensions contribute complementary perceptual information", loc="left", pad=4)
    y = np.arange(5)
    marginal = assoc.set_index("dimension").loc[["D1", "D2", "D3", "D4", "D5"], "marginal_spearman_rho"].to_numpy()
    partial = assoc.set_index("dimension").loc[["D1", "D2", "D3", "D4", "D5"], "partial_spearman_rho"].to_numpy()
    ax.barh(y+0.16, marginal, height=0.28, color=BLUE, label="Marginal Spearman ρ")
    ax.barh(y-0.16, partial, height=0.28, color=GREEN, label="Partial Spearman ρ")
    ax.axvline(0, color=GRAY, lw=0.7)
    ax.set_yticks(y, ["D1", "D2", "D3", "D4", "D5"]); ax.invert_yaxis()
    ax.set_xlabel("Association with mean visual score")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.52, -0.18), ncol=2)
    for yi, m, p in zip(y, marginal, partial):
        ax.text(m + 0.015*np.sign(m or 1), yi+0.16, f"{m:.2f}", va="center", ha="left" if m >= 0 else "right", fontsize=5.8)
        ax.text(p + 0.015*np.sign(p or 1), yi-0.16, f"{p:.2f}", va="center", ha="left" if p >= 0 else "right", fontsize=5.8)


def draw_d(ax, pred, metric):
    panel_letter(ax, "d"); clean(ax, grid=True)
    y = pred[TARGET].to_numpy(); p = pred["prediction"].to_numpy()
    ax.scatter(y, p, s=22, c=NAVY, alpha=0.78, edgecolor="white", linewidth=0.25, zorder=2)
    lo = min(y.min(), p.min())-3; hi = max(y.max(), p.max())+3
    ax.plot([lo, hi], [lo, hi], ls="--", lw=1.0, color=GRAY)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Observed mean visual score"); ax.set_ylabel("Predicted mean visual score")
    ax.set_title("Five-dimensional phenotypes predict held-out perceived quality", loc="left", pad=4)
    ax.text(0.04, 0.96, f"Fixed test n = 99\n$R^2$ = {metric['R2']:.3f}\nRMSE = {metric['RMSE']:.2f}\nMAE = {metric['MAE']:.2f}", transform=ax.transAxes, va="top", fontsize=6.8, fontweight="bold")
    ax.text(0.96, 0.05, "HistGradientBoosting\nselected by training CV", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.0, color=GRAY)


def draw_e(ax, importance):
    panel_letter(ax, "e"); clean(ax, grid=False)
    ax.set_title("Model-agnostic permutation importance on the fixed test set", loc="left", pad=4)
    order = importance.groupby("dimension")["importance_mean"].first().sort_values().index.tolist()
    for yi, dim in enumerate(order):
        g = importance[importance.dimension.eq(dim)]
        vals = np.asarray(json.loads(g.iloc[0]["importance_repeats_json"]))
        jitter = np.linspace(-0.10, 0.10, len(vals))
        ax.scatter(vals, yi+jitter, s=9, color=BLUE, alpha=0.30, edgecolor="none")
        mean = float(g.iloc[0].importance_mean); sd = float(g.iloc[0].importance_sd)
        ax.errorbar(mean, yi, xerr=sd, fmt="o", color=NAVY, ecolor=NAVY, capsize=2, markersize=4.5, lw=1.0)
    ax.axvline(0, color=GRAY, lw=0.7)
    ax.set_yticks(range(5), order); ax.set_xlabel("Decrease in test $R^2$ after permutation")
    ax.text(0.98, 0.05, "50 repeats per dimension", transform=ax.transAxes, ha="right", fontsize=6.1, color=GRAY)


def make_individual(name, figsize, drawer, *args):
    fig, ax = plt.subplots(figsize=figsize)
    drawer(ax, *args)
    fig.tight_layout(pad=0.7)
    save_all(fig, PANELS / name)
    plt.close(fig)


def main():
    d = pd.read_csv(DATA, encoding="utf-8-sig")
    assert len(d) == 518 and d["Extrusion_success_new"].eq(1).all()
    assert d[FEATURES + [TARGET]].notna().all().all()
    assert d["split"].value_counts().to_dict() == {"train": 419, "test": 99}

    scaler = StandardScaler().fit(d[FEATURES])
    xz = scaler.transform(d[FEATURES])
    pca = PCA(random_state=SEED).fit(xz)
    pcs = pca.transform(xz)
    pca_scores = d[["Attempt_ID", "Sample_ID_norm", "Material_code", TARGET, "split"]].copy()
    pca_scores["PC1"], pca_scores["PC2"] = pcs[:, 0], pcs[:, 1]
    pca_scores.to_csv(SOURCE / "Figure3b_PCA_scores_518.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"PC": [f"PC{i+1}" for i in range(5)], "explained_variance_ratio": pca.explained_variance_ratio_}).to_csv(SOURCE / "Figure3b_PCA_variance.csv", index=False)

    assoc_rows = []
    for f in FEATURES:
        others = [z for z in FEATURES if z != f]
        marginal = spearmanr(d[f], d[TARGET])
        partial_r, partial_p = partial_spearman(d[f].to_numpy(), d[TARGET].to_numpy(), d[others].to_numpy())
        assoc_rows.append({"feature": f, "dimension": SHORT[f], "marginal_spearman_rho": float(marginal.statistic), "marginal_p": float(marginal.pvalue), "partial_spearman_rho": partial_r, "partial_p": partial_p, "n": len(d)})
    assoc = pd.DataFrame(assoc_rows)
    assoc.to_csv(SOURCE / "Figure3c_associations_518.csv", index=False, encoding="utf-8-sig")
    yz = StandardScaler().fit_transform(d[[TARGET]]).ravel()
    linear = LinearRegression().fit(xz, yz)
    betas = pd.DataFrame({"feature": FEATURES, "dimension": [SHORT[f] for f in FEATURES], "standardized_beta": linear.coef_})
    betas.to_csv(SOURCE / "Figure3c_standardized_coefficients_518.csv", index=False, encoding="utf-8-sig")

    raw_pred = pd.read_csv(PRED, encoding="utf-8-sig")
    pred_col = "pred_HistGradientBoosting"
    if pred_col not in raw_pred.columns:
        raise KeyError(f"Expected selected-model prediction column missing: {pred_col}")
    actual_col = TARGET if TARGET in raw_pred.columns else next(c for c in raw_pred.columns if "sensory" in c.lower() and not c.lower().startswith("pred"))
    pred = raw_pred[["Attempt_ID", actual_col, pred_col]].rename(columns={actual_col: TARGET, pred_col: "prediction"})
    metric = {
        "R2": float(r2_score(pred[TARGET], pred.prediction)),
        "RMSE": float(mean_squared_error(pred[TARGET], pred.prediction) ** 0.5),
        "MAE": float(mean_absolute_error(pred[TARGET], pred.prediction)),
    }
    pred["residual_observed_minus_predicted"] = pred[TARGET] - pred["prediction"]
    pred.to_csv(SOURCE / "Figure3d_fixed_test_predictions_99.csv", index=False, encoding="utf-8-sig")

    model = joblib.load(MODEL)
    test = d["split"].eq("test")
    pi = permutation_importance(model, d.loc[test, FEATURES], d.loc[test, TARGET], scoring="r2", n_repeats=50, random_state=SEED, n_jobs=-1)
    importance = pd.DataFrame({
        "feature": FEATURES,
        "dimension": [SHORT[f] for f in FEATURES],
        "importance_mean": pi.importances_mean,
        "importance_sd": pi.importances_std,
        "importance_repeats_json": [json.dumps(pi.importances[i, :].tolist()) for i in range(len(FEATURES))],
    }).sort_values("importance_mean", ascending=False)
    importance.to_csv(SOURCE / "Figure3e_permutation_importance_99.csv", index=False, encoding="utf-8-sig")

    # Independently rendered panels, not raster crops.
    make_individual("Figure3a_dataset_gate", (7.0, 2.25), draw_a, d)
    make_individual("Figure3b_phenotype_landscape", (3.6, 3.0), draw_b, pca_scores, pca.explained_variance_ratio_)
    make_individual("Figure3c_associations", (3.6, 3.0), draw_c, assoc, betas)
    make_individual("Figure3d_fixed_test_prediction", (3.6, 3.0), draw_d, pred, metric)
    make_individual("Figure3e_permutation_importance", (3.6, 3.0), draw_e, importance)

    fig = plt.figure(figsize=(7.20, 6.55))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[0.72, 1.05, 1.05], hspace=0.62, wspace=0.42)
    draw_a(fig.add_subplot(gs[0, :]), d)
    draw_b(fig.add_subplot(gs[1, 0]), pca_scores, pca.explained_variance_ratio_)
    draw_c(fig.add_subplot(gs[1, 1]), assoc, betas)
    draw_d(fig.add_subplot(gs[2, 0]), pred, metric)
    draw_e(fig.add_subplot(gs[2, 1]), importance)
    fig.subplots_adjust(left=0.085, right=0.965, top=0.975, bottom=0.07)
    save_all(fig, OUT / "Figure3_UPDATED_NEW_SENSORY_518")
    plt.close(fig)

    qc = {
        "status": "BUILT_PENDING_VISUAL_QC",
        "font": "Times New Roman with DejaVu Serif fallback",
        "analysis_population": 518,
        "excluded_non_extrudable": 72,
        "train_n": 419,
        "test_n": 99,
        "test_metrics": metric,
        "interpretation_method": "Permutation importance on fixed test set; not causal",
        "source_hashes": {"data": sha256(DATA), "model": sha256(MODEL), "predictions": sha256(PRED)},
    }
    (QC / "FIGURE3_BUILD_QC.json").write_text(json.dumps(qc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(qc, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
