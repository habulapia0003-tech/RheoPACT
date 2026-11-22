from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from matplotlib.lines import Line2D
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT / "external_data" / "source_tree"
DATA = ROOT / "figure4_state_space_131072_v1_20261005" / "source_data"
OUT = ROOT / "supplementary_figure15_rebuilt_540_20261026"
OUT.mkdir(parents=True, exist_ok=True)

MEASURED = DATA / "measured_conditions_540_projection.csv"
ANCHORS = DATA / "measured_anchor_states.csv"
FEATURES = [
    "log10_G_prime_Pa",
    "log10_G_double_prime_Pa",
    "log10_eta10_Pa_s",
    "log10_1_plus_tau_y_Pa",
    "log10_K_Pa_s_n",
    "flow_index_n",
]
FEATURE_LABELS = [r"log$_{10}$ $G'$", r"log$_{10}$ $G''$", r"log$_{10}$ $\eta_{10}$", r"log$_{10}$ (1 + $\tau_0$)", r"log$_{10}$ $K$", r"$n$"]
REGIONS = ["Low elasticity", "Compatible / balanced", "High elastic recovery"]
REGION_LABELS = ["Low\nelasticity", "Compatible /\nbalanced", "High elastic\nrecovery"]
REGION_COLORS = ["#1768B3", "#16844A", "#D9470B"]


def main():
    measured = pd.read_csv(MEASURED)
    anchors = pd.read_csv(ANCHORS).set_index("State").loc[REGIONS].reset_index()
    assert len(measured) == 540
    assert measured["Rheology_condition_ID"].is_unique
    assert len(anchors) == 3

    # Descriptive Euclidean distance in the same frozen two-component projection.
    mxy = measured[["PC1", "PC2"]].to_numpy(float)
    axy = anchors[["PC1", "PC2"]].to_numpy(float)
    distances = np.sqrt(((mxy[:, None, :] - axy[None, :, :]) ** 2).sum(axis=2))
    nearest_distance = distances.min(axis=1)
    nearest_anchor = distances.argmin(axis=1)

    region_counts = (
        measured["anchor_defined_region_recalculated"]
        .value_counts()
        .reindex(REGIONS, fill_value=0)
    )

    # Z-scores are referenced to the 540-condition measured library.
    mu = measured[FEATURES].mean().to_numpy(float)
    sd = measured[FEATURES].std(ddof=0).to_numpy(float)
    anchor_z = (anchors[FEATURES].to_numpy(float) - mu) / sd

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8.5,
            "axes.linewidth": 0.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
        }
    )

    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.55))
    fig.subplots_adjust(left=0.085, right=0.94, bottom=0.17, top=0.82, wspace=0.34, hspace=0.62)
    axa, axb, axc, axd = axs.ravel()

    # a: all measured conditions, coloured only by the frozen anchor-defined regions.
    for i, region in enumerate(REGIONS):
        subset = measured[measured["anchor_defined_region_recalculated"] == region]
        axa.scatter(subset["PC1"], subset["PC2"], s=12, alpha=0.58, color=REGION_COLORS[i], edgecolors="none", label=region)
    for i, row in anchors.iterrows():
        axa.scatter(row.PC1, row.PC2, marker="*", s=140, color=REGION_COLORS[i], edgecolor="white", linewidth=0.9, zorder=5)
    axa.set_title("a  Measured conditions and current anchors", loc="left", fontweight="bold", pad=7)
    axa.set_xlabel("Physical state coordinate 1 (PCA, 70.6%)", fontweight="bold")
    axa.set_ylabel("Physical state coordinate 2 (PCA, 18.6%)", fontweight="bold")
    axa.grid(color="#DCE3E8", linewidth=0.55, alpha=0.85)
    handles = [Line2D([0], [0], marker="o", ls="", ms=4.8, color=REGION_COLORS[i], label=REGIONS[i]) for i in range(3)]
    handles.append(Line2D([0], [0], marker="*", ls="", ms=8, color="#6A6F73", label="Measured anchors"))
    axa.legend(handles=handles, ncol=1, fontsize=6.3, frameon=False, loc="upper left", handletextpad=0.35, columnspacing=0.8)

    # b: exact measured-condition counts in the anchor-defined regions.
    bars = axb.bar(np.arange(3), region_counts.to_numpy(), color=REGION_COLORS, width=0.66)
    for bar, value in zip(bars, region_counts.to_numpy()):
        axb.text(bar.get_x() + bar.get_width() / 2, value + 5, f"{int(value)}", ha="center", va="bottom", fontweight="bold")
    axb.set_xticks(np.arange(3), REGION_LABELS)
    axb.set_ylim(0, max(region_counts) * 1.18)
    axb.set_title("b  Coverage of anchor-defined regions", loc="left", fontweight="bold", pad=7)
    axb.set_ylabel("Measured conditions", fontweight="bold")
    axb.grid(axis="y", color="#DCE3E8", linewidth=0.55)

    # c: distance to the nearest of the three measured anchors.
    axc.hist(nearest_distance, bins=24, color="#5B93B4", edgecolor="white", linewidth=0.4)
    med = float(np.median(nearest_distance))
    axc.axvline(med, color="#D84A43", linestyle="--", linewidth=1.2)
    axc.text(0.97, 0.92, f"median = {med:.2f}", transform=axc.transAxes, ha="right", va="top", fontweight="bold", color="#7A2A26")
    axc.set_title("c  Distance to the nearest measured anchor", loc="left", fontweight="bold", pad=7)
    axc.set_xlabel("Euclidean distance in frozen PCA space", fontweight="bold")
    axc.set_ylabel("Measured conditions", fontweight="bold")
    axc.grid(axis="y", color="#DCE3E8", linewidth=0.55)

    # d: current three-anchor rheological fingerprints relative to the measured library.
    vmax = max(2.0, float(np.nanmax(np.abs(anchor_z))))
    im = axd.imshow(anchor_z, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axd.set_yticks(np.arange(3), REGION_LABELS)
    axd.set_xticks(np.arange(len(FEATURES)), FEATURE_LABELS, rotation=18, ha="right")
    axd.set_title("d  Anchor rheological fingerprints", loc="left", fontweight="bold", pad=7)
    axd.set_xlabel("Standardized rheological descriptor", fontweight="bold")
    for i in range(anchor_z.shape[0]):
        for j in range(anchor_z.shape[1]):
            color = "white" if abs(anchor_z[i, j]) > 0.55 * vmax else "#263238"
            axd.text(j, i, f"{anchor_z[i, j]:.1f}", ha="center", va="center", fontsize=6.6, color=color, fontweight="bold")
    cbar = fig.colorbar(im, ax=axd, fraction=0.046, pad=0.035)
    cbar.set_label("z-score", fontweight="bold")

    fig.suptitle("Measured rheological state-space coverage and anchor definition", y=0.965, fontsize=13, fontweight="bold")
    fig.text(0.5, 0.905, "540 unique measured conditions; three experimentally measured anchors in the frozen six-feature coordinate system", ha="center", fontsize=8.6, color="#58636B")
    fig.text(0.085, 0.025, "Distances and z-scores are descriptive summaries of the frozen measured-state coordinate system; no printing outcomes are used.", fontsize=6.8, color="#626D75")

    stem = OUT / "Supplementary_Figure15_540"
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=600, facecolor="white")
    plt.close(fig)
    with Image.open(stem.with_suffix(".png")) as image:
        rgb = np.asarray(image.convert("RGB"))
    tifffile.imwrite(stem.with_suffix(".tiff"), rgb, photometric="rgb", compression="deflate", resolution=(600, 600), resolutionunit="INCH")

    pd.DataFrame(
        {
            "Rheology_condition_ID": measured["Rheology_condition_ID"],
            "nearest_anchor": [REGIONS[i] for i in nearest_anchor],
            "nearest_anchor_distance_PCA": nearest_distance,
            "anchor_defined_region": measured["anchor_defined_region_recalculated"],
        }
    ).to_csv(OUT / "Supplementary_Figure15_source_data.csv", index=False)
    pd.DataFrame({"anchor_defined_region": REGIONS, "measured_n": region_counts.to_numpy()}).to_csv(OUT / "Supplementary_Figure15_region_counts.csv", index=False)
    print({"measured_n": len(measured), "region_counts": region_counts.to_dict(), "median_nearest_anchor_distance": med})


if __name__ == "__main__":
    main()
