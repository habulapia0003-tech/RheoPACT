from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT / "external_data" / "source_tree"
DATA = ROOT / "fig02_pipeline" / "outputs" / "chocolate_final" / "turtle"
OUT = ROOT / "outputs" / "Food3D_FINAL_HIGHRES_FIGURES_20261026" / "01_MAIN_FIGURES" / "Figure2" / "Panels"
STEM = OUT / "Figure2d_TNR"

HEADER = ["#DDEBFA", "#F8E8CE", "#DDF1E6", "#E9E0F5", "#F8E1E5"]
EDGE = ["#93BCE3", "#E6BE7D", "#8DCEAB", "#B8A0D8", "#E4A7B1"]
ACCENT = ["#1768B3", "#C47713", "#16844A", "#7250A5", "#C94C66"]
BODY = ["#F8FBFE", "#FEFBF6", "#F7FCF9", "#FBF9FD", "#FEF8F9"]


def read_ply_xyz(path: Path) -> np.ndarray:
    type_map = {
        "char": "i1", "uchar": "u1", "short": "<i2", "ushort": "<u2",
        "int": "<i4", "uint": "<u4", "float": "<f4", "double": "<f8",
        "int8": "i1", "uint8": "u1", "int16": "<i2", "uint16": "<u2",
        "int32": "<i4", "uint32": "<u4", "float32": "<f4", "float64": "<f8",
    }
    with path.open("rb") as handle:
        header = []
        while True:
            line = handle.readline().decode("ascii").strip()
            header.append(line)
            if line == "end_header":
                break
        fmt = next(line.split()[1] for line in header if line.startswith("format "))
        count = int(next(line.split()[2] for line in header if line.startswith("element vertex ")))
        props = []
        in_vertex = False
        for line in header:
            if line.startswith("element "):
                in_vertex = line.startswith("element vertex ")
            elif in_vertex and line.startswith("property ") and " list " not in line:
                _, typ, name = line.split()
                props.append((name, type_map[typ]))
        if fmt == "ascii":
            array = np.loadtxt(handle, max_rows=count)
            names = [prop[0] for prop in props]
            return array[:, [names.index("x"), names.index("y"), names.index("z")]]
        array = np.fromfile(handle, dtype=np.dtype(props), count=count)
        return np.column_stack([array["x"], array["y"], array["z"]]).astype(float, copy=False)


def clean(axis):
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def card(axis, index: int, code: str, title: str, subtitle: str, question: str):
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    clean(axis)
    axis.add_patch(
        FancyBboxPatch(
            (0.008, 0.008), 0.984, 0.984,
            boxstyle="round,pad=0.008,rounding_size=0.025",
            facecolor=BODY[index], edgecolor=EDGE[index], linewidth=1.0,
        )
    )
    axis.add_patch(
        FancyBboxPatch(
            (0.008, 0.775), 0.984, 0.217,
            boxstyle="round,pad=0.008,rounding_size=0.025",
            facecolor=HEADER[index], edgecolor="none",
        )
    )
    axis.text(0.5, 0.955, code, ha="center", va="center", fontsize=7.5, fontweight="bold")
    axis.text(0.5, 0.905, title, ha="center", va="center", fontsize=7.35, fontweight="bold", linespacing=0.95)
    axis.text(0.5, 0.812, subtitle, ha="center", va="center", fontsize=6.25, linespacing=0.98)
    axis.text(0.5, 0.700, question, ha="center", va="center", fontsize=6.55, linespacing=1.05)


def bullet_text(axis, lines: list[str]):
    y = 0.060 + 0.048 * (len(lines) - 1)
    for line in lines:
        axis.text(0.065, y, u"▪", ha="left", va="center", fontsize=6.4, color="#111111")
        axis.text(0.115, y, line, ha="left", va="center", fontsize=6.05)
        y -= 0.048


def inset(axis, rect):
    inner = axis.inset_axes(rect)
    clean(inner)
    return inner


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    deviations = np.load(DATA / "pointwise_deviation.npz")
    points_raw = deviations["points"].astype(float)
    signed = deviations["signed_deviation_mm"].astype(float)
    reference_raw = read_ply_xyz(DATA / "reference_sampled.ply")

    # Rotate the registered XY projection for a horizontal turtle silhouette.
    points = np.column_stack([points_raw[:, 1], -points_raw[:, 0], points_raw[:, 2]])
    reference = np.column_stack([reference_raw[:, 1], -reference_raw[:, 0], reference_raw[:, 2]])

    # Use every available point. Dense marks are rasterized inside vector containers.
    x_mid = float(np.median(points[:, 0]))
    extent = (
        min(reference[:, 0].min(), points[:, 0].min()),
        max(reference[:, 0].max(), points[:, 0].max()),
        min(reference[:, 1].min(), points[:, 1].min()),
        max(reference[:, 1].max(), points[:, 1].max()),
    )

    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
        }
    )

    fig = plt.figure(figsize=(7.2, 2.78), facecolor="white")
    grid = fig.add_gridspec(1, 5, left=0.012, right=0.993, bottom=0.035, top=0.84, wspace=0.025)
    axes = [fig.add_subplot(grid[0, i]) for i in range(5)]
    fig.text(0.006, 0.965, "d", fontsize=13, fontweight="bold", ha="left", va="top")
    fig.text(0.033, 0.95, "Five-domain structural phenotype", fontsize=11.2, fontweight="bold", ha="left", va="top")

    # D1 — global geometry.
    ax = axes[0]
    card(ax, 0, "D1", "Global geometric fidelity", "Overall dimensions\nand envelope", "Is the overall shape and\nscale retained?")
    art = inset(ax, [0.10, 0.25, 0.80, 0.34])
    art.scatter(reference[:, 0], reference[:, 1], s=0.05, color="#82B7C4", alpha=0.63, linewidths=0, rasterized=True)
    art.set_xlim(extent[0], extent[1]); art.set_ylim(extent[2], extent[3]); art.set_aspect("equal")
    art.add_patch(FancyArrowPatch((0.05, 0.03), (0.95, 0.03), transform=art.transAxes, arrowstyle="<->", mutation_scale=7, color=ACCENT[0], lw=0.8))
    art.add_patch(FancyArrowPatch((0.04, 0.08), (0.04, 0.92), transform=art.transAxes, arrowstyle="<->", mutation_scale=7, color=ACCENT[0], lw=0.8))
    art.text(0.5, -0.01, "L / W", transform=art.transAxes, ha="center", va="top", fontsize=6.5, fontweight="bold", color=ACCENT[0])
    art.text(-0.01, 0.50, "H", transform=art.transAxes, ha="right", va="center", fontsize=6.5, fontweight="bold", color=ACCENT[0])
    bullet_text(ax, ["Dimensional deviation", "Envelope fidelity"])

    # D2 — bidirectional structural coverage.
    ax = axes[1]
    card(ax, 1, "D2", "Structural completeness", "Presence of designed\nfeatures", "Are the designed structural\nfeatures recovered?")
    art = inset(ax, [0.08, 0.25, 0.84, 0.34])
    left = reference[:, 0] <= x_mid
    right = points[:, 0] > x_mid
    art.scatter(reference[left, 0], reference[left, 1], s=0.06, color="#6FAAB7", alpha=0.68, linewidths=0, rasterized=True)
    art.scatter(points[right, 0], points[right, 1], s=0.06, color="#D9C5A3", alpha=0.68, linewidths=0, rasterized=True)
    art.axvline(x_mid, color="#8B8B8B", lw=0.55, ls="--")
    art.set_xlim(extent[0], extent[1]); art.set_ylim(extent[2], extent[3]); art.set_aspect("equal")
    art.text(0.25, 0.03, "Design", transform=art.transAxes, ha="center", fontsize=5.8, color="#397787")
    art.text(0.75, 0.03, "Printed", transform=art.transAxes, ha="center", fontsize=5.8, color="#8F6F42")
    bullet_text(ax, ["Bidirectional surface coverage", "Feature retention"])

    # D3 — local deviation magnitude.
    ax = axes[2]
    card(ax, 2, "D3", "Surface-deviation\nmagnitude", "Local geometric error", "How large are local\nsurface deviations?")
    art = inset(ax, [0.07, 0.335, 0.86, 0.265])
    sc3 = art.scatter(points[:, 0], points[:, 1], c=np.clip(signed, -0.75, 0.75), s=0.055, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-0.75, vcenter=0, vmax=0.75), linewidths=0, rasterized=True)
    art.set_xlim(extent[0], extent[1]); art.set_ylim(extent[2], extent[3]); art.set_aspect("equal")
    cax = ax.inset_axes([0.18, 0.285, 0.64, 0.018])
    cb = fig.colorbar(sc3, cax=cax, orientation="horizontal", ticks=[-0.75, 0, 0.75])
    cb.set_label("Deviation (mm)", fontsize=5.4, labelpad=0.4)
    cb.ax.tick_params(labelsize=5.0, length=1.5, pad=1)
    bullet_text(ax, ["MAE / RMSE / P95", "Local deviation severity"])

    # D4 — signed direction and tolerance.
    ax = axes[3]
    card(ax, 3, "D4", "Deviation polarity\n& tolerance", "Direction and\nacceptable fraction", "Are deviations inward or outward,\nand within tolerance?")
    art = inset(ax, [0.07, 0.335, 0.86, 0.265])
    sc4 = art.scatter(points[:, 0], points[:, 1], c=np.clip(signed, -0.75, 0.75), s=0.055, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-0.75, vcenter=0, vmax=0.75), linewidths=0, rasterized=True)
    art.set_xlim(extent[0], extent[1]); art.set_ylim(extent[2], extent[3]); art.set_aspect("equal")
    art.annotate("Outward", xy=(0.23, 0.62), xytext=(0.02, 0.91), xycoords="axes fraction", textcoords="axes fraction", color="#C5232A", fontsize=5.3, fontweight="bold", arrowprops=dict(arrowstyle="->", color="#C5232A", lw=0.65))
    art.annotate("Inward", xy=(0.78, 0.45), xytext=(0.76, 0.08), xycoords="axes fraction", textcoords="axes fraction", color="#2455C3", fontsize=5.3, fontweight="bold", arrowprops=dict(arrowstyle="->", color="#2455C3", lw=0.65))
    cax = ax.inset_axes([0.18, 0.285, 0.64, 0.018])
    cb = fig.colorbar(sc4, cax=cax, orientation="horizontal", ticks=[-0.75, 0, 0.75])
    cb.set_label("Signed deviation (mm)", fontsize=5.25, labelpad=0.4)
    cb.ax.tick_params(labelsize=5.0, length=1.5, pad=1)
    bullet_text(ax, ["Mean signed deviation", "Positive / negative fractions", "Tolerance compliance"])

    # D5 — spatial localization of absolute deviations.
    ax = axes[4]
    card(ax, 4, "D5", "Spatial heterogeneity\n& localization", "Distribution of defects", "Where are the defects\nconcentrated?")
    art = inset(ax, [0.07, 0.335, 0.86, 0.265])
    abs_dev = np.abs(signed)
    vmax = float(np.quantile(abs_dev, 0.95))
    sc5 = art.scatter(points[:, 0], points[:, 1], c=np.clip(abs_dev, 0, vmax), s=0.055, cmap="YlOrRd", norm=Normalize(vmin=0, vmax=vmax), linewidths=0, rasterized=True)
    art.set_xlim(extent[0], extent[1]); art.set_ylim(extent[2], extent[3]); art.set_aspect("equal")
    cax = ax.inset_axes([0.20, 0.285, 0.60, 0.018])
    cb = fig.colorbar(sc5, cax=cax, orientation="horizontal", ticks=[])
    cb.ax.text(-0.04, -1.7, "Low", transform=cb.ax.transAxes, ha="right", va="center", fontsize=5.2)
    cb.ax.text(1.04, -1.7, "High", transform=cb.ax.transAxes, ha="left", va="center", fontsize=5.2)
    bullet_text(ax, ["Hotspot voxel fraction", "Defect clusters", "Spatial heterogeneity"])

    fig.savefig(STEM.with_suffix(".svg"), facecolor="white")
    fig.savefig(STEM.with_suffix(".pdf"), facecolor="white")
    fig.savefig(STEM.with_suffix(".png"), dpi=600, facecolor="white")
    plt.close(fig)

    with Image.open(STEM.with_suffix(".png")) as image:
        rgb = np.asarray(image.convert("RGB"))
    tifffile.imwrite(STEM.with_suffix(".tiff"), rgb, photometric="rgb", compression="deflate", resolution=(600, 600), resolutionunit="INCH")
    print({"panel": str(STEM), "points_n": int(len(points)), "reference_points_n": int(len(reference)), "outputs": ["svg", "pdf", "png", "tiff"]})


if __name__ == "__main__":
    main()
