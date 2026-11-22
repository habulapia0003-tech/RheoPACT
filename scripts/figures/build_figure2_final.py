from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT / "external_data" / "source_tree"
PACKAGE = ROOT / "outputs" / "Food3D_FINAL_HIGHRES_FIGURES_20261026"
PANEL_DIR = PACKAGE / "01_MAIN_FIGURES" / "Figure2" / "Panels"
WHOLE_DIR = PACKAGE / "01_MAIN_FIGURES" / "Figure2" / "Whole"
SOURCE = ROOT / "fig02_submission_ready" / "source_data" / "Fig02_Source_Data_submission_ready.xlsx"
TURTLE = ROOT / "fig02_pipeline" / "outputs" / "chocolate_final" / "turtle"
RAW_PLY = REPO_ROOT / "external_data" / "pointclouds" / "turtle_raw.ply"
PHOTO = ROOT / "tmp_fig02_ppt_audit" / "pptx" / "ppt" / "media" / "image4.png"

BLUE = "#2774B5"
CYAN = "#2DB5C3"
PALE = "#EDF6FC"
HEADER = "#DCEBF6"
EDGE = "#B8D7EC"
CARD_HEADER = ["#DDEBFA", "#F8E8CE", "#DDF1E6", "#E9E0F5", "#F8E1E5"]
CARD_EDGE = ["#93BCE3", "#E6BE7D", "#8DCEAB", "#B8A0D8", "#E4A7B1"]
CARD_BODY = ["#F8FBFE", "#FEFBF6", "#F7FCF9", "#FBF9FD", "#FEF8F9"]


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
        props, in_vertex = [], False
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


def orient(points: np.ndarray) -> np.ndarray:
    return np.column_stack([points[:, 1], -points[:, 0], points[:, 2]])


def pca_xy(points: np.ndarray) -> np.ndarray:
    centered = points - np.mean(points, axis=0)
    _, _, vt = np.linalg.svd(centered[::max(1, len(centered) // 20000)], full_matrices=False)
    projected = centered @ vt[:2].T
    if np.ptp(projected[:, 1]) > np.ptp(projected[:, 0]):
        projected = projected[:, ::-1]
    return np.column_stack([projected[:, 0], projected[:, 1], np.zeros(len(projected))])


def clean(axis):
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def panel_frame(axis, label: str, title: str):
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    clean(axis)
    axis.add_patch(FancyBboxPatch((0.003, 0.006), 0.994, 0.988,
                                  boxstyle="round,pad=0.002,rounding_size=0.012",
                                  facecolor="#FFFFFF", edgecolor=EDGE, linewidth=0.8))
    axis.add_patch(FancyBboxPatch((0.004, 0.885), 0.992, 0.108,
                                  boxstyle="round,pad=0.002,rounding_size=0.012",
                                  facecolor=HEADER, edgecolor="none"))
    axis.text(0.011, 0.945, label, fontsize=9.0, fontweight="bold", va="center")
    axis.text(0.036, 0.945, title, fontsize=8.4, fontweight="bold", va="center")


def add_arrow(axis, x0, x1, y=0.48):
    axis.add_patch(FancyArrowPatch((x0, y), (x1, y), transform=axis.transAxes,
                                  arrowstyle="-|>", mutation_scale=7, lw=1.0, color=CYAN))


def cloud(axis, points, color, size=0.05, alpha=0.70):
    axis.scatter(points[:, 0], points[:, 1], s=size, c=color, alpha=alpha,
                 linewidths=0, rasterized=True)
    axis.set_aspect("equal")
    clean(axis)


def set_cloud_bounds(axis, arrays):
    xs = np.concatenate([a[:, 0] for a in arrays])
    ys = np.concatenate([a[:, 1] for a in arrays])
    padx = 0.04 * (xs.max() - xs.min())
    pady = 0.04 * (ys.max() - ys.min())
    axis.set_xlim(xs.min() - padx, xs.max() + padx)
    axis.set_ylim(ys.min() - pady, ys.max() + pady)


def draw_panel_a(axis, data):
    panel_frame(axis, "a", "From conventional assessments to 3D digital phenotyping")

    # Conventional assessment block.
    axis.add_patch(FancyBboxPatch((0.012, 0.08), 0.175, 0.77, boxstyle="round,pad=0.004,rounding_size=0.012",
                                  facecolor="#F3F3F3", edgecolor="#E2E2E2", linewidth=0.6))
    axis.add_patch(FancyBboxPatch((0.015, 0.73), 0.169, 0.10, boxstyle="round,pad=0.003,rounding_size=0.010",
                                  facecolor="#E9E9E9", edgecolor="none"))
    axis.text(0.099, 0.78, "Conventional assessments", ha="center", va="center", fontsize=6.7, fontweight="bold")
    ia = axis.inset_axes([0.035, 0.32, 0.105, 0.36])
    cloud(ia, data["reference"], "#D7CDB6", 0.05, 0.70)
    set_cloud_bounds(ia, [data["reference"]])
    axis.add_patch(FancyArrowPatch((0.125, 0.535), (0.163, 0.535), transform=axis.transAxes,
                                  arrowstyle="<->", mutation_scale=6, lw=0.8, color="#111111"))
    axis.add_patch(Circle((0.143, 0.39), 0.012, transform=axis.transAxes,
                          facecolor="none", edgecolor="#111111", linewidth=0.8))
    for y, txt in [(0.25, "Visual inspection"), (0.18, "Manual dimensional measurement"), (0.11, "Qualitative scoring")]:
        axis.add_patch(Rectangle((0.022, y - 0.008), 0.008, 0.016, transform=axis.transAxes,
                                 facecolor="#111111", edgecolor="none"))
        axis.text(0.038, y, txt, fontsize=5.7, va="center")

    # Digital workflow block.
    axis.add_patch(FancyBboxPatch((0.202, 0.08), 0.608, 0.77, boxstyle="round,pad=0.004,rounding_size=0.012",
                                  facecolor="#F8FCFE", edgecolor="#B9DAEF", linewidth=0.7))
    axis.add_patch(FancyBboxPatch((0.205, 0.73), 0.602, 0.10, boxstyle="round,pad=0.003,rounding_size=0.010",
                                  facecolor="#D9EAFB", edgecolor="none"))
    axis.text(0.506, 0.78, "3D digital phenotyping workflow", ha="center", va="center", fontsize=6.8, fontweight="bold", color="#163E7A")
    axis.text(0.632, 0.78, "(objective and quantitative)", ha="left", va="center", fontsize=5.7, fontweight="bold", color="#1D60CD")
    steps = [
        ("Digital model", "reference"), ("Printing", "photo"), ("3D scanning", "scanner"),
        ("Point cloud", "registered"), ("Deviation mapping", "deviation")
    ]
    xstarts = [0.215, 0.333, 0.451, 0.569, 0.687]
    for i, ((label, kind), x0) in enumerate(zip(steps, xstarts)):
        axis.text(x0 + 0.050, 0.68, label, ha="center", va="center", fontsize=5.6, fontweight="bold")
        ia = axis.inset_axes([x0, 0.23, 0.100, 0.39])
        if kind == "reference":
            cloud(ia, data["reference"], "#2D57D4", 0.05, 0.78); set_cloud_bounds(ia, [data["reference"]])
        elif kind == "photo":
            ia.imshow(data["photo"]); clean(ia)
        elif kind == "scanner":
            clean(ia)
            ia.add_patch(Rectangle((0.15, 0.62), 0.70, 0.18, facecolor="#333333", edgecolor="#111111", lw=0.5))
            ia.add_patch(Rectangle((0.26, 0.68), 0.12, 0.07, facecolor="#2E6BFF", edgecolor="none"))
            ia.add_patch(Rectangle((0.62, 0.68), 0.12, 0.07, facecolor="#2E6BFF", edgecolor="none"))
            ia.fill_between([0.28, 0.50], [0.62, 0.05], [0.36, 0.05], color="#3FB7E8", alpha=0.45)
            ia.fill_between([0.64, 0.50], [0.62, 0.05], [0.72, 0.05], color="#3FB7E8", alpha=0.45)
            ia.set_xlim(0, 1); ia.set_ylim(0, 1)
        elif kind == "registered":
            cloud(ia, data["points"], "#6CB5BF", 0.045, 0.72); set_cloud_bounds(ia, [data["points"]])
        else:
            sc = ia.scatter(data["points"][:, 0], data["points"][:, 1], c=np.clip(data["signed"], -0.75, 0.75),
                            s=0.045, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-0.75, vcenter=0, vmax=0.75),
                            linewidths=0, rasterized=True)
            ia.set_aspect("equal"); clean(ia); set_cloud_bounds(ia, [data["points"]])
            cax = axis.inset_axes([x0 + 0.015, 0.18, 0.070, 0.018])
            cb = plt.colorbar(sc, cax=cax, orientation="horizontal", ticks=[-0.75, 0, 0.75])
            cb.ax.tick_params(labelsize=5.0, length=1, pad=0.5)
        if i < 4:
            add_arrow(axis, x0 + 0.102, xstarts[i + 1] - 0.004, 0.43)

    # Multidimensional phenotype block (schematic, not a quantitative radar).
    axis.add_patch(FancyBboxPatch((0.822, 0.08), 0.166, 0.77, boxstyle="round,pad=0.004,rounding_size=0.012",
                                  facecolor="#FCFCFF", edgecolor="#AFA9E8", linewidth=0.7))
    axis.add_patch(FancyBboxPatch((0.825, 0.73), 0.160, 0.10, boxstyle="round,pad=0.003,rounding_size=0.010",
                                  facecolor="#E8E7FB", edgecolor="none"))
    axis.text(0.905, 0.78, "Multidimensional\ndigital phenotype", ha="center", va="center", fontsize=6.6, fontweight="bold")
    ra = axis.inset_axes([0.855, 0.25, 0.100, 0.37], projection="polar")
    angles = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    values = np.ones(5) * 0.72
    ra.plot(np.r_[angles, angles[0]], np.r_[values, values[0]], color="#2367E8", lw=1.1)
    ra.fill(np.r_[angles, angles[0]], np.r_[values, values[0]], color="#7FA7F8", alpha=0.34)
    ra.scatter(angles, values, s=7, color="#1A55D2", zorder=3)
    ra.set_ylim(0, 1); ra.set_yticks([]); ra.set_xticks([])
    ra.grid(color="#B5C8EE", lw=0.5); ra.spines["polar"].set_visible(False)
    axis.text(0.905, 0.655, "D1\nFidelity", ha="center", va="center", fontsize=5.0)
    axis.text(0.968, 0.475, "D2\nCompleteness", ha="center", va="center", fontsize=5.0)
    axis.text(0.945, 0.215, "D3\nDeviation", ha="center", va="center", fontsize=5.0)
    axis.text(0.865, 0.215, "D4\nPolarity", ha="center", va="center", fontsize=5.0)
    axis.text(0.838, 0.475, "D5\nLocalization", ha="center", va="center", fontsize=5.0)
    axis.text(0.905, 0.12, "Schematic D1–D5 representation", ha="center", fontsize=5.0, color="#666666")


def draw_panel_b(axis, data):
    panel_frame(axis, "b", "3D reconstruction and signed-deviation mapping (example: turtle)")
    stages = [
        ("Reference digital model", "reference"), ("Scanned point cloud", "raw"),
        ("Registered point cloud", "registered"), ("Signed surface deviation", "deviation"),
        ("Distribution of signed deviation", "distribution")
    ]
    widths = [0.145, 0.145, 0.145, 0.225, 0.245]
    x = 0.018
    for i, ((title, kind), width) in enumerate(zip(stages, widths)):
        axis.text(x + width / 2, 0.81, title, ha="center", va="center", fontsize=5.9, fontweight="bold")
        ia = axis.inset_axes([x, 0.18, width, 0.56])
        if kind == "reference":
            cloud(ia, data["reference"], "#2788C7", 0.05, 0.75); set_cloud_bounds(ia, [data["reference"]])
            axis.text(x + width / 2, 0.105, f"$N$ = {len(data['reference']):,} points", ha="center", fontsize=5.0)
            axis.text(x + width / 2, 0.060, "(surface sampling)", ha="center", fontsize=5.0)
        elif kind == "raw":
            cloud(ia, data["raw"], "#63C5A4", 0.035, 0.62); set_cloud_bounds(ia, [data["raw"]])
            axis.text(x + width / 2, 0.105, f"$N$ = {len(data['raw']):,} points", ha="center", fontsize=5.0)
            axis.text(x + width / 2, 0.060, "(after filtering)", ha="center", fontsize=5.0)
        elif kind == "registered":
            cloud(ia, data["points"], "#C6A56D", 0.045, 0.70); set_cloud_bounds(ia, [data["points"]])
            axis.text(x + width / 2, 0.105, f"$N$ = {len(data['points']):,} points", ha="center", fontsize=5.0)
            axis.text(x + width / 2, 0.060, "(rigid registration)", ha="center", fontsize=5.0)
        elif kind == "deviation":
            sc = ia.scatter(data["points"][:, 0], data["points"][:, 1], c=np.clip(data["signed"], -0.75, 0.75),
                            s=0.045, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-0.75, vcenter=0, vmax=0.75),
                            linewidths=0, rasterized=True)
            ia.set_aspect("equal"); clean(ia); set_cloud_bounds(ia, [data["points"]])
            ia.annotate("Outward", xy=(0.22, 0.67), xytext=(0.03, 0.93), xycoords="axes fraction", textcoords="axes fraction",
                        color="#D11F2F", fontsize=5.0, fontweight="bold", arrowprops=dict(arrowstyle="->", color="#D11F2F", lw=0.55))
            ia.annotate("Inward", xy=(0.80, 0.48), xytext=(0.78, 0.08), xycoords="axes fraction", textcoords="axes fraction",
                        color="#1C58C7", fontsize=5.0, fontweight="bold", arrowprops=dict(arrowstyle="->", color="#1C58C7", lw=0.55))
            cax = axis.inset_axes([x + 0.025, 0.075, width - 0.050, 0.021])
            cb = plt.colorbar(sc, cax=cax, orientation="horizontal", ticks=[-0.75, 0, 0.75])
            cb.set_label("Signed deviation (mm)", fontsize=5.0, labelpad=0.2)
            cb.ax.tick_params(labelsize=5.0, length=1, pad=0.5)
        else:
            bins = np.linspace(-1.5, 1.5, 48)
            ia.hist(data["signed"], bins=bins, density=True, color="#8F8F8F", edgecolor="white", linewidth=0.15)
            ia.axvline(-0.5, color="#2879D0", ls="--", lw=0.8)
            ia.axvline(0.5, color="#D73A3A", ls="--", lw=0.8)
            ia.axvline(np.mean(data["signed"]), color="#222222", lw=0.7)
            ia.set_xlim(-1.5, 1.5)
            ia.set_xlabel("Signed deviation (mm)", fontsize=5.0, fontweight="bold", labelpad=1)
            ia.set_ylabel("Density", fontsize=5.0, fontweight="bold", labelpad=1)
            ia.tick_params(labelsize=5.0, length=2, pad=1)
            ia.spines["top"].set_visible(False); ia.spines["right"].set_visible(False)
            stats = (f"Mean = {np.mean(data['signed']):.3f} mm\n"
                     f"Median = {np.median(data['signed']):.3f} mm\n"
                     f"SD = {np.std(data['signed'], ddof=1):.3f} mm")
            ia.text(0.98, 0.95, stats, transform=ia.transAxes, ha="right", va="top", fontsize=5.0)
        x += width + 0.015


def draw_card(axis, index, code, title, subtitle, question, data):
    axis.set_xlim(0, 1); axis.set_ylim(0, 1); clean(axis)
    axis.add_patch(FancyBboxPatch((0.005, 0.005), 0.99, 0.99, boxstyle="round,pad=0.005,rounding_size=0.025",
                                  facecolor=CARD_BODY[index], edgecolor=CARD_EDGE[index], linewidth=0.75))
    axis.add_patch(FancyBboxPatch((0.005, 0.76), 0.99, 0.235, boxstyle="round,pad=0.005,rounding_size=0.025",
                                  facecolor=CARD_HEADER[index], edgecolor="none"))
    axis.text(0.5, 0.95, code, ha="center", va="center", fontsize=6.2, fontweight="bold")
    axis.text(0.5, 0.88, title, ha="center", va="center", fontsize=5.9, fontweight="bold", linespacing=0.95)
    axis.text(0.5, 0.79, subtitle, ha="center", va="center", fontsize=5.0, linespacing=0.95)
    axis.text(0.5, 0.69, question, ha="center", va="center", fontsize=5.0, linespacing=1.0)
    art = axis.inset_axes([0.08, 0.380, 0.84, 0.225])
    if index == 0:
        cloud(art, data["reference"], "#82B7C4", 0.045, 0.67); set_cloud_bounds(art, [data["reference"]])
        art.add_patch(FancyArrowPatch((0.07, 0.02), (0.93, 0.02), transform=art.transAxes, arrowstyle="<->", mutation_scale=5, color="#1768B3", lw=0.65))
        art.add_patch(FancyArrowPatch((0.05, 0.10), (0.05, 0.90), transform=art.transAxes, arrowstyle="<->", mutation_scale=5, color="#1768B3", lw=0.65))
        art.text(0.5, -0.02, "L / W", transform=art.transAxes, ha="center", va="top", fontsize=5.0, fontweight="bold", color="#1768B3")
        art.text(-0.005, 0.50, "H", transform=art.transAxes, ha="right", fontsize=5.0, fontweight="bold", color="#1768B3")
        bullets = ["Dimensional deviation", "Envelope fidelity"]
    elif index == 1:
        mid = np.median(data["points"][:, 0])
        left = data["reference"][:, 0] <= mid; right = data["points"][:, 0] > mid
        art.scatter(data["reference"][left, 0], data["reference"][left, 1], s=0.05, color="#6FAAB7", alpha=0.68, linewidths=0, rasterized=True)
        art.scatter(data["points"][right, 0], data["points"][right, 1], s=0.05, color="#D9C5A3", alpha=0.68, linewidths=0, rasterized=True)
        art.axvline(mid, color="#777777", lw=0.45, ls="--"); art.set_aspect("equal"); clean(art); set_cloud_bounds(art, [data["reference"], data["points"]])
        art.text(0.25, 0.02, "Design", transform=art.transAxes, ha="center", fontsize=5.0, color="#397787")
        art.text(0.75, 0.02, "Printed", transform=art.transAxes, ha="center", fontsize=5.0, color="#8F6F42")
        bullets = ["Bidirectional surface coverage", "Feature retention"]
    elif index in (2, 3):
        sc = art.scatter(data["points"][:, 0], data["points"][:, 1], c=np.clip(data["signed"], -0.75, 0.75), s=0.045,
                         cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-0.75, vcenter=0, vmax=0.75), linewidths=0, rasterized=True)
        art.set_aspect("equal"); clean(art); set_cloud_bounds(art, [data["points"]])
        if index == 3:
            art.annotate("Outward", xy=(0.22, 0.65), xytext=(0.02, 0.92), xycoords="axes fraction", textcoords="axes fraction",
                         color="#D11F2F", fontsize=5.0, fontweight="bold", arrowprops=dict(arrowstyle="->", color="#D11F2F", lw=0.45))
            art.annotate("Inward", xy=(0.79, 0.45), xytext=(0.76, 0.09), xycoords="axes fraction", textcoords="axes fraction",
                         color="#1C58C7", fontsize=5.0, fontweight="bold", arrowprops=dict(arrowstyle="->", color="#1C58C7", lw=0.45))
        cax = axis.inset_axes([0.18, 0.325, 0.64, 0.017])
        cb = plt.colorbar(sc, cax=cax, orientation="horizontal", ticks=[-0.75, 0, 0.75])
        cb.ax.tick_params(labelsize=5.0, length=1, pad=0.3)
        cb.set_label("Deviation (mm)" if index == 2 else "Signed deviation (mm)", fontsize=5.0, labelpad=0.2)
        bullets = ["MAE / RMSE / P95", "Local deviation severity"] if index == 2 else ["Mean signed deviation", "Positive / negative fractions", "Tolerance compliance"]
    else:
        absolute = np.abs(data["signed"]); vmax = float(np.quantile(absolute, 0.95))
        sc = art.scatter(data["points"][:, 0], data["points"][:, 1], c=np.clip(absolute, 0, vmax), s=0.045,
                         cmap="YlOrRd", norm=Normalize(0, vmax), linewidths=0, rasterized=True)
        art.set_aspect("equal"); clean(art); set_cloud_bounds(art, [data["points"]])
        cax = axis.inset_axes([0.20, 0.325, 0.60, 0.017])
        cb = plt.colorbar(sc, cax=cax, orientation="horizontal", ticks=[])
        cb.ax.text(-0.04, -1.4, "Low", transform=cb.ax.transAxes, ha="right", fontsize=5.0)
        cb.ax.text(1.04, -1.4, "High", transform=cb.ax.transAxes, ha="left", fontsize=5.0)
        bullets = ["Hotspot voxel fraction", "Defect clusters", "Spatial heterogeneity"]
    start = 0.055 + 0.048 * (len(bullets) - 1)
    for j, text in enumerate(bullets):
        axis.text(0.065, start - 0.048 * j, "▪", fontsize=5.0, va="center")
        axis.text(0.115, start - 0.048 * j, text, fontsize=5.0, va="center")


def draw_panel_c(axis, data):
    panel_frame(axis, "c", "Five-domain structural phenotype")
    specs = [
        ("D1", "Global geometric fidelity", "Overall dimensions & envelope", "Is the overall shape and size\nretained?"),
        ("D2", "Structural completeness", "Presence of designed features", "Are the designed structural\nfeatures recovered?"),
        ("D3", "Surface-deviation magnitude", "Local geometric error", "How large are local\nsurface deviations?"),
        ("D4", "Deviation polarity & tolerance", "Direction & acceptable fraction", "Are deviations inward or outward,\nand within tolerance?"),
        ("D5", "Spatial heterogeneity & localization", "Distribution of defects", "Where are the defects\nconcentrated?"),
    ]
    for i, spec in enumerate(specs):
        ia = axis.inset_axes([0.006 + i * 0.1985, 0.035, 0.193, 0.82])
        draw_card(ia, i, *spec, data)


def draw_panel_d(axis, data):
    panel_frame(axis, "d", "Geometric agreement, measurement repeatability and spatial-deviation characterization")
    dims = data["dimensions"]
    repeat = data["repeatability"]
    reg = data["registration"]
    axs = [axis.inset_axes([0.035, 0.13, 0.175, 0.65]), axis.inset_axes([0.248, 0.13, 0.155, 0.65]),
           axis.inset_axes([0.445, 0.13, 0.155, 0.65]), axis.inset_axes([0.640, 0.13, 0.165, 0.65]),
           axis.inset_axes([0.835, 0.13, 0.145, 0.65])]

    # i | dimensional agreement.
    ax = axs[0]
    colors = {"Length": "#2A6FB4", "Width": "#20A9B7", "Height": "#E69431"}
    markers = {"Length": "o", "Width": "s", "Height": "^"}
    for dim in ("Length", "Width", "Height"):
        part = dims[dims["dimension"] == dim]
        ax.scatter(part["reference_CAD_mm"], part["reconstructed_mm"], s=12, facecolor="white", edgecolor=colors[dim], marker=markers[dim], lw=0.8, label=dim)
    lo = 0; hi = max(dims["reference_CAD_mm"].max(), dims["reconstructed_mm"].max()) * 1.05
    ax.plot([lo, hi], [lo, hi], color="#DD3434", lw=0.7, ls="--")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Reference dimension (mm)", fontsize=5.0, fontweight="bold", labelpad=1)
    ax.set_ylabel("Reconstructed dimension (mm)", fontsize=5.0, fontweight="bold", labelpad=1)
    ax.tick_params(labelsize=5.0, length=2, pad=1)
    ax.set_title("Design–print dimensional agreement\n(11 geometries, 33 dimensions)", fontsize=5.3, fontweight="bold", pad=2)
    ax.text(0.04, 0.95, "R² = 0.988\nslope = 0.976\nRMSE = 2.34 mm", transform=ax.transAxes, va="top", fontsize=5.0)
    ax.legend(frameon=False, fontsize=5.0, loc="lower right", handletextpad=0.2, labelspacing=0.1)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # ii | repeatability.
    ax = axs[1]
    metrics = ["MAE_mm", "RMSE_mm", "P95_mm"]
    x = np.arange(3)
    scan_colors = ["#2A6FB4", "#26A8B6", "#E49A36"]
    scan_markers = ["o", "s", "^"]
    for i, row in repeat.iterrows():
        ax.plot(x, [row[m] for m in metrics], color=scan_colors[i], lw=0.75, alpha=0.85)
        ax.scatter(x, [row[m] for m in metrics], s=13, facecolor="white", edgecolor=scan_colors[i], marker=scan_markers[i], lw=0.8, label=f"Scan {i+2:02d}")
    ax.set_xticks(x, ["MAE", "RMSE", "P95"], fontsize=5.0)
    ax.set_ylabel("Bidirectional surface\ndistance (mm)", fontsize=5.0, fontweight="bold", labelpad=1)
    ax.tick_params(axis="y", labelsize=5.0, length=2, pad=1)
    ax.set_title("Scan-to-scan repeatability\n(unchanged specimen)", fontsize=5.3, fontweight="bold", pad=2)
    ax.legend(frameon=False, fontsize=5.0, ncol=3, loc="upper left", handlelength=0.7, handletextpad=0.2, columnspacing=0.4)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # iii | registration robustness.
    ax = axs[2]
    xx = np.arange(1, len(reg) + 1)
    y = reg["surface_mae_mm"].to_numpy()
    ax.fill_between([0.7, len(reg) + 0.3], [y.min(), y.min()], [y.max(), y.max()], color="#DCEAF4", alpha=0.8)
    ax.plot(xx, y, color="#2B72B5", lw=0.85, marker="o", ms=3.5)
    ax.set_xlim(0.7, len(reg) + 0.3)
    ax.set_xticks(xx, [str(i) for i in xx], fontsize=5.0)
    ax.set_ylabel("Surface MAE (mm)", fontsize=5.0, fontweight="bold", labelpad=1)
    ax.tick_params(axis="y", labelsize=5.0, length=2, pad=1)
    ax.set_title("Registration-method robustness\n(five rigid-registration strategies)", fontsize=5.3, fontweight="bold", pad=2)
    ax.text(0.5, 0.92, f"MAE = {y.min():.4f}–{y.max():.4f} mm", transform=ax.transAxes, ha="center", fontsize=5.0)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # iv | divergent domain scores.
    ax = axs[3]
    labels = ["D1", "D2", "D3", "D4", "D5"]
    values = [96.40, 27.23, 89.79, 57.35, 49.65]
    bar_colors = ["#75AEE2", "#E9B05B", "#63B85D", "#9A78CF", "#E987B1"]
    yy = np.arange(5)
    ax.barh(yy, values, color=bar_colors, height=0.63)
    ax.set_yticks(yy, labels, fontsize=5.0); ax.invert_yaxis(); ax.set_xlim(0, 105)
    for yi, val in zip(yy, values): ax.text(val + 1.3, yi, f"{val:.2f}", va="center", fontsize=5.0)
    ax.set_xlabel("Score (0–100)", fontsize=5.0, fontweight="bold", labelpad=1)
    ax.tick_params(axis="x", labelsize=5.0, length=2, pad=1)
    ax.set_title("Divergent structural phenotypes\nwithin a single print", fontsize=5.3, fontweight="bold", pad=2)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # v | localized deformation. Keep the quantitative summary outside the image field.
    container = axs[4]
    clean(container)
    container.text(0.50, 1.03, "Localized deformation patterns\n(defect hotspot analysis)",
                   transform=container.transAxes, ha="center", va="bottom", fontsize=5.1, fontweight="bold")
    ax = container.inset_axes([0.00, 0.19, 0.66, 0.66])
    sc = ax.scatter(data["points"][:, 0], data["points"][:, 1], c=np.clip(data["signed"], -0.5, 0.5), s=0.040,
                    cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=0.5), linewidths=0, rasterized=True)
    hot = np.abs(data["signed"]) > 0.5
    hot_voxels = np.unique(np.floor(data["points"][hot, :2] / 2.0), axis=0)
    hotspot_centers = (hot_voxels + 0.5) * 2.0
    ax.scatter(hotspot_centers[:, 0], hotspot_centers[:, 1], s=7.0, facecolors="none",
               edgecolors="#222222", marker="s", lw=0.35, rasterized=True)
    ax.set_aspect("equal"); clean(ax); set_cloud_bounds(ax, [data["points"]])
    container.text(0.69, 0.77, "Hotspot voxel\nfraction", transform=container.transAxes,
                   ha="left", va="top", fontsize=4.7, color="#444444")
    container.text(0.69, 0.61, "7.7%", transform=container.transAxes,
                   ha="left", va="top", fontsize=6.0, fontweight="bold", color="#183C5A")
    container.text(0.69, 0.45, "Defect clusters", transform=container.transAxes,
                   ha="left", va="top", fontsize=4.7, color="#444444")
    container.text(0.69, 0.34, "6", transform=container.transAxes,
                   ha="left", va="top", fontsize=6.0, fontweight="bold", color="#183C5A")
    container.text(0.69, 0.20, "Largest cluster\n160 mm³", transform=container.transAxes,
                   ha="left", va="top", fontsize=4.7, color="#444444")
    cax = container.inset_axes([0.03, 0.055, 0.59, 0.045])
    cb = plt.colorbar(sc, cax=cax, orientation="horizontal", ticks=[-0.5, 0, 0.5])
    cb.set_label("Deviation (mm)", fontsize=5.0, labelpad=0.2)
    cb.ax.tick_params(labelsize=5.0, length=1, pad=0.3)


def draw_d_component(axis, data, component):
    """Render one Figure 2d validation module as a publication-ready standalone panel."""
    axis.set_facecolor("white")
    if component == 1:
        dims = data["dimensions"]
        colors = {"Length": "#2A6FB4", "Width": "#20A9B7", "Height": "#E69431"}
        markers = {"Length": "o", "Width": "s", "Height": "^"}
        for dim in ("Length", "Width", "Height"):
            part = dims[dims["dimension"] == dim]
            axis.scatter(part["reference_CAD_mm"], part["reconstructed_mm"], s=32, facecolor="white",
                         edgecolor=colors[dim], marker=markers[dim], lw=1.2, label=dim)
        hi = max(dims["reference_CAD_mm"].max(), dims["reconstructed_mm"].max()) * 1.05
        axis.plot([0, hi], [0, hi], color="#DD3434", lw=1.0, ls="--")
        axis.set(xlim=(0, hi), ylim=(0, hi))
        axis.set_xlabel("Reference dimension (mm)", fontweight="bold")
        axis.set_ylabel("Reconstructed dimension (mm)", fontweight="bold")
        axis.set_title("Design–print dimensional agreement\n(11 geometries, 33 dimensions)", fontweight="bold", pad=6)
        axis.text(0.04, 0.95, "R² = 0.988\nslope = 0.976\nRMSE = 2.34 mm", transform=axis.transAxes, va="top")
        axis.legend(frameon=False, loc="lower right")
    elif component == 2:
        repeat = data["repeatability"]
        metrics = ["MAE_mm", "RMSE_mm", "P95_mm"]
        x = np.arange(3)
        colors = ["#2A6FB4", "#26A8B6", "#E49A36"]
        markers = ["o", "s", "^"]
        for i, row in repeat.iterrows():
            vals = [row[m] for m in metrics]
            axis.plot(x, vals, color=colors[i], lw=1.2, alpha=0.9)
            axis.scatter(x, vals, s=34, facecolor="white", edgecolor=colors[i], marker=markers[i], lw=1.2,
                         label=f"Scan {i+2:02d}")
        axis.set_xticks(x, ["MAE", "RMSE", "P95"])
        axis.set_ylabel("Bidirectional surface distance (mm)", fontweight="bold")
        axis.set_title("Scan-to-scan repeatability\n(unchanged specimen)", fontweight="bold", pad=6)
        axis.legend(frameon=False, ncol=3, loc="upper left", handlelength=0.8, columnspacing=0.7)
    elif component == 3:
        reg = data["registration"]
        xx = np.arange(1, len(reg) + 1)
        y = reg["surface_mae_mm"].to_numpy()
        axis.fill_between([0.7, len(reg) + 0.3], [y.min(), y.min()], [y.max(), y.max()], color="#DCEAF4", alpha=0.9)
        axis.plot(xx, y, color="#2B72B5", lw=1.4, marker="o", ms=6)
        axis.set_xlim(0.7, len(reg) + 0.3)
        axis.set_xticks(xx, [str(i) for i in xx])
        axis.set_xlabel("Registration strategy", fontweight="bold")
        axis.set_ylabel("Surface MAE (mm)", fontweight="bold")
        axis.set_title("Registration-method robustness\n(five rigid-registration strategies)", fontweight="bold", pad=6)
        axis.text(0.50, 0.93, f"MAE = {y.min():.4f}–{y.max():.4f} mm", transform=axis.transAxes, ha="center")
    elif component == 4:
        labels = ["D1", "D2", "D3", "D4", "D5"]
        values = [96.40, 27.23, 89.79, 57.35, 49.65]
        colors = ["#75AEE2", "#E9B05B", "#63B85D", "#9A78CF", "#E987B1"]
        yy = np.arange(5)
        axis.barh(yy, values, color=colors, height=0.62)
        axis.set_yticks(yy, labels); axis.invert_yaxis(); axis.set_xlim(0, 108)
        for yi, val in zip(yy, values):
            axis.text(val + 1.4, yi, f"{val:.2f}", va="center")
        axis.set_xlabel("Score (0–100)", fontweight="bold")
        axis.set_title("Divergent structural phenotypes\nwithin a single print", fontweight="bold", pad=6)
    elif component == 5:
        clean(axis)
        axis.text(0.50, 1.03, "Localized deformation patterns\n(defect hotspot analysis)",
                  transform=axis.transAxes, ha="center", va="bottom", fontweight="bold")
        map_ax = axis.inset_axes([0.02, 0.18, 0.65, 0.68])
        sc = map_ax.scatter(data["points"][:, 0], data["points"][:, 1], c=np.clip(data["signed"], -0.5, 0.5),
                            s=0.11, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=0.5),
                            linewidths=0, rasterized=True)
        hot = np.abs(data["signed"]) > 0.5
        centers = (np.unique(np.floor(data["points"][hot, :2] / 2.0), axis=0) + 0.5) * 2.0
        map_ax.scatter(centers[:, 0], centers[:, 1], s=15, facecolors="none", edgecolors="#222222",
                       marker="s", lw=0.55, rasterized=True)
        map_ax.set_aspect("equal"); clean(map_ax); set_cloud_bounds(map_ax, [data["points"]])
        axis.text(0.72, 0.75, "Hotspot voxel fraction", transform=axis.transAxes, color="#444444")
        axis.text(0.72, 0.65, "7.7%", transform=axis.transAxes, fontsize=10, fontweight="bold", color="#183C5A")
        axis.text(0.72, 0.50, "Defect clusters", transform=axis.transAxes, color="#444444")
        axis.text(0.72, 0.40, "6", transform=axis.transAxes, fontsize=10, fontweight="bold", color="#183C5A")
        axis.text(0.72, 0.25, "Largest cluster", transform=axis.transAxes, color="#444444")
        axis.text(0.72, 0.15, "160 mm³", transform=axis.transAxes, fontsize=9, fontweight="bold", color="#183C5A")
        cax = axis.inset_axes([0.08, 0.06, 0.52, 0.045])
        cb = plt.colorbar(sc, cax=cax, orientation="horizontal", ticks=[-0.5, 0, 0.5])
        cb.set_label("Deviation (mm)", fontweight="bold", labelpad=1)
    else:
        raise ValueError(f"Unknown Figure 2d component: {component}")
    if component != 5:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(length=3, pad=2)


def component_export(component, stem, data):
    fig = plt.figure(figsize=(3.65, 2.65), facecolor="white")
    ax = fig.add_axes([0.17, 0.17, 0.78, 0.72] if component != 5 else [0.06, 0.08, 0.90, 0.78])
    draw_d_component(ax, data, component)
    save_figure(fig, PANEL_DIR / stem)


def save_figure(fig, stem: Path):
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=600, facecolor="white")
    plt.close(fig)
    with Image.open(stem.with_suffix(".png")) as image:
        rgb = np.asarray(image.convert("RGB"))
    tifffile.imwrite(stem.with_suffix(".tiff"), rgb, photometric="rgb", compression="deflate",
                     resolution=(600, 600), resolutionunit="INCH")


def panel_export(draw_fn, stem: str, height: float, data):
    fig = plt.figure(figsize=(7.2, height), facecolor="white")
    ax = fig.add_axes([0.004, 0.015, 0.992, 0.97])
    draw_fn(ax, data)
    save_figure(fig, PANEL_DIR / stem)


def main():
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    WHOLE_DIR.mkdir(parents=True, exist_ok=True)
    dev = np.load(TURTLE / "pointwise_deviation.npz")
    points = orient(dev["points"].astype(float))
    reference = orient(read_ply_xyz(TURTLE / "reference_sampled.ply"))
    raw = pca_xy(read_ply_xyz(TURTLE / "sample_cleaned.ply"))
    photo = np.asarray(Image.open(PHOTO).convert("RGBA"))
    dimensions = pd.read_excel(SOURCE, sheet_name="Fig2e_dimension_pairs")
    repeatability = pd.read_excel(SOURCE, sheet_name="Fig2e_repeatability")
    registration = pd.read_excel(SOURCE, sheet_name="SuppFig7_reg_methods")
    data = dict(points=points, reference=reference, raw=raw, signed=dev["signed_deviation_mm"].astype(float),
                photo=photo, dimensions=dimensions, repeatability=repeatability, registration=registration)

    mpl.rcParams.update({
        "font.family": "Times New Roman", "font.serif": ["Times New Roman"],
        "font.sans-serif": ["Arial"], "mathtext.fontset": "stix",
        "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 6,
        "axes.linewidth": 0.6, "axes.labelweight": "bold", "axes.titleweight": "bold",
    })

    panel_export(draw_panel_a, "Figure2a_TNR", 2.05, data)
    panel_export(draw_panel_b, "Figure2b_TNR", 2.05, data)
    panel_export(draw_panel_c, "Figure2c_TNR", 2.15, data)
    panel_export(draw_panel_d, "Figure2d_TNR", 2.15, data)
    component_export(1, "Figure2d_i_dimensional_agreement_TNR", data)
    component_export(2, "Figure2d_ii_repeatability_TNR", data)
    component_export(3, "Figure2d_iii_registration_robustness_TNR", data)
    component_export(4, "Figure2d_iv_divergent_phenotypes_TNR", data)
    component_export(5, "Figure2d_v_localized_deformation_TNR", data)

    # Whole Figure 2: four stacked evidence bands, matching the requested layout.
    fig = plt.figure(figsize=(7.2, 5.82), facecolor="white")
    fig.text(0.008, 0.993, "Fig. 2 | Five-domain digital phenotyping resolves structural failures beyond global geometric fidelity.",
             fontsize=8.8, fontweight="bold", ha="left", va="top")
    positions = [(0.006, 0.748, 0.988, 0.202), (0.006, 0.553, 0.988, 0.185),
                 (0.006, 0.292, 0.988, 0.251), (0.006, 0.014, 0.988, 0.268)]
    for draw_fn, pos in zip((draw_panel_a, draw_panel_b, draw_panel_c, draw_panel_d), positions):
        ax = fig.add_axes(pos)
        draw_fn(ax, data)
    save_figure(fig, WHOLE_DIR / "Figure2_FINAL_TNR")
    print({"whole": str(WHOLE_DIR / "Figure2_FINAL_TNR"), "panels": 9,
           "reference_points": len(reference), "registered_points": len(points), "raw_points": len(raw)})


if __name__ == "__main__":
    main()
