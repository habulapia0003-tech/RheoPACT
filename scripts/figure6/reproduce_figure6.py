"""Reproduce Figure 6 from the twelve author-recorded prints and frozen estimators."""
from pathlib import Path
import argparse
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from PIL import Image
import fitz


from figure6_same_setting_oqi import comparison, draw_panel, metrics as comparison_metrics

DOMAINS = [f"D{i}" for i in range(1, 6)]
MATERIALS = ["Beef", "Pork", "Carrot", "Pumpkin", "Purple sweet potato", "Milk gel"]
PHOTO_ORDER = ["Beef", "Pork", "Milk gel", "Carrot", "Purple sweet potato", "Pumpkin"]
BLUE, ORANGE, NAVY = "#2F6F9F", "#E88932", "#173F5F"
PROCESS = ["Print_temp_C", "Model_height_mm", "Print_speed_mm_s", "Nozzle_mm", "Nozzle_height_mm"]
RHEO = ["Gp_mean_Pa", "Gpp_mean_Pa", "eta_10_Pa_s", "eta_100_Pa_s", "Selected_tau0", "Selected_K", "Selected_n"]

plt.rcParams.update({
    "font.family": "Times New Roman", "font.size": 7, "pdf.fonttype": 42,
    "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .7, "legend.frameon": False,
})


def calculate(prints, scorer, recommender):
    prints = prints.copy()
    assert len(prints) == 12 and not prints.duplicated(["Material_ID", "Condition"]).any()
    assert set(prints.Condition) == {"Reference", "Adjusted"}
    prints["Phenotype_OQI"] = scorer.predict(prints[DOMAINS])
    reference = prints[prints.Condition == "Reference"].set_index("Material").loc[MATERIALS]
    adjusted = prints[prints.Condition == "Adjusted"].set_index("Material").loc[MATERIALS]
    pairs = pd.DataFrame({
        "Material_ID": adjusted.Material_ID, "Reference_OQI": reference.Phenotype_OQI,
        "Adjusted_OQI": adjusted.Phenotype_OQI,
    })
    pairs["OQI_gain"] = pairs.Adjusted_OQI - pairs.Reference_OQI
    for domain in DOMAINS:
        pairs[f"{domain}_gain"] = adjusted[domain] - reference[domain]
    pairs["Mean_D1_D5_gain"] = pairs[[f"{d}_gain" for d in DOMAINS]].mean(axis=1)

    # Apply the frozen classifier at the exact recorded adjusted settings.
    x = adjusted[PROCESS].copy()
    for column in RHEO[:-1]:
        x[f"log10_{column}"] = np.log10(adjusted[column].clip(lower=0) + 1e-8)
    x["Selected_n"] = adjusted.Selected_n
    clf = recommender["feasibility_classifier"]
    pairs["P_extrudable"] = clf.predict_proba(x)[:, list(clf.classes_).index(1)]
    threshold = float(recommender["feasibility_threshold"])
    pairs["Feasibility_screen_pass"] = pairs.P_extrudable >= threshold
    pairs = pairs.reset_index()
    process_delta = (adjusted[PROCESS] - reference[PROCESS]).reset_index()
    return prints, pairs, process_delta, threshold


def save(fig, folder, stem):
    fig.savefig(folder / f"{stem}.pdf", facecolor="white")
    fig.savefig(folder / f"{stem}.svg", facecolor="white")
    fig.savefig(folder / f"{stem}.png", dpi=360, facecolor="white")
    plt.close(fig)


def title(fig, letter, description):
    fig.text(.025, .965, letter, va="top", size=9, weight="bold")
    fig.text(.095, .965, description, va="top", size=8, weight="bold", color=NAVY)


def workflow(out):
    fig, ax = plt.subplots(figsize=(3.55, 2.05))
    fig.subplots_adjust(left=.055, right=.98, bottom=.09, top=.83)
    title(fig, "a", "Cross-material process adjustment")
    ax.axis("off")
    stages = [
        ("Six external food inks", "Rheological state recorded for each material"),
        ("Paired printing", "Empirical reference and recorded adjusted settings"),
        ("Design-referenced phenotyping", "D1–D5 and phenotype-derived OQI"),
    ]
    for y, (heading, line) in zip([.71, .37, .03], stages):
        ax.add_patch(FancyBboxPatch((.025, y), .95, .24, boxstyle="round,pad=.016",
                                   ec="#B4CBDD", fc="#F4F8FB", lw=.7))
        ax.text(.50, y + .160, heading, ha="center", va="center", size=7.4, weight="bold", color=NAVY)
        ax.text(.50, y + .067, line, ha="center", va="center", size=6.5)
    for y in [.70, .36]:
        ax.annotate("", (.50, y - .08), (.50, y), arrowprops={"arrowstyle": "-|>", "color": BLUE, "lw": .8})
    save(fig, out, "Figure6a")


def photo_panel(out, path):
    # Column centers follow the dividers of the author-supplied photo plate.
    # Only the outer legend strip and blank bottom margin are cropped.
    image = np.asarray(Image.open(path).convert("RGB"))
    assert image.shape[:2] == (444, 900)
    plate = image[:415, 65:, :]
    centers = (np.array([126.5, 256.5, 395, 537, 682, 828]) - 65) / plate.shape[1]
    fig = plt.figure(figsize=(7.20, 3.48))
    ax = fig.add_axes([.105, .029, .88, .84])
    ax.imshow(plate, interpolation="none", aspect="equal")
    ax.axis("off")
    title(fig, "c", "Representative prints at the paired settings")
    for center, name in zip(centers, PHOTO_ORDER):
        ax.text(center, 1.012, name, transform=ax.transAxes, ha="center", va="bottom", size=7.1, weight="bold")
    for y, label in [(1 - 98 / 415, "Reference"), (1 - 306 / 415, "Adjusted")]:
        ax.text(-.019, y, label, transform=ax.transAxes, ha="right", va="center", weight="bold", color=NAVY)
    save(fig, out, "Figure6c")


def state_panel(out, state_map, projection):
    from sklearn.neighbors import KNeighborsRegressor

    measured = pd.read_csv(state_map / "measured_conditions_540_projection.csv")
    reduced = pd.read_csv(state_map / "reduced_order_states_131072_projection.csv")
    definition = pd.read_csv(state_map / "frozen_pca_definition_from_540_measured.csv")
    anchors = pd.read_csv(state_map / "measured_anchor_states.csv")
    external = pd.read_csv(projection)
    assert len(measured) == 540 and len(reduced) == 131072
    features = definition.feature.tolist()
    anchor_state = (anchors[features].to_numpy() - definition.scaler_mean.to_numpy()) / definition.scaler_scale.to_numpy()
    axis = anchor_state[2] - anchor_state[0]
    indices = (anchor_state - anchor_state[0]) @ (axis / np.linalg.norm(axis))
    thresholds = (indices[:-1] + indices[1:]) / 2
    pc = np.vstack([reduced[["PC1", "PC2"]], measured[["PC1", "PC2"]]])
    state = np.r_[reduced.anchor_axis_index, measured.elastic_state_index]
    pad = np.ptp(pc, axis=0) * [.035, .055]
    lower, upper = pc.min(axis=0) - pad, pc.max(axis=0) + pad
    gx, gy = np.linspace(lower[0], upper[0], 480), np.linspace(lower[1], upper[1], 360)
    xx, yy = np.meshgrid(gx, gy)
    guide = KNeighborsRegressor(61, weights="distance").fit(pc, state)
    zz = guide.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(3.55, 2.33))
    fig.subplots_adjust(left=.143, right=.975, top=.82, bottom=.22)
    title(fig, "b", "Projection onto the frozen rheological map")
    ax.contourf(xx, yy, zz, levels=[zz.min() - 1e-8, *thresholds, zz.max() + 1e-8],
                colors=["#EAF3FC", "#EAF7EE", "#FFF0E8"])
    ax.contour(xx, yy, zz, levels=thresholds, colors="#687780", linewidths=.65, linestyles="--")
    ax.scatter(measured.PC1, measured.PC2, s=4.5, c="#58656F", alpha=.17, edgecolors="none", rasterized=True)
    offsets = [(-7, -11), (-11, 6), (-5, 12), (11, -3), (7, -8), (10, 8)]
    for row, offset, color in zip(external.itertuples(), offsets, ["#8C5A45", "#AB7836", "#C56127", "#BD8517", "#74529B", "#397995"]):
        ax.scatter(row.PC1, row.PC2, s=22, c=color, ec="white", lw=.65, zorder=4)
        ax.annotate(str(row.Plot_ID), (row.PC1, row.PC2), xytext=offset, textcoords="offset points",
                    color=color, weight="bold", size=6.5, ha="center", va="center",
                    arrowprops={"arrowstyle": "-", "lw": .6, "color": color}, zorder=5)
    for x, y, text, color in [(.09, .17, "Low elasticity", "#1768B3"),
                              (.41, .72, "Balanced", "#16844A"), (.71, .88, "High recovery", "#B3461B")]:
        ax.text(x, y, text, transform=ax.transAxes, color=color, size=6.5, weight="bold")
    ax.set(xlim=(gx[0], gx[-1]), ylim=(gy[0], gy[-1]), xlabel="PC1 (70.6%)", ylabel="PC2 (18.6%)")
    fig.text(.14, .045, "1 Beef   2 Pork   3 Carrot   4 Pumpkin   5 Purple sweet potato   6 Milk gel", size=5.6)
    save(fig, out, "Figure6b")


def gains_heatmap(ax, pairs, columns, labels, high, cmap="Blues", fmt="+.0f"):
    values = pairs[columns].to_numpy()
    im = ax.imshow(values, cmap=cmap, vmin=0, vmax=high, aspect="auto")
    for i, row in enumerate(values):
        for j, v in enumerate(row):
            ax.text(j, i, format(v, fmt), ha="center", va="center", size=6.8,
                    color="white" if v > high * .69 else NAVY)
    ax.set_xticks(range(len(columns)), labels)
    ax.set_yticks(range(len(pairs)), pairs.Material)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def oqi_pairs(ax, pairs):
    y = np.arange(len(pairs))
    ax.hlines(y, pairs.Reference_OQI, pairs.Adjusted_OQI, color="#BAC9D4", lw=1.5)
    ax.scatter(pairs.Reference_OQI, y, s=25, color=BLUE, marker="s", zorder=3)
    ax.scatter(pairs.Adjusted_OQI, y, s=26, color=ORANGE, zorder=3)
    for i, row in pairs.iterrows():
        ax.text(100.8, i, f"+{row.OQI_gain:.2f}", va="center", ha="right", weight="bold", color=NAVY, size=6.5)
    ax.set(xlim=(40, 103), xticks=[40, 50, 60, 70, 80, 90], xlabel="Phenotype-derived OQI")
    ax.set_yticks(y, pairs.Material)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=.27)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.legend(handles=[
        Line2D([], [], marker="s", color=BLUE, lw=0, ms=4.1, label="Reference"),
        Line2D([], [], marker="o", color=ORANGE, lw=0, ms=4.1, label="Adjusted"),
    ], loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, handletextpad=.25, borderaxespad=0)


def figures(out, photo, panel_b, pairs, process_delta, threshold, same_setting):
    workflow(out)
    photo_panel(out, photo)
    fig, ax = plt.subplots(figsize=(3.55, 2.56))
    title(fig, "d", "Changes in structural phenotype")
    fig.subplots_adjust(left=.28, right=.93, top=.81, bottom=.25)
    im = gains_heatmap(ax, pairs, [f"{d}_gain" for d in DOMAINS], DOMAINS, 45)
    cax = fig.add_axes([.30, .141, .55, .035])
    fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0, 15, 30, 45], label="Gain (points)")
    save(fig, out, "Figure6d")

    draw_panel(same_setting, out)

    fig, axs = plt.subplots(1, 2, figsize=(7.20, 2.65), gridspec_kw={"wspace": .64})
    fig.subplots_adjust(left=.14, right=.99, top=.71, bottom=.22)
    title(fig, "a", "Paired D1–D5 gains")
    fig.text(.57, .965, "b", weight="bold", va="top", size=9)
    fig.text(.605, .965, "Paired OQI values", weight="bold", va="top", size=8, color=NAVY)
    im = gains_heatmap(axs[0], pairs, [f"{d}_gain" for d in DOMAINS], DOMAINS, 45)
    fig.colorbar(im, ax=axs[0], fraction=.037, pad=.03, ticks=[0, 15, 30, 45], label="Gain (points)")
    oqi_pairs(axs[1], pairs)
    save(fig, out, "Supplementary_Figure28")

    fig = plt.figure(figsize=(7.20, 3.67))
    gs = fig.add_gridspec(2, 2, left=.16, right=.98, top=.85, bottom=.14,
                          width_ratios=[1.42, 1], hspace=.60, wspace=.44)
    ax = fig.add_subplot(gs[:, 0])
    cols = PROCESS
    differences = process_delta[cols].to_numpy()
    scales = np.abs(differences).max(axis=0)
    scaled = np.divide(differences, scales, out=np.zeros_like(differences), where=scales > 0)
    ax.imshow(scaled, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
    for i, row in enumerate(differences):
        for j, v in enumerate(row):
            ax.text(j, i, f"{v:+.2f}".replace("+0.00", "0"), ha="center", va="center", size=6.4,
                    color="white" if abs(scaled[i, j]) > .74 else NAVY)
    ax.set_xticks(range(5), ["Temp.\n(°C)", "Height\n(mm)", "Speed\n(mm/s)", "Nozzle\n(mm)", "Gap\n(mm)"])
    ax.set_yticks(range(6), pairs.Material)
    ax.set_title("a  Recorded changes in process settings", loc="left", weight="bold", size=8, pad=6)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(pairs.Mean_D1_D5_gain, pairs.OQI_gain, color=BLUE, s=25)
    ax.set(xlabel="Mean D1–D5 gain (points)", ylabel="OQI gain (points)")
    ax.set_title("b  Gains across the six matrices", loc="left", weight="bold", size=8, pad=6)
    ax = fig.add_subplot(gs[1, 1])
    y = np.arange(6)
    ax.scatter(pairs.P_extrudable, y, color=BLUE, s=23, zorder=3)
    ax.axvline(threshold, lw=.8, ls="--", color="#6B737D")
    ax.set(xlim=(.44, 1.045), xlabel="P(extrudable)", xticks=[.5, .75, 1.0])
    ax.set_yticks(y, [f"M{i:02}" for i in range(1, 7)])
    ax.invert_yaxis()
    ax.set_title("c  Frozen feasibility screen", loc="left", weight="bold", size=8, pad=6)
    fig.text(.16, .025, "Panel a shows adjusted minus reference settings. Colors are scaled within each variable.", color="#52606D")
    save(fig, out, "Supplementary_Figure29")

    # Compose the five panel PDFs at their intended printed widths.
    assembly = fitz.open()
    sources = [fitz.open(p) for p in (
        out / "Figure6a.pdf", panel_b, out / "Figure6c.pdf",
        out / "Figure6d.pdf", out / "Figure6e.pdf",
    )]
    width, gap = 183 / 25.4 * 72, 8
    half = (width - gap) / 2
    widths = [half, half, width, half, half]
    heights = [w * s[0].rect.height / s[0].rect.width for w, s in zip(widths, sources)]
    top = max(heights[:2])
    bottom = top + heights[2] + 2 * gap
    y = [0, 0, top + gap, bottom, bottom]
    x = [0, half + gap, 0, 0, half + gap]
    page = assembly.new_page(width=width, height=bottom + max(heights[3:]))
    for i, source in enumerate(sources):
        page.show_pdf_page(fitz.Rect(x[i], y[i], x[i] + widths[i], y[i] + heights[i]), source, 0)
    assembly.save(out / "Figure6.pdf", deflate=True, garbage=4)
    page.get_pixmap(dpi=240, alpha=False).save(out / "Figure6.png")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["prints", "scorer", "recommender", "photo", "projection", "state-map", "fig5-oqi-scorer", "out"]:
        p.add_argument(f"--{name}", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    prints, pairs, changes, threshold = calculate(pd.read_csv(args.prints),
        joblib.load(args.scorer), joblib.load(args.recommender))
    prints.to_csv(args.out / "Figure6_12_prints.csv", index=False)
    pairs.to_csv(args.out / "Figure6_paired_results.csv", index=False)
    changes.to_csv(args.out / "Supplementary_Figure29a_process_changes.csv", index=False)
    comparison_inputs = prints.copy()
    summary_path = Path(__file__).resolve().parents[2] / "data/figure6/Figure6_summary.json"
    source_inputs = json.loads(summary_path.read_text(encoding="utf-8"))
    pumpkin_adjusted = comparison_inputs.Material_ID.eq("M04") & comparison_inputs.Condition.eq("Adjusted")
    for descriptor, value in source_inputs["same_setting_M04_rheology"].items():
        comparison_inputs.loc[pumpkin_adjusted, descriptor] = value
    same_setting = comparison(comparison_inputs, joblib.load(args.recommender), joblib.load(args.fig5_oqi_scorer))
    same_setting.to_csv(args.out / "Figure6e_same_setting_OQI.csv", index=False)
    metrics = {
        "same_setting_Figure5_OQI_comparison": {
            **comparison_metrics(same_setting),
            "result_role": "current_Figure6e_result",
            "report_in_main_text": True,
            "input_variant": "author_reported_later_M04_rheology",
        },
        "score_matrix_lineage_status": source_inputs["score_matrix_lineage_status"],
        "same_setting_M04_input_provenance": source_inputs["same_setting_M04_input_provenance"],
        "same_setting_M04_rheology": source_inputs["same_setting_M04_rheology"],
        "same_setting_M04_original_rheology": source_inputs["same_setting_M04_original_rheology"],
        "same_setting_Figure5_OQI_comparison_original_input": source_inputs[
            "same_setting_Figure5_OQI_comparison_original_input"
        ],
        "paired_materials": len(pairs), "recorded_prints": len(prints),
        "mean_OQI_gain": pairs.OQI_gain.mean(),
        "OQI_gain_range": [pairs.OQI_gain.min(), pairs.OQI_gain.max()],
        "mean_D1_D5_gains": [pairs[f"{d}_gain"].mean() for d in DOMAINS],
        "positive_D1_D5_comparisons": int((pairs[[f"{d}_gain" for d in DOMAINS]] > 0).sum().sum()),
        "feasibility_passes": int(pairs.Feasibility_screen_pass.sum()),
        "feasibility_threshold": threshold,
        "feasibility_probability_range": [pairs.P_extrudable.min(), pairs.P_extrudable.max()],
    }
    (args.out / "Figure6_summary.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    state_panel(args.out, args.state_map, args.projection)
    figures(args.out, args.photo, args.out / "Figure6b.pdf", pairs, changes, threshold, same_setting)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
