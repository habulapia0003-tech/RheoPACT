from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import scipy

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT = REPO_ROOT / "working" / "fig4d_recalculation_131072"
DATA = OUT / "source_data"
FIG = OUT / "outputs"
AUD = OUT / "audit"
for d in (DATA, FIG, AUD):
    d.mkdir(parents=True, exist_ok=True)

SIM = REPO_ROOT / "external_data" / "Table_S4-1_full_131072_state_space.csv"
MEA = REPO_ROOT / "data" / "state_space" / "Fig4a_measured_conditions_540.csv"
PCA = REPO_ROOT / "data" / "state_space" / "Fig4a_frozen_PCA_definition.csv"
ANC = REPO_ROOT / "data" / "state_space" / "Fig4a_anchor_states.csv"

FEATURES = [
    "log10_G_prime_Pa", "log10_G_double_prime_Pa", "log10_eta10_Pa_s",
    "log10_1_plus_tau_y_Pa", "log10_K_Pa_s_n", "flow_index_n",
]
REGIONS = ["Low elasticity", "Compatible / balanced", "High elastic recovery"]
REGION_COLORS = ["#1768B3", "#16844A", "#D9470B"]
GATES = [
    ("gate1_extrusion", "Extrusion"),
    ("gate2_filament", "Filament"),
    ("gate3_landing", "Landing"),
    ("gate4_road", "Road"),
    ("gate5_structure", "Structure"),
    ("all_five_gates", "All five"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    sim = pd.read_csv(SIM)
    measured = pd.read_csv(MEA)
    definition = pd.read_csv(PCA).set_index("feature").loc[FEATURES]
    anchors = pd.read_csv(ANC).set_index("State").loc[REGIONS].reset_index()

    require(len(sim) == 131072, "Expected exactly 131,072 reduced-order states")
    require(sim["ink_id"].is_unique and not sim.duplicated().any(), "Simulation states must be unique")
    require(len(measured) == 540 and measured["Rheology_condition_ID"].is_unique, "Expected 540 unique measured conditions")
    require(measured["Evidence_class"].eq("REAL_EXPERIMENTAL_INPUT").all(), "Measured evidence class changed")
    require(sim["parameter_provenance"].eq("CONTROLLED_IN_SILICO_PARAMETER").all(), "Parameter provenance changed")
    require(sim["output_provenance"].eq("PHYSICS_SIMULATED_OUTPUT_REDUCED_MODEL").all(), "Output provenance changed")

    core = ["K_ref_Pa_s_n", "n", "tau_y_ref_Pa", "G_prime_Pa", "G_double_prime_Pa"]
    require(not sim[core].isna().any().any(), "Core state variables contain missing values")
    require((sim[["K_ref_Pa_s_n", "G_prime_Pa", "G_double_prime_Pa"]] > 0).all().all(), "Log inputs must be positive")
    require((sim["tau_y_ref_Pa"] >= 0).all(), "Yield stress must be non-negative")

    eta10 = sim["tau_y_ref_Pa"] / 10 + sim["K_ref_Pa_s_n"] * 10 ** (sim["n"] - 1)
    require((eta10 > 0).all(), "Apparent viscosity at 10 s^-1 must be positive before log10 transformation")
    eta10_safe = eta10.clip(lower=np.finfo(float).tiny)
    feat = pd.DataFrame({
        "log10_G_prime_Pa": np.log10(sim["G_prime_Pa"]),
        "log10_G_double_prime_Pa": np.log10(sim["G_double_prime_Pa"]),
        "log10_eta10_Pa_s": np.log10(eta10_safe),
        "log10_1_plus_tau_y_Pa": np.log10(1 + sim["tau_y_ref_Pa"]),
        "log10_K_Pa_s_n": np.log10(sim["K_ref_Pa_s_n"]),
        "flow_index_n": sim["n"],
    })

    mean = definition["scaler_mean"].to_numpy()
    scale = definition["scaler_scale"].to_numpy()
    anchor_scaled = (anchors[FEATURES].to_numpy() - mean) / scale
    direction = anchor_scaled[2] - anchor_scaled[0]
    direction = direction / np.linalg.norm(direction)
    anchor_index = (anchor_scaled - anchor_scaled[0]) @ direction
    scaled = (feat.to_numpy() - mean) / scale
    sim_index = (scaled - anchor_scaled[0]) @ direction
    thresholds = [
        (anchor_index[0] + anchor_index[1]) / 2,
        (anchor_index[1] + anchor_index[2]) / 2,
    ]
    sim_region = np.asarray(REGIONS, dtype=object)[np.digitize(sim_index, thresholds)]

    gate_columns = [x[0] for x in GATES]
    require(not sim[gate_columns].isna().any().any(), "Gate outputs contain missing values")
    for col in gate_columns:
        require(set(sim[col].unique()).issubset({True, False}), f"{col} is not Boolean")

    work = sim[["ink_id", *gate_columns]].copy()
    work["anchor_defined_region"] = sim_region
    work["anchor_axis_index"] = sim_index

    outcomes = [
        "Non-extrudable", "Filament failure", "Landing failure",
        "Road failure", "Structure failure", "All five pass",
    ]
    work["first_failure_outcome"] = np.select(
        [~work[col].to_numpy(dtype=bool) for col in gate_columns[:-1]],
        outcomes[:-1], default=outcomes[-1],
    )
    first_failures = []
    for region in REGIONS:
        region_outcomes = work.loc[work["anchor_defined_region"].eq(region), "first_failure_outcome"]
        counts = region_outcomes.value_counts().reindex(outcomes, fill_value=0)
        for outcome, count in counts.items():
            first_failures.append({
                "anchor_defined_region": region,
                "n_states": len(region_outcomes),
                "first_failure_outcome": outcome,
                "count": int(count),
                "percent": 100 * int(count) / len(region_outcomes),
                "evidence_class": "PHYSICS_SIMULATED_OUTPUT_REDUCED_MODEL",
            })
    pd.DataFrame(first_failures).to_csv(DATA / "Figure4d_first_failure_by_region_131072.csv", index=False)
    counts = work["first_failure_outcome"].value_counts().reindex(outcomes, fill_value=0)
    overall = counts.rename("count").rename_axis("first_failure_outcome").reset_index()
    overall["percent"] = 100 * overall["count"] / len(work)
    overall.to_csv(DATA / "Figure4d_first_failure_overall_131072.csv", index=False)

    rows = []
    for region in REGIONS:
        d = work.loc[work["anchor_defined_region"].eq(region)]
        row = {"anchor_defined_region": region, "n_states": int(len(d))}
        for col, label in GATES:
            row[f"{col}_pass_n"] = int(d[col].sum())
            row[f"{col}_pass_pct"] = float(100 * d[col].mean())
        five = [(label, row[f"{col}_pass_pct"]) for col, label in GATES[:-1]]
        row["limiting_stage"] = min(five, key=lambda x: x[1])[0]
        row["limiting_stage_pass_pct"] = min(x[1] for x in five)
        rows.append(row)
    summary = pd.DataFrame(rows)
    require(summary["n_states"].sum() == 131072, "Region counts do not reconcile")
    summary.to_csv(DATA / "Fig4d_gate_pass_summary_131072.csv", index=False, encoding="utf-8-sig")

    gate_long = []
    for _, row in summary.iterrows():
        for col, label in GATES:
            gate_long.append({
                "anchor_defined_region": row["anchor_defined_region"],
                "n_states": int(row["n_states"]),
                "gate": label,
                "pass_n": int(row[f"{col}_pass_n"]),
                "pass_pct": float(row[f"{col}_pass_pct"]),
                "evidence_class": "PHYSICS_SIMULATED_OUTPUT_REDUCED_MODEL",
            })
    pd.DataFrame(gate_long).to_csv(DATA / "Fig4d_gate_pass_long_131072.csv", index=False, encoding="utf-8-sig")

    values = summary[[f"{c}_pass_pct" for c, _ in GATES]].to_numpy()
    mpl.rcParams.update({
        "font.family": "Times New Roman",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 8,
        "axes.linewidth": 0.8,
    })
    cmap = LinearSegmentedColormap.from_list("pass", ["#F4F6F7", "#C9E5DC", "#4BAA83", "#0C6948"])
    fig = plt.figure(figsize=(7.05, 3.15), constrained_layout=False)
    ax = fig.add_axes([0.18, 0.24, 0.66, 0.57])
    im = ax.imshow(values, aspect="auto", cmap=cmap, vmin=0, vmax=100)
    ax.set_xticks(range(len(GATES)), [x[1] for x in GATES])
    ylabels = [f"{r}\n(n = {n:,})" for r, n in zip(summary["anchor_defined_region"], summary["n_states"])]
    ax.set_yticks(range(len(REGIONS)), ylabels)
    ax.tick_params(axis="x", rotation=0, labelsize=8, length=0, pad=6)
    ax.tick_params(axis="y", labelsize=8, length=0, pad=7)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                    color="white" if v >= 57 else "#1E2528", fontsize=8,
                    fontweight="bold" if j == values.shape[1] - 1 else "normal")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(GATES), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(REGIONS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i, color in enumerate(REGION_COLORS):
        ax.add_patch(plt.Rectangle((-0.63, i - 0.48), 0.08, 0.96, color=color,
                                   transform=ax.transData, clip_on=False, linewidth=0))

    cax = fig.add_axes([0.86, 0.28, 0.018, 0.47])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Pass rate (%)", fontsize=8)
    cb.ax.tick_params(labelsize=7, length=2)
    cb.outline.set_linewidth(0.6)

    fig.text(0.025, 0.91, "d", fontsize=15, fontweight="bold", va="top")
    fig.text(0.18, 0.91, "Rheological regions exhibit distinct model-derived process bottlenecks",
             fontsize=11.2, fontweight="bold", va="top")
    fig.text(0.18, 0.855,
             "Five-stage pass rates across the complete 131,072-state reduced-order design",
             fontsize=8, color="#4B5258", va="top")
    limiting = "  |  ".join(
        f"{r.split(' / ')[0]}: {b}" for r, b in zip(summary["anchor_defined_region"], summary["limiting_stage"])
    )
    fig.text(0.18, 0.115, f"Limiting stage by region: {limiting}", fontsize=7.2, color="#30363A")
    fig.text(0.18, 0.065,
             "Values are deterministic model-output proportions, not additional experiments or COMSOL solutions.",
             fontsize=6.8, color="#535A60")

    prefix = FIG / "Fig4d_gate_bottlenecks_131072_TNR"
    fig.savefig(prefix.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    png_path = prefix.with_name(prefix.name + "_600dpi.png")
    tiff_path = prefix.with_name(prefix.name + "_600dpi.tiff")
    fig.savefig(png_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    from PIL import Image
    import tifffile
    with Image.open(png_path) as image:
        rgb = np.asarray(image.convert("RGB"))
    tifffile.imwrite(tiff_path, rgb, photometric="rgb",
                     resolution=(600, 600), resolutionunit="INCH")

    audit = {
        "decision": "PASS",
        "analysis_role": "replacement for legacy Fig. 4d; no print-association model used",
        "simulated_state_n": int(len(sim)),
        "simulated_unique_id_n": int(sim["ink_id"].nunique()),
        "measured_condition_n_defining_coordinate_system": int(len(measured)),
        "legacy_310_association_used": False,
        "region_counts": summary[["anchor_defined_region", "n_states"]].to_dict(orient="records"),
        "gate_missing_n": {c: int(sim[c].isna().sum()) for c in gate_columns},
        "source_sha256": {str(p): sha256(p) for p in (SIM, MEA, PCA, ANC)},
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": mpl.__version__,
            "scipy": scipy.__version__,
        },
        "method": (
            "All 131,072 unique reduced-order states were assigned to the three anchor-defined regions "
            "using the six-feature scaler, PCA definition, and low-to-high anchor axis frozen from 540 "
            "unique measured rheological conditions. Gate pass rates are exact proportions of the full "
            "controlled in silico design; no inferential error bars were added."
        ),
        "claim_limit": (
            "The panel describes reduced-order model behavior and stage-specific screening hypotheses. "
            "It is not experimental validation and does not add measured printing samples."
        ),
    }
    (AUD / "Fig4d_recalculation_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (AUD / "Fig4d_legend.md").write_text(
        "**Fig. 4d | Rheological regions exhibit distinct model-derived process bottlenecks.** "
        "All 131,072 unique reduced-order simulation states were assigned to the low-elasticity, "
        "compatible/balanced, or high-elastic-recovery region using the six-feature coordinate system "
        "and anchor axis frozen from 540 measured rheological conditions. Cells report the exact "
        "percentage of states passing each sequential process gate; the final column reports the "
        "percentage passing all five gates. These proportions summarize the complete controlled "
        "in silico design and therefore are shown without sampling-error bars. They are model-derived "
        "screening outputs, not additional experiments or COMSOL solutions.\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
