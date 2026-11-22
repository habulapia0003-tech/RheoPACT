from __future__ import annotations

import hashlib
import json
import os
import time
import traceback
from pathlib import Path

import mph
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_META = REPO_ROOT / "external_data" / "simulation" / "giesekus" / "Model_1"
SOURCE_MODELS = SOURCE_META / "models"
RERUN_ROOT = REPO_ROOT / "working" / "comsol" / "giesekus_rerun"
DATA_DIR = RERUN_ROOT / "Data"
MODEL_DIR = RERUN_ROOT / "Models"
LOG_DIR = RERUN_ROOT / "Logs"
REPORT_DIR = RERUN_ROOT / "Reports"
TRANSIENT_DIR = DATA_DIR / "Transient"

STATES = ["LOW_ELASTICITY", "BALANCED", "HIGH_ELASTIC_RECOVERY"]
OUTPUT_NAMES = {
    "LOW_ELASTICITY": "mechanics_field_low.csv",
    "BALANCED": "mechanics_field_balanced.csv",
    "HIGH_ELASTIC_RECOVERY": "mechanics_field_high.csv",
}
TARGET_KPA = {
    "LOW_ELASTICITY": 5.8,
    "BALANCED": 31.5,
    "HIGH_ELASTIC_RECOVERY": 135.6,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array(model, expression: str, unit: str | None = None) -> np.ndarray:
    values = np.asarray(model.evaluate(expression, unit=unit, inner="last"), dtype=float).ravel()
    if values.size == 0 or not np.isfinite(values).any():
        raise RuntimeError(f"COMSOL returned no finite values for {expression!r}")
    return values


def array_at(model, dataset, inner_index: int, expression: str, unit: str | None = None) -> np.ndarray:
    values = np.asarray(model.evaluate(expression, unit=unit, dataset=dataset, inner=[int(inner_index)]), dtype=float).ravel()
    if values.size == 0 or not np.isfinite(values).any():
        raise RuntimeError(f"COMSOL returned no finite transient values for {expression!r}")
    return values

def scalar_at(model, dataset, inner_index: int, expression: str, unit: str | None = None) -> float:
    values = array_at(model, dataset, inner_index, expression, unit)
    finite = values[np.isfinite(values)]
    if finite.size != 1:
        raise RuntimeError(f"Expected transient scalar {expression!r}, got {finite.size} values")
    return float(finite[0])

def scalar(model, expression: str, unit: str | None = None) -> float:
    values = array(model, expression, unit)
    finite = values[np.isfinite(values)]
    if finite.size != 1:
        raise RuntimeError(f"Expected scalar {expression!r}, got {finite.size} values")
    return float(finite[0])


def runtime_version(client) -> str:
    try:
        return str(client.java.getVersion())
    except Exception:
        return "COMSOL version unavailable through client API"


def prepare_directories() -> None:
    for directory in (DATA_DIR, MODEL_DIR, LOG_DIR, REPORT_DIR, TRANSIENT_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def geometry_radius(z: np.ndarray) -> np.ndarray:
    return np.where(z <= 25.0, 15.0, np.where(z <= 27.0, 15.0 - 6.0 * (z - 25.0), 3.0 + (0.42 - 3.0) * (z - 27.0) / 34.0))


def geometry_slope(z: np.ndarray) -> np.ndarray:
    return np.where(z <= 25.0, 0.0, np.where(z <= 27.0, -6.0, (0.42 - 3.0) / 34.0))


def solve_one(client, state: str, comsol_version: str) -> dict[str, object]:
    source_model = SOURCE_MODELS / f"model1_{state}.mph"
    source_hash_before = sha256(source_model)
    started = time.time()
    record: dict[str, object] = {
        "State_ID": state,
        "status": "RUNNING",
        "source_model": str(source_model),
        "source_sha256_before": source_hash_before,
        "comsol_version": comsol_version,
        "pressure_expression": "p",
        "shear_rate_expression": "vef.sr",
        "shear_stress_expression": "mu_s*(uz+wr)+vef.Te_1rz",
        "delta_p_expression": "Pin-Pout",
        "solver_action": "java.sol('sol1').runAll()",
    }
    model = None
    try:
        model = client.load(str(source_model))
        java = model.java
        try:
            client.java.showProgress(str(LOG_DIR / f"{state}_solver_progress.log"))
        except Exception:
            pass

        parameters = {
            "Qpres_mm3_s": scalar(model, "Qpres", "mm^3/s"),
            "Rs_mm": scalar(model, "Rs", "mm"),
            "Rt_mm": scalar(model, "Rt", "mm"),
            "Ro_mm": scalar(model, "Ro", "mm"),
            "Ls_mm": scalar(model, "Ls", "mm"),
            "Lt_mm": scalar(model, "Lt", "mm"),
            "Ln_mm": scalar(model, "Ln", "mm"),
            "mu_s_Pa_s": scalar(model, "mu_s", "Pa*s"),
            "mu_e_Pa_s": scalar(model, "mu_e", "Pa*s"),
            "lambda_e_s": scalar(model, "lambda_e", "s"),
            "alpha_e": scalar(model, "alpha_e", "1"),
        }
        java.sol("sol1").runAll()
        solve_elapsed_s = time.time() - started

        dataset = list(model / "datasets")[0]
        inner_indices, inner_times = model.inner(dataset)
        transient_rows = []
        transient_expr = [
            ("x", "r", "mm"), ("y", "z", "mm"),
            ("pressure(kPa)", "p", "kPa"),
            ("shear_rate(1/s)", "vef.sr", "1/s"),
            ("total_shear_rz(Pa)", "mu_s*(uz+wr)+vef.Te_1rz", "Pa"),
            ("abs_total_shear_rz(Pa)", "abs(mu_s*(uz+wr)+vef.Te_1rz)", "Pa"),
            ("speed(m/s)", "sqrt(u^2+w^2)", "m/s"),
        ]
        for inner_index, time_value in zip(inner_indices, inner_times):
            cols = {name: array_at(model, dataset, int(inner_index), expr, unit) for name, expr, unit in transient_expr}
            frame_t = pd.DataFrame(cols)
            out_name = f"{state.lower()}_t{float(time_value):.3f}s.csv"
            frame_t.to_csv(TRANSIENT_DIR / out_name, index=False, float_format="%.12g")
            pin_t = scalar_at(model, dataset, int(inner_index), "Pin", "kPa")
            pout_t = scalar_at(model, dataset, int(inner_index), "Pout", "kPa")
            transient_rows.append({
                "State_ID": state, "inner_index": int(inner_index), "time_s": float(time_value),
                "field_file": out_name, "field_nodes": int(len(frame_t)),
                "Pin_kPa": pin_t, "Pout_kPa": pout_t, "DeltaP_kPa": pin_t - pout_t,
                "Qin_mm3_s": scalar_at(model, dataset, int(inner_index), "Qin", "mm^3/s"),
                "Qout_mm3_s": scalar_at(model, dataset, int(inner_index), "Qout", "mm^3/s"),
                "p_min_kPa": float(frame_t["pressure(kPa)"].min()), "p_max_kPa": float(frame_t["pressure(kPa)"].max()),
                "shear_rate_max_1_s": float(frame_t["shear_rate(1/s)"].max()),
                "abs_total_shear_rz_max_Pa": float(frame_t["abs_total_shear_rz(Pa)"].max()),
            })
        pd.DataFrame(transient_rows).to_csv(TRANSIENT_DIR / f"{state.lower()}_time_summary.csv", index=False)

        expressions = [
            ("x", "r", "mm"),
            ("y", "z", "mm"),
            ("pressure(kPa)", "p", "kPa"),
            ("u(m/s)", "u", "m/s"),
            ("w(m/s)", "w", "m/s"),
            ("speed(m/s)", "sqrt(u^2+w^2)", "m/s"),
            ("shear_rate(1/s)", "vef.sr", "1/s"),
            ("ur(1/s)", "ur", "1/s"),
            ("uz(1/s)", "uz", "1/s"),
            ("wr(1/s)", "wr", "1/s"),
            ("wz(1/s)", "wz", "1/s"),
            ("Te_rr(Pa)", "vef.Te_1rr", "Pa"),
            ("Te_zz(Pa)", "vef.Te_1zz", "Pa"),
            ("Te_rz(Pa)", "vef.Te_1rz", "Pa"),
            ("total_shear_rz(Pa)", "mu_s*(uz+wr)+vef.Te_1rz", "Pa"),
            ("abs_total_shear_rz(Pa)", "abs(mu_s*(uz+wr)+vef.Te_1rz)", "Pa"),
        ]
        field_data = {name: array(model, expr, unit) for name, expr, unit in expressions}
        lengths = {name: len(values) for name, values in field_data.items()}
        if len(set(lengths.values())) != 1:
            raise RuntimeError(f"Inconsistent COMSOL field lengths: {lengths}")
        field = pd.DataFrame(field_data)
        if field.isna().any().any():
            raise RuntimeError("Rerun field contains non-finite values")
        field.to_csv(DATA_DIR / OUTPUT_NAMES[state], index=False, float_format="%.12g")

        z = field["y"].to_numpy()
        r = field["x"].to_numpy()
        rp = geometry_slope(z)
        denom = np.sqrt(1.0 + rp**2)
        tr, tz = rp / denom, 1.0 / denom
        nr, nz = 1.0 / denom, -rp / denom
        mu_s = parameters["mu_s_Pa_s"]
        s_rr = 2.0 * mu_s * field["ur(1/s)"].to_numpy() + field["Te_rr(Pa)"].to_numpy()
        s_zz = 2.0 * mu_s * field["wz(1/s)"].to_numpy() + field["Te_zz(Pa)"].to_numpy()
        s_rz = field["total_shear_rz(Pa)"].to_numpy()
        wall_shear = tr * (s_rr * nr + s_rz * nz) + tz * (s_rz * nr + s_zz * nz)
        wall = np.abs(r - geometry_radius(z)) < 1e-6
        wall_frame = pd.DataFrame({
            "axial_position_mm": z[wall],
            "wall_radius_mm": r[wall],
            "pressure_kPa": field.loc[wall, "pressure(kPa)"].to_numpy(),
            "shear_rate_1_s": field.loc[wall, "shear_rate(1/s)"].to_numpy(),
            "wall_shear_signed_Pa": wall_shear[wall],
            "wall_shear_magnitude_Pa": np.abs(wall_shear[wall]),
        }).sort_values(["axial_position_mm", "wall_radius_mm"])
        wall_frame.to_csv(DATA_DIR / f"wall_mechanics_{state.lower()}.csv", index=False, float_format="%.12g")

        qin = scalar(model, "Qin", "m^3/s")
        qout = scalar(model, "Qout", "m^3/s")
        pin = scalar(model, "Pin", "Pa")
        pout = scalar(model, "Pout", "Pa")
        delta_p = scalar(model, "Pin-Pout", "Pa")
        final_time = float(np.nanmax(array(model, "t", "s")))
        rerun_model = MODEL_DIR / f"model1_{state}_TRANSIENT_DYNAMIC_RERUN_20260926.mph"
        model.save(str(rerun_model))

        original = json.loads((SOURCE_META / f"{state}_final.json").read_text(encoding="utf-8"))
        source_hash_after = sha256(source_model)
        record.update({
            "status": "SOLVED_RERUN_SAVED",
            "solve_elapsed_s": solve_elapsed_s,
            "final_time_s": final_time,
            **parameters,
            "Qin_mm3_s": qin * 1e9,
            "Qout_mm3_s": qout * 1e9,
            "mass_error": abs(qin - qout) / max(abs(qin), abs(qout), 1e-30),
            "Pin_kPa": pin / 1000.0,
            "Pout_kPa": pout / 1000.0,
            "DeltaP_rerun_kPa": delta_p / 1000.0,
            "DeltaP_requested_kPa": TARGET_KPA[state],
            "DeltaP_original_saved_kPa": float(original["DeltaP_Pa"]) / 1000.0,
            "rerun_minus_original_kPa": delta_p / 1000.0 - float(original["DeltaP_Pa"]) / 1000.0,
            "minimum_pressure_kPa": float(field["pressure(kPa)"].min()),
            "maximum_pressure_kPa": float(field["pressure(kPa)"].max()),
            "minimum_shear_rate_1_s": float(field["shear_rate(1/s)"].min()),
            "maximum_shear_rate_1_s": float(field["shear_rate(1/s)"].max()),
            "minimum_abs_total_shear_rz_Pa": float(field["abs_total_shear_rz(Pa)"].min()),
            "maximum_abs_total_shear_rz_Pa": float(field["abs_total_shear_rz(Pa)"].max()),
            "wall_nodes": int(wall.sum()),
            "outlet_total_normal_traction_kPa": scalar(model, "2*iout(r*(-p+2*mu_s*wz+vef.Te_1zz))/Ro^2", "kPa"),
            "field_nodes": len(field),
            "transient_time_count": int(len(inner_times)),
            "transient_times_s": ";".join(f"{float(v):.6g}" for v in inner_times),
            "rerun_model": str(rerun_model),
            "rerun_model_sha256": sha256(rerun_model),
            "source_sha256_after": source_hash_after,
            "source_unchanged": source_hash_before == source_hash_after,
        })
    except Exception as error:
        record.update({
            "status": "FAILED",
            "solve_elapsed_s": time.time() - started,
            "error": repr(error),
            "traceback": traceback.format_exc(),
            "source_sha256_after": sha256(source_model),
        })
    finally:
        if model is not None:
            try:
                client.remove(model)
            except Exception:
                pass
    (LOG_DIR / f"{state}_rerun.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(record, indent=2, ensure_ascii=False), flush=True)
    return record


def write_report(summary: pd.DataFrame) -> None:
    rows = [
        "# COMSOL multifield mechanics rerun audit",
        "",
        "## Scope",
        "",
        "Each original saved MPH model was loaded read-only and solved again with `sol1.runAll()`. No parameter, geometry, mesh, physics, boundary condition, study time or solver setting was changed. Static pressure, shear rate, total rz shear-stress component and wall shear traction were exported from the new solution.",
        "",
        "## Release gate",
        "",
        f"- All three solves completed: **{bool((summary.status == 'SOLVED_RERUN_SAVED').all())}**",
        f"- All source MPH hashes remained unchanged: **{bool(summary.source_unchanged.all())}**",
        f"- All rerun Δp values are within 0.10 kPa of the requested rounded values: **{bool(summary.DeltaP_target_pass.all())}**",
        f"- Rerun ordering is Low < Balanced < High: **{bool(summary.DeltaP_order_pass.all())}**",
        f"- All mass errors are below 1e-10: **{bool((summary.mass_error < 1e-10).all())}**",
        f"- All exports contain 6874 domain nodes and 226 native wall nodes: **{bool(((summary.field_nodes == 6874) & (summary.wall_nodes == 226)).all())}**",
        f"- Outlet total-normal-traction closure is within 0.15 kPa: **{bool((summary.outlet_total_normal_traction_kPa.abs() < 0.15).all())}**",
        "",
        "## Rerun values",
        "",
        "```csv\n" + summary.to_csv(index=False, float_format="%.9g").strip() + "\n```",
        "",
        "## Evidence boundaries",
        "",
        "- Pressure field is the COMSOL variable `p` evaluated in kPa at the final solved time.",
        "- Shear-rate field is the COMSOL invariant `vef.sr` in 1/s.",
        "- Domain shear-stress field is the magnitude of the rz component `mu_s*(uz+wr)+vef.Te_1rz` in Pa; it is not labelled as an equivalent/von-Mises stress.",
        "- Wall shear traction is the tangential projection of the full solvent-plus-elastic deviatoric stress tensor on native wall nodes.",
        "- Δp is evaluated independently as `Pin-Pout`, where both operators are embedded in each MPH.",
        "- The CSV coordinate `x` is the native non-negative axisymmetric radial coordinate `r`; no mirroring, interpolation, smoothing or contrast manipulation occurs in the exported data.",
        "- The newly saved rerun MPH files are evidence that the solver was executed again; they are not replacements for the originals.",
        "",
    ]
    (REPORT_DIR / "COMSOL_MULTIFIELD_RERUN_AUDIT.md").write_text("\n".join(rows), encoding="utf-8")


def main() -> None:
    prepare_directories()
    if os.environ.get("FIG4_COMSOL_REPORT_ONLY") == "1":
        records = [
            json.loads((LOG_DIR / f"{state}_rerun.json").read_text(encoding="utf-8"))
            for state in STATES
        ]
    else:
        client = mph.Client(port=2036)
        version = runtime_version(client)
        records = [solve_one(client, state, version) for state in STATES]
        try:
            client.clear()
        except Exception:
            pass
    summary = pd.DataFrame(records)
    if not (summary["status"] == "SOLVED_RERUN_SAVED").all():
        summary.to_csv(DATA_DIR / "rerun_validation_summary.csv", index=False)
        raise SystemExit("At least one COMSOL rerun failed; figure generation is blocked")

    summary["DeltaP_target_abs_error_kPa"] = (
        summary["DeltaP_rerun_kPa"] - summary["DeltaP_requested_kPa"]
    ).abs()
    summary["DeltaP_target_pass"] = summary["DeltaP_target_abs_error_kPa"] <= 0.10
    ordering = bool(np.all(np.diff(summary["DeltaP_rerun_kPa"].to_numpy()) > 0))
    summary["DeltaP_order_pass"] = ordering
    summary.to_csv(DATA_DIR / "rerun_validation_summary.csv", index=False)
    time_parts = [pd.read_csv(TRANSIENT_DIR / f"{state.lower()}_time_summary.csv") for state in STATES]
    transient_summary = pd.concat(time_parts, ignore_index=True)
    transient_summary.to_csv(DATA_DIR / "transient_validation_summary.csv", index=False)
    write_report(summary)

    gates = {
        "all_solved": bool((summary.status == "SOLVED_RERUN_SAVED").all()),
        "source_models_unchanged": bool(summary.source_unchanged.all()),
        "target_delta_p_pass": bool(summary.DeltaP_target_pass.all()),
        "low_balanced_high_order": ordering,
        "mass_error_pass": bool((summary.mass_error < 1e-10).all()),
        "node_count_pass": bool(((summary.field_nodes == 6874) & (summary.wall_nodes == 226)).all()),
        "outlet_traction_closure_pass": bool((summary.outlet_total_normal_traction_kPa.abs() < 0.15).all()),
        "transient_time_grid_pass": bool(len(transient_summary) == 15 and np.allclose(sorted(transient_summary.time_s.unique()), [0, 0.02, 0.05, 0.10, 0.15])),
        "transient_node_count_pass": bool((transient_summary.field_nodes == 6874).all()),
    }
    gates["plot_allowed"] = all(gates.values())
    (REPORT_DIR / "RERUN_GATES.json").write_text(
        json.dumps(gates, indent=2), encoding="utf-8"
    )
    print(json.dumps(gates, indent=2), flush=True)
    if not gates["plot_allowed"]:
        raise SystemExit("COMSOL rerun gates failed; figure generation is blocked")


if __name__ == "__main__":
    main()
