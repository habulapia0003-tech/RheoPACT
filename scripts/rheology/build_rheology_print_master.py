from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import unicodedata
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from openpyxl import load_workbook


ALLOWED_CONCENTRATIONS = {8, 10, 12, 14, 16, 18}
ALLOWED_PREPARATION_T = {70, 80, 90}
ALLOWED_RHEOLOGY_T = {25, 35, 45}
ETA_TARGETS = (0.1, 0.5, 1, 5, 10, 20, 50, 100)
RANDOM_SEED = 20260825
np.random.seed(RANDOM_SEED)


@dataclass
class ParsedCondition:
    raw_sample_name: str
    normalized_sample_name: str
    raw_prefix: str | None
    concentration_pct: int | None
    preparation_t_c: int | None
    rheology_t_c: int | None
    rheology_replicate: int | None
    parsing_status: str
    parsing_note: str
    correction_applied: bool


def normalize_text(value) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace("\u3000", " ")
    text = re.sub(r"[‘’'\"“”]", "", text)
    text = re.sub(r"[－–—−]", "-", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s*-\s*", "-", text)
    return text


def normalize_sample_name(value) -> str:
    return normalize_text(value)


def parse_rheology_condition(value, default_replicate: int = 1) -> ParsedCondition:
    raw = "" if value is None else str(value)
    normalized = normalize_sample_name(raw)
    prefix_match = re.match(r"^([^\d-]+)-", normalized)
    prefix = prefix_match.group(1).strip() if prefix_match else None
    numeric_text = normalized[len(prefix_match.group(0)) :] if prefix_match else normalized
    nums = [int(x) for x in re.findall(r"\d+", numeric_text)]
    notes: list[str] = []
    correction = normalized != raw.strip()
    status = "OK"
    c = h = t = rep = None
    summary_label_without_replicate = False

    if len(nums) >= 3:
        c, h, t = nums[:3]
        if len(nums) == 3:
            rep = default_replicate
            summary_label_without_replicate = True
            status = "OK_inferred_replicate"
            notes.append("replicate=1 inferred from single curated condition block")
        elif len(nums) == 4:
            rep = nums[3]
        else:
            # Known pattern such as 10-70-45-1 2: retain the last token as replicate,
            # record the unexplained token instead of silently treating it as a key.
            rep = nums[-1]
            notes.append(f"extra_numeric_tokens={nums[3:-1]}")
            status = "Corrected_with_note"
            correction = True
    else:
        status = "Unparsed"
        notes.append("fewer_than_three_condition_numbers")

    invalid = []
    if c is not None and c not in ALLOWED_CONCENTRATIONS:
        invalid.append(f"concentration={c}")
    if h is not None and h not in ALLOWED_PREPARATION_T:
        invalid.append(f"preparation_T={h}")
    if t is not None and t not in ALLOWED_RHEOLOGY_T:
        invalid.append(f"rheology_T={t}")
    if invalid:
        status = "Invalid_level"
        notes.append("unexpected_allowed_level:" + ",".join(invalid))
    if prefix:
        notes.append(f"prefix={prefix}")

    suggested = normalized
    if c is not None and h is not None and t is not None:
        suggested = (
            f"{prefix + '-' if prefix else ''}{c}-{h}-{t}"
            if summary_label_without_replicate
            else f"{prefix + '-' if prefix else ''}{c}-{h}-{t} {rep or default_replicate}"
        )
        if normalize_text(suggested).lower() != normalized.lower():
            correction = True
            if status == "OK":
                status = "Normalized"
    if correction and status == "OK":
        status = "Normalized"
    return ParsedCondition(
        raw_sample_name=raw,
        normalized_sample_name=suggested,
        raw_prefix=prefix,
        concentration_pct=c,
        preparation_t_c=h,
        rheology_t_c=t,
        rheology_replicate=rep,
        parsing_status=status,
        parsing_note="; ".join(notes),
        correction_applied=correction,
    )


def condition_id(starch: str, c: int, h: int, t: int) -> str:
    safe = re.sub(r"\s+", "_", normalize_text(starch))
    return f"{safe}_C{int(c):02d}_H{int(h):02d}_T{int(t):02d}"


def ensure_xlsx(path: Path, conversion_dir: Path) -> Path:
    if path.suffix.lower() == ".xlsx":
        return path
    if path.suffix.lower() != ".xls":
        raise ValueError(f"Unsupported workbook type: {path}")
    converter = Path(__file__).with_name("convert_xls_readonly.ps1")
    if not converter.exists():
        raise FileNotFoundError(f"Required converter not found: {converter}")
    conversion_dir.mkdir(parents=True, exist_ok=True)
    expected = conversion_dir / f"{path.stem}.xlsx"
    if not expected.exists() or expected.stat().st_mtime < path.stat().st_mtime:
        subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(converter),
                "-InputPaths",
                str(path),
                "-OutputDirectory",
                str(conversion_dir),
            ],
            check=True,
        )
    return expected


def choose_files_gui():
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    print_path = filedialog.askopenfilename(title="Select the printing records workbook", filetypes=[("Excel", "*.xlsx *.xls")])
    shear_path = filedialog.askopenfilename(title="Select the steady-shear rheology workbook", filetypes=[("Excel", "*.xlsx *.xls")])
    dynamic_path = filedialog.askopenfilename(title="Select the oscillatory rheology workbook", filetypes=[("Excel", "*.xlsx *.xls")])
    output_dir = filedialog.askdirectory(title="Select an output directory")
    root.destroy()
    if not all((print_path, shear_path, dynamic_path, output_dir)):
        raise SystemExit("File selection cancelled")
    return Path(print_path), Path(shear_path), Path(dynamic_path), Path(output_dir)


def load_print_design(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=0, header=1, engine="openpyxl")
    raw = raw.dropna(how="all").copy()
    column_map = {
        "\u5b8c\u6210\u60c5\u51b5": "Completion_status",
        "\u5b9e\u9a8c\u7ec4\u522b": "Experiment_group",
        "\u6dc0\u7c89\u6d53\u5ea6(%)": "Concentration_pct",
        "\u6dc0\u7c89\u5236\u5907\u6e29\u5ea6(℃)": "Preparation_T_C",
        "\u6a21\u578b\u9ad8\u5ea6C(mm）": "Model_height_mm",
        "\u6253\u5370\u901f\u5ea6A(mm/s)": "Printing_speed_mm_s",
        "\u55b7\u5634\u76f4\u5f84(mm)": "Nozzle_diameter_mm",
        "\u5c42\u9ad8B(mm)": "Layer_height_code",
        "\u6253\u5370\u6e29\u5ea6D(℃)": "Printing_T_C",
        "\u8fb9\u957f（mm）": "Edge_length_mm",
        "\u8fb9\u9ad8（mm）": "Edge_height_mm",
        "\u4e2d\u5fc3\u9ad8\u5ea6（mm）": "Center_height_mm",
        "\u7ebf\u5bbd（mm）": "Line_width_mm",
    }
    missing = [col for col in column_map if col not in raw.columns]
    if missing:
        raise ValueError(f"Printing workbook missing expected columns: {missing}")
    df = raw[list(column_map)].rename(columns=column_map)
    numeric = [
        "Experiment_group", "Concentration_pct", "Preparation_T_C", "Model_height_mm",
        "Printing_speed_mm_s", "Nozzle_diameter_mm", "Layer_height_code", "Printing_T_C",
        "Edge_length_mm", "Edge_height_mm", "Center_height_mm", "Line_width_mm",
    ]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["Experiment_group"].between(1, 59, inclusive="both")].copy()
    df["Experiment_group"] = df["Experiment_group"].astype(int)
    df["Printing_condition_ID"] = df["Experiment_group"].map(lambda x: f"G{x:03d}")
    df["Layer_height_mm"] = np.nan
    df["Layer_height_note"] = "Source values are design codes; physical mm mapping not found in workbook"
    return df.reset_index(drop=True)


def _organized_sheet_names(path: Path) -> list[str]:
    wb = load_workbook(path, read_only=True, data_only=True)
    names = [ws.title for ws in wb.worksheets[1:]]
    wb.close()
    return names


def _naming_qc_record(source_file, source_sheet, parsed: ParsedCondition, parsed_starch: str):
    return {
        "Source_file": source_file,
        "Source_sheet": source_sheet,
        "Raw_sample_name": parsed.raw_sample_name,
        "Raw_prefix": parsed.raw_prefix,
        "Parsed_starch": parsed_starch,
        "Parsed_concentration": parsed.concentration_pct,
        "Parsed_preparation_T": parsed.preparation_t_c,
        "Parsed_rheology_T": parsed.rheology_t_c,
        "Parsed_replicate": parsed.rheology_replicate,
        "Parsing_status": parsed.parsing_status,
        "Suggested_normalized_name": parsed.normalized_sample_name,
        "Issue": parsed.parsing_note,
        "Correction_applied": parsed.correction_applied,
    }


def load_shear_rheology(path: Path):
    wb = load_workbook(path, read_only=True, data_only=True)
    source_file = path.name
    records, qc, mapping = [], [], []
    for ws in wb.worksheets[1:]:
        starch = ws.title
        shear_rates = [ws.cell(r, 1).value for r in range(2, 42)]
        seen_curves = set()
        for col in range(2, ws.max_column + 1):
            raw_name = ws.cell(1, col).value
            if raw_name in (None, ""):
                continue
            parsed = parse_rheology_condition(raw_name)
            qc.append(_naming_qc_record(source_file, starch, parsed, starch))
            mapping.append({"Raw_sheet": starch, "Raw_prefix": parsed.raw_prefix, "Standard_starch_name": starch, "Mapping_status": "Sheet_authoritative"})
            if None in (parsed.concentration_pct, parsed.preparation_t_c, parsed.rheology_t_c):
                continue
            rid = condition_id(starch, parsed.concentration_pct, parsed.preparation_t_c, parsed.rheology_t_c)
            curve_values = [ws.cell(r, col).value for r in range(2, 42)]
            fingerprint = (rid, parsed.rheology_replicate or 1, tuple(curve_values))
            if fingerprint in seen_curves:
                qc[-1]["Parsing_status"] = "Duplicate_source_column"
                qc[-1]["Issue"] = (qc[-1]["Issue"] + "; " if qc[-1]["Issue"] else "") + "exact duplicate curve excluded from analysis"
                qc[-1]["Correction_applied"] = False
                continue
            seen_curves.add(fingerprint)
            for point, (rate, viscosity) in enumerate(zip(shear_rates, curve_values), 1):
                rate = pd.to_numeric(rate, errors="coerce")
                viscosity = pd.to_numeric(viscosity, errors="coerce")
                if pd.isna(rate) or pd.isna(viscosity):
                    continue
                stress = float(rate) * float(viscosity)
                records.append({
                    "Rheology_condition_ID": rid,
                    "Starch_type": starch,
                    "Concentration_pct": parsed.concentration_pct,
                    "Preparation_T_C": parsed.preparation_t_c,
                    "Rheology_T_C": parsed.rheology_t_c,
                    "Rheology_replicate": parsed.rheology_replicate or 1,
                    "Measurement_point": point,
                    "Shear_rate_1_s": float(rate),
                    "Viscosity_Pa_s": float(viscosity),
                    "Shear_stress_Pa": stress,
                    "Raw_value": float(viscosity),
                    "Raw_unit": "Pa·s",
                    "Source_sheet": starch,
                    "Raw_sample_name": parsed.raw_sample_name,
                    "Normalized_sample_name": parsed.normalized_sample_name,
                })
    wb.close()
    return pd.DataFrame(records), pd.DataFrame(qc), pd.DataFrame(mapping).drop_duplicates()


def inspect_dynamic_protocol(path: Path):
    """Audit raw Sheet1 protocol metadata without assigning ambiguous raw blocks to starch sheets."""
    wb = load_workbook(path, read_only=True, data_only=True)
    ws = wb.worksheets[0]
    blocks, current = [], None
    for values in ws.iter_rows(values_only=True):
        cells = ["" if value is None else str(value) for value in values]
        if cells and cells[0] == "Data Series Information":
            if current:
                blocks.append(current)
            current = {"Raw_sample_name": None, "N_measurement_points": None, "Point_duration_s": None, "Strain_pct": None, "Angular_frequency_rad_s": None}
        if current is None:
            continue
        if cells and cells[0] == "Name:" and len(cells) > 3:
            current["Raw_sample_name"] = cells[3]
        text = " | ".join(cells)
        if "Number of Data Points:" in text and len(cells) > 3:
            current["N_measurement_points"] = pd.to_numeric(cells[3], errors="coerce")
        match = re.search(r"Meas\. Pt\. Duration\s*([0-9.]+)\s*s", text)
        if match:
            current["Point_duration_s"] = float(match.group(1))
        match = re.search(r"Amplitude gamma\s*=\s*([0-9.]+)\s*%", text)
        if match:
            current["Strain_pct"] = float(match.group(1))
        match = re.search(r"Angular Frequency omega\s*=\s*([0-9.]+)\s*rad/s", text)
        if match:
            current["Angular_frequency_rad_s"] = float(match.group(1))
    if current:
        blocks.append(current)
    wb.close()
    block_df = pd.DataFrame(blocks)

    def unique_values(column):
        return sorted(pd.to_numeric(block_df[column], errors="coerce").dropna().unique().tolist())

    strains = unique_values("Strain_pct")
    omegas = unique_values("Angular_frequency_rad_s")
    point_counts = unique_values("N_measurement_points")
    durations = unique_values("Point_duration_s")
    protocol = {
        "Oscillation_test_type": "fixed-frequency small-amplitude oscillatory measurement" if len(strains) == len(omegas) == 1 else "metadata mismatch - review required",
        "Strain_pct": strains[0] if len(strains) == 1 else np.nan,
        "Angular_frequency_rad_s": omegas[0] if len(omegas) == 1 else np.nan,
        "N_measurement_points": int(point_counts[0]) if len(point_counts) == 1 else np.nan,
        "Point_duration_s": durations[0] if len(durations) == 1 and block_df["Point_duration_s"].notna().all() else np.nan,
        "Total_duration_s": (durations[0] * point_counts[0]) if len(durations) == len(point_counts) == 1 and block_df["Point_duration_s"].notna().all() else np.nan,
        "Time_axis_status": "reliable_time" if len(durations) == 1 and block_df["Point_duration_s"].notna().all() else "unavailable_in_curated_sheets_use_point_index",
        "Protocol_metadata_blocks": len(block_df),
        "Observed_point_durations_s": ", ".join(str(x) for x in durations) if durations else "missing",
    }
    protocol_rows = [
        {"Protocol_field": key, "Observed_value": value, "Source": "Dynamic raw Sheet1", "Interpretation": "Automatically audited; raw blocks are not safely attributable to all 10 starch sheets"}
        for key, value in protocol.items()
    ]
    protocol_rows.extend([
        {"Protocol_field": "Raw_blocks_with_duration", "Observed_value": int(block_df["Point_duration_s"].notna().sum()), "Source": "Dynamic raw Sheet1", "Interpretation": "Duration metadata is incomplete and includes 5 s and 6 s; time slopes are therefore not assigned to curated conditions"},
        {"Protocol_field": "Raw_blocks_total", "Observed_value": len(block_df), "Source": "Dynamic raw Sheet1", "Interpretation": "All audited blocks had 10 points, 1% strain and 100 rad/s"},
    ])
    return protocol, pd.DataFrame(protocol_rows), block_df


def load_dynamic_rheology(path: Path, protocol: dict):
    wb = load_workbook(path, read_only=True, data_only=True)
    source_file = path.name
    records, qc, mapping = [], [], []
    for ws in wb.worksheets[1:]:
        starch = ws.title
        # Curated sheets store 54 condition blocks of 10 readings in columns A:E.
        sheet_rows = list(ws.iter_rows(min_row=1, max_row=541, max_col=4, values_only=True))
        row = 1
        while row <= 541:
            raw_name = sheet_rows[row - 1][0]
            if raw_name in (None, ""):
                row += 1
                continue
            parsed = parse_rheology_condition(raw_name)
            if None in (parsed.concentration_pct, parsed.preparation_t_c, parsed.rheology_t_c):
                row += 1
                continue
            qc.append(_naming_qc_record(source_file, starch, parsed, starch))
            mapping.append({"Raw_sheet": starch, "Raw_prefix": parsed.raw_prefix, "Standard_starch_name": starch, "Mapping_status": "Sheet_authoritative"})
            if None not in (parsed.concentration_pct, parsed.preparation_t_c, parsed.rheology_t_c):
                rid = condition_id(starch, parsed.concentration_pct, parsed.preparation_t_c, parsed.rheology_t_c)
                for point in range(10):
                    point_values = sheet_rows[row + point - 1]
                    gp = pd.to_numeric(point_values[1], errors="coerce")
                    gpp = pd.to_numeric(point_values[3], errors="coerce")
                    if pd.isna(gp) and pd.isna(gpp):
                        continue
                    gp_value = float(gp) if pd.notna(gp) else np.nan
                    gpp_value = float(gpp) if pd.notna(gpp) else np.nan
                    omega = protocol.get("Angular_frequency_rad_s", np.nan)
                    valid_pair = np.isfinite(gp_value) and np.isfinite(gpp_value) and gp_value != 0 and gpp_value != 0
                    tan_delta = gpp_value / gp_value if valid_pair else np.nan
                    g_complex = math.sqrt(gp_value**2 + gpp_value**2) if np.isfinite(gp_value) and np.isfinite(gpp_value) else np.nan
                    records.append({
                        "Rheology_condition_ID": rid,
                        "Starch_type": starch,
                        "Concentration_pct": parsed.concentration_pct,
                        "Preparation_T_C": parsed.preparation_t_c,
                        "Rheology_T_C": parsed.rheology_t_c,
                        "Rheology_replicate": parsed.rheology_replicate or 1,
                        "Point_index": point + 1,
                        "Measurement_point": point + 1,
                        "Time_s": np.nan,
                        "Time_axis_status": protocol.get("Time_axis_status"),
                        "Oscillation_test_type": protocol.get("Oscillation_test_type"),
                        "Angular_frequency_rad_s": omega,
                        "Strain_pct": protocol.get("Strain_pct", np.nan),
                        "Point_duration_s": protocol.get("Point_duration_s", np.nan),
                        "Total_duration_s": protocol.get("Total_duration_s", np.nan),
                        "G_prime_Pa": gp_value,
                        "G_double_prime_Pa": gpp_value,
                        "tan_delta_i": tan_delta,
                        "G_complex_Pa_i": g_complex,
                        "phase_angle_deg_i": math.degrees(math.atan(tan_delta)) if np.isfinite(tan_delta) else np.nan,
                        "Elastic_fraction_i": gp_value / (gp_value + gpp_value) if valid_pair and (gp_value + gpp_value) != 0 else np.nan,
                        "Elastic_viscous_ratio_i": gp_value / gpp_value if valid_pair else np.nan,
                        "Complex_viscosity_Pa_s_i": g_complex / omega if np.isfinite(g_complex) and np.isfinite(omega) and omega > 0 else np.nan,
                        "Source_sheet": starch,
                        "Raw_sample_name": parsed.raw_sample_name,
                        "Normalized_sample_name": parsed.normalized_sample_name,
                    })
            row += 10
    wb.close()
    return pd.DataFrame(records), pd.DataFrame(qc), pd.DataFrame(mapping).drop_duplicates()


def _metric_bundle(y, pred, k):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    residual = y - pred
    n = len(y)
    rss = float(np.sum(residual**2))
    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))
    tss = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1 - rss / tss) if tss > 0 else np.nan
    safe_rss = max(rss, np.finfo(float).tiny)
    aic = float(n * np.log(safe_rss / n) + 2 * k)
    bic = float(n * np.log(safe_rss / n) + k * np.log(n))
    return {"R2": r2, "RMSE": rmse, "MAE": mae, "AIC": aic, "BIC": bic}


def fit_power_law(rate, stress):
    x, y = np.asarray(rate, float), np.asarray(stress, float)
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 4:
        return {"PL_fit_success": False, "PL_fit_note": "fewer_than_4_positive_points"}
    n, log_k = np.polyfit(np.log(x), np.log(y), 1)
    k = float(np.exp(log_k))
    pred = k * x**n
    metrics = _metric_bundle(y, pred, 2)
    return {
        "K_PL": k, "n_PL": float(n),
        **{f"{key}_PL": value for key, value in metrics.items()},
        "PL_fit_success": True, "PL_fit_note": "log-linear fit; metrics on stress scale",
    }


def fit_herschel_bulkley(rate, stress):
    x, y = np.asarray(rate, float), np.asarray(stress, float)
    mask = (x > 0) & np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 6:
        return {"HB_fit_success": False, "HB_fit_note": "fewer_than_6_points"}
    best = None
    grids = [np.linspace(0.05, 1.5, 291)]
    for grid in grids:
        for n_hb in grid:
            z = x**n_hb
            design = np.column_stack([np.ones_like(z), z])
            tau0, k_hb = np.linalg.lstsq(design, y, rcond=None)[0]
            if tau0 < 0:
                tau0 = 0.0
                k_hb = float(np.dot(z, y) / np.dot(z, z))
            if k_hb <= 0:
                continue
            pred = tau0 + k_hb * z
            rss = float(np.sum((y - pred) ** 2))
            if best is None or rss < best[0]:
                best = (rss, float(tau0), float(k_hb), float(n_hb), pred)
    if best is None:
        return {"HB_fit_success": False, "HB_fit_note": "no_nonnegative_solution"}
    _, tau0, k_hb, n_hb, pred = best
    # Local refinement around grid optimum.
    lo, hi = max(0.01, n_hb - 0.01), min(2.0, n_hb + 0.01)
    for candidate in np.linspace(lo, hi, 201):
        z = x**candidate
        tau, kval = np.linalg.lstsq(np.column_stack([np.ones_like(z), z]), y, rcond=None)[0]
        if tau < 0:
            tau = 0.0
            kval = float(np.dot(z, y) / np.dot(z, z))
        if kval <= 0:
            continue
        candidate_pred = tau + kval * z
        rss = float(np.sum((y - candidate_pred) ** 2))
        if rss < best[0]:
            best = (rss, float(tau), float(kval), float(candidate), candidate_pred)
    _, tau0, k_hb, n_hb, pred = best
    boundary = tau0 <= 1e-10 or n_hb <= 0.011 or n_hb >= 1.999
    metrics = _metric_bundle(y, pred, 3)
    return {
        "tau0_HB": tau0, "K_HB": k_hb, "n_HB": n_hb,
        **{f"{key}_HB": value for key, value in metrics.items()},
        "HB_fit_success": not boundary,
        "HB_fit_note": "boundary_solution" if boundary else "nonnegative grid-refined least squares",
    }


def _log_interp(x, y, target):
    order = np.argsort(x)
    x, y = np.asarray(x)[order], np.asarray(y)[order]
    if target < x.min() or target > x.max():
        return np.nan, False
    exact = np.where(np.isclose(x, target, rtol=1e-8, atol=1e-12))[0]
    if len(exact):
        return float(y[exact[0]]), False
    value = float(np.exp(np.interp(np.log(target), np.log(x), np.log(y))))
    return value, True


def extract_shear_features(shear_long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["Rheology_condition_ID", "Starch_type", "Concentration_pct", "Preparation_T_C", "Rheology_T_C", "Rheology_replicate"]
    for key, group in shear_long.groupby(keys, dropna=False, sort=False):
        group = group.sort_values("Shear_rate_1_s")
        x = group["Shear_rate_1_s"].to_numpy(float)
        eta = group["Viscosity_Pa_s"].to_numpy(float)
        stress = group["Shear_stress_Pa"].to_numpy(float)
        valid = (x > 0) & (eta > 0) & np.isfinite(x) & np.isfinite(eta)
        x, eta, stress = x[valid], eta[valid], stress[valid]
        row = dict(zip(keys, key))
        row["Raw_sample_name"] = group["Raw_sample_name"].iloc[0]
        row["n_curve_points"] = len(x)
        interp_flags = []
        for target in ETA_TARGETS:
            val, interpolated = _log_interp(x, eta, target) if len(x) else (np.nan, False)
            label = str(target).replace(".", "p")
            row[f"eta_{label}"] = val
            row[f"eta_{label}_interpolated"] = interpolated
            interp_flags.append(interpolated)
        if len(x) >= 4:
            slope, _ = np.polyfit(np.log10(x), np.log10(eta), 1)
            row.update({
                "low_shear_viscosity": float(eta[0]),
                "high_shear_viscosity": float(eta[-1]),
                "viscosity_drop_ratio": float(eta[0] / eta[-1]) if eta[-1] else np.nan,
                "loglog_viscosity_slope": float(slope),
                "shear_thinning_index": float(slope + 1),
                "viscosity_curve_AUC": float(np.trapezoid(eta, np.log10(x))),
                "log_viscosity_AUC": float(np.trapezoid(np.log10(eta), np.log10(x))),
                "viscosity_dynamic_range": float(np.max(eta) / np.min(eta)),
                "any_loglog_interpolation": any(interp_flags),
            })
        row.update(fit_power_law(x, stress))
        row.update(fit_herschel_bulkley(x, stress))
        aic_pl, aic_hb = row.get("AIC_PL", np.nan), row.get("AIC_HB", np.nan)
        bic_pl, bic_hb = row.get("BIC_PL", np.nan), row.get("BIC_HB", np.nan)
        row["Delta_AIC_HB_minus_PL"] = aic_hb - aic_pl if np.isfinite(aic_pl) and np.isfinite(aic_hb) else np.nan
        row["Delta_BIC_HB_minus_PL"] = bic_hb - bic_pl if np.isfinite(bic_pl) and np.isfinite(bic_hb) else np.nan
        if row.get("HB_fit_success") and np.isfinite(aic_hb) and aic_hb + 2 < aic_pl:
            row["Best_constitutive_model"] = "Herschel-Bulkley"
        elif row.get("PL_fit_success"):
            row["Best_constitutive_model"] = "Power-law"
        else:
            row["Best_constitutive_model"] = "None"
        row["Steady_QC"] = "Pass" if row.get("PL_fit_success") and len(x) >= 30 else "Review"
        rows.append(row)
    return pd.DataFrame(rows)


def _distribution_features(values, prefix):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {f"{prefix}_{suffix}": np.nan for suffix in ["mean", "median", "SD", "CV_pct", "min", "max", "range"]}
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_SD": sd,
        f"{prefix}_CV_pct": float(sd / mean * 100) if np.isfinite(sd) and mean != 0 else np.nan,
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_range": float(np.max(values) - np.min(values)),
    }


def _first_last_drift(values, n_each=3):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2 * n_each:
        return np.nan
    first = float(np.mean(values[:n_each]))
    last = float(np.mean(values[-n_each:]))
    return float((last - first) / first * 100) if first != 0 else np.nan


def _linear_slope(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3 or len(np.unique(x)) < 2:
        return np.nan, np.nan
    slope, intercept = np.polyfit(x, y, 1)
    pred = intercept + slope * x
    tss = float(np.sum((y - np.mean(y)) ** 2))
    rss = float(np.sum((y - pred) ** 2))
    r2 = 1 - rss / tss if tss > 0 else np.nan
    return float(slope), float(r2)


def extract_dynamic_features(dynamic_long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["Rheology_condition_ID", "Starch_type", "Concentration_pct", "Preparation_T_C", "Rheology_T_C", "Rheology_replicate"]
    variables = {
        "G_prime": "G_prime_Pa",
        "G_double_prime": "G_double_prime_Pa",
        "G_complex": "G_complex_Pa_i",
        "tan_delta": "tan_delta_i",
        "phase_angle_deg": "phase_angle_deg_i",
        "elastic_fraction": "Elastic_fraction_i",
        "elastic_viscous_ratio": "Elastic_viscous_ratio_i",
        "Complex_viscosity": "Complex_viscosity_Pa_s_i",
    }
    for key, group in dynamic_long.groupby(keys, dropna=False, sort=False):
        group = group.sort_values("Point_index")
        gp = pd.to_numeric(group["G_prime_Pa"], errors="coerce").to_numpy(float)
        gpp = pd.to_numeric(group["G_double_prime_Pa"], errors="coerce").to_numpy(float)
        valid_pair = np.isfinite(gp) & np.isfinite(gpp)
        row = dict(zip(keys, key))
        row.update({
            "Raw_sample_name": group["Raw_sample_name"].iloc[0],
            "Oscillation_test_type": group["Oscillation_test_type"].iloc[0],
            "Reference_angular_frequency_rad_s": float(group["Angular_frequency_rad_s"].dropna().iloc[0]) if group["Angular_frequency_rad_s"].notna().any() else np.nan,
            "Reference_strain_pct": float(group["Strain_pct"].dropna().iloc[0]) if group["Strain_pct"].notna().any() else np.nan,
            "N_points": int(valid_pair.sum()),
            "n_within_run_points": int(valid_pair.sum()),
            "Point_duration_s": float(group["Point_duration_s"].dropna().iloc[0]) if group["Point_duration_s"].notna().any() else np.nan,
            "Total_duration_s": float(group["Total_duration_s"].dropna().iloc[0]) if group["Total_duration_s"].notna().any() else np.nan,
            "Time_axis_status": group["Time_axis_status"].iloc[0],
            "Nonphysical_point_count": int(((gp <= 0) | (gpp <= 0)).sum()),
            "G_prime_greater_than_G_double_all_points": bool(np.all(gp[valid_pair] > gpp[valid_pair])) if valid_pair.any() else False,
        })
        for prefix, column in variables.items():
            values = pd.to_numeric(group[column], errors="coerce").to_numpy(float)
            row.update(_distribution_features(values, prefix))
            row[f"{prefix}_drift_pct"] = _first_last_drift(values)

        point_x = pd.to_numeric(group["Point_index"], errors="coerce").to_numpy(float)
        time_x = pd.to_numeric(group["Time_s"], errors="coerce").to_numpy(float)
        for prefix, column in {"G_prime": "G_prime_Pa", "G_double_prime": "G_double_prime_Pa", "G_complex": "G_complex_Pa_i", "tan_delta": "tan_delta_i"}.items():
            values = pd.to_numeric(group[column], errors="coerce").to_numpy(float)
            point_slope, point_r2 = _linear_slope(point_x, values)
            row[f"{prefix}_point_slope"] = point_slope
            row[f"{prefix}_point_slope_R2"] = point_r2
            if np.isfinite(time_x).sum() >= 3:
                time_slope, time_r2 = _linear_slope(time_x, values)
            else:
                time_slope, time_r2 = np.nan, np.nan
            row[f"{prefix}_time_slope"] = time_slope
            row[f"{prefix}_time_slope_R2"] = time_r2

        # Backward-compatible aliases used by the first-stage database.
        row.update({
            "G_prime_ref": row.get("G_prime_mean"),
            "G_double_prime_ref": row.get("G_double_prime_mean"),
            "tan_delta_ref": row.get("tan_delta_mean"),
            "G_complex_ref": row.get("G_complex_mean"),
            "Gprime_to_Gdoubleprime": row.get("elastic_viscous_ratio_mean"),
            "G_prime_within_SD": row.get("G_prime_SD"),
            "G_double_prime_within_SD": row.get("G_double_prime_SD"),
            "G_prime_platform_CV": row.get("G_prime_CV_pct", np.nan) / 100,
            "G_double_prime_platform_CV": row.get("G_double_prime_CV_pct", np.nan) / 100,
            "Gprime_Gdoubleprime_stability": row.get("elastic_viscous_ratio_CV_pct", np.nan) / 100,
            "Dynamic_max_within_run_CV": np.nanmax([row.get("G_prime_CV_pct", np.nan), row.get("G_double_prime_CV_pct", np.nan), row.get("tan_delta_CV_pct", np.nan)]) / 100,
            "Complex_viscosity_100rad_s": row.get("Complex_viscosity_mean") if np.isclose(row.get("Reference_angular_frequency_rad_s", np.nan), 100) else np.nan,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def apply_dynamic_qc(dynamic_features: pd.DataFrame):
    df = dynamic_features.copy()
    df["Max_dynamic_CV_pct"] = df[["G_prime_CV_pct", "G_double_prime_CV_pct", "tan_delta_CV_pct"]].max(axis=1)
    df["Max_abs_dynamic_drift_pct"] = df[["G_prime_drift_pct", "G_double_prime_drift_pct", "tan_delta_drift_pct", "G_complex_drift_pct"]].abs().max(axis=1)

    distribution_rows = []
    for metric in ["G_prime_CV_pct", "G_double_prime_CV_pct", "tan_delta_CV_pct", "G_prime_drift_pct", "G_double_prime_drift_pct", "tan_delta_drift_pct", "Max_dynamic_CV_pct", "Max_abs_dynamic_drift_pct"]:
        values = pd.to_numeric(df[metric], errors="coerce").dropna()
        q1, median, q3 = values.quantile([0.25, 0.5, 0.75])
        iqr = q3 - q1
        mad = float(np.median(np.abs(values - median)))
        distribution_rows.append({
            "Metric": metric, "N": len(values), "Median": median, "Q1": q1, "Q3": q3, "IQR": iqr,
            "Upper_outlier_threshold": q3 + 1.5 * iqr, "MAD": mad,
        })
    distributions = pd.DataFrame(distribution_rows)
    cv_row = distributions.loc[distributions.Metric == "Max_dynamic_CV_pct"].iloc[0]
    drift_row = distributions.loc[distributions.Metric == "Max_abs_dynamic_drift_pct"].iloc[0]
    df["CV_robust_z"] = 0.6745 * (df["Max_dynamic_CV_pct"] - cv_row.Median) / cv_row.MAD if cv_row.MAD > 0 else np.nan
    df["Drift_robust_z"] = 0.6745 * (df["Max_abs_dynamic_drift_pct"] - drift_row.Median) / drift_row.MAD if drift_row.MAD > 0 else np.nan
    df["CV_warning_gt_20pct"] = df["Max_dynamic_CV_pct"] > 20
    df["CV_robust_outlier"] = df["Max_dynamic_CV_pct"] > cv_row.Upper_outlier_threshold
    df["Strong_drift_robust_outlier"] = df["Max_abs_dynamic_drift_pct"] > drift_row.Upper_outlier_threshold

    statuses, notes = [], []
    for _, row in df.iterrows():
        flags, explanations = [], []
        if row["N_points"] < 6:
            flags.append("Missing")
            explanations.append("fewer than 6 valid paired points")
        if row["Nonphysical_point_count"] > 0:
            flags.append("Nonphysical")
            explanations.append(f"{int(row['Nonphysical_point_count'])} point(s) with G'<=0 or G''<=0")
        if row["CV_robust_outlier"]:
            flags.append("Unstable")
            explanations.append("CV exceeds Q3+1.5×IQR")
        if row["Strong_drift_robust_outlier"]:
            flags.append("Strong drift")
            explanations.append("absolute drift exceeds Q3+1.5×IQR")
        if row["CV_warning_gt_20pct"]:
            explanations.append("fixed warning: max CV >20%; retained, not deleted")
        if row.get("Time_axis_status") != "reliable_time":
            explanations.append("time metadata not safely assignable; point-index slopes used")
        statuses.append("; ".join(flags) if flags else ("Warning" if row["CV_warning_gt_20pct"] else "Pass"))
        notes.append("; ".join(explanations) if explanations else "within robust distribution")
    df["Dynamic_QC_status"] = statuses
    df["Dynamic_QC_note"] = notes
    df["Dynamic_QC"] = df["Dynamic_QC_status"]
    qc_columns = [
        "Starch_type", "Rheology_condition_ID", "N_points", "Reference_strain_pct", "Reference_angular_frequency_rad_s",
        "G_prime_mean", "G_prime_CV_pct", "G_prime_drift_pct", "G_double_prime_mean", "G_double_prime_CV_pct",
        "G_double_prime_drift_pct", "tan_delta_mean", "tan_delta_CV_pct", "tan_delta_drift_pct",
        "Complex_viscosity_mean", "Max_dynamic_CV_pct", "Max_abs_dynamic_drift_pct", "CV_robust_z", "Drift_robust_z",
        "CV_warning_gt_20pct", "CV_robust_outlier", "Strong_drift_robust_outlier", "Nonphysical_point_count",
        "Dynamic_QC_status", "Dynamic_QC_note",
    ]
    qc = df[qc_columns].copy()
    qc["_QC_priority"] = (qc["Dynamic_QC_status"] != "Pass").astype(int)
    qc = qc.sort_values(["_QC_priority", "Max_dynamic_CV_pct"], ascending=[False, False]).drop(columns="_QC_priority")
    return df, qc, distributions


def aggregate_replicates(features: pd.DataFrame, kind: str) -> pd.DataFrame:
    id_cols = ["Rheology_condition_ID", "Starch_type", "Concentration_pct", "Preparation_T_C", "Rheology_T_C"]
    exclude = set(id_cols + ["Rheology_replicate", "Raw_sample_name"])
    numeric_cols = [c for c in features.select_dtypes(include=[np.number, "bool"]).columns if c not in exclude]
    categorical = [c for c in features.columns if c not in exclude and c not in numeric_cols and c != "Rheology_replicate"]
    rows = []
    for key, group in features.groupby(id_cols, dropna=False, sort=False):
        row = dict(zip(id_cols, key))
        nrep = int(group["Rheology_replicate"].nunique())
        row[f"{kind}_n_replicates"] = nrep
        for col in numeric_cols:
            vals = pd.to_numeric(group[col], errors="coerce").dropna()
            row[col] = float(vals.mean()) if len(vals) else np.nan
            if col in {"K_PL", "n_PL", "tau0_HB", "K_HB", "n_HB", "G_prime_ref", "G_double_prime_ref", "tan_delta_ref", "G_complex_ref"}:
                row[f"{col}_SD"] = float(vals.std(ddof=1)) if len(vals) > 1 else np.nan
                row[f"{col}_CV"] = float(vals.std(ddof=1) / vals.mean()) if len(vals) > 1 and vals.mean() != 0 else np.nan
        for col in categorical:
            vals = group[col].dropna().astype(str)
            row[col] = vals.mode().iloc[0] if len(vals) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def build_combined_rheology_fingerprint(shear_features, dynamic_features):
    keys = ["Rheology_condition_ID", "Starch_type", "Concentration_pct", "Preparation_T_C", "Rheology_T_C"]
    combined = shear_features.merge(dynamic_features, on=keys, how="outer", suffixes=("_steady", "_dynamic"))
    combined["Has_steady"] = combined["K_PL"].notna()
    combined["Has_dynamic"] = combined["G_prime_mean"].notna()
    combined["Overall_Rheology_QC"] = np.select(
        [combined.Has_steady & combined.Has_dynamic, combined.Has_steady | combined.Has_dynamic],
        ["Both_available", "Partial"], default="Missing_both",
    )
    eta100 = pd.to_numeric(combined.get("eta_100"), errors="coerce")
    eta_star = pd.to_numeric(combined.get("Complex_viscosity_100rad_s"), errors="coerce")
    valid = (eta100 > 0) & (eta_star > 0)
    combined["Dynamic_steady_viscosity_ratio_100"] = np.where(valid, eta_star / eta100, np.nan)
    combined["Dynamic_steady_log_viscosity_difference_100"] = np.where(valid, np.log10(eta_star) - np.log10(eta100), np.nan)
    combined["Cox_Merz_ratio_100"] = combined["Dynamic_steady_viscosity_ratio_100"]
    combined["Cox_Merz_log_deviation_100"] = combined["Dynamic_steady_log_viscosity_difference_100"]
    hb_reliable = pd.to_numeric(combined.get("HB_fit_success"), errors="coerce").fillna(0) > 0.5
    gp = pd.to_numeric(combined.get("G_prime_mean"), errors="coerce")
    tau0 = pd.to_numeric(combined.get("tau0_HB"), errors="coerce")
    combined["Yield_elastic_ratio"] = np.where(hb_reliable & (gp > 0), tau0 / gp, np.nan)
    combined["Approx_yield_strain_scale"] = combined["Yield_elastic_ratio"]
    combined["Yield_elastic_ratio_interpretation"] = "Approximate strain scale from steady HB yield stress / small-amplitude G'; not a true oscillatory yield strain"
    combined["Candidate_feature_status"] = "candidate_requires_collinearity_stability_and_ablation_screening"
    return combined


def add_exploratory_gravity_features(master: pd.DataFrame):
    df = master.copy()
    density = 1000.0
    gravity = 9.80665
    height_m = pd.to_numeric(df["Model_height_mm"], errors="coerce") / 1000
    stress_scale = density * gravity * height_m
    gp = pd.to_numeric(df.get("G_prime_mean"), errors="coerce")
    tau0 = pd.to_numeric(df.get("tau0_HB"), errors="coerce")
    hb_reliable = pd.to_numeric(df.get("HB_fit_success"), errors="coerce").fillna(0) > 0.5
    df["Density_kg_m3"] = density
    df["Density_source"] = "assumed"
    df["Gravity_m_s2"] = gravity
    df["Gravitational_stress_scale_Pa"] = stress_scale
    df["Elastic_gravity_ratio"] = np.where((gp > 0) & (stress_scale > 0), gp / stress_scale, np.nan)
    df["Yield_gravity_ratio"] = np.where(hb_reliable & (tau0 >= 0) & (stress_scale > 0), tau0 / stress_scale, np.nan)
    df["Physics_feature_status"] = "exploratory_assumed_density"
    df["Physics_feature_note"] = "rho=1000 kg/m3 is an explicit exploratory assumption; replace with measured density before confirmatory analysis"
    return df


def build_cox_merz_outputs(combined: pd.DataFrame):
    columns = ["Rheology_condition_ID", "Starch_type", "Concentration_pct", "Preparation_T_C", "Rheology_T_C", "eta_100", "Complex_viscosity_100rad_s", "Cox_Merz_ratio_100", "Cox_Merz_log_deviation_100"]
    qc = combined[columns].dropna(subset=["eta_100", "Complex_viscosity_100rad_s"]).copy()
    qc["Log10_eta_100"] = np.nan
    qc["Log10_complex_viscosity_100"] = np.nan
    qc["Reference_y_eq_x"] = np.nan
    qc = qc.sort_values("eta_100").reset_index(drop=True)

    summaries = []
    for factor in ["Starch_type", "Concentration_pct", "Preparation_T_C", "Rheology_T_C"]:
        for level, group in qc.groupby(factor, dropna=False):
            values = group["Cox_Merz_log_deviation_100"].dropna()
            summaries.append({
                "Factor": factor, "Level": level, "N": len(values), "Mean_log_deviation": values.mean(),
                "Median_log_deviation": values.median(), "SD_log_deviation": values.std(ddof=1),
                "Q1_log_deviation": values.quantile(0.25), "Q3_log_deviation": values.quantile(0.75),
                "Median_ratio": group["Cox_Merz_ratio_100"].median(),
            })
    summary = pd.DataFrame(summaries)
    return qc, summary


def build_modulus_feature_sets():
    sets = {
        "Modulus-M1": ["G_prime_mean", "G_double_prime_mean"],
        "Modulus-M2": ["G_prime_mean", "G_double_prime_mean", "tan_delta_mean", "G_complex_mean", "Complex_viscosity_mean"],
        "Modulus-M3": ["G_prime_mean", "G_double_prime_mean", "tan_delta_mean", "G_complex_mean", "Complex_viscosity_mean", "G_prime_CV_pct", "G_prime_drift_pct", "tan_delta_drift_pct"],
        "Modulus-M4": ["G_prime_mean", "G_double_prime_mean", "tan_delta_mean", "G_complex_mean", "Complex_viscosity_mean", "G_prime_CV_pct", "G_prime_drift_pct", "tan_delta_drift_pct", "Cox_Merz_log_deviation_100", "Yield_elastic_ratio", "Elastic_gravity_ratio", "Yield_gravity_ratio"],
    }
    rows = []
    for set_name, features in sets.items():
        for order, feature in enumerate(features, 1):
            rows.append({
                "Feature_set": set_name, "Order": order, "Feature_name": feature,
                "Scope": "printing-sample" if feature in {"Elastic_gravity_ratio", "Yield_gravity_ratio"} else "rheology-condition",
                "Candidate_note": "Candidate only; apply correlation clustering, VIF, stability and ablation before ML",
            })
    return pd.DataFrame(rows)


def build_dynamic_diagnostic(dynamic_features, combined, master):
    point_counts = dynamic_features["N_points"]
    gp_cv = dynamic_features["G_prime_CV_pct"]
    gpp_cv = dynamic_features["G_double_prime_CV_pct"]
    drift_flag = dynamic_features["Strong_drift_robust_outlier"]
    stable_by_starch = dynamic_features.groupby("Starch_type").agg(
        Median_max_CV_pct=("Max_dynamic_CV_pct", "median"),
        Median_abs_drift_pct=("Max_abs_dynamic_drift_pct", "median"),
    ).sort_values(["Median_max_CV_pct", "Median_abs_drift_pct"])
    tan = dynamic_features["tan_delta_mean"].dropna()
    eta_star = dynamic_features["Complex_viscosity_mean"].dropna()
    cox = combined.dropna(subset=["eta_100", "Complex_viscosity_100rad_s", "Cox_Merz_log_deviation_100"])
    log_x = np.log10(cox["eta_100"])
    log_y = np.log10(cox["Complex_viscosity_100rad_s"])
    pearson = float(log_x.corr(log_y, method="pearson"))
    spearman = float(log_x.rank(method="average").corr(log_y.rank(method="average"), method="pearson"))
    within_factor2 = float((cox["Cox_Merz_log_deviation_100"].abs() <= math.log10(2)).mean())
    yield_ratio = combined["Yield_elastic_ratio"].dropna()
    rows = [
        ("Material conditions", len(dynamic_features), "Oscillatory data are available for 540 conditions"),
        ("Valid points per condition", point_counts.mean(), f"Range {point_counts.min():.0f}–{point_counts.max():.0f}"),
        ("Within-run G-prime repeatability", gp_cv.median(), f"Median within-run CV; upper quartile = {gp_cv.quantile(0.75):.3g}%"),
        ("Within-run G-double-prime repeatability", gpp_cv.median(), f"Median within-run CV; upper quartile = {gpp_cv.quantile(0.75):.3g}%"),
        ("Conditions flagged for drift", int(drift_flag.sum()), "Flag threshold is Q3 + 1.5 × IQR of absolute drift. Records are retained."),
        ("Starch with the lowest median maximum CV", stable_by_starch.index[0], f"Median maximum CV = {stable_by_starch.iloc[0,0]:.3g}%"),
        ("Conditions with G-prime above G-double-prime throughout", int(dynamic_features["G_prime_greater_than_G_double_all_points"].sum()), f"/ {len(dynamic_features)} conditions"),
        ("Loss tangent range", f"{tan.min():.4g}–{tan.max():.4g}", f"Median = {tan.median():.4g}"),
        ("Complex viscosity range", f"{eta_star.min():.4g}–{eta_star.max():.4g} Pa·s", "Angular frequency of 100 rad/s"),
        ("9_eta100_vs_etaStar100", pearson, f"Pearson correlation on log10 scale; Spearman = {spearman:.4g}"),
        ("Cox–Merz deviation", cox["Cox_Merz_log_deviation_100"].median(), f"Median log deviation; {within_factor2:.1%} lie within a factor of two"),
        ("Condition with the largest absolute deviation", cox.loc[cox["Cox_Merz_log_deviation_100"].abs().idxmax(), "Rheology_condition_ID"], f"Absolute log deviation = {cox['Cox_Merz_log_deviation_100'].abs().max():.4g}"),
        ("12_tau0_over_Gprime", f"{yield_ratio.min():.4g}–{yield_ratio.max():.4g}", "Approximate strain scale from steady yield stress and small-amplitude elasticity; dynamic yield strain was not measured."),
        ("Gravity descriptors", "Exploratory descriptors", "Density of 1000 kg/m3 is assumed. This input workbook has no morphology labels for comparison."),
        ("Oscillation time axis", "Slopes are computed by point index", "The 36 input blocks each contain ten points at 1% strain and 100 rad/s. Recorded durations vary between 5 and 6 s or are missing."),
    ]
    return pd.DataFrame(rows, columns=["Diagnostic_item", "Result", "Interpretation"]), stable_by_starch.reset_index()


def expand_print_design(print_design: pd.DataFrame, starch_types: Iterable[str]) -> pd.DataFrame:
    starch_df = pd.DataFrame({"Starch_type": list(starch_types)})
    left = starch_df.assign(_cross=1)
    right = print_design.assign(_cross=1)
    expanded = left.merge(right, on="_cross").drop(columns="_cross")
    expanded["Sample_ID"] = expanded.apply(lambda r: f"{r['Starch_type']}_G{int(r['Experiment_group']):03d}", axis=1)
    expanded["Rheology_condition_ID"] = expanded.apply(
        lambda r: condition_id(r["Starch_type"], r["Concentration_pct"], r["Preparation_T_C"], r["Printing_T_C"]), axis=1
    )
    expanded["Group_for_ML"] = expanded["Rheology_condition_ID"]
    return expanded


def match_rheology_to_printing(expanded, rheology_all):
    master = expanded.merge(rheology_all, on=["Rheology_condition_ID", "Starch_type", "Concentration_pct", "Preparation_T_C"], how="left", suffixes=("", "_rheo"))
    master = master.drop(columns=[c for c in ["Rheology_T_C"] if c in master.columns])
    master["Matched_steady"] = master["K_PL"].notna() if "K_PL" in master else False
    master["Matched_dynamic"] = master["G_prime_ref"].notna() if "G_prime_ref" in master else False
    master["Matched_both"] = master["Matched_steady"] & master["Matched_dynamic"]
    master["Overall_Rheology_QC"] = np.select(
        [master["Matched_both"], master["Matched_steady"] | master["Matched_dynamic"]],
        ["Both_available", "Partial"], default="Missing_both"
    )
    return master


def build_coverage(starch_types, print_design, steady_features, dynamic_features):
    used = set(zip(print_design["Concentration_pct"].astype(int), print_design["Preparation_T_C"].astype(int), print_design["Printing_T_C"].astype(int)))
    steady_ids = set(steady_features["Rheology_condition_ID"])
    dynamic_ids = set(dynamic_features["Rheology_condition_ID"])
    rows = []
    for starch, c, h, t in product(starch_types, sorted(ALLOWED_CONCENTRATIONS), sorted(ALLOWED_PREPARATION_T), sorted(ALLOWED_RHEOLOGY_T)):
        rid = condition_id(starch, c, h, t)
        rows.append({
            "Starch_type": starch, "Concentration_pct": c, "Preparation_T_C": h, "Rheology_T_C": t,
            "Rheology_condition_ID": rid,
            "Steady_available": "Yes" if rid in steady_ids else "No",
            "Dynamic_available": "Yes" if rid in dynamic_ids else "No",
            "Both_available": "Yes" if rid in steady_ids and rid in dynamic_ids else "No",
            "Used_in_printing": "Yes" if (c, h, t) in used else "No",
        })
    return pd.DataFrame(rows)


def data_dictionary(master_columns):
    definitions = {
        "Sample_ID": ("Unique printing-specimen identifier", "Dimensionless", "Derived", "Starch type and printing group", "Yes"),
        "Rheology_condition_ID": ("Rheological condition identifier", "Dimensionless", "Derived", "Starch, concentration, preparation temperature and rheology or printing temperature", "Yes"),
        "Group_for_ML": ("Group identifier for model evaluation", "Dimensionless", "Derived", "Rheology_condition_ID groups records from the same measured state.", "Yes"),
        "Layer_height_code": ("Layer-height design code", "Code", "Reported", "Code in the printing records, without a length unit.", "Yes"),
        "Layer_height_mm": ("Layer height in millimetres", "mm", "Unavailable", "The source workbook does not contain a code-to-length mapping.", "No"),
        "K_PL": ("Power-law consistency coefficient", "Pa·s^n", "Derived", "τ=Kγ̇^n", "Yes"),
        "n_PL": ("Power-law flow behaviour index", "Dimensionless", "Derived", "τ=Kγ̇^n", "Yes"),
        "tau0_HB": ("Herschel–Bulkley yield stress", "Pa", "Derived", "τ=τ0+Kγ̇^n", "Yes"),
        "G_prime_ref": ("Reference storage modulus", "Pa", "Derived", "Mean of ten points at 100 rad/s and 1% strain", "Yes"),
        "G_double_prime_ref": ("Reference loss modulus", "Pa", "Derived", "Mean of ten points at 100 rad/s and 1% strain", "Yes"),
        "tan_delta_ref": ("Loss tangent", "Dimensionless", "Derived", "Mean of pointwise G-double-prime / G-prime", "Yes"),
        "G_complex_ref": ("Complex modulus magnitude", "Pa", "Derived", "Mean of pointwise sqrt(G-prime^2 + G-double-prime^2)", "Yes"),
        "G_prime_drift_pct": ("Storage modulus drift", "%", "Derived", "(Mean of last three points minus mean of first three) / mean of first three × 100%", "Candidate"),
        "G_double_prime_drift_pct": ("Loss modulus drift", "%", "Derived", "(Mean of last three points minus mean of first three) / mean of first three × 100%", "Candidate"),
        "tan_delta_drift_pct": ("Loss tangent drift", "%", "Derived", "(Mean of last three points minus mean of first three) / mean of first three × 100%", "Candidate"),
        "Complex_viscosity_mean": ("Mean magnitude of complex viscosity", "Pa·s", "Derived", "Mean of pointwise |G*| / angular frequency in rad/s", "Candidate"),
        "Cox_Merz_ratio_100": ("Oscillatory-to-steady viscosity ratio", "Dimensionless", "Derived", "|η*(100 rad/s)| / η(100 s^-1), a deviation measure", "Candidate"),
        "Cox_Merz_log_deviation_100": ("Log-scale Cox–Merz deviation", "Difference in log10(Pa·s)", "Derived", "log10|η*| minus log10η", "Candidate"),
        "Yield_elastic_ratio": ("Yield-to-elasticity scale", "Dimensionless", "Derived", "Herschel–Bulkley yield stress divided by mean storage modulus", "Candidate"),
        "Approx_yield_strain_scale": ("Approximate characteristic strain", "Dimensionless", "Derived", "Ratio of steady Herschel–Bulkley yield stress to small-amplitude storage modulus.", "Candidate"),
        "Elastic_gravity_ratio": ("Elastic-to-gravity ratio", "Dimensionless", "Exploratory derived value", "G'/(rho*g*H)，density assumed to be 1000 kg/m3", "Candidate"),
        "Yield_gravity_ratio": ("Yield-to-gravity ratio", "Dimensionless", "Exploratory derived value", "tau0/(rho*g*H)，density assumed to be 1000 kg/m3", "Candidate"),
        "Density_kg_m3": ("Density", "kg/m3", "Exploratory assumption", "Assumed density of 1000 kg/m3", "No"),
        "Density_source": ("Density source", "Dimensionless", "Derived flag", "Indicates use of an assumed density", "No"),
    }
    rows = []
    for col in master_columns:
        chinese, unit, source_type, formula, ml = definitions.get(col, (col, "See field name", "Reported or derived", "See the input sheets and calculation code", "Depends on model target"))
        rows.append({
            "Variable": col, "Description": chinese, "Unit": unit, "Source_or_type": source_type,
            "Formula_or_definition": formula, "Original_variable": "Yes" if source_type == "Reported" else "No",
            "Derived_variable": "Yes" if source_type == "Derived" else "No", "Recommended_for_ML": ml,
            "Notes": "Fixed-frequency stability at 100 rad/s. Screen closely related oscillatory descriptors before model fitting." if col.startswith("G_") or "tan_delta" in col or "phase_angle" in col or "elastic_fraction" in col else "",
        })
    return pd.DataFrame(rows)


def run_quality_control(print_design, expanded, shear_long, dynamic_long, shear_features, dynamic_features, rheology_all, master, naming_qc):
    unique_print_material = print_design[["Concentration_pct", "Preparation_T_C", "Printing_T_C"]].drop_duplicates()
    steady_ids, dynamic_ids = set(shear_features.Rheology_condition_ID), set(dynamic_features.Rheology_condition_ID)
    both_ids = steady_ids & dynamic_ids
    n = len(master)
    summary = [
        ("Printing_design_total", len(print_design), "expected 59"),
        ("Starch_types", expanded.Starch_type.nunique(), "expected 10"),
        ("Expanded_samples", len(expanded), "must equal 590"),
        ("Unique_material_conditions_in_59", len(unique_print_material), "C×Preparation_T×Printing_T"),
        ("Unique_rheology_ID_used", expanded.Rheology_condition_ID.nunique(), "includes starch type"),
        ("Steady_conditions_available", len(steady_ids), "condition-level"),
        ("Dynamic_conditions_available", len(dynamic_ids), "condition-level"),
        ("Both_conditions_available", len(both_ids), "condition-level"),
        ("Steady_only_conditions", len(steady_ids - dynamic_ids), "condition-level"),
        ("Dynamic_only_conditions", len(dynamic_ids - steady_ids), "condition-level"),
        ("Matched_steady_samples", int(master.Matched_steady.sum()), f"{master.Matched_steady.mean():.1%}"),
        ("Matched_dynamic_samples", int(master.Matched_dynamic.sum()), f"{master.Matched_dynamic.mean():.1%}"),
        ("Matched_both_samples", int(master.Matched_both.sum()), f"{master.Matched_both.mean():.1%}"),
        ("Missing_steady_samples", int((~master.Matched_steady).sum()), f"{(~master.Matched_steady).mean():.1%}"),
        ("Missing_dynamic_samples", int((~master.Matched_dynamic).sum()), f"{(~master.Matched_dynamic).mean():.1%}"),
        ("Missing_both_samples", int((~master.Matched_steady & ~master.Matched_dynamic).sum()), f"{((~master.Matched_steady) & (~master.Matched_dynamic)).mean():.1%}"),
        ("Naming_anomalies_or_duplicates", int((naming_qc.Correction_applied.astype(bool) | naming_qc.Parsing_status.isin(["Unparsed", "Invalid_level", "Duplicate_source_column", "Corrected_with_note"])).sum()), "true corrections, invalid names, or exact duplicate columns"),
        ("Layer_height_mm_available", int(print_design.Layer_height_mm.notna().sum()), "0 means mapping is required"),
        ("Edge_length_nonmissing", int(expanded.Edge_length_mm.notna().sum()), "printing-result target values in expanded data"),
        ("Edge_height_nonmissing", int(expanded.Edge_height_mm.notna().sum()), "printing-result target values in expanded data"),
        ("Center_height_nonmissing", int(expanded.Center_height_mm.notna().sum()), "printing-result target values in expanded data"),
        ("Line_width_nonmissing", int(expanded.Line_width_mm.notna().sum()), "printing-result target values in expanded data"),
    ]
    summary_df = pd.DataFrame(summary, columns=["QC_metric", "Value", "Note"])
    missing = master.loc[~master.Matched_both, ["Sample_ID", "Rheology_condition_ID", "Matched_steady", "Matched_dynamic"]].copy()
    return summary_df, missing


def build_diagnostic(summary_df, shear_features, dynamic_features, master, naming_qc):
    best_counts = shear_features["Best_constitutive_model"].value_counts(dropna=False).to_dict()
    poor_steady = int((shear_features.get("Steady_QC", pd.Series(dtype=str)) != "Pass").sum())
    poor_dynamic = int((dynamic_features.get("Dynamic_QC", pd.Series(dtype=str)) != "Pass").sum())
    rows = [
        ("Linked records", "The 590 printing records are linked by starch, concentration, preparation temperature and print temperature."),
        ("Oscillation protocol", "Ten-point stability measurement at 100 rad/s and 1% strain. Frequency-sweep descriptors are unavailable."),
        ("Layer-height code", "Only the codes -1, 0 and 1 are recorded. Layer_height_mm remains missing."),
        ("Power-law vs HB", f"Condition counts: {best_counts}. Herschel–Bulkley is selected when AIC improves by at least 2 and the fit does not reach a parameter bound."),
        ("Replication", "One independent rheology run was identified per condition. Between-run SD and CV remain missing. Ten-point oscillatory CV measures within-run stability."),
        ("Data review", f"Steady-shear review flags = {poor_steady}, oscillatory review flags = {poor_dynamic}, corrected, unmatched or duplicate source labels = {int((naming_qc.Correction_applied.astype(bool) | naming_qc.Parsing_status.isin(['Unparsed', 'Invalid_level', 'Duplicate_source_column', 'Corrected_with_note'])).sum())}。"),
        ("Scope of the linked workbook", "Rheology and print settings are linked. Edge length, edge height, centre height and line width are blank in these inputs. For state-grouped analysis use Rheology_condition_ID."),
    ]
    return pd.DataFrame(rows, columns=["Topic", "Diagnostic_conclusion"])


def _clean_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace([np.inf, -np.inf], np.nan)


def export_results(tables: dict[str, pd.DataFrame], output_dir: Path, manifest: dict):
    data_dir = output_dir / "processed_csv"
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for sheet, df in tables.items():
        path = data_dir / f"{sheet}.csv"
        _clean_for_csv(df).to_csv(path, index=False, encoding="utf-8-sig")
        paths[sheet] = str(path)
    manifest["tables"] = paths
    manifest_path = output_dir / "workbook_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def parse_args():
    parser = argparse.ArgumentParser(description="Build rheology-printing master database")
    parser.add_argument("--print", dest="print_path", type=Path)
    parser.add_argument("--shear", dest="shear_path", type=Path)
    parser.add_argument("--dynamic", dest="dynamic_path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if all((args.print_path, args.shear_path, args.dynamic_path, args.output_dir)):
        print_path, shear_path, dynamic_path, output_dir = args.print_path, args.shear_path, args.dynamic_path, args.output_dir
    else:
        print_path, shear_path, dynamic_path, output_dir = choose_files_gui()
    output_dir.mkdir(parents=True, exist_ok=True)
    converted = output_dir / "converted_readonly_copies"
    print_xlsx = ensure_xlsx(print_path, converted)
    shear_xlsx = ensure_xlsx(shear_path, converted)
    dynamic_xlsx = ensure_xlsx(dynamic_path, converted)

    print_design = load_print_design(print_xlsx)
    shear_long, shear_qc, shear_map = load_shear_rheology(shear_xlsx)
    dynamic_protocol, dynamic_protocol_report, dynamic_protocol_blocks = inspect_dynamic_protocol(dynamic_xlsx)
    dynamic_long, dynamic_qc, dynamic_map = load_dynamic_rheology(dynamic_xlsx, dynamic_protocol)
    starch_types = _organized_sheet_names(shear_xlsx)
    dynamic_starches = _organized_sheet_names(dynamic_xlsx)
    if set(starch_types) != set(dynamic_starches):
        print(f"WARNING: starch sheet mismatch: steady={starch_types}, dynamic={dynamic_starches}", file=sys.stderr)

    shear_rep = extract_shear_features(shear_long)
    dynamic_rep = extract_dynamic_features(dynamic_long)
    shear_features = aggregate_replicates(shear_rep, "steady")
    dynamic_features = aggregate_replicates(dynamic_rep, "dynamic")
    dynamic_features, dynamic_rheology_qc, dynamic_qc_distributions = apply_dynamic_qc(dynamic_features)
    rheology_all = build_combined_rheology_fingerprint(shear_features, dynamic_features)

    expanded = expand_print_design(print_design, starch_types)
    master = match_rheology_to_printing(expanded, rheology_all)
    master = add_exploratory_gravity_features(master)
    used_ids = set(expanded.Rheology_condition_ID)
    rheology_used = rheology_all[rheology_all.Rheology_condition_ID.isin(used_ids)].copy()
    mapping = expanded.groupby("Rheology_condition_ID", as_index=False).agg(
        Starch_type=("Starch_type", "first"),
        N_printing_conditions=("Experiment_group", "nunique"),
        Experiment_groups=("Experiment_group", lambda s: ", ".join(f"G{int(x):03d}" for x in sorted(set(s)))),
    )
    naming_qc = pd.concat([shear_qc, dynamic_qc], ignore_index=True)
    starch_mapping = pd.concat([shear_map, dynamic_map], ignore_index=True).drop_duplicates()
    coverage = build_coverage(starch_types, print_design, shear_features, dynamic_features)
    matching_summary, matching_missing = run_quality_control(
        print_design, expanded, shear_long, dynamic_long, shear_features, dynamic_features, rheology_all, master, naming_qc
    )
    matching_qc = pd.concat([
        matching_summary.assign(Record_type="Summary"),
        matching_missing.rename(columns={"Sample_ID": "QC_metric", "Rheology_condition_ID": "Value"}).assign(
            Note=lambda x: "steady=" + x["Matched_steady"].astype(str) + "; dynamic=" + x["Matched_dynamic"].astype(str),
            Record_type="Unmatched_sample",
        )[["QC_metric", "Value", "Note", "Record_type"]],
    ], ignore_index=True)
    diagnostic = build_diagnostic(matching_summary, shear_features, dynamic_features, master, naming_qc)
    cox_merz_qc, cox_merz_summary = build_cox_merz_outputs(rheology_all)
    modulus_feature_sets = build_modulus_feature_sets()
    dynamic_diagnostic, starch_stability = build_dynamic_diagnostic(dynamic_features, rheology_all, master)

    preferred_master = [
        "Sample_ID", "Starch_type", "Experiment_group", "Printing_condition_ID", "Rheology_condition_ID", "Group_for_ML",
        "Concentration_pct", "Preparation_T_C", "Printing_T_C",
        "Printing_speed_mm_s", "Nozzle_diameter_mm", "Layer_height_code", "Layer_height_mm", "Model_height_mm",
    ]
    result_cols = ["Completion_status", "Edge_length_mm", "Edge_height_mm", "Center_height_mm", "Line_width_mm"]
    tail = [c for c in master.columns if c not in preferred_master + result_cols]
    master = master[[c for c in preferred_master if c in master] + tail + [c for c in result_cols if c in master]]
    dictionary = data_dictionary(master.columns)

    structure_report = pd.DataFrame([
        {"Source_file": shear_path.name, "Source_sheet": "organized starch sheets", "Observed_structure": "Rows 2:41 are shear rates 0.1–100 1/s; columns are conditions; values are viscosity Pa·s", "Scientific_interpretation": "Steady shear viscosity; shear stress reliably calculated as η×γ̇"},
        {"Source_file": dynamic_path.name, "Source_sheet": "raw Sheet1 + organized starch sheets", "Observed_structure": "Raw audit: 36 blocks, all 10 points/1% strain/100 rad/s; duration metadata includes 5 s, 6 s and missing. Curated sheets: 54 blocks × 10 points per starch", "Scientific_interpretation": "Fixed-frequency small-amplitude oscillatory stability, not frequency sweep; point-index slopes used because time is not safely assignable"},
        {"Source_file": print_path.name, "Source_sheet": "Sheet1", "Observed_structure": "59 printing groups; layer-height field has codes -1/0/1", "Scientific_interpretation": "Cross join with 10 starch sheets; physical layer height unavailable"},
    ])

    tables = {
        "01_PrintDesign_59": print_design,
        "02_PrintExpanded_590": expanded,
        "03_Shear_Raw_Long": shear_long,
        "04_Dynamic_Raw_Long": dynamic_long,
        "05_Shear_Replicate": shear_rep,
        "05_Shear_Features": shear_features,
        "06_Dynamic_Replicate": dynamic_rep,
        "06_Dynamic_Features": dynamic_features,
        "07_Rheology_All": rheology_all,
        "07b_Rheology_Used": rheology_used,
        "08_Master_590": master,
        "09_Rheology_Print_Map": mapping,
        "10_Naming_QC": naming_qc,
        "11_Matching_QC": matching_qc,
        "12_Data_Dictionary": dictionary,
        "13_Coverage_Matrix": coverage,
        "14_Starch_Mapping": starch_mapping,
        "15_Structure_Report": structure_report,
        "16_Diagnostic": diagnostic,
        "Dynamic_Rheology_QC": dynamic_rheology_qc,
        "Dynamic_QC_Distributions": dynamic_qc_distributions,
        "Combined_Rheology_Fingerprint": rheology_all,
        "Cox_Merz_QC": cox_merz_qc,
        "Cox_Merz_Summary": cox_merz_summary,
        "Modulus_Feature_Sets": modulus_feature_sets,
        "Dynamic_Protocol": dynamic_protocol_report,
        "Dynamic_Protocol_Blocks": dynamic_protocol_blocks,
        "Dynamic_Diagnostic": dynamic_diagnostic,
        "Starch_Stability": starch_stability,
    }
    manifest = {
        "output_filename": "Food3D_590_rheology_printing_dataset.xlsx",
        "random_seed": RANDOM_SEED,
        "source_files": [str(print_path), str(shear_path), str(dynamic_path)],
        "starch_types": starch_types,
        "summary": dict(zip(matching_summary.QC_metric, matching_summary.Value)),
    }
    manifest_path = export_results(tables, output_dir, manifest)

    summary = manifest["summary"]
    print("========== DATA INTEGRATION SUMMARY ==========")
    print(f"Starch types: {summary['Starch_types']}")
    print(f"Original printing design: {summary['Printing_design_total']} conditions")
    print(f"Expanded printing samples: {summary['Expanded_samples']}")
    print(f"Unique material conditions used in printing: {summary['Unique_material_conditions_in_59']}")
    print(f"Total rheological conditions available: {len(rheology_all)}")
    print(f"Matched steady rheology: {summary['Matched_steady_samples']} / {len(master)}")
    print(f"Matched oscillatory rheology: {summary['Matched_dynamic_samples']} / {len(master)}")
    print(f"Matched both: {summary['Matched_both_samples']} / {len(master)}")
    print(f"Naming anomalies/duplicate records: {summary['Naming_anomalies_or_duplicates']}")
    print(f"Manifest: {manifest_path}")
    print("=============================================")


if __name__ == "__main__":
    main()
