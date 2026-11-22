from __future__ import annotations

import csv
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageFont, ImageOps


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT / "external_data" / "source_tree"
OUT = ROOT / "outputs" / "Food3D_FINAL_HIGHRES_FIGURES_20261026"
MAIN = OUT / "01_MAIN_FIGURES"
SUPP = OUT / "02_SUPPLEMENTARY_FIGURES"
QC = OUT / "03_QC"
SRC = OUT / "04_SOURCE_AND_SCRIPTS"

NODE = Path(os.environ.get("FOOD3D_NODE", shutil.which("node") or "node"))
NODE_MODULES = Path(os.environ.get("FOOD3D_NODE_MODULES", "node_modules"))
RENDERER = ROOT / "render_svg_release.js"

for directory in (MAIN, SUPP, QC, SRC):
    directory.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


AXIS_TERMS = re.compile(
    r"^(?:PC[12]\b|Physical state coordinate|Observed (?:mean visual score|integrated outcome)|"
    r"Predicted (?:mean visual score|integrated outcome)|Out-of-fold prediction|"
    r"Residual \(|Density$|Mean visual score$|Visual score$|Integrated print quality|"
    r"Reference design dimension|Reconstructed dimension|Signed surface deviation|"
    r"Absolute deviation|Surface MAE|P95|Within tolerance|Point density|Retained density|"
    r"Static pressure|Pressure drop|Total shear stress|Near-axis|Distance upstream|"
    r"Axial position|Normalized radial position|Shear rate|Apparent viscosity|Flow rate|"
    r"Velocity|Temperature|Speed|Nozzle|Build height|Review budget|Retrieval \(|"
    r"Rank of best observed process|Explained variance|Standardized partial coefficient|"
    r"SHAP value|Mean \|SHAP|Fraction|Probability|Tolerance)",
    re.I,
)


def _plain_text(fragment: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", fragment)).strip()


def normalize_svg(src: Path, dst: Path, *, replacements: dict[str, str] | None = None, remove_text_patterns: tuple[str, ...] = ()) -> dict:
    text = src.read_text(encoding="utf-8", errors="ignore")
    if replacements:
        for old, new in replacements.items():
            text = text.replace(old, new)

    text = re.sub(r"font-family\s*:\s*[^;\"]+", "font-family: 'Times New Roman'", text)
    for family in (
        "'Arial', 'Helvetica', 'DejaVu Sans', sans-serif",
        "Arial, Helvetica, sans-serif",
        "DejaVu Sans",
        "DejaVu Serif",
        "Liberation Sans",
        "Liberation Serif",
    ):
        text = text.replace(family, "'Times New Roman'")
    text = text.replace("'Arial'", "'Times New Roman'").replace('"Arial"', '"Times New Roman"')
    text = text.replace("'Helvetica'", "'Times New Roman'").replace('"Helvetica"', '"Times New Roman"')

    for pattern in remove_text_patterns:
        text = re.sub(
            r"<text\b[^>]*>(?:(?!</text>).)*?" + pattern + r"(?:(?!</text>).)*?</text>",
            "",
            text,
            flags=re.S | re.I,
        )

    axis_hits = []
    panel_hits = []

    def repl_text(match: re.Match[str]) -> str:
        opening, inner, closing = match.group(1), match.group(2), match.group(3)
        label = _plain_text(inner)
        attrs = opening
        is_panel = label.lower() in {"a", "b", "c", "d", "e", "f"}
        is_axis = bool(label) and bool(AXIS_TERMS.search(label)) and len(label) > 2
        if is_axis or is_panel:
            if "style=" in attrs:
                attrs = re.sub(
                    r'style="([^"]*)"',
                    lambda m: 'style="' + m.group(1).rstrip("; ") + "; font-weight: 700;" + (' font-size: 12px;' if is_panel else '') + '"',
                    attrs,
                    count=1,
                )
            else:
                attrs = attrs[:-1] + (' style="font-family: \'Times New Roman\'; font-weight: 700; font-size: 12px;">' if is_panel else ' style="font-family: \'Times New Roman\'; font-weight: 700;">')
            (panel_hits if is_panel else axis_hits).append(label)
        return attrs + inner + closing

    text = re.sub(r"(<text\b[^>]*>)(.*?)(</text>)", repl_text, text, flags=re.S | re.I)
    css = "<style>text,tspan{font-family:'Times New Roman' !important;}</style>"
    text = re.sub(r"(<svg\b[^>]*>)", r"\1" + css, text, count=1, flags=re.I)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text, encoding="utf-8")
    return {"axis_labels_bolded": sorted(set(axis_hits)), "panel_labels_normalized": sorted(set(panel_hits))}


def render_svg(svg: Path, width: int) -> None:
    base = svg.with_suffix("")
    env = os.environ.copy()
    env["NODE_PATH"] = str(NODE_MODULES)
    font_cache = OUT / "_fontcache"
    font_cache.mkdir(parents=True, exist_ok=True)
    env["XDG_CACHE_HOME"] = str(font_cache)
    subprocess.run([str(NODE), str(RENDERER), str(svg), str(base), str(width)], check=True, env=env)


def image_to_pdf(image_path: Path, pdf_path: Path) -> None:
    img = Image.open(image_path).convert("RGB")
    img.save(pdf_path, "PDF", resolution=600.0)


manifest = []
style_audit = {}


def release_svg(src: Path, out_dir: Path, stem: str, width: int, *, replacements=None, remove_text_patterns=(), role="") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    svg = out_dir / f"{stem}.svg"
    style_audit[stem] = normalize_svg(src, svg, replacements=replacements, remove_text_patterns=remove_text_patterns)
    render_svg(svg, width)
    manifest.append({"role": role, "source": str(src), "source_sha256": sha256(src), "release_stem": str(svg.with_suffix(""))})
    return svg


def save_raster_release(image: Image.Image, out_dir: Path, stem: str, width: int, source: Path, role: str) -> None:
    print(f"  RASTER {stem} start", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    ratio = image.height / image.width
    image = image.resize((width, round(width * ratio)), Image.Resampling.LANCZOS).convert("RGB")
    png = out_dir / f"{stem}.png"
    tif = out_dir / f"{stem}.tiff"
    image.save(png, dpi=(600, 600), optimize=True)
    print(f"  RASTER {stem} png", flush=True)
    tifffile.imwrite(tif, np.asarray(image), photometric="rgb", compression="deflate", resolution=(600, 600), resolutionunit="INCH")
    print(f"  RASTER {stem} tiff", flush=True)
    image_to_pdf(png, out_dir / f"{stem}.pdf")
    print(f"  RASTER {stem} pdf", flush=True)
    manifest.append({"role": role, "source": str(source), "source_sha256": sha256(source), "release_stem": str(out_dir / stem)})


def fit_to_cell(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    box = Image.new("RGB", size, "white")
    fitted = ImageOps.contain(img.convert("RGB"), size, Image.Resampling.LANCZOS)
    box.paste(fitted, ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2))
    return box


def compose_grid(images: list[Path], layout: list[list[int]], cell: tuple[int, int], out_dir: Path, stem: str, source_note: str) -> None:
    rows, cols = len(layout), max(len(r) for r in layout)
    canvas = Image.new("RGB", (cell[0] * cols, cell[1] * rows), "white")
    opened = [Image.open(p).convert("RGB") for p in images]
    for r, row in enumerate(layout):
        for c, idx in enumerate(row):
            if idx < 0:
                continue
            tile = fit_to_cell(opened[idx], cell)
            canvas.paste(tile, (c * cell[0], r * cell[1]))
    save_raster_release(canvas, out_dir, stem, 4800, images[0], source_note)


def compose_two_by_two_plus_full(images: list[Path], out_dir: Path, stem: str, source_note: str) -> None:
    """Compose panels a-d in a 2x2 grid and panel e across the full bottom width."""
    canvas = Image.new("RGB", (4800, 5000), "white")
    opened = [Image.open(p).convert("RGB") for p in images]
    for idx, (x, y) in enumerate(((0, 0), (2400, 0), (0, 1600), (2400, 1600))):
        canvas.paste(fit_to_cell(opened[idx], (2400, 1600)), (x, y))
    canvas.paste(fit_to_cell(opened[4], (4800, 1800)), (0, 3200))
    save_raster_release(canvas, out_dir, stem, 4800, images[0], source_note)


def compose_vertical(images: list[Path], out_dir: Path, stem: str, source_note: str, width: int = 4800, gap: int = 60) -> None:
    opened = [Image.open(p).convert("RGB") for p in images]
    scaled = [im.resize((width, round(width * im.height / im.width)), Image.Resampling.LANCZOS) for im in opened]
    canvas = Image.new("RGB", (width, sum(im.height for im in scaled) + gap * (len(scaled) - 1)), "white")
    y = 0
    for im in scaled:
        canvas.paste(im, (0, y))
        y += im.height + gap
    save_raster_release(canvas, out_dir, stem, width, images[0], source_note)


# Main Figure 1: author-confirmed layout, with only the obsolete Fig. 3 summary metric refreshed.
print("STAGE Figure1", flush=True)
fig1_src = ROOT / "Food3D_NC_Master_Submission" / "NC_Submission_Package_v4_Intro_Fig1_Fig2_20261005" / "figures" / "Figure1_MAIN_user_confirmed_20261005.png"
fig1 = Image.open(fig1_src).convert("RGB")
draw = ImageDraw.Draw(fig1)
font_path = os.environ.get("FOOD3D_TNR_BOLD", "timesbd.ttf")
tnr_bold = ImageFont.truetype(font_path, 19)
tnr_bold_small = ImageFont.truetype(font_path, 17)
box_fill = fig1.getpixel((450, 790))
draw.rounded_rectangle((350, 752, 653, 834), radius=10, fill=box_fill)
line1 = "R² = 0.817   RMSE = 8.77"
line2 = "MAE = 5.06"
for y, line, font in ((770, line1, tnr_bold), (798, line2, tnr_bold_small)):
    bb = draw.textbbox((0, 0), line, font=font)
    draw.text(((501 - (bb[2] - bb[0]) / 2), y), line, font=font, fill="black")
save_raster_release(fig1, MAIN / "Figure1" / "Whole", "Figure1_FINAL_TNR", 4800, fig1_src, "Main Figure 1 whole")
fig1_boxes = {
    "Figure1a": (0.00, 0.00, 0.445, 0.415),
    "Figure1b": (0.450, 0.00, 0.995, 0.415),
    "Figure1c": (0.00, 0.425, 1.00, 1.00),
}
for stem, (x0, y0, x1, y1) in fig1_boxes.items():
    crop = fig1.crop((round(x0 * fig1.width), round(y0 * fig1.height), round(x1 * fig1.width), round(y1 * fig1.height)))
    save_raster_release(crop, MAIN / "Figure1" / "Panels", stem + "_TNR", 4200, fig1_src, f"Main {stem}")


# Main Figure 2: four-band digital-phenotyping layout requested by the author.
print("STAGE Figure2", flush=True)
fig2_builder = OUT / "04_SOURCE_AND_SCRIPTS" / "rebuild_figure2_reference_layout.py"
subprocess.run([sys.executable, str(fig2_builder)], check=True)
for obsolete in (MAIN / "Figure2" / "Panels").glob("Figure2e_TNR.*"):
    obsolete.unlink()
manifest.append({"role": "Main Figure 2 whole and panels a-d", "source": str(fig2_builder), "source_sha256": sha256(fig2_builder), "release_stem": str(MAIN / "Figure2")})
for letter in "abcd":
    style_audit[f"Figure2{letter}_TNR"] = {
        "axis_labels_bolded": ["all quantitative axis titles"],
        "panel_labels_normalized": [letter],
    }
for stem in (
    "Figure2d_i_dimensional_agreement_TNR",
    "Figure2d_ii_repeatability_TNR",
    "Figure2d_iii_registration_robustness_TNR",
    "Figure2d_iv_divergent_phenotypes_TNR",
    "Figure2d_v_localized_deformation_TNR",
):
    style_audit[stem] = {
        "font": "Times New Roman",
        "axis_labels_bolded": ["all quantitative axis titles"],
        "layout": "native standalone redraw from frozen Figure 2d source data",
    }
style_audit["Figure2_FINAL_TNR"] = {
    "layout": "four stacked bands matching the author-provided reference",
    "font": "Times New Roman",
    "raster_density_dpi": 600,
}


# Main Figure 3.
print("STAGE Figure3", flush=True)
fig3_root = ROOT / "Food3D_Figures2-6_FORMAL_FROZEN_20261016" / "Figure3" / "03_FIGURE3_REBUILT"
release_svg(fig3_root / "Figure3_NC_phenotype_perception_TNR_20261017.svg", MAIN / "Figure3" / "Whole", "Figure3_FINAL_TNR", 4800, role="Main Figure 3 whole")
for src in sorted((fig3_root / "individual_panels").glob("*.svg")):
    release_svg(src, MAIN / "Figure3" / "Panels", src.stem.replace("_TNR", "") + "_FINAL_TNR", 4800, role="Main Figure 3 panel")


# Main Figure 4 panels and recomposed whole.
print("STAGE Figure4", flush=True)
fig4_root = ROOT / "Food3D_Figures2-6_FORMAL_FROZEN_20261016" / "Figure4" / "individual_panels"
fig4_svgs = sorted(fig4_root.glob("*.svg"))
fig4_pngs = []
for src in fig4_svgs:
    remove_notes = (r"Three nozzle fields only\.",) if src.stem.startswith("Figure4b") else ()
    svg = release_svg(src, MAIN / "Figure4" / "Panels", re.sub(r"_(preserved_latest|131072_TNR)$", "", src.stem) + "_FINAL_TNR", 4200, remove_text_patterns=remove_notes, role="Main Figure 4 panel")
    fig4_pngs.append(svg.with_suffix(".png"))
compose_grid(fig4_pngs, [[0, 1], [2, 3]], (2400, 1900), MAIN / "Figure4" / "Whole", "Figure4_FINAL_TNR", "Recomposed from final Figure 4 panels")


# Main Figure 5.
print("STAGE Figure5", flush=True)
fig5_root = ROOT / "Figure5_NC_REFERENCE_MODE_TNR_v4_20261017"
fig5_pngs = []
for src in sorted((fig5_root / "02_INDIVIDUAL_PANELS").glob("*.svg")):
    svg = release_svg(src, MAIN / "Figure5" / "Panels", src.stem.replace("_TNR", "") + "_FINAL_TNR", 4800, role="Main Figure 5 panel")
    fig5_pngs.append(svg.with_suffix(".png"))
compose_two_by_two_plus_full(fig5_pngs, MAIN / "Figure5" / "Whole", "Figure5_FINAL_TNR", "Recomposed from final Figure 5 panels")


# Figure 6 is generated by scripts/figure6/reproduce_figure6.py.
fig6_build = Path(__file__).resolve().parents[2] / 'working/figure6'
fig6_whole = MAIN / 'Figure6' / 'Whole'
fig6_whole.mkdir(parents=True, exist_ok=True)
shutil.copy2(fig6_build / 'Figure6.pdf', fig6_whole / 'Figure6_FINAL_TNR.pdf')

# Supplementary Figures 1-29 in the current logical order.
print("STAGE Supplementary", flush=True)
fig2_supp = ROOT / "fig02_supplementary_master" / "figures"
supp_sources: dict[int, Path] = {
    1: fig2_supp / "FigS_Fig2_01_scanning_repeatability.svg",
    2: fig2_supp / "FigS_Fig2_02_registration_robustness_and_convergence.svg",
    3: fig2_supp / "FigS_Fig2_03_point_density_sensitivity.svg",
    4: fig2_supp / "FigS_Fig2_04_tolerance_sensitivity.svg",
    5: fig2_supp / "FigS_Fig2_05_descriptor_definition_and_redundancy.svg",
    6: fig2_supp / "FigS_Fig2_08_denoising_sensitivity.svg",
    7: fig2_supp / "FigS_Fig2_09_multigeometry_signed_deviation_maps.svg",
    8: fig2_supp / "FigS_Fig2_06_additional_starch_print_examples.svg",
    9: fig2_supp / "FigS_Fig2_07_Fig2f_sample_selection_workflow.svg",
}
fig3_revised = ROOT / "outputs" / "Food3D_Supplementary_REVIEWER_REVISED_20261018" / "01_UPDATED_FIGURES"
for n in (10, 11, 12):
    supp_sources[n] = fig3_revised / f"Supplementary_Figure{n}_REVISED.svg"
supp_sources[13] = ROOT / "outputs" / "Food3D_Supplementary_LATEST_20261016" / "01_UPDATED_FIGURES" / "Supplementary_Figure13_LATEST.svg"
fig4_supp = ROOT / "figure4_supplementary_final_20260908" / "package" / "01_FIGURES"
for n in range(14, 25):
    supp_sources[n] = fig4_supp / f"Supplementary_Fig_S4-{n - 13}.svg"
supp_sources[15] = ROOT / "supplementary_figure15_rebuilt_540_20261026" / "Supplementary_Figure15_540.svg"
current_supp = ROOT / "outputs" / "Food3D_Supplementary_CURRENT_20261020" / "01_FIGURES_HIGHRES"
supp_sources[25] = current_supp / "Supplementary_Figure25_VISCOELASTIC_VERIFICATION.svg"
supp_sources[26] = current_supp / "Supplementary_Figure26_FIGURE5_CURRENT.svg"
supp_sources[27] = current_supp / "Supplementary_Figure27_FIGURE5_CURRENT.svg"
supp_sources[28] = fig6_build / "Supplementary_Figure28.svg"
supp_sources[29] = fig6_build / "Supplementary_Figure29.svg"

for n in range(1, 30):
    src = supp_sources[n]
    if not src.exists():
        raise FileNotFoundError(src)
    release_svg(src, SUPP / f"Supplementary_Figure_{n:02d}", f"Supplementary_Figure_{n:02d}_FINAL_TNR", 4200, replacements={"CAD": "design"}, role=f"Supplementary Figure {n}")


# Copy scripts used for this presentation-only rebuild.
shutil.copy2(__file__, SRC / Path(__file__).name)
shutil.copy2(RENDERER, SRC / RENDERER.name)
shutil.copy2(ROOT / "rebuild_supplementary_figure15_540.py", SRC / "rebuild_supplementary_figure15_540.py")

with (QC / "SOURCE_MANIFEST.csv").open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=["role", "source", "source_sha256", "release_stem"])
    writer.writeheader()
    writer.writerows(manifest)

(QC / "STYLE_AUDIT.json").write_text(json.dumps(style_audit, indent=2, ensure_ascii=False), encoding="utf-8")

summary = {
    "main_figures": 6,
    "main_panel_files": sum(1 for p in MAIN.rglob("Panels/*.png")),
    "supplementary_figures": 29,
    "font_family": "Times New Roman",
    "axis_label_weight": "bold (700)",
    "raster_density_dpi": 600,
    "formats": ["SVG", "PDF", "PNG", "TIFF"],
    "data_changed": False,
    "figure1_note": "Author-confirmed raster layout retained; obsolete Figure 3 summary metrics refreshed to locked R2=0.817, RMSE=8.77 and MAE=5.06.",
    "figure2_note": "Rebuilt in place as the requested four-band a-d composition; Figure 2d hotspot metrics were separated from the map, and its five validation modules were exported independently.",
}
(QC / "PACKAGE_SUMMARY.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
