# Study data

The data follow the measurement and analysis sequence, from reference
geometries and print phenotypes through rheological modelling and process
screening to comparisons at the tested settings.

| Analysis | Dataset | Contents |
| --- | --- | --- |
| Structural measurement | [Reference geometry](reference_geometry/) | Eleven STL designs used for design-referenced measurements |
| Visual-quality modelling | [Analysis ratings](frozen_tables/Food3D_590_visual_ratings_50_assessors.xlsx), [specimen-level inputs](frozen_tables/Food3D_D1D5_NewSensory_518_extrudable_training.csv) and [split assignment](frozen_tables/split_assignment_80_20.csv) | Analysis scores and D1–D5 inputs for the 518 successfully extruded specimens used for modelling, with 419 development and 99 test specimens |
| Rheology and nozzle-flow modelling | [Rheological state space](state_space/) | Measurements from 540 conditions, PCA coordinates and projection parameters for 131,072 reduced-order states, and simulation inputs |
| Print-outcome prediction | [Printing and rheology workbook](frozen_tables/Food3D_590_merged_dataset.xlsx) | 590 print attempts, rheological measurements, and specimen and condition identifiers |
| Paired printing and same-setting prediction | [Print records](figure6/Figure6_12_prints.csv), [paired outcomes](figure6/Figure6_paired_results.csv) and [predicted and print-derived OQI](figure6/Figure6e_same_setting_OQI.csv) | One reference and one adjusted print per food matrix, with D1–D5, paired OQI and predictions for the six tested adjusted settings |

The starch codes MS, WD, YM, MLS and XM identify cassava, pea, maize, potato and
wheat, respectively. HS, SP, AS, ADSP and HDSP identify the modified starches.

## Measurement and scoring records

Supplementary Data 1 contains the Excel workbook used to collect visual scores
on site. The supplied analysis matrix has different specimen means. The
transformation between the two records is undocumented (Supplementary Table 6).
The visual-quality modelling scripts use the supplied analysis matrix.

The paired comparison uses an Extra Trees scorer to convert measured D1–D5 to
OQI for all twelve prints. The same-setting comparison uses the process-quality
model for prediction and its corresponding D1–D5 scorer for print-derived OQI.
The two comparisons therefore use distinct OQI scales. The tested settings were
ranked retrospectively within the candidate grid.

The same-setting comparison for pumpkin (M04) uses the reported later
rheological measurements. The state-space projection and extrusion screen use
the initial measurements. The link between the later measurements, the
instrument export and the adjusted-print batch is unverified.
[Input records and sensitivity results](figure6/Figure6_summary.json) give both
sets of measurements and the resulting errors.

See the [figure index](../docs/FIGURE_CODE_MAP.md) for panel-level sources and
the [reproduction guide](../docs/VALIDATION.md) for commands.
