# Food3D Figure 3–5 update freeze record

Freeze date: 2026-10-15

## Frozen source scope

- Sensory registry: 590 printing attempts × 50 integer ratings = 29,500 ratings.
- Extrudable analysis cohort: 518 successful prints × 50 ratings = 25,900 ratings.
- Non-extrudable states: 72 rows. These are excluded before the D1–D5-to-perception regression.
- Fixed successful-print split: 419 training rows and 99 test rows.
- Material identity is excluded from the primary models.

Original files remain unchanged. Frozen copies and their hashes are stored in `00_FROZEN_SOURCE`.

## Frozen Figure 3 result

- Selected model: HistGradientBoosting, selected by five-fold training GroupKFold RMSE.
- Training grouped-CV: R2 = 0.8496, RMSE = 7.2259, MAE = 4.6338.
- Fixed 99-print test: R2 = 0.8175, RMSE = 8.7744, MAE = 5.0623.
- Interpretation uses model-agnostic permutation importance on the fixed test set and is not causal.

## D1–D5-derived objective quality score

- `OQI_D1D5_OOF`: five-fold group-out-of-fold prediction of the 50-rater mean from D1–D5 for all 518 successful prints.
- `OQI_D1D5_STRICT_FIG5`: inner-OOF values for the 419 training prints and train-only-to-test predictions for the 99 fixed test prints.
- The 72 non-extrudable rows are assigned protocol-defined zero only in the 590-row export. They are not model predictions and are not included in the quality regression.
- Global 518-print OOF agreement with sensory mean: R2 = 0.8536, RMSE = 7.3700, MAE = 4.7465.

## Updated Figure 5 current-domain process-selection analysis

Figure 5 uses all 590 printing attempts. The current-domain primary question is: given a measured rheological state, can the model score process candidates that were not used for training? Exact `Rheology_ID + process` candidate groups, including their replicates, are therefore kept wholly within either training or test; the same rheological state may occur with different process candidates in both partitions by design.

- Candidate-group overlap between training and test: zero.
- Process only, selected GradientBoosting: training grouped-CV R2 = 0.1186; test R2 = 0.1355, RMSE = 24.7207, MAE = 19.6020.
- Rheology + process, selected ExtraTrees: training grouped-CV R2 = 0.8103; test R2 = 0.6971, RMSE = 14.6339, MAE = 8.8311.

The earlier test R2 of 0.8522 is not frozen: 75% of its test rows reused rheological states present in training and 10 test rows duplicated a complete rheology-plus-process predictor vector found in training. Changing only the OQI target changed R2 from 0.8522 to 0.8496, whereas enforcing candidate-group separation reduced it to 0.6971. The decline therefore comes from the evaluation split, not from including the 72 failures or redefining the objective score.

## Updated Figure 5 cross-state robustness analysis

The 72 non-extrudable attempts are retained as observed failure states with protocol-defined integrated outcome zero. The 518-print conditional-quality analysis is retained only as a supplementary sensitivity analysis.

The outer evaluation split is exclusive by `Rheology_ID`; no rheological condition occurs in both training and test sets. The split contains 471 training attempts and 119 test attempts. D1–D5 objective targets are regenerated inside this outer split.

Integrated outcome regression:

- Process only, selected model GradientBoosting: training grouped-CV R2 = 0.0535; test R2 = 0.1192, RMSE = 24.3399, MAE = 18.8200.
- Rheology + process, selected model ExtraTrees: training grouped-CV R2 = 0.7316; test R2 = 0.5039, RMSE = 18.2663, MAE = 11.4829.
- Delta test R2 = +0.3847; RMSE reduction = 6.0736.
- Group-bootstrap 95% CI for rheology + process test R2: 0.1855 to 0.7556.

Extrusion-failure classification:

- Process only, selected model SVC: test failure PR-AUC = 0.1779; failure recall = 0; specificity = 1.0000.
- Rheology + process, selected model Random Forest: test failure PR-AUC = 0.7798; ROC-AUC = 0.9485; balanced accuracy = 0.7692; failure recall = 0.5385; specificity = 1.0000.

An experiment-group split produced a substantially higher apparent regression score, but the audit found repeated rheological conditions across training and test. That estimate is not retained as Figure 5 evidence. Figure 5 has not yet been redrawn in this package.

## Figure 3 figure QA

- Composite and five independently rendered panels are supplied as PNG, TIFF, SVG and PDF.
- Raster resolution: 600 dpi.
- Final composite width: approximately 183 mm.
- SVG retains editable text.
- Times New Roman was resolved from `the system Times New Roman font`; PDF font inspection found TimesNewRomanPS regular, bold and italic subsets.
- The generic Nature figure validator reports the serif font as a font-family failure because its rule accepts only Arial/Helvetica-style sans-serif fonts. This is a documented rule mismatch, not a font-substitution failure; the user-requested Times New Roman requirement is verified directly in SVG and PDF.
