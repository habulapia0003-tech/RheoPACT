# Visual-quality and print-outcome evaluation records

## Study populations

- The print-outcome dataset contains 590 attempts, including 518 successful extrusions and 72 non-extrudable attempts.
- The visual-quality analysis uses the 518 successfully extruded specimens and 50 analysis scores per specimen. The partition contains 419 development and 99 test specimens.
- Material identity is excluded from the primary models.

The [study data guide](../data/README.md#measurement-and-scoring-records)
describes the visual-score records.

## Visual-quality model

- Selected model: HistGradientBoosting, selected by five-fold training GroupKFold RMSE.
- Training grouped-CV: R2 = 0.8496, RMSE = 7.2259, MAE = 4.6338.
- Fixed 99-print test: R2 = 0.8175, RMSE = 8.7744, MAE = 5.0623.
- Test-set permutation importance measures each phenotype domain's contribution to predictive performance.

## D1–D5-derived objective quality score

- `OQI_D1D5_OOF`: five-fold grouped out-of-fold prediction of the specimen mean from D1–D5 for all 518 successful prints.
- `OQI_D1D5_STRICT_FIG5`: inner-OOF values for the 419 training prints and train-only-to-test predictions for the 99 fixed test prints.
- The 72 non-extrudable attempts are assigned OQI zero for integrated print-outcome regression. The D1–D5 scoring model is fitted on the successfully extruded specimens.
- Grouped out-of-fold agreement with specimen mean visual scores for the 518 prints: R2 = 0.8536, RMSE = 7.3700, MAE = 4.7465.

## Candidate-group evaluation

The evaluation uses all 590 printing attempts. Each candidate group combines
`Rheology_ID` with the five process settings, and all records in a group stay
in the same partition. A rheological state can occur in both partitions at
different process settings.

- Candidate-group overlap between training and test: zero.
- Process only, selected GradientBoosting: training grouped-CV R2 = 0.1186; test R2 = 0.1355, RMSE = 24.7207, MAE = 19.6020.
- Rheology + process, selected ExtraTrees: training grouped-CV R2 = 0.8103; test R2 = 0.6971, RMSE = 14.6339, MAE = 8.8311.

An exploratory split shared rheological states for 75% of its test rows and
shared complete predictor vectors for 10 test rows. Its R² was 0.8522 with
the original target and 0.8496 with the revised OQI target. The reported
candidate-group evaluation separates complete candidate groups and gives
R² = 0.6971 with the revised target.

## Complete rheological-state holdout

The integrated outcome includes the 72 non-extrudable attempts with OQI zero.
The supplementary analysis of quality after extrusion uses 518 prints.

The outer evaluation split is exclusive by `Rheology_ID`; no rheological condition occurs in both training and test sets. The split contains 471 training attempts and 119 test attempts. D1–D5 objective targets are regenerated inside this outer split.

Integrated outcome regression:

- Process only, selected model GradientBoosting: training grouped-CV R2 = 0.0535; test R2 = 0.1192, RMSE = 24.3399, MAE = 18.8200.
- Rheology + process, selected model ExtraTrees: training grouped-CV R2 = 0.7316; test R2 = 0.5039, RMSE = 18.2663, MAE = 11.4829.
- Delta test R2 = +0.3847; RMSE reduction = 6.0736.
- Group-bootstrap 95% CI for rheology + process test R2: 0.1855 to 0.7556.

Extrusion-failure classification:

- Process only, selected model SVC: test failure PR-AUC = 0.1779; failure recall = 0; specificity = 1.0000.
- Rheology + process, selected model Random Forest: test failure PR-AUC = 0.7798; ROC-AUC = 0.9485; balanced accuracy = 0.7692; failure recall = 0.5385; specificity = 1.0000.
