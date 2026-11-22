# Figure 5 current-domain result — FROZEN

Freeze date: 2026-10-15

## Primary scientific question

Given a measured rheological state, can rheology and process information predict the integrated print outcome for process candidates not used for training?

## Frozen population and validation contract

- All 590 printing attempts are included.
- The 72 non-extrudable attempts are retained as observed failure states with protocol-defined integrated outcome zero.
- The remaining 518 outcomes use a D1–D5-derived integrated objective score.
- Material identity is excluded from the primary models.
- The grouping key is the complete `Rheology_ID + process candidate` combination.
- Replicates of the same candidate are never divided between training and test.
- Training n = 469; independent candidate test n = 121.
- Candidate-group overlap between training and test = 0.
- Model family and hyperparameters are selected only by five-fold grouped training CV with the lowest RMSE.

## Frozen primary result

### Rheology + process

- Selected model: ExtraTrees.
- Training grouped-CV R2 = 0.8103420072.
- Training grouped-CV RMSE = 11.4295916106.
- Training grouped-CV MAE = 7.8608040903.
- Independent candidate-test R2 = 0.6970704359.
- Independent candidate-test RMSE = 14.6338839014.
- Independent candidate-test MAE = 8.8311066577.

### Process only comparator

- Selected model: GradientBoosting.
- Training grouped-CV R2 = 0.1185578585.
- Independent candidate-test R2 = 0.1355446379.
- Independent candidate-test RMSE = 24.7206600009.
- Independent candidate-test MAE = 19.6020486285.

## Reporting rule

The manuscript and Figure 5 must report both the grouped-CV and independent candidate-test results. The primary wording is:

> Rheology-informed prediction achieved a grouped-CV R2 of 0.810 and an independent candidate-test R2 of 0.697, compared with 0.119 and 0.136, respectively, for process-only prediction.

The earlier apparent test R2 of 0.852 is not part of this frozen result because its split allowed complete predictor duplicates across training and test. Cross-rheological-state extrapolation is a separate robustness question and must not replace or be conflated with the current-domain process-selection result.

## Change control

This result remains frozen unless at least one of the following changes:

- the frozen raw sensory or D1–D5 database;
- the definition of non-extrudable outcomes;
- the D1–D5 objective-score construction;
- the predictor set;
- the candidate-group definition;
- the train/test split or model-selection protocol.

Any such change requires a new versioned result rather than overwriting this package.
