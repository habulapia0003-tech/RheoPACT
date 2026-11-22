# Print-outcome prediction at held-out candidate settings

## Population and evaluation

- All 590 printing attempts are included.
- The 72 non-extrudable attempts are retained as observed failure states with protocol-defined integrated outcome zero.
- The remaining 518 outcomes use a D1–D5-derived integrated objective score.
- Material identity is excluded from the primary models.
- The grouping key is the complete `Rheology_ID + process candidate` combination.
- Replicates of the same candidate are never divided between training and test.
- Development: 469 attempts. Candidate-group test: 121 attempts.
- Candidate-group overlap between training and test = 0.
- Model family and hyperparameters are selected only by five-fold grouped training CV with the lowest RMSE.

## Model performance

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

The [evaluation records](figure3_figure5_freeze_record.md) give the results
for complete rheological-state holdout separately.
