# Model inputs and evaluation

## Visual-quality prediction

- Population: 518 successful extrusions.
- Split: 419 development samples and 99 fixed test samples, grouped by
  `Experiment_group`.
- Inputs: D1-D5 point-cloud phenotype descriptors.
- Target: specimen mean from the visual-score analysis matrix.
- Estimator: `HistGradientBoostingRegressor`.
- Fixed-test metrics: R2 0.8174787, RMSE 8.77438 and MAE 5.06228.

The remaining 72 attempts are retained as extrusion-feasibility records in the
590-attempt process dataset.

## Integrated print-outcome prediction

- Population: all 590 printing attempts.
- Split: 469 development records and 121 candidate-group test records.
  Groups combine the measured rheological condition and the five process settings.
- Quality estimator: `ExtraTreesRegressor(n_estimators=400,
  random_state=20260902)`.
- Grouped-CV R2: approximately 0.8103.
- Candidate-test R2: 0.6970704.
- Feasibility target: observed extrusion success/failure.

The recommender evaluates 243 combinations of the observed process levels, applies
`P(extrudable) >= 0.50`, and ranks feasible candidates by predicted OQI. The
probability is used as a feasibility screen and final tie-breaker.

## Quality scores for the printing comparisons

The [study data guide](../data/README.md#measurement-and-scoring-records)
defines the scores used for paired comparisons and same-setting predictions.
The [reproduction guide](VALIDATION.md) gives the commands for both analyses.

## File locations and units

The analysis scripts write derived files to `working/` or a user-specified
directory. The [accompanying data guide](EXTERNAL_DATA.md) lists the input
paths for PLY scans, the complete state table and COMSOL projects.

G′, G″ and yield stress are in Pa. Viscosities are in Pa·s, the consistency coefficient is in Pa·s<sup><i>n</i></sup>, and the flow index is dimensionless. Print temperature is in °C, speed is in mm/s, and lengths are in mm.
