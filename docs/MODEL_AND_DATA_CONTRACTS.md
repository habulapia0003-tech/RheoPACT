# Model inputs and evaluation

## Figure 3

- Population: 518 successful extrusions.
- Split: 419 development samples and 99 fixed test samples, grouped by
  `Experiment_group`.
- Inputs: D1-D5 point-cloud phenotype descriptors.
- Target: revised mean sensory score.
- Estimator: `HistGradientBoostingRegressor` with the frozen parameters stored
  in the packaged joblib object.
- Fixed-test metrics: R2 0.8174787, RMSE 8.77438 and MAE 5.06228.

The remaining 72 attempts are retained as extrusion-feasibility records in the
590-attempt process dataset.

## Figure 5

- Population: all 590 printing attempts.
- Split: 469 development records and 121 candidate-group test records.
  Groups combine the measured rheological condition and the five process settings.
- Quality estimator: `ExtraTreesRegressor(n_estimators=400,
  random_state=20260902)`.
- Grouped-CV R2: approximately 0.8103.
- Candidate-test R2: 0.6970704.
- Feasibility target: observed extrusion success/failure.

The recommender evaluates 243 process combinations, applies
`P(extrudable) >= 0.50`, and ranks feasible candidates by predicted OQI. The
probability is used as a feasibility screen and final tie-breaker.

## Figure 6

The current Figure 6 source is the twelve-print table in `data/figure6/`. The archived phenotype scorer computes the paired OQI values. The Figure 5 classifier computes extrusion feasibility separately. Extract `Code_and_Models.zip` and run the following command from the `Food3D_GitHub_Code` directory.

```bash
python scripts/figure6/reproduce_figure6.py --prints data/figure6/Figure6_12_print_inputs.csv --scorer models/Figure6_hierarchical_phenotype_OQI_scorer.joblib --recommender models/Figure5_RECOMMENDER_rheology_to_process_topk.joblib --photo data/figure6/photo_plate.png --projection data/figure6/Figure6b_external_PCA.csv --state-map data/state_space --fig5-oqi-scorer models/Figure5_development_phenotype_to_OQI.joblib --out working/figure6
```

## Data preservation

Bundled workbooks and frozen models are read-only research sources. Derived
outputs are written to `working/` or another user-selected output directory.
The raw PLY archive, complete state table and COMSOL projects are indexed in
`data_manifests/` and placed according to `docs/EXTERNAL_DATA.md`.

G′, G″ and yield stress are in Pa. Viscosities are in Pa·s, the consistency coefficient is in Pa·s^n, and the flow index is dimensionless. Print temperature is in °C, speed is in mm/s, and lengths are in mm.
