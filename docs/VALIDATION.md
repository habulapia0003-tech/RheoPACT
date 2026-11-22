# Numerical validation and reproduction

Run commands from the repository root after installing `requirements.txt`.

## Held-out predictions and candidate screening

```text
python scripts/verify_release.py
python scripts/figure5d/recalculate_figure5d.py
```

The release verifier checks 99 visual-quality predictions, 121 process-quality
predictions and the 243-setting candidate grid. The shortlist calculation uses
the grouped out-of-fold predictions supplied with the script. Its two summary
groups contain 132 and 42 states.

## Reduced-order processing constraints

Extract `Table_S4-1_full_131072_state_space.csv` from
`Supplementary_Data_3.zip` into `external_data/`. Then run:

```text
python scripts/simulation/reduced_order/recalculate_sequential_gates_131072.py
```

The first-failure counts are written under
`working/fig4d_recalculation_131072/source_data/`. The reference tables are in
`validation/figure4/`.

## External food-matrix comparisons

```text
python scripts/figure6/reproduce_figure6.py --prints data/figure6/Figure6_12_print_inputs.csv --scorer models/Figure6_hierarchical_phenotype_OQI_scorer.joblib --recommender models/Figure5_RECOMMENDER_rheology_to_process_topk.joblib --photo data/figure6/photo_plate.png --projection data/figure6/Figure6b_external_PCA.csv --state-map data/state_space --fig5-oqi-scorer models/Figure5_development_phenotype_to_OQI.joblib --out working/figure6
```

This command uses the recorded paired-print inputs, fitted scoring models and
photo plate in the repository. It writes the paired-comparison and same-setting
tables to `working/figure6/`. The same-setting comparison gives MAE 4.91 and
RMSE 5.73 points. The paired comparison uses its own scoring model and gives an
average gain of 34.47 points. The data notes in `data/figure6/` explain the
input records and comparisons.
