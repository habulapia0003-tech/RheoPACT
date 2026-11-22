# Reproducing the analyses

Run commands from the repository root after installing `requirements.txt`.

## Model predictions and candidate ranking

```text
python scripts/verify_release.py
python scripts/figure5d/recalculate_figure5d.py
```

The verifier checks 99 visual-quality predictions, 121 process-quality
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

## Printing comparisons at the tested settings

```text
python scripts/figure6/reproduce_figure6.py --prints data/figure6/Figure6_12_print_inputs.csv --scorer models/Figure6_hierarchical_phenotype_OQI_scorer.joblib --recommender models/Figure5_RECOMMENDER_rheology_to_process_topk.joblib --photo data/figure6/photo_plate.png --projection data/figure6/Figure6b_external_PCA.csv --state-map data/state_space --fig5-oqi-scorer models/Figure5_development_phenotype_to_OQI.joblib --out working/figure6
```

The command writes the paired-comparison and same-setting tables to
`working/figure6/`. See the [study data guide](../data/README.md#measurement-and-scoring-records)
for the measurement and scoring records.

## Reference results

| Analysis | Reference result |
| --- | --- |
| Visual-quality prediction | 99 test specimens, MAE 5.06 points |
| Integrated print-outcome prediction | 121 test attempts, MAE 19.60 for process only and 8.83 for rheology plus process |
| Candidate retrieval | The groups with at least two and at least three distinct candidates contain 132 and 42 rheological states, respectively |
| Paired printing comparison | 12 prints across six food matrices, mean paired OQI gain of 34.47 points |
| Same-setting prediction | Six tested adjusted settings, MAE 4.91 and RMSE 5.73 OQI points |
