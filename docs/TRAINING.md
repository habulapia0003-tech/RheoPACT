# Training and inference

## Visual-quality prediction from digital phenotypes

Input consists of D1-D5 point-cloud phenotype descriptors for 518 successful
extrusions. The target is the specimen mean from the visual-score analysis matrix
(`Sensory_mean_new`). The fixed assignment contains 419 development and 99 test
samples and is grouped by `Experiment_group`; the regression population is the
518 successfully extruded specimens. See the [data guide](../data/README.md#measurement-and-scoring-records) for the rating records.

The comparison code includes linear regression, Ridge, ElasticNet, KNN, RBF
SVR, Random Forest, Extra Trees, Gradient Boosting, Histogram Gradient
Boosting, and AdaBoost. Median imputation is used; scaled estimators additionally
use standardization. Hyperparameter grids are defined in
`scripts/figure3/run_update_analysis.py`: Ridge alpha 0.1-100; ElasticNet alpha
0.001-0.1 and l1 ratio 0.2/0.5/0.8; KNN neighbours 5/9/15/25; SVR C 1/10/100,
gamma scale/0.01/0.1 and epsilon 0.1/1/5; tree ensemble minimum leaf 1/2/4 and
max features sqrt/1.0; boosting grids are likewise explicit in the script.

Five-fold GroupKFold CV is scored by RMSE, R², and MAE, with RMSE used for
selection. The selected pipeline uses median imputation followed by
`HistGradientBoostingRegressor(max_iter=300, max_leaf_nodes=7,
l2_regularization=1.0, random_state=20260902)`. It is fitted on the development
partition and evaluated once on the 99 fixed test samples. The packaged
validation reproduces R² 0.8174787, RMSE about 8.7744, and MAE about 5.0623.

PCA, Spearman association, partial Spearman association, standardized multiple
regression, and SHAP/permutation interpretation are in the Figure 3 scripts.

## OQI generation

For successful extrusions, OQI is the visual-quality estimate derived from
D1–D5. Development scores are generated out of fold with group-separated
folds. Candidate-test scores are generated with a model fitted on the
development partition. The integrated outcome is zero for non-extrudable
attempts. These scores define the quality-regression target for all 590
attempts.

## Integrated print outcome and extrusion feasibility

Inputs are five process variables (print temperature, model height, print
speed, nozzle diameter, nozzle height) and seven rheological descriptors (G′,
G″, viscosity at 10 and 100 s⁻¹, yield stress, consistency coefficient, and
flow index). The first six rheology quantities are transformed as
`log10(value + 1e-8)` after clipping negative values to zero; `Selected_n` is
untransformed. Feature order is stored inside each joblib bundle. Missing
values are median-imputed.

The outer partition keeps `Candidate_group_ID` groups intact and yields
469 development and 121 candidate-test records. A candidate group combines
`Rheology_ID` with all five process settings. Complete-state holdout was
evaluated separately. Candidate outer folds were assigned by balancing the
number of records and the failure rate before model evaluation.
Regression candidates reuse the Figure 3 model library; five-fold GroupKFold
RMSE is the selection criterion. The selected quality pipeline uses median
imputation plus `ExtraTreesRegressor(n_estimators=400,
random_state=20260902, n_jobs=1)`. It has grouped-CV R² about 0.8103 and fixed
candidate-test R² about 0.6971.

Feasibility candidates include logistic regression, RBF SVC, Random Forest,
Extra Trees, and Histogram Gradient Boosting, with the explicit grids in
`scripts/figure5/reanalyze_all590_cross_state.py`. Five-fold
StratifiedGroupKFold uses `Rheology_ID` with PR-AUC as the refit metric. The deployed
classifier is a 700-tree Random Forest with minimum leaf 2, `max_features=sqrt`,
and seed 20260902. Its training label is the observed extrusion outcome. The deployed feasibility estimator was refitted on all 590 attempts for candidate screening. The 121-attempt held-out metrics above refer to the quality regressor.

## Recommendation

The recommender enumerates 243 combinations of the observed process levels. It
predicts both feasibility and OQI, filters at `P(extrudable) >= 0.50`, and then
sorts feasible candidates by predicted OQI descending, using probability as a
tie-breaker.

See the [reproduction guide](VALIDATION.md) for evaluation commands and
[accompanying data](EXTERNAL_DATA.md) for the additional inputs used by the
complete-state analysis and some figure builders.
