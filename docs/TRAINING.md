# Training and inference

## Figure 3: digital phenotype to sensory score

Input consists of D1-D5 point-cloud phenotype descriptors for 518 successful
extrusions. The target is the mean score from the revised sensory table
(`Sensory_mean_new`). The fixed assignment contains 419 development and 99 test
samples and is grouped by `Experiment_group`; the regression population is the
518 successful extrusions.

The comparison code includes linear regression, Ridge, ElasticNet, KNN, RBF
SVR, Random Forest, Extra Trees, Gradient Boosting, Histogram Gradient
Boosting, and AdaBoost. Median imputation is used; scaled estimators additionally
use standardization. Hyperparameter grids are preserved verbatim in
`scripts/figure3/run_update_analysis.py`: Ridge alpha 0.1-100; ElasticNet alpha
0.001-0.1 and l1 ratio 0.2/0.5/0.8; KNN neighbours 5/9/15/25; SVR C 1/10/100,
gamma scale/0.01/0.1 and epsilon 0.1/1/5; tree ensemble minimum leaf 1/2/4 and
max features sqrt/1.0; boosting grids are likewise explicit in the script.

Five-fold GroupKFold CV is scored by RMSE, R², and MAE, with RMSE used for
selection. The frozen estimator is a median-imputation pipeline followed by
`HistGradientBoostingRegressor(max_iter=300, max_leaf_nodes=7,
l2_regularization=1.0, random_state=20260902)`. It is fitted on the development
partition and evaluated once on the 99 fixed test samples. The packaged
validation reproduces R² 0.8174787, RMSE about 8.7744, and MAE about 5.0623.

PCA, Spearman association, partial Spearman association, standardized multiple
regression, and SHAP/permutation interpretation are in the Figure 3 scripts.

## OQI generation

For successful prints, OQI is a leakage-controlled D1-D5-to-sensory model
prediction. Development OQI is generated out of fold within the relevant
groups; candidate-test OQI is predicted by an evaluator fitted on the
development partition. Non-extrudable attempts retain protocol-defined zero
outcomes. This construction supplies all 590 attempts to Figure 5 while
preserving the successful-extrusion population of the Figure 3 perception
model.

## Figure 5: process quality and feasibility

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
evaluated separately. Candidate outer folds are
fixed by row-count and failure-rate balance before model evaluation.
Regression candidates reuse the Figure 3 model library; five-fold GroupKFold
RMSE is the selection criterion. The frozen quality pipeline is median
imputation plus `ExtraTreesRegressor(n_estimators=400,
random_state=20260902, n_jobs=1)`. It has grouped-CV R² about 0.8103 and fixed
candidate-test R² about 0.6971.

Feasibility candidates include logistic regression, RBF SVC, Random Forest,
Extra Trees, and Histogram Gradient Boosting, with the explicit grids in
`scripts/figure5/reanalyze_all590_cross_state.py`. Five-fold
StratifiedGroupKFold uses `Rheology_ID`; PR-AUC is the refit metric. The deployed
classifier is a 700-tree Random Forest with minimum leaf 2, `max_features=sqrt`,
and seed 20260902. Its training label is the observed extrusion outcome. The deployed feasibility estimator was refitted on all 590 attempts for candidate screening. The 121-attempt held-out metrics above refer to the quality regressor.

The packaged final estimator is the development-fitted frozen object evaluated
once on the 121 held-out records.

## Recommendation

The recommender enumerates the 243 observed discrete process combinations. It
predicts both feasibility and OQI, filters at `P(extrudable) >= 0.50`, and then
sorts feasible candidates by predicted OQI descending, using probability as a
tie-breaker.

## Commands

Frozen inference validation:

```bash
python scripts/verify_release.py
```

Recommendation:

```bash
python scripts/figure5/predict_process_from_rheology.py --help
```

The Figure 3 analysis and the Figure 5 candidate-group training script use the included inputs. The earlier complete-state workflow and some figure builders refer to the separate project source tree under external_data/source_tree. The portable release command validates the frozen
estimators, split contracts and fixed-test predictions.


## Score-source clarification

The study team identifies the on-site Excel workbook as the visual score collection record. Supplementary Data 1 includes an English-labelled copy with the numerical entries, row order and experiment-group numbers preserved. The processing relationship from that workbook to the different analytical score matrix remains unresolved. Numerical reproduction of the supplied analytical models does not establish this processing relationship. See Supplementary Table 6.
