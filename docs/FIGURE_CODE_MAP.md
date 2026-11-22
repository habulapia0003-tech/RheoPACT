# Code and source data by figure

| Figure | Code or data entry point | Files needed |
| --- | --- | --- |
| 2 | scripts/pointcloud/pointcloud_pipeline.py | Specimen PLY and design STL; measurement data are in Source Data |
| 3 | scripts/verify_release.py and scripts/figure3/run_update_analysis.py | Included fixed test and phenotype inputs |
| 4 | scripts/simulation/reduced_order/ and scripts/simulation/comsol/ | Included reference library and projection; complete state table and COMSOL model files supplied separately |
| 5 | scripts/verify_release.py and scripts/figure5/predict_process_from_rheology.py | Included estimators and test data |
| 5d | scripts/figure5d/recalculate_figure5d.py | Included out-of-fold predictions; results correspond to Supplementary Table 14 |
| 6 | scripts/figure6/reproduce_figure6.py | Recorded print inputs, two scoring models and the photograph plate |

| Panel or table | Source file |
| --- | --- |
| Fig. 5d and Supplementary Table 14 | [Candidate retrieval summary](../validation/figure5/Figure5d_distinct_candidate_summary.csv) |
| Fig. 6b | [Rheological projection](../data/figure6/Figure6b_external_PCA.csv) |
| Fig. 6c | [Photograph index](../data/figure6/Figure6c_photo_columns.csv) and [photograph plate](../data/figure6/photo_plate.png) |
| Fig. 6d and Supplementary Fig. 16a | [Paired print outcomes](../data/figure6/Figure6_paired_results.csv) |
| Fig. 6e and Supplementary Table 16 | [Predicted and print-derived OQI](../data/figure6/Figure6e_same_setting_OQI.csv) |
| Supplementary Fig. 16b | [Changes in printing settings](../data/figure6/Supplementary_Figure29a_process_changes.csv) |
| Supplementary Fig. 16c | `Supplementary_Figure29_source_data.csv` in Source Data |
| Supplementary Table 15 | [Twelve print records](../data/figure6/Figure6_12_prints.csv) and [paired print outcomes](../data/figure6/Figure6_paired_results.csv) |

Some figure builders read the input tree under `external_data/source_tree`.
The [figure file index](../source_records/figure_source_manifest.csv) lists
those files. See [accompanying data and models](EXTERNAL_DATA.md) for access
and input paths.
