# Code and source data by figure

| Figure | Code or data entry point | Files needed |
| --- | --- | --- |
| 2 | scripts/pointcloud/pointcloud_pipeline.py | Specimen PLY and design STL; measurement data are in Source Data |
| 3 | scripts/verify_release.py and scripts/figure3/run_update_analysis.py | Included fixed test and phenotype inputs |
| 4 | scripts/simulation/reduced_order/ and scripts/simulation/comsol/ | Included reference library and projection; complete state table and COMSOL model files supplied separately |
| 5 | scripts/verify_release.py and scripts/figure5/predict_process_from_rheology.py | Included estimators and test data |
| 5d | scripts/figure5d/recalculate_figure5d.py | Included out-of-fold predictions; results correspond to Supplementary Table 14 |
| 6 | scripts/figure6/reproduce_figure6.py | Recorded inputs, two scoring models and the photo plate are included. Figure 6e corresponds to Supplementary Table 16 |

Some earlier figure-assembly scripts refer to a project tree under external_data/source_tree. The figure_source_manifest.csv records the source filenames of those assembled plates. The final submission plates are supplied separately. See EXTERNAL_DATA.md for the scientific data and solver inputs.
