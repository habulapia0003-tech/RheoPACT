# Data supplied separately

The prediction and Figure 6 commands in README.md use the files in the code archive.

| Archive | Contents | Use |
| --- | --- | --- |
| Source_Data.zip | Numerical tables for main and supplementary panels | Inspect and verify reported values |
| Supplementary_Data_1.zip | Printing, phenotype, visual-rating and split records | See Original_Records for the on-site rating workbook |
| Supplementary_Data_2.zip | Measured rheology, PCA and nozzle state definitions | Rheology reference library |
| Supplementary_Data_3.zip | Complete 131,072-state table and its projection | Extract Table_S4-1_full_131072_state_space.csv to external_data/ for the reduced-order scripts |
| Point_Clouds_and_CAD.zip | Representative PLY scans and CAD/STL models | Pass the scan and design paths to scripts/pointcloud/pointcloud_pipeline.py |
| COMSOL_Models.zip | Giesekus, generalized-Newtonian and nozzle simulation projects | Model names and variants are listed in COMSOL_MODEL_INDEX.csv. COMSOL 6.3 and a valid license are required |

The complete point-cloud batch and the project files used by older figure-assembly scripts are referenced in data_manifests/. They are separate from the compact inference archive. The archive names above identify files in the manuscript data package. Use the access details provided with the manuscript to obtain external archives.

The supplementary geometry builder expects phenotype scores in external_data/figure2/digital_phenotype_scores.csv and the printing analysis table in external_data/figure2/printing_analysis_590.csv. It also uses the geometry-analysis tree described in the script.

For the point-cloud workflow, use a separate Python 3.12 environment with the Windows wheels for Open3D 0.19 and install `requirements-pointcloud.txt`. Provide a PLY scan and its STL reference in the same spatial units.

~~~text
python scripts/pointcloud/pointcloud_pipeline.py --sample external_data/pointclouds/scan.ply --stl external_data/designs/design.stl --output working/pointcloud/example --case-id example
~~~
