# Data files and source labels

- frozen_tables/Food3D_590_merged_dataset.xlsx contains the 590 printing attempts, 540 measured rheology records, linkage notes and source manifest.
- frozen_tables/Food3D_590_visual_ratings_50_assessors.xlsx contains the 590-attempt analysis ratings. The view uses English column headings. Values and row order follow the supplied analysis workbook.
- frozen_tables/Food3D_D1D5_NewSensory_518_extrudable_training.csv contains the Figure 3 specimen-level analysis targets and phenotypes. The split file records the 419/99 partition.
- state_space/ contains measured rheology, projection settings, the 131,072-state projection and simulation inputs.
- figure6/ contains the twelve recorded print conditions and the paired and same-setting comparison tables.
- reference_geometry/ contains eleven STL reference designs. The English filenames identify the designs; the mesh files are preserved.

Materials are identified by their English names and specimen codes. The native starch codes are MS (cassava), WD (pea), YM (maize), MLS (potato) and XM (wheat). HS, SP, AS, ADSP and HDSP identify the modified starches. The identifiers in the rheology tables use English material names with the original condition suffixes. The rheology importer recognizes the original instrument and printing workbook headers. Supplementary Data 1 documents the lineage of the analysis ratings.
