# Figure 6 source data

Each material contributed one reference print and one adjusted print. An
Extra Trees scorer converted measured D1–D5 to OQI for the 12 prints in
the paired comparison (Fig. 6d and Supplementary Fig. 16a). The mean OQI
gain was 34.47 points.

Figure 6e compares two scores at each of the six adjusted settings. The
Fig. 5 Extra Trees regressor predicts OQI from the process and rheology
inputs. The Fig. 5 HistGradientBoosting evaluator converts measured D1–D5
to OQI on the same scale. Signed error is predicted OQI minus the
print-derived score. MAE was 4.91 and RMSE was 5.73 points. The listed
candidate ranks are retrospective.

Figure 6b and the extrusion screen use the initial rheology records.
Figure 6e uses values reported as a later measurement for pumpkin (M04).
Their link to the instrument export and the adjusted-print batch remains
unverified. At the precision supplied, the later values match a set
previously evaluated in a sensitivity analysis. Using the initial M04
values gives MAE 9.40 and RMSE 14.97 points. Both comparisons are recorded
in `Figure6_summary.json`. Specimen-level linkage of the scans, D1–D5
scores and photographs remains unverified.

| Panel or table | File in Source Data |
| --- | --- |
| Fig. 6b | `Figure6b_external_PCA.csv` |
| Fig. 6c | `Figure6c_photo_columns.csv` and the Figure 6 photo plate |
| Fig. 6d and Supplementary Fig. 16a | `Figure6_paired_results.csv` |
| Fig. 6e and Supplementary Table 16 | `Figure6e_same_setting_OQI.csv` |
| Supplementary Fig. 16b | `Supplementary_Figure29a_process_changes.csv` |
| Supplementary Fig. 16c | `Supplementary_Figure29_source_data.csv` |
| Supplementary Table 15 | `Figure6_12_prints.csv` and `Figure6_paired_results.csv` |
