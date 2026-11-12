# Supplementary table register (working revision, Stage 14)

Publication-facing Supplementary Tables are numbered **S1-S28**. The redundant former Tables S6-S8 have been removed from the revised Word SI: their numerical data and model selections are already reported in the comprehensive **Table S18**, and the four unique repeated-fold R² standard deviations from the former S6 were retained in the new S18 caption and the numerical export. The legacy computational `results/` folders, whose file names include historical S numbers and reviewer-comment IDs, are deliberately retained as provenance and are **not** publication table identifiers.

| Publication table | Verified path in this working branch | Scientific content |
| --- | --- | --- |
| S1 | [`rheology_table_S1.csv`](../data/processed/rheology_table_S1.csv) | Curated rheology data (48 rows) |
| S2 | [`printability_table_S2.csv`](../data/processed/printability_table_S2.csv) | Curated 2D printability data (43 rows) |
| S3 | [`angle_fidelity_table_S3.csv`](../data/processed/angle_fidelity_table_S3.csv) | Curated angular-fidelity data (96 rows) |
| S4 | [`022_tables_feature_dictionary.csv`](../supplementary/tables/022_tables_feature_dictionary.csv) | Engineered-feature dictionary; active code defines duplicate-reduced features |
| S5 | [`024_tables_feature_target_correlations.csv`](../supplementary/tables/024_tables_feature_target_correlations.csv) | Archived feature-target associations |
| S6 | [Legacy generated source outputs](../results/) | Archived SHAP summaries (see original analytical archive) |
| S7 | [Legacy generated source outputs](../results/) | Archived held-out permutation summaries (see original analytical archive) |
| S8 | [Legacy generated source outputs](../results/) | Separate XAI rank lists (see original analytical archive) |
| S9 | [Legacy generated source outputs](../results/) | Cross-method rank agreement (see original analytical archive) |
| S10 | [Legacy generated source outputs](../results/) | Table 3 XAI export (see original analytical archive) |
| S11 | [Legacy generated source outputs](../results/) | Feature-bin response summaries (see original analytical archive) |
| S12 | [Legacy generated source outputs](../results/) | Jointly observed condition eligibility audit |
| S13 | [Legacy generated source outputs](../results/) | Unordered 14-condition screen |
| S14 | [Legacy generated source outputs](../results/) | Processed model dataset package |
| S15 | [Legacy generated source outputs](../results/) | Reference-model full-fit diagnostic package |
| S16 | [`Table_S16_Ablation.xlsx`](../supplementary/final_revision/tables/Table_S16_Ablation.xlsx) | Core-versus-engineered representation sensitivity (former S19) |
| S17 | [`Table_S17_Baselines.xlsx`](../supplementary/final_revision/tables/Table_S17_Baselines.xlsx) | AF and Qm simple baselines (former S20) |
| S18 | [`Table_S18_Nested_CV_With_R2_SD.csv`](../supplementary/final_revision/tables/Table_S18_Nested_CV_With_R2_SD.csv) | Comprehensive nested CV/LOFO performance, modal selections, frequencies, and preserved outer-fold R² SDs (former S21 plus unique data of S6) |
| S19 | [`Table_S19_XAI.xlsx`](../supplementary/final_revision/tables/Table_S19_XAI.xlsx) | Held-out permutation and SHAP attribution (former S22) |
| S20 | [`Table_S20_LOFO_Diagnostics.xlsx`](../supplementary/final_revision/tables/Table_S20_LOFO_Diagnostics.xlsx) | Formulation-level LOFO diagnostics (former S23) |
| S21 | Embedded in the SI DOCX; [source audit script](../src/audit_comment_2_R2.py) | Embedded source-data provenance table; optional source PDF audit script retained (former S24) |
| S22 | [`Table_S22_Bootstrap_Ranking_Stability.csv`](../supplementary/final_revision/tables/Table_S22_Bootstrap_Ranking_Stability.csv) | Historical model-ranked bootstrap sensitivity (former S25) |
| S23 | [`Table_S23_R2_Aggregation.csv`](../supplementary/final_revision/tables/Table_S23_R2_Aggregation.csv) | Fold-level versus repeat-pooled R² aggregation (former S26) |
| S24 | [`Table_S24_Table2_vs_TableS16_Reconciliation.csv`](../supplementary/final_revision/tables/Table_S24_Table2_vs_TableS16_Reconciliation.csv) | Table 2 versus Table S16 consistency audit (former S27) |
| S25 | [`Table_S25_Dimensionless_Stress_Ratio_Sensitivity.csv`](../supplementary/final_revision/tables/Table_S25_Dimensionless_Stress_Ratio_Sensitivity.csv) | Dimensionless stress-scale sensitivity (former S28) |
| S26 | [`Table_S26_LOFO_Formulation_Block_Uncertainty.csv`](../supplementary/final_revision/tables/Table_S26_LOFO_Formulation_Block_Uncertainty.csv) | LOFO uncertainty and delete-one analysis (former S29) |
| S27 | [`Table_S27_Baseline_Comparison.csv`](../supplementary/final_revision/tables/Table_S27_Baseline_Comparison.csv) | Matched baseline-versus-nested comparisons (former S30) |
| S28 | [`Table_S28_Pr_Threshold_Sensitivity.csv`](../supplementary/final_revision/tables/Table_S28_Pr_Threshold_Sensitivity.csv) | Pr acceptance threshold sensitivity (former S31) |

## Exact publication mapping

Former S19-S31 become S16-S28, respectively. The separate former S6/S7/S8 tables are **not** renumbered to S6/S7/S8: they are removed because their information duplicates former S21. Former S9-S18 become S6-S15 (archive-only entries). The revised comprehensive S18 retains four otherwise unique R² SD values from the former S6: Pr 0.928, SR 0.702, Qm 0.238, AF 0.156.

`src/publication_si_table_renumber.py` is run **after** the scientific analyses, supplementary consolidation, existing XAI audit and numerical validation in `run_full_analysis.py`. It creates publication S18 from the archived selection-aware results and independently checks all four S6 R² SD values before converting active SI file names/sheet labels and removing superseded **consolidated publication** copies. It does not alter raw data or legacy analytical outputs. The new reconciliation **Table S24** changes only publication-facing column headers `TableS19_*` to `TableS16_*`; all eight scientific rows remain unchanged. The old source CSV files retain their previous names.

This branch includes the **new public aliases** S16-S28, verified against prior blobs where the content is unchanged. The old consolidated binary outputs and historical aliases are still in the branch until a successful end-to-end run regenerates the active package and its checksums. Do **not** treat the current working branch as a completed clean-run-certified release.

The embedded new Table S21 (formerly S24) is the source provenance summary. Its reported values and its source-PDF-based procedure remain unchanged. The external publisher-PDF audit has **not** been rerun as part of this editorial change.

The manuscript, SI Word document and cumulative reviewer response are supplied to the corresponding author as updated DOCX files; no DOCX bytes were added to the GitHub repository. The GitHub main branch, released tags, historical files and Zenodo records are untouched.
