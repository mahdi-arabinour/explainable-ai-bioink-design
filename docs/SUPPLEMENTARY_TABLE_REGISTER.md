# Supplementary table file register

The publication-facing table names and the actual files in the corresponding manuscript's computational archive are mapped here. The original `main` branch and previous releases remain unchanged. This map was verified against the tree of `release/acs-omega-r1-audit-20261112`.

| SI table | Current location in the working release branch | Scientific scope |
| --- | --- | --- |
| S1 | [`rheology_table_S1.csv`](../data/processed/rheology_table_S1.csv) | 48 source rheology measurements |
| S2 | [`printability_table_S2.csv`](../data/processed/printability_table_S2.csv) | 43 source 2D printability conditions |
| S3 | [`angle_fidelity_table_S3.csv`](../data/processed/angle_fidelity_table_S3.csv) | 96 source angular fidelity measurements |
| S4 | [`022_tables_feature_dictionary.csv`](../supplementary/tables/022_tables_feature_dictionary.csv) | Archived descriptor dictionary; duplicate-reduced features defined in active code |
| S5 | [`024_tables_feature_target_correlations.csv`](../supplementary/tables/024_tables_feature_target_correlations.csv) | Feature-target association archive; active Figure S3 is separate |
| S6-S8 | [`Table_S6-S8_Nested_Performance.xlsx`](../supplementary/final_revision/tables/Table_S6-S8_Nested_Performance.xlsx) | Repeated, LOFO, and selection-frequency sheets |
| S9-S18 | [``](../supplementary/tables/) | Archived numbered outputs; see the embedded SI register for scientific scope |
| S19 | [`Table_S19_Ablation.xlsx`](../supplementary/final_revision/tables/Table_S19_Ablation.xlsx) | Core-versus-engineered comparison |
| S20 | [`Table_S20_Baselines.xlsx`](../supplementary/final_revision/tables/Table_S20_Baselines.xlsx) | AF and Qm simple baselines |
| S21 | [`Table_S21_Nested_CV.xlsx`](../supplementary/final_revision/tables/Table_S21_Nested_CV.xlsx) | Nested CV and LOFO evaluation |
| S22 | [`Table_S22_XAI.xlsx`](../supplementary/final_revision/tables/Table_S22_XAI.xlsx) | Held-out permutation and SHAP |
| S23 | [`Table_S23_LOFO_Diagnostics.xlsx`](../supplementary/final_revision/tables/Table_S23_LOFO_Diagnostics.xlsx) | Formulation-specific LOFO diagnostics |
| S24 | Embedded Table S24 in the Word Supplementary Information; [audit script](../src/audit_comment_2_R2.py); [method description](../COMMENTS_2-R2_README.md) | Embedded SI Table S24; PDF-based source audit script and README retained, no standalone audit CSV committed |
| S25 | [`Table_S25_Bootstrap_Ranking_Stability.csv`](../supplementary/final_revision/tables/Table_S25_Bootstrap_Ranking_Stability.csv) | Historical 17-condition ranking sensitivity |
| S26 | [`Table_S26_R2_Aggregation.csv`](../supplementary/final_revision/tables/Table_S26_R2_Aggregation.csv) | Publication-facing byte-identical content alias of the archived R2 aggregation CSV |
| S27 | [`Table_S27_Table2_vs_TableS19_Reconciliation.csv`](../supplementary/final_revision/tables/Table_S27_Table2_vs_TableS19_Reconciliation.csv) | Nested versus representation analysis reconciliation |
| S28 | [`Table_S28_Dimensionless_Stress_Ratio_Sensitivity.csv`](../supplementary/final_revision/tables/Table_S28_Dimensionless_Stress_Ratio_Sensitivity.csv) | Applied-pressure/yield-stress sensitivity |
| S29 | [`Table_S29_LOFO_Formulation_Block_Uncertainty.csv`](../supplementary/final_revision/tables/Table_S29_LOFO_Formulation_Block_Uncertainty.csv) | LOFO evaluation-stage uncertainty |
| S30 | [`Table_S30_Baseline_Comparison.csv`](../supplementary/final_revision/tables/Table_S30_Baseline_Comparison.csv) | Publication-facing exact-copy matched baseline comparison |
| S31 | [`Table_S31_Pr_Threshold_Sensitivity.csv`](../supplementary/final_revision/tables/Table_S31_Pr_Threshold_Sensitivity.csv) | Publication-facing exact-copy Pr cutoff sensitivity |

## Preservation and provenance

- The short filenames for **S26, S30, and S31** were added only on this working branch. Their complete UTF-8 contents were checked against their original archived files, and found identical. No numerical data or analyses were changed.
- The historical files remain accessible at their former exact paths:
  - `supplementary/final_revision/tables/Table_S26_R2_Aggregation_Summary_Comments_4-R1_16-R2.csv`
  - `supplementary/final_revision/tables/Table_S30_Baseline_vs_Nested_Apples_to_Apples_Comment_3-R2.csv`
  - `supplementary/final_revision/tables/Table_S31_Relaxed_Pr_Only_Conditions_Comment_7_R2.csv`
- The other public-facing files linked above are existing files whose names do not refer to reviewer comments; they were not copied or renamed.
- **Table S24 is preserved in the Supplementary Information DOCX**, reporting the source-data verification summary for three datasets. The repository includes the [source-PDF audit script](../src/audit_comment_2_R2.py) and [methodological README](../COMMENTS_2-R2_README.md). A separately committed `Table_S24_Provenance.csv` or `Source_vs_Repository_Audit_Summary_Comment_2-R2.csv` was not found in the inspected GitHub branch tree. The master runner only produces the separate S24 CSV if the optional source-PDF audit is run in that execution. The missing *standalone copy* should not be conflated with the embedded table; no new audit was invented or performed in this naming-only pass.
- No source measurement, numerical SI table, model code, prior GitHub tag, or published Zenodo archive was edited during this step.
