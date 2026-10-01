# Explainable AI-Guided Bioink Design

Data, code, notebooks, exact supplementary tables, and figure-generation materials accompanying the manuscript:

**Explainable AI-Guided Bioink Design: Model-Derived Polymer-Pressure Relationships in Alginate-Hyaluronic Acid Bioinks**

## Scope

This repository supports a secondary, target-specific machine-learning and explainable-AI reanalysis of the alginate-hyaluronic acid bioink dataset reported by Perin et al. No new experimental measurements were generated in the present study.

The analysis models four outputs independently:

- printability ratio (`Pr`)
- spreading ratio (`SR`)
- mass flow rate (`Qm`)
- angular fidelity (`AF`)

The repository includes curated source tables, processed modeling datasets, duplicate-reduced feature engineering, reviewer-requested simple baselines, a dimensionless pressure/yield-stress sensitivity analysis, nested repeated and nested leave-one-formulation-out validation, held-out permutation importance, separate SHAP attribution, unordered measured-domain candidate screening, core-versus-engineered representation sensitivity, explicit Table 2/Table S19 reconciliation, and source-to-repository provenance audit materials.

## Repository structure

```text
notebooks/                 Canonical master notebook and archived legacy notebooks
src/                       Canonical preparation, validation, XAI, screening, audit, and figure-generation scripts
data/processed/            Three curated source tables at release start; derived modeling datasets are regenerated
data/source/               Source-data citation and provenance notice
results/tables/             Main generated result tables
results/descriptive_figures/ Regenerated SI S1-S2 and manuscript Figure 3 plus exact prediction tables
supplementary/tables/       Archived numbered CSV/XLSX files supporting Tables S1-S18
supplementary/final_revision/ Consolidated final revision Tables S6-S8 and S19-S31 plus replacement figures
supplementary/figures/      Archived supplementary figure files
supplementary/figures_pdf/  Vector/PDF supplementary figure materials
docs/                       Cell index, file register, and reproducibility documentation
```

## Quick start

### Conda

```bash
conda env create -f environment.yml
conda activate explainable-ai-bioink-design
jupyter lab
```

For the canonical end-to-end analysis, run from the repository root:

```bash
python run_full_analysis.py
```

For the readable notebook path, open `notebooks/00_master_analysis.ipynb` and use **Run All** from a fresh kernel. The notebook follows the same canonical dependency order and does not rely on hidden state.

### Google Colab

Clone or extract the repository to any local directory. The canonical command-line entry point is `python run_full_analysis.py`. For notebook execution, start Jupyter from the repository root, open `notebooks/00_master_analysis.ipynb`, restart the kernel, and Run All. Paths are repository-relative, the workflow begins from the included curated tables, and publisher PDFs are not required for the core analysis.

## Reproducibility notes

The reportable multivariable performance uses nested repeated cross-validation and nested leave-one-formulation-out validation so model/feature-set selection is separated from outer performance estimation. The main revised candidate analysis reports an unordered measured-domain set based on jointly observed source-study Pr and AF values; no statistically unique model-derived optimum is claimed. A separate reviewer-requested formulation-cluster bootstrap tests the stability of the originally submitted candidate ordering and retains 1ALG8HA / 65 kPa / 120 degrees as the modal, but not statistically separable, leading hypothesis.

The archived notebook under `notebooks/archive/` preserves the original Colab workflow, including optional PDF-extraction and packaging cells. The canonical notebook is `notebooks/00_master_analysis.ipynb`. Retired pre-integration code/notebooks are stored only under `archive/`.


## Final descriptive-figure generation

`src/generate_descriptive_figures.py` regenerates Supplementary Figures S1-S2 and the manuscript Figure 3 prediction heatmaps from canonical data and current fixed duplicate-reduced reference models. The Figure 3 display grid is created from formulation-specific source pressure ranges at 5-kPa increments; it is descriptive and is not used for candidate eligibility or ranking. No archived prediction table is read by this generator.

## Data provenance

The source experimental data were reported by:

Perin, F.; Spessot, E.; Famà, A.; Bucciarelli, A.; Callone, E.; Mota, C.; Motta, A.; Maniglio, D. *Modeling a Dynamic Printability Window on Polysaccharide Blend Inks for Extrusion Bioprinting.* **ACS Biomaterials Science & Engineering** 2023, 9, 1320-1331. DOI: `10.1021/acsbiomaterials.2c01143`.

The source publication is Open Access under CC BY 4.0. The original data-extraction route and the Reviewer 2 Comment 2 source-table verification audit are documented in `data/source/SOURCE_DATA.md` and `COMMENTS_2-R2_README.md`.

Publisher PDFs are not redistributed in this repository. See `data/source/SOURCE_DATA.md` and `DATA_USE_NOTICE.md`.

## Citation

Citation metadata are provided in `CITATION.cff`. The manuscript cites the GitHub repository and Zenodo version v1.0.1 (DOI 10.5281/zenodo.22835304); confirm the public release metadata before resubmission.

## License and reuse

Code is released under the MIT License. Curated and processed data are provided for reproducibility subject to the source-data attribution and reuse notice in `DATA_USE_NOTICE.md`.


## Reviewer revision analysis: Comment 3-R2

The revision package includes a reviewer-requested simple-baseline analysis for AF and Qm. Run `python src/revision_comment_3_R2_baselines.py` after installing the repository environment. The script evaluates angle-only AF baselines and P/TP-only Qm baselines using the same repeated five-fold CV and LOFO structure used in the manuscript. Outputs are written to `results/revision_comment_3_R2/`.

## Reviewer revision analysis: Comment 10-R1

The revision tests a dimensionless applied-pressure/yield-stress descriptor before concluding that P/TP should be interpreted only empirically. Run `python src/revision_comment_10_R1_dimensionless_stress_ratio.py`. The script compares P/TP with `P_over_Yield_dimless = Pressure_kPa*1000/Yield_stress_mean_Pa` as fixed single-descriptor linear/power-law baselines for Qm and SR and performs a matched multivariable substitution sensitivity. The dimensionless descriptor remains non-predictive as a stand-alone rule and does not outperform P/TP in the matched engineered representation. Outputs are under `results/revision_comment_10_R1_dimensionless_stress_ratio/` and the compact summary is Supplementary Table S28.

## Reviewer revision analysis: Comments 4-R1 and 16-R2

The revision package includes nested model/feature-set selection to prevent the same resampling from being used both to choose a candidate and to report its performance. Run:

`python src/revision_comments_4_R1_16_R2_nested_cv.py`

The repeated analysis uses 20 repetitions of outer five-fold cross-validation with inner five-fold candidate selection. The formulation-level analysis uses outer leave-one-formulation-out validation with inner LOFO candidate selection. Complete outputs are stored in `results/revision_comments_4_R1_16_R2/`. Because R2 is non-additive and the outer test folds are small, `src/revision_comments_4_R1_16_R2_r2_aggregation.py` reports two complementary summaries of the same held-out predictions: the arithmetic mean of the 100 fold-level R2 values and the mean R2 obtained after pooling all out-of-fold predictions within each of the 20 repetitions. The latter is a reporting/aggregation sensitivity analysis and does not alter model selection or refit the models.

## Reviewer revision analysis: Comments 6-R1 and 8-R2

Table 2 and Table S19 answer different questions. Table 2 reports selection-aware nested outer-test performance. Table S19 preserves the submitted core-versus-engineered model identities after duplicate-feature cleanup. Six of eight S19 target-validation comparisons use the same algorithm and can be interpreted as feature-representation sensitivity conditional on that algorithm; Pr LOFO and SR repeated K-fold use different algorithms and therefore reflect combined model-plus-representation changes. Run `python src/revision_comment_6_R1_reconciliation.py` after the S19 script to generate the row-by-row reconciliation used in Supplementary Table S27. Then run `python src/revision_comment_8_R2_reconciliation_audit.py` to verify all eight S27 rows against the archived nested and S19 outputs; this audit performs no model fitting.

## Reviewer revision analysis: Comments 12-R1 and 9-R2
The revision package includes formulation-level uncertainty reporting for the nested LOFO analysis. Run:

`python src/revision_comments_12_R1_9_R2_lofo_uncertainty.py`

The script does not refit models. It summarizes the already-computed nested LOFO outer-fold results from Comments 4-R1/16-R2, exports the full per-formulation diagnostics, and performs a 10,000-replicate formulation-block bootstrap of the fixed outer predictions. The bootstrap resamples the eight held-out formulation blocks with replacement and preserves all observations within a sampled block. It therefore quantifies evaluation-sample uncertainty conditional on the nested-LOFO predictions rather than full training-procedure uncertainty. Supplementary Table S29 reports all-target bootstrap intervals and delete-one-formulation pooled-R2 sensitivity; SR shows pooled R2 = 0.393, bootstrap 95% interval 0.179-0.537, and delete-one R2 range 0.315-0.445, alongside heterogeneous formulation-specific R2 values (-2.218 to 0.485).


### Reviewer candidate-set revision (Comments 2-R1 and 6-7-R2)
The reviewer-responsive candidate analysis is implemented in `src/revision_comments_2_R1_6_7_R2_candidate_set.py`. It uses measured Pr and measured AF at jointly observed formulation-pressure conditions and writes the audit workbook, candidate table, and replacement candidate-set map to `results/revision_comments_2_R1_6_7_R2/`.


## Final reviewer revision

See `FINAL_REVISION_README.md` and `RUN_ORDER.md` for the authoritative reviewer-responsive analysis sequence and final revision artifacts.


## Reviewer-requested model-complexity controls

Simple AF/Qm baselines are generated by `src/revision_comment_3_R2_baselines.py`. A matched incremental-value audit in `src/revision_comment_3_R2_baseline_contrast.py` compares those held-out predictions with the nested multivariable outer predictions using the same repeat-pooled aggregation. The resulting Table S30 distinguishes geometry-dominated AF from the clear within-domain multivariable gain observed for Qm.


### Comment 17-R2 XAI audit
The revised XAI workflow reports held-out permutation importance and SHAP separately, excludes Pr from cross-method rank-agreement claims, and uses no composite consensus score. `src/revision_comment_17_R2_xai_audit.py` checks these conditions and verifies the repeated XAI permutation metadata (5 folds x 5 repeats; 25 held-out folds). Superseded pre-revision XAI outputs are quarantined under `archive/legacy_pre_revision_xai/`. Superseded model-ranked candidate/design-atlas artifacts are quarantined under `archive/legacy_pre_revision_candidate_ranking/`, and one-time/intermediate revision scripts are under `archive/intermediate_revision_scripts/`. Only scripts remaining in `src/` belong to the canonical final reviewer-revision workflow.

### Comment 18-R2 held-out permutation audit
`src/revision_comment_18_R2_oof_permutation_audit.py` verifies that permutation importance is computed only on held-out test data, uses delta RMSE with 10 permutations per held-out fold, and reproduces the 25-fold repeated and 8-fold LOFO outputs. The final XAI script also fixes the descriptive SHAP permutation-explainer seed at 42; this does not change held-out permutation results or leading features, but it makes SHAP summaries and rank-agreement decimals reproducible. See `COMMENT_18_R2_OOF_PERMUTATION_README.md`.


## Stage 12 clean-room certification status

**FULL CLEAN-ROOM PASS.** The final Master Runner completed from a fresh extraction with 50/50 mandatory numerical checks passing. The Master Notebook independently completed 19/19 code cells from a separate fresh extraction with no errors. Runner/notebook scientific outputs reconciled with zero discrepancies. See `CLEAN_ROOM_REPRODUCIBILITY_REPORT.md`.
