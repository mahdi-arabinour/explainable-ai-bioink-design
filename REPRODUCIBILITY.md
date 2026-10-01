# Reproducibility and claim boundaries

This repository supports internal validation and model interpretation for a sparse, published alginate-hyaluronic acid bioink dataset.

## Interpretation boundaries

- Repeated K-fold validation can place related formulation observations in both training and validation folds; it estimates within-domain interpolation rather than independent formulation transfer.
- Leave-one-formulation-out validation is the stricter formulation-level test.
- SHAP and permutation importance quantify model reliance, not causal mechanisms.
- Engineered ratios such as `P_over_TP` are empirical descriptors and are not dimensionless physical groups. Reviewer 1 Comment 10 is addressed with a separate dimensionless applied-pressure/yield-stress sensitivity analysis; the new descriptor does not establish a transferable physical rule.
- Candidate screening is restricted to combinations represented in the source data.
- Candidate rankings require prospective experimental validation.

## Expected dataset sizes

- Pr: 40 observations
- SR: 43 observations
- Qm: 40 observations
- AF: 96 observations

## Nested validation added for Comments 4-R1 and 16-R2

Reportable multivariable performance in the revision is based on nested validation. For repeated validation, the outer five-fold split is repeated 20 times. Within each outer-training partition, all eligible algorithm/feature-set candidates are compared by inner five-fold cross-validation, using pooled RMSE as the primary selection criterion and MAE as the tie-breaker. The selected candidate is refit on the full outer-training partition and evaluated only on the untouched outer-test fold.

For formulation-level transfer, one formulation is held out in the outer loop. Inner leave-one-formulation-out validation across the remaining formulations selects the algorithm and feature set before refitting and prediction on the outer held-out formulation. Hyperparameters remain fixed at the values defined in the analysis pipeline. Random seed 42 is used where applicable.

## Dimensionless stress-scale sensitivity for Comment 10-R1

`src/revision_comment_10_R1_dimensionless_stress_ratio.py` defines `P_over_Yield_dimless = Pressure_kPa*1000/Yield_stress_mean_Pa` and compares it with `P_over_TP` using prespecified linear and power-law single-descriptor models under repeated five-fold cross-validation (20 repeats) and pooled LOFO. It also replaces P/TP one-for-one with the dimensionless descriptor in the same duplicate-reduced engineered SVR representation used for the Qm and SR S19 comparisons. The analysis uses random seed 42. The dimensionless descriptor is treated as an applied-pressure/yield-stress stress-scale sensitivity, not an exact nozzle wall-shear calculation, because the curated source tables do not contain the local pressure drop, internal conical profile, or volumetric flow/shear rate needed for a constitutive nozzle-flow estimate without extra assumptions. The complete outputs are in `results/revision_comment_10_R1_dimensionless_stress_ratio/`, and the compact one-descriptor summary is Supplementary Table S28.

## XAI revision for Comments 8-R1 and 17-18-R2
The revised XAI analysis uses random seed 42. Out-of-fold permutation importance is computed on held-out test folds using five repetitions of five-fold cross-validation (25 held-out folds), with 10 feature permutations per fold; a LOFO sensitivity analysis is also generated. Delta RMSE is defined as permuted-test RMSE minus unpermuted-test RMSE. The descriptive SHAP permutation explainer also uses an explicit seed of 42. SHAP attribution is reported separately from permutation importance and no composite consensus score is calculated. Pr is excluded from the cross-method rank-agreement summary because its three-predictor reference model has unstable repeated-CV performance across small outer folds and negative nested LOFO R2; the complementary repeat-pooled OOF R2 is positive but does not establish a stable predictive design relationship. The canonical reviewer-specific outputs are under `results/revision_comments_8_R1_17_18_R2/`.

## Table 2 / Table S19 reconciliation for Comment 6-R1

Table 2 reports selection-aware nested outer-test performance and is the reportable predictive-performance analysis. Table S19 is a continuity/sensitivity analysis that preserves the submitted core and engineered model identities after duplicate-feature cleanup. In six of eight target-validation comparisons the same algorithm is used on both sides, allowing a feature-representation sensitivity interpretation conditional on that algorithm. In Pr LOFO and SR repeated K-fold, the submitted core and engineered algorithms differ, so those delta metrics reflect combined model-identity and representation changes and are not interpreted as pure feature-ablation effects. `src/revision_comment_6_R1_reconciliation.py` generates Supplementary Table S27 directly from the same-run canonical nested-validation and S19 outputs; it performs no new model fitting. The reconciliation explicitly uses the primary fold-wise Table 2 repeated R2 convention rather than the complementary repeat-pooled Table S26 convention.

## LOFO uncertainty reporting for Comments 12-R1 and 9-R2
The per-formulation uncertainty analysis uses the stored outer-fold metrics and outer predictions from the nested LOFO procedure generated for Comments 4-R1 and 16-R2. No new model fitting is required. For SR, eight held-out formulations are reported separately with N, selected model/feature set, R2, MAE, RMSE, and Spearman correlation. The pooled LOFO R2 remains the primary overall metric; fold-level R2 values are diagnostics because R2 is non-additive across small held-out groups. An evaluation-stage formulation-block bootstrap resamples the eight held-out formulation blocks with replacement for 10,000 replicates while preserving within-block observations. This bootstrap uses the fixed outer predictions and therefore does not include model-refit or model-selection uncertainty. Supplementary Table S29 reports the all-target percentile intervals and delete-one-formulation sensitivity. For SR, pooled R2 = 0.393, bootstrap 95% interval = 0.179-0.537, and delete-one pooled R2 = 0.315-0.445.

## Fixed model specifications and search space for Comments 13-R1 and 1-R2

No hyperparameter tuning or hyperparameter grid search was performed for the reportable nested-validation analysis. The nested procedure selected only the algorithm and eligible target-specific feature set within each inner training split.

Fixed model specifications were:

- Linear regression: ordinary least-squares linear regression after fold-specific standardization.
- RBF-SVR: `C=10.0`, `epsilon=0.05` after fold-specific standardization.
- Random forest: `n_estimators=300`, `max_depth=None`, `min_samples_leaf=2`, `random_state=42`.
- Gradient boosting: `n_estimators=200`, `learning_rate=0.05`, `max_depth=2`, `random_state=42`.
- XGBoost: `n_estimators=200`, `learning_rate=0.05`, `max_depth=2`, `subsample=0.9`, `colsample_bytree=0.9`, `objective="reg:squarederror"`, `random_state=42`, `n_jobs=1`.

The model-selection search space was the Cartesian set of these five prespecified algorithms and the eligible target-specific feature sets defined in `src/revision_comments_4_R1_16_R2_nested_cv.py`. Random seed 42 was used for stochastic procedures. Exact dependency versions for the certified clean-room execution are pinned in `environment.yml` and `requirements.txt`. These pins describe the environment actually used for the final clean-room run; they should be updated only if the full clean-room workflow is rerun and revalidated under a different environment.

This update is documentation-only. No model fitting, validation, XAI computation, or candidate screening was rerun for Comments 13-R1 and 1-R2 because these comments concern reproducibility wording and removal of prescriptive/self-review language rather than a change in the analytical procedure.


For Reviewer 2 Comment 3, run `src/revision_comment_3_R2_baselines.py`, then the nested-validation script, then `src/revision_comment_3_R2_baseline_contrast.py`. The contrast script regenerates Table S30 and the repeat-level matched comparison outputs.


### XAI figure reproducibility note
Figure S7 beeswarm generation also uses the fixed SHAP permutation-explainer seed of 42, matching the authoritative XAI analysis.



## Descriptive figure reproducibility

`src/generate_descriptive_figures.py` is an active canonical step immediately after data preparation. It regenerates Supplementary Figure S1 from the four target-specific modeling datasets and Supplementary Figure S2 from the canonical target-specific source observations. It also regenerates manuscript Figure 3 using the fixed reference models used by the downstream interpretation workflow: core linear regression for Pr, rheology-enhanced RBF-SVR for SR, engineered RBF-SVR for Qm, and angle-engineered XGBoost for AF. The display grid is generated programmatically from each formulation's source pressure range at 5-kPa increments; AF is predicted at the sampled angle levels and averaged by formulation-pressure condition. These heatmaps are descriptive full-data model outputs, not validation estimates and not candidate-screening inputs. No archived prediction artifact is read.

## Clean-room dependency rules

The canonical workflow starts from the curated source-derived CSV files in `data/processed/` and must not require pre-existing files under `results/` or `supplementary/` as analysis inputs. Reviewer-specific scripts may consume outputs only from earlier canonical steps in `RUN_ORDER.md`. Archived development scripts are never executed by the core workflow. The sole documented archive input is the historical strict-candidate artifact used by the retrospective candidate-ranking bootstrap, because that sensitivity analysis intentionally reconstructs the originally submitted ordering.

Core stochastic analyses use seed 42 where specified. The retrospective candidate-ranking bootstrap preserves its explicit fixed seed 20260926 so that the reviewer-response sensitivity result remains reproducible; this seed exception is documented rather than silently changed.


## Exact matched baseline-versus-nested comparison for Table S30

`src/revision_comment_3_R2_baseline_contrast.py` reuses the exact outer-test memberships generated by the nested procedure using `(Repeat, Outer_fold, Row_index)` keys. It does not instantiate a second repeated-CV splitter. Under this corrected apples-to-apples aggregation, the AF categorical baseline has repeat-pooled R2 = 0.793 versus 0.742 for nested AF and lower RMSE in all 20 matched repetitions. The Qm P/TP power-law baseline has repeat-pooled R2 = -0.027 versus 0.846 for nested Qm, with the nested procedure lower in RMSE in all 20 matched repetitions. These values supersede the earlier unmatched-partition summaries.
