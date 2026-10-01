# Final reviewer-revision package

This directory is the consolidated computational package supporting the revised manuscript **Explainable AI-Guided Bioink Design: Model-Derived Polymer-Pressure Relationships in Alginate-Hyaluronic Acid Bioinks**.

The final reportable analyses use duplicate-reduced features and incorporate the reviewer-requested checks as separate, auditable scripts rather than silently overwriting the original exploratory outputs.

## Final analysis sequence

The canonical base script and portable notebook already contain the duplicate-reduced feature definitions; the former one-time patch helper has been archived and is not part of the final run order. The optional provenance audit requires a legally obtained local copy of the source Supporting Information PDF and is not required for the predictive reruns.

1. `src/revision_comment_3_R2_baselines.py` evaluates angle-only AF and P/TP-only Qm baselines.
2. `src/revision_comment_10_R1_dimensionless_stress_ratio.py` tests the reviewer-requested dimensionless applied-pressure/yield-stress alternative and a matched P/TP substitution sensitivity (Supplementary Table S28).
3. `src/revision_comments_4_R1_16_R2_nested_cv.py` performs nested repeated five-fold and nested leave-one-formulation-out model/feature-set selection and evaluation.
4. `src/revision_comments_4_R1_16_R2_r2_aggregation.py` summarizes complementary fold-mean and repeat-pooled out-of-fold R2 aggregation from the same nested repeated-CV predictions (Supplementary Table S26).
5. `src/revision_comment_3_R2_baseline_contrast.py` performs the matched repeat-pooled baseline-versus-nested comparison used in Supplementary Table S30.
6. `src/revision_comments_12_R1_9_R2_lofo_uncertainty.py` summarizes formulation-level nested-LOFO diagnostics and performs the evaluation-stage formulation-block bootstrap reported in Supplementary Table S29.
7. `src/revision_comments_8_R1_17_18_R2_xai.py` computes held-out permutation importance and separate SHAP attribution; no composite consensus score is used.
8. `src/revision_comment_17_R2_xai_audit.py` audits the authoritative revised XAI outputs, verifies Pr exclusion from rank-agreement claims, and checks the five-fold x five-repeat held-out permutation metadata.
9. `src/revision_comment_18_R2_oof_permutation_audit.py` verifies the held-out permutation implementation, exact repeated/LOFO fold structure, active Figure S5/Table S22 copies, and deterministic XAI metadata.
10. `src/revision_comments_6_7_R1_8_R2_ablation_selected.py` regenerates the duplicate-reduced core-versus-engineered comparisons used for Supplementary Table S19 while preserving the submitted S19 model identities. Six of eight rows use the same algorithm; Pr LOFO and SR repeated K-fold use different core/engineered algorithms and are not interpreted as pure feature-ablation effects.
11. `src/revision_comment_6_R1_reconciliation.py` generates Supplementary Table S27, which reconciles Table 2 nested performance with Table S19 row by row and flags whether the S19 algorithm is held constant.
12. `src/revision_comment_8_R2_reconciliation_audit.py` rechecks all eight Table S27 rows against the archived nested and S19 outputs and fails on any mismatch; it performs no model fitting.
13. `src/revision_comments_2_R1_6_7_R2_candidate_set.py` creates the unordered measured-domain candidate set from measured Pr and AF values; no statistically resolved within-set ranking or model-derived optimum is assigned.
14. `src/revision_comment_7_R2_pr_gate_sensitivity.py` evaluates sensitivity of the measured-Pr operational gate and produces Supplementary Table S31.
15. `src/revision_comments_2_R1_6_R2_bootstrap_ranking.py` implements the reviewer-suggested formulation-cluster bootstrap sensitivity analysis of the originally submitted 17-candidate ordering using the archived legacy candidate list as its documented input.
16. `src/finalize_revision_assets.py` regenerates the final replacement figures and code-generated TOC graphic from the revised outputs.

Optional source-to-repository audit:

```bash
python src/audit_comment_2_R2.py /path/to/Perin_supporting_information.pdf
```

## Authoritative revision artifacts

`results/revision_*` contains the complete comment-specific outputs. For convenience, the exact final tables and replacement figures used to assemble the revised Supporting Information are copied to:

- `supplementary/final_revision/tables/`
- `supplementary/final_revision/figures/`

Where a final-revision file exists, the `supplementary/final_revision/` version is authoritative for the revised manuscript. Superseded pre-revision XAI artifacts were moved out of the active Supplementary folders to `archive/legacy_pre_revision_xai/` so the former consensus score and in-sample permutation outputs cannot be mistaken for revised results. Superseded model-ranked candidate/design-atlas artifacts are likewise quarantined under `archive/legacy_pre_revision_candidate_ranking/`, and one-time/intermediate development scripts are under `archive/intermediate_revision_scripts/`. The final SHAP permutation explainer uses an explicit seed of 42; seeded SHAP outputs are authoritative for Figure 2, Table 3, Figure S6, and Table S22.

## Reportable validation values

Nested repeated outer-fold mean R2: Pr -0.083, SR -0.074, Qm 0.762, AF 0.712. Complementary repeat-pooled out-of-fold R2 means across the 20 repetitions are Pr 0.321, SR 0.033, Qm 0.846, and AF 0.742. Both summaries use exactly the same untouched outer predictions; the difference reflects R2 aggregation across small folds rather than a change in model fitting.

Nested LOFO pooled R2: Pr -0.348, SR 0.393, Qm -1.458, AF 0.762. For SR, the 10,000-replicate formulation-block bootstrap of fixed outer predictions gave a 95% percentile R2 interval of 0.179-0.537; delete-one-formulation pooled R2 remained positive from 0.315 to 0.445, while individual formulation R2 values ranged from -2.218 to 0.485.

The AF angle-only categorical baseline achieved R2 = 0.775 under repeated five-fold cross-validation and R2 = 0.791 under LOFO. P/TP-only Qm baselines had negative R2 under both validation schemes. A reviewer-requested dimensionless applied-pressure/yield-stress ratio produced less negative Qm single-descriptor R2 (best power-law values -0.254 repeated and -0.179 LOFO) but remained below zero, did not improve SR, and did not outperform P/TP in matched multivariable substitution tests. The result supports P/TP only as a conditional empirical descriptor, not a transferable physical rule.

## Candidate analysis

The final screening step considers 34 formulation-pressure conditions jointly observed in the source printability and angular-fidelity datasets. Fourteen conditions meet measured Pr = 0.95-1.05 and have at least one measured AF = 1.00-1.25. They remain an unordered measured-domain set for the main analysis. As a reviewer-requested uncertainty sensitivity test, a 1,000-replicate formulation-cluster bootstrap of the originally submitted 17 strict candidates retained 1ALG8HA / 65 kPa / 120 degrees as the modal leader in 49.1% of replicates and in the top three in 63.3%; however, none of its 16 pairwise 95% bootstrap difference intervals excluded zero. The condition is therefore retained only as a recurrent leading hypothesis, not a statistically unique winner or model-derived optimum.


### Comment 3-R2 matched baseline contrast

Table S30 adds a directly matched comparison of the reviewer-requested simple baselines with the selection-aware nested multivariable procedures. The comparison uses identical held-out repeat partitions and repeat-pooled OOF metrics. AF shows no consistent incremental predictive gain from multivariable complexity, while Qm shows a large within-domain gain beyond P/TP-only baselines; Qm LOFO transfer remains poor.



## Master clean-room entry point

Use `python run_full_analysis.py` from the repository root for final certification. The readable companion notebook is `notebooks/00_master_analysis.ipynb`. Supplementary Figures S1-S2 and manuscript Figure 3 are regenerated by `src/generate_descriptive_figures.py` from canonical data/current fixed reference models.
