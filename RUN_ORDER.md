# Canonical run order

## Final clean-room entry point

Run from the repository root:

```bash
python run_full_analysis.py
```

`run_full_analysis.py` is the only canonical end-to-end entry point. It uses relative repository paths, performs preflight checks, removes reproducible generated outputs before a certification run, executes the analysis sequentially with fail-fast behavior, records per-step logs/timing, consolidates the final SI artifacts, and validates the manuscript-level numerical checks.

Useful non-scientific checks:

```bash
python run_full_analysis.py --preflight-only
python run_full_analysis.py --dry-run
```

`--no-clean` is available only for debugging and cannot be used to certify a final clean-room release.

## Canonical dependency order

1. `src/prepare_canonical_inputs.py`
   - rebuilds all derived modeling datasets from the three curated source tables S1-S3;
   - writes target-specific datasets, EDA summaries, feature definitions, and correlations;
   - keeps compatibility columns in distributed processed tables but excludes deterministic duplicates from canonical model feature sets.
2. `src/generate_descriptive_figures.py`
   - regenerates Supplementary Figures S1-S2 directly from canonical data;
   - regenerates manuscript Figure 3 from the current fixed duplicate-reduced reference models on source-bounded 5-kPa display grids;
   - reads no archived prediction artifact.
3. `src/revision_comment_3_R2_baselines.py`
4. `src/revision_comment_10_R1_dimensionless_stress_ratio.py`
5. `src/revision_comments_4_R1_16_R2_nested_cv.py`
6. `src/revision_comments_4_R1_16_R2_r2_aggregation.py`
7. `src/revision_comment_3_R2_baseline_contrast.py`
   - reuses the exact nested outer-test memberships keyed by `(Repeat, Outer_fold, Row_index)`;
   - does not create an independent repeated-CV splitter for Table S30.
8. `src/revision_comments_12_R1_9_R2_lofo_uncertainty.py`
9. `src/revision_comments_8_R1_17_18_R2_xai.py`
10. `src/revision_comments_6_7_R1_8_R2_ablation_selected.py`
11. `src/revision_comment_6_R1_reconciliation.py`
12. `src/revision_comment_8_R2_reconciliation_audit.py`
13. `src/revision_comments_2_R1_6_7_R2_candidate_set.py`
14. `src/revision_comment_7_R2_pr_gate_sensitivity.py`
15. `src/revision_comments_2_R1_6_R2_bootstrap_ranking.py`
16. `src/finalize_revision_assets.py`
17. master-runner SI consolidation
18. `src/revision_comment_17_R2_xai_audit.py`
19. `src/revision_comment_18_R2_oof_permutation_audit.py`
20. master-runner numerical validation

The XAI audit scripts intentionally run after SI consolidation because they verify final-review copies generated in the same run.

## Optional provenance audit

If a legally obtained local copy of the Perin Supporting Information PDF is available:

```bash
python run_full_analysis.py --provenance-pdf /path/to/Perin_supporting_information.pdf
```

Without that PDF, the provenance audit is `SKIPPED-OPTIONAL`; the core clean-room analysis does not fail.

## Archive rule

The only permitted active analysis dependency under `archive/` is:

`archive/legacy_pre_revision_candidate_ranking/tables/029_tables_final_top_strict_design_candidates.csv`

It is used only by the retrospective 1,000-replicate bootstrap of the originally submitted strict-candidate ordering. No script under `archive/intermediate_revision_scripts/` is part of the canonical workflow.

## Retired execution paths

Retired pre-integration execution paths have been moved under `archive/legacy_pre_integration_workflow/`. They are retained only for provenance/history and are not active. The master workflow uses `src/prepare_canonical_inputs.py` for authoritative data preparation and the reviewer-specific scripts listed above for reportable analyses.

## Validation settings

- Nested repeated CV: outer 5-fold cross-validation repeated 20 times; inner 5-fold model/feature-set selection.
- Nested LOFO: one formulation held out in the outer loop; inner LOFO among remaining formulations.
- Random seed: 42 for canonical core stochastic procedures.
- Retrospective candidate-ranking bootstrap: explicit fixed seed `20260926`.
- Model hyperparameters are prespecified; nested validation selects algorithm and feature set rather than tuning a broad hyperparameter grid.
