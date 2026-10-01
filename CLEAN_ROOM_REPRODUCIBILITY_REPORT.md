# CLEAN-ROOM REPRODUCIBILITY REPORT

## Final status

**FULL CLEAN-ROOM PASS**

The final integrated analysis was rebuilt from a freshly extracted Stage-12 clean input package. The canonical Master Runner completed from zero, the Master Notebook independently completed Run All from a separate fresh extraction, all mandatory numerical checks passed, runner/notebook scientific outputs reconciled, and the corrected Manuscript/SI/Response values remained synchronized with the final clean outputs.

## Certified clean-room inputs

The active release starts from only the three curated source-derived tables plus one explicitly documented historical bootstrap artifact:

- `data/processed/rheology_table_S1.csv` — SHA-256 `a2e37aee29e332144328114d0a3363adaade1983644046189ffcd928df5e1236`
- `data/processed/printability_table_S2.csv` — SHA-256 `2e415b3373434c8c1ba6ac2d37f3559bb9e8a5717923033513e13b3bb31412e1`
- `data/processed/angle_fidelity_table_S3.csv` — SHA-256 `d498ee609fa6ec34639a0ef0d0f30e6cde993e4b7f476211136a5ec7ac6e151e`
- `archive/legacy_pre_revision_candidate_ranking/tables/029_tables_final_top_strict_design_candidates.csv` — SHA-256 `9dd2a5feda135a84032f3330178271b777c3dd5c9de330fa5fb4f8a79d07427e`

The historical archive dependency is used only for the retrospective bootstrap of the originally submitted candidate ordering. No other active scientific script reads `archive/`. Publisher PDFs are not required for the core workflow.

Stage-12 clean input ZIP SHA-256: `8bed158a295a54a30a274f9fb1154c9579ba1c1ff654c873aad222adacd3c2e2`.

## Canonical entry points

- Master Runner: `run_full_analysis.py` — SHA-256 `d473787c7d540720a1dfb6ec33fdaa017fe5c98dfe2a050e7cfa800c33c277f7`
- Master Notebook: `notebooks/00_master_analysis.ipynb` — SHA-256 `3563549e3d6dbdebbe8d7da7908384ad4e62b66e0120bfce2db96697ee2884e6`
- Executed certification notebook snapshot: SHA-256 `02e2f6dfb5b3f6a45d8033ea41cb48afabc2647c7bc503042e4fbe79a41b5589`

All active paths are repository-relative. Canonical stochastic procedures use explicit seed 42; the retrospective historical bootstrap preserves its documented fixed seed `20260926`.

## Final Master Runner execution

Status: **PASS**

- Fresh extraction: `/mnt/data/master_integration_stage1/stage12_recert_runner/final_code`
- Runtime: `688.1 s`
- Master numerical validation: `50/50 PASS`
- Optional publisher-PDF provenance step: `SKIPPED-OPTIONAL`
- Final runner log: `STAGE_12_FINAL_RUNNER.log`

The master runner regenerated canonical data, descriptive figures, reviewer-specific analyses, final scientific assets, consolidated SI tables/figures, XAI audits, and final numerical validations without reading pre-existing active result outputs.

## Final Master Notebook execution

Status: **PASS**

- Fresh independent extraction: `/mnt/data/master_integration_stage1/stage12_recert_notebook/final_code`
- Runtime: `692.43 s`
- Code cells executed: `19/19`
- Execution counts: `1..19`
- Cell errors: `0`
- Final mandatory notebook assertions: PASS
- Executed notebook snapshot: `00_master_analysis_EXECUTED_STAGE12_RECERTIFIED.ipynb`

The notebook includes explicit Table S30 exact-partition checks in addition to the primary nested/LOFO/R2-reconciliation checks.

## Runner versus Notebook reproducibility

Status: **PASS**

Across `results/`:

- Runner files: 136
- Notebook files: 112
- Common scientific/control files: 112
- Exact SHA-256 matches: 101
- Semantic matches: 11
- Runner-only controls/logs: 24
- Scientific discrepancies: 0

Breakdown of non-binary exact matches: nine XLSX workbooks matched cell-by-cell within `1e-12`; one JSON differed only by elapsed runtime; one PDF differed only in container metadata while its corresponding source PNG was byte-identical. All 65 common CSV outputs and all 23 common PNG outputs were exact SHA-256 matches.

## Mandatory numerical results from the final run

### Nested repeated outer-fold mean R2

- Pr: `-0.0831518511` (reported `-0.083`)
- SR: `-0.0736261387` (reported `-0.074`)
- Qm: `0.7624001750` (reported `0.762`)
- AF: `0.7120623642` (reported `0.712`)

### Repeat-pooled OOF R2

- Pr: `0.3212343662` (reported `0.321`)
- SR: `0.0331663735` (reported `0.033`)
- Qm: `0.8455619520` (reported `0.846`)
- AF: `0.7418599125` (reported `0.742`)

### Nested LOFO R2

- Pr: `-0.3477439786`
- SR: `0.3929835700`
- Qm: `-1.4582603196`
- AF: `0.7623138807`

### Uncertainty, screening, ranking, and XAI

- SR formulation-block bootstrap 95% interval: `0.1792659725–0.5369577537`
- Jointly observed formulation-pressure conditions: `34`
- Strict measured-domain candidate set: `14`
- Relaxed Pr sensitivity set: `21`
- Additional relaxed conditions: `7`
- Retrospective bootstrap leader: `49.1%` rank 1; `63.3%` top 3
- Pairwise bootstrap intervals separated from leader: `0/16`
- XAI rank agreement: SR `0.693233`; Qm `0.894505`; AF `0.380620`; Pr excluded

### Corrected exact matched Table S30 comparison

A Stage-10 audit found that the earlier baseline contrast used the same repeat labels but not the exact nested outer-fold memberships. The script was repaired so baseline predictions reuse `(Repeat, Outer_fold, Row_index)` from the nested output. Final Stage-12 clean-room validation independently confirms exact membership equality for AF and Qm.

- AF categorical baseline repeat-pooled R2: `0.7928570933` (reported `0.793`)
- Nested AF repeat-pooled R2: `0.7418599125` (reported `0.742`)
- AF categorical baseline lower RMSE: `20/20` matched repetitions (nested lower RMSE `0/20`)
- Qm P/TP power-law baseline repeat-pooled R2: `-0.0274234387` (reported `-0.027`)
- Nested Qm repeat-pooled R2: `0.8455619520` (reported `0.846`)
- Nested Qm lower RMSE: `20/20` matched repetitions

These corrected matched-partition values supersede the earlier unmatched-partition Table S30 values `0.794`, `-0.046`, and the `19/20` AF statement. The scientific interpretation is unchanged: added multivariable complexity is unnecessary for AF point prediction, while it improves Qm prediction within the sampled domain; Qm formulation-level extrapolation remains poor.

## Manuscript / SI / reviewer-response synchronization

Status: **PASS**

The final document guard contains `60/60` PASS checks. It verifies, against the Stage-12 runner output:

- Manuscript Table 1 corrected SR maximum `7.511` and Qm maximum `3.336`.
- Supplementary Table S30 row-by-row rounded values against the final exact-matched clean output.
- Manuscript and Response contain corrected `0.793` and `-0.027` values and no longer contain the obsolete `19 of 20` claim.

Earlier Stage-10 audits also cross-checked Manuscript Tables 1-5, narrative claims, Supplementary Tables/figures, and the Response to Reviewers. Table S24 publisher-PDF cell-by-cell provenance verification remains optional because the external publisher PDF is not distributed with the package. The internal row/count consistency checks remain PASS.

## Discrepancies found and repaired during integration

1. **Table 1 source-statistics mismatch:** manuscript SR max `7.510` and Qm max `3.330` were corrected to source-derived `7.511` and `3.336`.
2. **Table S27 aggregation bug:** the reconciliation exporter incorrectly used repeat-pooled R2 where Table 2 uses mean outer-fold R2. It was repaired and independently recertified.
3. **Manuscript Figure 3 reproducibility gap:** an active canonical descriptive-figure generator was added so Figure 3 and S1-S2 regenerate directly from canonical data/current reference models.
4. **Table S30 partition mismatch:** simple baselines originally used an independent repeated-CV splitter. The contrast now reuses exact nested outer memberships. Final runner and notebook both reproduce the corrected values.

No unresolved scientific discrepancy remains in the active workflow.

## Environment

- Python: `3.13.5`
- Platform: `Linux-6.18.44-x86_64-with-glibc2.41`
- NumPy: `2.3.5`
- pandas: `2.2.3`
- scikit-learn: `1.8.0`
- SciPy: `1.17.0`
- XGBoost: `3.1.3`
- SHAP: `0.50.0`

The certified dependency versions are pinned in `requirements.txt` and `environment.yml`.

## Limitations / explicit exclusions

- The publisher Supporting Information PDF provenance re-audit is optional and was not rerun because the external PDF is not part of the release package. This does not block the core clean-room certification.
- The retrospective candidate-ranking bootstrap intentionally depends on the explicitly documented historical candidate artifact; no other archive artifact is an active analysis input.
- The Graphical Abstract / TOC graphic is outside this reproducibility workflow and remains pending separately.
- The statistical interpretation boundaries documented in `REPRODUCIBILITY.md` remain applicable, including limited formulation counts and the distinction between descriptive model reliance and causal evidence.

## Release gate

All scientific and technical prerequisites for Stage 13 have passed. Stage 13 may create the final release ZIP from the finalized Stage-12 package without changing scientific code or values.
