# Reviewer 2 Comment 17 - XAI separation audit

Reviewer 2 objected to the former composite/consensus XAI score because it averaged max-normalized held-out permutation importance and mean absolute SHAP values despite their different meanings and units.

## Final revised treatment

- No composite or averaged XAI score is used in the revised analysis.
- Held-out permutation importance is retained in its native delta-RMSE scale.
- Mean absolute SHAP is retained separately in the target scale.
- Table 3 is ordered by held-out permutation rank and reports SHAP rank independently.
- Cross-method Spearman rank agreement is descriptive only for SR, Qm, and AF.
- Pr is excluded from the rank-agreement summary and its attribution is descriptive only.

## Package audit

`src/revision_comment_17_R2_xai_audit.py` verifies the authoritative revised outputs and fails if:

- a composite/consensus score appears in the revised Table 3 output;
- Pr re-enters the cross-method rank-agreement summary;
- the repeated held-out permutation analysis is mislabeled;
- the authoritative Table S22 does not retain separate permutation and SHAP outputs.

The XAI analysis uses five-fold cross-validation with five repeats (25 held-out folds) for the primary repeated held-out permutation analysis, with 10 feature permutations per held-out fold. A legacy label `Repeated5x20` was corrected to `Repeated5x5`; that label correction itself was metadata-only. During the final Comment 18-R2 reproducibility audit, the SHAP permutation explainer was additionally given an explicit seed of 42. This leaves all held-out permutation results and leading features unchanged but refreshes some descriptive SHAP values and cross-method rank-agreement decimals.

Superseded pre-revision XAI tables and figures, including the former consensus score and original full-data permutation outputs, were moved out of the active Supplementary folders to `archive/legacy_pre_revision_xai/` and are retained only for historical traceability.
