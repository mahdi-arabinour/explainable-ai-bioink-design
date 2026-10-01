# Archived intermediate revision scripts

These scripts are retained only for historical traceability and are **not** part of the canonical final reviewer-revision workflow.

- `apply_reviewer_feature_cleanup.py` was a one-time patch utility. The canonical base script and portable notebook already contain the duplicate-reduced feature definitions, so re-running the patch is unnecessary and can emit misleading "block not found" warnings.
- `revision_comment4_5_*` files were intermediate development/rerun helpers used while resolving duplicate-feature issues.
- `revision_comments_6_7_R1_8_R2_ablation.py` is the superseded fixed-model S19 experiment. The authoritative S19 analysis is `src/revision_comments_6_7_R1_8_R2_ablation_selected.py`, which preserves the submitted S19 model identities and explicitly distinguishes the two mixed-model rows.

Do not call these archived scripts from the final run order or future master runner.
