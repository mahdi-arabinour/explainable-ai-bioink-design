# Reviewer 1 Comment 6 and Reviewer 2 Comment 8: Table 2 / Table S19 reconciliation

This revision makes the two analyses explicitly non-equivalent rather than treating their different values as inconsistencies.

- Table 2 reports selection-aware nested outer-test performance.
- Table S19 preserves the submitted core-versus-engineered model identities after duplicate-feature cleanup.
- Six of eight S19 comparisons use the same algorithm for core and engineered representations.
- Pr LOFO and SR repeated K-fold use different algorithms and are therefore not interpreted as pure feature-ablation effects.
- `src/revision_comment_6_R1_reconciliation.py` generates Supplementary Table S27 from the archived nested and S19 outputs and performs no model fitting.

Run after `src/revision_comments_6_7_R1_8_R2_ablation_selected.py`:

```bash
python src/revision_comment_6_R1_reconciliation.py
```

Reviewer 2 Comment 8 uses the same reconciliation and adds `src/revision_comment_8_R2_reconciliation_audit.py` as a row-by-row consistency check; it performs no model fitting.
