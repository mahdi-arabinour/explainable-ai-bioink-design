# Reviewer 2 Comment 8: Table 2 / Table S19 reconciliation audit

This comment overlaps analytically with Reviewer 1 Comment 6. No new predictive model is fitted for Comment 8-R2.

- Table 2 reports selection-aware nested outer-test performance.
- Table S19 is a separate core-versus-engineered representation-sensitivity/continuity analysis that preserves the submitted S19 model identities after duplicate-feature cleanup.
- Six of eight S19 comparisons use the same algorithm for the core and engineered representations.
- Pr LOFO and SR repeated K-fold use different algorithms and are not interpreted as pure feature-ablation effects.
- `src/revision_comment_6_R1_reconciliation.py` generates Supplementary Table S27 from the archived nested and S19 outputs.
- `src/revision_comment_8_R2_reconciliation_audit.py` independently checks every S27 row against those archived outputs and fails if any field disagrees.

Run after the S19 and S27 scripts:

```bash
python src/revision_comment_6_R1_reconciliation.py
python src/revision_comment_8_R2_reconciliation_audit.py
```

The audit writes machine-readable results to `results/revision_comment_8_R2_reconciliation_audit/`.
