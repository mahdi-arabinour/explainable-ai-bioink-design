# Reviewer 2 Comment 7: Pr candidate-screening gate

The final measured-domain candidate screen does not use predicted Pr or predicted AF for eligibility. The main set uses source-measured Pr in 0.95-1.05 and at least one source-measured AF in 1.00-1.25 at the same jointly observed formulation-pressure condition.

Because the 0.95-1.05 measured-Pr interval is an operational screening tolerance rather than a model-derived confidence boundary, the revision also reports a threshold-sensitivity check with measured Pr in 0.90-1.10 while keeping the AF criterion unchanged. The main screen contains 14 conditions; the relaxed sensitivity contains 21, adding seven conditions. The sensitivity does not redefine the main candidate set.

Run after the main candidate-set script:

```bash
python src/revision_comments_2_R1_6_7_R2_candidate_set.py
python src/revision_comment_7_R2_pr_gate_sensitivity.py
```

Primary outputs are stored in `results/revision_comment_7_R2_pr_gate_sensitivity/`.
