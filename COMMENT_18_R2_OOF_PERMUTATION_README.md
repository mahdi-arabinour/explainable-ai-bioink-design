# Reviewer 2 Comment 18 - held-out permutation importance and citation audit

Reviewer 2 noted that the submitted permutation-importance analysis appeared to be computed on full-data refits and that the cited Altmann PIMP paper did not match the implemented method.

## Final revised treatment

- Permutation importance is computed only on untouched held-out test observations.
- The fixed target-specific reference model is refit on each training fold, then one feature at a time is permuted only in the corresponding test fold.
- Importance is `permuted test RMSE - baseline test RMSE`.
- The primary repeated attribution uses five-fold cross-validation with five repeats (25 held-out folds) and 10 feature permutations per fold.
- A leave-one-formulation-out permutation analysis is retained as a formulation-level sensitivity check.
- The Altmann/PIMP citation is not used because no corrected PIMP p-value procedure is performed.
- The manuscript cites Breiman's permutation-based variable-importance principle and fully specifies the held-out delta-RMSE implementation.

## Reproducibility check added during final audit

The held-out permutation summaries were hash-compared before and after the SHAP reproducibility fix and were byte-for-byte unchanged for both repeated and LOFO analyses. During the final audit, the descriptive SHAP permutation explainer was found to lack an explicit seed. `seed=42` is now passed to the SHAP permutation explainer (and NumPy is initialized with the same seed). This does not change the held-out permutation outputs, target-leading features, nested-validation performance, or candidate screening. It only refreshes some descriptive SHAP values and cross-method rank-agreement decimals. Figure 2, Table 3, Figures S6-S7, Table S22, and the Comment 17-R2 response were refreshed from the seeded SHAP outputs. A complete package-wide clean-room execution remains a separate final-stage task.

The final descriptive cross-method rank agreements are:

- SR: 0.693233 (reported as 0.693)
- Qm: 0.894505 (reported as 0.895)
- AF: 0.380620 (reported as 0.381)

Pr remains excluded from the rank-agreement summary.

## Automated audit

Run:

```bash
python src/revision_comment_18_R2_oof_permutation_audit.py
```

The audit verifies the held-out `Xte/yte` implementation, RMSE scoring, 25 repeated held-out folds, 10 permutations per fold, eight LOFO folds, target-leading OOF features, the authoritative Figure S5/Table S22 copies, the recorded SHAP seed, and the absence of superseded in-sample XAI artifacts from active Supplementary folders.
