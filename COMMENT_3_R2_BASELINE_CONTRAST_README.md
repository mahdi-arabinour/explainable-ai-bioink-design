# Reviewer 2 Comment 3: simple baselines and matched incremental-value contrast

This revision evaluates whether multivariable modeling adds predictive value beyond deliberately simple baselines for AF and Qm.

Primary scripts:
- `src/revision_comment_3_R2_baselines.py`: angle-only AF and P/TP-only Qm baselines under repeated 5-fold CV and LOFO.
- `src/revision_comment_3_R2_baseline_contrast.py`: apples-to-apples comparison of the baseline predictions with the selection-aware nested multivariable outer predictions.

For repeated validation, the second script pools held-out predictions within each of the same 20 outer repetitions before calculating R2, MAE, and RMSE. It reports descriptive paired differences and the fraction of repeats in which the nested multivariable procedure improves each metric. The repeated partitions overlap and are therefore not treated as independent inferential replicates.

Key findings:
- AF categorical angle-only baseline: repeat-pooled R2 = 0.793 and RMSE = 0.150 versus nested multivariable R2 = 0.742 and RMSE = 0.167. The nested multivariable procedure had lower RMSE in 0/20 matched repetitions (the categorical baseline had lower RMSE in all 20). LOFO R2 was 0.791 for the categorical angle baseline versus 0.762 for the nested multivariable procedure.
- Qm P/TP power-law baseline: repeat-pooled R2 = -0.027 and RMSE = 0.674 versus nested multivariable R2 = 0.846 and RMSE = 0.255. The nested multivariable procedure had lower RMSE in 20/20 matched repetitions. LOFO remained poor for the nested multivariable procedure (R2 = -1.458), so the incremental value is restricted to within-domain prediction.

The corresponding manuscript interpretation is target-specific: multivariable complexity is unnecessary for AF point prediction, whereas Qm shows clear within-domain incremental value beyond the one-descriptor baselines.
