# Comment 12-R1 / Comment 9-R2: SR nested-LOFO uncertainty

The revised analysis reports both the eight formulation-specific nested-LOFO outer-fold diagnostics and an evaluation-stage formulation-block bootstrap of the already-held-out outer predictions.

Run:

```bash
python src/revision_comments_12_R1_9_R2_lofo_uncertainty.py
```

The bootstrap samples the eight held-out formulation blocks with replacement for 10,000 replicates while preserving all observations within a sampled formulation. It recomputes pooled R2, MAE, and RMSE from the fixed nested-LOFO outer predictions. It does **not** rerun model selection or refit the nested training procedure, so the interval quantifies evaluation-sample sensitivity conditional on the existing outer predictions rather than full training-procedure uncertainty.

For SR, the original pooled nested-LOFO R2 is 0.393. The formulation-block bootstrap median R2 is 0.364 with a 95% percentile interval of 0.179 to 0.537. Delete-one-formulation evaluation sensitivity keeps pooled R2 positive for all eight omissions, ranging from 0.315 to 0.445. Individual formulation R2 values nevertheless range from -2.218 to 0.485 (median 0.101; five of eight positive). The supported interpretation is therefore positive aggregate formulation-level signal with heterogeneous formulation-specific transfer.

Supplementary Table S29 reports the all-target formulation-block uncertainty summary to avoid selective reporting of SR alone.
