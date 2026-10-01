# Reviewer 1 Comment 10: dimensionless stress-scale sensitivity

Reviewer 1 correctly noted that `P_over_TP` has compound units and cannot be presented as a dimensionless physical matching rule. Rather than only demoting the descriptor, this revision tests a physically motivated dimensionless alternative using quantities directly available in the curated source tables.

The analysis defines:

`P_over_Yield_dimless = Pressure_kPa * 1000 / Yield_stress_mean_Pa`

This is a dimensionless applied-pressure/yield-stress ratio. Under a fixed-geometry capillary approximation, a nominal wall-stress/yield-stress ratio differs from it only by a constant geometry factor. It is intentionally **not** labeled as an exact in-nozzle wall shear stress because the curated source tables do not specify the local pressure drop, internal conical radius profile, or volumetric flow/shear-rate information required for a constitutive Cross-based nozzle-flow calculation without additional assumptions.

Run:

```bash
python src/revision_comment_10_R1_dimensionless_stress_ratio.py
```

The script performs two checks:

1. Single-descriptor linear and positive power-law baselines for `P_over_TP` and `P_over_Yield_dimless` for Qm and SR using repeated five-fold cross-validation (20 repeats) and pooled LOFO.
2. A matched multivariable substitution sensitivity in which `P_over_TP` is replaced one-for-one by `P_over_Yield_dimless` in the duplicate-reduced engineered SVR representation, with all other features and validation settings unchanged.

Key results:

- Qm power-law one-descriptor R2: P/TP = -0.490 repeated and -0.455 LOFO; P/yield = -0.254 repeated and -0.179 LOFO. The dimensionless ratio is less negative but remains non-predictive as a stand-alone model.
- SR one-descriptor R2 values remain negative for both descriptors under both validation schemes.
- Matched Qm engineered-SVR substitution: P/TP R2 = 0.697 repeated and -0.558 LOFO; P/yield R2 = 0.674 and -0.880.
- Matched SR engineered-SVR substitution: P/TP R2 = 0.055 repeated and 0.390 LOFO; P/yield R2 = 0.027 and 0.196.

The reviewer-requested physical sensitivity therefore does not support a transferable one-parameter rule and does not outperform P/TP in the matched multivariable representation. P/TP is retained as a conditional empirical descriptor and model-reliance finding, not as a dimensionless law or causal mechanism.

Primary outputs are under `results/revision_comment_10_R1_dimensionless_stress_ratio/`. Supplementary Table S28 is copied to `supplementary/final_revision/tables/`.
