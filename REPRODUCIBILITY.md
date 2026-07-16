# Reproducibility and claim boundaries

This repository supports internal validation and model interpretation for a sparse, published alginate-hyaluronic acid bioink dataset.

## Interpretation boundaries

- Repeated K-fold validation can place related formulation observations in both training and validation folds; it estimates within-domain interpolation rather than independent formulation transfer.
- Leave-one-formulation-out validation is the stricter formulation-level test.
- SHAP and permutation importance quantify model reliance, not causal mechanisms.
- Engineered ratios such as `P_over_TP` are empirical descriptors and are not dimensionless physical groups.
- Candidate screening is restricted to combinations represented in the source data.
- Candidate rankings require prospective experimental validation.

## Expected dataset sizes

- Pr: 40 observations
- SR: 43 observations
- Qm: 40 observations
- AF: 96 observations
