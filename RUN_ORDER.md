# Run order

## Recommended reproducibility path

1. Create the environment from `environment.yml` or install `requirements.txt`.
2. Start Jupyter from the repository root.
3. Open `notebooks/01_analysis_from_curated_data.ipynb`.
4. Run every cell in order.
5. Confirm that generated outputs appear under `data/processed/`, `results/tables/`, and `results/figures/`.
6. Compare key outputs with the exact files under `supplementary/tables/` and `supplementary/figures/`.

## Optional source-extraction audit

The archived notebook `notebooks/archive/original_colab_workflow_through_cell_24G.ipynb` retains the original PDF-extraction cells. Those cells require legally obtained copies of the source article and supplementary information. Publisher PDFs are not distributed in this repository.

## Main validation settings

- Repeated five-fold cross-validation: 20 repeats
- Leave-one-formulation-out validation: groups defined by polymer formulation
- Metrics: R2, MAE, RMSE, and Spearman correlation
- Random seed: 42 where specified in the workflow
