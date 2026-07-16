# Explainable AI-Guided Bioink Design

Data, code, notebooks, exact supplementary tables, and figure-generation materials accompanying the manuscript:

**Explainable AI-Guided Bioink Design: Model-Derived Polymer-Pressure Relationships in Alginate-Hyaluronic Acid Bioinks**

## Scope

This repository supports a secondary, target-specific machine-learning and explainable-AI reanalysis of the alginate-hyaluronic acid bioink dataset reported by Perin et al. No new experimental measurements were generated in the present study.

The analysis models four outputs independently:

- printability ratio (`Pr`)
- spreading ratio (`SR`)
- mass flow rate (`Qm`)
- angular fidelity (`AF`)

The repository includes curated source tables, processed modeling datasets, repeated five-fold cross-validation, leave-one-formulation-out validation, model-selection outputs, SHAP and permutation-importance analyses, candidate-screening outputs, and engineered-feature ablation results.

## Repository structure

```text
notebooks/                 Canonical analysis notebook and archived Colab workflow
src/                       Python export of the curated-data notebook
data/processed/            Curated and processed modeling datasets
data/source/               Source-data citation and provenance notice
results/tables/             Main generated result tables
results/figures/            Main generated figures
supplementary/tables/       Exact CSV/XLSX files supporting Tables S1-S19
supplementary/figures/      Exact supplementary figure files
supplementary/figures_pdf/  Vector/PDF supplementary figure materials
docs/                       Cell index, file register, and reproducibility documentation
```

## Quick start

### Conda

```bash
conda env create -f environment.yml
conda activate explainable-ai-bioink-design
jupyter lab
```

Open:

```text
notebooks/01_analysis_from_curated_data.ipynb
```

Run all cells in order from the repository root.

### Google Colab

Clone the repository into `/content/explainable-ai-bioink-design`, open `notebooks/01_analysis_from_curated_data.ipynb`, and run the cells in order. The main notebook begins from the included curated tables and does not require redistribution of publisher PDFs.

## Reproducibility notes

The repeated K-fold results estimate observation-level interpolation within the sampled design space. Leave-one-formulation-out validation is the stricter assessment of transfer to unseen formulations. Candidate rankings are model-derived, hypothesis-generating priorities within the supported source-data domain; they are not experimentally confirmed optima.

The archived notebook under `notebooks/archive/` preserves the original Colab workflow, including optional PDF-extraction and packaging cells. The portable notebook under `notebooks/01_analysis_from_curated_data.ipynb` is the recommended analysis entry point.

## Data provenance

The source experimental data were reported by:

Perin, F.; Spessot, E.; Famà, A.; Bucciarelli, A.; Callone, E.; Mota, C.; Motta, A.; Maniglio, D. *Modeling a Dynamic Printability Window on Polysaccharide Blend Inks for Extrusion Bioprinting.* **ACS Biomaterials Science & Engineering** 2023, 9, 1320-1331. DOI: `10.1021/acsbiomaterials.2c01143`.

Publisher PDFs are not redistributed in this repository. See `data/source/SOURCE_DATA.md` and `DATA_USE_NOTICE.md`.

## Citation

The final GitHub release and Zenodo DOI will be added after archival. Citation metadata are provided in `CITATION.cff`. Before public release, confirm the complete author list and identifiers in `RELEASE_CHECKLIST.md`.

## License and reuse

Code is released under the MIT License. Curated and processed data are provided for reproducibility subject to the source-data attribution and reuse notice in `DATA_USE_NOTICE.md`.
