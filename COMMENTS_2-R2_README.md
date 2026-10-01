# Reviewer 2, Comment 2: data provenance and extraction audit

This revision block documents the provenance of the experimental data used in the secondary analysis.

## Source publication

Perin, F.; Spessot, E.; Famà, A.; Bucciarelli, A.; Callone, E.; Mota, C.; Motta, A.; Maniglio, D. **Modeling a Dynamic Printability Window on Polysaccharide Blend Inks for Extrusion Bioprinting.** *ACS Biomaterials Science & Engineering* **2023**, *9*(3), 1320-1331. https://doi.org/10.1021/acsbiomaterials.2c01143.

The source publication is Open Access under the Creative Commons Attribution 4.0 International (CC BY 4.0) license.

## Extraction route used in the original analysis

The archived original Colab workflow (`notebooks/archive/original_colab_workflow_through_cell_24G.ipynb`) shows that the publisher Supporting Information PDF was read with `pdfplumber`, converted to text, and the numeric rows of Tables S1-S3 were parsed programmatically. The analysis did **not** digitize plotted figures to create the modeling tables.

The resulting curated source-derived tables are:

- `data/processed/rheology_table_S1.csv` (48 rows)
- `data/processed/printability_table_S2.csv` (43 rows)
- `data/processed/angle_fidelity_table_S3.csv` (96 rows)

## Verification audit performed for the revision

For Reviewer 2, Comment 2, the publisher Supporting Information was re-read and Tables S1-S3 were parsed using the documented numeric-table logic, then compared cell-by-cell with the repository CSVs. The audit covered 1,522 non-missing numeric entries (including run identifiers):

- Table S1: 48 rows, 432 numeric entries, 0 mismatches
- Table S2: 43 rows, 418 numeric entries, 0 mismatches
- Table S3: 96 rows, 672 numeric entries, 0 mismatches

Maximum absolute numeric difference was 0 for all three tables. Sparse Table S2 rows retained the same missing-value positions as the source table. No figure-digitization error applies because figures were not used to generate these datasets.

Audit outputs are stored in `results/revision_comment_2_R2/`.

## Files intentionally not redistributed

The publisher-formatted article and Supporting Information PDFs are not included in this repository. The audit script expects a legally obtained local copy of the Supporting Information when the source-extraction verification is repeated.
