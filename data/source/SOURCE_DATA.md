# Source data and provenance

The experimental data analyzed in this repository were reported in:

Perin, F.; Spessot, E.; Famà, A.; Bucciarelli, A.; Callone, E.; Mota, C.; Motta, A.; Maniglio, D. **Modeling a Dynamic Printability Window on Polysaccharide Blend Inks for Extrusion Bioprinting.** *ACS Biomaterials Science & Engineering* 2023, 9(3), 1320-1331. DOI: 10.1021/acsbiomaterials.2c01143.

## Extraction and verification

The archived original Colab workflow documents that the publisher Supporting Information PDF was read with `pdfplumber`, converted to text, and Tables S1-S3 were parsed programmatically. Values used in the modeling datasets were **not digitized from plotted figures**.

A revision audit for Reviewer 2, Comment 2 re-parsed the source Tables S1-S3 and compared them cell-by-cell with the curated CSVs. All 48 rheology rows, 43 printability rows, and 96 angular-fidelity rows matched, covering 1,522 non-missing numeric entries with zero detected mismatches and maximum absolute numeric difference 0. The audit outputs are stored in `results/revision_comment_2_R2/`.

The source publication is Open Access under the Creative Commons Attribution 4.0 International (CC BY 4.0) license.

## Included source-derived tables

The repository begins from three curated tables derived from the published supplementary information:

- `data/processed/rheology_table_S1.csv`
- `data/processed/printability_table_S2.csv`
- `data/processed/angle_fidelity_table_S3.csv`

No new experimental measurements were generated in the present study.

## Files intentionally not redistributed

The publisher article PDF and supplementary-information PDF are not included. Users who need to audit the original extraction should obtain the source publication through the publisher or an authorized institutional repository, then consult the archived provenance notebook in `notebooks/archive/`.
