# ACS Omega R1: Stage 13 supplementary figure deduplication

**Scope:** Working branch only. Historical `main`, tags, prior archives and datasets are unchanged.

## Verification
The embedded PNG of manuscript Figure 4 and the embedded PNG of previous Supplementary Figure S8 have identical SHA-256:
`1b9ac1e4bb9ee8a570ab62297c2e44fd9d670d36d4fe2adf61e52adce8a55285`.
Both show the 34 jointly observed formulation-pressure conditions and 14 measured-data eligible conditions. The duplicate is removed from the **Word Supplementary Information**. The measured-domain map remains manuscript Figure 4.

The **former** supplementary Figure S9 (core-versus-engineered representation sensitivity) has been renumbered **Figure S8** in the revised SI Word file. Main-manuscript Supporting Information inventory now states Figures S1-S8. The caption and quantitative graphic are otherwise unchanged.

## Generator changes in this branch
- `src/finalize_revision_assets.py`: no longer copies the main candidate map as Supplementary Figure S8; the sensitivity plot is generated as `FigureS8_Final_Ablation.png` rather than S9.
- `run_full_analysis.py`: consolidation excludes the redundant candidate-map supplement and uses the new sensitivity-plot file as `supplementary/final_revision/figures/Figure_S8_Final_Ablation.png`.

**Important:** The repository's previously generated binary PNG artifacts from the v1.0.1 content remain until the full pipeline is rerun and the regenerated deliverables are audited. Do not describe the current GitHub binary figure directory as having been fully regenerated. Run `python run_full_analysis.py` in a fresh environment prior to approving the next publication release; this cleans and reconstructs the active `results/` and `supplementary/final_revision/` output tree, after which exact images and checksums must be checked, and only then should superseded binary files be removed or archived. Neither that end-to-end rerun nor a new Zenodo deposit was performed during Stage 13.

## Word files
- Manuscript: `Manuscript_R1_STEP13_RED.docx`, one red insertion (Figures S1-S8).
- Supplementary Information: `Supplementary_Information_R1_STEP13_RED.docx`, previously duplicated image and caption removed; former S9 becomes S8 with red renumbering.
- Response letter: `Response_to_Reviewers_R1_WORKING_STEP13.docx`, cumulative stages 1-6 and 10-13, Stage 13 statement appended.

All three working DOCX files were edited locally and are supplied to the author; no DOCX file was committed to the GitHub repository at this stage.
