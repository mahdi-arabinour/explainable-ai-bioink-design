# Pre-release checklist for GitHub and Zenodo

Complete these items before making the repository public and creating the immutable Zenodo release.

- [ ] Confirm the complete repository author/contributor list.
- [ ] Add ORCID identifiers where available.
- [ ] Confirm the copyright holder named in `LICENSE`.
- [x] Execute `python run_full_analysis.py` from a freshly extracted clean-room package.
- [x] Restart the kernel and Run All `notebooks/00_master_analysis.ipynb` from a separate fresh extraction.
- [x] Verify that key metrics match the manuscript and supplementary files.
- [x] Verify regenerated Supplementary Figures S1-S2 and manuscript Figure 3 come from `src/generate_descriptive_figures.py`.
- [x] Confirm Table S24 publisher-PDF provenance status is either PASS or explicitly documented as optional/skipped.
- [x] Confirm that no publisher PDFs, confidential files, credentials, or personal paths are present.
- [ ] Create the GitHub repository named `explainable-ai-bioink-design`.
- [ ] Upload the repository contents and verify file rendering on GitHub.
- [ ] Make the repository public only after the final audit.
- [ ] Connect the GitHub repository to Zenodo.
- [ ] Create GitHub release/tag `v1.0.1` (or the next version selected after the integrated clean-room run).
- [ ] Record the Zenodo concept DOI and version DOI.
- [ ] Add the final GitHub URL and Zenodo DOI to `README.md`, `CITATION.cff`, and the manuscript data-availability statement.

## Stage 14 supplementary table consolidation (working branch)
- [x] Removed duplicate former SI Tables S6-S8 from the working DOCX and preserved all 4 otherwise unique outer-fold R2 SD values in the publication S18 table and its new CSV.
- [x] Renumbered publication-facing SI to S1-S28; reconciled manuscript, SI register, response-letter references, repository filenames and explanatory documentation.
- [x] Added `docs/SI_TABLE_RENUMBERING_STAGE14.csv` for the full original-to-current mapping.
- [x] Updated the canonical runner to normalize published SI files after scientific validation; retained original archives/results.
- [ ] Rerun the entire analysis and all current audits in a fresh pinned environment and verify output hashes/Word references before issuing the new GitHub or Zenodo release. The current revision branch is **not yet a certified scientific release**.
