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
