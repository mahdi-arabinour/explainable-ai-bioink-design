# ACS Omega Stage 14: clean-room verification result

**Status: PASS, for the scientific computational pipeline and publication SI-table normalization.**  
**Branch:** `release/acs-omega-r1-audit-20261112`  
**Tested commit:** `2551449a7a66774db5aa615c0a5241661364e9a0`  
**GitHub Actions workflow run:** https://github.com/mahdi-arabinour/explainable-ai-bioink-design/actions/runs/43330791556  
**Run completion timestamp:** 2026-11-13 00:07:21 UTC (2026-11-12 local U.S. Central time).

## Independent jobs

| Job | Result | Details |
| --- | --- | --- |
| Full master runner | **PASS** | Fresh checkout, Python 3.13.5, pinned `requirements.txt`, all stages P00-P20 completed |
| Clean master notebook | **PASS** | Separate fresh checkout; 19/19 code cells executed without error |
| Core numerical tests | **50/50 PASS** | See `results/master_run/numerical_validation.csv` in the run artifact |
| Publication table checks | **PASS** | 12 required publication output files are present; old consolidated S6-S8 and S21 files are absent |
| Stage 14 retained standard deviations | **PASS** | Pr 0.9279864185618046, SR 0.7015920267643413, Qm 0.238484137068371, AF 0.1563747426420349 |
| Master runner final status | **PASS** | `FULL RUN PASS in 1548.2s` from GitHub Actions log |

All major scientific stages passed independently: canonical source-data preparation, descriptive figures, baselines, dimensionless stress-ratio sensitivity, nested model selection/validation, R2 aggregation, matched baselines, LOFO sensitivity, XAI, ablation/representation sensitivity, reconciliation, candidate screening, retrospective bootstrap, final assets, supplementary consolidation, both XAI audits, and numerical validation. The newly introduced publication-facing SI renumbering script ran before the master status became PASS, and the separate downstream table-validation step succeeded.

The numerical verification step logged:

```
COMPLETE PASS NUMERICAL 50 / 50 SI_TABLES 12 R2_SD [
  {'Target': 'Pr', 'Nested_repeated_R2_SD': 0.9279864185618046},
  {'Target': 'SR', 'Nested_repeated_R2_SD': 0.7015920267643413},
  {'Target': 'Qm', 'Nested_repeated_R2_SD': 0.238484137068371},
  {'Target': 'AF', 'Nested_repeated_R2_SD': 0.1563747426420349}
]
```

## Input immutability

Independent comparison of the Git tree object SHA for the three canonical source CSVs and the one explicitly retained historical candidate-ranking input between `main` and this working branch showed all **four files byte-identical** at the Git blob level. No original measurements were changed.

## Archived execution evidence

- [Workflow-run page](https://github.com/mahdi-arabinour/explainable-ai-bioink-design/actions/runs/43330791556)
- [Master runner and Stage 14 validation output artifact](https://github.com/mahdi-arabinour/explainable-ai-bioink-design/actions/runs/43330791556/artifacts/13997260033)
- [Independently executed 19-cell notebook artifact](https://github.com/mahdi-arabinour/explainable-ai-bioink-design/actions/runs/43330791556/artifacts/13994523427)

GitHub artifacts have finite retention (14 days in this workflow). Source workflow and this recorded result remain in the commit history.

## Boundaries and remaining release steps

1. The optional publisher Supporting Information PDF-dependent external provenance check was not requested in this GitHub run; it remains explicitly optional and does not affect the core 50 numerical checks.
2. Success of the code-generation and SI-output checks does **not** establish a final, page-by-page audit of every Word manuscript/SI/response cross-reference after any subsequent editorial changes. Before submission, complete a focused final Word/EndNote/figure cross-reference comparison with the regenerated package, particularly the new Table S16-S28 numbering, including Table S21 embedded in Word.
3. Only a **working branch** was tested. No final pull request merge, GitHub release tag or Zenodo deposit was created.
4. The first workflow attempt failed **before computation** because Python 3.13.5 was not available on Ubuntu 26.04. The workflow was corrected to Ubuntu 24.04 and the second attempt above succeeded.

**Scientific reproducibility decision for Stage 14 code changes: PASS. Publication-release decision: conditional on final manuscript/SI synchronization and approved tagging/archiving.**
