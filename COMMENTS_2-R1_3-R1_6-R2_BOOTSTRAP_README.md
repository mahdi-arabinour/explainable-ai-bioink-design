# Formulation-cluster bootstrap ranking-stability analysis

This analysis implements the reviewer-suggested uncertainty option for the originally submitted strict-candidate ordering. It is intended to test whether the original leading condition recurs under formulation-level resampling, not to provide prospective calibration.

Run:

```bash
python src/revision_comments_2_R1_6_R2_bootstrap_ranking.py
```

Primary outputs are written to `results/revision_comments_2_R1_6_R2_bootstrap_ranking/`.

Key result: 1ALG8HA / 65 kPa / 120 degrees was first in 49.1% of 1,000 formulation-cluster bootstrap replicates and in the top three in 63.3%. None of the 16 pairwise 95% bootstrap intervals for the combined-error difference versus this condition excluded zero. The result supports recurrence of the original leading hypothesis, but not a statistically unique winner.
