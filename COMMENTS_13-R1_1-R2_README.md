# Comments 13-R1 and 1-R2 documentation update

This package documents the revisions made in response to:

- Reviewer 1, Comment 13: reproducibility statement should describe what was actually done and should state the search space and random seed.
- Reviewer 2, Comment 1: remove prescriptive/self-review language from the manuscript.

Changes in this package are documentation-only:

1. `REPRODUCIBILITY.md` now states the fixed model specifications, the algorithm-by-feature-set search space, the absence of hyperparameter tuning, and random seed 42.
2. No analytical code behavior was changed.
3. No model rerun was required for these two comments.

The reportable nested-validation code remains `src/revision_comments_4_R1_16_R2_nested_cv.py`.
