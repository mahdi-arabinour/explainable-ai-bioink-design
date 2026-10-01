# Reviewer revision: Comments 2-R1, 3-R1, and 6-7-R2

The final candidate analysis keeps the main measured-domain set scientifically unordered because the previous fine-grained model ranking was finer than the predictive resolution and the Pr model was not reliable enough to support a narrow predicted-Pr eligibility gate. However, rather than discarding the original prioritization signal without testing it, the reviewer-suggested uncertainty option was evaluated directly.

The revised workflow therefore has two complementary parts:

1. `src/revision_comments_2_R1_6_7_R2_candidate_set.py`
   - intersects the source printability and angular-fidelity data at jointly observed formulation-pressure conditions;
   - uses measured Pr and measured AF rather than predicted values for candidate eligibility;
   - defines the main set as measured Pr in [0.95, 1.05] plus at least one measured AF in [1.00, 1.25];
   - produces 14 candidate conditions from 34 jointly observed conditions; and
   - assigns no scientific within-set rank or model-derived optimum.

2. `src/revision_comments_2_R1_6_R2_bootstrap_ranking.py`
   - re-evaluates the 17 originally submitted strict candidates with a formulation-cluster bootstrap;
   - samples the eight polymer formulations with replacement while retaining all observations belonging to each sampled formulation;
   - refits the duplicate-reduced fixed reference models used in the candidate workflow (core linear regression for Pr and angle-engineered XGBoost for AF);
   - recomputes E = |Pr_pred - 1| + |AF_pred - 1| over 1,000 bootstrap replicates; and
   - reports percentile intervals, top-rank frequencies, top-three frequencies, and pairwise bootstrap differences.

The originally highlighted 1ALG8HA / 65 kPa / 120 degree condition is the modal bootstrap leader (P[top] = 0.491) and appears in the top three in 0.633 of replicates. None of its 16 pairwise 95% bootstrap difference intervals excludes zero. The result therefore supports retaining this condition as a recurrent leading hypothesis, but not as a statistically unique winner or experimentally validated optimum.

This bootstrap is a ranking-stability sensitivity analysis, not a calibrated prospective prediction interval and not a replacement for the nested-validation performance estimates.

## Comment 7-R2 measured-Pr threshold sensitivity

The main 14-condition measured-domain set uses measured Pr 0.95-1.05 plus measured AF 1.00-1.25. A separate measured-Pr sensitivity analysis relaxes only the Pr interval to 0.90-1.10, producing 21 conditions. This confirms that predicted Pr is not used in the final gate while also documenting that candidate-set membership depends on the operational measured-Pr tolerance. See `src/revision_comment_7_R2_pr_gate_sensitivity.py`.
