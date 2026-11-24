# Adversarial Privacy Attacks

Members: 602 training records; non-members: 200 holdout records (real, unseen patients). Membership inference AUC of 0.5 means the synthetic data reveals nothing about who was in the training set. Attribute inference reports the MEMBERSHIP ADVANTAGE -- accuracy on members minus accuracy on non-members; population-level inference (both above baseline, equally) is the intended use of released data, only member-specific advantage is leakage.

| run | MIA AUC (95% CI) | learned MIA AUC (95% CI) | empirical ε̂ lower bound | worst AIA membership advantage | anonymeter |
|---|---|---|---|---|---|
| mst_eps15_seed0 🚨 (worst: cv-logreg over (d1, d2, d1/d2)) | 0.4859 (0.4366-0.535) | 0.536 (0.4874-0.5773) | 0.0041 | 0.0332 | SO 0.0, link 0.0051 |

## Who is at risk: membership inference by patient atypicality

WITHIN-STRATUM AUC: members and non-members both get the same atypicality score (distance to their 5th-nearest member, same reference set and encoder), non-members are binned by the member quartile cut points, and each stratum's AUC compares members-in-Qi against non-members-in-Qi only. 0.5 = no leakage on that stratum; elevated values indicate SELECTIVE leakage on that stratum (e.g. a model that memorizes its unusual patients shows it in Q4). Strata with fewer than 30 non-members are skipped ('-'). Cell format: AUC (n members / n non-members).

| run | Q1 typical | Q2 | Q3 | Q4 atypical |
|---|---|---|---|---|
| mst_eps15_seed0 | 0.4723 (151/53) | 0.4411 (150/48) | 0.5029 (151/51) | 0.5256 (150/48) |

## Attribute inference detail

| run | sensitive attribute | baseline | member acc | non-member acc | advantage |
|---|---|---|---|---|---|
| mst_eps15_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.515 | 0.5465 | 0.555 | -0.0085 |
| mst_eps15_seed0 | ckd_severity_from_calculated_egfr | 0.42 | 0.2458 | 0.295 | -0.0492 |
| mst_eps15_seed0 | nyha_nyha_pET | 0.695 | 0.5332 | 0.5 | +0.0332 |
