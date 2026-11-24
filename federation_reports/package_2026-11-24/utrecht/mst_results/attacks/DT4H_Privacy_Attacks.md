# Adversarial Privacy Attacks

Members: 1862 training records; non-members: 620 holdout records (real, unseen patients). Membership inference AUC of 0.5 means the synthetic data reveals nothing about who was in the training set. Attribute inference reports the MEMBERSHIP ADVANTAGE -- accuracy on members minus accuracy on non-members; population-level inference (both above baseline, equally) is the intended use of released data, only member-specific advantage is leakage.

| run | MIA AUC (95% CI) | learned MIA AUC (95% CI) | empirical ε̂ lower bound | worst AIA membership advantage | anonymeter |
|---|---|---|---|---|---|
| mst_eps15_seed0 | 0.5218 (0.4929-0.5485) | 0.5027 (0.4756-0.5296) | 0.0 | 0.0158 | SO 0.01, link 0.005 |

## Who is at risk: membership inference by patient atypicality

WITHIN-STRATUM AUC: members and non-members both get the same atypicality score (distance to their 5th-nearest member, same reference set and encoder), non-members are binned by the member quartile cut points, and each stratum's AUC compares members-in-Qi against non-members-in-Qi only. 0.5 = no leakage on that stratum; elevated values indicate SELECTIVE leakage on that stratum (e.g. a model that memorizes its unusual patients shows it in Q4). Strata with fewer than 30 non-members are skipped ('-'). Cell format: AUC (n members / n non-members).

| run | Q1 typical | Q2 | Q3 | Q4 atypical |
|---|---|---|---|---|
| mst_eps15_seed0 | 0.5073 (466/147) | 0.4927 (465/149) | 0.5242 (466/146) | 0.5238 (465/178) |

## Attribute inference detail

| run | sensitive attribute | baseline | member acc | non-member acc | advantage |
|---|---|---|---|---|---|
| mst_eps15_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.6403 | 0.4984 | 0.4919 | +0.0065 |
| mst_eps15_seed0 | ckd_severity_from_calculated_egfr | 0.3145 | 0.319 | 0.3032 | +0.0158 |
| mst_eps15_seed0 | nyha_nyha_pET | 0.3968 | 0.3255 | 0.3145 | +0.0109 |
