# Adversarial Privacy Attacks

Members: 754 training records; non-members: 251 holdout records (real, unseen patients). Membership inference AUC of 0.5 means the synthetic data reveals nothing about who was in the training set. Attribute inference reports the MEMBERSHIP ADVANTAGE -- accuracy on members minus accuracy on non-members; population-level inference (both above baseline, equally) is the intended use of released data, only member-specific advantage is leakage.

| run | MIA AUC (95% CI) | learned MIA AUC (95% CI) | empirical ε̂ lower bound | worst AIA membership advantage | anonymeter |
|---|---|---|---|---|---|
| mst_eps15_seed0 | 0.4848 (0.4437-0.5258) | 0.4656 (0.4266-0.5033) | 0.0 | 0.0142 | SO 0.0083, link 0.0 |

## Who is at risk: membership inference by patient atypicality

WITHIN-STRATUM AUC: members and non-members both get the same atypicality score (distance to their 5th-nearest member, same reference set and encoder), non-members are binned by the member quartile cut points, and each stratum's AUC compares members-in-Qi against non-members-in-Qi only. 0.5 = no leakage on that stratum; elevated values indicate SELECTIVE leakage on that stratum (e.g. a model that memorizes its unusual patients shows it in Q4). Strata with fewer than 30 non-members are skipped ('-'). Cell format: AUC (n members / n non-members).

| run | Q1 typical | Q2 | Q3 | Q4 atypical |
|---|---|---|---|---|
| mst_eps15_seed0 | 0.5629 (189/62) | 0.4531 (188/77) | 0.5677 (189/61) | 0.4827 (188/51) |

## Attribute inference detail

| run | sensitive attribute | baseline | member acc | non-member acc | advantage |
|---|---|---|---|---|---|
| mst_eps15_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.6255 | 0.5809 | 0.5737 | +0.0072 |
| mst_eps15_seed0 | ckd_severity_from_calculated_egfr | 0.2908 | 0.2812 | 0.2669 | +0.0142 |
| mst_eps15_seed0 | nyha_nyha_pET | 1.0 | 1.0 | 1.0 | +0.0000 |
