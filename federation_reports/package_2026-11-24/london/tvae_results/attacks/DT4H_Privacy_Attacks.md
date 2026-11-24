# Adversarial Privacy Attacks

Members: 754 training records; non-members: 251 holdout records (real, unseen patients). Membership inference AUC of 0.5 means the synthetic data reveals nothing about who was in the training set. Attribute inference reports the MEMBERSHIP ADVANTAGE -- accuracy on members minus accuracy on non-members; population-level inference (both above baseline, equally) is the intended use of released data, only member-specific advantage is leakage.

| run | MIA AUC (95% CI) | learned MIA AUC (95% CI) | empirical ε̂ lower bound | worst AIA membership advantage | anonymeter |
|---|---|---|---|---|---|
| tvae_seed0 🚨 (worst: cv-logreg over (d1, d2, d1/d2)) | 0.5141 (0.4761-0.5533) | 0.5204 (0.483-0.5585) | 0.0 | 0.0 | SO 0.0083, link 0.0 |

## Who is at risk: membership inference by patient atypicality

WITHIN-STRATUM AUC: members and non-members both get the same atypicality score (distance to their 5th-nearest member, same reference set and encoder), non-members are binned by the member quartile cut points, and each stratum's AUC compares members-in-Qi against non-members-in-Qi only. 0.5 = no leakage on that stratum; elevated values indicate SELECTIVE leakage on that stratum (e.g. a model that memorizes its unusual patients shows it in Q4). Strata with fewer than 30 non-members are skipped ('-'). Cell format: AUC (n members / n non-members).

| run | Q1 typical | Q2 | Q3 | Q4 atypical |
|---|---|---|---|---|
| tvae_seed0 🚨 | 0.5735 (189/62) | 0.5789 (188/77) | 0.5577 (189/61) | 0.6223 (188/51) |

## Attribute inference detail

| run | sensitive attribute | baseline | member acc | non-member acc | advantage |
|---|---|---|---|---|---|
| tvae_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.6255 | 0.5517 | 0.5737 | -0.0220 |
| tvae_seed0 | ckd_severity_from_calculated_egfr | 0.2908 | 0.2692 | 0.2948 | -0.0256 |
| tvae_seed0 | nyha_nyha_pET | 1.0 | 1.0 | 1.0 | +0.0000 |
