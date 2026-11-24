# Adversarial Privacy Attacks

Members: 1862 training records; non-members: 620 holdout records (real, unseen patients). Membership inference AUC of 0.5 means the synthetic data reveals nothing about who was in the training set. Attribute inference reports the MEMBERSHIP ADVANTAGE -- accuracy on members minus accuracy on non-members; population-level inference (both above baseline, equally) is the intended use of released data, only member-specific advantage is leakage.

| run | MIA AUC (95% CI) | learned MIA AUC (95% CI) | empirical ε̂ lower bound | worst AIA membership advantage | anonymeter |
|---|---|---|---|---|---|
| tvae_seed0 🚨 (worst: nearest-synthetic-distance) | 0.5431 (0.5161-0.5709) | 0.5401 (0.5135-0.5666) | 0.2363 | 0.0109 | SO 0.0, link 0.0 |

## Who is at risk: membership inference by patient atypicality

WITHIN-STRATUM AUC: members and non-members both get the same atypicality score (distance to their 5th-nearest member, same reference set and encoder), non-members are binned by the member quartile cut points, and each stratum's AUC compares members-in-Qi against non-members-in-Qi only. 0.5 = no leakage on that stratum; elevated values indicate SELECTIVE leakage on that stratum (e.g. a model that memorizes its unusual patients shows it in Q4). Strata with fewer than 30 non-members are skipped ('-'). Cell format: AUC (n members / n non-members).

| run | Q1 typical | Q2 | Q3 | Q4 atypical |
|---|---|---|---|---|
| tvae_seed0 | 0.4715 (466/147) | 0.5591 (465/149) | 0.5619 (466/146) | 0.5849 (465/178) |

## Attribute inference detail

| run | sensitive attribute | baseline | member acc | non-member acc | advantage |
|---|---|---|---|---|---|
| tvae_seed0 | cause_of_death_isAllCause_f5a_w5a_first | 0.6403 | 0.5757 | 0.5935 | -0.0178 |
| tvae_seed0 | ckd_severity_from_calculated_egfr | 0.3145 | 0.3158 | 0.3097 | +0.0061 |
| tvae_seed0 | nyha_nyha_pET | 0.3968 | 0.4076 | 0.3968 | +0.0109 |
