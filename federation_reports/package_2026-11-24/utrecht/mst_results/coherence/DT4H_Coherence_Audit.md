# Row-Coherence Audit

466 rules ({'implication': 454, 'category_range': 4, 'days_bounds': 1, 'flag_days_consistency': 7}) mined/learned from the TRAIN split and validated on real data. The holdout row is the fair baseline: real, unseen patients violating the same rules. A synthetic dataset far above it produces rows that are individually implausible patients even when every column's distribution is correct.

Two measures, answering different questions: the violation RATE is per applicable rule-check, while the row SHARE is the fraction of patients carrying at least one violation -- the one a release decision turns on, read against the real holdout's own share.

The 'consequent Missing' column is the evasion check for implication rules: the share of antecedent-true checks whose consequent was Missing and thus undecidable. A generator can push its violation rate toward zero by emitting Missing consequents; a share far above the real frames' reveals exactly that.

| frame | applicable checks | violations | violation rate | rules violated | consequent Missing | rows with >=1 violation |
|---|---|---|---|---|---|---|
| train (real) | 90298 | 0 | 0.0 | 0/466 | 63.9% | 0.0% (0/1862) |
| holdout (real, unseen) | 31817 | 5 | 0.00016 | 2/466 | 63.0% | 0.8% (5/620) |
| synthetic[mst_eps15_seed0] | 107920 | 2719 | 0.02519 | 153/466 | 55.5% | 26.3% (490/1862) |

## Worst rules per synthetic dataset

**synthetic[mst_eps15_seed0]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 1.0 over 61 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 1.0 over 153 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w3mo_first` (flag_days_consistency): rate 1.0 over 239 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w3mo_first => encounter_primary_reason_non_CV_Disease_f5a_w3a_first` (implication): rate 0.64394 over 132 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w3mo_first => encounter_primary_reason_non_CV_Disease_f5a_w5a_first` (implication): rate 0.57895 over 133 rows

