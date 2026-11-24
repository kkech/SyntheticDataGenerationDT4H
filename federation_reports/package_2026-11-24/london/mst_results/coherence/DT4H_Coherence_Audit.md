# Row-Coherence Audit

256 rules ({'implication': 244, 'category_range': 4, 'days_bounds': 1, 'flag_days_consistency': 7}) mined/learned from the TRAIN split and validated on real data. The holdout row is the fair baseline: real, unseen patients violating the same rules. A synthetic dataset far above it produces rows that are individually implausible patients even when every column's distribution is correct.

Two measures, answering different questions: the violation RATE is per applicable rule-check, while the row SHARE is the fraction of patients carrying at least one violation -- the one a release decision turns on, read against the real holdout's own share.

The 'consequent Missing' column is the evasion check for implication rules: the share of antecedent-true checks whose consequent was Missing and thus undecidable. A generator can push its violation rate toward zero by emitting Missing consequents; a share far above the real frames' reveals exactly that.

| frame | applicable checks | violations | violation rate | rules violated | consequent Missing | rows with >=1 violation |
|---|---|---|---|---|---|---|
| train (real) | 32954 | 0 | 0.0 | 0/256 | 59.1% | 0.0% (0/754) |
| holdout (real, unseen) | 10301 | 3 | 0.00029 | 3/256 | 61.4% | 1.2% (3/251) |
| synthetic[mst_eps15_seed0] | 43556 | 1254 | 0.02879 | 67/256 | 43.3% | 47.5% (358/754) |

## Worst rules per synthetic dataset

**synthetic[mst_eps15_seed0]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 1.0 over 14 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 1.0 over 50 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w3mo_first` (flag_days_consistency): rate 1.0 over 87 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w6mo_first` (flag_days_consistency): rate 0.70629 over 143 rows
- `med_anti_plat_history => med_platelet_history` (implication): rate 0.44565 over 92 rows

