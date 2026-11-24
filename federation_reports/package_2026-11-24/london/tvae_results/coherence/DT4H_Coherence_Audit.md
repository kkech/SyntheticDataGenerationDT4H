# Row-Coherence Audit

256 rules ({'implication': 244, 'category_range': 4, 'days_bounds': 1, 'flag_days_consistency': 7}) mined/learned from the TRAIN split and validated on real data. The holdout row is the fair baseline: real, unseen patients violating the same rules. A synthetic dataset far above it produces rows that are individually implausible patients even when every column's distribution is correct.

Two measures, answering different questions: the violation RATE is per applicable rule-check, while the row SHARE is the fraction of patients carrying at least one violation -- the one a release decision turns on, read against the real holdout's own share.

The 'consequent Missing' column is the evasion check for implication rules: the share of antecedent-true checks whose consequent was Missing and thus undecidable. A generator can push its violation rate toward zero by emitting Missing consequents; a share far above the real frames' reveals exactly that.

| frame | applicable checks | violations | violation rate | rules violated | consequent Missing | rows with >=1 violation |
|---|---|---|---|---|---|---|
| train (real) | 32954 | 0 | 0.0 | 0/256 | 59.1% | 0.0% (0/754) |
| holdout (real, unseen) | 10301 | 3 | 0.00029 | 3/256 | 61.4% | 1.2% (3/251) |
| synthetic[tvae_seed0] | 34974 | 219 | 0.00626 | 31/256 | 56.7% | 19.4% (146/754) |

## Worst rules per synthetic dataset

**synthetic[tvae_seed0]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 1.0 over 1 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 0.6 over 25 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w3mo_first` (flag_days_consistency): rate 0.28571 over 84 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1a_first` (flag_days_consistency): rate 0.14054 over 370 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w6mo_first` (flag_days_consistency): rate 0.10417 over 192 rows

