# Row-Coherence Audit

220 rules ({'implication': 208, 'category_range': 4, 'days_bounds': 1, 'flag_days_consistency': 7}) mined/learned from the TRAIN split and validated on real data. The holdout row is the fair baseline: real, unseen patients violating the same rules. A synthetic dataset far above it produces rows that are individually implausible patients even when every column's distribution is correct.

Two measures, answering different questions: the violation RATE is per applicable rule-check, while the row SHARE is the fraction of patients carrying at least one violation -- the one a release decision turns on, read against the real holdout's own share.

The 'consequent Missing' column is the evasion check for implication rules: the share of antecedent-true checks whose consequent was Missing and thus undecidable. A generator can push its violation rate toward zero by emitting Missing consequents; a share far above the real frames' reveals exactly that.

| frame | applicable checks | violations | violation rate | rules violated | consequent Missing | rows with >=1 violation |
|---|---|---|---|---|---|---|
| train (real) | 43351 | 0 | 0.0 | 0/220 | 3.4% | 0.0% (0/602) |
| holdout (real, unseen) | 15156 | 2 | 0.00013 | 2/220 | 3.8% | 1.0% (2/200) |
| synthetic[mst_eps15_seed0] | 42759 | 1643 | 0.03842 | 177/220 | 4.1% | 34.6% (208/602) |

## Worst rules per synthetic dataset

**synthetic[mst_eps15_seed0]**
- `cause_of_death_number_of_days_to_death_for_CV_f5a_first / cause_of_death_isCV_f5a_w7d_first` (flag_days_consistency): rate 1.0 over 14 rows
- `cause_of_death_number_of_days_to_death_for_CV_f5a_first / cause_of_death_isCV_f5a_w1mo_first` (flag_days_consistency): rate 1.0 over 12 rows
- `cause_of_death_number_of_days_to_death_for_CV_f5a_first / cause_of_death_isCV_f5a_w3mo_first` (flag_days_consistency): rate 1.0 over 42 rows
- `cause_of_death_number_of_days_to_death_for_CV_f5a_first / cause_of_death_isCV_f5a_w6mo_first` (flag_days_consistency): rate 0.84615 over 78 rows
- `cause_of_death_number_of_days_to_death_for_CV_f5a_first / cause_of_death_isCV_f5a_w1a_first` (flag_days_consistency): rate 0.71429 over 105 rows

