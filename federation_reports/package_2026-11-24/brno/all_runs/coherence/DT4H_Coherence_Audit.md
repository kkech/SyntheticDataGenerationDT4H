# Row-Coherence Audit

511 rules ({'implication': 499, 'category_range': 4, 'days_bounds': 1, 'flag_days_consistency': 7}) mined/learned from the TRAIN split and validated on real data. The holdout row is the fair baseline: real, unseen patients violating the same rules. A synthetic dataset far above it produces rows that are individually implausible patients even when every column's distribution is correct.

Two measures, answering different questions: the violation RATE is per applicable rule-check, while the row SHARE is the fraction of patients carrying at least one violation -- the one a release decision turns on, read against the real holdout's own share.

The 'consequent Missing' column is the evasion check for implication rules: the share of antecedent-true checks whose consequent was Missing and thus undecidable. A generator can push its violation rate toward zero by emitting Missing consequents; a share far above the real frames' reveals exactly that.

| frame | applicable checks | violations | violation rate | rules violated | consequent Missing | rows with >=1 violation |
|---|---|---|---|---|---|---|
| train (real) | 113225 | 0 | 0.0 | 0/511 | 61.5% | 0.0% (0/1549) |
| holdout (real, unseen) | 34374 | 5 | 0.00015 | 3/511 | 64.7% | 0.8% (4/516) |
| synthetic[aim40_eps1_seed0] | 4045 | 179 | 0.04425 | 4/511 | 0.0% | 7.6% (117/1549) |
| synthetic[aim40_eps5_seed0] | 4007 | 48 | 0.01198 | 3/511 | 0.0% | 2.4% (37/1549) |
| synthetic[aim40_eps8_seed0] | 3869 | 43 | 0.01111 | 3/511 | 0.0% | 2.0% (31/1549) |
| synthetic[aim50_eps1_seed0] | 5908 | 539 | 0.09123 | 7/511 | 0.0% | 29.8% (462/1549) |
| synthetic[aim50_eps5_seed0] | 5856 | 138 | 0.02357 | 5/511 | 0.0% | 7.2% (112/1549) |
| synthetic[aim50_eps8_seed0] | 5821 | 52 | 0.00893 | 7/511 | 0.0% | 2.8% (43/1549) |
| synthetic[ctgan_qt_seed0] | 92224 | 12651 | 0.13718 | 117/511 | 66.9% | 99.7% (1545/1549) |
| synthetic[ctgan_seed0] | 91393 | 13889 | 0.15197 | 117/511 | 68.9% | 99.9% (1548/1549) |
| synthetic[ctgan_seed1] | 89518 | 12357 | 0.13804 | 117/511 | 66.5% | 99.7% (1544/1549) |
| synthetic[ctgan_seed2] | 104366 | 12064 | 0.11559 | 117/511 | 62.4% | 99.9% (1548/1549) |
| synthetic[ddpm_g_seed0] | 171877 | 6435 | 0.03744 | 11/511 | 42.7% | 96.3% (1492/1549) |
| synthetic[ddpm_seed0] | 171428 | 12686 | 0.074 | 117/511 | 50.7% | 99.0% (1534/1549) |
| synthetic[ddpm_seed1] | 177410 | 12115 | 0.06829 | 117/511 | 49.3% | 98.8% (1530/1549) |
| synthetic[ddpm_seed2] | 176939 | 12319 | 0.06962 | 117/511 | 49.6% | 98.5% (1525/1549) |
| synthetic[dpctgan_eps10_seed0] | 46241 | 14281 | 0.30884 | 45/511 | 82.0% | 100.0% (1549/1549) |
| synthetic[dpctgan_eps15_seed0] | 65328 | 13535 | 0.20719 | 43/511 | 76.1% | 100.0% (1549/1549) |
| synthetic[dpctgan_eps15_seed1] | 114218 | 7146 | 0.06256 | 30/511 | 54.5% | 100.0% (1549/1549) |
| synthetic[dpctgan_eps15_seed2] | 32564 | 19485 | 0.59836 | 51/511 | 89.0% | 100.0% (1549/1549) |
| synthetic[dpctgan_eps1_seed0] | 85409 | 19677 | 0.23039 | 63/511 | 72.7% | 100.0% (1549/1549) |
| synthetic[dpctgan_eps20_seed0] | 94833 | 12379 | 0.13053 | 38/511 | 69.8% | 100.0% (1549/1549) |
| synthetic[dpctgan_eps5_seed0] | 162395 | 12429 | 0.07654 | 44/511 | 41.8% | 100.0% (1549/1549) |
| synthetic[dpctgan_eps8_seed0] | 197758 | 19521 | 0.09871 | 73/511 | 47.7% | 100.0% (1549/1549) |
| synthetic[gaussian_copula_seed0] | 98846 | 7051 | 0.07133 | 117/511 | 66.1% | 95.4% (1477/1549) |
| synthetic[gaussian_copula_seed1] | 96702 | 6944 | 0.07181 | 117/511 | 66.8% | 95.7% (1482/1549) |
| synthetic[gaussian_copula_seed2] | 102935 | 7036 | 0.06835 | 117/511 | 64.9% | 95.2% (1475/1549) |
| synthetic[mst_eps10_seed0] | 129206 | 2766 | 0.02141 | 95/511 | 55.0% | 39.8% (617/1549) |
| synthetic[mst_eps15_seed0] | 126664 | 1539 | 0.01215 | 86/511 | 56.1% | 27.2% (422/1549) |
| synthetic[mst_eps15_seed1] | 127754 | 1936 | 0.01515 | 106/511 | 55.7% | 28.5% (442/1549) |
| synthetic[mst_eps15_seed2] | 123596 | 1656 | 0.0134 | 96/511 | 57.3% | 36.7% (569/1549) |
| synthetic[mst_eps1_seed0] | 132858 | 9133 | 0.06874 | 105/511 | 53.1% | 87.0% (1348/1549) |
| synthetic[mst_eps20_seed0] | 131034 | 1958 | 0.01494 | 104/511 | 54.5% | 27.6% (427/1549) |
| synthetic[mst_eps5_seed0] | 131446 | 2679 | 0.02038 | 96/511 | 54.5% | 50.4% (780/1549) |
| synthetic[mst_eps8_seed0] | 134318 | 2290 | 0.01705 | 103/511 | 53.2% | 45.2% (700/1549) |
| synthetic[patectgan_eps15_seed0] | 68816 | 2623 | 0.03812 | 106/511 | 75.9% | 66.9% (1036/1549) |
| synthetic[patectgan_eps1_seed0] | 98713 | 14878 | 0.15072 | 117/511 | 66.8% | 99.9% (1548/1549) |
| synthetic[patectgan_eps5_seed0] | 53192 | 3654 | 0.06869 | 106/511 | 83.1% | 83.9% (1299/1549) |
| synthetic[tvae_cap256_seed0] | 110276 | 1365 | 0.01238 | 105/511 | 61.4% | 30.7% (476/1549) |
| synthetic[tvae_ep1000_seed0] | 108687 | 1587 | 0.0146 | 109/511 | 62.8% | 34.7% (537/1549) |
| synthetic[tvae_ind_seed0] | 106957 | 817 | 0.00764 | 100/511 | 63.5% | 24.7% (383/1549) |
| synthetic[tvae_qt_seed0] | 112362 | 1294 | 0.01152 | 99/511 | 61.4% | 34.9% (540/1549) |
| synthetic[tvae_qt_seed1] | 100009 | 1112 | 0.01112 | 94/511 | 65.3% | 29.9% (463/1549) |
| synthetic[tvae_qt_seed2] | 108978 | 988 | 0.00907 | 103/511 | 62.2% | 28.5% (441/1549) |
| synthetic[tvae_seed0] | 108826 | 1144 | 0.01051 | 100/511 | 62.4% | 28.7% (445/1549) |
| synthetic[tvae_seed1] | 104750 | 1136 | 0.01084 | 95/511 | 63.5% | 28.9% (447/1549) |
| synthetic[tvae_seed2] | 111177 | 1220 | 0.01097 | 94/511 | 61.8% | 31.9% (494/1549) |

## Worst rules per synthetic dataset

**synthetic[aim40_eps1_seed0]**
- `ckd_severity_from_calculated_egfr vs lab_results_valideGFR_value_last` (category_range): rate 0.0611 over 1473 rows
- `ckd_severity_from_calculated_egfr vs lab_results_valideGFR_value_first` (category_range): rate 0.04739 over 1477 rows
- `med_acei => med_rasi` (implication): rate 0.01887 over 795 rows
- `med_sglt2i => med_oral_antidiabetic` (implication): rate 0.01333 over 300 rows

**synthetic[aim40_eps5_seed0]**
- `ckd_severity_from_calculated_egfr vs lab_results_valideGFR_value_last` (category_range): rate 0.01606 over 1494 rows
- `ckd_severity_from_calculated_egfr vs lab_results_valideGFR_value_first` (category_range): rate 0.01472 over 1495 rows
- `med_sglt2i => med_oral_antidiabetic` (implication): rate 0.00725 over 276 rows

**synthetic[aim40_eps8_seed0]**
- `med_sglt2i => med_oral_antidiabetic` (implication): rate 0.02907 over 172 rows
- `ckd_severity_from_calculated_egfr vs lab_results_valideGFR_value_first` (category_range): rate 0.01484 over 1482 rows
- `ckd_severity_from_calculated_egfr vs lab_results_valideGFR_value_last` (category_range): rate 0.01077 over 1486 rows

**synthetic[aim50_eps1_seed0]**
- `med_mra_history => med_diuretics_history` (implication): rate 0.46809 over 423 rows
- `med_diuretics_loop_history => med_diuretics_history` (implication): rate 0.29263 over 475 rows
- `ckd_severity_from_calculated_egfr vs lab_results_valideGFR_value_first` (category_range): rate 0.06582 over 1489 rows
- `ckd_severity_from_calculated_egfr vs lab_results_valideGFR_value_last` (category_range): rate 0.04993 over 1462 rows
- `med_sglt2i => med_oral_antidiabetic` (implication): rate 0.03548 over 310 rows

**synthetic[aim50_eps5_seed0]**
- `ckd_severity_from_calculated_egfr vs lab_results_valideGFR_value_last` (category_range): rate 0.04756 over 1493 rows
- `med_acei => med_rasi` (implication): rate 0.03252 over 738 rows
- `ckd_severity_from_calculated_egfr vs lab_results_valideGFR_value_first` (category_range): rate 0.0259 over 1467 rows
- `med_sglt2i => med_oral_antidiabetic` (implication): rate 0.00735 over 272 rows
- `med_diuretics_loop_history => med_diuretics_history` (implication): rate 0.00606 over 495 rows

**synthetic[aim50_eps8_seed0]**
- `ckd_severity_from_calculated_egfr vs lab_results_valideGFR_value_first` (category_range): rate 0.01549 over 1485 rows
- `med_acei => med_rasi` (implication): rate 0.01007 over 695 rows
- `ckd_severity_from_calculated_egfr vs lab_results_valideGFR_value_last` (category_range): rate 0.00939 over 1491 rows
- `med_diuretics_loop_history => med_diuretics_history` (implication): rate 0.00833 over 480 rows
- `med_sglt2i => med_oral_antidiabetic` (implication): rate 0.00361 over 277 rows

**synthetic[ctgan_qt_seed0]**
- `med_anti_plat => med_platelet` (implication): rate 0.91795 over 195 rows
- `med_platelet => med_anti_plat` (implication): rate 0.90909 over 176 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 0.8895 over 181 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 0.87755 over 98 rows
- `med_platelet_history => med_anti_plat_history` (implication): rate 0.87432 over 366 rows

**synthetic[ctgan_seed0]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 0.97183 over 142 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 0.9351 over 339 rows
- `med_anti_plat_history => med_platelet_history` (implication): rate 0.91275 over 149 rows
- `med_arni => med_arb` (implication): rate 0.89744 over 117 rows
- `med_platelet_history => med_anti_plat_history` (implication): rate 0.89516 over 124 rows

**synthetic[ctgan_seed1]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 1.0 over 91 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 0.95327 over 214 rows
- `med_platelet => med_anti_plat` (implication): rate 0.90698 over 215 rows
- `med_anti_plat_history => med_platelet_history` (implication): rate 0.89437 over 142 rows
- `med_platelet_history => med_anti_plat_history` (implication): rate 0.89437 over 142 rows

**synthetic[ctgan_seed2]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 1.0 over 82 rows
- `med_sglt2i_history => med_oral_antidiabetic_history` (implication): rate 0.93182 over 44 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 0.93035 over 402 rows
- `med_platelet_history => med_anti_plat_history` (implication): rate 0.90123 over 162 rows
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => encounter_primary_reason_CV_Disease_f5a_w1mo_first` (implication): rate 0.83333 over 18 rows

**synthetic[ddpm_g_seed0]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 0.94426 over 897 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 0.94403 over 536 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w3mo_first` (flag_days_consistency): rate 0.90612 over 703 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w6mo_first` (flag_days_consistency): rate 0.88287 over 1127 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1a_first` (flag_days_consistency): rate 0.8651 over 934 rows

**synthetic[ddpm_seed0]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 0.96503 over 286 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 0.94937 over 395 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w3mo_first` (flag_days_consistency): rate 0.93128 over 553 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w6mo_first` (flag_days_consistency): rate 0.91437 over 654 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1a_first` (flag_days_consistency): rate 0.87593 over 806 rows

**synthetic[ddpm_seed1]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 0.95349 over 301 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 0.92809 over 445 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w3mo_first` (flag_days_consistency): rate 0.90865 over 613 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w6mo_first` (flag_days_consistency): rate 0.86099 over 705 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1a_first` (flag_days_consistency): rate 0.82578 over 861 rows

**synthetic[ddpm_seed2]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 0.95745 over 423 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 0.95679 over 324 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w3mo_first` (flag_days_consistency): rate 0.91367 over 556 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w6mo_first` (flag_days_consistency): rate 0.87463 over 670 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1a_first` (flag_days_consistency): rate 0.84485 over 825 rows

**synthetic[dpctgan_eps10_seed0]**
- `encounter_primary_reason_CV_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w3mo_first` (implication): rate 1.0 over 6 rows
- `encounter_primary_reason_CV_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w3a_first` (implication): rate 1.0 over 6 rows
- `encounter_primary_reason_CV_Disease_f5a_w3mo_first => encounter_primary_reason_CV_Disease_f5a_w3a_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_CV_Disease_f5a_w6mo_first => encounter_primary_reason_CV_Disease_f5a_w3mo_first` (implication): rate 1.0 over 2 rows
- `encounter_primary_reason_CV_Disease_f5a_w6mo_first => encounter_primary_reason_CV_Disease_f5a_w3a_first` (implication): rate 1.0 over 2 rows

**synthetic[dpctgan_eps15_seed0]**
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => encounter_primary_reason_CV_Disease_f5a_w3mo_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => encounter_primary_reason_CV_Disease_f5a_w1a_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => encounter_primary_reason_CV_Disease_f5a_w3a_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_CV_Disease_f5a_w1a_first => encounter_primary_reason_CV_Disease_f5a_w3mo_first` (implication): rate 1.0 over 86 rows
- `encounter_primary_reason_CV_Disease_f5a_w3a_first => encounter_primary_reason_CV_Disease_f5a_w3mo_first` (implication): rate 1.0 over 7 rows

**synthetic[dpctgan_eps15_seed1]**
- `smoking_status_smoker_last => smoking_status_formerSmoker_last` (implication): rate 1.0 over 3 rows
- `encounter_primary_reason_CV_Disease_f5a_w1a_first => encounter_primary_reason_CV_Disease_f5a_w3mo_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_CV_Disease_f5a_w3a_first => encounter_primary_reason_CV_Disease_f5a_w6mo_first` (implication): rate 1.0 over 6 rows
- `encounter_primary_reason_CV_Disease_f5a_w5a_first => encounter_primary_reason_CV_Disease_f5a_w6mo_first` (implication): rate 1.0 over 8 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w1mo_first => encounter_primary_reason_non_CV_Disease_f5a_w7d_first` (implication): rate 1.0 over 16 rows

**synthetic[dpctgan_eps15_seed2]**
- `encounter_primary_reason_CV_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w6mo_first` (implication): rate 1.0 over 7 rows
- `encounter_primary_reason_CV_Disease_f5a_w3mo_first => encounter_primary_reason_CV_Disease_f5a_w6mo_first` (implication): rate 1.0 over 4 rows
- `encounter_primary_reason_CV_Disease_f5a_w3mo_first => encounter_primary_reason_CV_Disease_f5a_w3a_first` (implication): rate 1.0 over 3 rows
- `encounter_primary_reason_CV_Disease_f5a_w6mo_first => encounter_primary_reason_CV_Disease_f5a_w3a_first` (implication): rate 1.0 over 3 rows
- `encounter_primary_reason_CV_Disease_f5a_w3a_first => encounter_primary_reason_CV_Disease_f5a_w6mo_first` (implication): rate 1.0 over 61 rows

**synthetic[dpctgan_eps1_seed0]**
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => encounter_primary_reason_CV_Disease_f5a_w3mo_first` (implication): rate 1.0 over 4 rows
- `encounter_primary_reason_CV_Disease_f5a_w3mo_first => encounter_primary_reason_CV_Disease_f5a_w1mo_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_CV_Disease_f5a_w6mo_first => encounter_primary_reason_CV_Disease_f5a_w1mo_first` (implication): rate 1.0 over 32 rows
- `encounter_primary_reason_CV_Disease_f5a_w1a_first => encounter_primary_reason_CV_Disease_f5a_w1mo_first` (implication): rate 1.0 over 32 rows
- `encounter_primary_reason_CV_Disease_f5a_w3a_first => encounter_primary_reason_CV_Disease_f5a_w1mo_first` (implication): rate 1.0 over 32 rows

**synthetic[dpctgan_eps20_seed0]**
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => encounter_primary_reason_CV_Disease_f5a_w6mo_first` (implication): rate 1.0 over 4 rows
- `encounter_primary_reason_CV_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w6mo_first` (implication): rate 1.0 over 14 rows
- `encounter_primary_reason_CV_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w5a_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_CV_Disease_f5a_w3mo_first => encounter_primary_reason_CV_Disease_f5a_w1a_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_CV_Disease_f5a_w3a_first => encounter_primary_reason_CV_Disease_f5a_w1a_first` (implication): rate 1.0 over 1 rows

**synthetic[dpctgan_eps5_seed0]**
- `encounter_primary_reason_CV_Disease_f5a_w1a_first => encounter_primary_reason_CV_Disease_f5a_w1mo_first` (implication): rate 1.0 over 3 rows
- `encounter_primary_reason_CV_Disease_f5a_w1a_first => encounter_primary_reason_CV_Disease_f5a_w3mo_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_CV_Disease_f5a_w1a_first => encounter_primary_reason_CV_Disease_f5a_w5a_first` (implication): rate 1.0 over 10 rows
- `encounter_primary_reason_CV_Disease_f5a_w3a_first => encounter_primary_reason_CV_Disease_f5a_w1mo_first` (implication): rate 1.0 over 3 rows
- `encounter_primary_reason_CV_Disease_f5a_w3a_first => encounter_primary_reason_CV_Disease_f5a_w3mo_first` (implication): rate 1.0 over 2 rows

**synthetic[dpctgan_eps8_seed0]**
- `encounter_primary_reason_CV_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w3mo_first` (implication): rate 1.0 over 2 rows
- `encounter_primary_reason_CV_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w1a_first` (implication): rate 1.0 over 2 rows
- `encounter_primary_reason_CV_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w5a_first` (implication): rate 1.0 over 2 rows
- `encounter_primary_reason_CV_Disease_f5a_w3mo_first => encounter_primary_reason_CV_Disease_f5a_w1a_first` (implication): rate 1.0 over 11 rows
- `encounter_primary_reason_CV_Disease_f5a_w3mo_first => encounter_primary_reason_CV_Disease_f5a_w5a_first` (implication): rate 1.0 over 11 rows

**synthetic[gaussian_copula_seed0]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 1.0 over 152 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 0.95671 over 231 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w3mo_first` (flag_days_consistency): rate 0.91111 over 315 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w6mo_first` (flag_days_consistency): rate 0.82005 over 389 rows
- `med_sglt2i_history => med_oral_antidiabetic_history` (implication): rate 0.71739 over 92 rows

**synthetic[gaussian_copula_seed1]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 1.0 over 147 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 0.96 over 200 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w3mo_first` (flag_days_consistency): rate 0.92857 over 280 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w6mo_first` (flag_days_consistency): rate 0.88747 over 391 rows
- `med_arni => med_arb` (implication): rate 0.75758 over 66 rows

**synthetic[gaussian_copula_seed2]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 0.98726 over 157 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 0.97403 over 231 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w3mo_first` (flag_days_consistency): rate 0.94562 over 331 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w6mo_first` (flag_days_consistency): rate 0.82927 over 410 rows
- `med_sglt2i_history => med_oral_antidiabetic_history` (implication): rate 0.73563 over 87 rows

**synthetic[mst_eps10_seed0]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 1.0 over 144 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 1.0 over 222 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w3mo_first` (flag_days_consistency): rate 1.0 over 298 rows
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => encounter_primary_reason_CV_Disease_f5a_w5a_first` (implication): rate 0.9902 over 102 rows
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => encounter_primary_reason_CV_Disease_f5a_w1a_first` (implication): rate 0.98039 over 102 rows

**synthetic[mst_eps15_seed0]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 1.0 over 148 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 1.0 over 219 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w3mo_first` (flag_days_consistency): rate 1.0 over 293 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w5a_first => encounter_primary_reason_non_CV_Disease_f5a_w7d_first` (implication): rate 0.22936 over 109 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w3a_first => encounter_primary_reason_non_CV_Disease_f5a_w7d_first` (implication): rate 0.2243 over 107 rows

**synthetic[mst_eps15_seed1]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 1.0 over 154 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 1.0 over 228 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w3mo_first` (flag_days_consistency): rate 1.0 over 296 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w7d_first => encounter_primary_reason_non_CV_Disease_f5a_w5a_first` (implication): rate 0.36709 over 79 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w7d_first => encounter_primary_reason_non_CV_Disease_f5a_w1a_first` (implication): rate 0.30667 over 75 rows

**synthetic[mst_eps15_seed2]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 1.0 over 156 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 1.0 over 224 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w3mo_first` (flag_days_consistency): rate 1.0 over 300 rows
- `med_arb => med_rasi` (implication): rate 0.70432 over 301 rows
- `med_arni => med_rasi` (implication): rate 0.49315 over 73 rows

**synthetic[mst_eps1_seed0]**
- `encounter_primary_reason_non_CV_Disease_f5a_w7d_first => encounter_primary_reason_non_CV_Disease_f5a_w6mo_first` (implication): rate 1.0 over 11 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w6mo_first => encounter_primary_reason_non_CV_Disease_f5a_w7d_first` (implication): rate 1.0 over 57 rows
- `med_insulins_history => conditions_diabetes` (implication): rate 1.0 over 17 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 1.0 over 114 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 1.0 over 243 rows

**synthetic[mst_eps20_seed0]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 1.0 over 155 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 1.0 over 219 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w3mo_first` (flag_days_consistency): rate 1.0 over 295 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w5a_first => encounter_primary_reason_non_CV_Disease_f5a_w7d_first` (implication): rate 0.5 over 86 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w7d_first => encounter_primary_reason_non_CV_Disease_f5a_w5a_first` (implication): rate 0.46914 over 81 rows

**synthetic[mst_eps5_seed0]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 1.0 over 143 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 1.0 over 221 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w3mo_first` (flag_days_consistency): rate 1.0 over 313 rows
- `med_arni => med_arb` (implication): rate 0.92727 over 55 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w7d_first => encounter_primary_reason_non_CV_Disease_f5a_w3mo_first` (implication): rate 0.59524 over 84 rows

**synthetic[mst_eps8_seed0]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 1.0 over 159 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 1.0 over 233 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w3mo_first` (flag_days_consistency): rate 1.0 over 299 rows
- `med_arb => med_rasi` (implication): rate 0.90203 over 296 rows
- `med_arni => med_arb` (implication): rate 0.73418 over 79 rows

**synthetic[patectgan_eps15_seed0]**
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => encounter_primary_reason_CV_Disease_f5a_w1mo_first` (implication): rate 1.0 over 2 rows
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => encounter_primary_reason_CV_Disease_f5a_w3mo_first` (implication): rate 1.0 over 2 rows
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => encounter_primary_reason_CV_Disease_f5a_w1a_first` (implication): rate 1.0 over 3 rows
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => encounter_primary_reason_CV_Disease_f5a_w3a_first` (implication): rate 1.0 over 3 rows
- `encounter_primary_reason_CV_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w6mo_first` (implication): rate 1.0 over 3 rows

**synthetic[patectgan_eps1_seed0]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 1.0 over 158 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 1.0 over 159 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1a_first` (flag_days_consistency): rate 0.94595 over 444 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w3mo_first` (flag_days_consistency): rate 0.93725 over 255 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w5a_first` (flag_days_consistency): rate 0.93151 over 730 rows

**synthetic[patectgan_eps5_seed0]**
- `encounter_primary_reason_CV_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w7d_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_CV_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w3mo_first` (implication): rate 1.0 over 2 rows
- `encounter_primary_reason_CV_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w6mo_first` (implication): rate 1.0 over 2 rows
- `encounter_primary_reason_CV_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w1a_first` (implication): rate 1.0 over 2 rows
- `encounter_primary_reason_CV_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w3a_first` (implication): rate 1.0 over 2 rows

**synthetic[tvae_cap256_seed0]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 0.81197 over 117 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 0.43939 over 264 rows
- `med_arni => med_arb` (implication): rate 0.33333 over 12 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w6mo_first` (flag_days_consistency): rate 0.21317 over 638 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w3mo_first` (flag_days_consistency): rate 0.18615 over 462 rows

**synthetic[tvae_ep1000_seed0]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 0.91837 over 98 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 0.53465 over 202 rows
- `med_arni => med_arb` (implication): rate 0.36842 over 19 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w3mo_first` (flag_days_consistency): rate 0.20782 over 409 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w6mo_first` (flag_days_consistency): rate 0.19598 over 597 rows

**synthetic[tvae_ind_seed0]**
- `med_arni => med_arb` (implication): rate 0.8 over 5 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 0.38144 over 97 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 0.24051 over 237 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1a_first` (flag_days_consistency): rate 0.10142 over 848 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w3a_first` (flag_days_consistency): rate 0.08278 over 1208 rows

**synthetic[tvae_qt_seed0]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 0.37069 over 232 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 0.36449 over 107 rows
- `med_arni => med_arb` (implication): rate 0.33333 over 12 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w3mo_first` (flag_days_consistency): rate 0.27619 over 420 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w6mo_first` (flag_days_consistency): rate 0.21192 over 604 rows

**synthetic[tvae_qt_seed1]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 0.77778 over 45 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 0.55128 over 156 rows
- `med_arni => med_arb` (implication): rate 0.5 over 2 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w3mo_first` (flag_days_consistency): rate 0.41017 over 295 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w6mo_first` (flag_days_consistency): rate 0.28662 over 471 rows

**synthetic[tvae_qt_seed2]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 0.3945 over 109 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 0.2649 over 302 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w3mo_first` (flag_days_consistency): rate 0.17717 over 508 rows
- `med_arni => med_arb` (implication): rate 0.16667 over 6 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w6mo_first` (flag_days_consistency): rate 0.1179 over 687 rows

**synthetic[tvae_seed0]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 0.88119 over 101 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 0.53052 over 213 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w3mo_first` (flag_days_consistency): rate 0.17621 over 454 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w6mo_first` (flag_days_consistency): rate 0.16212 over 623 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w3a_first` (flag_days_consistency): rate 0.13891 over 1231 rows

**synthetic[tvae_seed1]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 0.93651 over 63 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 0.5288 over 191 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w3mo_first` (flag_days_consistency): rate 0.2228 over 386 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w6mo_first` (flag_days_consistency): rate 0.20287 over 557 rows
- `med_arni => med_arb` (implication): rate 0.14286 over 7 rows

**synthetic[tvae_seed2]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 0.92308 over 78 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 0.55981 over 209 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w6mo_first` (flag_days_consistency): rate 0.21405 over 598 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w3mo_first` (flag_days_consistency): rate 0.19545 over 440 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1a_first` (flag_days_consistency): rate 0.14964 over 842 rows

