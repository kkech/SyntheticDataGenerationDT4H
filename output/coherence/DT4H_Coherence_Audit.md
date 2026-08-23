# Row-Coherence Audit

363 rules ({'implication': 351, 'category_range': 4, 'days_bounds': 1, 'flag_days_consistency': 7}) mined/learned from the TRAIN split and validated on real data. The holdout row is the fair baseline: real, unseen patients violating the same rules. A synthetic dataset far above it produces rows that are individually implausible patients even when every column's distribution is correct.

| frame | applicable checks | violations | violation rate | rules violated |
|---|---|---|---|---|
| train (real) | 117993 | 17 | 0.00014 | 14/363 |
| holdout (real, unseen) | 37897 | 88 | 0.00232 | 70/363 |
| synthetic[aim50_eps1_seed0] | 7845 | 332 | 0.04232 | 3/363 |
| synthetic[aim50_eps5_seed0] | 7720 | 70 | 0.00907 | 5/363 |
| synthetic[ctgan_seed0] | 130103 | 39728 | 0.30536 | 347/363 |
| synthetic[ctgan_seed1] | 114041 | 40007 | 0.35081 | 336/363 |
| synthetic[ctgan_seed2] | 114694 | 34297 | 0.29903 | 341/363 |
| synthetic[dpctgan_eps10_seed0] | 88175 | 41612 | 0.47193 | 76/363 |
| synthetic[dpctgan_eps15_seed0] | 51269 | 35302 | 0.68856 | 77/363 |
| synthetic[dpctgan_eps15_seed1] | 55755 | 23597 | 0.42323 | 58/363 |
| synthetic[dpctgan_eps15_seed2] | 91577 | 33118 | 0.36164 | 73/363 |
| synthetic[dpctgan_eps1_seed0] | 64450 | 39847 | 0.61826 | 119/363 |
| synthetic[dpctgan_eps20_seed0] | 61359 | 34811 | 0.56733 | 49/363 |
| synthetic[dpctgan_eps5_seed0] | 117315 | 60613 | 0.51667 | 91/363 |
| synthetic[dpctgan_eps8_seed0] | 122731 | 54964 | 0.44784 | 97/363 |
| synthetic[gaussian_copula_seed0] | 84735 | 23597 | 0.27848 | 330/363 |
| synthetic[gaussian_copula_seed1] | 85871 | 23945 | 0.27885 | 324/363 |
| synthetic[gaussian_copula_seed2] | 83734 | 23629 | 0.28219 | 317/363 |
| synthetic[mst_eps10_seed0] | 121832 | 6037 | 0.04955 | 297/363 |
| synthetic[mst_eps15_seed0] | 122727 | 4458 | 0.03632 | 302/363 |
| synthetic[mst_eps15_seed1] | 122680 | 5491 | 0.04476 | 312/363 |
| synthetic[mst_eps15_seed2] | 123656 | 6363 | 0.05146 | 327/363 |
| synthetic[mst_eps1_seed0] | 115842 | 16729 | 0.14441 | 203/363 |
| synthetic[mst_eps20_seed0] | 122383 | 6374 | 0.05208 | 312/363 |
| synthetic[mst_eps5_seed0] | 123470 | 9236 | 0.0748 | 310/363 |
| synthetic[mst_eps8_seed0] | 126981 | 8181 | 0.06443 | 311/363 |
| synthetic[tvae_seed0] | 102586 | 2310 | 0.02252 | 200/363 |
| synthetic[tvae_seed1] | 99657 | 2332 | 0.0234 | 180/363 |
| synthetic[tvae_seed2] | 99090 | 1966 | 0.01984 | 153/363 |

## Worst rules per synthetic dataset

**synthetic[aim50_eps1_seed0]**
- `ckd_severity_from_calculated_egfr vs lab_results_valideGFR_value_last` (category_range): rate 0.09754 over 2881 rows
- `med_acei => med_rasi` (implication): rate 0.02073 over 1399 rows
- `ckd_severity_from_calculated_egfr vs lab_results_valideGFR_value_first` (category_range): rate 0.00705 over 3122 rows

**synthetic[aim50_eps5_seed0]**
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w1mo_first` (implication): rate 0.02913 over 103 rows
- `ckd_severity_from_calculated_egfr vs lab_results_valideGFR_value_last` (category_range): rate 0.01785 over 2970 rows
- `ckd_severity_from_calculated_egfr vs lab_results_valideGFR_value_first` (category_range): rate 0.00373 over 2953 rows
- `med_arni => med_rasi` (implication): rate 0.00309 over 324 rows
- `med_acei => med_rasi` (implication): rate 0.00146 over 1370 rows

**synthetic[ctgan_seed0]**
- `encounter_primary_reason_HF_Disease_f5a_w1a_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 2 rows
- `encounter_primary_reason_HF_Disease_f5a_w3a_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 5 rows
- `cause_of_death_isCV_f5a_w5a_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 1 rows
- `med_ivabradine_history => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 2 rows
- `med_ivabradine_history => cause_of_death_isRenal_f5a_w7d_first` (implication): rate 1.0 over 4 rows

**synthetic[ctgan_seed1]**
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 2 rows
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 5 rows
- `med_ivabradine_history => cause_of_death_isRenal_f5a_w7d_first` (implication): rate 1.0 over 2 rows
- `med_ivabradine_history => cause_of_death_isNonRenalAndNonCV_f5a_w7d_first` (implication): rate 1.0 over 2 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 0.98913 over 184 rows

**synthetic[ctgan_seed2]**
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 2 rows
- `encounter_primary_reason_CV_Disease_f5a_w1a_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 7 rows
- `med_ivabradine_history => cause_of_death_isRenal_f5a_w7d_first` (implication): rate 1.0 over 5 rows
- `med_ivabradine_history => cause_of_death_isNonRenalAndNonCV_f5a_w7d_first` (implication): rate 1.0 over 3 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w3mo_first` (implication): rate 0.96 over 50 rows

**synthetic[dpctgan_eps10_seed0]**
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w1a_first` (implication): rate 1.0 over 4 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w5a_first` (implication): rate 1.0 over 4 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w1a_first` (implication): rate 1.0 over 4 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 4 rows
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first => encounter_primary_reason_HF_Disease_f5a_w1a_first` (implication): rate 1.0 over 4 rows

**synthetic[dpctgan_eps15_seed0]**
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w5a_first` (implication): rate 1.0 over 3 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w6mo_first` (implication): rate 1.0 over 3 rows
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first => encounter_primary_reason_HF_Disease_f5a_w5a_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first => encounter_primary_reason_CV_Disease_f5a_w6mo_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w6mo_first => encounter_primary_reason_HF_Disease_f5a_w5a_first` (implication): rate 1.0 over 1 rows

**synthetic[dpctgan_eps15_seed1]**
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first => smoking_status_smoker_last` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first => encounter_primary_reason_HF_Disease_f5a_w6mo_first` (implication): rate 1.0 over 606 rows
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first => encounter_primary_reason_HF_Disease_f5a_w3a_first` (implication): rate 1.0 over 610 rows
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first => encounter_primary_reason_CV_Disease_f5a_w6mo_first` (implication): rate 1.0 over 2 rows
- `encounter_primary_reason_HF_Disease_f5a_w1a_first => encounter_primary_reason_HF_Disease_f5a_w6mo_first` (implication): rate 1.0 over 2 rows

**synthetic[dpctgan_eps15_seed2]**
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w6mo_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w1a_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w3a_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first => encounter_primary_reason_HF_Disease_f5a_w6mo_first` (implication): rate 1.0 over 2 rows
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first => encounter_primary_reason_HF_Disease_f5a_w1a_first` (implication): rate 1.0 over 2 rows

**synthetic[dpctgan_eps1_seed0]**
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first => encounter_primary_reason_HF_Disease_f5a_w6mo_first` (implication): rate 1.0 over 3 rows
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first => encounter_primary_reason_HF_Disease_f5a_w5a_first` (implication): rate 1.0 over 3 rows
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first => encounter_primary_reason_CV_Disease_f5a_w3mo_first` (implication): rate 1.0 over 3 rows
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first => encounter_primary_reason_CV_Disease_f5a_w5a_first` (implication): rate 1.0 over 2 rows
- `encounter_primary_reason_HF_Disease_f5a_w6mo_first => smoking_status_smoker_last` (implication): rate 1.0 over 1 rows

**synthetic[dpctgan_eps20_seed0]**
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w6mo_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first => encounter_primary_reason_HF_Disease_f5a_w6mo_first` (implication): rate 1.0 over 10 rows
- `encounter_primary_reason_HF_Disease_f5a_w1a_first => encounter_primary_reason_HF_Disease_f5a_w6mo_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w3a_first => encounter_primary_reason_HF_Disease_f5a_w6mo_first` (implication): rate 1.0 over 5 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w7d_first => encounter_primary_reason_non_CV_Disease_f5a_w6mo_first` (implication): rate 1.0 over 1 rows

**synthetic[dpctgan_eps5_seed0]**
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w6mo_first` (implication): rate 1.0 over 44 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w1a_first` (implication): rate 1.0 over 61 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w3a_first` (implication): rate 1.0 over 62 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w3mo_first` (implication): rate 1.0 over 61 rows
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first => encounter_primary_reason_HF_Disease_f5a_w6mo_first` (implication): rate 1.0 over 2347 rows

**synthetic[dpctgan_eps8_seed0]**
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first => encounter_primary_reason_HF_Disease_f5a_w1mo_first` (implication): rate 1.0 over 16 rows
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first => encounter_primary_reason_HF_Disease_f5a_w6mo_first` (implication): rate 1.0 over 16 rows
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first => encounter_primary_reason_HF_Disease_f5a_w1a_first` (implication): rate 1.0 over 3 rows
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first => encounter_primary_reason_HF_Disease_f5a_w5a_first` (implication): rate 1.0 over 16 rows
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first => encounter_primary_reason_CV_Disease_f5a_w1a_first` (implication): rate 1.0 over 16 rows

**synthetic[gaussian_copula_seed0]**
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 2 rows
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 6 rows
- `encounter_primary_reason_HF_Disease_f5a_w6mo_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 5 rows
- `encounter_primary_reason_HF_Disease_f5a_w1a_first => encounter_primary_reason_HF_Disease_f5a_w7d_first` (implication): rate 1.0 over 35 rows
- `encounter_primary_reason_HF_Disease_f5a_w3a_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 4 rows

**synthetic[gaussian_copula_seed1]**
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 4 rows
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 3 rows
- `encounter_primary_reason_HF_Disease_f5a_w6mo_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 2 rows
- `encounter_primary_reason_HF_Disease_f5a_w1a_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 5 rows
- `encounter_primary_reason_HF_Disease_f5a_w5a_first => encounter_primary_reason_HF_Disease_f5a_w7d_first` (implication): rate 1.0 over 11 rows

**synthetic[gaussian_copula_seed2]**
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 3 rows
- `encounter_primary_reason_HF_Disease_f5a_w6mo_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 6 rows
- `encounter_primary_reason_HF_Disease_f5a_w1a_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 4 rows
- `encounter_primary_reason_HF_Disease_f5a_w3a_first => encounter_primary_reason_HF_Disease_f5a_w7d_first` (implication): rate 1.0 over 10 rows
- `encounter_primary_reason_HF_Disease_f5a_w3a_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 8 rows

**synthetic[mst_eps10_seed0]**
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 18 rows
- `cause_of_death_isCV_f5a_w5a_first => cause_of_death_isCV_f5a_w1mo_first` (implication): rate 1.0 over 1 rows
- `cause_of_death_isCV_f5a_w5a_first => cause_of_death_isCV_f5a_w3mo_first` (implication): rate 1.0 over 17 rows
- `med_ivabradine_history => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 2 rows
- `med_ivabradine_history => cause_of_death_isRenal_f5a_w7d_first` (implication): rate 1.0 over 2 rows

**synthetic[mst_eps15_seed0]**
- `cause_of_death_isCV_f5a_w1a_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 22 rows
- `cause_of_death_isCV_f5a_w1a_first => cause_of_death_isCV_f5a_w1mo_first` (implication): rate 1.0 over 19 rows
- `cause_of_death_isCV_f5a_w5a_first => smoking_status_formerSmoker_last` (implication): rate 1.0 over 2 rows
- `med_ivabradine_history => cause_of_death_isRenal_f5a_w7d_first` (implication): rate 1.0 over 3 rows
- `med_ivabradine_history => cause_of_death_isNonRenalAndNonCV_f5a_w7d_first` (implication): rate 1.0 over 6 rows

**synthetic[mst_eps15_seed1]**
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => smoking_status_smoker_last` (implication): rate 1.0 over 3 rows
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first => smoking_status_smoker_last` (implication): rate 1.0 over 3 rows
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => smoking_status_smoker_last` (implication): rate 1.0 over 3 rows
- `encounter_primary_reason_CV_Disease_f5a_w1mo_first => smoking_status_smoker_last` (implication): rate 1.0 over 2 rows
- `encounter_primary_reason_CV_Disease_f5a_w3mo_first => smoking_status_smoker_last` (implication): rate 1.0 over 1 rows

**synthetic[mst_eps15_seed2]**
- `cause_of_death_isCV_f5a_w1a_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 6 rows
- `cause_of_death_isCV_f5a_w3a_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 2 rows
- `med_ivabradine_history => cause_of_death_isRenal_f5a_w7d_first` (implication): rate 1.0 over 11 rows
- `med_ivabradine_history => cause_of_death_isNonRenalAndNonCV_f5a_w7d_first` (implication): rate 1.0 over 8 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 1.0 over 109 rows

**synthetic[mst_eps1_seed0]**
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w6mo_first` (implication): rate 1.0 over 69 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w1a_first` (implication): rate 1.0 over 69 rows
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first => encounter_primary_reason_HF_Disease_f5a_w7d_first` (implication): rate 1.0 over 132 rows
- `encounter_primary_reason_HF_Disease_f5a_w6mo_first => encounter_primary_reason_HF_Disease_f5a_w7d_first` (implication): rate 1.0 over 191 rows
- `encounter_primary_reason_HF_Disease_f5a_w1a_first => encounter_primary_reason_HF_Disease_f5a_w7d_first` (implication): rate 1.0 over 148 rows

**synthetic[mst_eps20_seed0]**
- `encounter_primary_reason_HF_Disease_f5a_w1a_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 95 rows
- `encounter_primary_reason_CV_Disease_f5a_w1a_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 95 rows
- `cause_of_death_isCV_f5a_w5a_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 7 rows
- `cause_of_death_isCV_f5a_w5a_first => cause_of_death_isCV_f5a_w1mo_first` (implication): rate 1.0 over 11 rows
- `med_ivabradine_history => cause_of_death_isRenal_f5a_w7d_first` (implication): rate 1.0 over 5 rows

**synthetic[mst_eps5_seed0]**
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 3 rows
- `encounter_primary_reason_HF_Disease_f5a_w6mo_first => encounter_primary_reason_HF_Disease_f5a_w7d_first` (implication): rate 1.0 over 122 rows
- `encounter_primary_reason_HF_Disease_f5a_w6mo_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 5 rows
- `encounter_primary_reason_CV_Disease_f5a_w6mo_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 5 rows
- `encounter_primary_reason_CV_Disease_f5a_w1a_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 5 rows

**synthetic[mst_eps8_seed0]**
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w7d_first` (implication): rate 1.0 over 90 rows
- `cause_of_death_isCV_f5a_w1a_first => cause_of_death_isCV_f5a_w1mo_first` (implication): rate 1.0 over 15 rows
- `cause_of_death_isCV_f5a_w3a_first => cause_of_death_isCV_f5a_w1mo_first` (implication): rate 1.0 over 23 rows
- `cause_of_death_isCV_f5a_w5a_first => cause_of_death_isCV_f5a_w1mo_first` (implication): rate 1.0 over 13 rows
- `med_ivabradine_history => cause_of_death_isRenal_f5a_w7d_first` (implication): rate 1.0 over 56 rows

**synthetic[tvae_seed0]**
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_CV_Disease_f5a_w1mo_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 4 rows
- `encounter_primary_reason_CV_Disease_f5a_w3mo_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 5 rows
- `encounter_primary_reason_CV_Disease_f5a_w3a_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 5 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 0.91304 over 46 rows

**synthetic[tvae_seed1]**
- `encounter_primary_reason_CV_Disease_f5a_w3mo_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 3 rows
- `encounter_primary_reason_CV_Disease_f5a_w6mo_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 2 rows
- `encounter_primary_reason_CV_Disease_f5a_w1a_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 2 rows
- `encounter_primary_reason_CV_Disease_f5a_w3a_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 2 rows
- `encounter_primary_reason_CV_Disease_f5a_w5a_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 1 rows

**synthetic[tvae_seed2]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 0.92 over 50 rows
- `encounter_primary_reason_CV_Disease_f5a_w3mo_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 0.5 over 4 rows
- `encounter_primary_reason_CV_Disease_f5a_w6mo_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 0.5 over 4 rows
- `encounter_primary_reason_CV_Disease_f5a_w1a_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 0.5 over 4 rows
- `encounter_primary_reason_CV_Disease_f5a_w3a_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 0.5 over 4 rows

