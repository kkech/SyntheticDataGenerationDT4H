# Row-Coherence Audit

363 rules ({'implication': 351, 'category_range': 4, 'days_bounds': 1, 'flag_days_consistency': 7}) mined/learned from the TRAIN split and validated on real data. The holdout row is the fair baseline: real, unseen patients violating the same rules. A synthetic dataset far above it produces rows that are individually implausible patients even when every column's distribution is correct.

| frame | applicable checks | violations | violation rate | rules violated |
|---|---|---|---|---|
| train (real) | 117993 | 17 | 0.00014 | 14/363 |
| holdout (real, unseen) | 37897 | 88 | 0.00232 | 70/363 |
| synthetic[aim40_eps1_seed0] | 7336 | 272 | 0.03708 | 4/363 |
| synthetic[aim50_eps1_seed0] | 7667 | 405 | 0.05282 | 4/363 |
| synthetic[ctgan_qt_seed0] | 106820 | 41514 | 0.38864 | 350/363 |
| synthetic[ctgan_seed0] | 130103 | 39728 | 0.30536 | 347/363 |
| synthetic[ctgan_seed1] | 114041 | 40007 | 0.35081 | 336/363 |
| synthetic[ctgan_seed2] | 114694 | 34297 | 0.29903 | 341/363 |
| synthetic[ddpm_g_seed0] | 277702 | 66028 | 0.23777 | 362/363 |
| synthetic[ddpm_seed0] | 357843 | 140063 | 0.39141 | 362/363 |
| synthetic[ddpm_seed1] | 377931 | 155309 | 0.41095 | 362/363 |
| synthetic[ddpm_seed2] | 361630 | 158083 | 0.43714 | 362/363 |
| synthetic[dpctgan_eps10_seed0] | 54038 | 39532 | 0.73156 | 108/363 |
| synthetic[dpctgan_eps15_seed0] | 27141 | 8153 | 0.30039 | 73/363 |
| synthetic[dpctgan_eps15_seed1] | 59869 | 45599 | 0.76165 | 103/363 |
| synthetic[dpctgan_eps15_seed2] | 93852 | 8604 | 0.09168 | 77/363 |
| synthetic[dpctgan_eps1_seed0] | 98484 | 61242 | 0.62185 | 138/363 |
| synthetic[dpctgan_eps20_seed0] | 86149 | 41572 | 0.48256 | 82/363 |
| synthetic[dpctgan_eps5_seed0] | 47254 | 22359 | 0.47317 | 86/363 |
| synthetic[dpctgan_eps8_seed0] | 64997 | 31859 | 0.49016 | 74/363 |
| synthetic[gaussian_copula_seed0] | 84735 | 23597 | 0.27848 | 330/363 |
| synthetic[gaussian_copula_seed1] | 85871 | 23945 | 0.27885 | 324/363 |
| synthetic[gaussian_copula_seed2] | 83734 | 23629 | 0.28219 | 317/363 |
| synthetic[mst_eps0p5_seed0] | 100128 | 27948 | 0.27912 | 229/363 |
| synthetic[mst_eps10_seed0] | 123328 | 6108 | 0.04953 | 296/363 |
| synthetic[mst_eps15_seed0] | 126437 | 6681 | 0.05284 | 315/363 |
| synthetic[mst_eps15_seed1] | 122880 | 6996 | 0.05693 | 323/363 |
| synthetic[mst_eps15_seed2] | 121802 | 5363 | 0.04403 | 301/363 |
| synthetic[mst_eps1_seed0] | 120235 | 29290 | 0.24361 | 278/363 |
| synthetic[mst_eps20_seed0] | 121607 | 4838 | 0.03978 | 309/363 |
| synthetic[mst_eps5_seed0] | 123983 | 9319 | 0.07516 | 304/363 |
| synthetic[mst_eps8_seed0] | 121420 | 7405 | 0.06099 | 310/363 |
| synthetic[patectgan_eps15_seed0] | 66042 | 2863 | 0.04335 | 180/363 |
| synthetic[patectgan_eps1_seed0] | 110278 | 49162 | 0.4458 | 362/363 |
| synthetic[patectgan_eps5_seed0] | 111879 | 6534 | 0.0584 | 188/363 |
| synthetic[tvae_cap256_seed0] | 103546 | 2341 | 0.02261 | 191/363 |
| synthetic[tvae_ep1000_seed0] | 100037 | 2351 | 0.0235 | 219/363 |
| synthetic[tvae_ind_seed0] | 101760 | 1888 | 0.01855 | 175/363 |
| synthetic[tvae_qt_seed0] | 99411 | 2195 | 0.02208 | 191/363 |
| synthetic[tvae_qt_seed1] | 104217 | 2015 | 0.01933 | 164/363 |
| synthetic[tvae_qt_seed2] | 102664 | 2630 | 0.02562 | 174/363 |
| synthetic[tvae_seed0] | 102586 | 2310 | 0.02252 | 200/363 |
| synthetic[tvae_seed1] | 99657 | 2332 | 0.0234 | 180/363 |
| synthetic[tvae_seed2] | 99090 | 1966 | 0.01984 | 153/363 |

## Worst rules per synthetic dataset

**synthetic[aim40_eps1_seed0]**
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w1mo_first` (implication): rate 0.46296 over 54 rows
- `ckd_severity_from_calculated_egfr vs lab_results_valideGFR_value_last` (category_range): rate 0.07739 over 2946 rows
- `ckd_severity_from_calculated_egfr vs lab_results_valideGFR_value_first` (category_range): rate 0.00601 over 2997 rows
- `med_acei => med_rasi` (implication): rate 0.00075 over 1339 rows

**synthetic[aim50_eps1_seed0]**
- `med_arni => med_rasi` (implication): rate 0.39118 over 363 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w1mo_first` (implication): rate 0.35294 over 68 rows
- `ckd_severity_from_calculated_egfr vs lab_results_valideGFR_value_last` (category_range): rate 0.08389 over 2837 rows
- `med_acei => med_rasi` (implication): rate 0.00075 over 1325 rows

**synthetic[ctgan_qt_seed0]**
- `encounter_primary_reason_HF_Disease_f5a_w1a_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 11 rows
- `encounter_primary_reason_HF_Disease_f5a_w5a_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 12 rows
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 16 rows
- `med_ivabradine_history => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 2 rows
- `med_ivabradine_history => cause_of_death_isRenal_f5a_w7d_first` (implication): rate 1.0 over 8 rows

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

**synthetic[ddpm_g_seed0]**
- `med_ivabradine_history => cause_of_death_isRenal_f5a_w7d_first` (implication): rate 1.0 over 851 rows
- `med_ivabradine_history => cause_of_death_isNonRenalAndNonCV_f5a_w7d_first` (implication): rate 1.0 over 929 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 1.0 over 1709 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 1.0 over 1515 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w3mo_first` (flag_days_consistency): rate 1.0 over 1685 rows

**synthetic[ddpm_seed0]**
- `med_ivabradine_history => cause_of_death_isRenal_f5a_w7d_first` (implication): rate 1.0 over 1409 rows
- `med_ivabradine_history => cause_of_death_isNonRenalAndNonCV_f5a_w7d_first` (implication): rate 1.0 over 1497 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 1.0 over 2014 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 1.0 over 1382 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w3mo_first` (flag_days_consistency): rate 1.0 over 1637 rows

**synthetic[ddpm_seed1]**
- `med_ivabradine_history => cause_of_death_isRenal_f5a_w7d_first` (implication): rate 1.0 over 928 rows
- `med_ivabradine_history => cause_of_death_isNonRenalAndNonCV_f5a_w7d_first` (implication): rate 1.0 over 848 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 1.0 over 1665 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 1.0 over 1502 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w3mo_first` (flag_days_consistency): rate 1.0 over 1684 rows

**synthetic[ddpm_seed2]**
- `med_ivabradine_history => cause_of_death_isRenal_f5a_w7d_first` (implication): rate 1.0 over 1704 rows
- `med_ivabradine_history => cause_of_death_isNonRenalAndNonCV_f5a_w7d_first` (implication): rate 1.0 over 1552 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 1.0 over 1146 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 1.0 over 2037 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w3mo_first` (flag_days_consistency): rate 1.0 over 1696 rows

**synthetic[dpctgan_eps10_seed0]**
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w3mo_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w5a_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w3mo_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w1a_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w3a_first` (implication): rate 1.0 over 1 rows

**synthetic[dpctgan_eps15_seed0]**
- `smoking_status_formerSmoker_last => smoking_status_smoker_last` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w1a_first => smoking_status_smoker_last` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w1a_first => encounter_primary_reason_HF_Disease_f5a_w5a_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w1a_first => encounter_primary_reason_CV_Disease_f5a_w1mo_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w1a_first => encounter_primary_reason_CV_Disease_f5a_w3mo_first` (implication): rate 1.0 over 1 rows

**synthetic[dpctgan_eps15_seed1]**
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w1a_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w5a_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w3mo_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w3a_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w5a_first` (implication): rate 1.0 over 1 rows

**synthetic[dpctgan_eps15_seed2]**
- `encounter_primary_reason_HF_Disease_f5a_w6mo_first => encounter_primary_reason_CV_Disease_f5a_w1mo_first` (implication): rate 1.0 over 2 rows
- `encounter_primary_reason_HF_Disease_f5a_w6mo_first => encounter_primary_reason_CV_Disease_f5a_w6mo_first` (implication): rate 1.0 over 2 rows
- `encounter_primary_reason_HF_Disease_f5a_w6mo_first => encounter_primary_reason_CV_Disease_f5a_w3a_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w6mo_first => encounter_primary_reason_CV_Disease_f5a_w5a_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w1a_first => encounter_primary_reason_CV_Disease_f5a_w1mo_first` (implication): rate 1.0 over 3 rows

**synthetic[dpctgan_eps1_seed0]**
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w3mo_first` (implication): rate 1.0 over 2 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w1a_first` (implication): rate 1.0 over 2 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w3a_first` (implication): rate 1.0 over 2 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w5a_first` (implication): rate 1.0 over 2 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w3a_first` (implication): rate 1.0 over 2 rows

**synthetic[dpctgan_eps20_seed0]**
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first => encounter_primary_reason_HF_Disease_f5a_w1mo_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first => encounter_primary_reason_HF_Disease_f5a_w3a_first` (implication): rate 1.0 over 3518 rows
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first => encounter_primary_reason_CV_Disease_f5a_w1mo_first` (implication): rate 1.0 over 3369 rows
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first => encounter_primary_reason_CV_Disease_f5a_w6mo_first` (implication): rate 1.0 over 3518 rows
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 1 rows

**synthetic[dpctgan_eps5_seed0]**
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w3mo_first` (implication): rate 1.0 over 2 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w3a_first` (implication): rate 1.0 over 2 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w5a_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w3mo_first` (implication): rate 1.0 over 2 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w6mo_first` (implication): rate 1.0 over 2 rows

**synthetic[dpctgan_eps8_seed0]**
- `encounter_primary_reason_HF_Disease_f5a_w6mo_first => smoking_status_smoker_last` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w6mo_first => encounter_primary_reason_HF_Disease_f5a_w7d_first` (implication): rate 1.0 over 9 rows
- `encounter_primary_reason_HF_Disease_f5a_w6mo_first => encounter_primary_reason_HF_Disease_f5a_w1mo_first` (implication): rate 1.0 over 3513 rows
- `encounter_primary_reason_HF_Disease_f5a_w6mo_first => encounter_primary_reason_HF_Disease_f5a_w3mo_first` (implication): rate 1.0 over 3505 rows
- `encounter_primary_reason_HF_Disease_f5a_w6mo_first => encounter_primary_reason_HF_Disease_f5a_w1a_first` (implication): rate 1.0 over 3518 rows

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

**synthetic[mst_eps0p5_seed0]**
- `smoking_status_formerSmoker_last => smoking_status_smoker_last` (implication): rate 1.0 over 35 rows
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first => smoking_status_smoker_last` (implication): rate 1.0 over 10 rows
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first => smoking_status_formerSmoker_last` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first => encounter_primary_reason_HF_Disease_f5a_w1mo_first` (implication): rate 1.0 over 15 rows
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first => encounter_primary_reason_HF_Disease_f5a_w6mo_first` (implication): rate 1.0 over 101 rows

**synthetic[mst_eps10_seed0]**
- `encounter_primary_reason_HF_Disease_f5a_w6mo_first => encounter_primary_reason_HF_Disease_f5a_w7d_first` (implication): rate 1.0 over 71 rows
- `cause_of_death_isCV_f5a_w1a_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 25 rows
- `cause_of_death_isCV_f5a_w5a_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 20 rows
- `med_ivabradine_history => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 2 rows
- `med_ivabradine_history => cause_of_death_isRenal_f5a_w7d_first` (implication): rate 1.0 over 3 rows

**synthetic[mst_eps15_seed0]**
- `encounter_primary_reason_HF_Disease_f5a_w1a_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 96 rows
- `med_ivabradine_history => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 9 rows
- `med_ivabradine_history => cause_of_death_isRenal_f5a_w7d_first` (implication): rate 1.0 over 9 rows
- `med_ivabradine_history => cause_of_death_isNonRenalAndNonCV_f5a_w7d_first` (implication): rate 1.0 over 9 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 1.0 over 107 rows

**synthetic[mst_eps15_seed1]**
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 47 rows
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 87 rows
- `encounter_primary_reason_HF_Disease_f5a_w6mo_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 110 rows
- `encounter_primary_reason_HF_Disease_f5a_w1a_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 111 rows
- `encounter_primary_reason_HF_Disease_f5a_w3a_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 111 rows

**synthetic[mst_eps15_seed2]**
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 3 rows
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w6mo_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 2 rows
- `encounter_primary_reason_HF_Disease_f5a_w1a_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 2 rows
- `encounter_primary_reason_HF_Disease_f5a_w3a_first => smoking_status_formerSmoker_last` (implication): rate 1.0 over 1 rows

**synthetic[mst_eps1_seed0]**
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => smoking_status_formerSmoker_last` (implication): rate 1.0 over 111 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w3mo_first` (implication): rate 1.0 over 65 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w1a_first` (implication): rate 1.0 over 68 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w3a_first` (implication): rate 1.0 over 68 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w5a_first` (implication): rate 1.0 over 68 rows

**synthetic[mst_eps20_seed0]**
- `cause_of_death_isCV_f5a_w1a_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 14 rows
- `cause_of_death_isCV_f5a_w1a_first => cause_of_death_isCV_f5a_w3mo_first` (implication): rate 1.0 over 35 rows
- `cause_of_death_isCV_f5a_w5a_first => smoking_status_smoker_last` (implication): rate 1.0 over 1 rows
- `med_potassium_binders => smoking_status_smoker_last` (implication): rate 1.0 over 1 rows
- `med_ivabradine_history => smoking_status_smoker_last` (implication): rate 1.0 over 1 rows

**synthetic[mst_eps5_seed0]**
- `encounter_primary_reason_HF_Disease_f5a_w1a_first => smoking_status_smoker_last` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w3a_first => smoking_status_smoker_last` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => encounter_primary_reason_CV_Disease_f5a_w1mo_first` (implication): rate 1.0 over 109 rows
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => encounter_primary_reason_CV_Disease_f5a_w3mo_first` (implication): rate 1.0 over 109 rows
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 23 rows

**synthetic[mst_eps8_seed0]**
- `encounter_primary_reason_non_CV_Disease_f5a_w7d_first => cause_of_death_isAllCause_f5a_w7d_first` (implication): rate 1.0 over 26 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w1mo_first => cause_of_death_isAllCause_f5a_w7d_first` (implication): rate 1.0 over 12 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w3mo_first => cause_of_death_isAllCause_f5a_w7d_first` (implication): rate 1.0 over 13 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w6mo_first => cause_of_death_isAllCause_f5a_w7d_first` (implication): rate 1.0 over 6 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w1a_first => cause_of_death_isAllCause_f5a_w7d_first` (implication): rate 1.0 over 15 rows

**synthetic[patectgan_eps15_seed0]**
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first => encounter_primary_reason_HF_Disease_f5a_w7d_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first => encounter_primary_reason_HF_Disease_f5a_w1mo_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first => encounter_primary_reason_HF_Disease_f5a_w1a_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first => encounter_primary_reason_CV_Disease_f5a_w7d_first` (implication): rate 1.0 over 2 rows
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 1 rows

**synthetic[patectgan_eps1_seed0]**
- `med_ivabradine_history => cause_of_death_isRenal_f5a_w7d_first` (implication): rate 1.0 over 20 rows
- `med_ivabradine_history => cause_of_death_isNonRenalAndNonCV_f5a_w7d_first` (implication): rate 1.0 over 18 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 1.0 over 283 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 0.97872 over 376 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w3mo_first` (flag_days_consistency): rate 0.95324 over 556 rows

**synthetic[patectgan_eps5_seed0]**
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w3mo_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w6mo_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w1a_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w3a_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w5a_first` (implication): rate 1.0 over 1 rows

**synthetic[tvae_cap256_seed0]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 0.9322 over 59 rows
- `encounter_primary_reason_CV_Disease_f5a_w1a_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 0.78571 over 14 rows
- `encounter_primary_reason_CV_Disease_f5a_w6mo_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 0.76923 over 13 rows
- `encounter_primary_reason_CV_Disease_f5a_w5a_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 0.75 over 12 rows
- `encounter_primary_reason_CV_Disease_f5a_w3mo_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 0.7 over 10 rows

**synthetic[tvae_ep1000_seed0]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 0.89474 over 38 rows
- `encounter_primary_reason_CV_Disease_f5a_w3mo_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 0.75 over 8 rows
- `encounter_primary_reason_CV_Disease_f5a_w3a_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 0.71429 over 7 rows
- `encounter_primary_reason_CV_Disease_f5a_w1mo_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 0.66667 over 6 rows
- `encounter_primary_reason_CV_Disease_f5a_w6mo_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 0.66667 over 9 rows

**synthetic[tvae_ind_seed0]**
- `encounter_primary_reason_CV_Disease_f5a_w1a_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 0.5 over 2 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 0.29464 over 224 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 0.28947 over 76 rows
- `encounter_primary_reason_HF_Disease_f5a_w1a_first => encounter_primary_reason_HF_Disease_f5a_w7d_first` (implication): rate 0.27273 over 11 rows
- `encounter_primary_reason_HF_Disease_f5a_w3a_first => encounter_primary_reason_HF_Disease_f5a_w7d_first` (implication): rate 0.2 over 10 rows

**synthetic[tvae_qt_seed0]**
- `med_potassium_binders_history => cause_of_death_isAllCause_f5a_w7d_first` (implication): rate 1.0 over 2 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 0.80952 over 63 rows
- `encounter_primary_reason_CV_Disease_f5a_w3mo_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 0.66667 over 3 rows
- `encounter_primary_reason_CV_Disease_f5a_w6mo_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 0.66667 over 3 rows
- `encounter_primary_reason_CV_Disease_f5a_w1a_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 0.66667 over 3 rows

**synthetic[tvae_qt_seed1]**
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_CV_Disease_f5a_w1mo_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_CV_Disease_f5a_w3mo_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_CV_Disease_f5a_w6mo_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_CV_Disease_f5a_w1a_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 3 rows

**synthetic[tvae_qt_seed2]**
- `encounter_primary_reason_CV_Disease_f5a_w3mo_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 6 rows
- `encounter_primary_reason_CV_Disease_f5a_w6mo_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 6 rows
- `encounter_primary_reason_CV_Disease_f5a_w1a_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 8 rows
- `encounter_primary_reason_CV_Disease_f5a_w3a_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 8 rows
- `encounter_primary_reason_CV_Disease_f5a_w5a_first => cause_of_death_isCV_f5a_w7d_first` (implication): rate 1.0 over 6 rows

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

