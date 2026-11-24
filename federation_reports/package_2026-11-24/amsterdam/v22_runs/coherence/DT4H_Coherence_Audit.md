# Row-Coherence Audit

230 rules ({'implication': 218, 'category_range': 4, 'days_bounds': 1, 'flag_days_consistency': 7}) mined/learned from the TRAIN split and validated on real data. The holdout row is the fair baseline: real, unseen patients violating the same rules. A synthetic dataset far above it produces rows that are individually implausible patients even when every column's distribution is correct.

Two measures, answering different questions: the violation RATE is per applicable rule-check, while the row SHARE is the fraction of patients carrying at least one violation -- the one a release decision turns on, read against the real holdout's own share.

The 'consequent Missing' column is the evasion check for implication rules: the share of antecedent-true checks whose consequent was Missing and thus undecidable. A generator can push its violation rate toward zero by emitting Missing consequents; a share far above the real frames' reveals exactly that.

| frame | applicable checks | violations | violation rate | rules violated | consequent Missing | rows with >=1 violation |
|---|---|---|---|---|---|---|
| train (real) | 105478 | 0 | 0.0 | 0/230 | 24.8% | 0.0% (0/3334) |
| holdout (real, unseen) | 35390 | 13 | 0.00037 | 7/230 | 25.5% | 0.4% (5/1111) |
| synthetic[aim40_eps10_seed0] | 10168 | 206 | 0.02026 | 5/230 | 13.3% | 5.6% (188/3334) |
| synthetic[aim40_eps15_seed0] | 10158 | 227 | 0.02235 | 5/230 | 13.4% | 5.4% (181/3334) |
| synthetic[aim40_eps1_seed0] | 10095 | 262 | 0.02595 | 2/230 | 13.9% | 7.1% (236/3334) |
| synthetic[aim40_eps20_seed0] | 10084 | 130 | 0.01289 | 5/230 | 13.3% | 3.3% (109/3334) |
| synthetic[aim40_eps8_seed0] | 10139 | 88 | 0.00868 | 4/230 | 13.5% | 2.2% (74/3334) |
| synthetic[aim50_eps1_seed0] | 13024 | 558 | 0.04284 | 5/230 | 8.7% | 13.4% (446/3334) |
| synthetic[aim50_eps5_seed0] | 12884 | 152 | 0.0118 | 7/230 | 7.5% | 2.9% (96/3334) |
| synthetic[ctgan_qt_seed0] | 91408 | 32466 | 0.35518 | 226/230 | 37.6% | 92.4% (3081/3334) |
| synthetic[ctgan_seed0] | 82653 | 31851 | 0.38536 | 225/230 | 39.0% | 96.6% (3220/3334) |
| synthetic[ctgan_seed1] | 79976 | 32462 | 0.4059 | 228/230 | 41.6% | 97.4% (3248/3334) |
| synthetic[ctgan_seed2] | 93349 | 33991 | 0.36413 | 229/230 | 44.7% | 98.5% (3285/3334) |
| synthetic[ddpm_g_seed0] | 14544 | 118 | 0.00811 | 4/230 | 5.4% | 3.0% (99/3334) |
| synthetic[ddpm_seed0] | 103141 | 15513 | 0.15041 | 227/230 | 35.1% | 85.8% (2861/3334) |
| synthetic[ddpm_seed1] | 106876 | 17528 | 0.164 | 227/230 | 34.6% | 87.2% (2908/3334) |
| synthetic[ddpm_seed2] | 153673 | 23572 | 0.15339 | 225/230 | 29.6% | 89.1% (2972/3334) |
| synthetic[dpctgan_eps10_seed0] | 45246 | 20123 | 0.44475 | 73/230 | 52.0% | 100.0% (3334/3334) |
| synthetic[dpctgan_eps15_seed0] | 49477 | 22854 | 0.46191 | 63/230 | 50.5% | 100.0% (3334/3334) |
| synthetic[dpctgan_eps15_seed1] | 33004 | 13370 | 0.4051 | 45/230 | 61.4% | 100.0% (3334/3334) |
| synthetic[dpctgan_eps15_seed2] | 59947 | 27718 | 0.46238 | 51/230 | 60.1% | 100.0% (3334/3334) |
| synthetic[dpctgan_eps1_seed0] | 53512 | 29194 | 0.54556 | 87/230 | 61.7% | 100.0% (3334/3334) |
| synthetic[dpctgan_eps20_seed0] | 96515 | 37703 | 0.39064 | 95/230 | 50.5% | 100.0% (3334/3334) |
| synthetic[dpctgan_eps5_seed0] | 27274 | 23333 | 0.8555 | 58/230 | 42.1% | 100.0% (3334/3334) |
| synthetic[dpctgan_eps8_seed0] | 37004 | 30011 | 0.81102 | 71/230 | 68.4% | 100.0% (3334/3334) |
| synthetic[gaussian_copula_seed0] | 81597 | 27519 | 0.33726 | 226/230 | 42.3% | 95.1% (3172/3334) |
| synthetic[gaussian_copula_seed1] | 80835 | 26683 | 0.33009 | 225/230 | 43.0% | 95.8% (3194/3334) |
| synthetic[gaussian_copula_seed2] | 82572 | 27373 | 0.3315 | 226/230 | 42.4% | 94.8% (3162/3334) |
| synthetic[mst_eps0p5_seed0] | 100728 | 21822 | 0.21664 | 170/230 | 30.0% | 83.6% (2786/3334) |
| synthetic[mst_eps10_seed0] | 106242 | 4002 | 0.03767 | 188/230 | 23.5% | 24.4% (812/3334) |
| synthetic[mst_eps15_seed0] | 106345 | 4460 | 0.04194 | 213/230 | 23.3% | 23.6% (786/3334) |
| synthetic[mst_eps15_seed1] | 105059 | 4451 | 0.04237 | 201/230 | 24.4% | 25.1% (836/3334) |
| synthetic[mst_eps15_seed2] | 105718 | 4174 | 0.03948 | 208/230 | 24.0% | 24.9% (829/3334) |
| synthetic[mst_eps1_seed0] | 102149 | 9064 | 0.08873 | 169/230 | 28.6% | 46.0% (1534/3334) |
| synthetic[mst_eps20_seed0] | 105527 | 2911 | 0.02759 | 187/230 | 23.8% | 18.4% (615/3334) |
| synthetic[mst_eps5_seed0] | 104982 | 4690 | 0.04467 | 168/230 | 24.4% | 26.5% (882/3334) |
| synthetic[mst_eps8_seed0] | 105638 | 4420 | 0.04184 | 193/230 | 23.6% | 23.7% (790/3334) |
| synthetic[patectgan_eps15_seed0] | 45963 | 2354 | 0.05122 | 101/230 | 33.1% | 48.1% (1604/3334) |
| synthetic[patectgan_eps1_seed0] | 95351 | 43696 | 0.45826 | 229/230 | 47.7% | 100.0% (3333/3334) |
| synthetic[patectgan_eps5_seed0] | 96611 | 6817 | 0.07056 | 165/230 | 27.9% | 82.0% (2733/3334) |
| synthetic[tvae_cap256_seed0] | 94085 | 1723 | 0.01831 | 162/230 | 29.8% | 17.6% (588/3334) |
| synthetic[tvae_ep1000_seed0] | 89943 | 1914 | 0.02128 | 161/230 | 32.6% | 17.9% (598/3334) |
| synthetic[tvae_ind_seed0] | 90779 | 2964 | 0.03265 | 146/230 | 31.1% | 17.5% (583/3334) |
| synthetic[tvae_qt_seed0] | 96975 | 1905 | 0.01964 | 160/230 | 29.4% | 18.8% (627/3334) |
| synthetic[tvae_qt_seed1] | 95139 | 1726 | 0.01814 | 139/230 | 30.5% | 18.1% (604/3334) |
| synthetic[tvae_qt_seed2] | 85717 | 1460 | 0.01703 | 139/230 | 33.3% | 14.5% (484/3334) |
| synthetic[tvae_seed0] | 91245 | 1931 | 0.02116 | 158/230 | 31.7% | 18.4% (615/3334) |
| synthetic[tvae_seed1] | 90025 | 2037 | 0.02263 | 163/230 | 32.0% | 18.9% (631/3334) |
| synthetic[tvae_seed2] | 90548 | 1934 | 0.02136 | 159/230 | 31.2% | 19.0% (634/3334) |

## Worst rules per synthetic dataset

**synthetic[aim40_eps10_seed0]**
- `med_mra_history => med_diuretics_history` (implication): rate 0.08121 over 1256 rows
- `ckd_severity_from_calculated_egfr vs lab_results_valideGFR_value_last` (category_range): rate 0.02586 over 3093 rows
- `conditions_mc => cause_of_death_isAllCause_f5a_w1mo_first` (implication): rate 0.02041 over 49 rows
- `ckd_severity_from_calculated_egfr vs lab_results_valideGFR_value_first` (category_range): rate 0.00644 over 3104 rows
- `med_diuretics_loop_history => med_diuretics_history` (implication): rate 0.00113 over 2666 rows

**synthetic[aim40_eps15_seed0]**
- `conditions_mc => cause_of_death_isAllCause_f5a_w1mo_first` (implication): rate 0.04082 over 49 rows
- `ckd_severity_from_calculated_egfr vs lab_results_valideGFR_value_last` (category_range): rate 0.03788 over 3089 rows
- `med_mra_history => med_diuretics_history` (implication): rate 0.03424 over 1256 rows
- `ckd_severity_from_calculated_egfr vs lab_results_valideGFR_value_first` (category_range): rate 0.01773 over 3102 rows
- `med_diuretics_loop_history => med_diuretics_history` (implication): rate 0.00376 over 2662 rows

**synthetic[aim40_eps1_seed0]**
- `ckd_severity_from_calculated_egfr vs lab_results_valideGFR_value_last` (category_range): rate 0.06808 over 3026 rows
- `ckd_severity_from_calculated_egfr vs lab_results_valideGFR_value_first` (category_range): rate 0.01798 over 3115 rows

**synthetic[aim40_eps20_seed0]**
- `med_mra_history => med_diuretics_history` (implication): rate 0.03822 over 1256 rows
- `conditions_mc => cause_of_death_isAllCause_f5a_w1mo_first` (implication): rate 0.03704 over 54 rows
- `ckd_severity_from_calculated_egfr vs lab_results_valideGFR_value_last` (category_range): rate 0.01244 over 3054 rows
- `ckd_severity_from_calculated_egfr vs lab_results_valideGFR_value_first` (category_range): rate 0.01113 over 3055 rows
- `med_diuretics_loop_history => med_diuretics_history` (implication): rate 0.003 over 2665 rows

**synthetic[aim40_eps8_seed0]**
- `conditions_mc => cause_of_death_isAllCause_f5a_w1mo_first` (implication): rate 0.04082 over 49 rows
- `ckd_severity_from_calculated_egfr vs lab_results_valideGFR_value_last` (category_range): rate 0.01816 over 3083 rows
- `ckd_severity_from_calculated_egfr vs lab_results_valideGFR_value_first` (category_range): rate 0.00873 over 3094 rows
- `med_diuretics_loop_history => med_diuretics_history` (implication): rate 0.00113 over 2661 rows

**synthetic[aim50_eps1_seed0]**
- `ckd_severity_from_calculated_egfr vs lab_results_valideGFR_value_last` (category_range): rate 0.09502 over 3094 rows
- `med_mra_history => med_diuretics_history` (implication): rate 0.07633 over 1297 rows
- `ckd_severity_from_calculated_egfr vs lab_results_valideGFR_value_first` (category_range): rate 0.03791 over 3060 rows
- `med_acei => med_rasi` (implication): rate 0.02829 over 1202 rows
- `med_diuretics_loop_history => med_diuretics_history` (implication): rate 0.00557 over 2695 rows

**synthetic[aim50_eps5_seed0]**
- `ckd_severity_from_calculated_egfr vs lab_results_valideGFR_value_first` (category_range): rate 0.02167 over 3046 rows
- `ckd_severity_from_calculated_egfr vs lab_results_valideGFR_value_last` (category_range): rate 0.01872 over 3045 rows
- `med_arb => med_rasi` (implication): rate 0.00609 over 821 rows
- `med_sglt2i => med_oral_antidiabetic` (implication): rate 0.00556 over 720 rows
- `med_acei => med_rasi` (implication): rate 0.00542 over 1291 rows

**synthetic[ctgan_qt_seed0]**
- `smoking_status_smoker_last => smoking_status_formerSmoker_last` (implication): rate 1.0 over 1 rows
- `smoking_status_formerSmoker_last => smoking_status_smoker_last` (implication): rate 1.0 over 77 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 0.94595 over 74 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w6mo_first` (implication): rate 0.93333 over 120 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w3a_first` (implication): rate 0.92701 over 137 rows

**synthetic[ctgan_seed0]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 1.0 over 38 rows
- `smoking_status_formerSmoker_last => smoking_status_smoker_last` (implication): rate 0.95597 over 159 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w6mo_first` (implication): rate 0.93684 over 95 rows
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first => encounter_primary_reason_HF_Disease_f5a_w6mo_first` (implication): rate 0.91346 over 208 rows
- `encounter_primary_reason_HF_Disease_f5a_w5a_first => encounter_primary_reason_HF_Disease_f5a_w6mo_first` (implication): rate 0.90805 over 174 rows

**synthetic[ctgan_seed1]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 1.0 over 68 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 0.97895 over 95 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w1a_first` (implication): rate 0.9562 over 137 rows
- `encounter_primary_reason_HF_Disease_f5a_w5a_first => encounter_primary_reason_HF_Disease_f5a_w3a_first` (implication): rate 0.94211 over 190 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w3a_first` (implication): rate 0.94203 over 138 rows

**synthetic[ctgan_seed2]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 0.98571 over 140 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 0.95781 over 474 rows
- `smoking_status_formerSmoker_last => smoking_status_smoker_last` (implication): rate 0.93651 over 189 rows
- `med_arni => med_arb` (implication): rate 0.91047 over 592 rows
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first => encounter_primary_reason_HF_Disease_f5a_w1a_first` (implication): rate 0.88281 over 128 rows

**synthetic[ddpm_g_seed0]**
- `hyperkalemia_severity_categorizedValue vs lab_results_potassium_value_last` (category_range): rate 0.0198 over 2374 rows
- `ckd_severity_from_calculated_egfr vs lab_results_valideGFR_value_first` (category_range): rate 0.01806 over 1827 rows
- `ckd_severity_from_calculated_egfr vs lab_results_valideGFR_value_last` (category_range): rate 0.01551 over 2127 rows
- `hyperkalemia_severity_categorizedValue vs lab_results_potassium_value_first` (category_range): rate 0.00207 over 2413 rows

**synthetic[ddpm_seed0]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 0.98473 over 131 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 0.93285 over 417 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w3mo_first` (flag_days_consistency): rate 0.89264 over 652 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w6mo_first` (flag_days_consistency): rate 0.8184 over 837 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1a_first` (flag_days_consistency): rate 0.73347 over 953 rows

**synthetic[ddpm_seed1]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 0.97531 over 81 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 0.94502 over 291 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w3mo_first` (flag_days_consistency): rate 0.88214 over 543 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w6mo_first` (flag_days_consistency): rate 0.81793 over 725 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1a_first` (flag_days_consistency): rate 0.72771 over 1043 rows

**synthetic[ddpm_seed2]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 1.0 over 108 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 0.95796 over 333 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w3mo_first` (flag_days_consistency): rate 0.89882 over 593 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w6mo_first` (flag_days_consistency): rate 0.84284 over 789 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1a_first` (flag_days_consistency): rate 0.78189 over 1027 rows

**synthetic[dpctgan_eps10_seed0]**
- `smoking_status_formerSmoker_last => smoking_status_smoker_last` (implication): rate 1.0 over 2 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w3mo_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w6mo_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w1a_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w3a_first` (implication): rate 1.0 over 1 rows

**synthetic[dpctgan_eps15_seed0]**
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w3mo_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w6mo_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w1a_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w3a_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w5a_first` (implication): rate 1.0 over 1 rows

**synthetic[dpctgan_eps15_seed1]**
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w1a_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first => encounter_primary_reason_CV_Disease_f5a_w1a_first` (implication): rate 1.0 over 4 rows
- `encounter_primary_reason_HF_Disease_f5a_w1a_first => encounter_primary_reason_HF_Disease_f5a_w3a_first` (implication): rate 1.0 over 12 rows
- `encounter_primary_reason_HF_Disease_f5a_w5a_first => encounter_primary_reason_HF_Disease_f5a_w1a_first` (implication): rate 1.0 over 4 rows
- `encounter_primary_reason_HF_Disease_f5a_w5a_first => encounter_primary_reason_HF_Disease_f5a_w3a_first` (implication): rate 1.0 over 1 rows

**synthetic[dpctgan_eps15_seed2]**
- `smoking_status_formerSmoker_last => smoking_status_smoker_last` (implication): rate 1.0 over 2 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w3mo_first` (implication): rate 1.0 over 2 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w1a_first` (implication): rate 1.0 over 2 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w5a_first` (implication): rate 1.0 over 2 rows
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first => encounter_primary_reason_HF_Disease_f5a_w1a_first` (implication): rate 1.0 over 22 rows

**synthetic[dpctgan_eps1_seed0]**
- `smoking_status_formerSmoker_last => smoking_status_smoker_last` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w1a_first` (implication): rate 1.0 over 33 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w3mo_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first => encounter_primary_reason_HF_Disease_f5a_w6mo_first` (implication): rate 1.0 over 6 rows
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first => encounter_primary_reason_HF_Disease_f5a_w5a_first` (implication): rate 1.0 over 6 rows

**synthetic[dpctgan_eps20_seed0]**
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w6mo_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w1a_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w3a_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w5a_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w3mo_first` (implication): rate 1.0 over 1 rows

**synthetic[dpctgan_eps5_seed0]**
- `smoking_status_formerSmoker_last => smoking_status_smoker_last` (implication): rate 1.0 over 2 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w3a_first` (implication): rate 1.0 over 10 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w5a_first` (implication): rate 1.0 over 8 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w3mo_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w6mo_first` (implication): rate 1.0 over 10 rows

**synthetic[dpctgan_eps8_seed0]**
- `smoking_status_formerSmoker_last => smoking_status_smoker_last` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w6mo_first => encounter_primary_reason_HF_Disease_f5a_w3mo_first` (implication): rate 1.0 over 8 rows
- `encounter_primary_reason_HF_Disease_f5a_w6mo_first => encounter_primary_reason_HF_Disease_f5a_w1a_first` (implication): rate 1.0 over 3 rows
- `encounter_primary_reason_HF_Disease_f5a_w6mo_first => encounter_primary_reason_HF_Disease_f5a_w3a_first` (implication): rate 1.0 over 8 rows
- `encounter_primary_reason_HF_Disease_f5a_w6mo_first => encounter_primary_reason_HF_Disease_f5a_w5a_first` (implication): rate 1.0 over 8 rows

**synthetic[gaussian_copula_seed0]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 1.0 over 74 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 0.98298 over 235 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w3mo_first` (flag_days_consistency): rate 0.92111 over 431 rows
- `encounter_primary_reason_HF_Disease_f5a_w5a_first => encounter_primary_reason_CV_Disease_f5a_w3a_first` (implication): rate 0.89778 over 225 rows
- `encounter_primary_reason_HF_Disease_f5a_w3a_first => encounter_primary_reason_CV_Disease_f5a_w3a_first` (implication): rate 0.89238 over 223 rows

**synthetic[gaussian_copula_seed1]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 1.0 over 79 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 0.94894 over 235 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w3mo_first` (flag_days_consistency): rate 0.92111 over 431 rows
- `smoking_status_formerSmoker_last => smoking_status_smoker_last` (implication): rate 0.89286 over 168 rows
- `encounter_primary_reason_HF_Disease_f5a_w6mo_first => encounter_primary_reason_CV_Disease_f5a_w6mo_first` (implication): rate 0.88571 over 175 rows

**synthetic[gaussian_copula_seed2]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 1.0 over 99 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 0.98347 over 242 rows
- `smoking_status_formerSmoker_last => smoking_status_smoker_last` (implication): rate 0.93827 over 162 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w3mo_first` (flag_days_consistency): rate 0.89952 over 418 rows
- `encounter_primary_reason_HF_Disease_f5a_w1a_first => encounter_primary_reason_CV_Disease_f5a_w3a_first` (implication): rate 0.89894 over 188 rows

**synthetic[mst_eps0p5_seed0]**
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w1mo_first` (implication): rate 1.0 over 310 rows
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first => encounter_primary_reason_CV_Disease_f5a_w1mo_first` (implication): rate 1.0 over 101 rows
- `encounter_primary_reason_HF_Disease_f5a_w1a_first => encounter_primary_reason_CV_Disease_f5a_w1mo_first` (implication): rate 1.0 over 17 rows
- `encounter_primary_reason_HF_Disease_f5a_w3a_first => encounter_primary_reason_CV_Disease_f5a_w1mo_first` (implication): rate 1.0 over 130 rows
- `encounter_primary_reason_HF_Disease_f5a_w5a_first => encounter_primary_reason_CV_Disease_f5a_w1mo_first` (implication): rate 1.0 over 86 rows

**synthetic[mst_eps10_seed0]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 1.0 over 95 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 1.0 over 231 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w3mo_first` (flag_days_consistency): rate 1.0 over 398 rows
- `encounter_primary_reason_HF_Disease_f5a_w5a_first => encounter_primary_reason_HF_Disease_f5a_w1mo_first` (implication): rate 0.55645 over 248 rows
- `encounter_primary_reason_HF_Disease_f5a_w3a_first => encounter_primary_reason_HF_Disease_f5a_w1mo_first` (implication): rate 0.53419 over 234 rows

**synthetic[mst_eps15_seed0]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 1.0 over 101 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 1.0 over 231 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w3mo_first` (flag_days_consistency): rate 1.0 over 405 rows
- `encounter_primary_reason_HF_Disease_f5a_w3a_first => encounter_primary_reason_HF_Disease_f5a_w1mo_first` (implication): rate 0.63713 over 237 rows
- `encounter_primary_reason_HF_Disease_f5a_w5a_first => encounter_primary_reason_HF_Disease_f5a_w1mo_first` (implication): rate 0.62281 over 228 rows

**synthetic[mst_eps15_seed1]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 1.0 over 104 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 1.0 over 242 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w3mo_first` (flag_days_consistency): rate 1.0 over 397 rows
- `encounter_primary_reason_HF_Disease_f5a_w5a_first => encounter_primary_reason_HF_Disease_f5a_w3mo_first` (implication): rate 0.76887 over 212 rows
- `encounter_primary_reason_HF_Disease_f5a_w3a_first => encounter_primary_reason_HF_Disease_f5a_w3mo_first` (implication): rate 0.75229 over 218 rows

**synthetic[mst_eps15_seed2]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 1.0 over 96 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 1.0 over 240 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w3mo_first` (flag_days_consistency): rate 1.0 over 400 rows
- `conditions_stroke => cause_of_death_isAllCause_f5a_w3mo_first` (implication): rate 0.73333 over 30 rows
- `encounter_primary_reason_HF_Disease_f5a_w1a_first => encounter_primary_reason_HF_Disease_f5a_w1mo_first` (implication): rate 0.63054 over 203 rows

**synthetic[mst_eps1_seed0]**
- `encounter_primary_reason_non_CV_Disease_f5a_w7d_first => encounter_primary_reason_non_CV_Disease_f5a_w3mo_first` (implication): rate 1.0 over 28 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w7d_first => encounter_primary_reason_non_CV_Disease_f5a_w6mo_first` (implication): rate 1.0 over 37 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w7d_first => encounter_primary_reason_non_CV_Disease_f5a_w1a_first` (implication): rate 1.0 over 71 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w7d_first => encounter_primary_reason_non_CV_Disease_f5a_w3a_first` (implication): rate 1.0 over 71 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w7d_first => encounter_primary_reason_non_CV_Disease_f5a_w5a_first` (implication): rate 1.0 over 71 rows

**synthetic[mst_eps20_seed0]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 1.0 over 98 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 1.0 over 235 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w3mo_first` (flag_days_consistency): rate 1.0 over 398 rows
- `conditions_stroke => cause_of_death_isAllCause_f5a_w3mo_first` (implication): rate 0.75862 over 29 rows
- `encounter_primary_reason_HF_Disease_f5a_w3a_first => encounter_primary_reason_HF_Disease_f5a_w1mo_first` (implication): rate 0.55981 over 209 rows

**synthetic[mst_eps5_seed0]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 1.0 over 96 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 1.0 over 225 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w3mo_first` (flag_days_consistency): rate 1.0 over 400 rows
- `encounter_primary_reason_HF_Disease_f5a_w1a_first => encounter_primary_reason_HF_Disease_f5a_w1mo_first` (implication): rate 0.48804 over 209 rows
- `encounter_primary_reason_HF_Disease_f5a_w3a_first => encounter_primary_reason_HF_Disease_f5a_w1mo_first` (implication): rate 0.47639 over 233 rows

**synthetic[mst_eps8_seed0]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 1.0 over 93 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 1.0 over 223 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w3mo_first` (flag_days_consistency): rate 1.0 over 397 rows
- `conditions_stroke => cause_of_death_isAllCause_f5a_w3mo_first` (implication): rate 0.7027 over 37 rows
- `conditions_stroke => cause_of_death_isAllCause_f5a_w6mo_first` (implication): rate 0.64706 over 17 rows

**synthetic[patectgan_eps15_seed0]**
- `smoking_status_formerSmoker_last => smoking_status_smoker_last` (implication): rate 1.0 over 53 rows
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first => encounter_primary_reason_HF_Disease_f5a_w6mo_first` (implication): rate 1.0 over 2 rows
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first => encounter_primary_reason_HF_Disease_f5a_w1a_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first => encounter_primary_reason_HF_Disease_f5a_w3a_first` (implication): rate 1.0 over 2 rows
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first => encounter_primary_reason_HF_Disease_f5a_w5a_first` (implication): rate 1.0 over 2 rows

**synthetic[patectgan_eps1_seed0]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 1.0 over 215 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 0.98769 over 325 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w3mo_first` (flag_days_consistency): rate 0.98323 over 477 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w3a_first` (flag_days_consistency): rate 0.97038 over 1688 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1a_first` (flag_days_consistency): rate 0.96634 over 1010 rows

**synthetic[patectgan_eps5_seed0]**
- `smoking_status_formerSmoker_last => smoking_status_smoker_last` (implication): rate 1.0 over 80 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w3mo_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w6mo_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w1a_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w3a_first` (implication): rate 1.0 over 1 rows

**synthetic[tvae_cap256_seed0]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 0.83607 over 61 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 0.46552 over 232 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w6mo_first` (flag_days_consistency): rate 0.13508 over 955 rows
- `med_potassium_binders_history => cause_of_death_isAllCause_f5a_w1a_first` (implication): rate 0.11111 over 9 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w3a_first` (flag_days_consistency): rate 0.10096 over 1882 rows

**synthetic[tvae_ep1000_seed0]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 0.77273 over 44 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 0.44318 over 176 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w5a_first` (implication): rate 0.18182 over 66 rows
- `med_potassium_binders_history => cause_of_death_isAllCause_f5a_w1a_first` (implication): rate 0.17647 over 17 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w3a_first` (implication): rate 0.15152 over 66 rows

**synthetic[tvae_ind_seed0]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 0.59091 over 44 rows
- `encounter_primary_reason_CV_Disease_f5a_w1a_first => encounter_primary_reason_CV_Disease_f5a_w7d_first` (implication): rate 0.55491 over 173 rows
- `encounter_primary_reason_CV_Disease_f5a_w6mo_first => encounter_primary_reason_CV_Disease_f5a_w7d_first` (implication): rate 0.49677 over 155 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w7d_first => encounter_primary_reason_non_CV_Disease_f5a_w1a_first` (implication): rate 0.46988 over 166 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w7d_first => encounter_primary_reason_non_CV_Disease_f5a_w3mo_first` (implication): rate 0.43373 over 166 rows

**synthetic[tvae_qt_seed0]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 0.62264 over 53 rows
- `med_potassium_binders_history => cause_of_death_isAllCause_f5a_w1a_first` (implication): rate 0.35294 over 17 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 0.35 over 200 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w3mo_first` (flag_days_consistency): rate 0.14395 over 521 rows
- `encounter_primary_reason_HF_Disease_f5a_w5a_first => encounter_primary_reason_HF_Disease_f5a_w1mo_first` (implication): rate 0.14035 over 57 rows

**synthetic[tvae_qt_seed1]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 0.6875 over 64 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 0.40708 over 226 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w1a_first` (implication): rate 0.18182 over 33 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w3a_first` (implication): rate 0.18182 over 33 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w3mo_first` (flag_days_consistency): rate 0.15736 over 591 rows

**synthetic[tvae_qt_seed2]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 0.65854 over 41 rows
- `med_potassium_binders_history => cause_of_death_isAllCause_f5a_w1a_first` (implication): rate 0.52632 over 19 rows
- `conditions_mc => cause_of_death_isAllCause_f5a_w1mo_first` (implication): rate 0.4 over 5 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 0.28934 over 197 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w3mo_first` (flag_days_consistency): rate 0.16942 over 484 rows

**synthetic[tvae_seed0]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 0.89474 over 57 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 0.5 over 240 rows
- `med_potassium_binders_history => cause_of_death_isAllCause_f5a_w1a_first` (implication): rate 0.42857 over 7 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w7d_first => encounter_primary_reason_non_CV_Disease_f5a_w6mo_first` (implication): rate 0.22917 over 96 rows
- `encounter_primary_reason_CV_Disease_f5a_w6mo_first => encounter_primary_reason_CV_Disease_f5a_w7d_first` (implication): rate 0.18333 over 120 rows

**synthetic[tvae_seed1]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 0.92157 over 51 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 0.45238 over 168 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w5a_first` (implication): rate 0.19298 over 57 rows
- `med_potassium_binders_history => cause_of_death_isAllCause_f5a_w1a_first` (implication): rate 0.16667 over 18 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w3a_first` (implication): rate 0.15789 over 57 rows

**synthetic[tvae_seed2]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 0.87097 over 31 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 0.43558 over 163 rows
- `med_potassium_binders_history => cause_of_death_isAllCause_f5a_w1a_first` (implication): rate 0.3125 over 16 rows
- `encounter_primary_reason_CV_Disease_f5a_w6mo_first => encounter_primary_reason_CV_Disease_f5a_w7d_first` (implication): rate 0.2963 over 135 rows
- `encounter_primary_reason_CV_Disease_f5a_w1a_first => encounter_primary_reason_CV_Disease_f5a_w7d_first` (implication): rate 0.17949 over 117 rows

