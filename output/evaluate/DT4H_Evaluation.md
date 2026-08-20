# Evaluation: original vs preprocessed vs synthetic

Metrics are computed per column over observed values (nulls excluded); missingness rates are compared separately. KS and TVD are in [0,1], lower is closer; `W/std` is the Wasserstein distance in units of the reference standard deviation.

| comparison | cols | KS mean | KS median | KS<0.1 | W/std mean | TVD mean | TVD<0.05 | missing-rate MAD |
|---|---|---|---|---|---|---|---|---|
| original vs preprocessed | 164 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 1.0 | 0.0 |
| preprocessed vs synthetic[ctgan] | 249 | 0.3199 | 0.2956 | 0.082 | 0.6724 | 0.0597 | 0.5957 | 0.0938 |
| original vs synthetic[ctgan] | 164 | 0.3199 | 0.2956 | 0.082 | 0.6724 | 0.0426 | 0.699 | 0.0938 |
| preprocessed vs synthetic[gaussian_copula] | 245 | 0.4542 | 0.4193 | 0.1053 | 1.285 | 0.0054 | 0.9947 | 0.1174 |
| original vs synthetic[gaussian_copula] | 160 | 0.4542 | 0.4193 | 0.1053 | 1.285 | 0.0066 | 0.9903 | 0.1174 |
| preprocessed vs synthetic[mst] | 232 | 0.3352 | 0.24 | 0.0227 | 0.3164 | 0.0016 | 1.0 | 0.0185 |
| original vs synthetic[mst] | 147 | 0.3352 | 0.24 | 0.0227 | 0.3164 | 0.0016 | 1.0 | 0.0185 |
| preprocessed vs synthetic[tvae] | 249 | 0.2059 | 0.2181 | 0.2131 | 0.2757 | 0.0487 | 0.5957 | 0.0441 |
| original vs synthetic[tvae] | 164 | 0.2059 | 0.2181 | 0.2131 | 0.2757 | 0.042 | 0.6602 | 0.0441 |

## original vs preprocessed

Worst numeric columns (by KS):
- `patient_demographics_age`: KS=0.0, W/std=0.0, mean 70.9842 -> 70.9842, missing 0% -> 0%
- `encounters_lengthOfStay`: KS=0.0, W/std=0.0, mean 10.5254 -> 10.5254, missing 0% -> 0%
- `encounters_numOfPreviousHFStays_count`: KS=0.0, W/std=0.0, mean 52.5614 -> 52.5614, missing 0% -> 0%
- `vital_signs_weight_value_p6mo_last`: KS=0.0, W/std=0.0, mean 77.4701 -> 77.4701, missing 8% -> 8%
- `vital_signs_weight_value_p6mo_first`: KS=0.0, W/std=0.0, mean 79.7398 -> 79.7398, missing 8% -> 8%
Worst categorical columns (by TVD):
- `patient_demographics_gender`: TVD=0.0, 2 -> 2 categories, missing 0% -> 0%
- `encounters_encounterClass`: TVD=0.0, 1 -> 1 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.0, 10 -> 10 categories, missing 0% -> 0%
- `symptoms_Ankle_swelling_display_pET_any`: TVD=0.0, 1 -> 1 categories, missing 0% -> 0%
- `symptoms_Ascites_display_pET_any`: TVD=0.0, 1 -> 1 categories, missing 0% -> 0%

## preprocessed vs synthetic[ctgan]

Worst numeric columns (by KS):
- `electrocardiographs_ecg_qt_duration_corrected_pET_first`: KS=0.6488, W/std=1.9653, mean 472.4841 -> 572.2142, missing 49% -> 42%
- `vital_signs_heartRate_value_last`: KS=0.6437, W/std=0.828, mean 109.6174 -> 97.7177, missing 48% -> 20%
- `lab_results_ferritin_value_first`: KS=0.6256, W/std=0.4723, mean 522.5048 -> 1366.6507, missing 80% -> 55%
- `lab_results_tropTHs_value_last`: KS=0.6039, W/std=0.2693, mean 0.4651 -> 0.8967, missing 63% -> 89%
- `lab_results_albuminBS_value_last`: KS=0.6018, W/std=1.6354, mean 29.985 -> 18.7145, missing 57% -> 71%
Worst categorical columns (by TVD):
- `conditions_vd`: TVD=0.2714, 2 -> 2 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.2657, 10 -> 10 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.2495, 7 -> 7 categories, missing 0% -> 0%
- `med_inotropes_history`: TVD=0.2484, 2 -> 2 categories, missing 0% -> 0%
- `med_insulins`: TVD=0.2467, 2 -> 2 categories, missing 0% -> 0%

## original vs synthetic[ctgan]

Worst numeric columns (by KS):
- `electrocardiographs_ecg_qt_duration_corrected_pET_first`: KS=0.6488, W/std=1.9653, mean 472.4841 -> 572.2142, missing 49% -> 42%
- `vital_signs_heartRate_value_last`: KS=0.6437, W/std=0.828, mean 109.6174 -> 97.7177, missing 48% -> 20%
- `lab_results_ferritin_value_first`: KS=0.6256, W/std=0.4723, mean 522.5048 -> 1366.6507, missing 80% -> 55%
- `lab_results_tropTHs_value_last`: KS=0.6039, W/std=0.2693, mean 0.4651 -> 0.8967, missing 63% -> 89%
- `lab_results_albuminBS_value_last`: KS=0.6018, W/std=1.6354, mean 29.985 -> 18.7145, missing 57% -> 71%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.2657, 10 -> 10 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.2495, 7 -> 7 categories, missing 1% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.215, 6 -> 6 categories, missing 11% -> 0%
- `conditions_heart_failure_occurred_prior_to_18_months_any`: TVD=0.1958, 2 -> 2 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.1721, 7 -> 7 categories, missing 6% -> 0%

## preprocessed vs synthetic[gaussian_copula]

Worst numeric columns (by KS):
- `lab_results_ldl_value_first`: KS=0.9928, W/std=1.8886, mean 2.1159 -> 0.3065, missing 91% -> 100%
- `lab_results_hdl_value_first`: KS=0.9785, W/std=1.9288, mean 1.1884 -> 0.3228, missing 91% -> 100%
- `lab_results_ldl_value_last`: KS=0.9686, W/std=1.8388, mean 2.1058 -> 0.3296, missing 91% -> 100%
- `lab_results_ferritin_value_last`: KS=0.9548, W/std=0.9082, mean 493.6046 -> 1785.425, missing 80% -> 100%
- `echocardiographs_lvef_pET_last`: KS=0.9076, W/std=2.778, mean 41.0664 -> 86.0552, missing 83% -> 83%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.1741, 10 -> 9 categories, missing 0% -> 0%
- `med_cortico_syst_history`: TVD=0.0219, 2 -> 2 categories, missing 0% -> 0%
- `encounter_primary_reason_HF_Disease_f5a_w6mo_first`: TVD=0.0164, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_CV_Disease_f5a_w1a_first`: TVD=0.016, 3 -> 3 categories, missing 0% -> 0%
- `cause_of_death_isRenal_f5a_w3a_first`: TVD=0.0158, 2 -> 2 categories, missing 0% -> 0%

## original vs synthetic[gaussian_copula]

Worst numeric columns (by KS):
- `lab_results_ldl_value_first`: KS=0.9928, W/std=1.8886, mean 2.1159 -> 0.3065, missing 91% -> 100%
- `lab_results_hdl_value_first`: KS=0.9785, W/std=1.9288, mean 1.1884 -> 0.3228, missing 91% -> 100%
- `lab_results_ldl_value_last`: KS=0.9686, W/std=1.8388, mean 2.1058 -> 0.3296, missing 91% -> 100%
- `lab_results_ferritin_value_last`: KS=0.9548, W/std=0.9082, mean 493.6046 -> 1785.425, missing 80% -> 100%
- `echocardiographs_lvef_pET_last`: KS=0.9076, W/std=2.778, mean 41.0664 -> 86.0552, missing 83% -> 83%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.1741, 10 -> 9 categories, missing 0% -> 0%
- `encounter_primary_reason_HF_Disease_f5a_w6mo_first`: TVD=0.0164, 3 -> 3 categories, missing 53% -> 0%
- `encounter_primary_reason_CV_Disease_f5a_w1a_first`: TVD=0.016, 3 -> 3 categories, missing 49% -> 0%
- `cause_of_death_isRenal_f5a_w3a_first`: TVD=0.0158, 2 -> 2 categories, missing 63% -> 0%
- `encounter_primary_reason_renal_complications_f5a_w3a_first`: TVD=0.0151, 3 -> 3 categories, missing 44% -> 0%

## preprocessed vs synthetic[mst]

Worst numeric columns (by KS):
- `smoking_status_smoker_startTime_count`: KS=0.9823, W/std=0.1346, mean 0.0807 -> 0.066, missing 0% -> 0%
- `lab_results_tropTHs_value_first`: KS=0.9264, W/std=0.6037, mean 0.2243 -> 0.55, missing 63% -> 89%
- `conditions_heartFailure_timeFromEarliest_first`: KS=0.6716, W/std=0.1595, mean 11.3756 -> 12.3051, missing 0% -> 0%
- `lab_results_sodium_value_last`: KS=0.665, W/std=1.3398, mean 137.8828 -> 136.2524, missing 4% -> 5%
- `lab_results_sodium_value_first`: KS=0.6466, W/std=1.1967, mean 137.1365 -> 135.5718, missing 4% -> 4%
Worst categorical columns (by TVD):
- `hyperkalemia_severity_categorizedValue`: TVD=0.0081, 5 -> 5 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.0077, 7 -> 7 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.0072, 10 -> 10 categories, missing 0% -> 0%
- `med_anti_coag`: TVD=0.007, 2 -> 2 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.0066, 6 -> 6 categories, missing 0% -> 0%

## original vs synthetic[mst]

Worst numeric columns (by KS):
- `smoking_status_smoker_startTime_count`: KS=0.9823, W/std=0.1346, mean 0.0807 -> 0.066, missing 0% -> 0%
- `lab_results_tropTHs_value_first`: KS=0.9264, W/std=0.6037, mean 0.2243 -> 0.55, missing 63% -> 89%
- `conditions_heartFailure_timeFromEarliest_first`: KS=0.6716, W/std=0.1595, mean 11.3756 -> 12.3051, missing 0% -> 0%
- `lab_results_sodium_value_last`: KS=0.665, W/std=1.3398, mean 137.8828 -> 136.2524, missing 4% -> 5%
- `lab_results_sodium_value_first`: KS=0.6466, W/std=1.1967, mean 137.1365 -> 135.5718, missing 4% -> 4%
Worst categorical columns (by TVD):
- `hyperkalemia_severity_categorizedValue`: TVD=0.0081, 5 -> 5 categories, missing 4% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.0077, 7 -> 7 categories, missing 6% -> 0%
- `encounters_admissionYear`: TVD=0.0072, 10 -> 10 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.0066, 6 -> 6 categories, missing 11% -> 0%
- `encounter_primary_reason_HF_Disease_f5a_w7d_first`: TVD=0.006, 3 -> 3 categories, missing 90% -> 0%

## preprocessed vs synthetic[tvae]

Worst numeric columns (by KS):
- `conditions_heartFailure_timeFromEarliest_first`: KS=0.5376, W/std=0.1812, mean 11.3756 -> 14.2919, missing 0% -> 27%
- `lab_results_creatUS_value_last`: KS=0.5117, W/std=0.6927, mean 678.5222 -> 382.0103, missing 90% -> 99%
- `lab_results_tropTHs_value_last`: KS=0.4857, W/std=0.1812, mean 0.4651 -> 0.2611, missing 63% -> 64%
- `lab_results_ferritin_value_first`: KS=0.429, W/std=0.1686, mean 522.5048 -> 465.1138, missing 80% -> 90%
- `lab_results_ferritin_value_last`: KS=0.3772, W/std=0.1593, mean 493.6046 -> 434.7788, missing 80% -> 90%
Worst categorical columns (by TVD):
- `ckd_severity_from_calculated_egfr`: TVD=0.2863, 6 -> 6 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.2825, 7 -> 7 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.2286, 10 -> 10 categories, missing 0% -> 0%
- `med_bb`: TVD=0.2201, 2 -> 2 categories, missing 0% -> 0%
- `conditions_cm`: TVD=0.1868, 2 -> 2 categories, missing 0% -> 0%

## original vs synthetic[tvae]

Worst numeric columns (by KS):
- `conditions_heartFailure_timeFromEarliest_first`: KS=0.5376, W/std=0.1812, mean 11.3756 -> 14.2919, missing 0% -> 27%
- `lab_results_creatUS_value_last`: KS=0.5117, W/std=0.6927, mean 678.5222 -> 382.0103, missing 90% -> 99%
- `lab_results_tropTHs_value_last`: KS=0.4857, W/std=0.1812, mean 0.4651 -> 0.2611, missing 63% -> 64%
- `lab_results_ferritin_value_first`: KS=0.429, W/std=0.1686, mean 522.5048 -> 465.1138, missing 80% -> 90%
- `lab_results_ferritin_value_last`: KS=0.3772, W/std=0.1593, mean 493.6046 -> 434.7788, missing 80% -> 90%
Worst categorical columns (by TVD):
- `ckd_severity_from_calculated_egfr`: TVD=0.2863, 6 -> 6 categories, missing 11% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.2825, 7 -> 7 categories, missing 1% -> 0%
- `encounters_admissionYear`: TVD=0.2286, 10 -> 10 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.1685, 7 -> 7 categories, missing 6% -> 0%
- `beta_blocker_use_pre_dc`: TVD=0.1623, 2 -> 2 categories, missing 0% -> 0%
