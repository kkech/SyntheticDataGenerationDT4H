# Evaluation: original vs preprocessed vs synthetic

Metrics are computed per column over observed values (nulls excluded); missingness rates are compared separately. KS and TVD are in [0,1], lower is closer; `W/std` is the Wasserstein distance in units of the reference standard deviation.

| comparison | cols | KS mean | KS median | KS<0.1 | W/std mean | TVD mean | TVD<0.05 | missing-rate MAD |
|---|---|---|---|---|---|---|---|---|
| original vs preprocessed | 164 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 1.0 | 0.0 |
| preprocessed vs synthetic[ctgan] | 249 | 0.3913 | 0.3902 | 0.0328 | 0.7793 | 0.0597 | 0.5957 | 0.0747 |
| original vs synthetic[ctgan] | 164 | 0.3913 | 0.3902 | 0.0328 | 0.7793 | 0.0426 | 0.699 | 0.0747 |
| preprocessed vs synthetic[gaussian_copula] | 248 | 0.5243 | 0.4798 | 0.0667 | 1.5988 | 0.0054 | 0.9947 | 0.1142 |
| original vs synthetic[gaussian_copula] | 163 | 0.5243 | 0.4798 | 0.0667 | 1.5988 | 0.0066 | 0.9903 | 0.1142 |
| preprocessed vs synthetic[mst] | 235 | 0.3881 | 0.2689 | 0.0 | 0.4663 | 0.0016 | 1.0 | 0.0265 |
| original vs synthetic[mst] | 150 | 0.3881 | 0.2689 | 0.0 | 0.4663 | 0.0016 | 1.0 | 0.0265 |
| preprocessed vs synthetic[tvae] | 249 | 0.2097 | 0.1902 | 0.2131 | 0.312 | 0.0487 | 0.5957 | 0.0462 |
| original vs synthetic[tvae] | 164 | 0.2097 | 0.1902 | 0.2131 | 0.312 | 0.042 | 0.6602 | 0.0462 |

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
- `nyha_nyha_pET`: KS=0.899, W/std=2.6716, mean 2.4579 -> 0.2501, missing 75% -> 0%
- `lab_results_crpNonHs_value_first`: KS=0.7346, W/std=0.6178, mean 45.4252 -> 1.6987, missing 12% -> 8%
- `lab_results_tropTHs_value_first`: KS=0.7045, W/std=0.3264, mean 0.2243 -> -0.0684, missing 63% -> 72%
- `encounter_primary_reason_number_of_days_to_rehosp_for_CV_f5a_first`: KS=0.6904, W/std=0.5494, mean 119.1399 -> -4.9861, missing 80% -> 82%
- `lab_results_albuminBS_value_last`: KS=0.6743, W/std=2.1241, mean 29.985 -> 15.3451, missing 57% -> 60%
Worst categorical columns (by TVD):
- `conditions_vd`: TVD=0.2714, 2 -> 2 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.2657, 10 -> 10 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.2495, 7 -> 7 categories, missing 0% -> 0%
- `med_inotropes_history`: TVD=0.2484, 2 -> 2 categories, missing 0% -> 0%
- `med_insulins`: TVD=0.2467, 2 -> 2 categories, missing 0% -> 0%

## original vs synthetic[ctgan]

Worst numeric columns (by KS):
- `nyha_nyha_pET`: KS=0.899, W/std=2.6716, mean 2.4579 -> 0.2501, missing 75% -> 0%
- `lab_results_crpNonHs_value_first`: KS=0.7346, W/std=0.6178, mean 45.4252 -> 1.6987, missing 12% -> 8%
- `lab_results_tropTHs_value_first`: KS=0.7045, W/std=0.3264, mean 0.2243 -> -0.0684, missing 63% -> 72%
- `encounter_primary_reason_number_of_days_to_rehosp_for_CV_f5a_first`: KS=0.6904, W/std=0.5494, mean 119.1399 -> -4.9861, missing 80% -> 82%
- `lab_results_albuminBS_value_last`: KS=0.6743, W/std=2.1241, mean 29.985 -> 15.3451, missing 57% -> 60%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.2657, 10 -> 10 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.2495, 7 -> 7 categories, missing 1% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.215, 6 -> 6 categories, missing 11% -> 0%
- `conditions_heart_failure_occurred_prior_to_18_months_any`: TVD=0.1958, 2 -> 2 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.1721, 7 -> 7 categories, missing 6% -> 0%

## preprocessed vs synthetic[gaussian_copula]

Worst numeric columns (by KS):
- `lab_results_tropTnHs_value_first`: KS=1.0, W/std=1.5417, mean 211.6467 -> -623.3631, missing 88% -> 100%
- `lab_results_triGly_value_last`: KS=1.0, W/std=2.4877, mean 1.5056 -> -1.0872, missing 91% -> 100%
- `lab_results_triGly_value_first`: KS=1.0, W/std=2.265, mean 1.5481 -> -1.3779, missing 91% -> 100%
- `lab_results_ldl_value_first`: KS=0.9928, W/std=2.7426, mean 2.1159 -> -0.5155, missing 91% -> 99%
- `lab_results_hdl_value_first`: KS=0.9785, W/std=2.9503, mean 1.1884 -> -0.1361, missing 91% -> 99%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.1741, 10 -> 9 categories, missing 0% -> 0%
- `med_cortico_syst_history`: TVD=0.0219, 2 -> 2 categories, missing 0% -> 0%
- `encounter_primary_reason_HF_Disease_f5a_w6mo_first`: TVD=0.0164, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_CV_Disease_f5a_w1a_first`: TVD=0.016, 3 -> 3 categories, missing 0% -> 0%
- `cause_of_death_isRenal_f5a_w3a_first`: TVD=0.0158, 2 -> 2 categories, missing 0% -> 0%

## original vs synthetic[gaussian_copula]

Worst numeric columns (by KS):
- `lab_results_tropTnHs_value_first`: KS=1.0, W/std=1.5417, mean 211.6467 -> -623.3631, missing 88% -> 100%
- `lab_results_triGly_value_last`: KS=1.0, W/std=2.4877, mean 1.5056 -> -1.0872, missing 91% -> 100%
- `lab_results_triGly_value_first`: KS=1.0, W/std=2.265, mean 1.5481 -> -1.3779, missing 91% -> 100%
- `lab_results_ldl_value_first`: KS=0.9928, W/std=2.7426, mean 2.1159 -> -0.5155, missing 91% -> 99%
- `lab_results_hdl_value_first`: KS=0.9785, W/std=2.9503, mean 1.1884 -> -0.1361, missing 91% -> 99%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.1741, 10 -> 9 categories, missing 0% -> 0%
- `encounter_primary_reason_HF_Disease_f5a_w6mo_first`: TVD=0.0164, 3 -> 3 categories, missing 53% -> 0%
- `encounter_primary_reason_CV_Disease_f5a_w1a_first`: TVD=0.016, 3 -> 3 categories, missing 49% -> 0%
- `cause_of_death_isRenal_f5a_w3a_first`: TVD=0.0158, 2 -> 2 categories, missing 63% -> 0%
- `encounter_primary_reason_renal_complications_f5a_w3a_first`: TVD=0.0151, 3 -> 3 categories, missing 44% -> 0%

## preprocessed vs synthetic[mst]

Worst numeric columns (by KS):
- `lab_results_tropTHs_value_last`: KS=1.0, W/std=0.4783, mean 0.4651 -> -0.65, missing 63% -> 63%
- `lab_results_hdl_value_last`: KS=1.0, W/std=2.8104, mean 1.1871 -> -0.051, missing 91% -> 91%
- `lab_results_hdl_value_first`: KS=1.0, W/std=2.7591, mean 1.1884 -> -0.0502, missing 91% -> 91%
- `smoking_status_smoker_startTime_count`: KS=0.9823, W/std=0.1346, mean 0.0807 -> 0.066, missing 0% -> 0%
- `nyha_nyha_pET`: KS=0.7507, W/std=2.0298, mean 2.4579 -> 0.7805, missing 75% -> 0%
Worst categorical columns (by TVD):
- `hyperkalemia_severity_categorizedValue`: TVD=0.0081, 5 -> 5 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.0077, 7 -> 7 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.0072, 10 -> 10 categories, missing 0% -> 0%
- `med_anti_coag`: TVD=0.007, 2 -> 2 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.0066, 6 -> 6 categories, missing 0% -> 0%

## original vs synthetic[mst]

Worst numeric columns (by KS):
- `lab_results_tropTHs_value_last`: KS=1.0, W/std=0.4783, mean 0.4651 -> -0.65, missing 63% -> 63%
- `lab_results_hdl_value_last`: KS=1.0, W/std=2.8104, mean 1.1871 -> -0.051, missing 91% -> 91%
- `lab_results_hdl_value_first`: KS=1.0, W/std=2.7591, mean 1.1884 -> -0.0502, missing 91% -> 91%
- `smoking_status_smoker_startTime_count`: KS=0.9823, W/std=0.1346, mean 0.0807 -> 0.066, missing 0% -> 0%
- `nyha_nyha_pET`: KS=0.7507, W/std=2.0298, mean 2.4579 -> 0.7805, missing 75% -> 0%
Worst categorical columns (by TVD):
- `hyperkalemia_severity_categorizedValue`: TVD=0.0081, 5 -> 5 categories, missing 4% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.0077, 7 -> 7 categories, missing 6% -> 0%
- `encounters_admissionYear`: TVD=0.0072, 10 -> 10 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.0066, 6 -> 6 categories, missing 11% -> 0%
- `encounter_primary_reason_HF_Disease_f5a_w7d_first`: TVD=0.006, 3 -> 3 categories, missing 90% -> 0%

## preprocessed vs synthetic[tvae]

Worst numeric columns (by KS):
- `nyha_nyha_pET`: KS=0.9578, W/std=2.8587, mean 2.4579 -> 0.0954, missing 75% -> 0%
- `lab_results_creatUS_value_last`: KS=0.5117, W/std=0.6927, mean 678.5222 -> 382.0103, missing 90% -> 99%
- `lab_results_ferritin_value_first`: KS=0.429, W/std=0.1686, mean 522.5048 -> 465.1138, missing 80% -> 90%
- `lab_results_ferritin_value_last`: KS=0.3735, W/std=0.1586, mean 493.6046 -> 432.8454, missing 80% -> 90%
- `lab_results_tropTHs_value_last`: KS=0.3644, W/std=0.1749, mean 0.4651 -> 0.2091, missing 63% -> 57%
Worst categorical columns (by TVD):
- `ckd_severity_from_calculated_egfr`: TVD=0.2863, 6 -> 6 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.2825, 7 -> 7 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.2286, 10 -> 10 categories, missing 0% -> 0%
- `med_bb`: TVD=0.2201, 2 -> 2 categories, missing 0% -> 0%
- `conditions_cm`: TVD=0.1868, 2 -> 2 categories, missing 0% -> 0%

## original vs synthetic[tvae]

Worst numeric columns (by KS):
- `nyha_nyha_pET`: KS=0.9578, W/std=2.8587, mean 2.4579 -> 0.0954, missing 75% -> 0%
- `lab_results_creatUS_value_last`: KS=0.5117, W/std=0.6927, mean 678.5222 -> 382.0103, missing 90% -> 99%
- `lab_results_ferritin_value_first`: KS=0.429, W/std=0.1686, mean 522.5048 -> 465.1138, missing 80% -> 90%
- `lab_results_ferritin_value_last`: KS=0.3735, W/std=0.1586, mean 493.6046 -> 432.8454, missing 80% -> 90%
- `lab_results_tropTHs_value_last`: KS=0.3644, W/std=0.1749, mean 0.4651 -> 0.2091, missing 63% -> 57%
Worst categorical columns (by TVD):
- `ckd_severity_from_calculated_egfr`: TVD=0.2863, 6 -> 6 categories, missing 11% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.2825, 7 -> 7 categories, missing 1% -> 0%
- `encounters_admissionYear`: TVD=0.2286, 10 -> 10 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.1685, 7 -> 7 categories, missing 6% -> 0%
- `beta_blocker_use_pre_dc`: TVD=0.1623, 2 -> 2 categories, missing 0% -> 0%
