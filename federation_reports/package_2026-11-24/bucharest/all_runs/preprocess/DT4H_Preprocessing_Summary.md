# Preprocessing Summary

- Input: 6274 rows x 540 columns (6274 unique patients)
- Output: 6274 rows x 229 columns
- Remaining missing cells: 0 null, 0 NaN
- Holdout split (seed 0): 4706 train / 1568 holdout rows (25% held out, never seen by any generator)

## Metadata validation
- 540 / 540 declared columns matched in data

## Expected non-null pair checks
- ⚠️ lab_results_hdl_value_first: 0 vs lab_results_ldl_value_first: 0 (expected similar (ordered together))
- lab_results_potassium_value_first: 6252 vs lab_results_sodium_value_first: 6248 (expected similar (ordered together))
- ⚠️ lab_results_albuminBS_value_first: 0 vs lab_results_ntProBnp_value_first: 0 (expected similar)
- ⚠️ lab_results_albuminBS_value_first: 0 vs lab_results_crpNonHs_value_first: 0 (expected similar)
- ⚠️ lab_results_albuminBS_value_first: 0 vs lab_results_hba1c_value_first: 0 (expected similar)
- vital_signs_heartRate_value_first: 3769 vs vital_signs_oxygenSaturation_value_first: 3310 (expected similar, oxygen sat maybe slightly lower)

## Transformations
- Out-of-domain invalidation: 18 cell(s) outside their declared public range -> null (invalid measurement): {'vital_signs_systolicBp_value_first': 7, 'vital_signs_systolicBp_value_last': 5, 'lab_results_triGly_value_last': 1, 'lab_results_triGly_value_first': 1, 'lab_results_cholTot_value_last': 1, 'lab_results_potassium_value_last': 3}
- ARRAY[NOMINAL] columns flattened: ['electrocardiographs_ecg_type_of_rhythms_pET_first', 'electrocardiographs_ecg_type_of_rhythms_pET_last']
- Symptom columns: 30 present, 30 currently constant, kept (not dropped)
- Medications combined into 58 feature(s) (from 114 source columns)
- Conditions combined into 31 feature(s) (from 78 source columns)
- NYHA encoding: {'LA28404-4': 1, 'LA28405-1': 2, 'LA28406-9': 3, 'LA28407-7': 4}
- Numeric aggregate columns dropped (bare/_min/_max/_avg/_stddev): 140
- IDENTIFIER/DATETIME columns dropped: ['pid', 'encounterId', 'eventTime', 'exitTime', 'referenceTimePoint', 'encounters_admissionDate', 'encounters_dischargeDate']
- Near-unique identifier-like columns dropped (safety net, not caught by declared type): ['patient_demographics_sourceIdentifier']
- Decimal columns cast to Float64: ['vital_signs_weight_value_p6mo_max', 'vital_signs_weight_value_p6mo_min', 'vital_signs_weight_value_p6mo_first', 'vital_signs_weight_value_p6mo_avg', 'vital_signs_weight_value_p6mo_last', 'vital_signs_height_value_p1a_avg', 'vital_signs_weight_value_last', 'vital_signs_height_value_last', 'vital_signs_diastolicBp_value_first', 'vital_signs_diastolicBp_value_min', 'vital_signs_diastolicBp_value_max', 'vital_signs_diastolicBp_value_last', 'vital_signs_diastolicBp_value_avg', 'vital_signs_heartRate_value_first', 'vital_signs_heartRate_value_min', 'vital_signs_heartRate_value_max', 'vital_signs_heartRate_value_last', 'vital_signs_heartRate_value_avg', 'vital_signs_systolicBp_value_first', 'vital_signs_systolicBp_value_min', 'vital_signs_systolicBp_value_max', 'vital_signs_systolicBp_value_last', 'vital_signs_systolicBp_value_avg', 'vital_signs_oxygenSaturation_value_avg', 'vital_signs_oxygenSaturation_value_max', 'vital_signs_oxygenSaturation_value_last', 'vital_signs_oxygenSaturation_value_first', 'vital_signs_oxygenSaturation_value_min', 'lab_results_hemoglobin_value_first', 'lab_results_hemoglobin_value_avg', 'lab_results_hemoglobin_value_min', 'lab_results_hemoglobin_value_max', 'lab_results_hemoglobin_value_last', 'lab_results_ferritin_value_first', 'lab_results_ferritin_value_avg', 'lab_results_ferritin_value_min', 'lab_results_ferritin_value_max', 'lab_results_ferritin_value_last', 'lab_results_tfs_value_first', 'lab_results_tfs_value_avg', 'lab_results_tfs_value_min', 'lab_results_tfs_value_max', 'lab_results_tfs_value_last', 'lab_results_ntProBnp_value_first', 'lab_results_ntProBnp_value_avg', 'lab_results_ntProBnp_value_min', 'lab_results_ntProBnp_value_max', 'lab_results_ntProBnp_value_last', 'lab_results_bnp_value_first', 'lab_results_bnp_value_avg', 'lab_results_bnp_value_min', 'lab_results_bnp_value_max', 'lab_results_bnp_value_last', 'lab_results_crpNonHs_value_first', 'lab_results_crpNonHs_value_avg', 'lab_results_crpNonHs_value_min', 'lab_results_crpNonHs_value_max', 'lab_results_crpNonHs_value_last', 'lab_results_crpHs_value_first', 'lab_results_crpHs_value_avg', 'lab_results_crpHs_value_min', 'lab_results_crpHs_value_max', 'lab_results_crpHs_value_last', 'lab_results_tropIHs_value_first', 'lab_results_tropIHs_value_avg', 'lab_results_tropIHs_value_min', 'lab_results_tropIHs_value_max', 'lab_results_tropIHs_value_last', 'lab_results_tropInHs_value_first', 'lab_results_tropInHs_value_avg', 'lab_results_tropInHs_value_min', 'lab_results_tropInHs_value_max', 'lab_results_tropInHs_value_last', 'lab_results_tropTHs_value_first', 'lab_results_tropTHs_value_avg', 'lab_results_tropTHs_value_min', 'lab_results_tropTHs_value_max', 'lab_results_tropTHs_value_last', 'lab_results_tropTnHs_value_first', 'lab_results_tropTnHs_value_avg', 'lab_results_tropTnHs_value_min', 'lab_results_tropTnHs_value_max', 'lab_results_tropTnHs_value_last', 'lab_results_triGly_value_first', 'lab_results_triGly_value_avg', 'lab_results_triGly_value_min', 'lab_results_triGly_value_max', 'lab_results_triGly_value_last', 'lab_results_cholTot_value_first', 'lab_results_cholTot_value_avg', 'lab_results_cholTot_value_min', 'lab_results_cholTot_value_max', 'lab_results_cholTot_value_last', 'lab_results_hdl_value_first', 'lab_results_hdl_value_avg', 'lab_results_hdl_value_min', 'lab_results_hdl_value_max', 'lab_results_hdl_value_last', 'lab_results_creatUS_value_first', 'lab_results_creatUS_value_avg', 'lab_results_creatUS_value_min', 'lab_results_creatUS_value_max', 'lab_results_creatUS_value_last', 'lab_results_albuminUS_value_first', 'lab_results_albuminUS_value_avg', 'lab_results_albuminUS_value_min', 'lab_results_albuminUS_value_max', 'lab_results_albuminUS_value_last', 'lab_results_bun_value_first', 'lab_results_bun_value_avg', 'lab_results_bun_value_min', 'lab_results_bun_value_max', 'lab_results_bun_value_last', 'lab_results_acr_value_first', 'lab_results_acr_value_avg', 'lab_results_acr_value_min', 'lab_results_acr_value_max', 'lab_results_acr_value_last', 'lab_results_ldl_value_avg', 'lab_results_ldl_value_max', 'lab_results_ldl_value_last', 'lab_results_ldl_value_min', 'lab_results_ldl_value_first', 'lab_results_potassium_value_first', 'lab_results_potassium_value_max', 'lab_results_potassium_value_avg', 'lab_results_potassium_value_min', 'lab_results_potassium_value_last', 'lab_results_sodium_value_first', 'lab_results_sodium_value_min', 'lab_results_sodium_value_last', 'lab_results_sodium_value_max', 'lab_results_sodium_value_avg', 'lab_results_albuminBS_value_min', 'lab_results_albuminBS_value_avg', 'lab_results_albuminBS_value_first', 'lab_results_albuminBS_value_max', 'lab_results_albuminBS_value_last', 'lab_results_hba1c%_value_avg', 'lab_results_hba1c%_value_first', 'lab_results_hba1c%_value_max', 'lab_results_hba1c%_value_last', 'lab_results_hba1c%_value_min', 'lab_results_hba1c_value_first', 'lab_results_hba1c_value_max', 'lab_results_hba1c_value_avg', 'lab_results_hba1c_value_min', 'lab_results_hba1c_value_last', 'lab_results_validSerumCreatinine_value_first', 'lab_results_validSerumCreatinine_value_min', 'lab_results_validSerumCreatinine_value_max', 'lab_results_validSerumCreatinine_value_avg', 'lab_results_valideGFR_value_min', 'lab_results_valideGFR_value_first', 'lab_results_valideGFR_value_max', 'lab_results_valideGFR_value_last', 'lab_results_valideGFR_value_avg', 'echocardiographs_lvef', 'echocardiographs_lvef_pET_min', 'echocardiographs_lvef_pET_max', 'echocardiographs_lvef_pET_last', 'echocardiographs_lvef_pET_first', 'echocardiographs_lvef_pET_avg', 'electrocardiographs_ecg_qrs_duration_pET_first', 'electrocardiographs_ecg_qrs_duration_pET_avg', 'electrocardiographs_ecg_qrs_duration_pET_min', 'electrocardiographs_ecg_qrs_duration_pET_max', 'electrocardiographs_ecg_qrs_duration_pET_last', 'electrocardiographs_ecg_qrs_axis_pET_max', 'electrocardiographs_ecg_qrs_axis_pET_avg', 'electrocardiographs_ecg_qrs_axis_pET_first', 'electrocardiographs_ecg_qrs_axis_pET_last', 'electrocardiographs_ecg_qrs_axis_pET_min', 'electrocardiographs_ecg_qt_duration_corrected_pET_max', 'electrocardiographs_ecg_qt_duration_corrected_pET_min', 'electrocardiographs_ecg_qt_duration_corrected_pET_first', 'electrocardiographs_ecg_qt_duration_corrected_pET_avg', 'electrocardiographs_ecg_qt_duration_corrected_pET_last', 'vital_signs_systolicBpDuringEncounter_value_pET', 'vital_signs_bmi_value_pET', 'lab_results_creatBS_value_p3a_avg', 'lab_results_validSerumCreatinine_value_pET', 'eGFR_2021_ckd_epi_creatinine']

## Final null cleanup
- NYHA: filled 0 missing value(s) with sentinel 0
- Numeric nulls are NOT imputed -- missingness carries meaning. 23 column(s) sentinel-encoded (0 time-to-event 'no event', 23 'not measured'), each with a per-column sentinel below the observed range, decoded back to null in the synthetic output (map: `/home/translated/cardioData/SyntheticDataGenerationTabularDT4H/output/preprocess/DT4H_Numeric_Missing_Encoding.json`).
- Dropped 60 numeric column(s) with fewer than 313 observed values:
  - `vital_signs_weight_value_p6mo_first` (only 0 observed)
  - `vital_signs_weight_value_p6mo_last` (only 0 observed)
  - `vital_signs_height_value_p1a_avg` (only 0 observed)
  - `vital_signs_weight_value_last` (only 0 observed)
  - `vital_signs_height_value_last` (only 0 observed)
  - `lab_results_ferritin_value_first` (only 0 observed)
  - `lab_results_ferritin_value_last` (only 0 observed)
  - `lab_results_tfs_value_first` (only 0 observed)
  - `lab_results_tfs_value_last` (only 0 observed)
  - `lab_results_ntProBnp_value_first` (only 0 observed)
  - `lab_results_ntProBnp_value_last` (only 0 observed)
  - `lab_results_bnp_value_first` (only 0 observed)
  - `lab_results_bnp_value_last` (only 0 observed)
  - `lab_results_crpNonHs_value_first` (only 0 observed)
  - `lab_results_crpNonHs_value_last` (only 0 observed)
  - `lab_results_crpHs_value_first` (only 0 observed)
  - `lab_results_crpHs_value_last` (only 0 observed)
  - `lab_results_tropIHs_value_first` (only 196 observed)
  - `lab_results_tropIHs_value_last` (only 196 observed)
  - `lab_results_tropInHs_value_first` (only 0 observed)
  - `lab_results_tropInHs_value_last` (only 0 observed)
  - `lab_results_tropTHs_value_first` (only 0 observed)
  - `lab_results_tropTHs_value_last` (only 0 observed)
  - `lab_results_tropTnHs_value_first` (only 0 observed)
  - `lab_results_tropTnHs_value_last` (only 0 observed)
  - `lab_results_hdl_value_first` (only 0 observed)
  - `lab_results_hdl_value_last` (only 0 observed)
  - `lab_results_creatUS_value_first` (only 0 observed)
  - `lab_results_creatUS_value_last` (only 0 observed)
  - `lab_results_albuminUS_value_first` (only 0 observed)
  - `lab_results_albuminUS_value_last` (only 0 observed)
  - `lab_results_bun_value_first` (only 0 observed)
  - `lab_results_bun_value_last` (only 0 observed)
  - `lab_results_acr_value_first` (only 0 observed)
  - `lab_results_acr_value_last` (only 0 observed)
  - `lab_results_ldl_value_first` (only 0 observed)
  - `lab_results_ldl_value_last` (only 0 observed)
  - `lab_results_albuminBS_value_first` (only 0 observed)
  - `lab_results_albuminBS_value_last` (only 0 observed)
  - `lab_results_hba1c%_value_first` (only 0 observed)
  - `lab_results_hba1c%_value_last` (only 0 observed)
  - `lab_results_hba1c_value_first` (only 0 observed)
  - `lab_results_hba1c_value_last` (only 0 observed)
  - `lab_results_valideGFR_value_first` (only 0 observed)
  - `lab_results_valideGFR_value_last` (only 0 observed)
  - `electrocardiographs_ecg_qrs_duration_pET_first` (only 0 observed)
  - `electrocardiographs_ecg_qrs_duration_pET_last` (only 0 observed)
  - `electrocardiographs_ecg_qrs_axis_pET_last` (only 0 observed)
  - `electrocardiographs_ecg_qrs_axis_pET_first` (only 0 observed)
  - `electrocardiographs_ecg_qt_duration_corrected_pET_last` (only 0 observed)
  - `electrocardiographs_ecg_qt_duration_corrected_pET_first` (only 0 observed)
  - `smoking_status_smoker_totalSmokingDuration_sum` (only 0 observed)
  - `encounter_primary_reason_number_of_days_to_rehosp_for_heart_failure_f5a_first` (only 3 observed)
  - `encounter_primary_reason_number_of_days_to_rehosp_for_CV_f5a_first` (only 96 observed)
  - `encounter_primary_reason_number_of_days_to_rehosp_for_non_CV_f5a_first` (only 71 observed)
  - `encounter_primary_reason_number_of_days_to_rehosp_for_renal_complications_f5a_first` (only 0 observed)
  - `cause_of_death_number_of_days_to_death_for_CV_f5a_first` (only 0 observed)
  - `cause_of_death_number_of_days_to_death_for_renal_f5a_first` (only 0 observed)
  - `cause_of_death_number_of_days_to_death_for_non_renal_and_non_CV_f5a_first` (only 0 observed)
  - `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first` (only 1 observed)
- Categorical/boolean: normalized 190 column(s) to String; 63 of them had nulls filled with an explicit 'Missing' category
