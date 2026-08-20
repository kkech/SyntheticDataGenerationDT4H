# Preprocessing Summary

- Input: 4694 rows x 528 columns (4694 unique patients)
- Output: 4694 rows x 312 columns
- Remaining null cells: 0

## Metadata validation
- 528 / 528 declared columns matched in data

## Expected non-null pair checks
- lab_results_hdl_value_first: 418 vs lab_results_ldl_value_first: 414 (expected similar (ordered together))
- lab_results_potassium_value_first: 4497 vs lab_results_sodium_value_first: 4494 (expected similar (ordered together))
- ⚠️ lab_results_albuminBS_value_first: 1995 vs lab_results_ntProBnp_value_first: 3787 (expected similar)
- ⚠️ lab_results_albuminBS_value_first: 1995 vs lab_results_crpNonHs_value_first: 4152 (expected similar)
- ⚠️ lab_results_albuminBS_value_first: 1995 vs lab_results_hba1c_value_first: 0 (expected similar)
- ⚠️ vital_signs_heartRate_value_first: 2436 vs vital_signs_oxygenSaturation_value_first: 0 (expected similar, oxygen sat maybe slightly lower)

## Transformations
- ARRAY[NOMINAL] columns flattened: ['electrocardiographs_ecg_type_of_rhythms_pET_first', 'electrocardiographs_ecg_type_of_rhythms_pET_last']
- Symptom columns: 30 present, 30 currently constant, kept (not dropped)
- Medications combined into 54 feature(s) (from 108 source columns)
- Conditions combined into 31 feature(s) (from 78 source columns)
- NYHA encoding: {'LA28404-4': 1, 'LA28405-1': 2, 'LA28406-9': 3, 'LA28407-7': 4}
- Numeric aggregate columns dropped (bare/_min/_max/_avg/_stddev): 140
- IDENTIFIER/DATETIME columns dropped: ['pid', 'encounterId', 'eventTime', 'exitTime', 'referenceTimePoint', 'encounters_admissionDate', 'encounters_dischargeDate']

## Dummy imputation (Machteld's temporary placeholder rules)
- Filled 1995 value(s) in `lab_results_hba1c%_value_first` (triggered by `lab_results_albuminBS_value` present)
- Filled 1995 value(s) in `lab_results_hba1c%_value_last` (triggered by `lab_results_albuminBS_value` present)
- Filled 1995 value(s) in `lab_results_hba1c_value_first` (triggered by `lab_results_albuminBS_value` present)
- Filled 1995 value(s) in `lab_results_hba1c_value_last` (triggered by `lab_results_albuminBS_value` present)
- Filled 277 value(s) in `lab_results_ntProBnp_value_first` (triggered by `lab_results_albuminBS_value` present)
- Filled 277 value(s) in `lab_results_ntProBnp_value_last` (triggered by `lab_results_albuminBS_value` present)
- Filled 114 value(s) in `lab_results_crpNonHs_value_first` (triggered by `lab_results_albuminBS_value` present)
- Filled 114 value(s) in `lab_results_crpNonHs_value_last` (triggered by `lab_results_albuminBS_value` present)
- Filled 391 value(s) in `lab_results_tropTnHs_value_first` (triggered by `lab_results_hdl_value` present)
- Filled 391 value(s) in `lab_results_tropTnHs_value_last` (triggered by `lab_results_hdl_value` present)
- Filled 4 value(s) in `lab_results_ldl_value_first` (triggered by `lab_results_hdl_value` present)
- Filled 4 value(s) in `lab_results_ldl_value_last` (triggered by `lab_results_hdl_value` present)
- Filled 37 value(s) in `lab_results_sodium_value_first` (triggered by `lab_results_potassium_value` present)
- Filled 37 value(s) in `lab_results_sodium_value_last` (triggered by `lab_results_potassium_value` present)
- Filled 2436 value(s) in `vital_signs_oxygenSaturation_value_first` (triggered by `vital_signs_heartRate_value` present)
- Filled 2436 value(s) in `vital_signs_oxygenSaturation_value_last` (triggered by `vital_signs_heartRate_value` present)
- Filled 2385 value(s) in `electrocardiographs_ecg_qrs_axis_pET_first` (triggered by `electrocardiographs_ecg_qrs_duration_pET` present)
- Filled 2385 value(s) in `electrocardiographs_ecg_qrs_axis_pET_last` (triggered by `electrocardiographs_ecg_qrs_duration_pET` present)
- (skipped) {'reason': 'target not found', 'trigger': 'lab_results_albuminBS_value', 'target': 'lab_results_glucose_value'}

## Final null cleanup
- NYHA: filled 3530 missing value(s) with sentinel 0
- Numeric: imputed 50 column(s) (bootstrap from observed values), added 50 '_was_missing' flag(s), dropped 18 column(s) with too few observed values
  - dropped: `lab_results_tfs_value_first`
  - dropped: `lab_results_tfs_value_last`
  - dropped: `lab_results_bnp_value_first`
  - dropped: `lab_results_bnp_value_last`
  - dropped: `lab_results_crpHs_value_first`
  - dropped: `lab_results_crpHs_value_last`
  - dropped: `lab_results_tropIHs_value_first`
  - dropped: `lab_results_tropIHs_value_last`
  - dropped: `lab_results_tropInHs_value_first`
  - dropped: `lab_results_tropInHs_value_last`
  - dropped: `lab_results_albuminUS_value_first`
  - dropped: `lab_results_albuminUS_value_last`
  - dropped: `lab_results_bun_value_first`
  - dropped: `lab_results_bun_value_last`
  - dropped: `lab_results_acr_value_first`
  - dropped: `lab_results_acr_value_last`
  - dropped: `cause_of_death_number_of_days_to_death_for_renal_f5a_first`
  - dropped: `cause_of_death_number_of_days_to_death_for_non_renal_and_non_CV_f5a_first`
- Categorical/boolean: filled 66 column(s) with explicit 'Missing' category
