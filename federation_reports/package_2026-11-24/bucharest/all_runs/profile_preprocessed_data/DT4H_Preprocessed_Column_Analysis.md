# Column Analysis

Total rows: 6274
Total columns: 229

## patient_demographics_gender

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `male`: 3958
  - `female`: 2316

## patient_demographics_age

- dtype: `Int32` (numeric)
- nulls: 0 (0.00%)
- min/max: 19.0 / 98.0
- mean/std: 67.3738 / 12.976787831576617
- quantiles: {'0.05': 44.0, '0.25': 59.0, '0.5': 68.0, '0.75': 76.0, '0.95': 87.0}

## encounters_encounterClass

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `IMP`: 6274

## encounters_admissionYear

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 6
- top values (shown only where count ≥ 5):
  - `2024`: 1491
  - `2023`: 1424
  - `2022`: 1313
  - `2021`: 1127
  - `2020`: 893
  - `2019`: 26

## encounters_lengthOfStay

- dtype: `Int32` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 130.0
- mean/std: 6.7018 / 7.2008049925318165
- quantiles: {'0.05': 1.0, '0.25': 2.0, '0.5': 5.0, '0.75': 9.0, '0.95': 18.0}

## encounters_numOfPreviousHFStays_count

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 10.0
- mean/std: 0.2246 / 0.6500279127654152
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 1.0}

## vital_signs_diastolicBp_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -42.0 / 200.0
- mean/std: 53.0892 / 49.67009604341682
- quantiles: {'0.05': -41.25, '0.25': 60.0, '0.5': 70.0, '0.75': 80.0, '0.95': 100.0}

## vital_signs_diastolicBp_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -42.0 / 200.0
- mean/std: 53.0447 / 49.63118933951972
- quantiles: {'0.05': -41.25, '0.25': 60.0, '0.5': 70.0, '0.75': 80.0, '0.95': 100.0}

## vital_signs_heartRate_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -58.0 / 250.0
- mean/std: 27.0240 / 71.76437084414742
- quantiles: {'0.05': -57.5, '0.25': -57.5, '0.5': 60.0, '0.75': 80.0, '0.95': 122.69999999999891}

## vital_signs_heartRate_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -45.0 / 200.0
- mean/std: 31.8852 / 65.84624227757124
- quantiles: {'0.05': -45.0, '0.25': -45.0, '0.5': 60.0, '0.75': 80.0, '0.95': 125.0}

## vital_signs_systolicBp_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -32.0 / 280.0
- mean/std: 100.2737 / 70.4658154512298
- quantiles: {'0.05': -31.5, '0.25': 100.0, '0.5': 124.0, '0.75': 140.0, '0.95': 180.0}

## vital_signs_systolicBp_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -5.0 / 270.0
- mean/std: 105.5773 / 60.4589875626723
- quantiles: {'0.05': -5.0, '0.25': 100.0, '0.5': 124.0, '0.75': 140.0, '0.95': 180.0}

## vital_signs_oxygenSaturation_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -17.0 / 100.0
- mean/std: 42.7384 / 55.94598059199665
- quantiles: {'0.05': -16.25, '0.25': -16.25, '0.5': 88.0, '0.75': 97.0, '0.95': 98.0}

## vital_signs_oxygenSaturation_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -17.0 / 100.0
- mean/std: 42.7769 / 55.97677205754241
- quantiles: {'0.05': -16.25, '0.25': -16.25, '0.5': 88.0, '0.75': 97.0, '0.95': 98.0}

## lab_results_hemoglobin_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -1.3 / 210.0
- mean/std: 132.6283 / 25.106607333626314
- quantiles: {'0.05': 92.0, '0.25': 119.0, '0.5': 135.55, '0.75': 149.375, '0.95': 168.0}

## lab_results_hemoglobin_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -1.5 / 210.0
- mean/std: 128.5880 / 23.663608136724665
- quantiles: {'0.05': 90.0, '0.25': 114.525, '0.5': 131.0, '0.75': 144.0, '0.95': 162.03499999999994}

## lab_results_triGly_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -4.0 / 17.0
- mean/std: 1.0066 / 1.797465251795757
- quantiles: {'0.05': -3.9462425000000008, '0.25': 0.8022999999999999, '0.5': 1.1726575000000001, '0.75': 1.6724, '0.95': 2.9719}

## lab_results_triGly_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -5.0 / 20.0
- mean/std: 0.8614 / 2.0163698159352275
- quantiles: {'0.05': -4.9155, '0.25': 0.7797, '0.5': 1.1074, '0.75': 1.5933, '0.95': 2.8701999999999996}

## lab_results_cholTot_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -1.8 / 13.0
- mean/std: 3.8489 / 1.972799469169995
- quantiles: {'0.05': -1.7094, '0.25': 3.0969925, '0.5': 3.9575199999999997, '0.75': 4.9727999999999986, '0.95': 6.6304}

## lab_results_cholTot_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -2.8 / 14.0
- mean/std: 3.6865 / 2.1520749407137414
- quantiles: {'0.05': -2.7454000000000005, '0.25': 3.0303, '0.5': 3.8591, '0.75': 4.8413575, '0.95': 6.505302999999997}

## lab_results_potassium_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.13 / 10.0
- mean/std: 4.3458 / 0.6384106871296379
- quantiles: {'0.05': 3.47, '0.25': 4.0, '0.5': 4.32, '0.75': 4.6775, '0.95': 5.32}

## lab_results_potassium_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -0.18 / 9.5
- mean/std: 4.0283 / 0.6318365645112244
- quantiles: {'0.05': 3.08, '0.25': 3.69, '0.5': 4.03, '0.75': 4.4, '0.95': 4.97}

## lab_results_sodium_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 74.0 / 180.0
- mean/std: 139.3903 / 5.786814196304675
- quantiles: {'0.05': 133.2, '0.25': 138.0, '0.5': 140.0, '0.75': 142.0, '0.95': 145.0}

## lab_results_sodium_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 97.0 / 160.0
- mean/std: 137.6088 / 4.770075524705261
- quantiles: {'0.05': 130.1, '0.25': 136.0, '0.5': 138.0, '0.75': 140.0, '0.95': 143.0}

## lab_results_validSerumCreatinine_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -4.9 / 23.0
- mean/std: 10.6068 / 4.033432605635688
- quantiles: {'0.05': 6.3, '0.25': 8.0, '0.5': 9.4, '0.75': 12.3, '0.95': 19.734999999999946}

## symptoms_Ankle_swelling_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## symptoms_Ascites_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## symptoms_Breathlessness_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## symptoms_Cardiac_murmur_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## symptoms_Chest_pain_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## symptoms_Cheyne_stokes_respiration_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## symptoms_Depression_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## symptoms_Dizziness_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## symptoms_Elevated_jugular_venous_pressure_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## symptoms_Fatigue_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## symptoms_Hepatojugular_reflux_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## symptoms_Hepatomegaly_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## symptoms_Intermittent_claudication_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## symptoms_Irregular_pulse_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## symptoms_Loss_of_appetite_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## symptoms_Nocturnal_cough_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## symptoms_Oliguria_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## symptoms_Orthopnoea_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## symptoms_Palpitations_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## symptoms_Paroxysmal_nocturnal_dyspnea_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## symptoms_Peripheral_edema_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## symptoms_Pleural_effusion_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## symptoms_Pulmonary_crepitations_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## symptoms_Reduced_exercise_tolerance_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## symptoms_Syncope_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## symptoms_Tachycardia_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## symptoms_Tachypnoea_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## symptoms_Third_heart_sound_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## symptoms_Weight_gain_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## symptoms_Weight_loss_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## echocardiographs_lvef

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -16.0 / 89.0
- mean/std: 4.0735 / 27.701124954103346
- quantiles: {'0.05': -16.0, '0.25': -16.0, '0.5': -16.0, '0.75': 30.0, '0.95': 55.0}

## echocardiographs_lvef_pET_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -16.0 / 89.0
- mean/std: 3.3253 / 27.51640827718896
- quantiles: {'0.05': -16.0, '0.25': -16.0, '0.5': -16.0, '0.75': 30.0, '0.95': 55.0}

## echocardiographs_lvef_pET_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -16.0 / 89.0
- mean/std: 3.3515 / 27.53190416841347
- quantiles: {'0.05': -16.0, '0.25': -16.0, '0.5': -16.0, '0.75': 30.0, '0.95': 55.0}

## electrocardiographs_ecg_st_pET

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `Missing`: 6274

## electrocardiographs_ecg_ischemia_without_st_pET

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `Missing`: 6274

## electrocardiographs_ecg_type_of_rhythms_pET_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `Missing`: 6274

## electrocardiographs_ecg_type_of_rhythms_pET_last

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `Missing`: 6274

## smoking_status_smoker_last

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## smoking_status_formerSmoker_last

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `Missing`: 6274

## smoking_status_smoker_startTime_count

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 0.0
- mean/std: 0.0000 / 0.0
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 0.0}
- ⚠️ constant column (single value)

## nyha_nyha

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 1.0 / 4.0
- mean/std: 3.0646 / 0.7851296488555279
- quantiles: {'0.05': 2.0, '0.25': 2.0, '0.5': 3.0, '0.75': 4.0, '0.95': 4.0}

## nyha_nyha_pET

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 1.0 / 4.0
- mean/std: 3.0598 / 0.7853052268369851
- quantiles: {'0.05': 2.0, '0.25': 2.0, '0.5': 3.0, '0.75': 4.0, '0.95': 4.0}

## vital_signs_systolicBpDuringEncounter_value_pET

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 50.0 / 260.0
- mean/std: 133.3756 / 23.67274188581887
- quantiles: {'0.05': 100.0, '0.25': 120.0, '0.5': 130.0, '0.75': 143.48808827908186, '0.95': 180.0}

## vital_signs_bmi_value_pET

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 18.0 / 25.0
- mean/std: 21.7532 / 1.8701470751402411
- quantiles: {'0.05': 18.8277678123233, '0.25': 20.108431914413185, '0.5': 21.755805692912602, '0.75': 23.380551958881806, '0.95': 24.66705846226328}

## lab_results_creatBS_value_p3a_avg

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 3.3 / 59.0
- mean/std: 10.9768 / 4.908453943141329
- quantiles: {'0.05': 6.282166666666667, '0.25': 8.0, '0.5': 9.700000000000001, '0.75': 12.344047619047618, '0.95': 20.281025641025618}

## lab_results_validSerumCreatinine_value_pET

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 1.5 / 23.0
- mean/std: 10.2746 / 3.5583064671602864
- quantiles: {'0.05': 5.865000000000003, '0.25': 7.7, '0.5': 10.0, '0.75': 11.799999999999999, '0.95': 17.8}

## hyperkalemia_severity_categorizedValue

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 5
- top values (shown only where count ≥ 5):
  - `normal`: 5969
  - `mild`: 227
  - `moderate`: 46
  - `severe`: 24
  - `Missing`: 8

## ckd_severity_categorizedValue

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `Missing`: 6274

## conditions_heartFailure_timeFromEarliest_first

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 57.0
- mean/std: 1.2816 / 5.277918483874049
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 8.0}

## conditions_heart_failure_hf_within_18mo_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `true`: 6274

## conditions_heart_failure_occurred_prior_to_18_months_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 6119
  - `true`: 155

## encounter_primary_reason_HF_Disease_f5a_w7d_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 6259
  - `false`: 14
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## encounter_primary_reason_HF_Disease_f5a_w1mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 6231
  - `false`: 42
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## encounter_primary_reason_HF_Disease_f5a_w3mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 6201
  - `false`: 71
  - 1 other distinct value(s) suppressed, covering 2 row(s) (count below 5 and/or ranked beyond top 20)

## encounter_primary_reason_HF_Disease_f5a_w6mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 6171
  - `false`: 100
  - 1 other distinct value(s) suppressed, covering 3 row(s) (count below 5 and/or ranked beyond top 20)

## encounter_primary_reason_HF_Disease_f5a_w1a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 6147
  - `false`: 124
  - 1 other distinct value(s) suppressed, covering 3 row(s) (count below 5 and/or ranked beyond top 20)

## encounter_primary_reason_HF_Disease_f5a_w3a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 6110
  - `false`: 161
  - 1 other distinct value(s) suppressed, covering 3 row(s) (count below 5 and/or ranked beyond top 20)

## encounter_primary_reason_HF_Disease_f5a_w5a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 6107
  - `false`: 164
  - 1 other distinct value(s) suppressed, covering 3 row(s) (count below 5 and/or ranked beyond top 20)

## encounter_primary_reason_number_of_HF_rehospitalizations_5a_f5a_count

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 2.0
- mean/std: 0.0065 / 0.0863117376825872
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 0.0}

## encounter_primary_reason_CV_Disease_f5a_w7d_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 6259
  - `false`: 8
  - `true`: 7

## encounter_primary_reason_CV_Disease_f5a_w1mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 6231
  - `true`: 27
  - `false`: 16

## encounter_primary_reason_CV_Disease_f5a_w3mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 6201
  - `true`: 41
  - `false`: 32

## encounter_primary_reason_CV_Disease_f5a_w6mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 6171
  - `true`: 60
  - `false`: 43

## encounter_primary_reason_CV_Disease_f5a_w1a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 6147
  - `true`: 73
  - `false`: 54

## encounter_primary_reason_CV_Disease_f5a_w3a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 6110
  - `true`: 94
  - `false`: 70

## encounter_primary_reason_CV_Disease_f5a_w5a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 6107
  - `true`: 96
  - `false`: 71

## encounter_primary_reason_number_of_CV_rehospitalizations_5a_f5a_count

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 14.0
- mean/std: 0.1302 / 0.8962264173980161
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 0.0}

## encounter_primary_reason_non_CV_Disease_f5a_w7d_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 6259
  - `true`: 8
  - `false`: 7

## encounter_primary_reason_non_CV_Disease_f5a_w1mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 6231
  - `false`: 27
  - `true`: 16

## encounter_primary_reason_non_CV_Disease_f5a_w3mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 6201
  - `false`: 41
  - `true`: 32

## encounter_primary_reason_non_CV_Disease_f5a_w6mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 6171
  - `false`: 60
  - `true`: 43

## encounter_primary_reason_non_CV_Disease_f5a_w1a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 6147
  - `false`: 73
  - `true`: 54

## encounter_primary_reason_non_CV_Disease_f5a_w3a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 6110
  - `false`: 94
  - `true`: 70

## encounter_primary_reason_non_CV_Disease_f5a_w5a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 6107
  - `false`: 96
  - `true`: 71

## encounter_primary_reason_number_of_non_CV_rehospitalizations_5a_f5a_count

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 16.0
- mean/std: 0.1111 / 0.8018756804699785
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 0.0}

## encounter_primary_reason_renal_complications_f5a_w7d_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 6259
  - `false`: 15

## encounter_primary_reason_renal_complications_f5a_w1mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 6231
  - `false`: 43

## encounter_primary_reason_renal_complications_f5a_w3mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 6201
  - `false`: 73

## encounter_primary_reason_renal_complications_f5a_w6mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 6171
  - `false`: 103

## encounter_primary_reason_renal_complications_f5a_w1a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 6147
  - `false`: 127

## encounter_primary_reason_renal_complications_f5a_w3a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 6110
  - `false`: 164

## encounter_primary_reason_renal_complications_f5a_w5a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 6107
  - `false`: 167

## encounter_primary_reason_number_of_renal_rehospitalizations_5a_f5a_count

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 3.0
- mean/std: 0.0040 / 0.07018499674933434
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 0.0}

## cause_of_death_isCV_f5a_w7d_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 6273
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## cause_of_death_isCV_f5a_w1mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 6273
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## cause_of_death_isCV_f5a_w3mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 6273
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## cause_of_death_isCV_f5a_w6mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 6273
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## cause_of_death_isCV_f5a_w1a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 6273
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## cause_of_death_isCV_f5a_w3a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 6273
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## cause_of_death_isCV_f5a_w5a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 6273
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## cause_of_death_isRenal_f5a_w7d_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 6273
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## cause_of_death_isRenal_f5a_w1mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 6273
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## cause_of_death_isRenal_f5a_w3mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 6273
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## cause_of_death_isRenal_f5a_w6mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 6273
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## cause_of_death_isRenal_f5a_w1a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 6273
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## cause_of_death_isRenal_f5a_w3a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 6273
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## cause_of_death_isRenal_f5a_w5a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 6273
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## cause_of_death_isNonRenalAndNonCV_f5a_w7d_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 6273
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## cause_of_death_isNonRenalAndNonCV_f5a_w1mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 6273
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## cause_of_death_isNonRenalAndNonCV_f5a_w3mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 6273
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## cause_of_death_isNonRenalAndNonCV_f5a_w6mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 6273
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## cause_of_death_isNonRenalAndNonCV_f5a_w1a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 6273
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## cause_of_death_isNonRenalAndNonCV_f5a_w3a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 6273
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## cause_of_death_isNonRenalAndNonCV_f5a_w5a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 6273
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## cause_of_death_isAllCause_f5a_w7d_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 6273
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## cause_of_death_isAllCause_f5a_w1mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 6273
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## cause_of_death_isAllCause_f5a_w3mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 6273
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## cause_of_death_isAllCause_f5a_w6mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 6273
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## cause_of_death_isAllCause_f5a_w1a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 6273
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## cause_of_death_isAllCause_f5a_w3a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 6273
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## cause_of_death_isAllCause_f5a_w5a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 6273
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## eGFR_2021_ckd_epi_creatinine

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 20.0 / 220.0
- mean/std: 76.8784 / 23.923231143294167
- quantiles: {'0.05': 35.52869065, '0.25': 57.874977, '0.5': 79.350515, '0.75': 95.771454, '0.95': 111.2269085}

## ckd_severity_from_calculated_egfr

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 5
- top values (shown only where count ≥ 5):
  - `mildly_decreased`: 2359
  - `normal_or_high`: 2260
  - `mild_to_moderate_decrease`: 965
  - `moderate_to_severe_decrease`: 574
  - `severe_decrease`: 116

## ckd_severity_calculated_or_measured

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 5
- top values (shown only where count ≥ 5):
  - `mildly_decreased`: 2359
  - `normal_or_high`: 2260
  - `mild_to_moderate_decrease`: 965
  - `moderate_to_severe_decrease`: 574
  - `severe_decrease`: 116

## maggic_total_score

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -1.0 / 44.0
- mean/std: 8.4127 / 13.299770257961285
- quantiles: {'0.05': -1.0, '0.25': -1.0, '0.5': -1.0, '0.75': 22.0, '0.95': 32.0}

## med_acei

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## med_activeDuringEncounter_ace_inhibitors_arb_use

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## med_activeDuringEncounter_bb

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## med_anti_coag

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## med_anti_plat

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## med_antiarrhytmic

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## med_antiinfl

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## med_arb

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## med_ari

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## med_arni

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## med_bb

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## med_ccb

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## med_cortico_syst

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## med_digitalis

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## med_diuretics

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## med_diuretics_loop

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## med_inotropes

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## med_insulins

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## med_ivabradine

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## med_ll

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## med_mra

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## med_oral_antidiabetic

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## med_platelet

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## med_potassium_binders

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## med_rasi

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## med_rdoad

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## med_rdoad_syst

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## med_sglt2i

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## med_thrombolytic

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## med_vasodil

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## med_acei_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## med_anti_coag_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## med_anti_plat_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## med_antiarrhytmic_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## med_antiinfl_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## med_arb_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## med_ari_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## med_arni_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## med_bb_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## med_ccb_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## med_cortico_syst_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## med_digitalis_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## med_diuretics_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## med_diuretics_loop_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## med_inotropes_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## med_insulins_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## med_ivabradine_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## med_ll_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## med_mra_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## med_oral_antidiabetic_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## med_platelet_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## med_potassium_binders_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## med_rasi_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## med_rdoad_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## med_rdoad_syst_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## med_sglt2i_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## med_thrombolytic_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## med_vasodil_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## conditions_af

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 3724
  - `true`: 2550

## conditions_aidshiv

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 6270
  - 1 other distinct value(s) suppressed, covering 4 row(s) (count below 5 and/or ranked beyond top 20)

## conditions_ap

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 5141
  - `true`: 1133

## conditions_ckd_chronic

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 5421
  - `true`: 853

## conditions_cm

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 4779
  - `true`: 1495

## conditions_copd

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 5978
  - `true`: 296

## conditions_dem

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 6170
  - `true`: 104

## conditions_dep

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 6160
  - `true`: 114

## conditions_devices

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 5275
  - `true`: 999

## conditions_dia

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 6274

## conditions_diabetes

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 4371
  - `true`: 1903

## conditions_dysl

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 3597
  - `false`: 2677

## conditions_hf

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `true`: 6274

## conditions_hyp

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 4623
  - `false`: 1651

## conditions_hyperthyroid

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 6248
  - `true`: 26

## conditions_hypothyroid

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 6034
  - `true`: 240

## conditions_ibd

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 6267
  - `true`: 7

## conditions_ihd

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 3524
  - `false`: 2750

## conditions_ld

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 5991
  - `true`: 283

## conditions_mc

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 6070
  - `true`: 204

## conditions_mi

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 5065
  - `true`: 1209

## conditions_myocarditis

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 6264
  - `true`: 10

## conditions_osa

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 5982
  - `true`: 292

## conditions_pad

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 5555
  - `true`: 719

## conditions_pericardial

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 6151
  - `true`: 123

## conditions_rd

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 6244
  - `true`: 30

## conditions_revasc

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 5018
  - `true`: 1256

## conditions_stroke

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 6172
  - `true`: 102

## conditions_substance_abuse

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 6240
  - `true`: 34

## conditions_tia

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 6221
  - `true`: 53

## conditions_vd

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 3273
  - `false`: 3001
