# Column Analysis

Total rows: 5031
Total columns: 255

## patient_demographics_gender

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `male`: 3125
  - `female`: 1906

## patient_demographics_age

- dtype: `Int32` (numeric)
- nulls: 0 (0.00%)
- min/max: 19.0 / 110.0
- mean/std: 73.1650 / 12.710987318177024
- quantiles: {'0.05': 50.0, '0.25': 66.0, '0.5': 75.0, '0.75': 82.0, '0.95': 90.0}

## encounters_encounterClass

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `IMP`: 5031

## encounters_admissionYear

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 9
- top values (shown only where count ≥ 5):
  - `2019`: 1101
  - `2023`: 1015
  - `2021`: 985
  - `2020`: 908
  - `2022`: 843
  - `2024`: 147
  - `2018`: 14
  - `2017`: 10
  - `2016`: 8

## encounters_lengthOfStay

- dtype: `Int32` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 320.0
- mean/std: 10.6466 / 10.958203976983903
- quantiles: {'0.05': 2.0, '0.25': 5.0, '0.5': 8.0, '0.75': 13.0, '0.95': 27.0}

## encounters_numOfPreviousHFStays_count

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 420.0
- mean/std: 9.0952 / 19.371123774654777
- quantiles: {'0.05': 0.0, '0.25': 1.0, '0.5': 2.0, '0.75': 10.0, '0.95': 38.0}

## vital_signs_weight_value_p6mo_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -9.5 / 180.0
- mean/std: 44.6166 / 42.214083366632984
- quantiles: {'0.05': -9.5, '0.25': -9.5, '0.5': 63.0, '0.75': 78.0, '0.95': 95.0}

## vital_signs_weight_value_p6mo_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -7.5 / 180.0
- mean/std: 45.3632 / 41.347001195477645
- quantiles: {'0.05': -7.5, '0.25': -7.5, '0.5': 64.0, '0.75': 78.0, '0.95': 96.0}

## vital_signs_height_value_p1a_avg

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 61.0 / 220.0
- mean/std: 143.1504 / 45.680360557455785
- quantiles: {'0.05': 61.749999999999986, '0.25': 150.0, '0.5': 165.0, '0.75': 172.0, '0.95': 180.41666666666669}

## vital_signs_weight_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -9.5 / 180.0
- mean/std: 30.7685 / 43.25507965322784
- quantiles: {'0.05': -9.5, '0.25': -9.5, '0.5': -9.5, '0.75': 73.0, '0.95': 91.95}

## vital_signs_height_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 75.0 / 200.0
- mean/std: 130.1276 / 46.16978687094605
- quantiles: {'0.05': 75.5, '0.25': 75.5, '0.5': 160.0, '0.75': 170.0, '0.95': 180.0}

## vital_signs_diastolicBp_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 13.0 / 130.0
- mean/std: 54.8517 / 27.152710882633546
- quantiles: {'0.05': 13.75, '0.25': 13.75, '0.5': 65.0, '0.75': 74.0, '0.95': 85.0}

## vital_signs_diastolicBp_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 5.0 / 130.0
- mean/std: 50.6033 / 29.518084582226376
- quantiles: {'0.05': 5.0, '0.25': 5.0, '0.5': 60.0, '0.75': 70.0, '0.95': 80.0}

## vital_signs_heartRate_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -22.0 / 170.0
- mean/std: 48.5438 / 46.10476462629484
- quantiles: {'0.05': -22.0, '0.25': -22.0, '0.5': 68.0, '0.75': 79.0, '0.95': 100.0}

## vital_signs_heartRate_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -23.0 / 150.0
- mean/std: 47.1043 / 44.8405570932374
- quantiles: {'0.05': -22.5, '0.25': -22.5, '0.5': 68.0, '0.75': 78.0, '0.95': 94.0}

## vital_signs_systolicBp_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -1.3 / 230.0
- mean/std: 81.0266 / 60.339378120519434
- quantiles: {'0.05': -1.25, '0.25': -1.25, '0.5': 110.0, '0.75': 130.0, '0.95': 150.0}

## vital_signs_systolicBp_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -7.5 / 230.0
- mean/std: 76.0442 / 60.7737859901
- quantiles: {'0.05': -7.5, '0.25': -7.5, '0.5': 110.0, '0.75': 120.0, '0.95': 140.0}

## vital_signs_oxygenSaturation_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 19.0 / 100.0
- mean/std: 72.5621 / 35.40666600071252
- quantiles: {'0.05': 19.5, '0.25': 19.5, '0.5': 95.0, '0.75': 97.0, '0.95': 99.0}

## vital_signs_oxygenSaturation_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -22.0 / 100.0
- mean/std: 59.9969 / 54.14071818195312
- quantiles: {'0.05': -21.25, '0.25': -21.25, '0.5': 95.0, '0.75': 97.0, '0.95': 99.0}

## lab_results_hemoglobin_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 17.0 / 210.0
- mean/std: 113.1563 / 31.115783491332017
- quantiles: {'0.05': 17.25, '0.25': 98.0, '0.5': 115.0, '0.75': 133.0, '0.95': 155.0}

## lab_results_hemoglobin_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 12.0 / 230.0
- mean/std: 118.3684 / 32.742178481957396
- quantiles: {'0.05': 12.25, '0.25': 105.0, '0.5': 124.0, '0.75': 139.0, '0.95': 158.0}

## lab_results_ferritin_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -490.0 / 2000.0
- mean/std: -284.0972 / 361.6075287724346
- quantiles: {'0.05': -489.5, '0.25': -489.5, '0.5': -489.5, '0.75': 26.0, '0.95': 401.5}

## lab_results_ferritin_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -490.0 / 2000.0
- mean/std: -286.2228 / 357.25282665246976
- quantiles: {'0.05': -489.5, '0.25': -489.5, '0.5': -489.5, '0.75': 25.0, '0.95': 384.5}

## lab_results_ntProBnp_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -13000.0 / 50000.0
- mean/std: -2293.6748 / 11177.686896752337
- quantiles: {'0.05': -12472.0, '0.25': -12472.0, '0.5': 296.0, '0.75': 3097.0, '0.95': 17271.5}

## lab_results_ntProBnp_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -13000.0 / 50000.0
- mean/std: -1800.1584 / 11743.745249945936
- quantiles: {'0.05': -12463.25, '0.25': -12463.25, '0.5': 315.0, '0.75': 3891.0, '0.95': 19980.0}

## lab_results_crpNonHs_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -120.0 / 460.0
- mean/std: -33.8694 / 79.93059701979509
- quantiles: {'0.05': -112.35, '0.25': -112.35, '0.5': 1.3, '0.75': 19.0, '0.95': 94.35}

## lab_results_crpNonHs_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -120.0 / 480.0
- mean/std: -30.4798 / 92.86949871687578
- quantiles: {'0.05': -119.525, '0.25': -119.525, '0.5': 1.3, '0.75': 24.0, '0.95': 137.3}

## lab_results_triGly_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -2.4 / 11.0
- mean/std: 0.0960 / 1.7954620633452694
- quantiles: {'0.05': -2.3956, '0.25': -2.3956, '0.5': 0.9153, '0.75': 1.3334, '0.95': 2.1018}

## lab_results_triGly_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -2.4 / 11.0
- mean/std: 0.0821 / 1.7858485012599115
- quantiles: {'0.05': -2.3956, '0.25': -2.3956, '0.5': 0.8927, '0.75': 1.3108, '0.95': 2.0792}

## lab_results_bun_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -12.0 / 57.0
- mean/std: 8.3972 / 7.005756306935179
- quantiles: {'0.05': 2.8568, '0.25': 5.3565, '0.5': 7.4991, '0.75': 10.3559, '0.95': 20.7118}

## lab_results_bun_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -13.0 / 59.0
- mean/std: 8.8633 / 7.540186912487196
- quantiles: {'0.05': 3.2139, '0.25': 5.7136, '0.5': 7.4991, '0.75': 11.0701, '0.95': 22.4973}

## lab_results_potassium_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.27 / 9.4
- mean/std: 3.7950 / 0.8673961311143167
- quantiles: {'0.05': 2.8, '0.25': 3.5, '0.5': 3.9, '0.75': 4.2, '0.95': 4.9}

## lab_results_potassium_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.7 / 7.2
- mean/std: 3.8356 / 0.7537093541031262
- quantiles: {'0.05': 3.0, '0.25': 3.6, '0.5': 3.9, '0.75': 4.2, '0.95': 4.8}

## lab_results_sodium_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 110.0 / 170.0
- mean/std: 138.7886 / 6.165466404120085
- quantiles: {'0.05': 131.0, '0.25': 137.0, '0.5': 140.0, '0.75': 142.0, '0.95': 145.0}

## lab_results_sodium_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 100.0 / 170.0
- mean/std: 138.8018 / 7.887723848500286
- quantiles: {'0.05': 130.0, '0.25': 138.0, '0.5': 140.0, '0.75': 142.0, '0.95': 146.0}

## lab_results_albuminBS_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -3.0 / 52.0
- mean/std: 25.9754 / 15.182762956844488
- quantiles: {'0.05': -3.0, '0.25': 24.0, '0.5': 31.0, '0.75': 36.0, '0.95': 42.0}

## lab_results_albuminBS_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -3.0 / 52.0
- mean/std: 27.1634 / 15.717395460341429
- quantiles: {'0.05': -3.0, '0.25': 25.0, '0.5': 33.0, '0.75': 38.0, '0.95': 43.0}

## lab_results_hba1c%_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.72 / 14.0
- mean/std: 2.4814 / 2.8074580450453195
- quantiles: {'0.05': 0.7249999999999996, '0.25': 0.7249999999999996, '0.5': 0.7249999999999996, '0.75': 5.6, '0.95': 7.6}

## lab_results_hba1c%_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.75 / 14.0
- mean/std: 2.4948 / 2.78698696597667
- quantiles: {'0.05': 0.75, '0.25': 0.75, '0.5': 0.75, '0.75': 5.6, '0.95': 7.6}

## lab_results_validSerumCreatinine_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -4.4 / 23.0
- mean/std: 9.9856 / 5.592734760754799
- quantiles: {'0.05': -4.4, '0.25': 7.7, '0.5': 9.9, '0.75': 12.8, '0.95': 19.8}

## lab_results_valideGFR_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -30.0 / 140.0
- mean/std: 48.3766 / 42.82894068604677
- quantiles: {'0.05': -29.25, '0.25': 28.0, '0.5': 58.0, '0.75': 83.0, '0.95': 101.0}

## lab_results_valideGFR_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -28.0 / 130.0
- mean/std: 48.5482 / 42.34320918088336
- quantiles: {'0.05': -27.25, '0.25': 26.0, '0.5': 58.0, '0.75': 83.0, '0.95': 101.0}

## symptoms_Ankle_swelling_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## symptoms_Ascites_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## symptoms_Breathlessness_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## symptoms_Cardiac_murmur_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## symptoms_Chest_pain_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## symptoms_Cheyne_stokes_respiration_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## symptoms_Depression_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## symptoms_Dizziness_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## symptoms_Elevated_jugular_venous_pressure_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## symptoms_Fatigue_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## symptoms_Hepatojugular_reflux_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## symptoms_Hepatomegaly_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## symptoms_Intermittent_claudication_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## symptoms_Irregular_pulse_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## symptoms_Loss_of_appetite_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## symptoms_Nocturnal_cough_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## symptoms_Oliguria_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## symptoms_Orthopnoea_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## symptoms_Palpitations_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## symptoms_Paroxysmal_nocturnal_dyspnea_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## symptoms_Peripheral_edema_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## symptoms_Pleural_effusion_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## symptoms_Pulmonary_crepitations_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## symptoms_Reduced_exercise_tolerance_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## symptoms_Syncope_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## symptoms_Tachycardia_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## symptoms_Tachypnoea_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## symptoms_Third_heart_sound_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## symptoms_Weight_gain_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## symptoms_Weight_loss_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## echocardiographs_lvef

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -19.0 / 98.0
- mean/std: 31.8756 / 31.360248735985337
- quantiles: {'0.05': -18.66666666666667, '0.25': 17.18333339691162, '0.5': 40.777038955688475, '0.75': 57.62916692097981, '0.95': 68.0}

## echocardiographs_lvef_pET_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -19.0 / 98.0
- mean/std: 30.5177 / 32.01955917323062
- quantiles: {'0.05': -18.66666666666667, '0.25': -18.66666666666667, '0.5': 39.5, '0.75': 57.5, '0.95': 67.76666641235353}

## echocardiographs_lvef_pET_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -19.0 / 98.0
- mean/std: 29.5829 / 32.228630547890674
- quantiles: {'0.05': -18.66666666666667, '0.25': -18.66666666666667, '0.5': 38.936532211303714, '0.75': 56.699999173482254, '0.95': 67.76666641235353}

## electrocardiographs_ecg_qrs_duration_pET_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -39.0 / 290.0
- mean/std: 52.4474 / 80.06704676064557
- quantiles: {'0.05': -39.0, '0.25': -39.0, '0.5': 89.0, '0.75': 111.0, '0.95': 166.0}

## electrocardiographs_ecg_qrs_duration_pET_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -33.0 / 260.0
- mean/std: 54.5015 / 77.45123231711185
- quantiles: {'0.05': -33.0, '0.25': -33.0, '0.5': 89.0, '0.75': 111.0, '0.95': 166.0}

## electrocardiographs_ecg_qrs_axis_pET_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -180.0 / 270.0
- mean/std: -66.1805 / 105.83743063658564
- quantiles: {'0.05': -178.75, '0.25': -178.75, '0.5': -48.0, '0.75': 14.0, '0.95': 87.0}

## electrocardiographs_ecg_qrs_axis_pET_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -180.0 / 270.0
- mean/std: -69.5369 / 104.36020823900758
- quantiles: {'0.05': -180.0, '0.25': -180.0, '0.5': -47.0, '0.75': 8.0, '0.95': 81.0}

## electrocardiographs_ecg_qt_duration_corrected_pET_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 59.0 / 650.0
- mean/std: 276.6541 / 185.81277597604185
- quantiles: {'0.05': 59.0, '0.25': 59.0, '0.5': 394.0, '0.75': 435.0, '0.95': 487.0}

## electrocardiographs_ecg_qt_duration_corrected_pET_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 19.0 / 700.0
- mean/std: 259.0554 / 202.39215447597812
- quantiles: {'0.05': 19.25, '0.25': 19.25, '0.5': 390.0, '0.75': 429.0, '0.95': 484.0}

## electrocardiographs_ecg_st_pET

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `Missing`: 5031

## electrocardiographs_ecg_ischemia_without_st_pET

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `Missing`: 5031

## electrocardiographs_ecg_type_of_rhythms_pET_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `Missing`: 5031

## electrocardiographs_ecg_type_of_rhythms_pET_last

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `Missing`: 5031

## smoking_status_smoker_last

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 2738
  - `true`: 2293

## smoking_status_formerSmoker_last

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 2738
  - `true`: 2293

## smoking_status_smoker_totalSmokingDuration_sum

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -1200.0 / 4600.0
- mean/std: -562.5961 / 703.382998657153
- quantiles: {'0.05': -1147.0, '0.25': -1147.0, '0.5': -1147.0, '0.75': 0.0, '0.95': 306.5}

## smoking_status_smoker_startTime_count

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 1.0
- mean/std: 0.4558 / 0.4980897424650877
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 1.0, '0.95': 1.0}

## nyha_nyha

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 1.0 / 4.0
- mean/std: 2.8630 / 0.8210259821676238
- quantiles: {'0.05': 2.0, '0.25': 2.0, '0.5': 3.0, '0.75': 3.0, '0.95': 4.0}

## nyha_nyha_pET

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 1.0 / 4.0
- mean/std: 2.7217 / 0.7707907463287467
- quantiles: {'0.05': 1.0, '0.25': 2.0, '0.5': 3.0, '0.75': 3.0, '0.95': 4.0}

## vital_signs_systolicBpDuringEncounter_value_pET

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 40.0 / 230.0
- mean/std: 118.8421 / 13.818118826304353
- quantiles: {'0.05': 100.0, '0.25': 110.0, '0.5': 120.0, '0.75': 127.04562384313843, '0.95': 140.0}

## vital_signs_bmi_value_pET

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 12.0 / 50.0
- mean/std: 25.4165 / 4.700438298120964
- quantiles: {'0.05': 19.237176475628306, '0.25': 22.28, '0.5': 24.577905022875516, '0.75': 27.76, '0.95': 34.16}

## lab_results_creatBS_value_p3a_avg

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 1.0 / 170.0
- mean/std: 12.5787 / 9.13645642919575
- quantiles: {'0.05': 6.372282608695652, '0.25': 8.440000000000001, '0.5': 10.333333333333334, '0.75': 13.504545454545454, '0.95': 24.50333333333333}

## lab_results_validSerumCreatinine_value_pET

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 1.0 / 23.0
- mean/std: 11.0710 / 3.8592211452896983
- quantiles: {'0.05': 6.2, '0.25': 8.399999999999999, '0.5': 10.299999999999999, '0.75': 12.9, '0.95': 19.5}

## hyperkalemia_severity_categorizedValue

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 5
- top values (shown only where count ≥ 5):
  - `normal`: 4504
  - `Missing`: 386
  - `mild`: 105
  - `moderate`: 27
  - `severe`: 9

## ckd_severity_categorizedValue

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 7
- top values (shown only where count ≥ 5):
  - `mildly_decreased`: 1639
  - `Missing`: 891
  - `normal_or_high`: 870
  - `mild_to_moderate_decrease`: 725
  - `moderate_to_severe_decrease`: 523
  - `severe_decrease`: 270
  - `kidney_failure`: 113

## conditions_heartFailure_timeFromEarliest_first

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 290.0
- mean/std: 5.1435 / 22.581121205829668
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 26.0}

## conditions_heart_failure_hf_within_18mo_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `true`: 5031

## conditions_heart_failure_occurred_prior_to_18_months_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 4709
  - `true`: 322

## encounter_primary_reason_HF_Disease_f5a_w7d_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 4920
  - `false`: 107
  - 1 other distinct value(s) suppressed, covering 4 row(s) (count below 5 and/or ranked beyond top 20)

## encounter_primary_reason_HF_Disease_f5a_w1mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 4676
  - `false`: 347
  - `true`: 8

## encounter_primary_reason_HF_Disease_f5a_w3mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 4261
  - `false`: 746
  - `true`: 24

## encounter_primary_reason_HF_Disease_f5a_w6mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 4011
  - `false`: 987
  - `true`: 33

## encounter_primary_reason_HF_Disease_f5a_w1a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 3743
  - `false`: 1245
  - `true`: 43

## encounter_primary_reason_HF_Disease_f5a_w3a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 3412
  - `false`: 1567
  - `true`: 52

## encounter_primary_reason_HF_Disease_f5a_w5a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 3351
  - `false`: 1625
  - `true`: 55

## encounter_primary_reason_number_of_HF_rehospitalizations_5a_f5a_count

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 2.0
- mean/std: 0.0183 / 0.13837859058377863
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 0.0}

## encounter_primary_reason_CV_Disease_f5a_w7d_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 4920
  - `true`: 59
  - `false`: 52

## encounter_primary_reason_CV_Disease_f5a_w1mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 4676
  - `false`: 186
  - `true`: 169

## encounter_primary_reason_CV_Disease_f5a_w3mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 4261
  - `true`: 392
  - `false`: 378

## encounter_primary_reason_CV_Disease_f5a_w6mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 4011
  - `false`: 517
  - `true`: 503

## encounter_primary_reason_CV_Disease_f5a_w1a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 3743
  - `false`: 660
  - `true`: 628

## encounter_primary_reason_CV_Disease_f5a_w3a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 3412
  - `false`: 850
  - `true`: 769

## encounter_primary_reason_CV_Disease_f5a_w5a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 3351
  - `false`: 881
  - `true`: 799

## encounter_primary_reason_number_of_days_to_rehosp_for_CV_f5a_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -430.0 / 1800.0
- mean/std: -323.2995 / 277.39535848377074
- quantiles: {'0.05': -429.75, '0.25': -429.75, '0.5': -429.75, '0.75': -429.75, '0.95': 242.5}

## encounter_primary_reason_number_of_CV_rehospitalizations_5a_f5a_count

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 11.0
- mean/std: 0.4075 / 0.9291693408995324
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 2.0}

## encounter_primary_reason_non_CV_Disease_f5a_w7d_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 4920
  - `false`: 56
  - `true`: 55

## encounter_primary_reason_non_CV_Disease_f5a_w1mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 4676
  - `true`: 194
  - `false`: 161

## encounter_primary_reason_non_CV_Disease_f5a_w3mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 4261
  - `true`: 386
  - `false`: 384

## encounter_primary_reason_non_CV_Disease_f5a_w6mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 4011
  - `true`: 525
  - `false`: 495

## encounter_primary_reason_non_CV_Disease_f5a_w1a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 3743
  - `true`: 669
  - `false`: 619

## encounter_primary_reason_non_CV_Disease_f5a_w3a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 3412
  - `true`: 861
  - `false`: 758

## encounter_primary_reason_non_CV_Disease_f5a_w5a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 3351
  - `true`: 893
  - `false`: 787

## encounter_primary_reason_number_of_days_to_rehosp_for_non_CV_f5a_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -450.0 / 1800.0
- mean/std: -314.9131 / 305.1309195084433
- quantiles: {'0.05': -440.5, '0.25': -440.5, '0.5': -440.5, '0.75': -440.5, '0.95': 316.0}

## encounter_primary_reason_number_of_non_CV_rehospitalizations_5a_f5a_count

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 23.0
- mean/std: 0.6190 / 1.4883611401400834
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 1.0, '0.95': 4.0}

## encounter_primary_reason_renal_complications_f5a_w7d_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 4920
  - `false`: 108
  - 1 other distinct value(s) suppressed, covering 3 row(s) (count below 5 and/or ranked beyond top 20)

## encounter_primary_reason_renal_complications_f5a_w1mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 4676
  - `false`: 346
  - `true`: 9

## encounter_primary_reason_renal_complications_f5a_w3mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 4261
  - `false`: 747
  - `true`: 23

## encounter_primary_reason_renal_complications_f5a_w6mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 4011
  - `false`: 988
  - `true`: 32

## encounter_primary_reason_renal_complications_f5a_w1a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 3743
  - `false`: 1245
  - `true`: 43

## encounter_primary_reason_renal_complications_f5a_w3a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 3412
  - `false`: 1563
  - `true`: 56

## encounter_primary_reason_renal_complications_f5a_w5a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 3351
  - `false`: 1621
  - `true`: 59

## encounter_primary_reason_number_of_renal_rehospitalizations_5a_f5a_count

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 2.0
- mean/std: 0.0503 / 0.23008213681232698
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 0.0}

## cause_of_death_isCV_f5a_w7d_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 5026
  - `false`: 5

## cause_of_death_isCV_f5a_w1mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 5002
  - `false`: 29

## cause_of_death_isCV_f5a_w3mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 4973
  - `false`: 58

## cause_of_death_isCV_f5a_w6mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 4922
  - `false`: 109

## cause_of_death_isCV_f5a_w1a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 4894
  - `false`: 137

## cause_of_death_isCV_f5a_w3a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 4889
  - `false`: 142

## cause_of_death_isCV_f5a_w5a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 4889
  - `false`: 142

## cause_of_death_isRenal_f5a_w7d_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 5026
  - `false`: 5

## cause_of_death_isRenal_f5a_w1mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 5002
  - `false`: 29

## cause_of_death_isRenal_f5a_w3mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 4973
  - `false`: 58

## cause_of_death_isRenal_f5a_w6mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 4922
  - `false`: 109

## cause_of_death_isRenal_f5a_w1a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 4894
  - `false`: 137

## cause_of_death_isRenal_f5a_w3a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 4889
  - `false`: 142

## cause_of_death_isRenal_f5a_w5a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 4889
  - `false`: 142

## cause_of_death_isNonRenalAndNonCV_f5a_w7d_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 5026
  - `false`: 5

## cause_of_death_isNonRenalAndNonCV_f5a_w1mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 5002
  - `false`: 29

## cause_of_death_isNonRenalAndNonCV_f5a_w3mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 4973
  - `false`: 58

## cause_of_death_isNonRenalAndNonCV_f5a_w6mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 4922
  - `false`: 109

## cause_of_death_isNonRenalAndNonCV_f5a_w1a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 4894
  - `false`: 137

## cause_of_death_isNonRenalAndNonCV_f5a_w3a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 4889
  - `false`: 142

## cause_of_death_isNonRenalAndNonCV_f5a_w5a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 4889
  - `false`: 142

## cause_of_death_isAllCause_f5a_w7d_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 5026
  - `true`: 5

## cause_of_death_isAllCause_f5a_w1mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 5002
  - `true`: 29

## cause_of_death_isAllCause_f5a_w3mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 4973
  - `true`: 58

## cause_of_death_isAllCause_f5a_w6mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 4922
  - `true`: 109

## cause_of_death_isAllCause_f5a_w1a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 4894
  - `true`: 137

## cause_of_death_isAllCause_f5a_w3a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 4889
  - `true`: 142

## cause_of_death_isAllCause_f5a_w5a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 4889
  - `true`: 142

## eGFR_2021_ckd_epi_creatinine

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 19.0 / 250.0
- mean/std: 69.7106 / 24.535108012353703
- quantiles: {'0.05': 31.432885, '0.25': 51.162541, '0.5': 70.069352, '0.75': 88.051175, '0.95': 104.1607915}

## ckd_severity_from_calculated_egfr

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 5
- top values (shown only where count ≥ 5):
  - `mildly_decreased`: 2113
  - `normal_or_high`: 1190
  - `mild_to_moderate_decrease`: 871
  - `moderate_to_severe_decrease`: 694
  - `severe_decrease`: 163

## ckd_severity_calculated_or_measured

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 5
- top values (shown only where count ≥ 5):
  - `mildly_decreased`: 2113
  - `normal_or_high`: 1190
  - `mild_to_moderate_decrease`: 871
  - `moderate_to_severe_decrease`: 694
  - `severe_decrease`: 163

## maggic_total_score

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -1.3 / 45.0
- mean/std: 19.0700 / 13.631580428179484
- quantiles: {'0.05': -1.25, '0.25': -1.25, '0.5': 24.0, '0.75': 30.0, '0.95': 36.0}

## med_acei

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## med_activeDuringEncounter_ace_inhibitors_arb_use

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## med_activeDuringEncounter_bb

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## med_anti_coag

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## med_anti_plat

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## med_antiarrhytmic

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## med_antiinfl

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## med_arb

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## med_ari

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## med_arni

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## med_bb

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## med_ccb

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## med_cortico_syst

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## med_digitalis

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## med_diuretics

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## med_diuretics_loop

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## med_inotropes

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## med_insulins

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## med_ivabradine

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## med_ll

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## med_mra

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## med_oral_antidiabetic

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## med_platelet

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## med_potassium_binders

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## med_rasi

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## med_rdoad

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## med_rdoad_syst

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## med_sglt2i

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## med_thrombolytic

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## med_vasodil

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## med_acei_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## med_anti_coag_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## med_anti_plat_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## med_antiarrhytmic_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## med_antiinfl_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## med_arb_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## med_ari_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## med_arni_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## med_bb_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## med_ccb_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## med_cortico_syst_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## med_digitalis_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## med_diuretics_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## med_diuretics_loop_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## med_inotropes_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## med_insulins_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## med_ivabradine_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## med_ll_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## med_mra_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## med_oral_antidiabetic_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## med_platelet_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## med_potassium_binders_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## med_rasi_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## med_rdoad_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## med_rdoad_syst_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## med_sglt2i_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## med_thrombolytic_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## med_vasodil_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## conditions_af

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 2862
  - `true`: 2169

## conditions_aidshiv

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 4973
  - `true`: 58

## conditions_ap

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 4721
  - `true`: 310

## conditions_ckd_chronic

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 3522
  - `true`: 1509

## conditions_cm

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 4371
  - `true`: 660

## conditions_copd

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 4785
  - `true`: 246

## conditions_dem

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 4711
  - `true`: 320

## conditions_dep

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## conditions_devices

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## conditions_dia

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## conditions_diabetes

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 2976
  - `true`: 2055

## conditions_dysl

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 2538
  - `false`: 2493

## conditions_hf

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `true`: 5031

## conditions_hyp

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 3851
  - `false`: 1180

## conditions_hyperthyroid

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## conditions_hypothyroid

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## conditions_ibd

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## conditions_ihd

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 2820
  - `false`: 2211

## conditions_ld

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 4643
  - `true`: 388

## conditions_mc

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 4865
  - `true`: 166

## conditions_mi

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 2629
  - `false`: 2402

## conditions_myocarditis

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## conditions_osa

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 5030
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## conditions_pad

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 4444
  - `true`: 587

## conditions_pericardial

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 5026
  - `true`: 5

## conditions_rd

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 5029
  - 1 other distinct value(s) suppressed, covering 2 row(s) (count below 5 and/or ranked beyond top 20)

## conditions_revasc

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## conditions_stroke

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 3828
  - `true`: 1203

## conditions_substance_abuse

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 5031

## conditions_tia

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 5029
  - 1 other distinct value(s) suppressed, covering 2 row(s) (count below 5 and/or ranked beyond top 20)

## conditions_vd

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 4402
  - `true`: 629
