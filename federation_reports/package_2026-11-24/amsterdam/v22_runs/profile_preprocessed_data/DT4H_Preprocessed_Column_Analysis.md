# Column Analysis

Total rows: 4445
Total columns: 254

## patient_demographics_gender

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `male`: 2572
  - `female`: 1873

## patient_demographics_age

- dtype: `Int32` (numeric)
- nulls: 0 (0.00%)
- min/max: 18.0 / 110.0
- mean/std: 71.1062 / 13.871646089276911
- quantiles: {'0.05': 45.0, '0.25': 64.0, '0.5': 73.0, '0.75': 81.0, '0.95': 90.0}

## encounters_encounterClass

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `IMP`: 4445

## encounters_admissionYear

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 11
- top values (shown only where count ≥ 5):
  - `2019`: 523
  - `2023`: 518
  - `2022`: 491
  - `2020`: 469
  - `2024`: 468
  - `2021`: 465
  - `2018`: 432
  - `2017`: 406
  - `2016`: 363
  - `2025`: 282
  - `2015`: 28

## encounters_lengthOfStay

- dtype: `Int32` (numeric)
- nulls: 0 (0.00%)
- min/max: 1.0 / 260.0
- mean/std: 11.7102 / 13.933224319255253
- quantiles: {'0.05': 1.0, '0.25': 4.0, '0.5': 8.0, '0.75': 15.0, '0.95': 35.0}

## encounters_numOfPreviousHFStays_count

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 790.0
- mean/std: 50.4160 / 65.15365392434441
- quantiles: {'0.05': 0.0, '0.25': 4.0, '0.5': 26.0, '0.75': 71.0, '0.95': 180.0}

## vital_signs_weight_value_p6mo_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -13.0 / 210.0
- mean/std: 65.4111 / 35.41373147642459
- quantiles: {'0.05': -12.149999999999999, '0.25': 57.6, '0.5': 71.0, '0.75': 85.0, '0.95': 110.78000000000002}

## vital_signs_weight_value_p6mo_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -18.0 / 250.0
- mean/std: 66.7976 / 38.03393493619137
- quantiles: {'0.05': -17.875, '0.25': 59.4, '0.5': 73.9, '0.75': 87.1, '0.95': 114.34000000000006}

## vital_signs_height_value_p1a_avg

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 85.0 / 210.0
- mean/std: 151.2012 / 37.306796334207824
- quantiles: {'0.05': 85.75, '0.25': 150.0, '0.5': 167.0, '0.75': 176.0, '0.95': 186.0}

## vital_signs_weight_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -13.0 / 210.0
- mean/std: 60.7814 / 38.939297930025305
- quantiles: {'0.05': -12.149999999999999, '0.25': 53.3, '0.5': 69.6, '0.75': 83.8, '0.95': 110.0}

## vital_signs_height_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 85.0 / 210.0
- mean/std: 131.6936 / 43.39048806859017
- quantiles: {'0.05': 85.75, '0.25': 85.75, '0.5': 156.0, '0.75': 172.0, '0.95': 185.0}

## vital_signs_diastolicBp_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -32.0 / 160.0
- mean/std: 58.2724 / 33.3232439224049
- quantiles: {'0.05': -32.0, '0.25': 57.0, '0.5': 66.0, '0.75': 76.0, '0.95': 90.0}

## vital_signs_diastolicBp_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -41.0 / 210.0
- mean/std: 63.3206 / 39.54804062415152
- quantiles: {'0.05': -40.25, '0.25': 58.0, '0.5': 71.0, '0.75': 84.0, '0.95': 107.0}

## vital_signs_heartRate_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -44.0 / 180.0
- mean/std: 81.1751 / 18.044777245635125
- quantiles: {'0.05': 57.0, '0.25': 69.0, '0.5': 80.0, '0.75': 91.0, '0.95': 112.0}

## vital_signs_heartRate_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -64.0 / 260.0
- mean/std: 90.7512 / 25.838028016597967
- quantiles: {'0.05': 57.0, '0.25': 73.0, '0.5': 88.0, '0.75': 105.0, '0.95': 136.0}

## vital_signs_systolicBp_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 8.5 / 220.0
- mean/std: 108.9980 / 40.22996166890388
- quantiles: {'0.05': 8.5, '0.25': 100.0, '0.5': 115.0, '0.75': 132.0, '0.95': 158.0}

## vital_signs_systolicBp_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -6.8 / 260.0
- mean/std: 115.8929 / 49.94276207792295
- quantiles: {'0.05': -6.75, '0.25': 103.0, '0.5': 122.0, '0.75': 145.0, '0.95': 180.0}

## vital_signs_oxygenSaturation_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -25.0 / 100.0
- mean/std: 28.7820 / 61.86281271329469
- quantiles: {'0.05': -25.0, '0.25': -25.0, '0.5': -25.0, '0.75': 100.0, '0.95': 100.0}

## vital_signs_oxygenSaturation_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -25.0 / 100.0
- mean/std: 28.8245 / 61.87919971311252
- quantiles: {'0.05': -25.0, '0.25': -25.0, '0.5': -25.0, '0.75': 100.0, '0.95': 100.0}

## lab_results_hemoglobin_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -11.0 / 220.0
- mean/std: 79.4848 / 60.3960821872784
- quantiles: {'0.05': -10.474099999999993, '0.25': -10.474099999999993, '0.5': 101.5182, '0.75': 124.0778, '0.95': 153.083}

## lab_results_hemoglobin_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -8.1 / 240.0
- mean/std: 80.2057 / 59.02292954226999
- quantiles: {'0.05': -8.057000000000002, '0.25': -8.057000000000002, '0.5': 101.5182, '0.75': 122.4664, '0.95': 151.4716}

## lab_results_ferritin_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -11000.0 / 43000.0
- mean/std: -8880.3476 / 4015.1831006150596
- quantiles: {'0.05': -10506.5, '0.25': -10506.5, '0.5': -10506.5, '0.75': -10506.5, '0.95': 343.8000000000002}

## lab_results_ferritin_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -11000.0 / 43000.0
- mean/std: -8887.5079 / 3984.4464783994476
- quantiles: {'0.05': -10507.75, '0.25': -10507.75, '0.5': -10507.75, '0.75': -10507.75, '0.95': 349.0}

## lab_results_ntProBnp_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -18000.0 / 70000.0
- mean/std: -6994.2564 / 16028.988284422516
- quantiles: {'0.05': -17458.75, '0.25': -17458.75, '0.5': -17458.75, '0.75': 2734.0, '0.95': 22584.200000000008}

## lab_results_ntProBnp_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -18000.0 / 70000.0
- mean/std: -7143.8893 / 15887.026386716554
- quantiles: {'0.05': -17458.75, '0.25': -17458.75, '0.5': -17458.75, '0.75': 2476.9999999999995, '0.95': 21579.60000000003}

## lab_results_crpNonHs_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -140.0 / 550.0
- mean/std: -11.3115 / 122.1789774689581
- quantiles: {'0.05': -135.1, '0.25': -135.1, '0.5': 9.0, '0.75': 57.0, '0.95': 208.0}

## lab_results_crpNonHs_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -160.0 / 630.0
- mean/std: -32.0799 / 116.33119149112895
- quantiles: {'0.05': -157.05, '0.25': -157.05, '0.5': 8.0, '0.75': 41.0, '0.95': 149.3}

## lab_results_triGly_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -3.9 / 17.0
- mean/std: -3.5392 / 1.2639329359249012
- quantiles: {'0.05': -3.825, '0.25': -3.825, '0.5': -3.825, '0.75': -3.825, '0.95': 0.5680000000000018}

## lab_results_triGly_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -2.6 / 12.0
- mean/std: -2.3218 / 0.9530478156330756
- quantiles: {'0.05': -2.5374999999999996, '0.25': -2.5374999999999996, '0.5': -2.5374999999999996, '0.75': -2.5374999999999996, '0.95': 0.5940000000000054}

## lab_results_cholTot_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -1.1 / 12.0
- mean/std: -0.7601 / 1.107532651862625
- quantiles: {'0.05': -1.0099999999999998, '0.25': -1.0099999999999998, '0.5': -1.0099999999999998, '0.75': -1.0099999999999998, '0.95': 1.9940000000000055}

## lab_results_cholTot_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -1.1 / 12.0
- mean/std: -0.7606 / 1.1057343614991357
- quantiles: {'0.05': -1.0099999999999998, '0.25': -1.0099999999999998, '0.5': -1.0099999999999998, '0.75': -1.0099999999999998, '0.95': 1.9640000000000055}

## lab_results_creatUS_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -590.0 / 2900.0
- mean/std: -470.9523 / 380.4716357944593
- quantiles: {'0.05': -582.568, '0.25': -582.568, '0.5': -582.568, '0.75': -582.568, '0.95': 512.6598400000008}

## lab_results_creatUS_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -770.0 / 3200.0
- mean/std: -638.2260 / 422.8928079945608
- quantiles: {'0.05': -763.5600000000001, '0.25': -763.5600000000001, '0.5': -763.5600000000001, '0.75': -763.5600000000001, '0.95': 474.1990400000008}

## lab_results_potassium_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.48 / 8.1
- mean/std: 4.0370 / 0.8924018599702389
- quantiles: {'0.05': 3.1, '0.25': 3.72, '0.5': 4.1, '0.75': 4.48, '0.95': 5.19}

## lab_results_potassium_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -0.4 / 7.6
- mean/std: 3.9260 / 1.035371332778549
- quantiles: {'0.05': 2.97, '0.25': 3.6, '0.5': 4.0, '0.75': 4.4, '0.95': 5.2}

## lab_results_sodium_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 100.0 / 170.0
- mean/std: 136.1199 / 8.78546870595893
- quantiles: {'0.05': 104.5, '0.25': 135.0, '0.5': 138.0, '0.75': 140.7, '0.95': 144.5}

## lab_results_sodium_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 87.0 / 160.0
- mean/std: 135.1929 / 12.312997195817623
- quantiles: {'0.05': 87.5, '0.25': 134.9, '0.5': 138.0, '0.75': 141.0, '0.95': 144.5}

## lab_results_albuminBS_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -1.4 / 47.0
- mean/std: 8.9974 / 14.956731423194839
- quantiles: {'0.05': -1.375, '0.25': -1.375, '0.5': -1.375, '0.75': 25.0, '0.95': 36.0}

## lab_results_albuminBS_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -0.5 / 47.0
- mean/std: 9.4031 / 14.340204235154065
- quantiles: {'0.05': -0.5, '0.25': -0.5, '0.5': -0.5, '0.75': 24.0, '0.95': 36.0}

## lab_results_validSerumCreatinine_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -4.3 / 23.0
- mean/std: 10.5455 / 7.256635887286583
- quantiles: {'0.05': -4.22994, '0.25': 7.8039, '0.5': 11.1969, '0.75': 15.4947, '0.95': 21.133866000000015}

## lab_results_valideGFR_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -18.0 / 90.0
- mean/std: 45.6381 / 29.349864282756425
- quantiles: {'0.05': -17.5, '0.25': 26.0, '0.5': 46.0, '0.75': 67.0, '0.95': 90.0}

## lab_results_valideGFR_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -19.0 / 95.0
- mean/std: 46.7796 / 30.0554369269213
- quantiles: {'0.05': -18.75, '0.25': 27.0, '0.5': 48.0, '0.75': 70.0, '0.95': 90.0}

## symptoms_Ankle_swelling_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 4445

## symptoms_Ascites_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 4445

## symptoms_Breathlessness_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 4445

## symptoms_Cardiac_murmur_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 4445

## symptoms_Chest_pain_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 4445

## symptoms_Cheyne_stokes_respiration_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 4445

## symptoms_Depression_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 4445

## symptoms_Dizziness_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 4445

## symptoms_Elevated_jugular_venous_pressure_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 4445

## symptoms_Fatigue_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 4445

## symptoms_Hepatojugular_reflux_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 4445

## symptoms_Hepatomegaly_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 4445

## symptoms_Intermittent_claudication_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 4445

## symptoms_Irregular_pulse_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 4445

## symptoms_Loss_of_appetite_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 4445

## symptoms_Nocturnal_cough_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 4445

## symptoms_Oliguria_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 4445

## symptoms_Orthopnoea_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 4445

## symptoms_Palpitations_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 4445

## symptoms_Paroxysmal_nocturnal_dyspnea_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 4445

## symptoms_Peripheral_edema_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 4445

## symptoms_Pleural_effusion_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 4445

## symptoms_Pulmonary_crepitations_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 4445

## symptoms_Reduced_exercise_tolerance_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 4445

## symptoms_Syncope_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 4445

## symptoms_Tachycardia_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 4445

## symptoms_Tachypnoea_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 4445

## symptoms_Third_heart_sound_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 4445

## symptoms_Weight_gain_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 4445

## symptoms_Weight_loss_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 4445

## echocardiographs_lvef

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -45.0 / 100.0
- mean/std: -22.1189 / 39.699608360895084
- quantiles: {'0.05': -44.99956028, '0.25': -44.99956028, '0.5': -44.99956028, '0.75': 15.411754608154, '0.95': 59.0}

## echocardiographs_lvef_pET_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -45.0 / 97.0
- mean/std: -26.4911 / 35.311070082054314
- quantiles: {'0.05': -44.029908212500004, '0.25': -44.029908212500004, '0.5': -44.029908212500004, '0.75': -44.029908212500004, '0.95': 52.56920840800002}

## echocardiographs_lvef_pET_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -79.0 / 96.0
- mean/std: -52.4561 / 50.374591710507296
- quantiles: {'0.05': -78.77288183, '0.25': -78.77288183, '0.5': -78.77288183, '0.75': -78.77288183, '0.95': 54.148841164000025}

## electrocardiographs_ecg_qrs_duration_pET_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -48.0 / 260.0
- mean/std: 43.4162 / 86.99865248465755
- quantiles: {'0.05': -48.0, '0.25': -48.0, '0.5': 82.0, '0.75': 113.0, '0.95': 172.0}

## electrocardiographs_ecg_qrs_duration_pET_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -41.0 / 270.0
- mean/std: 47.1694 / 83.79627761942244
- quantiles: {'0.05': -40.75, '0.25': -40.75, '0.5': 83.0, '0.75': 113.0, '0.95': 172.0}

## electrocardiographs_ecg_qt_duration_corrected_pET_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -4.3 / 770.0
- mean/std: 252.0211 / 235.77918943971858
- quantiles: {'0.05': -4.25, '0.25': -4.25, '0.5': 401.0, '0.75': 463.0, '0.95': 534.0}

## electrocardiographs_ecg_qt_duration_corrected_pET_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -24.0 / 810.0
- mean/std: 243.3688 / 246.07204080349416
- quantiles: {'0.05': -23.5, '0.25': -23.5, '0.5': 401.0, '0.75': 465.0, '0.95': 535.0}

## electrocardiographs_ecg_st_pET

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 4396
  - `true`: 49

## electrocardiographs_ecg_ischemia_without_st_pET

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `Missing`: 4445

## electrocardiographs_ecg_type_of_rhythms_pET_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `Missing`: 4445

## electrocardiographs_ecg_type_of_rhythms_pET_last

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `Missing`: 4445

## smoking_status_smoker_last

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 4204
  - `true`: 241

## smoking_status_formerSmoker_last

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 4185
  - `true`: 241
  - `false`: 19

## smoking_status_smoker_startTime_count

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 19.0
- mean/std: 0.0785 / 0.8415939775764368
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 0.0}

## nyha_nyha

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 1.0 / 4.0
- mean/std: 2.3969 / 0.8300656477882464
- quantiles: {'0.05': 1.0, '0.25': 2.0, '0.5': 2.0, '0.75': 3.0, '0.95': 4.0}

## nyha_nyha_pET

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 1.0 / 4.0
- mean/std: 2.4974 / 0.8420780636182507
- quantiles: {'0.05': 1.0, '0.25': 2.0, '0.5': 3.0, '0.75': 3.0, '0.95': 4.0}

## vital_signs_systolicBpDuringEncounter_value_pET

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 41.0 / 220.0
- mean/std: 121.3406 / 22.468568188906428
- quantiles: {'0.05': 90.0, '0.25': 106.0, '0.5': 118.0, '0.75': 135.0, '0.95': 161.0}

## vital_signs_bmi_value_pET

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 13.0 / 72.0
- mean/std: 25.7383 / 6.112361118004038
- quantiles: {'0.05': 18.592, '0.25': 21.7, '0.5': 24.49, '0.75': 28.5, '0.95': 37.42400000000001}

## lab_results_creatBS_value_p3a_avg

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 1.3 / 190.0
- mean/std: 15.4279 / 10.991683455577855
- quantiles: {'0.05': 6.583676666666667, '0.25': 9.498515000000001, '0.5': 12.441, '0.75': 17.44002, '0.95': 33.83369304818483}

## lab_results_validSerumCreatinine_value_pET

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.22 / 23.0
- mean/std: 12.9473 / 4.761819027399907
- quantiles: {'0.05': 6.064422, '0.25': 9.274199999999999, '0.5': 12.2148, '0.75': 16.3995, '0.95': 21.489}

## hyperkalemia_severity_categorizedValue

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 5
- top values (shown only where count ≥ 5):
  - `normal`: 3785
  - `Missing`: 345
  - `mild`: 213
  - `moderate`: 72
  - `severe`: 30

## ckd_severity_categorizedValue

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 7
- top values (shown only where count ≥ 5):
  - `mildly_decreased`: 1170
  - `moderate_to_severe_decrease`: 795
  - `mild_to_moderate_decrease`: 731
  - `severe_decrease`: 659
  - `normal_or_high`: 540
  - `Missing`: 309
  - `kidney_failure`: 241

## conditions_heartFailure_timeFromEarliest_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -35.0 / 140.0
- mean/std: 12.8710 / 24.7485430216172
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 13.0, '0.95': 72.80000000000018}

## conditions_heart_failure_hf_within_18mo_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 4432
  - `false`: 13

## conditions_heart_failure_occurred_prior_to_18_months_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 3431
  - `true`: 1014

## encounter_primary_reason_HF_Disease_f5a_w7d_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 3983
  - `false`: 418
  - `true`: 44

## encounter_primary_reason_HF_Disease_f5a_w1mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 3325
  - `false`: 988
  - `true`: 132

## encounter_primary_reason_HF_Disease_f5a_w3mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 2632
  - `false`: 1600
  - `true`: 213

## encounter_primary_reason_HF_Disease_f5a_w6mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 2291
  - `false`: 1901
  - `true`: 253

## encounter_primary_reason_HF_Disease_f5a_w1a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `false`: 2099
  - `Missing`: 2073
  - `true`: 273

## encounter_primary_reason_HF_Disease_f5a_w3a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `false`: 2289
  - `Missing`: 1864
  - `true`: 292

## encounter_primary_reason_HF_Disease_f5a_w5a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `false`: 2331
  - `Missing`: 1820
  - `true`: 294

## encounter_primary_reason_number_of_days_to_rehosp_for_heart_failure_f5a_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -450.0 / 1900.0
- mean/std: -413.2463 / 147.43396609798552
- quantiles: {'0.05': -450.0, '0.25': -450.0, '0.5': -450.0, '0.75': -450.0, '0.95': 12.0}

## encounter_primary_reason_number_of_HF_rehospitalizations_5a_f5a_count

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 20.0
- mean/std: 0.3638 / 1.1068854651359008
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 2.0}

## encounter_primary_reason_CV_Disease_f5a_w7d_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 3983
  - `false`: 286
  - `true`: 176

## encounter_primary_reason_CV_Disease_f5a_w1mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 3325
  - `false`: 685
  - `true`: 435

## encounter_primary_reason_CV_Disease_f5a_w3mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 2632
  - `false`: 1133
  - `true`: 680

## encounter_primary_reason_CV_Disease_f5a_w6mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 2291
  - `false`: 1332
  - `true`: 822

## encounter_primary_reason_CV_Disease_f5a_w1a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 2073
  - `false`: 1476
  - `true`: 896

## encounter_primary_reason_CV_Disease_f5a_w3a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 1864
  - `false`: 1607
  - `true`: 974

## encounter_primary_reason_CV_Disease_f5a_w5a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 1820
  - `false`: 1637
  - `true`: 988

## encounter_primary_reason_number_of_days_to_rehosp_for_CV_f5a_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -460.0 / 1900.0
- mean/std: -323.0275 / 264.1527225368795
- quantiles: {'0.05': -451.25, '0.25': -451.25, '0.5': -451.25, '0.75': -451.25, '0.95': 139.80000000000018}

## encounter_primary_reason_number_of_CV_rehospitalizations_5a_f5a_count

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 37.0
- mean/std: 1.7444 / 3.3288714219527487
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 2.0, '0.95': 8.0}

## encounter_primary_reason_non_CV_Disease_f5a_w7d_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 3983
  - `true`: 286
  - `false`: 176

## encounter_primary_reason_non_CV_Disease_f5a_w1mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 3325
  - `true`: 685
  - `false`: 435

## encounter_primary_reason_non_CV_Disease_f5a_w3mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 2632
  - `true`: 1133
  - `false`: 680

## encounter_primary_reason_non_CV_Disease_f5a_w6mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 2291
  - `true`: 1332
  - `false`: 822

## encounter_primary_reason_non_CV_Disease_f5a_w1a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 2073
  - `true`: 1476
  - `false`: 896

## encounter_primary_reason_non_CV_Disease_f5a_w3a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 1864
  - `true`: 1607
  - `false`: 974

## encounter_primary_reason_non_CV_Disease_f5a_w5a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 1820
  - `true`: 1637
  - `false`: 988

## encounter_primary_reason_number_of_days_to_rehosp_for_non_CV_f5a_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -440.0 / 1800.0
- mean/std: -225.7019 / 309.0627201258508
- quantiles: {'0.05': -433.75, '0.25': -433.75, '0.5': -433.75, '0.75': 20.0, '0.95': 269.8000000000002}

## encounter_primary_reason_number_of_non_CV_rehospitalizations_5a_f5a_count

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -13.0 / 50.0
- mean/std: 4.2461 / 8.0938916077754
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 1.0, '0.75': 5.0, '0.95': 21.0}

## encounter_primary_reason_renal_complications_f5a_w7d_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 3983
  - `false`: 458
  - 1 other distinct value(s) suppressed, covering 4 row(s) (count below 5 and/or ranked beyond top 20)

## encounter_primary_reason_renal_complications_f5a_w1mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 3325
  - `false`: 1096
  - `true`: 24

## encounter_primary_reason_renal_complications_f5a_w3mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 2632
  - `false`: 1770
  - `true`: 43

## encounter_primary_reason_renal_complications_f5a_w6mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 2291
  - `false`: 2102
  - `true`: 52

## encounter_primary_reason_renal_complications_f5a_w1a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `false`: 2312
  - `Missing`: 2073
  - `true`: 60

## encounter_primary_reason_renal_complications_f5a_w3a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `false`: 2517
  - `Missing`: 1864
  - `true`: 64

## encounter_primary_reason_renal_complications_f5a_w5a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `false`: 2561
  - `Missing`: 1820
  - `true`: 64

## encounter_primary_reason_number_of_renal_rehospitalizations_5a_f5a_count

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 20.0
- mean/std: 0.2065 / 1.020096313759256
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 1.0}

## cause_of_death_isCV_f5a_w7d_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 4308
  - `false`: 124
  - `true`: 13

## cause_of_death_isCV_f5a_w1mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 4124
  - `false`: 304
  - `true`: 17

## cause_of_death_isCV_f5a_w3mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 3887
  - `false`: 534
  - `true`: 24

## cause_of_death_isCV_f5a_w6mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 3675
  - `false`: 738
  - `true`: 32

## cause_of_death_isCV_f5a_w1a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 3459
  - `false`: 947
  - `true`: 39

## cause_of_death_isCV_f5a_w3a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 3086
  - `false`: 1302
  - `true`: 57

## cause_of_death_isCV_f5a_w5a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 2926
  - `false`: 1457
  - `true`: 62

## cause_of_death_isRenal_f5a_w7d_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 4308
  - `false`: 137

## cause_of_death_isRenal_f5a_w1mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 4124
  - `false`: 321

## cause_of_death_isRenal_f5a_w3mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 3887
  - `false`: 558

## cause_of_death_isRenal_f5a_w6mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 3675
  - `false`: 770

## cause_of_death_isRenal_f5a_w1a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 3459
  - `false`: 986

## cause_of_death_isRenal_f5a_w3a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 3086
  - `false`: 1359

## cause_of_death_isRenal_f5a_w5a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 2926
  - `false`: 1519

## cause_of_death_isNonRenalAndNonCV_f5a_w7d_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 4308
  - `false`: 137

## cause_of_death_isNonRenalAndNonCV_f5a_w1mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 4124
  - `false`: 321

## cause_of_death_isNonRenalAndNonCV_f5a_w3mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 3887
  - `false`: 558

## cause_of_death_isNonRenalAndNonCV_f5a_w6mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 3675
  - `false`: 770

## cause_of_death_isNonRenalAndNonCV_f5a_w1a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 3459
  - `false`: 986

## cause_of_death_isNonRenalAndNonCV_f5a_w3a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 3086
  - `false`: 1359

## cause_of_death_isNonRenalAndNonCV_f5a_w5a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 2926
  - `false`: 1519

## cause_of_death_isAllCause_f5a_w7d_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 4308
  - `true`: 124
  - `false`: 13

## cause_of_death_isAllCause_f5a_w1mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 4124
  - `true`: 304
  - `false`: 17

## cause_of_death_isAllCause_f5a_w3mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 3887
  - `true`: 534
  - `false`: 24

## cause_of_death_isAllCause_f5a_w6mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 3675
  - `true`: 738
  - `false`: 32

## cause_of_death_isAllCause_f5a_w1a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 3459
  - `true`: 947
  - `false`: 39

## cause_of_death_isAllCause_f5a_w3a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 3086
  - `true`: 1302
  - `false`: 57

## cause_of_death_isAllCause_f5a_w5a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 2926
  - `true`: 1457
  - `false`: 62

## cause_of_death_number_of_days_to_death_for_all_cause_f5a_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -460.0 / 1900.0
- mean/std: -180.8495 / 465.8547630215122
- quantiles: {'0.05': -453.25, '0.25': -453.25, '0.5': -453.25, '0.75': 38.0, '0.95': 894.4000000000005}

## eGFR_2021_ckd_epi_creatinine

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 20.0 / 240.0
- mean/std: 60.4753 / 25.849684976520514
- quantiles: {'0.05': 27.6183966, '0.25': 38.86505, '0.5': 56.230212, '0.75': 78.517054, '0.95': 105.77368680000001}

## ckd_severity_from_calculated_egfr

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 5
- top values (shown only where count ≥ 5):
  - `mildly_decreased`: 1325
  - `moderate_to_severe_decrease`: 1191
  - `mild_to_moderate_decrease`: 909
  - `normal_or_high`: 731
  - `severe_decrease`: 289

## ckd_severity_calculated_or_measured

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 5
- top values (shown only where count ≥ 5):
  - `mildly_decreased`: 1325
  - `moderate_to_severe_decrease`: 1191
  - `mild_to_moderate_decrease`: 909
  - `normal_or_high`: 731
  - `severe_decrease`: 289

## maggic_total_score

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -7.3 / 44.0
- mean/std: -0.9282 / 12.843906457404596
- quantiles: {'0.05': -7.25, '0.25': -7.25, '0.5': -7.25, '0.75': -7.25, '0.95': 29.0}

## med_acei

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 2734
  - `true`: 1711

## med_activeDuringEncounter_ace_inhibitors_arb_use

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 2433
  - `false`: 2012

## med_activeDuringEncounter_bb

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 2854
  - `false`: 1591

## med_anti_coag

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 3712
  - `false`: 733

## med_anti_plat

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 2619
  - `true`: 1826

## med_antiarrhytmic

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 3711
  - `true`: 734

## med_antiinfl

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 4392
  - `true`: 53

## med_arb

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 3392
  - `true`: 1053

## med_ari

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 4445

## med_arni

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 4068
  - `true`: 377

## med_bb

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 3182
  - `false`: 1263

## med_ccb

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 3339
  - `true`: 1106

## med_cortico_syst

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 3421
  - `true`: 1024

## med_digitalis

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 3354
  - `true`: 1091

## med_diuretics

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 4258
  - `false`: 187

## med_diuretics_loop

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 4208
  - `false`: 237

## med_inotropes

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 3484
  - `true`: 961

## med_insulins

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 3240
  - `true`: 1205

## med_ivabradine

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 4337
  - `true`: 108

## med_ll

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 2244
  - `false`: 2201

## med_mra

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 2549
  - `false`: 1896

## med_oral_antidiabetic

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 2864
  - `true`: 1581

## med_platelet

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 2590
  - `true`: 1855

## med_potassium_binders

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 4264
  - `true`: 181

## med_rasi

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 2586
  - `false`: 1859

## med_rdoad

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 2729
  - `true`: 1716

## med_rdoad_syst

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 4423
  - `true`: 22

## med_sglt2i

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 3509
  - `true`: 936

## med_thrombolytic

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 4418
  - `true`: 27

## med_vasodil

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 3378
  - `true`: 1067

## med_acei_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 2779
  - `true`: 1666

## med_anti_coag_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 3142
  - `false`: 1303

## med_anti_plat_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 2840
  - `true`: 1605

## med_antiarrhytmic_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 3913
  - `true`: 532

## med_antiinfl_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 4260
  - `true`: 185

## med_arb_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 3383
  - `true`: 1062

## med_ari_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 4445

## med_arni_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 4140
  - `true`: 305

## med_bb_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 2580
  - `false`: 1865

## med_ccb_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 3000
  - `true`: 1445

## med_cortico_syst_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 3068
  - `true`: 1377

## med_digitalis_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 3578
  - `true`: 867

## med_diuretics_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 3612
  - `false`: 833

## med_diuretics_loop_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 3552
  - `false`: 893

## med_inotropes_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 2945
  - `true`: 1500

## med_insulins_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 3372
  - `true`: 1073

## med_ivabradine_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 4333
  - `true`: 112

## med_ll_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 2433
  - `true`: 2012

## med_mra_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 2784
  - `true`: 1661

## med_oral_antidiabetic_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 3319
  - `true`: 1126

## med_platelet_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 2819
  - `true`: 1626

## med_potassium_binders_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 4223
  - `true`: 222

## med_rasi_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 2318
  - `false`: 2127

## med_rdoad_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 2897
  - `true`: 1548

## med_rdoad_syst_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 4403
  - `true`: 42

## med_sglt2i_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 3986
  - `true`: 459

## med_thrombolytic_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 4401
  - `true`: 44

## med_vasodil_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 3011
  - `true`: 1434

## conditions_af

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 2247
  - `true`: 2198

## conditions_aidshiv

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 4403
  - `true`: 42

## conditions_ap

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 3787
  - `true`: 658

## conditions_ckd_chronic

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 2654
  - `true`: 1791

## conditions_cm

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 3547
  - `true`: 898

## conditions_copd

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 3663
  - `true`: 782

## conditions_dem

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 4341
  - `true`: 104

## conditions_dep

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 4332
  - `true`: 113

## conditions_devices

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 3386
  - `true`: 1059

## conditions_dia

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 4445

## conditions_diabetes

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 3014
  - `true`: 1431

## conditions_dysl

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 3619
  - `true`: 826

## conditions_hf

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 4432
  - `false`: 13

## conditions_hyp

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 2427
  - `false`: 2018

## conditions_hyperthyroid

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 4317
  - `true`: 128

## conditions_hypothyroid

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 4119
  - `true`: 326

## conditions_ibd

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 4242
  - `true`: 203

## conditions_ihd

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 2340
  - `true`: 2105

## conditions_ld

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 4099
  - `true`: 346

## conditions_mc

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 3554
  - `true`: 891

## conditions_mi

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 3710
  - `true`: 735

## conditions_myocarditis

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 4430
  - `true`: 15

## conditions_osa

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 4122
  - `true`: 323

## conditions_pad

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 3602
  - `true`: 843

## conditions_pericardial

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 4270
  - `true`: 175

## conditions_rd

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 4202
  - `true`: 243

## conditions_revasc

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 3713
  - `true`: 732

## conditions_stroke

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 4027
  - `true`: 418

## conditions_substance_abuse

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 3796
  - `true`: 649

## conditions_tia

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 4141
  - `true`: 304

## conditions_vd

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 2525
  - `true`: 1920
