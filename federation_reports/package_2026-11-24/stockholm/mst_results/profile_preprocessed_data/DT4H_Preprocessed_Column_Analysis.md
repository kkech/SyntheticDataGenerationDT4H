# Column Analysis

Total rows: 802
Total columns: 253

## patient_demographics_gender

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `male`: 522
  - `female`: 280

## patient_demographics_age

- dtype: `Int32` (numeric)
- nulls: 0 (0.00%)
- min/max: 19.0 / 110.0
- mean/std: 72.5224 / 13.634890253397389
- quantiles: {'0.05': 48.0, '0.25': 65.0, '0.5': 74.0, '0.75': 82.0, '0.95': 91.0}

## encounters_encounterClass

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `IMP`: 802

## encounters_admissionYear

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 13
- top values (shown only where count ≥ 5):
  - `2022`: 105
  - `2018`: 91
  - `2019`: 80
  - `2021`: 77
  - `2014`: 76
  - `2015`: 69
  - `2016`: 68
  - `2020`: 63
  - `2017`: 55
  - `2012`: 53
  - `2013`: 51
  - `2011`: 12
  - 1 other distinct value(s) suppressed, covering 2 row(s) (count below 5 and/or ranked beyond top 20)

## encounters_lengthOfStay

- dtype: `Int32` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 240.0
- mean/std: 5.8691 / 9.6975004255416
- quantiles: {'0.05': 1.0, '0.25': 2.0, '0.5': 4.0, '0.75': 7.0, '0.95': 16.0}

## encounters_numOfPreviousHFStays_count

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -240.0 / 940.0
- mean/std: 75.0087 / 107.20512464518985
- quantiles: {'0.05': 0.0, '0.25': 12.0, '0.5': 40.5, '0.75': 95.75, '0.95': 284.4999999999993}

## vital_signs_weight_value_p6mo_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -15.0 / 240.0
- mean/std: 54.9988 / 47.03316801149052
- quantiles: {'0.05': -14.700000000000003, '0.25': -14.700000000000003, '0.5': 70.0, '0.75': 87.0, '0.95': 116.94999999999993}

## vital_signs_weight_value_p6mo_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -9.5 / 230.0
- mean/std: 55.2553 / 44.01328663848931
- quantiles: {'0.05': -9.5, '0.25': -9.5, '0.5': 69.2, '0.75': 85.2, '0.95': 111.97999999999998}

## vital_signs_height_value_p1a_avg

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 67.0 / 200.0
- mean/std: 117.9294 / 52.98810235583806
- quantiles: {'0.05': 67.75, '0.25': 67.75, '0.5': 67.75, '0.75': 173.0, '0.95': 185.0}

## vital_signs_weight_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -9.5 / 230.0
- mean/std: 45.4088 / 46.871503172465594
- quantiles: {'0.05': -9.5, '0.25': -9.5, '0.5': 63.5, '0.75': 82.0, '0.95': 109.0}

## vital_signs_height_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 66.0 / 200.0
- mean/std: 100.1650 / 49.90885513398515
- quantiles: {'0.05': 66.75, '0.25': 66.75, '0.5': 66.75, '0.75': 164.75, '0.95': 183.0}

## vital_signs_diastolicBp_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 8.5 / 130.0
- mean/std: 66.7606 / 20.568907207512474
- quantiles: {'0.05': 8.5, '0.25': 60.0, '0.5': 70.0, '0.75': 80.0, '0.95': 90.0}

## vital_signs_diastolicBp_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -1.3 / 170.0
- mean/std: 71.8653 / 26.907206483541593
- quantiles: {'0.05': -1.25, '0.25': 60.25, '0.5': 75.0, '0.75': 86.0, '0.95': 110.0}

## vital_signs_heartRate_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 5.0 / 170.0
- mean/std: 71.1347 / 24.76217607045466
- quantiles: {'0.05': 5.0, '0.25': 62.0, '0.5': 72.0, '0.75': 86.0, '0.95': 103.0}

## vital_signs_heartRate_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -24.0 / 220.0
- mean/std: 79.1930 / 38.21130761415396
- quantiles: {'0.05': -23.75, '0.25': 65.0, '0.5': 82.0, '0.75': 100.0, '0.95': 130.94999999999993}

## vital_signs_systolicBp_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 28.0 / 230.0
- mean/std: 115.8769 / 32.7400007787401
- quantiles: {'0.05': 28.75, '0.25': 104.0, '0.5': 120.0, '0.75': 134.0, '0.95': 160.0}

## vital_signs_systolicBp_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 16.0 / 260.0
- mean/std: 123.8825 / 41.44098599276697
- quantiles: {'0.05': 16.25, '0.25': 110.0, '0.5': 126.0, '0.75': 145.0, '0.95': 184.0}

## vital_signs_oxygenSaturation_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -18.0 / 100.0
- mean/std: 83.3603 / 33.99355265120862
- quantiles: {'0.05': -17.5, '0.25': 92.0, '0.5': 95.0, '0.75': 97.0, '0.95': 99.0}

## vital_signs_oxygenSaturation_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 60.0 / 100.0
- mean/std: 91.9813 / 11.105707019533238
- quantiles: {'0.05': 60.0, '0.25': 93.0, '0.5': 96.0, '0.75': 98.0, '0.95': 99.0}

## lab_results_hemoglobin_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 9.0 / 190.0
- mean/std: 118.5461 / 40.00431925833344
- quantiles: {'0.05': 9.0, '0.25': 111.0, '0.5': 128.0, '0.75': 142.0, '0.95': 160.0}

## lab_results_hemoglobin_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 46.0 / 190.0
- mean/std: 119.1253 / 29.482429990037126
- quantiles: {'0.05': 46.5, '0.25': 108.0, '0.5': 124.5, '0.75': 138.0, '0.95': 156.0}

## lab_results_ferritin_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -290.0 / 1300.0
- mean/std: -247.3647 / 168.35708363762524
- quantiles: {'0.05': -286.75, '0.25': -286.75, '0.5': -286.75, '0.75': -286.75, '0.95': 87.89999999999986}

## lab_results_ferritin_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -390.0 / 1600.0
- mean/std: -335.6627 / 193.31518266946787
- quantiles: {'0.05': -381.75, '0.25': -381.75, '0.5': -381.75, '0.75': -381.75, '0.95': 87.89999999999986}

## lab_results_tfs_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -17.0 / 87.0
- mean/std: -14.9380 / 8.25357724337187
- quantiles: {'0.05': -16.75, '0.25': -16.75, '0.5': -16.75, '0.75': -16.75, '0.95': 5.0}

## lab_results_tfs_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -17.0 / 87.0
- mean/std: -14.9168 / 8.33519694036611
- quantiles: {'0.05': -16.75, '0.25': -16.75, '0.5': -16.75, '0.75': -16.75, '0.95': 5.0}

## lab_results_ntProBnp_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -8800.0 / 35000.0
- mean/std: 35.7216 / 10162.700406425965
- quantiles: {'0.05': -8716.25, '0.25': -8716.25, '0.5': 445.0, '0.75': 4445.0, '0.95': 20394.999999999993}

## lab_results_ntProBnp_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -8800.0 / 35000.0
- mean/std: -143.1936 / 9919.0449645965
- quantiles: {'0.05': -8716.25, '0.25': -8716.25, '0.5': 463.5, '0.75': 4195.0, '0.95': 19994.999999999993}

## lab_results_crpNonHs_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -93.0 / 380.0
- mean/std: 11.2304 / 62.582061559471924
- quantiles: {'0.05': -93.0, '0.25': 1.0, '0.5': 8.0, '0.75': 27.75, '0.95': 120.94999999999993}

## lab_results_crpNonHs_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -85.0 / 350.0
- mean/std: 12.6329 / 58.653258828793824
- quantiles: {'0.05': -84.75, '0.25': 2.0, '0.5': 9.0, '0.75': 30.75, '0.95': 98.94999999999993}

## lab_results_tropTnHs_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -8700.0 / 35000.0
- mean/std: -2740.3089 / 4389.5553947153785
- quantiles: {'0.05': -8649.575, '0.25': -8649.575, '0.5': 22.0, '0.75': 52.0, '0.95': 319.8499999999998}

## lab_results_tropTnHs_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -2000.0 / 7800.0
- mean/std: -505.8739 / 1153.9355302957995
- quantiles: {'0.05': -1939.925, '0.25': -1939.925, '0.5': 24.0, '0.75': 62.75, '0.95': 527.249999999999}

## lab_results_triGly_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -0.65 / 5.2
- mean/std: -0.3883 / 0.752203112558011
- quantiles: {'0.05': -0.6499999999999999, '0.25': -0.6499999999999999, '0.5': -0.6499999999999999, '0.75': -0.6499999999999999, '0.95': 1.3}

## lab_results_triGly_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -0.48 / 4.2
- mean/std: -0.2470 / 0.6627939742375759
- quantiles: {'0.05': -0.48, '0.25': -0.48, '0.5': -0.48, '0.75': -0.48, '0.95': 1.3}

## lab_results_cholTot_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.8 / 7.8
- mean/std: 1.1446 / 1.0495596515176395
- quantiles: {'0.05': 0.8000000000000003, '0.25': 0.8000000000000003, '0.5': 0.8000000000000003, '0.75': 0.8000000000000003, '0.95': 3.894999999999993}

## lab_results_cholTot_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.8 / 7.8
- mean/std: 1.1448 / 1.0497624753562498
- quantiles: {'0.05': 0.8000000000000003, '0.25': 0.8000000000000003, '0.5': 0.8000000000000003, '0.75': 0.8000000000000003, '0.95': 3.9}

## lab_results_hdl_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -0.5 / 3.0
- mean/std: -0.3170 / 0.537378591878661
- quantiles: {'0.05': -0.5, '0.25': -0.5, '0.5': -0.5, '0.75': -0.5, '0.95': 1.1949999999999932}

## lab_results_hdl_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -0.6 / 3.0
- mean/std: -0.4064 / 0.5668732478467181
- quantiles: {'0.05': -0.6, '0.25': -0.6, '0.5': -0.6, '0.75': -0.6, '0.95': 1.1949999999999932}

## lab_results_ldl_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -0.56 / 5.2
- mean/std: -0.2594 / 0.892977780746801
- quantiles: {'0.05': -0.5500000000000002, '0.25': -0.5500000000000002, '0.5': -0.5500000000000002, '0.75': -0.5500000000000002, '0.95': 2.0}

## lab_results_ldl_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -0.56 / 5.2
- mean/std: -0.2596 / 0.8921079081306902
- quantiles: {'0.05': -0.5500000000000002, '0.25': -0.5500000000000002, '0.5': -0.5500000000000002, '0.75': -0.5500000000000002, '0.95': 2.0}

## lab_results_potassium_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 1.5 / 6.9
- mean/std: 4.0020 / 0.8683898401176924
- quantiles: {'0.05': 1.525, '0.25': 3.8, '0.5': 4.1, '0.75': 4.4, '0.95': 5.1}

## lab_results_potassium_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 1.7 / 6.0
- mean/std: 3.9153 / 0.7430420980190287
- quantiles: {'0.05': 1.7000000000000002, '0.25': 3.7, '0.5': 4.0, '0.75': 4.3, '0.95': 4.8}

## lab_results_sodium_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 100.0 / 160.0
- mean/std: 137.2007 / 9.144059937944242
- quantiles: {'0.05': 105.0, '0.25': 137.0, '0.5': 140.0, '0.75': 142.0, '0.95': 145.0}

## lab_results_sodium_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 120.0 / 160.0
- mean/std: 138.6291 / 5.771314842043633
- quantiles: {'0.05': 120.75, '0.25': 137.0, '0.5': 140.0, '0.75': 142.0, '0.95': 145.0}

## lab_results_albuminBS_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 4.0 / 39.0
- mean/std: 9.3628 / 10.137839816872473
- quantiles: {'0.05': 4.0, '0.25': 4.0, '0.5': 4.0, '0.75': 4.0, '0.95': 31.0}

## lab_results_albuminBS_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 5.2 / 39.0
- mean/std: 10.5792 / 10.06325792777474
- quantiles: {'0.05': 5.25, '0.25': 5.25, '0.5': 5.25, '0.75': 5.25, '0.95': 32.0}

## lab_results_validSerumCreatinine_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -1.6 / 23.0
- mean/std: 10.4739 / 5.442370094112767
- quantiles: {'0.05': -1.52685, '0.25': 8.0301, '0.5': 10.6314, '0.75': 13.3458, '0.95': 19.786844999999992}

## lab_results_valideGFR_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -14.0 / 90.0
- mean/std: 23.6072 / 37.031864135726
- quantiles: {'0.05': -13.75, '0.25': -13.75, '0.5': 22.0, '0.75': 58.0, '0.95': 83.0}

## lab_results_valideGFR_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -15.0 / 90.0
- mean/std: 24.1035 / 38.43636065634879
- quantiles: {'0.05': -15.0, '0.25': -15.0, '0.5': 24.0, '0.75': 60.0, '0.95': 85.0}

## symptoms_Ankle_swelling_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 802

## symptoms_Ascites_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 802

## symptoms_Breathlessness_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 801
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## symptoms_Cardiac_murmur_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 802

## symptoms_Chest_pain_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 802

## symptoms_Cheyne_stokes_respiration_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 802

## symptoms_Depression_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 802

## symptoms_Dizziness_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 802

## symptoms_Elevated_jugular_venous_pressure_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 802

## symptoms_Fatigue_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 802

## symptoms_Hepatojugular_reflux_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 802

## symptoms_Hepatomegaly_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 802

## symptoms_Intermittent_claudication_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 802

## symptoms_Irregular_pulse_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 802

## symptoms_Loss_of_appetite_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 802

## symptoms_Nocturnal_cough_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 802

## symptoms_Oliguria_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 802

## symptoms_Orthopnoea_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 802

## symptoms_Palpitations_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 802

## symptoms_Paroxysmal_nocturnal_dyspnea_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 802

## symptoms_Peripheral_edema_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 788
  - `true`: 14

## symptoms_Pleural_effusion_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 802

## symptoms_Pulmonary_crepitations_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 802

## symptoms_Reduced_exercise_tolerance_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 802

## symptoms_Syncope_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 802

## symptoms_Tachycardia_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 802

## symptoms_Tachypnoea_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 802

## symptoms_Third_heart_sound_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 802

## symptoms_Weight_gain_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 802

## symptoms_Weight_loss_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 802

## electrocardiographs_ecg_st_pET

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `Missing`: 802

## electrocardiographs_ecg_ischemia_without_st_pET

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `Missing`: 802

## electrocardiographs_ecg_type_of_rhythms_pET_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 5
- top values (shown only where count ≥ 5):
  - `Missing`: 738
  - `LA17718-0`: 32
  - `LA17093-8`: 16
  - `LA17084-7`: 14
  - 1 other distinct value(s) suppressed, covering 2 row(s) (count below 5 and/or ranked beyond top 20)

## electrocardiographs_ecg_type_of_rhythms_pET_last

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 5
- top values (shown only where count ≥ 5):
  - `Missing`: 738
  - `LA17718-0`: 31
  - `LA17093-8`: 17
  - `LA17084-7`: 14
  - 1 other distinct value(s) suppressed, covering 2 row(s) (count below 5 and/or ranked beyond top 20)

## smoking_status_smoker_last

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 802

## smoking_status_formerSmoker_last

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `Missing`: 802

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
- mean/std: 2.6209 / 0.6168342517507901
- quantiles: {'0.05': 1.0500000000000043, '0.25': 2.0, '0.5': 3.0, '0.75': 3.0, '0.95': 3.0}

## nyha_nyha_pET

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 1.0 / 3.0
- mean/std: 2.6097 / 0.6087626258610859
- quantiles: {'0.05': 1.0, '0.25': 2.0, '0.5': 3.0, '0.75': 3.0, '0.95': 3.0}

## vital_signs_systolicBpDuringEncounter_value_pET

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 69.0 / 230.0
- mean/std: 122.7915 / 22.12843958945778
- quantiles: {'0.05': 90.0, '0.25': 110.0, '0.5': 120.0, '0.75': 135.0, '0.95': 160.0}

## vital_signs_bmi_value_pET

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 18.0 / 25.0
- mean/std: 21.7927 / 1.8698867100680474
- quantiles: {'0.05': 18.905743680004385, '0.25': 20.186238417670396, '0.5': 21.743702857855464, '0.75': 23.398181056379606, '0.95': 24.718109418765835}

## lab_results_creatBS_value_p3a_avg

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 3.5 / 190.0
- mean/std: 18.1150 / 20.889083960680995
- quantiles: {'0.05': 6.72945, '0.25': 9.048, '0.5': 11.639875, '0.75': 16.074974464285717, '0.95': 70.12938958333307}

## lab_results_validSerumCreatinine_value_pET

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 3.5 / 23.0
- mean/std: 11.5930 / 3.9545920974917363
- quantiles: {'0.05': 6.4467, '0.25': 8.8218, '0.5': 10.8576, '0.75': 13.572000000000001, '0.95': 20.017765}

## hyperkalemia_severity_categorizedValue

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 5
- top values (shown only where count ≥ 5):
  - `normal`: 722
  - `Missing`: 53
  - `mild`: 19
  - `moderate`: 7
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## ckd_severity_categorizedValue

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 7
- top values (shown only where count ≥ 5):
  - `Missing`: 356
  - `mildly_decreased`: 187
  - `mild_to_moderate_decrease`: 105
  - `moderate_to_severe_decrease`: 63
  - `severe_decrease`: 48
  - `normal_or_high`: 27
  - `kidney_failure`: 16

## conditions_heartFailure_timeFromEarliest_first

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 280.0
- mean/std: 44.4800 / 65.21534653911789
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 5.0, '0.75': 77.0, '0.95': 189.94999999999993}

## conditions_heart_failure_hf_within_18mo_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `true`: 802

## conditions_heart_failure_occurred_prior_to_18_months_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 473
  - `true`: 329

## encounter_primary_reason_HF_Disease_f5a_w7d_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `false`: 536
  - `true`: 161
  - `Missing`: 105

## encounter_primary_reason_HF_Disease_f5a_w1mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `false`: 558
  - `true`: 173
  - `Missing`: 71

## encounter_primary_reason_HF_Disease_f5a_w3mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `false`: 566
  - `true`: 184
  - `Missing`: 52

## encounter_primary_reason_HF_Disease_f5a_w6mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `false`: 572
  - `true`: 188
  - `Missing`: 42

## encounter_primary_reason_HF_Disease_f5a_w1a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `false`: 578
  - `true`: 192
  - `Missing`: 32

## encounter_primary_reason_HF_Disease_f5a_w3a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `false`: 587
  - `true`: 195
  - `Missing`: 20

## encounter_primary_reason_HF_Disease_f5a_w5a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `false`: 588
  - `true`: 195
  - `Missing`: 19

## encounter_primary_reason_number_of_days_to_rehosp_for_heart_failure_f5a_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -160.0 / 620.0
- mean/std: -112.3270 / 83.5183689757948
- quantiles: {'0.05': -154.75, '0.25': -154.75, '0.5': -154.75, '0.75': -154.75, '0.95': 0.9499999999999318}

## encounter_primary_reason_number_of_HF_rehospitalizations_5a_f5a_count

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -12.0 / 47.0
- mean/std: 4.1646 / 6.740487754888427
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 2.0, '0.75': 6.0, '0.95': 17.0}

## encounter_primary_reason_CV_Disease_f5a_w7d_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `true`: 513
  - `false`: 184
  - `Missing`: 105

## encounter_primary_reason_CV_Disease_f5a_w1mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `true`: 542
  - `false`: 189
  - `Missing`: 71

## encounter_primary_reason_CV_Disease_f5a_w3mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `true`: 559
  - `false`: 191
  - `Missing`: 52

## encounter_primary_reason_CV_Disease_f5a_w6mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `true`: 567
  - `false`: 193
  - `Missing`: 42

## encounter_primary_reason_CV_Disease_f5a_w1a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `true`: 575
  - `false`: 195
  - `Missing`: 32

## encounter_primary_reason_CV_Disease_f5a_w3a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `true`: 581
  - `false`: 201
  - `Missing`: 20

## encounter_primary_reason_CV_Disease_f5a_w5a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `true`: 582
  - `false`: 201
  - `Missing`: 19

## encounter_primary_reason_number_of_days_to_rehosp_for_CV_f5a_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -420.0 / 1700.0
- mean/std: -103.0274 / 209.3892712437672
- quantiles: {'0.05': -418.0, '0.25': -418.0, '0.5': 0.0, '0.75': 0.0, '0.95': 28.899999999999864}

## encounter_primary_reason_number_of_CV_rehospitalizations_5a_f5a_count

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -13.0 / 50.0
- mean/std: 11.0642 / 13.459280753290473
- quantiles: {'0.05': -12.5, '0.25': 3.0, '0.5': 8.0, '0.75': 17.0, '0.95': 39.0}

## encounter_primary_reason_non_CV_Disease_f5a_w7d_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `false`: 513
  - `true`: 184
  - `Missing`: 105

## encounter_primary_reason_non_CV_Disease_f5a_w1mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `false`: 542
  - `true`: 189
  - `Missing`: 71

## encounter_primary_reason_non_CV_Disease_f5a_w3mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `false`: 559
  - `true`: 191
  - `Missing`: 52

## encounter_primary_reason_non_CV_Disease_f5a_w6mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `false`: 567
  - `true`: 193
  - `Missing`: 42

## encounter_primary_reason_non_CV_Disease_f5a_w1a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `false`: 575
  - `true`: 195
  - `Missing`: 32

## encounter_primary_reason_non_CV_Disease_f5a_w3a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `false`: 581
  - `true`: 201
  - `Missing`: 20

## encounter_primary_reason_non_CV_Disease_f5a_w5a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `false`: 582
  - `true`: 201
  - `Missing`: 19

## encounter_primary_reason_number_of_days_to_rehosp_for_non_CV_f5a_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -210.0 / 820.0
- mean/std: -147.5982 / 108.63111525178677
- quantiles: {'0.05': -203.75, '0.25': -203.75, '0.5': -203.75, '0.75': -50.9375, '0.95': 0.0}

## encounter_primary_reason_number_of_non_CV_rehospitalizations_5a_f5a_count

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -13.0 / 49.0
- mean/std: 6.4174 / 10.593740045737778
- quantiles: {'0.05': -12.25, '0.25': 0.0, '0.5': 3.0, '0.75': 10.0, '0.95': 28.949999999999932}

## encounter_primary_reason_renal_complications_f5a_w7d_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `false`: 673
  - `Missing`: 105
  - `true`: 24

## encounter_primary_reason_renal_complications_f5a_w1mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `false`: 707
  - `Missing`: 71
  - `true`: 24

## encounter_primary_reason_renal_complications_f5a_w3mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `false`: 726
  - `Missing`: 52
  - `true`: 24

## encounter_primary_reason_renal_complications_f5a_w6mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `false`: 736
  - `Missing`: 42
  - `true`: 24

## encounter_primary_reason_renal_complications_f5a_w1a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `false`: 746
  - `Missing`: 32
  - `true`: 24

## encounter_primary_reason_renal_complications_f5a_w3a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `false`: 757
  - `true`: 25
  - `Missing`: 20

## encounter_primary_reason_renal_complications_f5a_w5a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `false`: 758
  - `true`: 25
  - `Missing`: 19

## encounter_primary_reason_number_of_renal_rehospitalizations_5a_f5a_count

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -7.8 / 31.0
- mean/std: 1.0193 / 3.5439677145198814
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 6.0}

## cause_of_death_isCV_f5a_w7d_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 781
  - `true`: 15
  - `false`: 6

## cause_of_death_isCV_f5a_w1mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 757
  - `true`: 29
  - `false`: 16

## cause_of_death_isCV_f5a_w3mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 703
  - `true`: 59
  - `false`: 40

## cause_of_death_isCV_f5a_w6mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 668
  - `true`: 82
  - `false`: 52

## cause_of_death_isCV_f5a_w1a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 606
  - `true`: 112
  - `false`: 84

## cause_of_death_isCV_f5a_w3a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 500
  - `true`: 168
  - `false`: 134

## cause_of_death_isCV_f5a_w5a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 413
  - `true`: 212
  - `false`: 177

## cause_of_death_number_of_days_to_death_for_CV_f5a_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -450.0 / 1800.0
- mean/std: -193.6901 / 504.4898337489394
- quantiles: {'0.05': -448.75, '0.25': -448.75, '0.5': -448.75, '0.75': 6.0, '0.95': 1132.199999999999}

## cause_of_death_isRenal_f5a_w7d_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 781
  - `false`: 21

## cause_of_death_isRenal_f5a_w1mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 757
  - `false`: 45

## cause_of_death_isRenal_f5a_w3mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 703
  - `false`: 96
  - 1 other distinct value(s) suppressed, covering 3 row(s) (count below 5 and/or ranked beyond top 20)

## cause_of_death_isRenal_f5a_w6mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 668
  - `false`: 130
  - 1 other distinct value(s) suppressed, covering 4 row(s) (count below 5 and/or ranked beyond top 20)

## cause_of_death_isRenal_f5a_w1a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 606
  - `false`: 191
  - `true`: 5

## cause_of_death_isRenal_f5a_w3a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 500
  - `false`: 295
  - `true`: 7

## cause_of_death_isRenal_f5a_w5a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 413
  - `false`: 382
  - `true`: 7

## cause_of_death_isNonRenalAndNonCV_f5a_w7d_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 781
  - `false`: 15
  - `true`: 6

## cause_of_death_isNonRenalAndNonCV_f5a_w1mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 757
  - `false`: 29
  - `true`: 16

## cause_of_death_isNonRenalAndNonCV_f5a_w3mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 703
  - `false`: 63
  - `true`: 36

## cause_of_death_isNonRenalAndNonCV_f5a_w6mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 668
  - `false`: 87
  - `true`: 47

## cause_of_death_isNonRenalAndNonCV_f5a_w1a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 606
  - `false`: 119
  - `true`: 77

## cause_of_death_isNonRenalAndNonCV_f5a_w3a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 500
  - `false`: 185
  - `true`: 117

## cause_of_death_isNonRenalAndNonCV_f5a_w5a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 413
  - `false`: 240
  - `true`: 149

## cause_of_death_number_of_days_to_death_for_non_renal_and_non_CV_f5a_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -460.0 / 1900.0
- mean/std: -266.8167 / 455.6458998910533
- quantiles: {'0.05': -455.0, '0.25': -455.0, '0.5': -455.0, '0.75': -455.0, '0.95': 820.6999999999996}

## cause_of_death_isAllCause_f5a_w7d_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 781
  - `false`: 21

## cause_of_death_isAllCause_f5a_w1mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 757
  - `false`: 45

## cause_of_death_isAllCause_f5a_w3mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 703
  - `false`: 98
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## cause_of_death_isAllCause_f5a_w6mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 668
  - `false`: 133
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## cause_of_death_isAllCause_f5a_w1a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 606
  - `false`: 194
  - 1 other distinct value(s) suppressed, covering 2 row(s) (count below 5 and/or ranked beyond top 20)

## cause_of_death_isAllCause_f5a_w3a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 500
  - `false`: 292
  - `true`: 10

## cause_of_death_isAllCause_f5a_w5a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 413
  - `false`: 368
  - `true`: 21

## eGFR_2021_ckd_epi_creatinine

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 22.0 / 130.0
- mean/std: 67.0761 / 22.972531291210956
- quantiles: {'0.05': 30.99572615, '0.25': 48.25591075, '0.5': 66.63973899999999, '0.75': 85.83066500000001, '0.95': 102.4398268}

## ckd_severity_from_calculated_egfr

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 5
- top values (shown only where count ≥ 5):
  - `mildly_decreased`: 326
  - `normal_or_high`: 173
  - `mild_to_moderate_decrease`: 140
  - `moderate_to_severe_decrease`: 140
  - `severe_decrease`: 23

## ckd_severity_calculated_or_measured

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 5
- top values (shown only where count ≥ 5):
  - `mildly_decreased`: 326
  - `normal_or_high`: 173
  - `mild_to_moderate_decrease`: 140
  - `moderate_to_severe_decrease`: 140
  - `severe_decrease`: 23

## med_acei

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 802

## med_activeDuringEncounter_ace_inhibitors_arb_use

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 651
  - `false`: 151

## med_activeDuringEncounter_bb

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 673
  - `false`: 129

## med_anti_coag

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 802

## med_anti_plat

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 802

## med_antiarrhytmic

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 802

## med_antiinfl

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 802

## med_arb

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 802

## med_ari

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 802

## med_arni

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 802

## med_bb

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 802

## med_ccb

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 802

## med_cortico_syst

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 802

## med_digitalis

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 802

## med_diuretics

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 802

## med_diuretics_loop

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 802

## med_inotropes

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 802

## med_insulins

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 802

## med_ivabradine

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 802

## med_ll

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 802

## med_mra

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 802

## med_oral_antidiabetic

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 802

## med_platelet

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 802

## med_potassium_binders

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 802

## med_rasi

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 802

## med_rdoad

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 802

## med_rdoad_syst

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 802

## med_sglt2i

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 802

## med_thrombolytic

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 802

## med_vasodil

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 802

## med_acei_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 502
  - `false`: 300

## med_anti_coag_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 563
  - `false`: 239

## med_anti_plat_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 540
  - `false`: 262

## med_antiarrhytmic_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 716
  - `true`: 86

## med_antiinfl_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 410
  - `true`: 392

## med_arb_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 465
  - `true`: 337

## med_ari_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 802

## med_arni_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 759
  - `true`: 43

## med_bb_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 623
  - `false`: 179

## med_ccb_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 423
  - `true`: 379

## med_cortico_syst_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 485
  - `true`: 317

## med_digitalis_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 643
  - `true`: 159

## med_diuretics_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 618
  - `false`: 184

## med_diuretics_loop_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 550
  - `false`: 252

## med_inotropes_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 756
  - `true`: 46

## med_insulins_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 628
  - `true`: 174

## med_ivabradine_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 800
  - 1 other distinct value(s) suppressed, covering 2 row(s) (count below 5 and/or ranked beyond top 20)

## med_ll_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 483
  - `false`: 319

## med_mra_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 465
  - `true`: 337

## med_oral_antidiabetic_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 587
  - `true`: 215

## med_platelet_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 540
  - `false`: 262

## med_potassium_binders_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 781
  - `true`: 21

## med_rasi_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 612
  - `false`: 190

## med_rdoad_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 458
  - `true`: 344

## med_rdoad_syst_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 766
  - `true`: 36

## med_sglt2i_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 754
  - `true`: 48

## med_thrombolytic_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 794
  - `true`: 8

## med_vasodil_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 425
  - `false`: 377

## conditions_af

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 439
  - `true`: 363

## conditions_aidshiv

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 798
  - 1 other distinct value(s) suppressed, covering 4 row(s) (count below 5 and/or ranked beyond top 20)

## conditions_ap

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 528
  - `true`: 274

## conditions_ckd_chronic

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 623
  - `true`: 179

## conditions_cm

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 714
  - `true`: 88

## conditions_copd

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 666
  - `true`: 136

## conditions_dem

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 759
  - `true`: 43

## conditions_dep

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 712
  - `true`: 90

## conditions_devices

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 599
  - `true`: 203

## conditions_dia

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 802

## conditions_diabetes

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 549
  - `true`: 253

## conditions_dysl

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 802

## conditions_hf

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `true`: 802

## conditions_hyp

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 610
  - `false`: 192

## conditions_hyperthyroid

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 802

## conditions_hypothyroid

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 727
  - `true`: 75

## conditions_ibd

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 802

## conditions_ihd

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 463
  - `false`: 339

## conditions_ld

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 765
  - `true`: 37

## conditions_mc

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 611
  - `true`: 191

## conditions_mi

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 519
  - `true`: 283

## conditions_myocarditis

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 799
  - 1 other distinct value(s) suppressed, covering 3 row(s) (count below 5 and/or ranked beyond top 20)

## conditions_osa

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 721
  - `true`: 81

## conditions_pad

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 657
  - `true`: 145

## conditions_pericardial

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 777
  - `true`: 25

## conditions_rd

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 771
  - `true`: 31

## conditions_revasc

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 606
  - `true`: 196

## conditions_stroke

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 688
  - `true`: 114

## conditions_substance_abuse

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 802

## conditions_tia

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 695
  - `true`: 107

## conditions_vd

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 666
  - `true`: 136
