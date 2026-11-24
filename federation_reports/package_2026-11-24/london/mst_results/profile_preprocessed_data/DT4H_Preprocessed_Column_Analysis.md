# Column Analysis

Total rows: 1005
Total columns: 257

## patient_demographics_gender

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `male`: 575
  - `female`: 430

## patient_demographics_age

- dtype: `Int32` (numeric)
- nulls: 0 (0.00%)
- min/max: 22.0 / 89.0
- mean/std: 72.3453 / 13.222400031739843
- quantiles: {'0.05': 47.0, '0.25': 64.0, '0.5': 75.0, '0.75': 83.0, '0.95': 88.0}

## encounters_encounterClass

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `IMP`: 1005

## encounters_admissionYear

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 64
- ⚠️ high-cardinality column (64 distinct values)
- top values (shown only where count ≥ 5):
  - `2054`: 52
  - `2052`: 48
  - `2056`: 48
  - `2051`: 44
  - `2058`: 43
  - `2045`: 35
  - `2053`: 34
  - `2044`: 33
  - `2049`: 32
  - `2057`: 30
  - `2059`: 30
  - `2047`: 28
  - `2050`: 28
  - `2055`: 27
  - `2041`: 26
  - `2043`: 26
  - `2046`: 24
  - `2048`: 24
  - `2031`: 23
  - `2037`: 23
  - 44 other distinct value(s) suppressed, covering 347 row(s) (count below 5 and/or ranked beyond top 20)

## encounters_lengthOfStay

- dtype: `Int32` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 130.0
- mean/std: 12.4587 / 14.17279309318242
- quantiles: {'0.05': 1.0, '0.25': 4.0, '0.5': 8.0, '0.75': 15.0, '0.95': 41.0}

## encounters_numOfPreviousHFStays_count

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 320.0
- mean/std: 5.7841 / 15.856557746853968
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 6.0, '0.95': 28.799999999999955}

## vital_signs_weight_value_p6mo_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -17.0 / 230.0
- mean/std: 74.5079 / 33.74555458927338
- quantiles: {'0.05': -16.474945908743557, '0.25': 62.0256107001102, '0.5': 75.78887415356844, '0.75': 92.28209393301107, '0.95': 122.51457831323108}

## vital_signs_weight_value_p6mo_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -17.0 / 230.0
- mean/std: 71.3616 / 32.26677691621627
- quantiles: {'0.05': -16.395477727518333, '0.25': 59.316238520159345, '0.5': 73.64212680690996, '0.75': 89.46552326157592, '0.95': 114.20991208800137}

## vital_signs_height_value_p1a_avg

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 71.0 / 200.0
- mean/std: 151.9901 / 35.932066845474374
- quantiles: {'0.05': 71.42505799201318, '0.25': 151.27680365089327, '0.5': 163.74759855556488, '0.75': 173.2443819807656, '0.95': 185.35459950826873}

## vital_signs_weight_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -17.0 / 230.0
- mean/std: 69.3783 / 34.47703519548283
- quantiles: {'0.05': -16.395477727518333, '0.25': 58.2179319352939, '0.5': 72.94576546474813, '0.75': 89.01001083573126, '0.95': 114.01298929042358}

## vital_signs_height_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 71.0 / 200.0
- mean/std: 146.9673 / 40.08515517046911
- quantiles: {'0.05': 71.42505799201318, '0.25': 146.39407089773565, '0.5': 162.99427030168474, '0.75': 172.73125208869575, '0.95': 186.12227926593272}

## vital_signs_diastolicBp_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 7.0 / 130.0
- mean/std: 69.2776 / 13.011776242493484
- quantiles: {'0.05': 52.0, '0.25': 60.0, '0.5': 68.0, '0.75': 77.0, '0.95': 90.0}

## vital_signs_diastolicBp_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -2.5 / 200.0
- mean/std: 79.2687 / 17.702245189454835
- quantiles: {'0.05': 55.0, '0.25': 67.0, '0.5': 78.0, '0.75': 89.0, '0.95': 110.0}

## vital_signs_heartRate_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 35.0 / 160.0
- mean/std: 76.3542 / 15.322492597802887
- quantiles: {'0.05': 54.2, '0.25': 65.0, '0.5': 75.0, '0.75': 86.0, '0.95': 102.0}

## vital_signs_heartRate_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 30.0 / 230.0
- mean/std: 90.4925 / 25.1058554548014
- quantiles: {'0.05': 55.0, '0.25': 73.0, '0.5': 88.0, '0.75': 104.0, '0.95': 137.79999999999995}

## vital_signs_systolicBp_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 23.0 / 210.0
- mean/std: 123.1194 / 22.362754759645128
- quantiles: {'0.05': 93.0, '0.25': 108.0, '0.5': 119.0, '0.75': 136.0, '0.95': 165.0}

## vital_signs_systolicBp_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 12.0 / 280.0
- mean/std: 141.5244 / 30.96847537204315
- quantiles: {'0.05': 98.0, '0.25': 120.0, '0.5': 137.0, '0.75': 160.0, '0.95': 198.0}

## vital_signs_oxygenSaturation_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 68.0 / 100.0
- mean/std: 70.6413 / 6.955039254689428
- quantiles: {'0.05': 68.75, '0.25': 68.75, '0.5': 68.75, '0.75': 68.75, '0.95': 95.0}

## vital_signs_oxygenSaturation_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 68.0 / 100.0
- mean/std: 70.6572 / 7.014305615379273
- quantiles: {'0.05': 68.75, '0.25': 68.75, '0.5': 68.75, '0.75': 68.75, '0.95': 95.0}

## lab_results_hemoglobin_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 56.0 / 210.0
- mean/std: 119.6209 / 23.056887926833223
- quantiles: {'0.05': 84.0, '0.25': 102.0, '0.5': 119.0, '0.75': 136.0, '0.95': 160.0}

## lab_results_hemoglobin_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 52.0 / 200.0
- mean/std: 120.0517 / 23.03257068162216
- quantiles: {'0.05': 84.0, '0.25': 104.0, '0.5': 120.0, '0.75': 136.0, '0.95': 157.0}

## lab_results_ferritin_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -3600.0 / 15000.0
- mean/std: -2723.8721 / 1691.628704900672
- quantiles: {'0.05': -3520.75, '0.25': -3520.75, '0.5': -3520.75, '0.75': -3520.75, '0.95': 473.99999999999886}

## lab_results_ferritin_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -1300.0 / 5100.0
- mean/std: -919.5896 / 732.0232665797191
- quantiles: {'0.05': -1250.2500000000002, '0.25': -1250.2500000000002, '0.5': -1250.2500000000002, '0.75': -1250.2500000000002, '0.95': 426.9999999999998}

## lab_results_tfs_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -20.0 / 93.0
- mean/std: -5.1209 / 20.708216732328676
- quantiles: {'0.05': -19.5, '0.25': -19.5, '0.5': -19.5, '0.75': 11.0, '0.95': 29.799999999999955}

## lab_results_tfs_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -24.0 / 93.0
- mean/std: -7.5913 / 22.15988929449403
- quantiles: {'0.05': -23.25, '0.25': -23.25, '0.5': -23.25, '0.75': 11.0, '0.95': 29.0}

## lab_results_ntProBnp_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -18000.0 / 70000.0
- mean/std: 1917.2373 / 16314.237334470436
- quantiles: {'0.05': -17437.5, '0.25': -17437.5, '0.5': 2195.0, '0.75': 7568.0, '0.95': 29580.19999999997}

## lab_results_ntProBnp_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -18000.0 / 70000.0
- mean/std: 2060.5244 / 16373.474296997416
- quantiles: {'0.05': -17437.5, '0.25': -17437.5, '0.5': 2325.0, '0.75': 7549.0, '0.95': 29940.999999999985}

## lab_results_crpNonHs_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -110.0 / 420.0
- mean/std: 25.4587 / 46.374147003231215
- quantiles: {'0.05': 1.0, '0.25': 4.2, '0.5': 11.2, '0.75': 29.3, '0.95': 100.35999999999986}

## lab_results_crpNonHs_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -79.0 / 320.0
- mean/std: 27.1780 / 45.10663151017952
- quantiles: {'0.05': 1.0, '0.25': 4.3, '0.5': 12.3, '0.75': 31.3, '0.95': 107.59999999999998}

## lab_results_tropTnHs_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -1100.0 / 4500.0
- mean/std: -224.3582 / 589.7982866785054
- quantiles: {'0.05': -1097.0, '0.25': -1097.0, '0.5': 27.0, '0.75': 54.0, '0.95': 216.59999999999968}

## lab_results_tropTnHs_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -1900.0 / 7400.0
- mean/std: -434.8517 / 886.255912667723
- quantiles: {'0.05': -1832.0, '0.25': -1832.0, '0.5': 28.0, '0.75': 54.0, '0.95': 156.5999999999999}

## lab_results_triGly_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -0.6 / 3.2
- mean/std: -0.3756 / 0.6305413600166458
- quantiles: {'0.05': -0.6, '0.25': -0.6, '0.5': -0.6, '0.75': -0.6, '0.95': 1.3}

## lab_results_triGly_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -0.6 / 3.2
- mean/std: -0.3756 / 0.631992939845336
- quantiles: {'0.05': -0.6, '0.25': -0.6, '0.5': -0.6, '0.75': -0.6, '0.95': 1.3}

## lab_results_cholTot_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.44 / 6.7
- mean/std: 0.8706 / 1.2110191461334694
- quantiles: {'0.05': 0.44999999999999996, '0.25': 0.44999999999999996, '0.5': 0.44999999999999996, '0.75': 0.44999999999999996, '0.95': 4.0}

## lab_results_cholTot_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -1.1 / 6.7
- mean/std: -0.4491 / 1.6836864298478016
- quantiles: {'0.05': -1.05, '0.25': -1.05, '0.5': -1.05, '0.75': -1.05, '0.95': 4.1}

## lab_results_hdl_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -0.6 / 2.7
- mean/std: -0.3887 / 0.5934995532335391
- quantiles: {'0.05': -0.6, '0.25': -0.6, '0.5': -0.6, '0.75': -0.6, '0.95': 1.2799999999999954}

## lab_results_hdl_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -0.5 / 2.7
- mean/std: -0.2999 / 0.5632415342869943
- quantiles: {'0.05': -0.5, '0.25': -0.5, '0.5': -0.5, '0.75': -0.5, '0.95': 1.3}

## lab_results_creatUS_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -530.0 / 2400.0
- mean/std: -348.1681 / 455.86230240583893
- quantiles: {'0.05': -527.413575, '0.25': -527.413575, '0.5': -527.413575, '0.75': -527.413575, '0.95': 763.0630799999998}

## lab_results_creatUS_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -530.0 / 2400.0
- mean/std: -347.7852 / 459.6944267249686
- quantiles: {'0.05': -527.413575, '0.25': -527.413575, '0.5': -527.413575, '0.75': -527.413575, '0.95': 759.33078}

## lab_results_bun_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -12.0 / 56.0
- mean/std: 7.4287 / 11.140228330623613
- quantiles: {'0.05': -11.925, '0.25': 4.2, '0.5': 7.3, '0.75': 12.6, '0.95': 26.58}

## lab_results_bun_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -11.0 / 50.0
- mean/std: 7.3629 / 10.332373100902608
- quantiles: {'0.05': -10.425, '0.25': 4.3, '0.5': 7.2, '0.75': 12.0, '0.95': 25.679999999999996}

## lab_results_potassium_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 1.7 / 7.4
- mean/std: 4.3224 / 0.6393708602800748
- quantiles: {'0.05': 3.4, '0.25': 3.9, '0.5': 4.3, '0.75': 4.7, '0.95': 5.3}

## lab_results_potassium_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 1.4 / 7.3
- mean/std: 4.4546 / 0.6992402644897511
- quantiles: {'0.05': 3.4, '0.25': 4.1, '0.5': 4.4, '0.75': 4.8, '0.95': 5.6}

## lab_results_sodium_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 96.0 / 160.0
- mean/std: 136.2022 / 7.37181003079745
- quantiles: {'0.05': 125.0, '0.25': 134.0, '0.5': 138.0, '0.75': 140.0, '0.95': 143.0}

## lab_results_sodium_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 100.0 / 170.0
- mean/std: 137.4229 / 6.382033549187046
- quantiles: {'0.05': 129.0, '0.25': 135.0, '0.5': 138.0, '0.75': 141.0, '0.95': 144.0}

## lab_results_albuminBS_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 8.7 / 50.0
- mean/std: 35.1552 / 8.471288812103774
- quantiles: {'0.05': 8.75, '0.25': 33.0, '0.5': 37.0, '0.75': 40.0, '0.95': 44.0}

## lab_results_albuminBS_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -0.5 / 52.0
- mean/std: 35.6592 / 10.664828787697461
- quantiles: {'0.05': -0.5, '0.25': 35.0, '0.5': 38.0, '0.75': 41.0, '0.95': 45.0}

## lab_results_hba1c%_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 2.5 / 13.0
- mean/std: 2.7803 / 1.1099748796768811
- quantiles: {'0.05': 2.5249999999999995, '0.25': 2.5249999999999995, '0.5': 2.5249999999999995, '0.75': 2.5249999999999995, '0.95': 5.659999999999991}

## lab_results_hba1c%_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 2.4 / 13.0
- mean/std: 2.7574 / 1.1186473020956593
- quantiles: {'0.05': 2.4999999999999996, '0.25': 2.4999999999999996, '0.5': 2.4999999999999996, '0.75': 2.4999999999999996, '0.95': 5.659999999999991}

## lab_results_hba1c_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -3.3 / 140.0
- mean/std: 9.3898 / 25.273363179730172
- quantiles: {'0.05': -3.25, '0.25': -3.25, '0.5': -3.25, '0.75': -3.25, '0.95': 59.0}

## lab_results_hba1c_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -3.3 / 140.0
- mean/std: 9.4027 / 25.297602262943972
- quantiles: {'0.05': -3.25, '0.25': -3.25, '0.5': -3.25, '0.75': -3.25, '0.95': 59.799999999999955}

## lab_results_validSerumCreatinine_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -3.0 / 23.0
- mean/std: 11.1160 / 5.819313307496114
- quantiles: {'0.05': -2.9406000000000003, '0.25': 8.2563, '0.5': 11.0838, '0.75': 14.5899, '0.95': 20.8104}

## lab_results_valideGFR_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -20.0 / 90.0
- mean/std: 54.0925 / 24.066731472610726
- quantiles: {'0.05': 16.0, '0.25': 35.0, '0.5': 54.0, '0.75': 75.0, '0.95': 90.0}

## lab_results_valideGFR_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -19.0 / 90.0
- mean/std: 54.3687 / 23.8718052385065
- quantiles: {'0.05': 16.0, '0.25': 36.0, '0.5': 55.0, '0.75': 74.0, '0.95': 90.0}

## symptoms_Ankle_swelling_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 1004
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## symptoms_Ascites_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 962
  - `true`: 43

## symptoms_Breathlessness_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 838
  - `true`: 167

## symptoms_Cardiac_murmur_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 1005

## symptoms_Chest_pain_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 964
  - `true`: 41

## symptoms_Cheyne_stokes_respiration_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 1005

## symptoms_Depression_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 996
  - `true`: 9

## symptoms_Dizziness_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 1001
  - 1 other distinct value(s) suppressed, covering 4 row(s) (count below 5 and/or ranked beyond top 20)

## symptoms_Elevated_jugular_venous_pressure_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 1005

## symptoms_Fatigue_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 1005

## symptoms_Hepatojugular_reflux_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 1005

## symptoms_Hepatomegaly_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 996
  - `true`: 9

## symptoms_Intermittent_claudication_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 1005

## symptoms_Irregular_pulse_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 1005

## symptoms_Loss_of_appetite_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 1005

## symptoms_Nocturnal_cough_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 1005

## symptoms_Oliguria_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 1004
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## symptoms_Orthopnoea_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 1005

## symptoms_Palpitations_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 1000
  - `true`: 5

## symptoms_Paroxysmal_nocturnal_dyspnea_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 1005

## symptoms_Peripheral_edema_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 1001
  - 1 other distinct value(s) suppressed, covering 4 row(s) (count below 5 and/or ranked beyond top 20)

## symptoms_Pleural_effusion_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 625
  - `true`: 380

## symptoms_Pulmonary_crepitations_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 1004
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## symptoms_Reduced_exercise_tolerance_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 1004
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## symptoms_Syncope_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 1005

## symptoms_Tachycardia_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 974
  - `true`: 31

## symptoms_Tachypnoea_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 1004
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## symptoms_Third_heart_sound_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 1005

## symptoms_Weight_gain_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 1005

## symptoms_Weight_loss_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 1005

## electrocardiographs_ecg_st_pET

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `Missing`: 1005

## electrocardiographs_ecg_ischemia_without_st_pET

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `Missing`: 1005

## electrocardiographs_ecg_type_of_rhythms_pET_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `Missing`: 1005

## electrocardiographs_ecg_type_of_rhythms_pET_last

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `Missing`: 1005

## smoking_status_smoker_last

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 971
  - `true`: 34

## smoking_status_formerSmoker_last

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 785
  - `false`: 111
  - `true`: 109

## smoking_status_smoker_startTime_count

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 3.0
- mean/std: 0.0507 / 0.29337291902973484
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 0.0}

## nyha_nyha

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 0.0
- mean/std: 0.0000 / 0.0
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 0.0}
- ⚠️ constant column (single value)

## nyha_nyha_pET

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 0.0
- mean/std: 0.0000 / 0.0
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 0.0}
- ⚠️ constant column (single value)

## vital_signs_systolicBpDuringEncounter_value_pET

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 60.0 / 210.0
- mean/std: 123.3262 / 21.91681126158149
- quantiles: {'0.05': 94.0, '0.25': 108.0, '0.5': 119.0, '0.75': 136.0, '0.95': 165.0}

## vital_signs_bmi_value_pET

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 14.0 / 94.0
- mean/std: 27.2290 / 7.382153593815175
- quantiles: {'0.05': 18.599999999999998, '0.25': 22.1, '0.5': 25.5, '0.75': 30.8, '0.95': 40.66}

## lab_results_creatBS_value_p3a_avg

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 3.9 / 170.0
- mean/std: 15.1085 / 12.165783354758137
- quantiles: {'0.05': 6.50325, '0.25': 9.33075, '0.5': 12.04515, '0.75': 16.851899999999997, '0.95': 30.073289999999982}

## lab_results_validSerumCreatinine_value_pET

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 3.9 / 23.0
- mean/std: 12.6543 / 4.504124153913538
- quantiles: {'0.05': 6.5598, '0.25': 9.161100000000001, '0.5': 11.706554367197109, '0.75': 15.834000000000001, '0.95': 21.353279999999994}

## hyperkalemia_severity_categorizedValue

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 5
- top values (shown only where count ≥ 5):
  - `normal`: 869
  - `mild`: 88
  - `severe`: 20
  - `moderate`: 18
  - `Missing`: 10

## ckd_severity_categorizedValue

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 7
- top values (shown only where count ≥ 5):
  - `mildly_decreased`: 325
  - `mild_to_moderate_decrease`: 197
  - `moderate_to_severe_decrease`: 195
  - `severe_decrease`: 136
  - `normal_or_high`: 118
  - `kidney_failure`: 32
  - 1 other distinct value(s) suppressed, covering 2 row(s) (count below 5 and/or ranked beyond top 20)

## conditions_heartFailure_timeFromEarliest_first

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 67.0
- mean/std: 7.4547 / 13.803804186155077
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 8.0, '0.95': 40.799999999999955}

## conditions_heart_failure_hf_within_18mo_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `true`: 1005

## conditions_heart_failure_occurred_prior_to_18_months_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 830
  - `true`: 175

## encounter_primary_reason_HF_Disease_f5a_w7d_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 974
  - `false`: 30
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## encounter_primary_reason_HF_Disease_f5a_w1mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 901
  - `false`: 102
  - 1 other distinct value(s) suppressed, covering 2 row(s) (count below 5 and/or ranked beyond top 20)

## encounter_primary_reason_HF_Disease_f5a_w3mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 823
  - `false`: 177
  - `true`: 5

## encounter_primary_reason_HF_Disease_f5a_w6mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 778
  - `false`: 220
  - `true`: 7

## encounter_primary_reason_HF_Disease_f5a_w1a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 731
  - `false`: 267
  - `true`: 7

## encounter_primary_reason_HF_Disease_f5a_w3a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 688
  - `false`: 310
  - `true`: 7

## encounter_primary_reason_HF_Disease_f5a_w5a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 679
  - `false`: 319
  - `true`: 7

## encounter_primary_reason_number_of_HF_rehospitalizations_5a_f5a_count

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 1.0
- mean/std: 0.0080 / 0.0889083776115733
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 0.0}

## encounter_primary_reason_CV_Disease_f5a_w7d_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 974
  - `false`: 27
  - 1 other distinct value(s) suppressed, covering 4 row(s) (count below 5 and/or ranked beyond top 20)

## encounter_primary_reason_CV_Disease_f5a_w1mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 901
  - `false`: 91
  - `true`: 13

## encounter_primary_reason_CV_Disease_f5a_w3mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 823
  - `false`: 156
  - `true`: 26

## encounter_primary_reason_CV_Disease_f5a_w6mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 778
  - `false`: 192
  - `true`: 35

## encounter_primary_reason_CV_Disease_f5a_w1a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 731
  - `false`: 229
  - `true`: 45

## encounter_primary_reason_CV_Disease_f5a_w3a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 688
  - `false`: 262
  - `true`: 55

## encounter_primary_reason_CV_Disease_f5a_w5a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 679
  - `false`: 269
  - `true`: 57

## encounter_primary_reason_number_of_days_to_rehosp_for_CV_f5a_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -380.0 / 1500.0
- mean/std: -338.6816 / 158.76173093856812
- quantiles: {'0.05': -373.25, '0.25': -373.25, '0.5': -373.25, '0.75': -373.25, '0.95': 14.199999999999818}

## encounter_primary_reason_number_of_CV_rehospitalizations_5a_f5a_count

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 6.0
- mean/std: 0.1453 / 0.5269528749533089
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 1.0}

## encounter_primary_reason_non_CV_Disease_f5a_w7d_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 974
  - `true`: 27
  - 1 other distinct value(s) suppressed, covering 4 row(s) (count below 5 and/or ranked beyond top 20)

## encounter_primary_reason_non_CV_Disease_f5a_w1mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 901
  - `true`: 91
  - `false`: 13

## encounter_primary_reason_non_CV_Disease_f5a_w3mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 823
  - `true`: 159
  - `false`: 23

## encounter_primary_reason_non_CV_Disease_f5a_w6mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 778
  - `true`: 198
  - `false`: 29

## encounter_primary_reason_non_CV_Disease_f5a_w1a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 731
  - `true`: 238
  - `false`: 36

## encounter_primary_reason_non_CV_Disease_f5a_w3a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 688
  - `true`: 274
  - `false`: 43

## encounter_primary_reason_non_CV_Disease_f5a_w5a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 679
  - `true`: 281
  - `false`: 45

## encounter_primary_reason_number_of_days_to_rehosp_for_non_CV_f5a_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -440.0 / 1800.0
- mean/std: -263.8328 / 321.34665003460685
- quantiles: {'0.05': -439.75, '0.25': -439.75, '0.5': -439.75, '0.75': 8.0, '0.95': 290.0}

## encounter_primary_reason_number_of_non_CV_rehospitalizations_5a_f5a_count

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -9.8 / 39.0
- mean/std: 1.2490 / 3.4462649400224312
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 1.0, '0.95': 6.0}

## encounter_primary_reason_renal_complications_f5a_w7d_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 974
  - `false`: 31

## encounter_primary_reason_renal_complications_f5a_w1mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 901
  - `false`: 104

## encounter_primary_reason_renal_complications_f5a_w3mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 823
  - `false`: 182

## encounter_primary_reason_renal_complications_f5a_w6mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 778
  - `false`: 226
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## encounter_primary_reason_renal_complications_f5a_w1a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 731
  - `false`: 272
  - 1 other distinct value(s) suppressed, covering 2 row(s) (count below 5 and/or ranked beyond top 20)

## encounter_primary_reason_renal_complications_f5a_w3a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 688
  - `false`: 315
  - 1 other distinct value(s) suppressed, covering 2 row(s) (count below 5 and/or ranked beyond top 20)

## encounter_primary_reason_renal_complications_f5a_w5a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 679
  - `false`: 324
  - 1 other distinct value(s) suppressed, covering 2 row(s) (count below 5 and/or ranked beyond top 20)

## encounter_primary_reason_number_of_renal_rehospitalizations_5a_f5a_count

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 2.0
- mean/std: 0.0030 / 0.07050645456677775
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 0.0}

## cause_of_death_isCV_f5a_w7d_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 989
  - `false`: 16

## cause_of_death_isCV_f5a_w1mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 950
  - `false`: 55

## cause_of_death_isCV_f5a_w3mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 894
  - `false`: 111

## cause_of_death_isCV_f5a_w6mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 845
  - `false`: 160

## cause_of_death_isCV_f5a_w1a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 761
  - `false`: 244

## cause_of_death_isCV_f5a_w3a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 653
  - `false`: 352

## cause_of_death_isCV_f5a_w5a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 620
  - `false`: 385

## cause_of_death_isRenal_f5a_w7d_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 989
  - `false`: 16

## cause_of_death_isRenal_f5a_w1mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 950
  - `false`: 55

## cause_of_death_isRenal_f5a_w3mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 894
  - `false`: 111

## cause_of_death_isRenal_f5a_w6mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 845
  - `false`: 160

## cause_of_death_isRenal_f5a_w1a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 761
  - `false`: 244

## cause_of_death_isRenal_f5a_w3a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 653
  - `false`: 352

## cause_of_death_isRenal_f5a_w5a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 620
  - `false`: 385

## cause_of_death_isNonRenalAndNonCV_f5a_w7d_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 989
  - `false`: 16

## cause_of_death_isNonRenalAndNonCV_f5a_w1mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 950
  - `false`: 55

## cause_of_death_isNonRenalAndNonCV_f5a_w3mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 894
  - `false`: 111

## cause_of_death_isNonRenalAndNonCV_f5a_w6mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 845
  - `false`: 160

## cause_of_death_isNonRenalAndNonCV_f5a_w1a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 761
  - `false`: 244

## cause_of_death_isNonRenalAndNonCV_f5a_w3a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 653
  - `false`: 352

## cause_of_death_isNonRenalAndNonCV_f5a_w5a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 620
  - `false`: 385

## cause_of_death_isAllCause_f5a_w7d_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 989
  - `true`: 16

## cause_of_death_isAllCause_f5a_w1mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 950
  - `true`: 55

## cause_of_death_isAllCause_f5a_w3mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 894
  - `true`: 111

## cause_of_death_isAllCause_f5a_w6mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 845
  - `true`: 160

## cause_of_death_isAllCause_f5a_w1a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 761
  - `true`: 244

## cause_of_death_isAllCause_f5a_w3a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 653
  - `true`: 352

## cause_of_death_isAllCause_f5a_w5a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 620
  - `true`: 385

## cause_of_death_number_of_days_to_death_for_all_cause_f5a_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -450.0 / 1800.0
- mean/std: -128.4587 / 477.7843444141399
- quantiles: {'0.05': -447.75, '0.25': -447.75, '0.5': -447.75, '0.75': 114.0, '0.95': 894.3999999999992}

## eGFR_2021_ckd_epi_creatinine

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 20.0 / 160.0
- mean/std: 60.9953 / 24.080843215236936
- quantiles: {'0.05': 26.7078992, '0.25': 40.86016, '0.5': 58.332979, '0.75': 79.39538, '0.95': 103.07151579999999}

## ckd_severity_from_calculated_egfr

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 5
- top values (shown only where count ≥ 5):
  - `mildly_decreased`: 343
  - `moderate_to_severe_decrease`: 232
  - `mild_to_moderate_decrease`: 210
  - `normal_or_high`: 153
  - `severe_decrease`: 67

## ckd_severity_calculated_or_measured

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 5
- top values (shown only where count ≥ 5):
  - `mildly_decreased`: 343
  - `moderate_to_severe_decrease`: 232
  - `mild_to_moderate_decrease`: 210
  - `normal_or_high`: 153
  - `severe_decrease`: 67

## med_acei

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 687
  - `true`: 318

## med_activeDuringEncounter_ace_inhibitors_arb_use

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 1005

## med_activeDuringEncounter_bb

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 1005

## med_anti_coag

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 806
  - `false`: 199

## med_anti_plat

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 794
  - `true`: 211

## med_antiarrhytmic

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 932
  - `true`: 73

## med_antiinfl

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 976
  - `true`: 29

## med_arb

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 698
  - `true`: 307

## med_ari

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 1005

## med_arni

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 909
  - `true`: 96

## med_bb

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 718
  - `true`: 287

## med_ccb

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 765
  - `true`: 240

## med_cortico_syst

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 817
  - `true`: 188

## med_digitalis

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 817
  - `true`: 188

## med_diuretics

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 974
  - `false`: 31

## med_diuretics_loop

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 883
  - `false`: 122

## med_inotropes

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 830
  - `true`: 175

## med_insulins

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 814
  - `true`: 191

## med_ivabradine

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 997
  - `true`: 8

## med_ll

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 571
  - `false`: 434

## med_mra

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 910
  - `false`: 95

## med_oral_antidiabetic

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 594
  - `true`: 411

## med_platelet

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 794
  - `true`: 211

## med_potassium_binders

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 1005

## med_rasi

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 607
  - `true`: 398

## med_rdoad

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 622
  - `true`: 383

## med_rdoad_syst

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 699
  - `true`: 306

## med_sglt2i

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 726
  - `true`: 279

## med_thrombolytic

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 982
  - `true`: 23

## med_vasodil

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 875
  - `true`: 130

## med_acei_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 845
  - `true`: 160

## med_anti_coag_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 698
  - `true`: 307

## med_anti_plat_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 905
  - `true`: 100

## med_antiarrhytmic_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 940
  - `true`: 65

## med_antiinfl_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 982
  - `true`: 23

## med_arb_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 874
  - `true`: 131

## med_ari_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 1005

## med_arni_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 969
  - `true`: 36

## med_bb_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 877
  - `true`: 128

## med_ccb_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 891
  - `true`: 114

## med_cortico_syst_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 882
  - `true`: 123

## med_digitalis_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 921
  - `true`: 84

## med_diuretics_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 656
  - `true`: 349

## med_diuretics_loop_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 696
  - `true`: 309

## med_inotropes_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 895
  - `true`: 110

## med_insulins_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 909
  - `true`: 96

## med_ivabradine_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 1001
  - 1 other distinct value(s) suppressed, covering 4 row(s) (count below 5 and/or ranked beyond top 20)

## med_ll_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 764
  - `true`: 241

## med_mra_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 682
  - `true`: 323

## med_oral_antidiabetic_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 836
  - `true`: 169

## med_platelet_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 905
  - `true`: 100

## med_potassium_binders_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 1005

## med_rasi_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 822
  - `true`: 183

## med_rdoad_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 816
  - `true`: 189

## med_rdoad_syst_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 836
  - `true`: 169

## med_sglt2i_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 917
  - `true`: 88

## med_thrombolytic_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 1002
  - 1 other distinct value(s) suppressed, covering 3 row(s) (count below 5 and/or ranked beyond top 20)

## med_vasodil_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 943
  - `true`: 62

## conditions_af

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 544
  - `false`: 461

## conditions_aidshiv

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 945
  - `true`: 60

## conditions_ap

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 924
  - `true`: 81

## conditions_ckd_chronic

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 609
  - `true`: 396

## conditions_cm

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 882
  - `true`: 123

## conditions_copd

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 768
  - `true`: 237

## conditions_dem

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 913
  - `true`: 92

## conditions_dep

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 994
  - `true`: 11

## conditions_devices

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 1005

## conditions_dia

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 1005

## conditions_diabetes

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 593
  - `true`: 412

## conditions_dysl

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 686
  - `true`: 319

## conditions_hf

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `true`: 1005

## conditions_hyp

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 712
  - `false`: 293

## conditions_hyperthyroid

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 979
  - `true`: 26

## conditions_hypothyroid

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 864
  - `true`: 141

## conditions_ibd

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 849
  - `true`: 156

## conditions_ihd

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 557
  - `true`: 448

## conditions_ld

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 850
  - `true`: 155

## conditions_mc

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 822
  - `true`: 183

## conditions_mi

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 895
  - `true`: 110

## conditions_myocarditis

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 1001
  - 1 other distinct value(s) suppressed, covering 4 row(s) (count below 5 and/or ranked beyond top 20)

## conditions_osa

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 911
  - `true`: 94

## conditions_pad

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 701
  - `true`: 304

## conditions_pericardial

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 858
  - `true`: 147

## conditions_rd

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 855
  - `true`: 150

## conditions_revasc

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 1005

## conditions_stroke

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 906
  - `true`: 99

## conditions_substance_abuse

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 817
  - `true`: 188

## conditions_tia

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 949
  - `true`: 56

## conditions_vd

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 917
  - `false`: 88
