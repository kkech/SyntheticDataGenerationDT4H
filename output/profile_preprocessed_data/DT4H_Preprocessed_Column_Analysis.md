# Column Analysis

Total rows: 4694
Total columns: 311

## patient_demographics_gender

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `male`: 2711
  - `female`: 1983

## patient_demographics_age

- dtype: `Int32` (numeric)
- nulls: 0 (0.00%)
- min/max: 18 / 104
- mean/std: 70.9842 / 13.760506966385604
- quantiles: {'0.05': 45.0, '0.25': 64.0, '0.5': 73.0, '0.75': 81.0, '0.95': 90.0}

## encounters_encounterClass

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `IMP`: 4694

## encounters_admissionYear

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 10
- top values (shown only where count ≥ 5):
  - `2023`: 615
  - `2022`: 596
  - `2019`: 583
  - `2021`: 536
  - `2020`: 516
  - `2018`: 483
  - `2017`: 471
  - `2024`: 438
  - `2016`: 422
  - `2015`: 34

## encounters_lengthOfStay

- dtype: `Int32` (numeric)
- nulls: 0 (0.00%)
- min/max: 1 / 257
- mean/std: 10.5254 / 13.764715567948564
- quantiles: {'0.05': 1.0, '0.25': 2.0, '0.5': 6.0, '0.75': 13.0, '0.95': 34.0}

## encounters_numOfPreviousHFStays_count

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0 / 757
- mean/std: 52.5614 / 71.64670202080976
- quantiles: {'0.05': 0.0, '0.25': 4.0, '0.5': 25.0, '0.75': 74.0, '0.95': 201.0}

## vital_signs_weight_value_p6mo_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 32.0 / 208.6
- mean/std: 77.4137 / 19.82206258247233
- quantiles: {'0.05': 50.2, '0.25': 63.6, '0.5': 74.7, '0.75': 87.9, '0.95': 113.0}

## vital_signs_weight_value_p6mo_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 33.9 / 241.0
- mean/std: 79.8160 / 20.27092061865717
- quantiles: {'0.05': 52.0, '0.25': 66.0, '0.5': 77.0, '0.75': 90.0, '0.95': 116.0}

## vital_signs_height_value_p1a_avg

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 96.95 / 207.0
- mean/std: 171.1361 / 10.645669102075088
- quantiles: {'0.05': 154.0, '0.25': 164.0, '0.5': 171.5, '0.75': 178.0, '0.95': 188.0}

## vital_signs_weight_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 32.0 / 208.6
- mean/std: 77.3783 / 19.749029501610078
- quantiles: {'0.05': 50.2, '0.25': 63.6, '0.5': 74.5, '0.75': 87.7, '0.95': 113.0}

## vital_signs_height_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 110.0 / 207.0
- mean/std: 171.5836 / 10.38403261877741
- quantiles: {'0.05': 155.0, '0.25': 165.0, '0.5': 172.0, '0.75': 178.0, '0.95': 188.0}

## vital_signs_bmi_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 13.21178491214163 / 79.58477508650519
- mean/std: 26.8488 / 6.413166116548382
- quantiles: {'0.05': 18.39741049015722, '0.25': 22.545071323723135, '0.5': 25.858572201189833, '0.75': 29.950132879452955, '0.95': 38.204081632653065}

## vital_signs_systolicBp_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 46.0 / 257.0
- mean/std: 130.2904 / 28.71232965890652
- quantiles: {'0.05': 91.0, '0.25': 110.0, '0.5': 126.0, '0.75': 148.0, '0.95': 181.0}

## vital_signs_systolicBp_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 50.0 / 221.0
- mean/std: 121.1858 / 22.70389542659586
- quantiles: {'0.05': 90.0, '0.25': 105.0, '0.5': 118.0, '0.75': 135.0, '0.95': 161.0}

## vital_signs_diastolicBp_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 8.0 / 201.0
- mean/std: 75.6174 / 18.361876155391204
- quantiles: {'0.05': 50.0, '0.25': 63.0, '0.5': 73.0, '0.75': 85.0, '0.95': 109.0}

## vital_signs_diastolicBp_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 6.0 / 158.0
- mean/std: 69.0782 / 13.55072158338932
- quantiles: {'0.05': 48.0, '0.25': 60.0, '0.5': 68.0, '0.75': 77.0, '0.95': 92.0}

## vital_signs_heartRate_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 1.0 / 223.0
- mean/std: 112.6747 / 17.60663685872408
- quantiles: {'0.05': 100.0, '0.25': 102.0, '0.5': 108.0, '0.75': 119.0, '0.95': 146.0}

## vital_signs_heartRate_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 254.0
- mean/std: 109.1623 / 14.926357337424573
- quantiles: {'0.05': 100.0, '0.25': 101.0, '0.5': 105.0, '0.75': 112.0, '0.95': 135.0}

## vital_signs_oxygenSaturation_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 85.14609103804591 / 98.99998835996726
- mean/std: nan / nan
- quantiles: {'0.05': 91.04103841325049, '0.25': 96.8855573003089, '0.5': 98.9818871211591, '0.75': nan, '0.95': nan}

## vital_signs_oxygenSaturation_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 85.2815288561821 / 98.99999998947361
- mean/std: nan / nan
- quantiles: {'0.05': 91.31017468606436, '0.25': 96.92960922306507, '0.5': 98.98602371057228, '0.75': nan, '0.95': nan}

## lab_results_hemoglobin_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 32.228 / 233.653
- mean/std: 119.8794 / 23.00745249989437
- quantiles: {'0.05': 85.4042, '0.25': 103.1296, '0.5': 119.2436, '0.75': 135.3576, '0.95': 159.5286}

## lab_results_hemoglobin_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.16114 / 219.1504
- mean/std: 121.5586 / 24.231952331178267
- quantiles: {'0.05': 82.1814, '0.25': 104.741, '0.5': 122.4664, '0.75': 138.5804, '0.95': 159.5286}

## lab_results_ferritin_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 4.0 / 42076.0
- mean/std: 522.1137 / 2097.7279320117946
- quantiles: {'0.05': 21.7, '0.25': 77.4, '0.5': 196.0, '0.75': 446.0, '0.95': 1378.0}

## lab_results_ferritin_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 4.0 / 42076.0
- mean/std: 537.8192 / 2221.4999668521164
- quantiles: {'0.05': 20.9, '0.25': 72.73, '0.5': 183.0, '0.75': 439.0, '0.95': 1350.0}

## lab_results_ntProBnp_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 33.0 / 70000.0
- mean/std: nan / nan
- quantiles: {'0.05': 543.5, '0.25': 2241.0, '0.5': 5510.465645875448, '0.75': 20082.0, '0.95': nan}

## lab_results_ntProBnp_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 34.5 / 70000.0
- mean/std: nan / nan
- quantiles: {'0.05': 575.0, '0.25': 2333.0, '0.5': 5855.0, '0.75': 20345.0, '0.95': nan}

## lab_results_crpNonHs_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.3 / 629.7
- mean/std: nan / nan
- quantiles: {'0.05': 1.7, '0.25': 8.0, '0.5': 25.9, '0.75': 77.0, '0.95': nan}

## lab_results_crpNonHs_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.3 / 578.1
- mean/std: nan / nan
- quantiles: {'0.05': 1.4, '0.25': 6.0, '0.5': 21.0, '0.75': 78.0, '0.95': nan}

## lab_results_tropTHs_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.004 / 65.65
- mean/std: 0.4757 / 2.4338591860463894
- quantiles: {'0.05': 0.015, '0.25': 0.033, '0.5': 0.062, '0.75': 0.172, '0.95': 1.48}

## lab_results_tropTHs_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.004 / 17.389
- mean/std: 0.2397 / 0.9622199404121635
- quantiles: {'0.05': 0.013, '0.25': 0.028, '0.5': 0.051, '0.75': 0.115, '0.95': 0.82}

## lab_results_tropTnHs_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 3.0 / 15400.0
- mean/std: nan / nan
- quantiles: {'0.05': 56.0, '0.25': nan, '0.5': nan, '0.75': nan, '0.95': nan}

## lab_results_tropTnHs_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 3.0 / 6140.0
- mean/std: nan / nan
- quantiles: {'0.05': 44.0, '0.25': nan, '0.5': nan, '0.75': nan, '0.95': nan}

## lab_results_triGly_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.2 / 11.35
- mean/std: 1.4996 / 1.0162255762862134
- quantiles: {'0.05': 0.6, '0.25': 0.873, '0.5': 1.16, '0.75': 1.84, '0.95': 3.27}

## lab_results_triGly_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.2 / 16.5
- mean/std: 1.5482 / 1.3008404467661214
- quantiles: {'0.05': 0.61, '0.25': 0.887, '0.5': 1.18, '0.75': 1.82, '0.95': 3.27}

## lab_results_cholTot_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 1.17 / 8.71
- mean/std: 3.9091 / 1.1287612235486775
- quantiles: {'0.05': 2.28, '0.25': 3.14, '0.5': 3.8, '0.75': 4.5, '0.95': 6.06}

## lab_results_cholTot_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 1.17 / 8.71
- mean/std: 3.9754 / 1.148051944847041
- quantiles: {'0.05': 2.3, '0.25': 3.21, '0.5': 3.88, '0.75': 4.58, '0.95': 6.1}

## lab_results_hdl_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.22 / 3.76
- mean/std: 1.1793 / 0.43556925869503366
- quantiles: {'0.05': 0.62, '0.25': 0.89, '0.5': 1.1, '0.75': 1.39, '0.95': 2.01}

## lab_results_hdl_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.16 / 3.85
- mean/std: 1.1911 / 0.4533860757749135
- quantiles: {'0.05': 0.62, '0.25': 0.89, '0.5': 1.11, '0.75': 1.39, '0.95': 2.03}

## lab_results_ldl_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.04 / 7.124584861764171
- mean/std: nan / nan
- quantiles: {'0.05': 2.1, '0.25': nan, '0.5': nan, '0.75': nan, '0.95': nan}

## lab_results_ldl_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.04 / 6.4475
- mean/std: nan / nan
- quantiles: {'0.05': 2.12, '0.25': nan, '0.5': nan, '0.75': nan, '0.95': nan}

## lab_results_potassium_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 1.85 / 7.5
- mean/std: 4.1776 / 0.5810725389491181
- quantiles: {'0.05': 3.3, '0.25': 3.8, '0.5': 4.11, '0.75': 4.5, '0.95': 5.2}

## lab_results_potassium_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 1.85 / 8.8
- mean/std: 4.2900 / 0.6847061421837667
- quantiles: {'0.05': 3.3, '0.25': 3.89, '0.5': 4.21, '0.75': 4.66, '0.95': 5.5}

## lab_results_sodium_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 109.0 / 160.0
- mean/std: nan / nan
- quantiles: {'0.05': 130.0, '0.25': 135.8, '0.5': 138.3, '0.75': 141.0, '0.95': 147.5}

## lab_results_sodium_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 109.0 / 165.0
- mean/std: nan / nan
- quantiles: {'0.05': 128.5, '0.25': 135.0, '0.5': 138.0, '0.75': 140.9, '0.95': 146.0}

## lab_results_creatUS_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 44.56928 / 2706.9616
- mean/std: 690.8570 / 488.6408249367045
- quantiles: {'0.05': 175.336, '0.25': 339.36, '0.5': 542.976, '0.75': 877.8112, '0.95': 1708.112}

## lab_results_creatUS_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 44.56928 / 2839.312
- mean/std: 737.4743 / 495.2540940506342
- quantiles: {'0.05': 180.992, '0.25': 373.296, '0.5': 622.16, '0.75': 961.52, '0.95': 1708.112}

## lab_results_albuminBS_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 9.0 / 50.0
- mean/std: 29.8841 / 6.800737302854493
- quantiles: {'0.05': 18.1, '0.25': 25.0, '0.5': 30.0, '0.75': 35.0, '0.95': 40.9}

## lab_results_albuminBS_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 10.0 / 50.0
- mean/std: 31.2396 / 6.809028167417907
- quantiles: {'0.05': 19.0, '0.25': 26.9, '0.5': 31.7, '0.75': 36.0, '0.95': 42.0}

## lab_results_hba1c%_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 3.057644111307257 / 14.72464629199165
- mean/std: nan / nan
- quantiles: {'0.05': 4.5359769982365705, '0.25': 7.237471880362289, '0.5': nan, '0.75': nan, '0.95': nan}

## lab_results_hba1c%_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 3.030634140968014 / 15.084526345323734
- mean/std: nan / nan
- quantiles: {'0.05': 4.488018187785859, '0.25': 7.259655634034583, '0.5': nan, '0.75': nan, '0.95': nan}

## lab_results_hba1c_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 15.281738621585285 / 130.36519624223948
- mean/std: nan / nan
- quantiles: {'0.05': 27.074286329881666, '0.25': 51.841746012140796, '0.5': nan, '0.75': nan, '0.95': nan}

## lab_results_hba1c_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 15.336777091546635 / 151.99290291300636
- mean/std: nan / nan
- quantiles: {'0.05': 27.435696164201683, '0.25': 51.29739629351176, '0.5': nan, '0.75': nan, '0.95': nan}

## lab_results_validSerumCreatinine_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.7917 / 22.57476
- mean/std: 12.2191 / 4.5569709466201305
- quantiles: {'0.05': 5.9943, '0.25': 8.7087, '0.5': 11.4231, '0.75': 15.16671, '0.95': 21.0366}

## lab_results_validSerumCreatinine_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.9048 / 22.57476
- mean/std: 12.3884 / 4.487922946788882
- quantiles: {'0.05': 6.2205, '0.25': 8.9349, '0.5': 11.4231, '0.75': 15.4947, '0.95': 20.98005}

## lab_results_valideGFR_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 3.0 / 90.0
- mean/std: 53.1059 / 24.63013585220488
- quantiles: {'0.05': 13.0, '0.25': 33.0, '0.5': 54.0, '0.75': 74.0, '0.95': 90.0}

## lab_results_valideGFR_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 3.0 / 90.0
- mean/std: 53.8319 / 25.093009933909165
- quantiles: {'0.05': 13.0, '0.25': 33.0, '0.5': 54.0, '0.75': 76.0, '0.95': 90.0}

## symptoms_Ankle_swelling_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4694

## symptoms_Ascites_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4694

## symptoms_Breathlessness_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4694

## symptoms_Cardiac_murmur_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4694

## symptoms_Chest_pain_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4694

## symptoms_Cheyne_stokes_respiration_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4694

## symptoms_Depression_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4694

## symptoms_Dizziness_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4694

## symptoms_Elevated_jugular_venous_pressure_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4694

## symptoms_Fatigue_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4694

## symptoms_Hepatojugular_reflux_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4694

## symptoms_Hepatomegaly_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4694

## symptoms_Intermittent_claudication_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4694

## symptoms_Irregular_pulse_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4694

## symptoms_Loss_of_appetite_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4694

## symptoms_Nocturnal_cough_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4694

## symptoms_Oliguria_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4694

## symptoms_Orthopnoea_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4694

## symptoms_Palpitations_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4694

## symptoms_Paroxysmal_nocturnal_dyspnea_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4694

## symptoms_Peripheral_edema_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4694

## symptoms_Pleural_effusion_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4694

## symptoms_Pulmonary_crepitations_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4694

## symptoms_Reduced_exercise_tolerance_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4694

## symptoms_Syncope_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4694

## symptoms_Tachycardia_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4694

## symptoms_Tachypnoea_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4694

## symptoms_Third_heart_sound_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4694

## symptoms_Weight_gain_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4694

## symptoms_Weight_loss_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4694

## echocardiographs_lvef_pET_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -179.0 / 96.0
- mean/std: 41.3116 / 18.658897448751244
- quantiles: {'0.05': 16.34, '0.25': 27.56, '0.5': 39.83, '0.75': 53.29, '0.95': 74.0}

## echocardiographs_lvef_pET_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -44.0 / 100.0
- mean/std: 41.1342 / 18.196033727841005
- quantiles: {'0.05': 14.89, '0.25': 27.57, '0.5': 40.0, '0.75': 53.41, '0.95': 73.0}

## electrocardiographs_ecg_qrs_duration_pET_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 20.0 / 252.0
- mean/std: 119.5618 / 34.709882752444486
- quantiles: {'0.05': 80.0, '0.25': 93.0, '0.5': 108.0, '0.75': 143.0, '0.95': 188.0}

## electrocardiographs_ecg_qrs_duration_pET_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 56.0 / 250.0
- mean/std: 118.2045 / 34.21510920487622
- quantiles: {'0.05': 78.0, '0.25': 92.0, '0.5': 108.0, '0.75': 141.0, '0.95': 184.0}

## electrocardiographs_ecg_qrs_axis_pET_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 31.883716394672575 / 119.3168883233668
- mean/std: nan / nan
- quantiles: {'0.05': 47.943978012164116, '0.25': 74.59084744701855, '0.5': 113.45073677816418, '0.75': nan, '0.95': nan}

## electrocardiographs_ecg_qrs_axis_pET_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 31.304142944626363 / 118.15545187557217
- mean/std: nan / nan
- quantiles: {'0.05': 47.923494321808036, '0.25': 73.75004853549743, '0.5': 114.28718291903022, '0.75': nan, '0.95': nan}

## electrocardiographs_ecg_qt_duration_corrected_pET_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 186.0 / 767.0
- mean/std: 472.9218 / 51.47418987965087
- quantiles: {'0.05': 399.0, '0.25': 439.0, '0.5': 469.0, '0.75': 502.0, '0.95': 559.0}

## electrocardiographs_ecg_qt_duration_corrected_pET_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 168.0 / 733.0
- mean/std: 469.7867 / 52.91375679911473
- quantiles: {'0.05': 393.0, '0.25': 436.0, '0.5': 464.0, '0.75': 499.0, '0.95': 562.0}

## electrocardiographs_ecg_st_pET

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `Missing`: 4694

## electrocardiographs_ecg_ischemia_without_st_pET

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `Missing`: 4694

## electrocardiographs_ecg_type_of_rhythms_pET_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `Missing`: 4694

## electrocardiographs_ecg_type_of_rhythms_pET_last

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `Missing`: 4694

## smoking_status_smoker_last

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 4409
  - `true`: 263
  - `false`: 22

## smoking_status_formerSmoker_last

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 4409
  - `true`: 264
  - `false`: 21

## smoking_status_smoker_totalSmokingDuration_sum

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 889.0 / 339891.0
- mean/std: 70621.8713 / 75198.96129122387
- quantiles: {'0.05': 3600.0, '0.25': 17046.0, '0.5': 41118.0, '0.75': 99016.0, '0.95': 233189.0}

## smoking_status_smoker_startTime_count

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0 / 19
- mean/std: 0.0807 / 0.8406362465188728
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 0.0}

## nyha_nyha_pET

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0 / 4
- mean/std: 0.6095 / 1.1385217979468278
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 3.0}

## hyperkalemia_severity_categorizedValue

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 5
- top values (shown only where count ≥ 5):
  - `normal`: 4130
  - `mild`: 241
  - `Missing`: 197
  - `moderate`: 85
  - `severe`: 41

## ckd_severity_categorizedValue

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 7
- top values (shown only where count ≥ 5):
  - `mildly_decreased`: 1341
  - `moderate_to_severe_decrease`: 814
  - `mild_to_moderate_decrease`: 782
  - `normal_or_high`: 626
  - `severe_decrease`: 617
  - `Missing`: 280
  - `kidney_failure`: 234

## conditions_heartFailure_timeFromEarliest_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 123.0
- mean/std: 11.3958 / 22.02786990758603
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 11.0, '0.95': 65.0}

## conditions_heart_failure_hf_within_18mo_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 4682
  - `False`: 12

## conditions_heart_failure_occurred_prior_to_18_months_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3717
  - `True`: 977

## encounter_primary_reason_HF_Disease_f5a_w7d_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 4222
  - `false`: 434
  - `true`: 38

## encounter_primary_reason_HF_Disease_f5a_w1mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 3577
  - `false`: 1001
  - `true`: 116

## encounter_primary_reason_HF_Disease_f5a_w3mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 2829
  - `false`: 1676
  - `true`: 189

## encounter_primary_reason_HF_Disease_f5a_w6mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 2508
  - `false`: 1969
  - `true`: 217

## encounter_primary_reason_HF_Disease_f5a_w1a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 2290
  - `false`: 2175
  - `true`: 229

## encounter_primary_reason_HF_Disease_f5a_w3a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `false`: 2375
  - `Missing`: 2079
  - `true`: 240

## encounter_primary_reason_HF_Disease_f5a_w5a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `false`: 2413
  - `Missing`: 2039
  - `true`: 242

## encounter_primary_reason_number_of_days_to_rehosp_for_heart_failure_f5a_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 1.0 / 1805.0
- mean/std: 91.3091 / 189.74733751076613
- quantiles: {'0.05': 1.0, '0.25': 13.0, '0.5': 35.0, '0.75': 83.0, '0.95': 460.0}

## encounter_primary_reason_CV_Disease_f5a_w7d_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 4222
  - `false`: 296
  - `true`: 176

## encounter_primary_reason_CV_Disease_f5a_w1mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 3577
  - `false`: 699
  - `true`: 418

## encounter_primary_reason_CV_Disease_f5a_w3mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 2829
  - `false`: 1214
  - `true`: 651

## encounter_primary_reason_CV_Disease_f5a_w6mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 2508
  - `false`: 1403
  - `true`: 783

## encounter_primary_reason_CV_Disease_f5a_w1a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 2290
  - `false`: 1555
  - `true`: 849

## encounter_primary_reason_CV_Disease_f5a_w3a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 2079
  - `false`: 1698
  - `true`: 917

## encounter_primary_reason_CV_Disease_f5a_w5a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 2039
  - `false`: 1726
  - `true`: 929

## encounter_primary_reason_number_of_days_to_rehosp_for_CV_f5a_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 1805.0
- mean/std: 117.6331 / 216.7712534507779
- quantiles: {'0.05': 2.0, '0.25': 12.0, '0.5': 38.0, '0.75': 116.0, '0.95': 573.0}

## encounter_primary_reason_non_CV_Disease_f5a_w7d_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 4222
  - `true`: 296
  - `false`: 176

## encounter_primary_reason_non_CV_Disease_f5a_w1mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 3577
  - `true`: 699
  - `false`: 418

## encounter_primary_reason_non_CV_Disease_f5a_w3mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 2829
  - `true`: 1214
  - `false`: 651

## encounter_primary_reason_non_CV_Disease_f5a_w6mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 2508
  - `true`: 1403
  - `false`: 783

## encounter_primary_reason_non_CV_Disease_f5a_w1a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 2290
  - `true`: 1555
  - `false`: 849

## encounter_primary_reason_non_CV_Disease_f5a_w3a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 2079
  - `true`: 1698
  - `false`: 917

## encounter_primary_reason_non_CV_Disease_f5a_w5a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 2039
  - `true`: 1726
  - `false`: 929

## encounter_primary_reason_number_of_days_to_rehosp_for_non_CV_f5a_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 1694.0
- mean/std: 134.1493 / 240.8074891218386
- quantiles: {'0.05': 2.0, '0.25': 14.0, '0.5': 45.0, '0.75': 120.0, '0.95': 653.0}

## encounter_primary_reason_renal_complications_f5a_w7d_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 4222
  - `false`: 466
  - `true`: 6

## encounter_primary_reason_renal_complications_f5a_w1mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 3577
  - `false`: 1095
  - `true`: 22

## encounter_primary_reason_renal_complications_f5a_w3mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 2829
  - `false`: 1823
  - `true`: 42

## encounter_primary_reason_renal_complications_f5a_w6mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 2508
  - `false`: 2137
  - `true`: 49

## encounter_primary_reason_renal_complications_f5a_w1a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `false`: 2348
  - `Missing`: 2290
  - `true`: 56

## encounter_primary_reason_renal_complications_f5a_w3a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `false`: 2554
  - `Missing`: 2079
  - `true`: 61

## encounter_primary_reason_renal_complications_f5a_w5a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `false`: 2594
  - `Missing`: 2039
  - `true`: 61

## encounter_primary_reason_number_of_days_to_rehosp_for_renal_complications_f5a_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 3.0 / 815.0
- mean/std: 123.1864 / 178.82265354778536
- quantiles: {'0.05': 5.0, '0.25': 17.0, '0.5': 56.0, '0.75': 135.0, '0.95': 648.0}

## cause_of_death_isCV_f5a_w7d_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 4541
  - `false`: 141
  - `true`: 12

## cause_of_death_isCV_f5a_w1mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 4334
  - `false`: 339
  - `true`: 21

## cause_of_death_isCV_f5a_w3mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 4051
  - `false`: 610
  - `true`: 33

## cause_of_death_isCV_f5a_w6mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 3804
  - `false`: 841
  - `true`: 49

## cause_of_death_isCV_f5a_w1a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 3521
  - `false`: 1108
  - `true`: 65

## cause_of_death_isCV_f5a_w3a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 2965
  - `false`: 1625
  - `true`: 104

## cause_of_death_isCV_f5a_w5a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 2680
  - `false`: 1904
  - `true`: 110

## cause_of_death_number_of_days_to_death_for_CV_f5a_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 1.0 / 1499.0
- mean/std: 367.8144 / 367.9076003904201
- quantiles: {'0.05': 1.0, '0.25': 72.0, '0.5': 258.0, '0.75': 563.0, '0.95': 1099.0}

## cause_of_death_isRenal_f5a_w7d_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 4541
  - `false`: 153

## cause_of_death_isRenal_f5a_w1mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 4334
  - `false`: 360

## cause_of_death_isRenal_f5a_w3mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 4051
  - `false`: 643

## cause_of_death_isRenal_f5a_w6mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 3804
  - `false`: 890

## cause_of_death_isRenal_f5a_w1a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 3521
  - `false`: 1173

## cause_of_death_isRenal_f5a_w3a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 2965
  - `false`: 1729

## cause_of_death_isRenal_f5a_w5a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 2680
  - `false`: 2014

## cause_of_death_isNonRenalAndNonCV_f5a_w7d_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 4541
  - `false`: 153

## cause_of_death_isNonRenalAndNonCV_f5a_w1mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 4334
  - `false`: 360

## cause_of_death_isNonRenalAndNonCV_f5a_w3mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 4051
  - `false`: 643

## cause_of_death_isNonRenalAndNonCV_f5a_w6mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 3804
  - `false`: 890

## cause_of_death_isNonRenalAndNonCV_f5a_w1a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 3521
  - `false`: 1173

## cause_of_death_isNonRenalAndNonCV_f5a_w3a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 2965
  - `false`: 1729

## cause_of_death_isNonRenalAndNonCV_f5a_w5a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 2680
  - `false`: 2014

## cause_of_death_isAllCause_f5a_w7d_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 4541
  - `true`: 141
  - `false`: 12

## cause_of_death_isAllCause_f5a_w1mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 4334
  - `true`: 339
  - `false`: 21

## cause_of_death_isAllCause_f5a_w3mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 4051
  - `true`: 610
  - `false`: 33

## cause_of_death_isAllCause_f5a_w6mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 3804
  - `true`: 841
  - `false`: 49

## cause_of_death_isAllCause_f5a_w1a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 3521
  - `true`: 1108
  - `false`: 65

## cause_of_death_isAllCause_f5a_w3a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 2965
  - `true`: 1625
  - `false`: 104

## cause_of_death_isAllCause_f5a_w5a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 2680
  - `true`: 1904
  - `false`: 110

## cause_of_death_number_of_days_to_death_for_all_cause_f5a_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 1.0 / 1826.0
- mean/std: 451.2049 / 490.1994838367664
- quantiles: {'0.05': 5.0, '0.25': 54.0, '0.5': 244.0, '0.75': 735.0, '0.95': 1500.0}

## eGFR_2021_ckd_epi_creatinine

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 20.091671 / 231.686837
- mean/std: 64.1440 / 25.885709754232593
- quantiles: {'0.05': 29.356253, '0.25': 42.152854, '0.5': 60.948432, '0.75': 84.605864, '0.95': 106.57567}

## ckd_severity_from_calculated_egfr

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 6
- top values (shown only where count ≥ 5):
  - `mildly_decreased`: 1360
  - `moderate_to_severe_decrease`: 954
  - `normal_or_high`: 840
  - `mild_to_moderate_decrease`: 833
  - `Missing`: 504
  - `severe_decrease`: 203

## ckd_severity_calculated_or_measured

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 7
- top values (shown only where count ≥ 5):
  - `mildly_decreased`: 1361
  - `moderate_to_severe_decrease`: 959
  - `normal_or_high`: 841
  - `mild_to_moderate_decrease`: 833
  - `severe_decrease`: 455
  - `kidney_failure`: 208
  - `Missing`: 37

## beta_blocker_use_pre_dc

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 3563
  - `False`: 1131

## ace_inhibitors_arb_use_pre_dc

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 3399
  - `False`: 1295

## maggic_total_score

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 14.0 / 29.0
- mean/std: 22.4026 / 6.440784165750511
- quantiles: {'0.05': 14.0, '0.25': 16.0, '0.5': 28.0, '0.75': 29.0, '0.95': 29.0}

## med_acei

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2894
  - `True`: 1800

## med_anti_coag

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 3434
  - `False`: 1260

## med_anti_plat

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2959
  - `True`: 1735

## med_antiarrhytmic

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4054
  - `True`: 640

## med_antiinfl

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4638
  - `True`: 56

## med_arb

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3593
  - `True`: 1101

## med_ari

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4694

## med_arni

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4300
  - `True`: 394

## med_bb

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 3277
  - `False`: 1417

## med_ccb

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3667
  - `True`: 1027

## med_cortico_syst

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3760
  - `True`: 934

## med_digitalis

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3714
  - `True`: 980

## med_diuretics

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 4367
  - `False`: 327

## med_diuretics_loop

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 4271
  - `False`: 423

## med_inotropes

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3812
  - `True`: 882

## med_insulins

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3586
  - `True`: 1108

## med_ivabradine

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4583
  - `True`: 111

## med_ll

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2449
  - `True`: 2245

## med_mra

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 2457
  - `False`: 2237

## med_oral_antidiabetic

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3260
  - `True`: 1434

## med_platelet

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2932
  - `True`: 1762

## med_potassium_binders

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4521
  - `True`: 173

## med_rasi

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 2728
  - `False`: 1966

## med_rdoad

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3072
  - `True`: 1622

## med_rdoad_syst

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4674
  - `True`: 20

## med_thrombolytic

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4666
  - `True`: 28

## med_vasodil

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3700
  - `True`: 994

## med_acei_history

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3021
  - `True`: 1673

## med_anti_coag_history

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 2354
  - `False`: 2340

## med_anti_plat_history

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3408
  - `True`: 1286

## med_antiarrhytmic_history

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4299
  - `True`: 395

## med_antiinfl_history

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4518
  - `True`: 176

## med_arb_history

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3628
  - `True`: 1066

## med_ari_history

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4694

## med_arni_history

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4380
  - `True`: 314

## med_bb_history

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 2611
  - `False`: 2083

## med_ccb_history

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3430
  - `True`: 1264

## med_cortico_syst_history

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3550
  - `True`: 1144

## med_digitalis_history

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4120
  - `True`: 574

## med_diuretics_history

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 3644
  - `False`: 1050

## med_diuretics_loop_history

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 3529
  - `False`: 1165

## med_inotropes_history

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3514
  - `True`: 1180

## med_insulins_history

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3910
  - `True`: 784

## med_ivabradine_history

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4604
  - `True`: 90

## med_ll_history

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3317
  - `True`: 1377

## med_mra_history

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3067
  - `True`: 1627

## med_oral_antidiabetic_history

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3827
  - `True`: 867

## med_platelet_history

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3395
  - `True`: 1299

## med_potassium_binders_history

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4512
  - `True`: 182

## med_rasi_history

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 2358
  - `False`: 2336

## med_rdoad_history

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3558
  - `True`: 1136

## med_rdoad_syst_history

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4663
  - `True`: 31

## med_thrombolytic_history

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4660
  - `True`: 34

## med_vasodil_history

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3532
  - `True`: 1162

## conditions_af

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2420
  - `True`: 2274

## conditions_aidshiv

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4653
  - `True`: 41

## conditions_ap

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3998
  - `True`: 696

## conditions_ckd_chronic

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2852
  - `True`: 1842

## conditions_cm

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3708
  - `True`: 986

## conditions_copd

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3848
  - `True`: 846

## conditions_dem

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4574
  - `True`: 120

## conditions_dep

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4585
  - `True`: 109

## conditions_devices

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3573
  - `True`: 1121

## conditions_dia

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4694

## conditions_diabetes

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3172
  - `True`: 1522

## conditions_dysl

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3818
  - `True`: 876

## conditions_hf

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 4683
  - `False`: 11

## conditions_hyp

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 2571
  - `False`: 2123

## conditions_hyperthyroid

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4566
  - `True`: 128

## conditions_hypothyroid

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4359
  - `True`: 335

## conditions_ibd

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4493
  - `True`: 201

## conditions_ihd

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2396
  - `True`: 2298

## conditions_ld

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4355
  - `True`: 339

## conditions_mc

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3908
  - `True`: 786

## conditions_mi

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3906
  - `True`: 788

## conditions_myocarditis

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4676
  - `True`: 18

## conditions_osa

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4361
  - `True`: 333

## conditions_pad

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3848
  - `True`: 846

## conditions_pericardial

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4520
  - `True`: 174

## conditions_rd

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4438
  - `True`: 256

## conditions_revasc

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3856
  - `True`: 838

## conditions_stroke

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4268
  - `True`: 426

## conditions_substance_abuse

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4024
  - `True`: 670

## conditions_tia

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4374
  - `True`: 320

## conditions_vd

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2714
  - `True`: 1980

## vital_signs_weight_value_p6mo_first_was_missing

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4303
  - `True`: 391

## vital_signs_weight_value_p6mo_last_was_missing

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4303
  - `True`: 391

## vital_signs_height_value_p1a_avg_was_missing

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3735
  - `True`: 959

## vital_signs_weight_value_last_was_missing

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3976
  - `True`: 718

## vital_signs_height_value_last_was_missing

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2637
  - `True`: 2057

## vital_signs_bmi_value_last_was_missing

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2499
  - `True`: 2195

## vital_signs_systolicBp_value_first_was_missing

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4453
  - `True`: 241

## vital_signs_systolicBp_value_last_was_missing

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4453
  - `True`: 241

## vital_signs_diastolicBp_value_first_was_missing

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4453
  - `True`: 241

## vital_signs_diastolicBp_value_last_was_missing

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4453
  - `True`: 241

## vital_signs_heartRate_value_first_was_missing

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2436
  - `True`: 2258

## vital_signs_heartRate_value_last_was_missing

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2436
  - `True`: 2258

## lab_results_hemoglobin_value_first_was_missing

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4547
  - `True`: 147

## lab_results_hemoglobin_value_last_was_missing

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4547
  - `True`: 147

## lab_results_ferritin_value_first_was_missing

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 3765
  - `False`: 929

## lab_results_ferritin_value_last_was_missing

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 3765
  - `False`: 929

## lab_results_tropTHs_value_first_was_missing

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 2969
  - `False`: 1725

## lab_results_tropTHs_value_last_was_missing

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 2969
  - `False`: 1725

## lab_results_triGly_value_first_was_missing

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 4251
  - `False`: 443

## lab_results_triGly_value_last_was_missing

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 4251
  - `False`: 443

## lab_results_cholTot_value_first_was_missing

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 4180
  - `False`: 514

## lab_results_cholTot_value_last_was_missing

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 4180
  - `False`: 514

## lab_results_hdl_value_first_was_missing

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 4276
  - `False`: 418

## lab_results_hdl_value_last_was_missing

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 4276
  - `False`: 418

## lab_results_potassium_value_first_was_missing

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4497
  - `True`: 197

## lab_results_potassium_value_last_was_missing

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4497
  - `True`: 197

## lab_results_creatUS_value_first_was_missing

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 4217
  - `False`: 477

## lab_results_creatUS_value_last_was_missing

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 4217
  - `False`: 477

## lab_results_albuminBS_value_first_was_missing

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 2699
  - `False`: 1995

## lab_results_albuminBS_value_last_was_missing

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 2699
  - `False`: 1995

## lab_results_validSerumCreatinine_value_first_was_missing

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4190
  - `True`: 504

## lab_results_validSerumCreatinine_value_last_was_missing

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4190
  - `True`: 504

## lab_results_valideGFR_value_first_was_missing

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4414
  - `True`: 280

## lab_results_valideGFR_value_last_was_missing

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4414
  - `True`: 280

## echocardiographs_lvef_pET_last_was_missing

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 3902
  - `False`: 792

## echocardiographs_lvef_pET_first_was_missing

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 3853
  - `False`: 841

## electrocardiographs_ecg_qrs_duration_pET_first_was_missing

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2385
  - `True`: 2309

## electrocardiographs_ecg_qrs_duration_pET_last_was_missing

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2384
  - `True`: 2310

## electrocardiographs_ecg_qt_duration_corrected_pET_last_was_missing

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2383
  - `True`: 2311

## electrocardiographs_ecg_qt_duration_corrected_pET_first_was_missing

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2384
  - `True`: 2310

## smoking_status_smoker_totalSmokingDuration_sum_was_missing

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 4611
  - `False`: 83

## conditions_heartFailure_timeFromEarliest_first_was_missing

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4683
  - `True`: 11

## eGFR_2021_ckd_epi_creatinine_was_missing

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4190
  - `True`: 504

## maggic_total_score_was_missing

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 4688
  - `False`: 6

## encounter_primary_reason_number_of_days_to_rehosp_for_heart_failure_f5a_first_was_missing

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 4452
  - `False`: 242

## encounter_primary_reason_number_of_days_to_rehosp_for_CV_f5a_first_was_missing

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 3765
  - `False`: 929

## encounter_primary_reason_number_of_days_to_rehosp_for_non_CV_f5a_first_was_missing

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 2968
  - `False`: 1726

## encounter_primary_reason_number_of_days_to_rehosp_for_renal_complications_f5a_first_was_missing

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 4633
  - `False`: 61

## cause_of_death_number_of_days_to_death_for_CV_f5a_first_was_missing

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 4584
  - `False`: 110

## cause_of_death_number_of_days_to_death_for_all_cause_f5a_first_was_missing

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 2789
  - `False`: 1905
