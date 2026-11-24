# Column Analysis

Total rows: 2065
Total columns: 256

## patient_demographics_gender

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `male`: 1280
  - `female`: 785

## patient_demographics_age

- dtype: `Int32` (numeric)
- nulls: 0 (0.00%)
- min/max: 18.0 / 100.0
- mean/std: 72.7971 / 12.495774141626429
- quantiles: {'0.05': 49.0, '0.25': 66.0, '0.5': 74.0, '0.75': 82.0, '0.95': 90.0}

## encounters_encounterClass

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `IMP`: 2065

## encounters_admissionYear

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 14
- top values (shown only where count ≥ 5):
  - `2025`: 243
  - `2016`: 221
  - `2017`: 203
  - `2018`: 171
  - `2015`: 162
  - `2014`: 159
  - `2021`: 149
  - `2019`: 139
  - `2024`: 119
  - `2023`: 117
  - `2022`: 108
  - `2026`: 108
  - `2020`: 106
  - `2013`: 60

## encounters_lengthOfStay

- dtype: `Int32` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 69.0
- mean/std: 6.5278 / 6.328715507587193
- quantiles: {'0.05': 1.0, '0.25': 2.0, '0.5': 4.0, '0.75': 8.0, '0.95': 18.0}

## encounters_numOfPreviousHFStays_count

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 450.0
- mean/std: 10.6475 / 22.56793522169907
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 4.0, '0.75': 12.0, '0.95': 41.799999999999955}

## vital_signs_weight_value_p6mo_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 3.7 / 190.0
- mean/std: 74.5113 / 32.34740039581911
- quantiles: {'0.05': 3.75, '0.25': 65.0, '0.5': 80.0, '0.75': 92.0, '0.95': 116.0}

## vital_signs_weight_value_p6mo_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 3.7 / 190.0
- mean/std: 74.2657 / 32.154440770882225
- quantiles: {'0.05': 3.75, '0.25': 65.0, '0.5': 80.0, '0.75': 92.0, '0.95': 115.0}

## vital_signs_height_value_p1a_avg

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 69.0 / 200.0
- mean/std: 169.8684 / 14.2131994470703
- quantiles: {'0.05': 155.0, '0.25': 165.0, '0.5': 171.0, '0.75': 178.0, '0.95': 185.0}

## vital_signs_weight_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 3.7 / 190.0
- mean/std: 71.9119 / 34.14163357372316
- quantiles: {'0.05': 3.75, '0.25': 62.0, '0.5': 79.0, '0.75': 92.0, '0.95': 115.0}

## vital_signs_height_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 69.0 / 200.0
- mean/std: 169.3821 / 15.90855556113342
- quantiles: {'0.05': 155.0, '0.25': 165.0, '0.5': 171.0, '0.75': 178.0, '0.95': 185.79999999999995}

## vital_signs_diastolicBp_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -30.0 / 150.0
- mean/std: 22.1424 / 53.22465979284772
- quantiles: {'0.05': -29.5, '0.25': -29.5, '0.5': -29.5, '0.75': 75.0, '0.95': 90.0}

## vital_signs_diastolicBp_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -30.0 / 150.0
- mean/std: 22.1424 / 53.22465979284772
- quantiles: {'0.05': -29.5, '0.25': -29.5, '0.5': -29.5, '0.75': 75.0, '0.95': 90.0}

## vital_signs_heartRate_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -14.0 / 180.0
- mean/std: 26.9527 / 50.06809285529284
- quantiles: {'0.05': -13.75, '0.25': -13.75, '0.5': -13.75, '0.75': 75.0, '0.95': 110.0}

## vital_signs_heartRate_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -14.0 / 180.0
- mean/std: 26.9527 / 50.06809285529284
- quantiles: {'0.05': -13.75, '0.25': -13.75, '0.5': -13.75, '0.75': 75.0, '0.95': 110.0}

## vital_signs_systolicBp_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 17.0 / 230.0
- mean/std: 72.4908 / 57.86121512454399
- quantiles: {'0.05': 17.5, '0.25': 17.5, '0.5': 17.5, '0.75': 125.0, '0.95': 157.0}

## vital_signs_systolicBp_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 17.0 / 230.0
- mean/std: 72.4908 / 57.86121512454399
- quantiles: {'0.05': 17.5, '0.25': 17.5, '0.5': 17.5, '0.75': 125.0, '0.95': 157.0}

## lab_results_hemoglobin_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 13.0 / 200.0
- mean/std: 32.1036 / 43.42843519971091
- quantiles: {'0.05': 13.0, '0.25': 13.0, '0.5': 13.0, '0.75': 13.0, '0.95': 140.0}

## lab_results_hemoglobin_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 13.0 / 200.0
- mean/std: 32.1312 / 43.497991997621455
- quantiles: {'0.05': 13.0, '0.25': 13.0, '0.5': 13.0, '0.75': 13.0, '0.95': 140.0}

## lab_results_ntProBnp_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -8800.0 / 35000.0
- mean/std: 580.9390 / 10166.20477948227
- quantiles: {'0.05': -8750.0, '0.25': -8750.0, '0.5': 838.0, '0.75': 4498.0, '0.95': 21042.0}

## lab_results_ntProBnp_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -8800.0 / 35000.0
- mean/std: 585.2295 / 10166.609772156173
- quantiles: {'0.05': -8750.0, '0.25': -8750.0, '0.5': 838.0, '0.75': 4500.0, '0.95': 21042.0}

## lab_results_crpNonHs_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -99.0 / 400.0
- mean/std: 35.7782 / 66.07340372591341
- quantiles: {'0.05': 0.6, '0.25': 2.7, '0.5': 10.1, '0.75': 44.5, '0.95': 180.07999999999996}

## lab_results_crpNonHs_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -99.0 / 400.0
- mean/std: 35.7839 / 66.21493628562759
- quantiles: {'0.05': 0.6, '0.25': 2.7, '0.5': 10.0, '0.75': 44.4, '0.95': 180.07999999999996}

## lab_results_tropTHs_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -2.5 / 10.0
- mean/std: -0.9488 / 2.2261261751280905
- quantiles: {'0.05': -2.5, '0.25': -2.5, '0.5': -2.5, '0.75': 0.1, '0.95': 3.0437999999999983}

## lab_results_tropTHs_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -2.5 / 10.0
- mean/std: -0.9374 / 2.2530221608966716
- quantiles: {'0.05': -2.5, '0.25': -2.5, '0.5': -2.5, '0.75': 0.102, '0.95': 3.1145999999999994}

## lab_results_triGly_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -3.4 / 14.0
- mean/std: -1.1487 / 2.4514169850659324
- quantiles: {'0.05': -3.325, '0.25': -3.325, '0.5': -3.325, '0.75': 1.04, '0.95': 2.1779999999999995}

## lab_results_triGly_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -3.0 / 14.0
- mean/std: -0.9280 / 2.256373244949068
- quantiles: {'0.05': -2.9125, '0.25': -2.9125, '0.5': -2.9125, '0.75': 1.04, '0.95': 2.1779999999999995}

## lab_results_cholTot_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -0.69 / 8.9
- mean/std: 1.5086 / 2.534732565602363
- quantiles: {'0.05': -0.685, '0.25': -0.685, '0.5': -0.685, '0.75': 3.81, '0.95': 5.71}

## lab_results_cholTot_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -0.69 / 8.9
- mean/std: 1.5107 / 2.534926626123274
- quantiles: {'0.05': -0.685, '0.25': -0.685, '0.5': -0.685, '0.75': 3.81, '0.95': 5.71}

## lab_results_hdl_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -1.0 / 3.0
- mean/std: -0.0572 / 1.059332219668732
- quantiles: {'0.05': -1.0, '0.25': -1.0, '0.5': -1.0, '0.75': 0.99, '0.95': 1.5179999999999996}

## lab_results_hdl_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -0.87 / 3.0
- mean/std: 0.0144 / 0.9966289140334424
- quantiles: {'0.05': -0.87, '0.25': -0.87, '0.5': -0.87, '0.75': 0.99, '0.95': 1.5179999999999996}

## lab_results_potassium_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.57 / 7.7
- mean/std: 4.0281 / 0.900648034435819
- quantiles: {'0.05': 3.0, '0.25': 3.7, '0.5': 4.1, '0.75': 4.5, '0.95': 5.2}

## lab_results_potassium_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.72 / 7.1
- mean/std: 4.0393 / 0.872037828989452
- quantiles: {'0.05': 3.0, '0.25': 3.7, '0.5': 4.1, '0.75': 4.5, '0.95': 5.2}

## lab_results_sodium_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 100.0 / 170.0
- mean/std: 136.5238 / 7.962329181520606
- quantiles: {'0.05': 126.0, '0.25': 135.0, '0.5': 138.0, '0.75': 141.0, '0.95': 144.0}

## lab_results_sodium_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 98.0 / 170.0
- mean/std: 136.5446 / 8.206708301662278
- quantiles: {'0.05': 126.0, '0.25': 135.0, '0.5': 138.0, '0.75': 141.0, '0.95': 144.0}

## lab_results_albuminBS_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 8.8 / 56.0
- mean/std: 19.2545 / 14.582831479721575
- quantiles: {'0.05': 8.825000000000001, '0.25': 8.825000000000001, '0.5': 8.825000000000001, '0.75': 35.8, '0.95': 44.08}

## lab_results_albuminBS_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 8.8 / 56.0
- mean/std: 19.2547 / 14.581599430294832
- quantiles: {'0.05': 8.825000000000001, '0.25': 8.825000000000001, '0.5': 8.825000000000001, '0.75': 35.9, '0.95': 44.08}

## lab_results_hba1c_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -2.5 / 130.0
- mean/std: 11.0816 / 24.2568818726025
- quantiles: {'0.05': -2.5, '0.25': -2.5, '0.5': -2.5, '0.75': 35.0, '0.95': 59.0}

## lab_results_hba1c_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -2.5 / 130.0
- mean/std: 11.0816 / 24.2568818726025
- quantiles: {'0.05': -2.5, '0.25': -2.5, '0.5': -2.5, '0.75': 35.0, '0.95': 59.0}

## lab_results_validSerumCreatinine_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -1.5 / 23.0
- mean/std: 10.9978 / 5.434242938213378
- quantiles: {'0.05': -1.4125, '0.25': 8.701, '0.5': 11.074, '0.75': 14.238, '0.95': 20.001}

## lab_results_valideGFR_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -35.0 / 150.0
- mean/std: 51.0446 / 29.57371005646807
- quantiles: {'0.05': 8.4, '0.25': 33.6, '0.5': 51.0, '0.75': 72.0, '0.95': 95.4}

## lab_results_valideGFR_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -35.0 / 150.0
- mean/std: 51.0504 / 29.61117739652722
- quantiles: {'0.05': 8.4, '0.25': 33.6, '0.5': 51.0, '0.75': 71.4, '0.95': 96.0}

## symptoms_Ankle_swelling_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 1633
  - `true`: 432

## symptoms_Ascites_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 2052
  - `true`: 13

## symptoms_Breathlessness_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 1953
  - `true`: 112

## symptoms_Cardiac_murmur_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 2063
  - 1 other distinct value(s) suppressed, covering 2 row(s) (count below 5 and/or ranked beyond top 20)

## symptoms_Chest_pain_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 2019
  - `true`: 46

## symptoms_Cheyne_stokes_respiration_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 2065

## symptoms_Depression_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 1745
  - `true`: 320

## symptoms_Dizziness_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 2036
  - `true`: 29

## symptoms_Elevated_jugular_venous_pressure_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 2065

## symptoms_Fatigue_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 2052
  - `true`: 13

## symptoms_Hepatojugular_reflux_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 2065

## symptoms_Hepatomegaly_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 2051
  - `true`: 14

## symptoms_Intermittent_claudication_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 2065

## symptoms_Irregular_pulse_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 2058
  - `true`: 7

## symptoms_Loss_of_appetite_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 2065

## symptoms_Nocturnal_cough_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 2065

## symptoms_Oliguria_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 2065

## symptoms_Orthopnoea_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 2062
  - 1 other distinct value(s) suppressed, covering 3 row(s) (count below 5 and/or ranked beyond top 20)

## symptoms_Palpitations_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 2063
  - 1 other distinct value(s) suppressed, covering 2 row(s) (count below 5 and/or ranked beyond top 20)

## symptoms_Paroxysmal_nocturnal_dyspnea_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 2064
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## symptoms_Peripheral_edema_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 1795
  - `true`: 270

## symptoms_Pleural_effusion_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 1942
  - `true`: 123

## symptoms_Pulmonary_crepitations_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 2043
  - `true`: 22

## symptoms_Reduced_exercise_tolerance_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 2065

## symptoms_Syncope_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 2052
  - `true`: 13

## symptoms_Tachycardia_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 2001
  - `true`: 64

## symptoms_Tachypnoea_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 2063
  - 1 other distinct value(s) suppressed, covering 2 row(s) (count below 5 and/or ranked beyond top 20)

## symptoms_Third_heart_sound_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 2060
  - `true`: 5

## symptoms_Weight_gain_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 2065

## symptoms_Weight_loss_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 2065

## echocardiographs_lvef

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -8.8 / 75.0
- mean/std: 20.5730 / 26.825780216018938
- quantiles: {'0.05': -8.75, '0.25': -8.75, '0.5': 25.0, '0.75': 44.0, '0.95': 62.799999999999955}

## echocardiographs_lvef_pET_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -18.0 / 75.0
- mean/std: 9.0601 / 30.011791937099137
- quantiles: {'0.05': -17.75, '0.25': -17.75, '0.5': -17.75, '0.75': 35.0, '0.95': 60.0}

## echocardiographs_lvef_pET_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -8.8 / 75.0
- mean/std: 13.7289 / 25.986296176652505
- quantiles: {'0.05': -8.75, '0.25': -8.75, '0.5': -8.75, '0.75': 35.0, '0.95': 60.0}

## electrocardiographs_ecg_qrs_duration_pET_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -31.0 / 250.0
- mean/std: 38.2165 / 80.93416024564344
- quantiles: {'0.05': -31.0, '0.25': -31.0, '0.5': -31.0, '0.75': 107.0, '0.95': 180.0}

## electrocardiographs_ecg_qrs_duration_pET_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -44.0 / 280.0
- mean/std: 30.8168 / 86.5757153184939
- quantiles: {'0.05': -43.75, '0.25': -43.75, '0.5': -43.75, '0.75': 108.0, '0.95': 173.0}

## electrocardiographs_ecg_qrs_axis_pET_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -180.0 / 270.0
- mean/std: -93.8840 / 106.58733720222159
- quantiles: {'0.05': -177.5, '0.25': -177.5, '0.5': -177.5, '0.75': -14.0, '0.95': 91.0}

## electrocardiographs_ecg_qrs_axis_pET_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -180.0 / 270.0
- mean/std: -94.2770 / 105.10164033015673
- quantiles: {'0.05': -177.75, '0.25': -177.75, '0.5': -177.75, '0.75': -13.0, '0.95': 89.0}

## electrocardiographs_ecg_qt_duration_corrected_pET_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 58.0 / 690.0
- mean/std: 240.5519 / 205.81177118559626
- quantiles: {'0.05': 58.25, '0.25': 58.25, '0.5': 58.25, '0.75': 458.0, '0.95': 522.0}

## electrocardiographs_ecg_qt_duration_corrected_pET_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 92.0 / 670.0
- mean/std: 260.4523 / 189.8348808879219
- quantiles: {'0.05': 92.5, '0.25': 92.5, '0.5': 92.5, '0.75': 458.0, '0.95': 528.8}

## electrocardiographs_ecg_st_pET

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 1148
  - `false`: 896
  - `true`: 21

## electrocardiographs_ecg_ischemia_without_st_pET

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 1148
  - `false`: 898
  - `true`: 19

## electrocardiographs_ecg_type_of_rhythms_pET_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 1141
  - `LA17059-9`: 924

## electrocardiographs_ecg_type_of_rhythms_pET_last

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 1141
  - `LA17059-9`: 924

## smoking_status_smoker_last

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 1667
  - `true`: 398

## smoking_status_formerSmoker_last

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `false`: 743
  - `true`: 682
  - `Missing`: 640

## smoking_status_smoker_totalSmokingDuration_sum

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -7600.0 / 31000.0
- mean/std: -5626.1288 / 3715.1474873260513
- quantiles: {'0.05': -7573.0, '0.25': -7573.0, '0.5': -7573.0, '0.75': -7573.0, '0.95': 1064.399999999999}

## smoking_status_smoker_startTime_count

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 15.0
- mean/std: 0.4029 / 1.053046191893137
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 2.0}

## nyha_nyha

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 1.0 / 3.0
- mean/std: 2.3627 / 0.5325312586629632
- quantiles: {'0.05': 2.0, '0.25': 2.0, '0.5': 2.0, '0.75': 3.0, '0.95': 3.0}

## nyha_nyha_pET

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 1.0 / 3.0
- mean/std: 2.3792 / 0.5525244217848939
- quantiles: {'0.05': 2.0, '0.25': 2.0, '0.5': 2.0, '0.75': 3.0, '0.95': 3.0}

## vital_signs_systolicBpDuringEncounter_value_pET

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 60.0 / 230.0
- mean/std: 129.3101 / 17.60202276907315
- quantiles: {'0.05': 100.2, '0.25': 120.0, '0.5': 128.29355177418137, '0.75': 137.04476072292678, '0.95': 160.0}

## vital_signs_bmi_value_pET

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 1.9 / 71.0
- mean/std: 27.9619 / 6.103675420463612
- quantiles: {'0.05': 19.782647899074277, '0.25': 23.875114784205692, '0.5': 27.34375, '0.75': 31.141868512110726, '0.95': 38.56011693933772}

## lab_results_creatBS_value_p3a_avg

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 3.7 / 87.0
- mean/std: 13.9429 / 7.587868038994555
- quantiles: {'0.05': 7.29528, '0.25': 9.680333333333333, '0.5': 12.091, '0.75': 15.661799999999998, '0.95': 26.540471428571426}

## lab_results_validSerumCreatinine_value_pET

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 3.3 / 23.0
- mean/std: 12.5272 / 3.927252612022608
- quantiles: {'0.05': 7.232, '0.25': 9.605, '0.5': 11.751999999999999, '0.75': 14.802999999999999, '0.95': 20.34}

## hyperkalemia_severity_categorizedValue

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 5
- top values (shown only where count ≥ 5):
  - `normal`: 1827
  - `mild`: 123
  - `Missing`: 61
  - `moderate`: 37
  - `severe`: 17

## ckd_severity_categorizedValue

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 7
- top values (shown only where count ≥ 5):
  - `mildly_decreased`: 607
  - `mild_to_moderate_decrease`: 462
  - `moderate_to_severe_decrease`: 388
  - `severe_decrease`: 250
  - `normal_or_high`: 197
  - `kidney_failure`: 83
  - `Missing`: 78

## conditions_heartFailure_timeFromEarliest_first

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 150.0
- mean/std: 5.7351 / 18.33675229660521
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 41.799999999999955}

## conditions_heart_failure_hf_within_18mo_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `true`: 2065

## conditions_heart_failure_occurred_prior_to_18_months_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 1875
  - `true`: 190

## encounter_primary_reason_HF_Disease_f5a_w7d_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 1823
  - `false`: 229
  - `true`: 13

## encounter_primary_reason_HF_Disease_f5a_w1mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 1470
  - `false`: 565
  - `true`: 30

## encounter_primary_reason_HF_Disease_f5a_w3mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 1030
  - `false`: 990
  - `true`: 45

## encounter_primary_reason_HF_Disease_f5a_w6mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `false`: 1114
  - `Missing`: 906
  - `true`: 45

## encounter_primary_reason_HF_Disease_f5a_w1a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `false`: 1202
  - `Missing`: 817
  - `true`: 46

## encounter_primary_reason_HF_Disease_f5a_w3a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `false`: 1312
  - `Missing`: 706
  - `true`: 47

## encounter_primary_reason_HF_Disease_f5a_w5a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `false`: 1341
  - `Missing`: 677
  - `true`: 47

## encounter_primary_reason_number_of_HF_rehospitalizations_5a_f5a_count

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -7.5 / 30.0
- mean/std: 0.3305 / 1.5797099451190986
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 2.0}

## encounter_primary_reason_CV_Disease_f5a_w7d_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 1823
  - `true`: 129
  - `false`: 113

## encounter_primary_reason_CV_Disease_f5a_w1mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 1470
  - `true`: 305
  - `false`: 290

## encounter_primary_reason_CV_Disease_f5a_w3mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 1030
  - `false`: 536
  - `true`: 499

## encounter_primary_reason_CV_Disease_f5a_w6mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 906
  - `false`: 590
  - `true`: 569

## encounter_primary_reason_CV_Disease_f5a_w1a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 817
  - `false`: 636
  - `true`: 612

## encounter_primary_reason_CV_Disease_f5a_w3a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 706
  - `false`: 688
  - `true`: 671

## encounter_primary_reason_CV_Disease_f5a_w5a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `false`: 704
  - `true`: 684
  - `Missing`: 677

## encounter_primary_reason_number_of_days_to_rehosp_for_CV_f5a_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -430.0 / 1700.0
- mean/std: -238.9844 / 301.5391370404057
- quantiles: {'0.05': -422.75, '0.25': -422.75, '0.5': -422.75, '0.75': 13.0, '0.95': 211.79999999999995}

## encounter_primary_reason_number_of_CV_rehospitalizations_5a_f5a_count

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -13.0 / 50.0
- mean/std: 4.8753 / 8.446684720604887
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 1.0, '0.75': 6.0, '0.95': 24.0}

## encounter_primary_reason_non_CV_Disease_f5a_w7d_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 1823
  - `false`: 129
  - `true`: 113

## encounter_primary_reason_non_CV_Disease_f5a_w1mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 1470
  - `false`: 305
  - `true`: 290

## encounter_primary_reason_non_CV_Disease_f5a_w3mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 1030
  - `true`: 536
  - `false`: 499

## encounter_primary_reason_non_CV_Disease_f5a_w6mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 906
  - `true`: 590
  - `false`: 569

## encounter_primary_reason_non_CV_Disease_f5a_w1a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 817
  - `true`: 636
  - `false`: 612

## encounter_primary_reason_non_CV_Disease_f5a_w3a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 706
  - `true`: 688
  - `false`: 671

## encounter_primary_reason_non_CV_Disease_f5a_w5a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `true`: 704
  - `false`: 684
  - `Missing`: 677

## encounter_primary_reason_number_of_days_to_rehosp_for_non_CV_f5a_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -460.0 / 1900.0
- mean/std: -255.2407 / 319.4826838235619
- quantiles: {'0.05': -455.25, '0.25': -455.25, '0.5': -455.25, '0.75': 17.0, '0.95': 220.79999999999973}

## encounter_primary_reason_number_of_non_CV_rehospitalizations_5a_f5a_count

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -13.0 / 50.0
- mean/std: 4.6741 / 8.235560064431771
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 1.0, '0.75': 7.0, '0.95': 21.0}

## encounter_primary_reason_renal_complications_f5a_w7d_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 1823
  - `false`: 242

## encounter_primary_reason_renal_complications_f5a_w1mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `Missing`: 1470
  - `false`: 594
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## encounter_primary_reason_renal_complications_f5a_w3mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `false`: 1034
  - `Missing`: 1030
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## encounter_primary_reason_renal_complications_f5a_w6mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `false`: 1158
  - `Missing`: 906
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## encounter_primary_reason_renal_complications_f5a_w1a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `false`: 1246
  - `Missing`: 817
  - 1 other distinct value(s) suppressed, covering 2 row(s) (count below 5 and/or ranked beyond top 20)

## encounter_primary_reason_renal_complications_f5a_w3a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `false`: 1357
  - `Missing`: 706
  - 1 other distinct value(s) suppressed, covering 2 row(s) (count below 5 and/or ranked beyond top 20)

## encounter_primary_reason_renal_complications_f5a_w5a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `false`: 1386
  - `Missing`: 677
  - 1 other distinct value(s) suppressed, covering 2 row(s) (count below 5 and/or ranked beyond top 20)

## encounter_primary_reason_number_of_renal_rehospitalizations_5a_f5a_count

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 8.0
- mean/std: 0.0576 / 0.3710335272370817
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 0.0}

## cause_of_death_isCV_f5a_w7d_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 1873
  - `false`: 192

## cause_of_death_isCV_f5a_w1mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 1787
  - `false`: 278

## cause_of_death_isCV_f5a_w3mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 1700
  - `false`: 365

## cause_of_death_isCV_f5a_w6mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 1606
  - `false`: 459

## cause_of_death_isCV_f5a_w1a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 1502
  - `false`: 563

## cause_of_death_isCV_f5a_w3a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 1244
  - `false`: 821

## cause_of_death_isCV_f5a_w5a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 1088
  - `false`: 977

## cause_of_death_isRenal_f5a_w7d_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 1873
  - `false`: 192

## cause_of_death_isRenal_f5a_w1mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 1787
  - `false`: 278

## cause_of_death_isRenal_f5a_w3mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 1700
  - `false`: 365

## cause_of_death_isRenal_f5a_w6mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 1606
  - `false`: 459

## cause_of_death_isRenal_f5a_w1a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 1502
  - `false`: 563

## cause_of_death_isRenal_f5a_w3a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 1244
  - `false`: 821

## cause_of_death_isRenal_f5a_w5a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 1088
  - `false`: 977

## cause_of_death_isNonRenalAndNonCV_f5a_w7d_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 1873
  - `false`: 192

## cause_of_death_isNonRenalAndNonCV_f5a_w1mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 1787
  - `false`: 278

## cause_of_death_isNonRenalAndNonCV_f5a_w3mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 1700
  - `false`: 365

## cause_of_death_isNonRenalAndNonCV_f5a_w6mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 1606
  - `false`: 459

## cause_of_death_isNonRenalAndNonCV_f5a_w1a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 1502
  - `false`: 563

## cause_of_death_isNonRenalAndNonCV_f5a_w3a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 1244
  - `false`: 821

## cause_of_death_isNonRenalAndNonCV_f5a_w5a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 1088
  - `false`: 977

## cause_of_death_isAllCause_f5a_w7d_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 1873
  - `true`: 192

## cause_of_death_isAllCause_f5a_w1mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 1787
  - `true`: 278

## cause_of_death_isAllCause_f5a_w3mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 1700
  - `true`: 365

## cause_of_death_isAllCause_f5a_w6mo_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 1606
  - `true`: 459

## cause_of_death_isAllCause_f5a_w1a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 1502
  - `true`: 563

## cause_of_death_isAllCause_f5a_w3a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 1244
  - `true`: 821

## cause_of_death_isAllCause_f5a_w5a_first

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `Missing`: 1088
  - `true`: 977

## cause_of_death_number_of_days_to_death_for_all_cause_f5a_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -460.0 / 1900.0
- mean/std: -23.1496 / 579.9250071648815
- quantiles: {'0.05': -454.75, '0.25': -454.75, '0.5': -454.75, '0.75': 181.0, '0.95': 1297.7999999999997}

## eGFR_2021_ckd_epi_creatinine

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 20.0 / 140.0
- mean/std: 61.1687 / 22.487808215982625
- quantiles: {'0.05': 28.1155732, '0.25': 42.845709, '0.5': 59.125829, '0.75': 77.762526, '0.95': 99.9534602}

## ckd_severity_from_calculated_egfr

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 5
- top values (shown only where count ≥ 5):
  - `mildly_decreased`: 752
  - `mild_to_moderate_decrease`: 487
  - `moderate_to_severe_decrease`: 423
  - `normal_or_high`: 286
  - `severe_decrease`: 117

## ckd_severity_calculated_or_measured

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 5
- top values (shown only where count ≥ 5):
  - `mildly_decreased`: 752
  - `mild_to_moderate_decrease`: 487
  - `moderate_to_severe_decrease`: 423
  - `normal_or_high`: 286
  - `severe_decrease`: 117

## maggic_total_score

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -6.3 / 40.0
- mean/std: 6.9689 / 14.55348518154287
- quantiles: {'0.05': -6.25, '0.25': -6.25, '0.5': -6.25, '0.75': 21.0, '0.95': 29.0}

## med_acei

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 1078
  - `true`: 987

## med_activeDuringEncounter_ace_inhibitors_arb_use

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 1555
  - `false`: 510

## med_activeDuringEncounter_bb

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 1619
  - `false`: 446

## med_anti_coag

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 1218
  - `false`: 847

## med_anti_plat

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 1475
  - `true`: 590

## med_antiarrhytmic

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 1746
  - `true`: 319

## med_antiinfl

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 2054
  - `true`: 11

## med_arb

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 1679
  - `true`: 386

## med_ari

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 2065

## med_arni

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 1967
  - `true`: 98

## med_bb

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 1461
  - `false`: 604

## med_ccb

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 1679
  - `true`: 386

## med_cortico_syst

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 2065

## med_digitalis

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 1715
  - `true`: 350

## med_diuretics

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 1790
  - `false`: 275

## med_diuretics_loop

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 1628
  - `false`: 437

## med_inotropes

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 2056
  - `true`: 9

## med_insulins

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 1963
  - `true`: 102

## med_ivabradine

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 2044
  - `true`: 21

## med_ll

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 1103
  - `false`: 962

## med_mra

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 1289
  - `false`: 776

## med_oral_antidiabetic

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 1413
  - `true`: 652

## med_platelet

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 1475
  - `true`: 590

## med_potassium_binders

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 2063
  - 1 other distinct value(s) suppressed, covering 2 row(s) (count below 5 and/or ranked beyond top 20)

## med_rasi

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 1344
  - `false`: 721

## med_rdoad

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 1934
  - `true`: 131

## med_rdoad_syst

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 1959
  - `true`: 106

## med_sglt2i

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 1689
  - `true`: 376

## med_thrombolytic

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 2064
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## med_vasodil

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 2015
  - `true`: 50

## med_acei_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 1516
  - `true`: 549

## med_anti_coag_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 1485
  - `true`: 580

## med_anti_plat_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 1780
  - `true`: 285

## med_antiarrhytmic_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 1851
  - `true`: 214

## med_antiinfl_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 2046
  - `true`: 19

## med_arb_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 1813
  - `true`: 252

## med_ari_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 2065

## med_arni_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 2018
  - `true`: 47

## med_bb_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 1366
  - `true`: 699

## med_ccb_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 1746
  - `true`: 319

## med_cortico_syst_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 2065

## med_digitalis_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 1874
  - `true`: 191

## med_diuretics_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 1312
  - `true`: 753

## med_diuretics_loop_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 1412
  - `true`: 653

## med_inotropes_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 2059
  - `true`: 6

## med_insulins_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 1976
  - `true`: 89

## med_ivabradine_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 2048
  - `true`: 17

## med_ll_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 1523
  - `true`: 542

## med_mra_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 1520
  - `true`: 545

## med_oral_antidiabetic_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 1732
  - `true`: 333

## med_platelet_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 1780
  - `true`: 285

## med_potassium_binders_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 2064
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## med_rasi_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 1346
  - `true`: 719

## med_rdoad_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 1959
  - `true`: 106

## med_rdoad_syst_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 1966
  - `true`: 99

## med_sglt2i_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 1941
  - `true`: 124

## med_thrombolytic_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 2065

## med_vasodil_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 2002
  - `true`: 63

## conditions_af

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 1052
  - `false`: 1013

## conditions_aidshiv

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 2065

## conditions_ap

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 1866
  - `true`: 199

## conditions_ckd_chronic

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 1844
  - `true`: 221

## conditions_cm

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 1731
  - `true`: 334

## conditions_copd

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 1669
  - `true`: 396

## conditions_dem

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 2040
  - `true`: 25

## conditions_dep

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 1987
  - `true`: 78

## conditions_devices

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 1526
  - `true`: 539

## conditions_dia

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 2065

## conditions_diabetes

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 1191
  - `true`: 874

## conditions_dysl

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 1095
  - `true`: 970

## conditions_hf

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `true`: 2065

## conditions_hyp

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 1666
  - `false`: 399

## conditions_hyperthyroid

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 2049
  - `true`: 16

## conditions_hypothyroid

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 1883
  - `true`: 182

## conditions_ibd

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 2047
  - `true`: 18

## conditions_ihd

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 1555
  - `false`: 510

## conditions_ld

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 2041
  - `true`: 24

## conditions_mc

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 1982
  - `true`: 83

## conditions_mi

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 1287
  - `true`: 778

## conditions_myocarditis

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 2039
  - `true`: 26

## conditions_osa

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 2038
  - `true`: 27

## conditions_pad

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 1590
  - `true`: 475

## conditions_pericardial

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 1923
  - `true`: 142

## conditions_rd

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 2034
  - `true`: 31

## conditions_revasc

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 2034
  - `true`: 31

## conditions_stroke

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 1815
  - `true`: 250

## conditions_substance_abuse

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 2050
  - `true`: 15

## conditions_tia

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 1958
  - `true`: 107

## conditions_vd

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 1756
  - `true`: 309
