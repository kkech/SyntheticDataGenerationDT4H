# Column Analysis

Total rows: 4694
Total columns: 249

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
- min/max: -12.149999999999999 / 208.6
- mean/std: 70.0049 / 31.243358495877064
- quantiles: {'0.05': -12.149999999999999, '0.25': 60.6, '0.5': 72.8, '0.75': 86.2, '0.95': 111.8}

## vital_signs_weight_value_p6mo_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -17.875 / 241.0
- mean/std: 71.6087 / 33.280508520322165
- quantiles: {'0.05': -17.875, '0.25': 62.1, '0.5': 75.0, '0.75': 89.0, '0.95': 115.0}

## vital_signs_height_value_p1a_avg

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 69.43749999999999 / 207.0
- mean/std: 150.3669 / 42.08120802741982
- quantiles: {'0.05': 69.43749999999999, '0.25': 155.0, '0.5': 168.0, '0.75': 176.0, '0.95': 186.0}

## vital_signs_weight_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -12.149999999999999 / 208.6
- mean/std: 63.7557 / 37.111593537189556
- quantiles: {'0.05': -12.149999999999999, '0.25': 56.0, '0.5': 70.7, '0.75': 85.0, '0.95': 111.0}

## vital_signs_height_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 85.75 / 207.0
- mean/std: 133.9408 / 43.29328737413144
- quantiles: {'0.05': 85.75, '0.25': 85.75, '0.5': 158.5, '0.75': 173.0, '0.95': 185.0}

## vital_signs_bmi_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -3.3814626314492653 / 79.5847750865052
- mean/std: 12.7445 / 15.841776803245297
- quantiles: {'0.05': -3.3814626314492653, '0.25': -3.3814626314492653, '0.5': 18.920068027210885, '0.75': 26.299357208448118, '0.95': 34.85952133194589}

## vital_signs_systolicBp_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -6.75 / 257.0
- mean/std: 123.3156 / 41.169908358972776
- quantiles: {'0.05': -6.75, '0.25': 107.0, '0.5': 124.0, '0.75': 147.0, '0.95': 180.0}

## vital_signs_systolicBp_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 7.25 / 221.0
- mean/std: 115.2953 / 33.34934022541762
- quantiles: {'0.05': 7.25, '0.25': 103.0, '0.5': 116.0, '0.75': 134.0, '0.95': 160.0}

## vital_signs_diastolicBp_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -40.25 / 201.0
- mean/std: 69.5996 / 31.135519313945515
- quantiles: {'0.05': -40.25, '0.25': 62.0, '0.5': 72.0, '0.75': 85.0, '0.95': 108.0}

## vital_signs_diastolicBp_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -32.0 / 158.0
- mean/std: 63.9617 / 25.906621057394084
- quantiles: {'0.05': -32.0, '0.25': 59.0, '0.5': 68.0, '0.75': 77.0, '0.95': 91.0}

## vital_signs_heartRate_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -54.5 / 223.0
- mean/std: 32.1949 / 84.32726490839673
- quantiles: {'0.05': -54.5, '0.25': -54.5, '0.5': 100.0, '0.75': 108.0, '0.95': 134.0}

## vital_signs_heartRate_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -63.5 / 254.0
- mean/std: 26.3411 / 87.1626820559667
- quantiles: {'0.05': -63.5, '0.25': -63.5, '0.5': 100.0, '0.75': 106.0, '0.95': 126.0}

## lab_results_hemoglobin_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -18.128249999999994 / 233.653
- mean/std: 115.4606 / 32.98599848397717
- quantiles: {'0.05': 77.3472, '0.25': 101.5182, '0.5': 117.6322, '0.75': 135.3576, '0.95': 157.9172}

## lab_results_hemoglobin_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -54.586175 / 219.1504
- mean/std: 116.0506 / 38.87815542080427
- quantiles: {'0.05': 72.513, '0.25': 103.1296, '0.5': 120.855, '0.75': 136.969, '0.95': 159.5286}

## lab_results_ferritin_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -10514.0 / 42076.0
- mean/std: -8335.4605 / 4453.0885458139255
- quantiles: {'0.05': -10514.0, '0.25': -10514.0, '0.5': -10514.0, '0.75': -10514.0, '0.95': 438.1}

## lab_results_ferritin_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -10514.0 / 42076.0
- mean/std: -8329.7407 / 4481.541767800647
- quantiles: {'0.05': -10514.0, '0.25': -10514.0, '0.5': -10514.0, '0.75': -10514.0, '0.95': 423.0}

## lab_results_ntProBnp_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -17458.75 / 70000.0
- mean/std: 4707.8375 / 16573.254571901765
- quantiles: {'0.05': -17458.75, '0.25': 613.0, '0.5': 3131.0000000000005, '0.75': 8951.0, '0.95': 34274.0}

## lab_results_ntProBnp_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -17456.875 / 70000.0
- mean/std: 4777.6985 / 16294.573073119645
- quantiles: {'0.05': -17456.875, '0.25': 649.0, '0.5': 3344.9999999999995, '0.75': 9516.000000000002, '0.95': 33083.0}

## lab_results_crpNonHs_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -157.05 / 629.7
- mean/std: 20.7614 / 86.32173961764433
- quantiles: {'0.05': -157.05, '0.25': 3.8, '0.5': 15.2, '0.75': 47.2, '0.95': 159.0}

## lab_results_crpNonHs_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -144.15 / 578.1
- mean/std: 23.5356 / 90.0225384239732
- quantiles: {'0.05': -144.15, '0.25': 2.9, '0.5': 12.0, '0.75': 44.0, '0.95': 191.8}

## lab_results_tropTHs_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -16.4075 / 65.65
- mean/std: -10.2070 / 8.25741333184113
- quantiles: {'0.05': -16.4075, '0.25': -16.4075, '0.5': -16.4075, '0.75': 0.039, '0.95': 0.496}

## lab_results_tropTHs_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -4.34225 / 17.389
- mean/std: -2.6641 / 2.268041932929854
- quantiles: {'0.05': -4.34225, '0.25': -4.34225, '0.5': -4.34225, '0.75': 0.034, '0.95': 0.25}

## lab_results_tropTnHs_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -3846.25 / 15400.0
- mean/std: -3344.2135 / 1385.3238580691257
- quantiles: {'0.05': -3846.25, '0.25': -3846.25, '0.5': -3846.25, '0.75': -3846.25, '0.95': 98.0}

## lab_results_tropTnHs_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -1531.25 / 6140.0
- mean/std: -1319.9785 / 599.3517801696339
- quantiles: {'0.05': -1531.25, '0.25': -1531.25, '0.5': -1531.25, '0.75': -1531.25, '0.95': 76.0}

## lab_results_triGly_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -2.5875 / 11.35
- mean/std: -2.2012 / 1.2388397444044164
- quantiles: {'0.05': -2.5875, '0.25': -2.5875, '0.5': -2.5875, '0.75': -2.5875, '0.95': 1.12}

## lab_results_triGly_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -3.875 / 16.5
- mean/std: -3.3632 / 1.6345195301522157
- quantiles: {'0.05': -3.875, '0.25': -3.875, '0.5': -3.875, '0.75': -3.875, '0.95': 1.13}

## lab_results_cholTot_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -0.7150000000000003 / 8.71
- mean/std: -0.2076 / 1.4954801556956043
- quantiles: {'0.05': -0.7150000000000003, '0.25': -0.7150000000000003, '0.5': -0.7150000000000003, '0.75': -0.7150000000000003, '0.95': 3.92}

## lab_results_cholTot_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -0.7150000000000003 / 8.71
- mean/std: -0.2065 / 1.4979863803754487
- quantiles: {'0.05': -0.7150000000000003, '0.25': -0.7150000000000003, '0.5': -0.7150000000000003, '0.75': -0.7150000000000003, '0.95': 3.95}

## lab_results_hdl_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -0.78 / 3.76
- mean/std: -0.6048 / 0.5755344590611388
- quantiles: {'0.05': -0.78, '0.25': -0.78, '0.5': -0.78, '0.75': -0.78, '0.95': 1.06}

## lab_results_hdl_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -0.84 / 3.85
- mean/std: -0.6594 / 0.5931275774185593
- quantiles: {'0.05': -0.84, '0.25': -0.84, '0.5': -0.84, '0.75': -0.84, '0.95': 1.06}

## lab_results_ldl_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -1.561875 / 6.4475
- mean/std: -1.2384 / 1.0790694183895384
- quantiles: {'0.05': -1.561875, '0.25': -1.561875, '0.5': -1.561875, '0.75': -1.561875, '0.95': 1.85}

## lab_results_ldl_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -1.561875 / 6.4475
- mean/std: -1.2375 / 1.0812921968667804
- quantiles: {'0.05': -1.561875, '0.25': -1.561875, '0.5': -1.561875, '0.75': -1.561875, '0.95': 1.86}

## lab_results_potassium_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.4375 / 7.5
- mean/std: 4.0209 / 0.9405741270962902
- quantiles: {'0.05': 3.0, '0.25': 3.74, '0.5': 4.1, '0.75': 4.49, '0.95': 5.16}

## lab_results_potassium_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.11249999999999982 / 8.8
- mean/std: 4.1142 / 1.0730361932794357
- quantiles: {'0.05': 2.76, '0.25': 3.8, '0.5': 4.2, '0.75': 4.61, '0.95': 5.5}

## lab_results_sodium_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 96.25 / 160.0
- mean/std: 136.1089 / 9.537349683837508
- quantiles: {'0.05': 124.0, '0.25': 135.0, '0.5': 138.0, '0.75': 140.6, '0.95': 144.2}

## lab_results_sodium_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 95.0 / 165.0
- mean/std: 135.3411 / 9.809221150922442
- quantiles: {'0.05': 119.0, '0.25': 134.0, '0.5': 137.7, '0.75': 140.1, '0.95': 143.4}

## lab_results_creatUS_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -621.0287999999999 / 2706.9616
- mean/std: -488.9696 / 421.4404675052635
- quantiles: {'0.05': -621.0287999999999, '0.25': -621.0287999999999, '0.5': -621.0287999999999, '0.75': -621.0287999999999, '0.95': 535.0576}

## lab_results_creatUS_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -654.1163999999999 / 2839.312
- mean/std: -513.1379 / 448.24269065801525
- quantiles: {'0.05': -654.1163999999999, '0.25': -654.1163999999999, '0.5': -654.1163999999999, '0.75': -654.1163999999999, '0.95': 622.16}

## lab_results_albuminBS_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -1.25 / 50.0
- mean/std: 12.0252 / 16.083019280566806
- quantiles: {'0.05': -1.25, '0.25': -1.25, '0.5': -1.25, '0.75': 28.6, '0.95': 38.0}

## lab_results_albuminBS_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 50.0
- mean/std: 13.2616 / 16.04857350219392
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 30.0, '0.95': 39.0}

## lab_results_validSerumCreatinine_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -4.654065000000001 / 22.57476
- mean/std: 10.4043 / 6.769829361724951
- quantiles: {'0.05': -4.654065000000001, '0.25': 7.79259, '0.5': 10.72188, '0.75': 14.5899, '0.95': 20.8104}

## lab_results_validSerumCreatinine_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -4.51269 / 22.57476
- mean/std: 10.5510 / 6.727587290513293
- quantiles: {'0.05': -4.51269, '0.25': 8.00748, '0.5': 10.8576, '0.75': 14.8161, '0.95': 20.6973}

## lab_results_valideGFR_value_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -18.75 / 90.0
- mean/std: 48.8560 / 29.332048881379112
- quantiles: {'0.05': -18.75, '0.25': 29.0, '0.5': 51.0, '0.75': 72.0, '0.95': 90.0}

## lab_results_valideGFR_value_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -18.75 / 90.0
- mean/std: 49.3460 / 29.732852868819776
- quantiles: {'0.05': -18.75, '0.25': 29.0, '0.5': 51.0, '0.75': 74.0, '0.95': 90.0}

## symptoms_Ankle_swelling_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 4694

## symptoms_Ascites_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 4694

## symptoms_Breathlessness_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 4694

## symptoms_Cardiac_murmur_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 4694

## symptoms_Chest_pain_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 4694

## symptoms_Cheyne_stokes_respiration_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 4694

## symptoms_Depression_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 4694

## symptoms_Dizziness_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 4694

## symptoms_Elevated_jugular_venous_pressure_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 4694

## symptoms_Fatigue_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 4694

## symptoms_Hepatojugular_reflux_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 4694

## symptoms_Hepatomegaly_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 4694

## symptoms_Intermittent_claudication_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 4694

## symptoms_Irregular_pulse_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 4694

## symptoms_Loss_of_appetite_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 4694

## symptoms_Nocturnal_cough_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 4694

## symptoms_Oliguria_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 4694

## symptoms_Orthopnoea_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 4694

## symptoms_Palpitations_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 4694

## symptoms_Paroxysmal_nocturnal_dyspnea_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 4694

## symptoms_Peripheral_edema_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 4694

## symptoms_Pleural_effusion_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 4694

## symptoms_Pulmonary_crepitations_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 4694

## symptoms_Reduced_exercise_tolerance_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 4694

## symptoms_Syncope_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 4694

## symptoms_Tachycardia_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 4694

## symptoms_Tachypnoea_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 4694

## symptoms_Third_heart_sound_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 4694

## symptoms_Weight_gain_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 4694

## symptoms_Weight_loss_display_pET_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 4694

## echocardiographs_lvef_pET_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -247.75 / 96.0
- mean/std: -199.0191 / 108.46347836737515
- quantiles: {'0.05': -247.75, '0.25': -247.75, '0.5': -247.75, '0.75': -247.75, '0.95': 50.86}

## echocardiographs_lvef_pET_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -80.0 / 100.0
- mean/std: -58.2251 / 47.244731609764926
- quantiles: {'0.05': -80.0, '0.25': -80.0, '0.5': -80.0, '0.75': -80.0, '0.95': 52.0}

## electrocardiographs_ecg_qrs_duration_pET_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -38.0 / 252.0
- mean/std: 41.9022 / 82.35581789334437
- quantiles: {'0.05': -38.0, '0.25': -38.0, '0.5': 72.0, '0.75': 109.0, '0.95': 170.0}

## electrocardiographs_ecg_qrs_duration_pET_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 7.5 / 250.0
- mean/std: 63.9122 / 60.78353781785869
- quantiles: {'0.05': 7.5, '0.25': 7.5, '0.5': 71.0, '0.75': 109.0, '0.95': 170.0}

## electrocardiographs_ecg_qt_duration_corrected_pET_first

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 40.75 / 767.0
- mean/std: 260.0201 / 218.87256073341533
- quantiles: {'0.05': 40.75, '0.25': 40.75, '0.5': 377.0, '0.75': 470.0, '0.95': 537.0}

## electrocardiographs_ecg_qt_duration_corrected_pET_last

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 26.75 / 733.0
- mean/std: 251.1647 / 224.2075725693615
- quantiles: {'0.05': 26.75, '0.25': 26.75, '0.5': 368.0, '0.75': 464.0, '0.95': 536.0}

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
- min/max: -30.75 / 123.0
- mean/std: 11.2769 / 22.073221388517265
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 11.0, '0.95': 65.0}

## conditions_heart_failure_hf_within_18mo_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 4682
  - `false`: 12

## conditions_heart_failure_occurred_prior_to_18_months_any

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 3717
  - `true`: 977

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
- min/max: -450.0 / 1805.0
- mean/std: -422.1253 / 127.59553841778593
- quantiles: {'0.05': -450.0, '0.25': -450.0, '0.5': -450.0, '0.75': -450.0, '0.95': 1.0}

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
- min/max: -451.25 / 1805.0
- mean/std: -338.3629 / 248.5220964548068
- quantiles: {'0.05': -451.25, '0.25': -451.25, '0.5': -451.25, '0.75': -451.25, '0.95': 110.0}

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
- min/max: -423.5 / 1694.0
- mean/std: -219.3941 / 304.524922949788
- quantiles: {'0.05': -423.5, '0.25': -423.5, '0.5': -423.5, '0.75': 20.0, '0.95': 279.0}

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
- min/max: -455.25 / 1826.0
- mean/std: -86.3181 / 544.6524489735771
- quantiles: {'0.05': -455.25, '0.25': -455.25, '0.5': -455.25, '0.75': 130.0, '0.95': 1157.0}

## eGFR_2021_ckd_epi_creatinine

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: -32.807120499999996 / 231.686837
- mean/std: 53.6113 / 38.7027671330604
- quantiles: {'0.05': -32.807120499999996, '0.25': 35.872781, '0.5': 56.39836, '0.75': 80.832657, '0.95': 105.515767}

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

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 3563
  - `false`: 1131

## ace_inhibitors_arb_use_pre_dc

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 3399
  - `false`: 1295

## med_acei

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 2894
  - `true`: 1800

## med_anti_coag

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 3434
  - `false`: 1260

## med_anti_plat

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 2959
  - `true`: 1735

## med_antiarrhytmic

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 4054
  - `true`: 640

## med_antiinfl

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 4638
  - `true`: 56

## med_arb

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 3593
  - `true`: 1101

## med_ari

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 4694

## med_arni

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 4300
  - `true`: 394

## med_bb

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 3277
  - `false`: 1417

## med_ccb

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 3667
  - `true`: 1027

## med_cortico_syst

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 3760
  - `true`: 934

## med_digitalis

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 3714
  - `true`: 980

## med_diuretics

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 4367
  - `false`: 327

## med_diuretics_loop

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 4271
  - `false`: 423

## med_inotropes

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 3812
  - `true`: 882

## med_insulins

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 3586
  - `true`: 1108

## med_ivabradine

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 4583
  - `true`: 111

## med_ll

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 2449
  - `true`: 2245

## med_mra

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 2457
  - `false`: 2237

## med_oral_antidiabetic

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 3260
  - `true`: 1434

## med_platelet

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 2932
  - `true`: 1762

## med_potassium_binders

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 4521
  - `true`: 173

## med_rasi

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 2728
  - `false`: 1966

## med_rdoad

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 3072
  - `true`: 1622

## med_rdoad_syst

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 4674
  - `true`: 20

## med_thrombolytic

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 4666
  - `true`: 28

## med_vasodil

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 3700
  - `true`: 994

## med_acei_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 3021
  - `true`: 1673

## med_anti_coag_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 2354
  - `false`: 2340

## med_anti_plat_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 3408
  - `true`: 1286

## med_antiarrhytmic_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 4299
  - `true`: 395

## med_antiinfl_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 4518
  - `true`: 176

## med_arb_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 3628
  - `true`: 1066

## med_ari_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 4694

## med_arni_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 4380
  - `true`: 314

## med_bb_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 2611
  - `false`: 2083

## med_ccb_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 3430
  - `true`: 1264

## med_cortico_syst_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 3550
  - `true`: 1144

## med_digitalis_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 4120
  - `true`: 574

## med_diuretics_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 3644
  - `false`: 1050

## med_diuretics_loop_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 3529
  - `false`: 1165

## med_inotropes_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 3514
  - `true`: 1180

## med_insulins_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 3910
  - `true`: 784

## med_ivabradine_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 4604
  - `true`: 90

## med_ll_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 3317
  - `true`: 1377

## med_mra_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 3067
  - `true`: 1627

## med_oral_antidiabetic_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 3827
  - `true`: 867

## med_platelet_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 3395
  - `true`: 1299

## med_potassium_binders_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 4512
  - `true`: 182

## med_rasi_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 2358
  - `false`: 2336

## med_rdoad_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 3558
  - `true`: 1136

## med_rdoad_syst_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 4663
  - `true`: 31

## med_thrombolytic_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 4660
  - `true`: 34

## med_vasodil_history

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 3532
  - `true`: 1162

## conditions_af

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 2420
  - `true`: 2274

## conditions_aidshiv

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 4653
  - `true`: 41

## conditions_ap

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 3998
  - `true`: 696

## conditions_ckd_chronic

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 2852
  - `true`: 1842

## conditions_cm

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 3708
  - `true`: 986

## conditions_copd

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 3848
  - `true`: 846

## conditions_dem

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 4574
  - `true`: 120

## conditions_dep

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 4585
  - `true`: 109

## conditions_devices

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 3573
  - `true`: 1121

## conditions_dia

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `false`: 4694

## conditions_diabetes

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 3172
  - `true`: 1522

## conditions_dysl

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 3818
  - `true`: 876

## conditions_hf

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 4683
  - `false`: 11

## conditions_hyp

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `true`: 2571
  - `false`: 2123

## conditions_hyperthyroid

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 4566
  - `true`: 128

## conditions_hypothyroid

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 4359
  - `true`: 335

## conditions_ibd

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 4493
  - `true`: 201

## conditions_ihd

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 2396
  - `true`: 2298

## conditions_ld

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 4355
  - `true`: 339

## conditions_mc

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 3908
  - `true`: 786

## conditions_mi

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 3906
  - `true`: 788

## conditions_myocarditis

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 4676
  - `true`: 18

## conditions_osa

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 4361
  - `true`: 333

## conditions_pad

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 3848
  - `true`: 846

## conditions_pericardial

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 4520
  - `true`: 174

## conditions_rd

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 4438
  - `true`: 256

## conditions_revasc

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 3856
  - `true`: 838

## conditions_stroke

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 4268
  - `true`: 426

## conditions_substance_abuse

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 4024
  - `true`: 670

## conditions_tia

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 4374
  - `true`: 320

## conditions_vd

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `false`: 2714
  - `true`: 1980
