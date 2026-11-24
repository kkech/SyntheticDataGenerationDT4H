# Column Analysis

Total rows: 5031
Total columns: 540

## pid

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 5031
- ⚠️ high-cardinality column (5031 distinct values)
- top values (shown only where count ≥ 5):
  - (none met the display threshold)
  - 5031 other distinct value(s) suppressed, covering 5031 row(s) (count below 5 and/or ranked beyond top 20)

## encounterId

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 5031
- ⚠️ high-cardinality column (5031 distinct values)
- top values (shown only where count ≥ 5):
  - (none met the display threshold)
  - 5031 other distinct value(s) suppressed, covering 5031 row(s) (count below 5 and/or ranked beyond top 20)

## referenceTimePoint

- dtype: `Datetime(time_unit='ns', time_zone=None)` (categorical)
- nulls: 0 (0.00%)
- unique values: 4989
- ⚠️ high-cardinality column (4989 distinct values)
- top values (shown only where count ≥ 5):
  - (none met the display threshold)
  - 4989 other distinct value(s) suppressed, covering 5031 row(s) (count below 5 and/or ranked beyond top 20)

## eventTime

- dtype: `Datetime(time_unit='ns', time_zone=None)` (categorical)
- nulls: 0 (0.00%)
- unique values: 4798
- ⚠️ high-cardinality column (4798 distinct values)
- top values (shown only where count ≥ 5):
  - `2020-03-02 08:00:00`: 6
  - `2020-05-26 08:00:00`: 6
  - `2019-11-25 08:00:00`: 5
  - `2019-12-16 08:00:00`: 5
  - `2020-02-10 08:00:00`: 5
  - `2020-04-22 08:00:00`: 5
  - `2020-06-25 08:00:00`: 5
  - `2021-03-29 08:00:00`: 5
  - 4790 other distinct value(s) suppressed, covering 4989 row(s) (count below 5 and/or ranked beyond top 20)

## exitTime

- dtype: `Datetime(time_unit='ns', time_zone=None)` (categorical)
- nulls: 5031 (100.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 5031

## patient_demographics_sourceIdentifier

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 5031
- ⚠️ high-cardinality column (5031 distinct values)
- top values (shown only where count ≥ 5):
  - (none met the display threshold)
  - 5031 other distinct value(s) suppressed, covering 5031 row(s) (count below 5 and/or ranked beyond top 20)

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

## encounters_admissionDate

- dtype: `Datetime(time_unit='ns', time_zone=None)` (categorical)
- nulls: 0 (0.00%)
- unique values: 4798
- ⚠️ high-cardinality column (4798 distinct values)
- top values (shown only where count ≥ 5):
  - `2020-03-02 08:00:00`: 6
  - `2020-05-26 08:00:00`: 6
  - `2019-11-25 08:00:00`: 5
  - `2019-12-16 08:00:00`: 5
  - `2020-02-10 08:00:00`: 5
  - `2020-04-22 08:00:00`: 5
  - `2020-06-25 08:00:00`: 5
  - `2021-03-29 08:00:00`: 5
  - 4790 other distinct value(s) suppressed, covering 4989 row(s) (count below 5 and/or ranked beyond top 20)

## encounters_dischargeDate

- dtype: `Datetime(time_unit='ns', time_zone=None)` (categorical)
- nulls: 0 (0.00%)
- unique values: 4989
- ⚠️ high-cardinality column (4989 distinct values)
- top values (shown only where count ≥ 5):
  - (none met the display threshold)
  - 4989 other distinct value(s) suppressed, covering 5031 row(s) (count below 5 and/or ranked beyond top 20)

## encounters_numOfPreviousHFStays_count

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 420.0
- mean/std: 9.0952 / 19.371123774654777
- quantiles: {'0.05': 0.0, '0.25': 1.0, '0.5': 2.0, '0.75': 10.0, '0.95': 38.0}

## vital_signs_weight_value_p6mo_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1788 (35.54%)
- min/max: 0.0 / 180.0
- mean/std: 74.4644 / 16.048544075669756
- quantiles: {'0.05': 50.0, '0.25': 65.0, '0.5': 74.0, '0.75': 83.0, '0.95': 100.0}

## vital_signs_weight_value_p6mo_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1788 (35.54%)
- min/max: 0.0 / 180.0
- mean/std: 75.2622 / 16.051884616560915
- quantiles: {'0.05': 52.0, '0.25': 65.0, '0.5': 75.0, '0.75': 84.0, '0.95': 100.0}

## vital_signs_weight_value_p6mo_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 1788 (35.54%)
- min/max: 0.0 / 180.0
- mean/std: 74.5047 / 15.877362959501108
- quantiles: {'0.05': 50.410000000000004, '0.25': 65.0, '0.5': 74.0, '0.75': 83.0, '0.95': 100.0}

## vital_signs_weight_value_p6mo_stddev

- dtype: `Float64` (numeric)
- nulls: 1788 (35.54%)
- min/max: 0.0 / 41.0
- mean/std: 0.6847 / 2.385269330479403
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 4.714045207910316}

## vital_signs_weight_value_p6mo_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1788 (35.54%)
- min/max: 0.0 / 180.0
- mean/std: 74.5281 / 16.081491717135243
- quantiles: {'0.05': 50.0, '0.25': 65.0, '0.5': 74.0, '0.75': 83.0, '0.95': 100.0}

## vital_signs_weight_value_p6mo_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1788 (35.54%)
- min/max: 0.0 / 180.0
- mean/std: 73.7074 / 16.132819839086643
- quantiles: {'0.05': 50.0, '0.25': 64.0, '0.5': 72.0, '0.75': 82.0, '0.95': 100.0}

## vital_signs_height_value_p1a_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 1170 (23.26%)
- min/max: 0.0 / 220.0
- mean/std: 167.8143 / 10.198950306591216
- quantiles: {'0.05': 153.0, '0.25': 160.375, '0.5': 168.0, '0.75': 175.0, '0.95': 182.0}

## vital_signs_weight_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2612 (51.92%)
- min/max: 0.0 / 180.0
- mean/std: 74.2537 / 15.76192123893108
- quantiles: {'0.05': 50.0, '0.25': 65.0, '0.5': 74.0, '0.75': 83.0, '0.95': 100.0}

## vital_signs_height_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2058 (40.91%)
- min/max: 0.0 / 200.0
- mean/std: 167.9084 / 10.919899072422655
- quantiles: {'0.05': 153.0, '0.25': 162.0, '0.5': 169.0, '0.75': 175.0, '0.95': 182.0}

## vital_signs_diastolicBp_value_stddev

- dtype: `Float64` (numeric)
- nulls: 1412 (28.07%)
- min/max: 0.0 / 30.0
- mean/std: 6.8780 / 2.805194828148075
- quantiles: {'0.05': 0.0, '0.25': 5.357864684673976, '0.5': 7.043392293490404, '0.75': 8.434628394048094, '0.95': 10.957261858054306}

## vital_signs_diastolicBp_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1412 (28.07%)
- min/max: 37.0 / 130.0
- mean/std: 70.8881 / 10.41459086058263
- quantiles: {'0.05': 57.0, '0.25': 60.0, '0.5': 70.0, '0.75': 80.0, '0.95': 90.0}

## vital_signs_diastolicBp_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1412 (28.07%)
- min/max: 45.0 / 150.0
- mean/std: 81.8218 / 9.70746870449402
- quantiles: {'0.05': 70.0, '0.25': 80.0, '0.5': 80.0, '0.75': 90.0, '0.95': 100.0}

## vital_signs_diastolicBp_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1412 (28.07%)
- min/max: 30.0 / 110.0
- mean/std: 57.9632 / 8.274888120498549
- quantiles: {'0.05': 45.0, '0.25': 50.0, '0.5': 60.0, '0.75': 60.0, '0.95': 70.0}

## vital_signs_diastolicBp_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1412 (28.07%)
- min/max: 30.0 / 130.0
- mean/std: 68.3960 / 9.114249061717848
- quantiles: {'0.05': 56.0, '0.25': 60.0, '0.5': 70.0, '0.75': 71.0, '0.95': 80.0}

## vital_signs_diastolicBp_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 1412 (28.07%)
- min/max: 40.0 / 110.0
- mean/std: 68.7978 / 6.152571903648371
- quantiles: {'0.05': 60.0, '0.25': 64.6282894736842, '0.5': 68.5, '0.75': 72.39047619047619, '0.95': 80.0}

## vital_signs_heartRate_value_stddev

- dtype: `Float64` (numeric)
- nulls: 1415 (28.13%)
- min/max: 0.0 / 40.0
- mean/std: 8.7998 / 5.200780823672562
- quantiles: {'0.05': 0.4714045207910317, '0.25': 5.770114189709903, '0.5': 8.09741898313391, '0.75': 11.04181283611476, '0.95': 18.853525021889073}

## vital_signs_heartRate_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1415 (28.13%)
- min/max: 15.0 / 170.0
- mean/std: 76.1488 / 15.73335872169592
- quantiles: {'0.05': 56.0, '0.25': 66.0, '0.5': 74.0, '0.75': 83.0, '0.95': 105.0}

## vital_signs_heartRate_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1415 (28.13%)
- min/max: 15.0 / 200.0
- mean/std: 92.2530 / 18.473976811608637
- quantiles: {'0.05': 70.0, '0.25': 80.0, '0.5': 89.0, '0.75': 100.0, '0.95': 130.0}

## vital_signs_heartRate_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1415 (28.13%)
- min/max: 0.0 / 140.0
- mean/std: 60.0827 / 11.460529527523983
- quantiles: {'0.05': 45.0, '0.25': 55.0, '0.5': 60.0, '0.75': 66.0, '0.95': 76.0}

## vital_signs_heartRate_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1415 (28.13%)
- min/max: 11.0 / 150.0
- mean/std: 74.3415 / 12.620355038054669
- quantiles: {'0.05': 58.0, '0.25': 66.0, '0.5': 73.0, '0.75': 80.0, '0.95': 97.0}

## vital_signs_heartRate_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 1415 (28.13%)
- min/max: 15.0 / 140.0
- mean/std: 74.5938 / 9.806151389380682
- quantiles: {'0.05': 61.30219780219779, '0.25': 68.27562862669245, '0.5': 73.5, '0.75': 79.66666666666667, '0.95': 92.06911764705883}

## vital_signs_systolicBp_value_stddev

- dtype: `Float64` (numeric)
- nulls: 1687 (33.53%)
- min/max: 0.0 / 35.0
- mean/std: 10.8852 / 5.051752076925559
- quantiles: {'0.05': 0.0, '0.25': 8.291561975888499, '0.5': 10.897247358851684, '0.75': 13.764202806786187, '0.95': 19.0746276389217}

## vital_signs_systolicBp_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1687 (33.53%)
- min/max: 45.0 / 230.0
- mean/std: 122.5340 / 18.402668920851745
- quantiles: {'0.05': 95.0, '0.25': 110.0, '0.5': 120.0, '0.75': 130.0, '0.95': 154.0}

## vital_signs_systolicBp_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1687 (33.53%)
- min/max: 95.0 / 230.0
- mean/std: 140.0090 / 19.332579966953585
- quantiles: {'0.05': 110.0, '0.25': 130.0, '0.5': 140.0, '0.75': 150.0, '0.95': 180.0}

## vital_signs_systolicBp_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1687 (33.53%)
- min/max: 40.0 / 230.0
- mean/std: 101.1787 / 14.682004286124188
- quantiles: {'0.05': 80.0, '0.25': 90.0, '0.5': 100.0, '0.75': 110.0, '0.95': 125.0}

## vital_signs_systolicBp_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1687 (33.53%)
- min/max: 40.0 / 230.0
- mean/std: 118.1911 / 16.069685356214702
- quantiles: {'0.05': 95.0, '0.25': 110.0, '0.5': 120.0, '0.75': 130.0, '0.95': 140.0}

## vital_signs_systolicBp_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 1687 (33.53%)
- min/max: 72.0 / 230.0
- mean/std: 119.2335 / 12.520969728886499
- quantiles: {'0.05': 100.00671641791044, '0.25': 110.28373015873015, '0.5': 118.5934065934066, '0.75': 126.72088068181819, '0.95': 140.0}

## vital_signs_oxygenSaturation_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 1542 (30.65%)
- min/max: 66.0 / 100.0
- mean/std: 95.9814 / 1.684336097303494
- quantiles: {'0.05': 93.51069402007856, '0.25': 95.37931034482759, '0.5': 96.25, '0.75': 96.92857142857143, '0.95': 97.8888888888889}

## vital_signs_oxygenSaturation_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1542 (30.65%)
- min/max: 35.0 / 100.0
- mean/std: 96.0134 / 3.604380017533291
- quantiles: {'0.05': 92.0, '0.25': 95.0, '0.5': 96.0, '0.75': 98.0, '0.95': 99.0}

## vital_signs_oxygenSaturation_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1542 (30.65%)
- min/max: 0.0 / 100.0
- mean/std: 91.2396 / 11.09191898728494
- quantiles: {'0.05': 84.0, '0.25': 92.0, '0.5': 94.0, '0.75': 95.0, '0.95': 97.0}

## vital_signs_oxygenSaturation_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1542 (30.65%)
- min/max: 77.0 / 100.0
- mean/std: 98.2929 / 1.3674873096840343
- quantiles: {'0.05': 96.0, '0.25': 98.0, '0.5': 98.0, '0.75': 99.0, '0.95': 100.0}

## vital_signs_oxygenSaturation_value_stddev

- dtype: `Float64` (numeric)
- nulls: 1542 (30.65%)
- min/max: 0.0 / 43.0
- mean/std: 1.8379 / 2.505834003902135
- quantiles: {'0.05': 0.0, '0.25': 1.019803902718557, '0.5': 1.4317821063276353, '0.75': 1.8929694486000919, '0.95': 3.4748988876327207}

## vital_signs_oxygenSaturation_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1542 (30.65%)
- min/max: 3.0 / 100.0
- mean/std: 95.9048 / 4.367783305453988
- quantiles: {'0.05': 92.0, '0.25': 95.0, '0.5': 96.0, '0.75': 98.0, '0.95': 99.0}

## lab_results_hemoglobin_value_stddev

- dtype: `Float64` (numeric)
- nulls: 266 (5.29%)
- min/max: 0.0 / 35.0
- mean/std: 5.7139 / 5.191825737136898
- quantiles: {'0.05': 0.0, '0.25': 1.833030277982335, '0.5': 4.758150901348127, '0.75': 8.013876853447538, '0.95': 16.467797749999466}

## lab_results_hemoglobin_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 266 (5.29%)
- min/max: 54.0 / 220.0
- mean/std: 120.2319 / 20.91749356638611
- quantiles: {'0.05': 89.5, '0.25': 103.13043478260869, '0.5': 119.1764705882353, '0.75': 135.20000000000002, '0.95': 155.0}

## lab_results_hemoglobin_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 266 (5.29%)
- min/max: 54.0 / 210.0
- mean/std: 118.5102 / 21.909040056041718
- quantiles: {'0.05': 87.0, '0.25': 101.0, '0.5': 117.0, '0.75': 134.0, '0.95': 155.0}

## lab_results_hemoglobin_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 266 (5.29%)
- min/max: 48.0 / 210.0
- mean/std: 112.5024 / 23.906176907695905
- quantiles: {'0.05': 76.0, '0.25': 93.0, '0.5': 112.0, '0.75': 130.0, '0.95': 151.0}

## lab_results_hemoglobin_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 266 (5.29%)
- min/max: 54.0 / 230.0
- mean/std: 129.3975 / 20.48991258121877
- quantiles: {'0.05': 97.0, '0.25': 114.0, '0.5': 130.0, '0.75': 144.0, '0.95': 162.0}

## lab_results_hemoglobin_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 266 (5.29%)
- min/max: 54.0 / 230.0
- mean/std: 124.2923 / 21.634208897676444
- quantiles: {'0.05': 88.0, '0.25': 108.0, '0.5': 126.0, '0.75': 140.0, '0.95': 158.0}

## lab_results_ferritin_value_stddev

- dtype: `Float64` (numeric)
- nulls: 3604 (71.64%)
- min/max: 0.0 / 800.0
- mean/std: 10.2141 / 52.64411025217608
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 45.85000000000002}

## lab_results_ferritin_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 3604 (71.64%)
- min/max: 5.0 / 2000.0
- mean/std: 232.0332 / 286.99721079616944
- quantiles: {'0.05': 15.0, '0.25': 50.5, '0.5': 133.0, '0.75': 295.24999999999994, '0.95': 803.7}

## lab_results_ferritin_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 3604 (71.64%)
- min/max: 5.0 / 2000.0
- mean/std: 234.6636 / 292.07026011903446
- quantiles: {'0.05': 15.0, '0.25': 50.5, '0.5': 134.0, '0.75': 298.5, '0.95': 801.8000000000002}

## lab_results_ferritin_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 3604 (71.64%)
- min/max: 5.0 / 2000.0
- mean/std: 221.2186 / 277.5606977628686
- quantiles: {'0.05': 15.0, '0.25': 47.0, '0.5': 128.0, '0.75': 269.0, '0.95': 778.8000000000002}

## lab_results_ferritin_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 3604 (71.64%)
- min/max: 5.0 / 2000.0
- mean/std: 242.8984 / 306.92785775188366
- quantiles: {'0.05': 15.0, '0.25': 51.0, '0.5': 135.0, '0.75': 308.5, '0.95': 832.4000000000001}

## lab_results_ferritin_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 3604 (71.64%)
- min/max: 5.0 / 2000.0
- mean/std: 227.1696 / 286.36230222339265
- quantiles: {'0.05': 15.0, '0.25': 48.0, '0.5': 130.0, '0.75': 278.0, '0.95': 802.7}

## lab_results_tfs_value_stddev

- dtype: `Float64` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_tfs_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_tfs_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_tfs_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_tfs_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_tfs_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_ntProBnp_value_stddev

- dtype: `Float64` (numeric)
- nulls: 2251 (44.74%)
- min/max: 0.0 / 17000.0
- mean/std: 1022.9745 / 2189.7321598348767
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 971.7403330989464, '0.95': 5666.182861762728}

## lab_results_ntProBnp_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 2251 (44.74%)
- min/max: 21.0 / 50000.0
- mean/std: 6350.1633 / 8460.001761068583
- quantiles: {'0.05': 183.80000000000007, '0.25': 1118.75, '0.5': 3088.5, '0.75': 7673.0, '0.95': 26377.779999999897}

## lab_results_ntProBnp_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2251 (44.74%)
- min/max: 20.0 / 50000.0
- mean/std: 5947.8396 / 8618.647742113837
- quantiles: {'0.05': 177.0, '0.25': 950.5, '0.5': 2584.0, '0.75': 6656.0, '0.95': 27276.19999999996}

## lab_results_ntProBnp_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2251 (44.74%)
- min/max: 18.0 / 50000.0
- mean/std: 5251.2342 / 7703.618752436469
- quantiles: {'0.05': 171.95000000000005, '0.25': 868.0, '0.5': 2372.0, '0.75': 5853.500000000001, '0.95': 22722.149999999998}

## lab_results_ntProBnp_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2251 (44.74%)
- min/max: 27.0 / 50000.0
- mean/std: 7705.9270 / 10104.557082444402
- quantiles: {'0.05': 185.90000000000003, '0.25': 1225.5000000000002, '0.5': 3650.0, '0.75': 9475.5, '0.95': 32590.09999999999}

## lab_results_ntProBnp_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2251 (44.74%)
- min/max: 27.0 / 50000.0
- mean/std: 6833.8773 / 9107.996808185899
- quantiles: {'0.05': 177.95000000000005, '0.25': 1103.5, '0.5': 3313.4999999999995, '0.75': 8301.0, '0.95': 29331.749999999996}

## lab_results_bnp_value_stddev

- dtype: `Float64` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_bnp_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_bnp_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_bnp_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_bnp_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_bnp_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_crpNonHs_value_stddev

- dtype: `Float64` (numeric)
- nulls: 2320 (46.11%)
- min/max: 0.0 / 160.0
- mean/std: 14.5377 / 22.072419969648067
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 3.0999999999999996, '0.75': 21.432788545595685, '0.95': 62.58885697875924}

## lab_results_crpNonHs_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 2320 (46.11%)
- min/max: 0.5 / 350.0
- mean/std: 41.0637 / 45.55706298273117
- quantiles: {'0.05': 1.1, '0.25': 7.788888888888889, '0.5': 23.825, '0.75': 61.2, '0.95': 130.56666666666666}

## lab_results_crpNonHs_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2320 (46.11%)
- min/max: 0.5 / 460.0
- mean/std: 33.2922 / 45.53232931886581
- quantiles: {'0.05': 0.9, '0.25': 5.65, '0.5': 16.3, '0.75': 40.5, '0.95': 129.15}

## lab_results_crpNonHs_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2320 (46.11%)
- min/max: 0.28 / 350.0
- mean/std: 24.5574 / 35.480488887148645
- quantiles: {'0.05': 0.8, '0.25': 4.3, '0.5': 12.0, '0.75': 28.2, '0.95': 97.9}

## lab_results_crpNonHs_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2320 (46.11%)
- min/max: 0.5 / 480.0
- mean/std: 63.7779 / 73.20751325565077
- quantiles: {'0.05': 1.1, '0.25': 8.9, '0.5': 32.9, '0.75': 100.1, '0.95': 212.9}

## lab_results_crpNonHs_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2320 (46.11%)
- min/max: 0.28 / 480.0
- mean/std: 45.7227 / 58.40685678595838
- quantiles: {'0.05': 1.0, '0.25': 6.6, '0.5': 20.9, '0.75': 63.2, '0.95': 166.5}

## lab_results_crpHs_value_stddev

- dtype: `Float64` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_crpHs_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_crpHs_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_crpHs_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_crpHs_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_crpHs_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_tropIHs_value_stddev

- dtype: `Float64` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_tropIHs_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_tropIHs_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_tropIHs_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_tropIHs_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_tropIHs_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_tropInHs_value_stddev

- dtype: `Float64` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_tropInHs_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_tropInHs_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_tropInHs_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_tropInHs_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_tropInHs_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_tropTHs_value_stddev

- dtype: `Float64` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_tropTHs_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_tropTHs_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_tropTHs_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_tropTHs_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_tropTHs_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_tropTnHs_value_stddev

- dtype: `Float64` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_tropTnHs_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_tropTnHs_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_tropTnHs_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_tropTnHs_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_tropTnHs_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_triGly_value_stddev

- dtype: `Float64` (numeric)
- nulls: 1631 (32.42%)
- min/max: 0.0 / 1.6
- mean/std: 0.0615 / 0.13416534062826643
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.06214999999999998, '0.95': 0.34464999999999996}

## lab_results_triGly_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 1631 (32.42%)
- min/max: 0.27 / 11.0
- mean/std: 1.2799 / 0.5863062765906244
- quantiles: {'0.05': 0.6441, '0.25': 0.9115333333333333, '0.5': 1.15825, '0.75': 1.5142, '0.95': 2.2374}

## lab_results_triGly_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1631 (32.42%)
- min/max: 0.15 / 11.0
- mean/std: 1.2913 / 0.6021476933348985
- quantiles: {'0.05': 0.6328, '0.25': 0.904, '0.5': 1.1752, '0.75': 1.5368, '0.95': 2.2713}

## lab_results_triGly_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1631 (32.42%)
- min/max: 0.12 / 11.0
- mean/std: 1.2106 / 0.5747251838317499
- quantiles: {'0.05': 0.5989, '0.25': 0.8475, '0.5': 1.0961, '0.75': 1.4351, '0.95': 2.147564999999997}

## lab_results_triGly_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1631 (32.42%)
- min/max: 0.27 / 11.0
- mean/std: 1.3550 / 0.6506257537078329
- quantiles: {'0.05': 0.6554, '0.25': 0.9379, '0.5': 1.2204, '0.75': 1.6046, '0.95': 2.4295}

## lab_results_triGly_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1631 (32.42%)
- min/max: 0.15 / 11.0
- mean/std: 1.2706 / 0.6006583738625619
- quantiles: {'0.05': 0.6215, '0.25': 0.889875, '0.5': 1.1413, '0.75': 1.5029, '0.95': 2.2713}

## lab_results_cholTot_value_stddev

- dtype: `Float64` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_cholTot_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_cholTot_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_cholTot_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_cholTot_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_cholTot_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_hdl_value_stddev

- dtype: `Float64` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_hdl_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_hdl_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_hdl_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_hdl_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_hdl_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_creatUS_value_stddev

- dtype: `Float64` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_creatUS_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_creatUS_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_creatUS_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_creatUS_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_creatUS_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_albuminUS_value_stddev

- dtype: `Float64` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_albuminUS_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_albuminUS_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_albuminUS_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_albuminUS_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_albuminUS_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_bun_value_stddev

- dtype: `Float64` (numeric)
- nulls: 185 (3.68%)
- min/max: 0.0 / 13.0
- mean/std: 1.5026 / 1.5826003514765035
- quantiles: {'0.05': 0.0, '0.25': 0.4998358783702134, '0.5': 1.0782279058181934, '0.75': 1.9686190406944277, '0.95': 4.520094997668238}

## lab_results_bun_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 185 (3.68%)
- min/max: 2.1 / 67.0
- mean/std: 9.4593 / 5.795512549097589
- quantiles: {'0.05': 4.166166666666666, '0.25': 5.89215, '0.5': 7.737166666666666, '0.75': 10.843192708333335, '0.95': 21.184957500000003}

## lab_results_bun_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 185 (3.68%)
- min/max: 1.7 / 61.0
- mean/std: 9.1861 / 5.926412199651713
- quantiles: {'0.05': 3.571, '0.25': 5.7136, '0.5': 7.4991, '0.75': 10.713, '0.95': 20.7118}

## lab_results_bun_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 185 (3.68%)
- min/max: 1.7 / 59.0
- mean/std: 7.4424 / 4.8414099918120055
- quantiles: {'0.05': 2.8568, '0.25': 4.6423, '0.5': 6.0707, '0.75': 8.5704, '0.95': 16.4266}

## lab_results_bun_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 185 (3.68%)
- min/max: 2.4 / 73.0
- mean/std: 11.8495 / 7.578141446714282
- quantiles: {'0.05': 4.6423, '0.25': 6.7849, '0.5': 9.6417, '0.75': 13.9269, '0.95': 27.8538}

## lab_results_bun_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 185 (3.68%)
- min/max: 1.7 / 59.0
- mean/std: 9.6788 / 6.398113526440716
- quantiles: {'0.05': 3.9281, '0.25': 5.7136, '0.5': 7.8562, '0.75': 11.0701, '0.95': 22.4973}

## lab_results_acr_value_stddev

- dtype: `Float64` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_acr_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_acr_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_acr_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_acr_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_acr_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_ldl_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_ldl_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_ldl_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_ldl_value_stddev

- dtype: `Float64` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_ldl_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_ldl_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_potassium_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 170 (3.38%)
- min/max: 2.1 / 9.7
- mean/std: 4.3952 / 0.6285033869974455
- quantiles: {'0.05': 3.6, '0.25': 4.0, '0.5': 4.3, '0.75': 4.7, '0.95': 5.5}

## lab_results_potassium_value_stddev

- dtype: `Float64` (numeric)
- nulls: 170 (3.38%)
- min/max: 0.0 / 2.5
- mean/std: 0.2868 / 0.21806242684138458
- quantiles: {'0.05': 0.0, '0.25': 0.13266499161421585, '0.5': 0.27519689920733703, '0.75': 0.4075141921157211, '0.95': 0.6442049363362565}

## lab_results_potassium_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 170 (3.38%)
- min/max: 2.0 / 7.1
- mean/std: 3.9290 / 0.401466763553173
- quantiles: {'0.05': 3.3249999999999997, '0.25': 3.6750000000000003, '0.5': 3.9000000000000004, '0.75': 4.15, '0.95': 4.6000000000000005}

## lab_results_potassium_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 170 (3.38%)
- min/max: 2.1 / 9.4
- mean/std: 3.9181 / 0.5745625083642529
- quantiles: {'0.05': 3.0, '0.25': 3.6, '0.5': 3.9, '0.75': 4.2, '0.95': 4.9}

## lab_results_potassium_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 170 (3.38%)
- min/max: 1.6 / 6.5
- mean/std: 3.5409 / 0.4916855754250776
- quantiles: {'0.05': 2.8, '0.25': 3.2, '0.5': 3.5, '0.75': 3.8, '0.95': 4.4}

## lab_results_potassium_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 170 (3.38%)
- min/max: 2.0 / 7.2
- mean/std: 3.9452 / 0.4816728075489031
- quantiles: {'0.05': 3.2, '0.25': 3.6, '0.5': 3.9, '0.75': 4.2, '0.95': 4.8}

## lab_results_sodium_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 171 (3.40%)
- min/max: 120.0 / 170.0
- mean/std: 139.7399 / 3.56607162550816
- quantiles: {'0.05': 133.0, '0.25': 138.0, '0.5': 140.0, '0.75': 142.0, '0.95': 145.0}

## lab_results_sodium_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 171 (3.40%)
- min/max: 120.0 / 180.0
- mean/std: 142.2455 / 3.6539482492920388
- quantiles: {'0.05': 137.0, '0.25': 140.0, '0.5': 142.0, '0.75': 144.0, '0.95': 148.0}

## lab_results_sodium_value_stddev

- dtype: `Float64` (numeric)
- nulls: 171 (3.40%)
- min/max: 0.0 / 17.0
- mean/std: 1.6585 / 1.3031193275051431
- quantiles: {'0.05': 0.0, '0.25': 0.816496580927726, '0.5': 1.5000000000000038, '0.75': 2.2587697572631322, '0.95': 3.838272323870407}

## lab_results_sodium_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 171 (3.40%)
- min/max: 100.0 / 160.0
- mean/std: 137.3257 / 4.1764287502939546
- quantiles: {'0.05': 130.0, '0.25': 135.0, '0.5': 138.0, '0.75': 140.0, '0.95': 143.0}

## lab_results_sodium_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 171 (3.40%)
- min/max: 120.0 / 170.0
- mean/std: 139.8024 / 3.161795540077995
- quantiles: {'0.05': 134.5, '0.25': 138.0, '0.5': 140.0, '0.75': 141.83333333333334, '0.95': 144.5}

## lab_results_sodium_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 171 (3.40%)
- min/max: 110.0 / 170.0
- mean/std: 140.0967 / 3.8813581122401386
- quantiles: {'0.05': 133.0, '0.25': 138.0, '0.5': 140.0, '0.75': 142.0, '0.95': 146.0}

## lab_results_albuminBS_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 972 (19.32%)
- min/max: 8.0 / 52.0
- mean/std: 32.0776 / 6.54443345499455
- quantiles: {'0.05': 21.0, '0.25': 27.0, '0.5': 32.0, '0.75': 37.0, '0.95': 42.0}

## lab_results_albuminBS_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 972 (19.32%)
- min/max: 8.0 / 52.0
- mean/std: 32.9140 / 6.039401438022128
- quantiles: {'0.05': 23.0, '0.25': 28.0, '0.5': 33.0, '0.75': 37.0, '0.95': 43.0}

## lab_results_albuminBS_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 972 (19.32%)
- min/max: 8.0 / 52.0
- mean/std: 34.3865 / 6.00772677977561
- quantiles: {'0.05': 24.0, '0.25': 30.0, '0.5': 35.0, '0.75': 39.0, '0.95': 43.0}

## lab_results_albuminBS_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 972 (19.32%)
- min/max: 8.0 / 52.0
- mean/std: 35.0394 / 5.588344490267423
- quantiles: {'0.05': 26.0, '0.25': 31.0, '0.5': 35.0, '0.75': 39.0, '0.95': 44.0}

## lab_results_albuminBS_value_stddev

- dtype: `Float64` (numeric)
- nulls: 972 (19.32%)
- min/max: 0.0 / 12.0
- mean/std: 1.1433 / 1.5596777918301497
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.5, '0.75': 1.8856180831641267, '0.95': 4.5}

## lab_results_albuminBS_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 972 (19.32%)
- min/max: 8.0 / 52.0
- mean/std: 33.3736 / 5.755136904054157
- quantiles: {'0.05': 24.0, '0.25': 29.2, '0.5': 33.4, '0.75': 37.666666666666664, '0.95': 43.0}

## lab_results_hba1c%_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 3538 (70.32%)
- min/max: 3.2 / 14.0
- mean/std: 6.6435 / 1.3862669332521245
- quantiles: {'0.05': 5.2, '0.25': 5.8, '0.5': 6.3, '0.75': 7.1, '0.95': 9.5}

## lab_results_hba1c%_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 3538 (70.32%)
- min/max: 3.2 / 14.0
- mean/std: 6.6370 / 1.3721869534164395
- quantiles: {'0.05': 5.2, '0.25': 5.8, '0.5': 6.3, '0.75': 7.1000000000000005, '0.95': 9.469999999999994}

## lab_results_hba1c%_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 3538 (70.32%)
- min/max: 3.2 / 14.0
- mean/std: 6.6296 / 1.3634031961842643
- quantiles: {'0.05': 5.2, '0.25': 5.8, '0.5': 6.3, '0.75': 7.1, '0.95': 9.4}

## lab_results_hba1c%_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 3538 (70.32%)
- min/max: 3.2 / 14.0
- mean/std: 6.6212 / 1.3626767864625573
- quantiles: {'0.05': 5.2, '0.25': 5.8, '0.5': 6.3, '0.75': 7.1, '0.95': 9.4}

## lab_results_hba1c%_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 3538 (70.32%)
- min/max: 3.2 / 14.0
- mean/std: 6.6523 / 1.3871114236932098
- quantiles: {'0.05': 5.2, '0.25': 5.8, '0.5': 6.3, '0.75': 7.1, '0.95': 9.5}

## lab_results_hba1c%_value_stddev

- dtype: `Float64` (numeric)
- nulls: 3538 (70.32%)
- min/max: 0.0 / 2.3
- mean/std: 0.0149 / 0.08338858227725127
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 0.10000000000000003}

## lab_results_hba1c_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_hba1c_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_hba1c_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_hba1c_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_hba1c_value_stddev

- dtype: `Float64` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_hba1c_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## lab_results_validSerumCreatinine_value_stddev

- dtype: `Float64` (numeric)
- nulls: 360 (7.16%)
- min/max: 0.0 / 6.7
- mean/std: 0.9106 / 0.8467606780898739
- quantiles: {'0.05': 0.0, '0.25': 0.31341298998071393, '0.5': 0.7210019070709869, '0.75': 1.2588766420296311, '0.95': 2.6386561506026123}

## lab_results_validSerumCreatinine_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 360 (7.16%)
- min/max: 1.0 / 23.0
- mean/std: 11.0943 / 4.06292135459385
- quantiles: {'0.05': 6.1, '0.25': 8.2, '0.5': 10.1, '0.75': 13.1, '0.95': 20.0}

## lab_results_validSerumCreatinine_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 360 (7.16%)
- min/max: 1.0 / 23.0
- mean/std: 11.0088 / 3.751449272275622
- quantiles: {'0.05': 6.2625, '0.25': 8.375, '0.5': 10.233333333333334, '0.75': 12.957142857142857, '0.95': 18.836666666666666}

## lab_results_validSerumCreatinine_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 360 (7.16%)
- min/max: 1.0 / 23.0
- mean/std: 12.4914 / 4.480824574882683
- quantiles: {'0.05': 6.8, '0.25': 9.2, '0.5': 11.4, '0.75': 15.2, '0.95': 21.6}

## lab_results_validSerumCreatinine_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 360 (7.16%)
- min/max: 1.0 / 23.0
- mean/std: 9.8026 / 3.5004459064619717
- quantiles: {'0.05': 5.4, '0.25': 7.4, '0.5': 9.2, '0.75': 11.4, '0.95': 17.15}

## lab_results_valideGFR_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 891 (17.71%)
- min/max: 4.0 / 130.0
- mean/std: 65.0496 / 25.049513039078466
- quantiles: {'0.05': 21.099545454545456, '0.25': 46.77840909090909, '0.5': 67.0, '0.75': 85.14411764705882, '0.95': 101.83333333333333}

## lab_results_valideGFR_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 891 (17.71%)
- min/max: 3.0 / 140.0
- mean/std: 65.0832 / 25.55039968771937
- quantiles: {'0.05': 19.0, '0.25': 47.0, '0.5': 67.0, '0.75': 86.0, '0.95': 102.0}

## lab_results_valideGFR_value_stddev

- dtype: `Float64` (numeric)
- nulls: 891 (17.71%)
- min/max: 0.0 / 38.0
- mean/std: 5.0434 / 4.679846523036985
- quantiles: {'0.05': 0.0, '0.25': 1.5, '0.5': 4.0, '0.75': 7.288213421321281, '0.95': 14.124351939300832}

## lab_results_valideGFR_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 891 (17.71%)
- min/max: 3.0 / 130.0
- mean/std: 57.4626 / 25.988877770778515
- quantiles: {'0.05': 15.0, '0.25': 37.0, '0.5': 57.0, '0.75': 78.0, '0.95': 99.0}

## lab_results_valideGFR_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 891 (17.71%)
- min/max: 4.0 / 150.0
- mean/std: 72.2998 / 25.184887377415528
- quantiles: {'0.05': 25.0, '0.25': 55.0, '0.5': 77.0, '0.75': 92.0, '0.95': 106.0}

## lab_results_valideGFR_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 891 (17.71%)
- min/max: 4.0 / 130.0
- mean/std: 64.8613 / 25.998649770076852
- quantiles: {'0.05': 19.0, '0.25': 45.0, '0.5': 67.0, '0.75': 87.0, '0.95': 102.0}

## symptoms_Ankle_swelling_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## symptoms_Ascites_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## symptoms_Breathlessness_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## symptoms_Cardiac_murmur_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## symptoms_Chest_pain_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## symptoms_Cheyne_stokes_respiration_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## symptoms_Depression_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## symptoms_Dizziness_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## symptoms_Elevated_jugular_venous_pressure_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## symptoms_Fatigue_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## symptoms_Hepatojugular_reflux_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## symptoms_Hepatomegaly_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## symptoms_Intermittent_claudication_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## symptoms_Irregular_pulse_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## symptoms_Loss_of_appetite_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## symptoms_Nocturnal_cough_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## symptoms_Oliguria_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## symptoms_Orthopnoea_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## symptoms_Palpitations_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## symptoms_Paroxysmal_nocturnal_dyspnea_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## symptoms_Peripheral_edema_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## symptoms_Pleural_effusion_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## symptoms_Pulmonary_crepitations_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## symptoms_Reduced_exercise_tolerance_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## symptoms_Syncope_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## symptoms_Tachycardia_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## symptoms_Tachypnoea_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## symptoms_Third_heart_sound_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## symptoms_Weight_gain_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## symptoms_Weight_loss_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## echocardiographs_lvef

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1220 (24.25%)
- min/max: 4.5 / 98.0
- mean/std: 48.0554 / 14.783147167995038
- quantiles: {'0.05': 23.252604611714684, '0.25': 35.666666666666664, '0.5': 49.43333307902019, '0.75': 60.5, '0.95': 68.84772618611655}

## echocardiographs_lvef_pET_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 1270 (25.24%)
- min/max: 4.5 / 98.0
- mean/std: 47.8291 / 14.863171935735432
- quantiles: {'0.05': 23.320000076293944, '0.25': 35.467675399780276, '0.5': 49.06666819254557, '0.75': 60.46666717529296, '0.95': 68.38333257039388}

## echocardiographs_lvef_pET_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1270 (25.24%)
- min/max: 4.5 / 98.0
- mean/std: 48.9471 / 14.94529218545599
- quantiles: {'0.05': 24.0, '0.25': 36.66666666666666, '0.5': 50.17624855041504, '0.75': 61.275662740071624, '0.95': 69.7400001525879}

## echocardiographs_lvef_pET_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1311 (26.06%)
- min/max: 4.5 / 98.0
- mean/std: 47.8512 / 15.276393227398884
- quantiles: {'0.05': 22.596489707628887, '0.25': 35.0999995470047, '0.5': 49.34742291768392, '0.75': 60.56666692097981, '0.95': 68.96124320983887}

## echocardiographs_lvef_pET_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1270 (25.24%)
- min/max: 4.5 / 98.0
- mean/std: 46.7313 / 15.214136955404552
- quantiles: {'0.05': 21.666666666666668, '0.25': 34.0, '0.5': 48.0, '0.75': 59.56666564941406, '0.95': 68.0}

## echocardiographs_lvef_pET_stddev

- dtype: `Float64` (numeric)
- nulls: 1270 (25.24%)
- min/max: 0.0 / 23.0
- mean/std: 1.0430 / 2.3947569079478193
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.5, '0.95': 6.310506142922096}

## echocardiographs_lvef_pET_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1379 (27.41%)
- min/max: 4.5 / 98.0
- mean/std: 47.8021 / 14.82118597089776
- quantiles: {'0.05': 23.099879789352414, '0.25': 35.5, '0.5': 49.0, '0.75': 60.24166679382324, '0.95': 68.84999968210857}

## electrocardiographs_ecg_qrs_duration_pET_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2063 (41.01%)
- min/max: 6.0 / 290.0
- mean/std: 116.0414 / 31.682341760828862
- quantiles: {'0.05': 81.0, '0.25': 93.0, '0.5': 106.0, '0.75': 134.0, '0.95': 178.0}

## electrocardiographs_ecg_qrs_duration_pET_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2052 (40.79%)
- min/max: -4.0 / 260.0
- mean/std: 111.5334 / 30.6295124642711
- quantiles: {'0.05': 78.0, '0.25': 90.0, '0.5': 102.0, '0.75': 129.5, '0.95': 172.0}

## electrocardiographs_ecg_qrs_duration_pET_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 2052 (40.79%)
- min/max: 11.0 / 270.0
- mean/std: 116.2813 / 30.458297567388225
- quantiles: {'0.05': 82.97999999999999, '0.25': 94.0, '0.5': 106.0, '0.75': 135.0, '0.95': 176.41}

## electrocardiographs_ecg_qrs_duration_pET_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2086 (41.46%)
- min/max: -4.0 / 260.0
- mean/std: 116.5002 / 31.2411957701768
- quantiles: {'0.05': 82.0, '0.25': 94.0, '0.5': 106.0, '0.75': 136.0, '0.95': 177.0}

## electrocardiographs_ecg_qrs_duration_pET_stddev

- dtype: `Float64` (numeric)
- nulls: 2052 (40.79%)
- min/max: 0.0 / 70.0
- mean/std: 3.9072 / 6.718686627025236
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 1.699673171197595, '0.75': 5.0, '0.95': 16.0}

## electrocardiographs_ecg_qrs_duration_pET_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2052 (40.79%)
- min/max: 26.0 / 290.0
- mean/std: 121.3800 / 32.90784213133278
- quantiles: {'0.05': 85.0, '0.25': 97.0, '0.5': 110.0, '0.75': 142.0, '0.95': 184.0999999999999}

## electrocardiographs_ecg_qrs_axis_pET_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2052 (40.79%)
- min/max: -89.0 / 270.0
- mean/std: 25.9114 / 72.18303400148385
- quantiles: {'0.05': -63.0, '0.25': -24.0, '0.5': 13.0, '0.75': 57.0, '0.95': 195.0}

## electrocardiographs_ecg_qrs_axis_pET_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2063 (41.01%)
- min/max: -89.0 / 270.0
- mean/std: 12.0644 / 63.67808190779619
- quantiles: {'0.05': -69.0, '0.25': -31.0, '0.5': 3.0, '0.75': 45.0, '0.95': 116.0}

## electrocardiographs_ecg_qrs_axis_pET_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 2052 (40.79%)
- min/max: -89.0 / 270.0
- mean/std: 10.4640 / 56.81595581663847
- quantiles: {'0.05': -65.7, '0.25': -28.733333333333334, '0.5': 4.25, '0.75': 40.0, '0.95': 104.0}

## electrocardiographs_ecg_qrs_axis_pET_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2086 (41.46%)
- min/max: -90.0 / 270.0
- mean/std: 8.7063 / 61.95179943353007
- quantiles: {'0.05': -69.0, '0.25': -33.0, '0.5': -2.0, '0.75': 38.0, '0.95': 116.0}

## electrocardiographs_ecg_qrs_axis_pET_stddev

- dtype: `Float64` (numeric)
- nulls: 2052 (40.79%)
- min/max: 0.0 / 180.0
- mean/std: 11.2930 / 25.587662922565343
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 2.5, '0.75': 10.0, '0.95': 56.23825782644894}

## electrocardiographs_ecg_qrs_axis_pET_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2052 (40.79%)
- min/max: -90.0 / 270.0
- mean/std: -1.4924 / 56.63901510669734
- quantiles: {'0.05': -76.0, '0.25': -40.0, '0.5': -9.0, '0.75': 27.0, '0.95': 91.0999999999999}

## electrocardiographs_ecg_qt_duration_corrected_pET_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 2052 (40.79%)
- min/max: 210.0 / 620.0
- mean/std: 428.4625 / 37.63879239219761
- quantiles: {'0.05': 377.0, '0.25': 404.0, '0.5': 424.66666666666663, '0.75': 449.0, '0.95': 495.5499999999999}

## electrocardiographs_ecg_qt_duration_corrected_pET_stddev

- dtype: `Float64` (numeric)
- nulls: 2052 (40.79%)
- min/max: 0.0 / 150.0
- mean/std: 9.7891 / 13.854147534095434
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 4.5, '0.75': 15.737481969250938, '0.95': 34.748348653926435}

## electrocardiographs_ecg_qt_duration_corrected_pET_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2052 (40.79%)
- min/max: 76.0 / 610.0
- mean/std: 415.6368 / 43.256365610038934
- quantiles: {'0.05': 355.0, '0.25': 392.0, '0.5': 414.0, '0.75': 439.5, '0.95': 488.0}

## electrocardiographs_ecg_qt_duration_corrected_pET_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2052 (40.79%)
- min/max: 210.0 / 700.0
- mean/std: 440.8711 / 42.16729490263555
- quantiles: {'0.05': 383.0, '0.25': 413.0, '0.5': 436.0, '0.75': 465.0, '0.95': 514.0999999999999}

## electrocardiographs_ecg_qt_duration_corrected_pET_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2086 (41.46%)
- min/max: 170.0 / 650.0
- mean/std: 430.8228 / 40.589790242329315
- quantiles: {'0.05': 374.2, '0.25': 405.0, '0.5': 427.0, '0.75': 454.0, '0.95': 500.7999999999997}

## electrocardiographs_ecg_qt_duration_corrected_pET_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2063 (41.01%)
- min/max: 150.0 / 700.0
- mean/std: 425.7396 / 40.8201794412612
- quantiles: {'0.05': 369.0, '0.25': 401.0, '0.5': 422.0, '0.75': 447.0, '0.95': 497.0}

## electrocardiographs_ecg_st_pET

- dtype: `Boolean` (boolean)
- nulls: 5031 (100.00%)
- unique values: 0
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 5031

## electrocardiographs_ecg_ischemia_without_st_pET

- dtype: `Boolean` (boolean)
- nulls: 5031 (100.00%)
- unique values: 0
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 5031

## electrocardiographs_ecg_type_of_rhythms_pET_first

- dtype: `List(String)` (categorical)
- nulls: 2047 (40.69%)
- unique values: 2
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `[]`: 2984
  - null (missing): 2047

## electrocardiographs_ecg_type_of_rhythms_pET_last

- dtype: `List(String)` (categorical)
- nulls: 2047 (40.69%)
- unique values: 2
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `[]`: 2984
  - null (missing): 2047

## smoking_status_smoker_last

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2738
  - `True`: 2293

## smoking_status_formerSmoker_last

- dtype: `Boolean` (boolean)
- nulls: 2738 (54.42%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 2738
  - `True`: 2293

## smoking_status_smoker_totalSmokingDuration_sum

- dtype: `Int64` (numeric)
- nulls: 2738 (54.42%)
- min/max: 0.0 / 4600.0
- mean/std: 135.2224 / 436.5996297781669
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 18.0, '0.95': 857.4000000000001}

## smoking_status_smoker_startTime_count

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 1.0
- mean/std: 0.4558 / 0.4980897424650877
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 1.0, '0.95': 1.0}

## nyha_nyha

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 4
- top values (shown only where count ≥ 5):
  - `LA28406-9`: 2191
  - `LA28405-1`: 1442
  - `LA28407-7`: 1183
  - `LA28404-4`: 215

## nyha_nyha_pET

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 4
- top values (shown only where count ≥ 5):
  - `LA28406-9`: 2487
  - `LA28405-1`: 1555
  - `LA28407-7`: 711
  - `LA28404-4`: 278

## vital_signs_systolicBpDuringEncounter_value_pET

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 0 (0.00%)
- min/max: 40.0 / 230.0
- mean/std: 118.8421 / 13.818118826304353
- quantiles: {'0.05': 100.0, '0.25': 110.0, '0.5': 120.0, '0.75': 127.04562384313843, '0.95': 140.0}

## vital_signs_bmi_value_pET

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 0 (0.00%)
- min/max: 12.0 / 50.0
- mean/std: 25.4165 / 4.700438298120964
- quantiles: {'0.05': 19.237176475628303, '0.25': 22.28, '0.5': 24.577905022875516, '0.75': 27.76, '0.95': 34.16}

## lab_results_creatBS_value_p3a_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 0 (0.00%)
- min/max: 1.0 / 170.0
- mean/std: 12.5787 / 9.13645642919575
- quantiles: {'0.05': 6.372282608695652, '0.25': 8.440000000000001, '0.5': 10.333333333333334, '0.75': 13.504545454545454, '0.95': 24.503333333333334}

## lab_results_validSerumCreatinine_value_pET

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 0 (0.00%)
- min/max: 1.0 / 23.0
- mean/std: 11.0710 / 3.8592211452896983
- quantiles: {'0.05': 6.2, '0.25': 8.399999999999999, '0.5': 10.299999999999999, '0.75': 12.9, '0.95': 19.5}

## hyperkalemia_severity_categorizedValue

- dtype: `String` (categorical)
- nulls: 386 (7.67%)
- unique values: 5
- top values (shown only where count ≥ 5):
  - `normal`: 4504
  - null (missing): 386
  - `mild`: 105
  - `moderate`: 27
  - `severe`: 9

## ckd_severity_categorizedValue

- dtype: `String` (categorical)
- nulls: 891 (17.71%)
- unique values: 7
- top values (shown only where count ≥ 5):
  - `mildly_decreased`: 1639
  - null (missing): 891
  - `normal_or_high`: 870
  - `mild_to_moderate_decrease`: 725
  - `moderate_to_severe_decrease`: 523
  - `severe_decrease`: 270
  - `kidney_failure`: 113

## med_requests_activeDuringEncounter_bb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_requests_activeDuringEncounter_ace_inhibitors_arb_use_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## conditions_heartFailure_timeFromEarliest_first

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 290.0
- mean/std: 5.1435 / 22.581121205829668
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 26.0}

## conditions_heart_failure_hf_within_18mo_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `True`: 5031

## conditions_heart_failure_occurred_prior_to_18_months_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4709
  - `True`: 322

## conditions_ap_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4827
  - `True`: 204

## conditions_af_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3923
  - `True`: 1108

## conditions_cm_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4629
  - `True`: 402

## conditions_dysl_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4092
  - `True`: 939

## conditions_hf_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 4469
  - `False`: 562

## conditions_hyp_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3722
  - `True`: 1309

## conditions_ihd_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3179
  - `True`: 1852

## conditions_mi_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3729
  - `True`: 1302

## conditions_pad_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4760
  - `True`: 271

## conditions_stroke_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4551
  - `True`: 480

## conditions_tia_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## conditions_vd_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4496
  - `True`: 535

## conditions_revasc_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## conditions_devices_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## conditions_aidshiv_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4996
  - `True`: 35

## conditions_copd_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4964
  - `True`: 67

## conditions_diabetes_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4156
  - `True`: 875

## conditions_dem_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4879
  - `True`: 152

## conditions_dep_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## conditions_dia_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## conditions_hyperthyroid_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## conditions_hypothyroid_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## conditions_ibd_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## conditions_ld_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4862
  - `True`: 169

## conditions_mc_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4980
  - `True`: 51

## conditions_osa_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 5030
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## conditions_rd_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 5030
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## conditions_ckd_chronic_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4296
  - `True`: 735

## conditions_myocarditis_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## conditions_pericardial_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 5027
  - 1 other distinct value(s) suppressed, covering 4 row(s) (count below 5 and/or ranked beyond top 20)

## conditions_substance_abuse_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## conditions_ap_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4721
  - `True`: 310

## conditions_af_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3883
  - `True`: 1148

## conditions_cm_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4757
  - `True`: 274

## conditions_dysl_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3367
  - `True`: 1664

## conditions_hf_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 3772
  - `False`: 1259

## conditions_hyp_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 2625
  - `False`: 2406

## conditions_ihd_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2733
  - `True`: 2298

## conditions_mi_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3427
  - `True`: 1604

## conditions_pad_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4711
  - `True`: 320

## conditions_stroke_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4302
  - `True`: 729

## conditions_tia_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 5029
  - 1 other distinct value(s) suppressed, covering 2 row(s) (count below 5 and/or ranked beyond top 20)

## conditions_vd_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4404
  - `True`: 627

## conditions_revasc_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## conditions_devices_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## conditions_aidshiv_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 5008
  - `True`: 23

## conditions_copd_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4852
  - `True`: 179

## conditions_diabetes_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3837
  - `True`: 1194

## conditions_dem_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4862
  - `True`: 169

## conditions_dep_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## conditions_dia_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## conditions_hyperthyroid_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## conditions_hypothyroid_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## conditions_ibd_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## conditions_ld_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4785
  - `True`: 246

## conditions_mc_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4910
  - `True`: 121

## conditions_osa_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 5030
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## conditions_rd_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 5029
  - 1 other distinct value(s) suppressed, covering 2 row(s) (count below 5 and/or ranked beyond top 20)

## conditions_ckd_chronic_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4202
  - `True`: 829

## conditions_myocarditis_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## conditions_pericardial_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 5026
  - `True`: 5

## conditions_substance_abuse_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## conditions_af_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2862
  - `True`: 2169

## conditions_ckd_chronic_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3522
  - `True`: 1509

## conditions_cm_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4371
  - `True`: 660

## conditions_copd_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4785
  - `True`: 246

## conditions_dem_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4711
  - `True`: 320

## conditions_dep_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## conditions_diabetes_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2976
  - `True`: 2055

## conditions_hypothyroid_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## conditions_hyp_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 3851
  - `False`: 1180

## conditions_ihd_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 2820
  - `False`: 2211

## conditions_mc_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4865
  - `True`: 166

## conditions_mi_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 2629
  - `False`: 2402

## conditions_pad_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4444
  - `True`: 587

## conditions_rd_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 5029
  - 1 other distinct value(s) suppressed, covering 2 row(s) (count below 5 and/or ranked beyond top 20)

## conditions_stroke_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3828
  - `True`: 1203

## conditions_vd_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4402
  - `True`: 629

## med_admins_rasi_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_admins_arni_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_admins_acei_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_admins_arb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_admins_mra_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_admins_diuretics_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_admins_diuretics_loop_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_admins_anti_coag_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_admins_anti_plat_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_admins_thrombolytic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_admins_bb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_admins_ccb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_admins_digitalis_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_admins_antiarrhytmic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_admins_inotropes_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_admins_vasodil_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_admins_platelet_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_admins_ll_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_admins_ivabradine_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_admins_potassium_binders_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_admins_insulins_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_admins_oral_antidiabetic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_admins_sglt2i_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_admins_ari_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_admins_rdoad_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_admins_rdoad_syst_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_admins_cortico_syst_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_admins_antiinfl_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_requests_rasi_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_requests_arni_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_requests_acei_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_requests_arb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_requests_mra_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_requests_diuretics_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_requests_diuretics_loop_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_requests_anti_coag_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_requests_anti_plat_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_requests_thrombolytic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_requests_bb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_requests_ccb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_requests_digitalis_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_requests_antiarrhytmic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_requests_inotropes_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_requests_vasodil_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_requests_platelet_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_requests_ll_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_requests_ivabradine_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_requests_potassium_binders_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_requests_insulins_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_requests_oral_antidiabetic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_requests_sglt2i_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_requests_ari_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_requests_rdoad_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_requests_rdoad_syst_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_requests_cortico_syst_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_requests_antiinfl_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_admins_history_rasi_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_admins_history_arni_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_admins_history_acei_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_admins_history_arb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_admins_history_mra_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_admins_history_diuretics_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_admins_history_diuretics_loop_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_admins_history_anti_coag_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_admins_history_anti_plat_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_admins_history_thrombolytic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_admins_history_bb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_admins_history_ccb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_admins_history_digitalis_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_admins_history_antiarrhytmic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_admins_history_inotropes_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_admins_history_vasodil_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_admins_history_platelet_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_admins_history_ll_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_admins_history_ivabradine_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_admins_history_potassium_binders_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_admins_history_insulins_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_admins_history_oral_antidiabetic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_admins_history_sglt2i_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_admins_history_ari_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_admins_history_rdoad_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_admins_history_rdoad_syst_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_admins_history_cortico_syst_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_admins_history_antiinfl_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_requests_history_rasi_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_requests_history_arni_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_requests_history_acei_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_requests_history_arb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_requests_history_mra_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_requests_history_diuretics_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_requests_history_diuretics_loop_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_requests_history_anti_coag_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_requests_history_anti_plat_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_requests_history_thrombolytic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_requests_history_bb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_requests_history_ccb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_requests_history_digitalis_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_requests_history_antiarrhytmic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_requests_history_inotropes_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_requests_history_vasodil_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_requests_history_platelet_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_requests_history_ll_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_requests_history_ivabradine_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_requests_history_potassium_binders_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_requests_history_insulins_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_requests_history_oral_antidiabetic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_requests_history_sglt2i_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_requests_history_ari_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_requests_history_rdoad_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_requests_history_rdoad_syst_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_requests_history_cortico_syst_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## med_requests_history_antiinfl_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 5031

## encounter_primary_reason_HF_Disease_f5a_w7d_first

- dtype: `Boolean` (boolean)
- nulls: 4920 (97.79%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 4920
  - `False`: 107
  - 1 other distinct value(s) suppressed, covering 4 row(s) (count below 5 and/or ranked beyond top 20)

## encounter_primary_reason_HF_Disease_f5a_w1mo_first

- dtype: `Boolean` (boolean)
- nulls: 4676 (92.94%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 4676
  - `False`: 347
  - `True`: 8

## encounter_primary_reason_HF_Disease_f5a_w3mo_first

- dtype: `Boolean` (boolean)
- nulls: 4261 (84.69%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 4261
  - `False`: 746
  - `True`: 24

## encounter_primary_reason_HF_Disease_f5a_w6mo_first

- dtype: `Boolean` (boolean)
- nulls: 4011 (79.73%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 4011
  - `False`: 987
  - `True`: 33

## encounter_primary_reason_HF_Disease_f5a_w1a_first

- dtype: `Boolean` (boolean)
- nulls: 3743 (74.40%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 3743
  - `False`: 1245
  - `True`: 43

## encounter_primary_reason_HF_Disease_f5a_w3a_first

- dtype: `Boolean` (boolean)
- nulls: 3412 (67.82%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 3412
  - `False`: 1567
  - `True`: 52

## encounter_primary_reason_HF_Disease_f5a_w5a_first

- dtype: `Boolean` (boolean)
- nulls: 3351 (66.61%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 3351
  - `False`: 1625
  - `True`: 55

## encounter_primary_reason_number_of_days_to_rehosp_for_heart_failure_f5a_first

- dtype: `Int32` (numeric)
- nulls: 4976 (98.91%)
- min/max: 0.0 / 1600.0
- mean/std: 276.2545 / 350.0068897830768
- quantiles: {'0.05': 6.7, '0.25': 55.5, '0.5': 136.0, '0.75': 314.5, '0.95': 1040.1999999999994}

## encounter_primary_reason_number_of_HF_rehospitalizations_5a_f5a_count

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 2.0
- mean/std: 0.0183 / 0.13837859058377863
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 0.0}

## encounter_primary_reason_CV_Disease_f5a_w7d_first

- dtype: `Boolean` (boolean)
- nulls: 4920 (97.79%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 4920
  - `True`: 59
  - `False`: 52

## encounter_primary_reason_CV_Disease_f5a_w1mo_first

- dtype: `Boolean` (boolean)
- nulls: 4676 (92.94%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 4676
  - `False`: 186
  - `True`: 169

## encounter_primary_reason_CV_Disease_f5a_w3mo_first

- dtype: `Boolean` (boolean)
- nulls: 4261 (84.69%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 4261
  - `True`: 392
  - `False`: 378

## encounter_primary_reason_CV_Disease_f5a_w6mo_first

- dtype: `Boolean` (boolean)
- nulls: 4011 (79.73%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 4011
  - `False`: 517
  - `True`: 503

## encounter_primary_reason_CV_Disease_f5a_w1a_first

- dtype: `Boolean` (boolean)
- nulls: 3743 (74.40%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 3743
  - `False`: 660
  - `True`: 628

## encounter_primary_reason_CV_Disease_f5a_w3a_first

- dtype: `Boolean` (boolean)
- nulls: 3412 (67.82%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 3412
  - `False`: 850
  - `True`: 769

## encounter_primary_reason_CV_Disease_f5a_w5a_first

- dtype: `Boolean` (boolean)
- nulls: 3351 (66.61%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 3351
  - `False`: 881
  - `True`: 799

## encounter_primary_reason_number_of_days_to_rehosp_for_CV_f5a_first

- dtype: `Int32` (numeric)
- nulls: 4232 (84.12%)
- min/max: 0.0 / 1800.0
- mean/std: 240.5282 / 326.5414095115792
- quantiles: {'0.05': 5.0, '0.25': 34.0, '0.5': 94.0, '0.75': 320.0, '0.95': 983.8999999999992}

## encounter_primary_reason_number_of_CV_rehospitalizations_5a_f5a_count

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 11.0
- mean/std: 0.4075 / 0.9291693408995324
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 2.0}

## encounter_primary_reason_non_CV_Disease_f5a_w7d_first

- dtype: `Boolean` (boolean)
- nulls: 4920 (97.79%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 4920
  - `False`: 56
  - `True`: 55

## encounter_primary_reason_non_CV_Disease_f5a_w1mo_first

- dtype: `Boolean` (boolean)
- nulls: 4676 (92.94%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 4676
  - `True`: 194
  - `False`: 161

## encounter_primary_reason_non_CV_Disease_f5a_w3mo_first

- dtype: `Boolean` (boolean)
- nulls: 4261 (84.69%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 4261
  - `True`: 386
  - `False`: 384

## encounter_primary_reason_non_CV_Disease_f5a_w6mo_first

- dtype: `Boolean` (boolean)
- nulls: 4011 (79.73%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 4011
  - `True`: 525
  - `False`: 495

## encounter_primary_reason_non_CV_Disease_f5a_w1a_first

- dtype: `Boolean` (boolean)
- nulls: 3743 (74.40%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 3743
  - `True`: 669
  - `False`: 619

## encounter_primary_reason_non_CV_Disease_f5a_w3a_first

- dtype: `Boolean` (boolean)
- nulls: 3412 (67.82%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 3412
  - `True`: 861
  - `False`: 758

## encounter_primary_reason_non_CV_Disease_f5a_w5a_first

- dtype: `Boolean` (boolean)
- nulls: 3351 (66.61%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 3351
  - `True`: 893
  - `False`: 787

## encounter_primary_reason_number_of_days_to_rehosp_for_non_CV_f5a_first

- dtype: `Int32` (numeric)
- nulls: 4138 (82.25%)
- min/max: 0.0 / 1800.0
- mean/std: 267.0336 / 335.8737801310977
- quantiles: {'0.05': 6.0, '0.25': 36.0, '0.5': 121.0, '0.75': 366.0, '0.95': 1009.3999999999999}

## encounter_primary_reason_number_of_non_CV_rehospitalizations_5a_f5a_count

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 23.0
- mean/std: 0.6190 / 1.4883611401400834
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 1.0, '0.95': 4.0}

## encounter_primary_reason_renal_complications_f5a_w7d_first

- dtype: `Boolean` (boolean)
- nulls: 4920 (97.79%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 4920
  - `False`: 108
  - 1 other distinct value(s) suppressed, covering 3 row(s) (count below 5 and/or ranked beyond top 20)

## encounter_primary_reason_renal_complications_f5a_w1mo_first

- dtype: `Boolean` (boolean)
- nulls: 4676 (92.94%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 4676
  - `False`: 346
  - `True`: 9

## encounter_primary_reason_renal_complications_f5a_w3mo_first

- dtype: `Boolean` (boolean)
- nulls: 4261 (84.69%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 4261
  - `False`: 747
  - `True`: 23

## encounter_primary_reason_renal_complications_f5a_w6mo_first

- dtype: `Boolean` (boolean)
- nulls: 4011 (79.73%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 4011
  - `False`: 988
  - `True`: 32

## encounter_primary_reason_renal_complications_f5a_w1a_first

- dtype: `Boolean` (boolean)
- nulls: 3743 (74.40%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 3743
  - `False`: 1245
  - `True`: 43

## encounter_primary_reason_renal_complications_f5a_w3a_first

- dtype: `Boolean` (boolean)
- nulls: 3412 (67.82%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 3412
  - `False`: 1563
  - `True`: 56

## encounter_primary_reason_renal_complications_f5a_w5a_first

- dtype: `Boolean` (boolean)
- nulls: 3351 (66.61%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 3351
  - `False`: 1621
  - `True`: 59

## encounter_primary_reason_number_of_days_to_rehosp_for_renal_complications_f5a_first

- dtype: `Int32` (numeric)
- nulls: 4972 (98.83%)
- min/max: 0.0 / 1600.0
- mean/std: 306.3220 / 368.699038123282
- quantiles: {'0.05': 7.800000000000001, '0.25': 59.0, '0.5': 131.0, '0.75': 382.0, '0.95': 1082.4999999999995}

## encounter_primary_reason_number_of_renal_rehospitalizations_5a_f5a_count

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 2.0
- mean/std: 0.0503 / 0.23008213681232698
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 0.0}

## cause_of_death_isCV_f5a_w7d_first

- dtype: `Boolean` (boolean)
- nulls: 5026 (99.90%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 5026
  - `False`: 5

## cause_of_death_isCV_f5a_w1mo_first

- dtype: `Boolean` (boolean)
- nulls: 5002 (99.42%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 5002
  - `False`: 29

## cause_of_death_isCV_f5a_w3mo_first

- dtype: `Boolean` (boolean)
- nulls: 4973 (98.85%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 4973
  - `False`: 58

## cause_of_death_isCV_f5a_w6mo_first

- dtype: `Boolean` (boolean)
- nulls: 4922 (97.83%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 4922
  - `False`: 109

## cause_of_death_isCV_f5a_w1a_first

- dtype: `Boolean` (boolean)
- nulls: 4894 (97.28%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 4894
  - `False`: 137

## cause_of_death_isCV_f5a_w3a_first

- dtype: `Boolean` (boolean)
- nulls: 4889 (97.18%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 4889
  - `False`: 142

## cause_of_death_isCV_f5a_w5a_first

- dtype: `Boolean` (boolean)
- nulls: 4889 (97.18%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 4889
  - `False`: 142

## cause_of_death_number_of_days_to_death_for_CV_f5a_first

- dtype: `Int32` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## cause_of_death_isRenal_f5a_w7d_first

- dtype: `Boolean` (boolean)
- nulls: 5026 (99.90%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 5026
  - `False`: 5

## cause_of_death_isRenal_f5a_w1mo_first

- dtype: `Boolean` (boolean)
- nulls: 5002 (99.42%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 5002
  - `False`: 29

## cause_of_death_isRenal_f5a_w3mo_first

- dtype: `Boolean` (boolean)
- nulls: 4973 (98.85%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 4973
  - `False`: 58

## cause_of_death_isRenal_f5a_w6mo_first

- dtype: `Boolean` (boolean)
- nulls: 4922 (97.83%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 4922
  - `False`: 109

## cause_of_death_isRenal_f5a_w1a_first

- dtype: `Boolean` (boolean)
- nulls: 4894 (97.28%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 4894
  - `False`: 137

## cause_of_death_isRenal_f5a_w3a_first

- dtype: `Boolean` (boolean)
- nulls: 4889 (97.18%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 4889
  - `False`: 142

## cause_of_death_isRenal_f5a_w5a_first

- dtype: `Boolean` (boolean)
- nulls: 4889 (97.18%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 4889
  - `False`: 142

## cause_of_death_number_of_days_to_death_for_renal_f5a_first

- dtype: `Int32` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## cause_of_death_isNonRenalAndNonCV_f5a_w7d_first

- dtype: `Boolean` (boolean)
- nulls: 5026 (99.90%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 5026
  - `False`: 5

## cause_of_death_isNonRenalAndNonCV_f5a_w1mo_first

- dtype: `Boolean` (boolean)
- nulls: 5002 (99.42%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 5002
  - `False`: 29

## cause_of_death_isNonRenalAndNonCV_f5a_w3mo_first

- dtype: `Boolean` (boolean)
- nulls: 4973 (98.85%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 4973
  - `False`: 58

## cause_of_death_isNonRenalAndNonCV_f5a_w6mo_first

- dtype: `Boolean` (boolean)
- nulls: 4922 (97.83%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 4922
  - `False`: 109

## cause_of_death_isNonRenalAndNonCV_f5a_w1a_first

- dtype: `Boolean` (boolean)
- nulls: 4894 (97.28%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 4894
  - `False`: 137

## cause_of_death_isNonRenalAndNonCV_f5a_w3a_first

- dtype: `Boolean` (boolean)
- nulls: 4889 (97.18%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 4889
  - `False`: 142

## cause_of_death_isNonRenalAndNonCV_f5a_w5a_first

- dtype: `Boolean` (boolean)
- nulls: 4889 (97.18%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 4889
  - `False`: 142

## cause_of_death_number_of_days_to_death_for_non_renal_and_non_CV_f5a_first

- dtype: `Int32` (numeric)
- nulls: 5031 (100.00%)
- All values are null.

## cause_of_death_isAllCause_f5a_w7d_first

- dtype: `Boolean` (boolean)
- nulls: 5026 (99.90%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 5026
  - `True`: 5

## cause_of_death_isAllCause_f5a_w1mo_first

- dtype: `Boolean` (boolean)
- nulls: 5002 (99.42%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 5002
  - `True`: 29

## cause_of_death_isAllCause_f5a_w3mo_first

- dtype: `Boolean` (boolean)
- nulls: 4973 (98.85%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 4973
  - `True`: 58

## cause_of_death_isAllCause_f5a_w6mo_first

- dtype: `Boolean` (boolean)
- nulls: 4922 (97.83%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 4922
  - `True`: 109

## cause_of_death_isAllCause_f5a_w1a_first

- dtype: `Boolean` (boolean)
- nulls: 4894 (97.28%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 4894
  - `True`: 137

## cause_of_death_isAllCause_f5a_w3a_first

- dtype: `Boolean` (boolean)
- nulls: 4889 (97.18%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 4889
  - `True`: 142

## cause_of_death_isAllCause_f5a_w5a_first

- dtype: `Boolean` (boolean)
- nulls: 4889 (97.18%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 4889
  - `True`: 142

## cause_of_death_number_of_days_to_death_for_all_cause_f5a_first

- dtype: `Int32` (numeric)
- nulls: 4889 (97.18%)
- min/max: 2.0 / 870.0
- mean/std: 133.5282 / 131.43294780326173
- quantiles: {'0.05': 11.05, '0.25': 47.5, '0.5': 110.0, '0.75': 175.0, '0.95': 285.9}

## eGFR_2021_ckd_epi_creatinine

- dtype: `Decimal(precision=38, scale=6)` (numeric)
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

- dtype: `Int32` (numeric)
- nulls: 1379 (27.41%)
- min/max: 8.0 / 45.0
- mean/std: 26.7429 / 6.415657308112522
- quantiles: {'0.05': 15.0, '0.25': 22.0, '0.5': 27.0, '0.75': 31.0, '0.95': 37.0}
