# Column Analysis

Total rows: 6274
Total columns: 540

## pid

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 6274
- ⚠️ high-cardinality column (6274 distinct values)
- top values (shown only where count ≥ 5):
  - (none met the display threshold)
  - 6274 other distinct value(s) suppressed, covering 6274 row(s) (count below 5 and/or ranked beyond top 20)

## encounterId

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 6274
- ⚠️ high-cardinality column (6274 distinct values)
- top values (shown only where count ≥ 5):
  - (none met the display threshold)
  - 6274 other distinct value(s) suppressed, covering 6274 row(s) (count below 5 and/or ranked beyond top 20)

## referenceTimePoint

- dtype: `Datetime(time_unit='ns', time_zone=None)` (categorical)
- nulls: 0 (0.00%)
- unique values: 1603
- ⚠️ high-cardinality column (1603 distinct values)
- top values (shown only where count ≥ 5):
  - `2023-12-22 00:00:00`: 16
  - `2024-10-25 00:00:00`: 15
  - `2022-07-28 00:00:00`: 13
  - `2022-12-23 00:00:00`: 13
  - `2023-03-03 00:00:00`: 13
  - `2024-11-01 00:00:00`: 13
  - `2024-11-15 00:00:00`: 13
  - `2020-03-13 00:00:00`: 12
  - `2022-09-16 00:00:00`: 12
  - `2022-11-29 00:00:00`: 12
  - `2023-09-06 00:00:00`: 12
  - `2024-02-02 00:00:00`: 12
  - `2024-02-28 00:00:00`: 12
  - `2024-08-14 00:00:00`: 12
  - `2024-11-04 00:00:00`: 12
  - `2020-01-10 00:00:00`: 11
  - `2022-04-06 00:00:00`: 11
  - `2022-12-19 00:00:00`: 11
  - `2023-02-03 00:00:00`: 11
  - `2023-05-29 00:00:00`: 11
  - 1583 other distinct value(s) suppressed, covering 6027 row(s) (count below 5 and/or ranked beyond top 20)

## eventTime

- dtype: `Datetime(time_unit='ns', time_zone=None)` (categorical)
- nulls: 0 (0.00%)
- unique values: 1680
- ⚠️ high-cardinality column (1680 distinct values)
- top values (shown only where count ≥ 5):
  - `2023-06-27 00:00:00`: 14
  - `2024-02-28 00:00:00`: 13
  - `2024-11-07 00:00:00`: 12
  - `2022-05-24 00:00:00`: 11
  - `2022-11-21 00:00:00`: 11
  - `2022-12-19 00:00:00`: 11
  - `2023-11-15 00:00:00`: 11
  - `2024-03-27 00:00:00`: 11
  - `2024-04-04 00:00:00`: 11
  - `2024-04-22 00:00:00`: 11
  - `2024-06-10 00:00:00`: 11
  - `2024-10-11 00:00:00`: 11
  - `2024-10-24 00:00:00`: 11
  - `2022-03-22 00:00:00`: 10
  - `2023-03-23 00:00:00`: 10
  - `2023-04-09 00:00:00`: 10
  - `2024-03-22 00:00:00`: 10
  - `2024-03-24 00:00:00`: 10
  - `2024-03-31 00:00:00`: 10
  - `2024-04-03 00:00:00`: 10
  - 1660 other distinct value(s) suppressed, covering 6055 row(s) (count below 5 and/or ranked beyond top 20)

## exitTime

- dtype: `Datetime(time_unit='ns', time_zone=None)` (categorical)
- nulls: 6274 (100.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 6274

## patient_demographics_sourceIdentifier

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 6274
- ⚠️ high-cardinality column (6274 distinct values)
- top values (shown only where count ≥ 5):
  - (none met the display threshold)
  - 6274 other distinct value(s) suppressed, covering 6274 row(s) (count below 5 and/or ranked beyond top 20)

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

## encounters_admissionDate

- dtype: `Datetime(time_unit='ns', time_zone=None)` (categorical)
- nulls: 0 (0.00%)
- unique values: 1680
- ⚠️ high-cardinality column (1680 distinct values)
- top values (shown only where count ≥ 5):
  - `2023-06-27 00:00:00`: 14
  - `2024-02-28 00:00:00`: 13
  - `2024-11-07 00:00:00`: 12
  - `2022-05-24 00:00:00`: 11
  - `2022-11-21 00:00:00`: 11
  - `2022-12-19 00:00:00`: 11
  - `2023-11-15 00:00:00`: 11
  - `2024-03-27 00:00:00`: 11
  - `2024-04-04 00:00:00`: 11
  - `2024-04-22 00:00:00`: 11
  - `2024-06-10 00:00:00`: 11
  - `2024-10-11 00:00:00`: 11
  - `2024-10-24 00:00:00`: 11
  - `2022-03-22 00:00:00`: 10
  - `2023-03-23 00:00:00`: 10
  - `2023-04-09 00:00:00`: 10
  - `2024-03-22 00:00:00`: 10
  - `2024-03-24 00:00:00`: 10
  - `2024-03-31 00:00:00`: 10
  - `2024-04-03 00:00:00`: 10
  - 1660 other distinct value(s) suppressed, covering 6055 row(s) (count below 5 and/or ranked beyond top 20)

## encounters_dischargeDate

- dtype: `Datetime(time_unit='ns', time_zone=None)` (categorical)
- nulls: 0 (0.00%)
- unique values: 1603
- ⚠️ high-cardinality column (1603 distinct values)
- top values (shown only where count ≥ 5):
  - `2023-12-22 00:00:00`: 16
  - `2024-10-25 00:00:00`: 15
  - `2022-07-28 00:00:00`: 13
  - `2022-12-23 00:00:00`: 13
  - `2023-03-03 00:00:00`: 13
  - `2024-11-01 00:00:00`: 13
  - `2024-11-15 00:00:00`: 13
  - `2020-03-13 00:00:00`: 12
  - `2022-09-16 00:00:00`: 12
  - `2022-11-29 00:00:00`: 12
  - `2023-09-06 00:00:00`: 12
  - `2024-02-02 00:00:00`: 12
  - `2024-02-28 00:00:00`: 12
  - `2024-08-14 00:00:00`: 12
  - `2024-11-04 00:00:00`: 12
  - `2020-01-10 00:00:00`: 11
  - `2022-04-06 00:00:00`: 11
  - `2022-12-19 00:00:00`: 11
  - `2023-02-03 00:00:00`: 11
  - `2023-05-29 00:00:00`: 11
  - 1583 other distinct value(s) suppressed, covering 6027 row(s) (count below 5 and/or ranked beyond top 20)

## encounters_numOfPreviousHFStays_count

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 10.0
- mean/std: 0.2246 / 0.6500279127654152
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 1.0}

## vital_signs_weight_value_p6mo_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## vital_signs_weight_value_p6mo_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## vital_signs_weight_value_p6mo_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## vital_signs_weight_value_p6mo_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## vital_signs_weight_value_p6mo_stddev

- dtype: `Float64` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## vital_signs_weight_value_p6mo_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## vital_signs_height_value_p1a_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## vital_signs_weight_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## vital_signs_height_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## vital_signs_diastolicBp_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1294 (20.62%)
- min/max: 7.0 / 200.0
- mean/std: 77.6022 / 13.938744824810634
- quantiles: {'0.05': 60.0, '0.25': 70.0, '0.5': 80.0, '0.75': 85.0, '0.95': 100.0}

## vital_signs_diastolicBp_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1294 (20.62%)
- min/max: 7.0 / 200.0
- mean/std: 76.5869 / 13.927042502820248
- quantiles: {'0.05': 58.95000000000002, '0.25': 70.0, '0.5': 80.0, '0.75': 82.0, '0.95': 100.0}

## vital_signs_diastolicBp_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1294 (20.62%)
- min/max: 7.0 / 200.0
- mean/std: 78.6080 / 13.923346796089993
- quantiles: {'0.05': 60.0, '0.25': 70.0, '0.5': 80.0, '0.75': 86.0, '0.95': 100.0}

## vital_signs_diastolicBp_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1294 (20.62%)
- min/max: 7.0 / 200.0
- mean/std: 77.5462 / 13.86245214005141
- quantiles: {'0.05': 60.0, '0.25': 70.0, '0.5': 80.0, '0.75': 85.0, '0.95': 100.0}

## vital_signs_diastolicBp_value_stddev

- dtype: `Float64` (numeric)
- nulls: 1294 (20.62%)
- min/max: 0.0 / 55.0
- mean/std: 0.9776 / 3.4540908776392323
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 7.5}

## vital_signs_diastolicBp_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 1294 (20.62%)
- min/max: 7.0 / 200.0
- mean/std: 77.5957 / 13.45252623516276
- quantiles: {'0.05': 60.0, '0.25': 70.0, '0.5': 80.0, '0.75': 85.0, '0.95': 100.0}

## vital_signs_heartRate_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2505 (39.93%)
- min/max: 4.0 / 250.0
- mean/std: 83.2014 / 25.838723597767085
- quantiles: {'0.05': 50.0, '0.25': 68.0, '0.5': 79.0, '0.75': 95.0, '0.95': 140.0}

## vital_signs_heartRate_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2505 (39.93%)
- min/max: 4.0 / 200.0
- mean/std: 78.4643 / 24.046153043294545
- quantiles: {'0.05': 50.0, '0.25': 65.0, '0.5': 75.0, '0.75': 90.0, '0.95': 130.0}

## vital_signs_heartRate_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2505 (39.93%)
- min/max: 4.0 / 250.0
- mean/std: 87.8888 / 27.543906368970767
- quantiles: {'0.05': 55.0, '0.25': 70.0, '0.5': 80.0, '0.75': 100.0, '0.95': 150.0}

## vital_signs_heartRate_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2505 (39.93%)
- min/max: 4.0 / 200.0
- mean/std: 82.9857 / 26.005752396887736
- quantiles: {'0.05': 50.0, '0.25': 67.0, '0.5': 78.0, '0.75': 94.0, '0.95': 140.0}

## vital_signs_heartRate_value_stddev

- dtype: `Float64` (numeric)
- nulls: 2505 (39.93%)
- min/max: 0.0 / 79.0
- mean/std: 4.3874 / 8.269565601699973
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 5.0, '0.95': 20.992589323656645}

## vital_signs_heartRate_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 2505 (39.93%)
- min/max: 4.0 / 200.0
- mean/std: 83.0904 / 24.172359247103813
- quantiles: {'0.05': 50.26666666666667, '0.25': 68.5, '0.5': 79.0, '0.75': 95.0, '0.95': 130.47999999999996}

## vital_signs_systolicBp_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1264 (20.15%)
- min/max: 0.11 / 280.0
- mean/std: 133.5775 / 26.72908931638307
- quantiles: {'0.05': 100.0, '0.25': 120.0, '0.5': 130.0, '0.75': 147.0, '0.95': 180.0}

## vital_signs_systolicBp_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1264 (20.15%)
- min/max: 0.11 / 270.0
- mean/std: 131.6040 / 26.551420859548177
- quantiles: {'0.05': 95.0, '0.25': 116.0, '0.5': 130.0, '0.75': 144.0, '0.95': 180.0}

## vital_signs_systolicBp_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1264 (20.15%)
- min/max: 0.11 / 280.0
- mean/std: 135.5635 / 26.914520174568707
- quantiles: {'0.05': 100.0, '0.25': 120.0, '0.5': 130.0, '0.75': 150.0, '0.95': 190.0}

## vital_signs_systolicBp_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1264 (20.15%)
- min/max: 0.11 / 270.0
- mean/std: 133.4869 / 26.658929581194883
- quantiles: {'0.05': 100.0, '0.25': 120.0, '0.5': 130.0, '0.75': 147.0, '0.95': 180.0}

## vital_signs_systolicBp_value_stddev

- dtype: `Float64` (numeric)
- nulls: 1264 (20.15%)
- min/max: 0.0 / 75.0
- mean/std: 1.9100 / 6.512129941769924
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 15.0}

## vital_signs_systolicBp_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 1264 (20.15%)
- min/max: 0.11 / 280.0
- mean/std: 133.5635 / 25.83243893246207
- quantiles: {'0.05': 100.0, '0.25': 120.0, '0.5': 130.0, '0.75': 146.0, '0.95': 180.0}

## vital_signs_oxygenSaturation_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 2964 (47.24%)
- min/max: 7.0 / 100.0
- mean/std: 95.6025 / 4.797438658284817
- quantiles: {'0.05': 88.0, '0.25': 95.0, '0.5': 97.0, '0.75': 98.0, '0.95': 99.0}

## vital_signs_oxygenSaturation_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2964 (47.24%)
- min/max: 7.0 / 100.0
- mean/std: 95.9152 / 4.651928946149448
- quantiles: {'0.05': 88.0, '0.25': 96.0, '0.5': 97.0, '0.75': 98.0, '0.95': 99.0}

## vital_signs_oxygenSaturation_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2964 (47.24%)
- min/max: 7.0 / 100.0
- mean/std: 95.5606 / 5.070182715446869
- quantiles: {'0.05': 88.0, '0.25': 95.0, '0.5': 97.0, '0.75': 98.0, '0.95': 99.0}

## vital_signs_oxygenSaturation_value_stddev

- dtype: `Float64` (numeric)
- nulls: 2964 (47.24%)
- min/max: 0.0 / 20.0
- mean/std: 0.3137 / 1.3646325402581947
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 2.0}

## vital_signs_oxygenSaturation_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2964 (47.24%)
- min/max: 7.0 / 100.0
- mean/std: 95.6335 / 4.952497763956681
- quantiles: {'0.05': 88.0, '0.25': 95.0, '0.5': 97.0, '0.75': 98.0, '0.95': 99.0}

## vital_signs_oxygenSaturation_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2964 (47.24%)
- min/max: 7.0 / 100.0
- mean/std: 95.2589 / 5.39963986741914
- quantiles: {'0.05': 86.0, '0.25': 95.0, '0.5': 97.0, '0.75': 98.0, '0.95': 99.0}

## lab_results_hemoglobin_value_stddev

- dtype: `Float64` (numeric)
- nulls: 37 (0.59%)
- min/max: 0.0 / 43.0
- mean/std: 5.6050 / 4.242873583826192
- quantiles: {'0.05': 0.0, '0.25': 2.75, '0.5': 5.252722508376013, '0.75': 7.84573486395988, '0.95': 13.000786205917143}

## lab_results_hemoglobin_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 37 (0.59%)
- min/max: 40.0 / 210.0
- mean/std: 133.4225 / 22.958792442415177
- quantiles: {'0.05': 93.0, '0.25': 119.5, '0.5': 136.0, '0.75': 149.60000000000002, '0.95': 168.0}

## lab_results_hemoglobin_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 37 (0.59%)
- min/max: 40.0 / 190.0
- mean/std: 131.3258 / 21.185748403873323
- quantiles: {'0.05': 92.0, '0.25': 117.82499999999999, '0.5': 133.32, '0.75': 146.0, '0.95': 163.0821818181818}

## lab_results_hemoglobin_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 37 (0.59%)
- min/max: 40.0 / 190.0
- mean/std: 123.5224 / 23.325245691871164
- quantiles: {'0.05': 78.5, '0.25': 110.0, '0.5': 126.30000000000001, '0.75': 140.0, '0.95': 157.0}

## lab_results_hemoglobin_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 37 (0.59%)
- min/max: 40.0 / 220.0
- mean/std: 139.8559 / 21.232548802671033
- quantiles: {'0.05': 102.69999999999999, '0.25': 126.0, '0.5': 141.0, '0.75': 154.2, '0.95': 173.0}

## lab_results_hemoglobin_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 37 (0.59%)
- min/max: 40.0 / 210.0
- mean/std: 129.3597 / 21.500808212612096
- quantiles: {'0.05': 91.0, '0.25': 115.0, '0.5': 131.0, '0.75': 144.0, '0.95': 162.12}

## lab_results_ferritin_value_stddev

- dtype: `Float64` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_ferritin_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_ferritin_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_ferritin_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_ferritin_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_ferritin_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_tfs_value_stddev

- dtype: `Float64` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_tfs_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_tfs_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_tfs_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_tfs_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_tfs_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_ntProBnp_value_stddev

- dtype: `Float64` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_ntProBnp_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_ntProBnp_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_ntProBnp_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_ntProBnp_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_ntProBnp_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_bnp_value_stddev

- dtype: `Float64` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_bnp_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_bnp_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_bnp_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_bnp_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_bnp_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_crpNonHs_value_stddev

- dtype: `Float64` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_crpNonHs_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_crpNonHs_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_crpNonHs_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_crpNonHs_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_crpNonHs_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_crpHs_value_stddev

- dtype: `Float64` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_crpHs_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_crpHs_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_crpHs_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_crpHs_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_crpHs_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_tropIHs_value_stddev

- dtype: `Float64` (numeric)
- nulls: 6078 (96.88%)
- min/max: 0.0 / 1.4
- mean/std: 0.0241 / 0.13778575725535755
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 0.054}

## lab_results_tropIHs_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6078 (96.88%)
- min/max: 0.012 / 62.0
- mean/std: 1.1632 / 4.929681345095346
- quantiles: {'0.05': 0.012, '0.25': 0.012, '0.5': 0.0415, '0.75': 0.41425, '0.95': 4.775}

## lab_results_tropIHs_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 6078 (96.88%)
- min/max: 0.012 / 62.0
- mean/std: 1.1476 / 4.920167063559258
- quantiles: {'0.05': 0.012, '0.25': 0.012, '0.5': 0.0415, '0.75': 0.41425, '0.95': 4.452500000000001}

## lab_results_tropIHs_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6078 (96.88%)
- min/max: 0.012 / 62.0
- mean/std: 1.1228 / 4.916284510658795
- quantiles: {'0.05': 0.012, '0.25': 0.012, '0.5': 0.0415, '0.75': 0.406, '0.95': 4.4525}

## lab_results_tropIHs_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6078 (96.88%)
- min/max: 0.012 / 62.0
- mean/std: 1.1750 / 4.930241507570624
- quantiles: {'0.05': 0.012, '0.25': 0.012, '0.5': 0.0415, '0.75': 0.41425, '0.95': 4.775}

## lab_results_tropIHs_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6078 (96.88%)
- min/max: 0.012 / 62.0
- mean/std: 1.1346 / 4.9169438364725755
- quantiles: {'0.05': 0.012, '0.25': 0.012, '0.5': 0.0415, '0.75': 0.406, '0.95': 4.4525}

## lab_results_tropInHs_value_stddev

- dtype: `Float64` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_tropInHs_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_tropInHs_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_tropInHs_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_tropInHs_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_tropInHs_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_tropTHs_value_stddev

- dtype: `Float64` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_tropTHs_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_tropTHs_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_tropTHs_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_tropTHs_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_tropTHs_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_tropTnHs_value_stddev

- dtype: `Float64` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_tropTnHs_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_tropTnHs_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_tropTnHs_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_tropTnHs_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_tropTnHs_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_triGly_value_stddev

- dtype: `Float64` (numeric)
- nulls: 545 (8.69%)
- min/max: 0.0 / 21.0
- mean/std: 0.1111 / 0.41588153076310624
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.1359135264301043, '0.95': 0.46594285603182395}

## lab_results_triGly_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 545 (8.69%)
- min/max: 0.16 / 44.0
- mean/std: 1.4861 / 1.1350640442468025
- quantiles: {'0.05': 0.5876, '0.25': 0.9039999999999999, '0.5': 1.2317, '0.75': 1.7402, '0.95': 3.0848548}

## lab_results_triGly_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 545 (8.69%)
- min/max: 0.16 / 22.0
- mean/std: 1.4479 / 0.9795979302484581
- quantiles: {'0.05': 0.6102, '0.25': 0.9039999999999999, '0.5': 1.2147499999999998, '0.75': 1.6723999999999999, '0.95': 2.9718999999999998}

## lab_results_triGly_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 545 (8.69%)
- min/max: 0.033 / 17.0
- mean/std: 1.3329 / 0.8946637487260516
- quantiles: {'0.05': 0.5424, '0.25': 0.8136, '0.5': 1.1187, '0.75': 1.5594, '0.95': 2.7685}

## lab_results_triGly_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 545 (8.69%)
- min/max: 0.16 / 60.0
- mean/std: 1.5832 / 1.4273204567191242
- quantiles: {'0.05': 0.6327999999999999, '0.25': 0.9604999999999999, '0.5': 1.2995, '0.75': 1.8419, '0.95': 3.2657}

## lab_results_triGly_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 545 (8.69%)
- min/max: 0.033 / 60.0
- mean/std: 1.4222 / 1.2464545841109465
- quantiles: {'0.05': 0.565, '0.25': 0.8587999999999999, '0.5': 1.1752, '0.75': 1.6498, '0.95': 2.9449607999999965}

## lab_results_cholTot_value_stddev

- dtype: `Float64` (numeric)
- nulls: 423 (6.74%)
- min/max: 0.0 / 3.2
- mean/std: 0.1517 / 0.2703014153410973
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.20720000000000027, '0.95': 0.7087129627212989}

## lab_results_cholTot_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 423 (6.74%)
- min/max: 1.0 / 13.0
- mean/std: 4.2507 / 1.3333969825970504
- quantiles: {'0.05': 2.4087, '0.25': 3.2893, '0.5': 4.0663, '0.75': 5.0505, '0.95': 6.6822}

## lab_results_cholTot_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 423 (6.74%)
- min/max: 1.1 / 13.0
- mean/std: 4.2006 / 1.287041052214548
- quantiles: {'0.05': 2.4346, '0.25': 3.2595149999999995, '0.5': 4.026155, '0.75': 4.969562499999999, '0.95': 6.5527}

## lab_results_cholTot_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 423 (6.74%)
- min/max: 0.12 / 13.0
- mean/std: 4.0320 / 1.304239956653338
- quantiles: {'0.05': 2.2015, '0.25': 3.083395, '0.5': 3.8591, '0.75': 4.7915, '0.95': 6.4491}

## lab_results_cholTot_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 423 (6.74%)
- min/max: 1.1 / 14.0
- mean/std: 4.3764 / 1.3458498794483278
- quantiles: {'0.05': 2.5382, '0.25': 3.3929, '0.5': 4.1699, '0.75': 5.18, '0.95': 6.79875}

## lab_results_cholTot_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 423 (6.74%)
- min/max: 0.12 / 14.0
- mean/std: 4.1520 / 1.324146064043447
- quantiles: {'0.05': 2.3569, '0.25': 3.2116, '0.5': 3.9627, '0.75': 4.921, '0.95': 6.5527}

## lab_results_hdl_value_stddev

- dtype: `Float64` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_hdl_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_hdl_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_hdl_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_hdl_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_hdl_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_creatUS_value_stddev

- dtype: `Float64` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_creatUS_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_creatUS_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_creatUS_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_creatUS_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_creatUS_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_albuminUS_value_stddev

- dtype: `Float64` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_albuminUS_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_albuminUS_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_albuminUS_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_albuminUS_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_albuminUS_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_bun_value_stddev

- dtype: `Float64` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_bun_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_bun_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_bun_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_bun_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_bun_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_acr_value_stddev

- dtype: `Float64` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_acr_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_acr_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_acr_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_acr_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_acr_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_ldl_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_ldl_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_ldl_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_ldl_value_stddev

- dtype: `Float64` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_ldl_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_ldl_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_potassium_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 22 (0.35%)
- min/max: 2.1 / 10.0
- mean/std: 4.3606 / 0.5886032074056682
- quantiles: {'0.05': 3.5, '0.25': 4.0, '0.5': 4.32, '0.75': 4.68, '0.95': 5.32}

## lab_results_potassium_value_stddev

- dtype: `Float64` (numeric)
- nulls: 22 (0.35%)
- min/max: 0.0 / 3.0
- mean/std: 0.2815 / 0.22610628729195884
- quantiles: {'0.05': 0.0, '0.25': 0.125, '0.5': 0.26140881065822064, '0.75': 0.39828189988417584, '0.95': 0.6651862848658411}

## lab_results_potassium_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 22 (0.35%)
- min/max: 2.1 / 14.0
- mean/std: 4.6600 / 0.6789340864580311
- quantiles: {'0.05': 3.8155, '0.25': 4.27, '0.5': 4.58, '0.75': 4.94, '0.95': 5.7}

## lab_results_potassium_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 22 (0.35%)
- min/max: 2.1 / 9.5
- mean/std: 4.1987 / 0.4440927841669646
- quantiles: {'0.05': 3.50185, '0.25': 3.9128571428571433, '0.5': 4.183333333333333, '0.75': 4.4603125, '0.95': 4.930000000000001}

## lab_results_potassium_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 22 (0.35%)
- min/max: 1.4 / 9.5
- mean/std: 3.8169 / 0.5633874804781202
- quantiles: {'0.05': 2.9, '0.25': 3.47, '0.5': 3.81, '0.75': 4.17, '0.95': 4.72}

## lab_results_potassium_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 22 (0.35%)
- min/max: 1.7 / 14.0
- mean/std: 4.0495 / 0.6086661072820427
- quantiles: {'0.05': 3.1, '0.25': 3.6975, '0.5': 4.04, '0.75': 4.4, '0.95': 4.97}

## lab_results_sodium_value_stddev

- dtype: `Float64` (numeric)
- nulls: 26 (0.41%)
- min/max: 0.0 / 24.0
- mean/std: 1.7473 / 1.5074095870706945
- quantiles: {'0.05': 0.0, '0.25': 0.7870194175947698, '0.5': 1.5499999999999972, '0.75': 2.416609194718914, '0.95': 4.383782029933015}

## lab_results_sodium_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 26 (0.41%)
- min/max: 94.0 / 180.0
- mean/std: 139.6613 / 3.9865624990886035
- quantiles: {'0.05': 133.8, '0.25': 138.0, '0.5': 140.0, '0.75': 142.0, '0.95': 145.0}

## lab_results_sodium_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 26 (0.41%)
- min/max: 84.0 / 150.0
- mean/std: 136.1772 / 4.439247352096076
- quantiles: {'0.05': 128.0, '0.25': 134.2, '0.5': 137.0, '0.75': 139.0, '0.95': 142.0}

## lab_results_sodium_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 26 (0.41%)
- min/max: 100.0 / 160.0
- mean/std: 137.7741 / 4.03114416914789
- quantiles: {'0.05': 130.8, '0.25': 136.0, '0.5': 138.0, '0.75': 140.0, '0.95': 143.0}

## lab_results_sodium_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 26 (0.41%)
- min/max: 110.0 / 180.0
- mean/std: 141.2161 / 3.9077958264345787
- quantiles: {'0.05': 136.0, '0.25': 139.0, '0.5': 141.0, '0.75': 143.0, '0.95': 147.0}

## lab_results_sodium_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 26 (0.41%)
- min/max: 110.0 / 160.0
- mean/std: 138.6990 / 3.206083186479062
- quantiles: {'0.05': 133.13116666666667, '0.25': 137.23333333333335, '0.5': 139.0, '0.75': 140.54999999999998, '0.95': 143.0}

## lab_results_albuminBS_value_stddev

- dtype: `Float64` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_albuminBS_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_albuminBS_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_albuminBS_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_albuminBS_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_albuminBS_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_hba1c%_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_hba1c%_value_stddev

- dtype: `Float64` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_hba1c%_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_hba1c%_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_hba1c%_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_hba1c%_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_hba1c_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_hba1c_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_hba1c_value_stddev

- dtype: `Float64` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_hba1c_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_hba1c_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_hba1c_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_validSerumCreatinine_value_stddev

- dtype: `Float64` (numeric)
- nulls: 15 (0.24%)
- min/max: 0.0 / 7.5
- mean/std: 1.0567 / 1.0393829597978526
- quantiles: {'0.05': 0.0, '0.25': 0.34326034972889197, '0.5': 0.7949056547792324, '0.75': 1.4334747329133322, '0.95': 3.2970184340215023}

## lab_results_validSerumCreatinine_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 15 (0.24%)
- min/max: 0.6 / 23.0
- mean/std: 10.6439 / 3.9660845693495586
- quantiles: {'0.05': 6.3, '0.25': 8.0, '0.5': 9.4, '0.75': 12.3, '0.95': 19.8}

## lab_results_validSerumCreatinine_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 15 (0.24%)
- min/max: 0.6 / 23.0
- mean/std: 9.1244 / 3.2391490927388866
- quantiles: {'0.05': 5.300000000000001, '0.25': 7.0, '0.5': 8.4, '0.75': 10.5, '0.95': 15.8}

## lab_results_validSerumCreatinine_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 15 (0.24%)
- min/max: 3.2 / 23.0
- mean/std: 12.1807 / 4.469404297527048
- quantiles: {'0.05': 6.800000000000001, '0.25': 8.8, '0.5': 10.9, '0.75': 14.7, '0.95': 21.5}

## lab_results_validSerumCreatinine_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 15 (0.24%)
- min/max: 3.2 / 23.0
- mean/std: 10.5005 / 3.505200744953102
- quantiles: {'0.05': 6.268974358974359, '0.25': 8.0, '0.5': 9.65, '0.75': 12.21168831168831, '0.95': 17.83499999999999}

## lab_results_valideGFR_value_stddev

- dtype: `Float64` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_valideGFR_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_valideGFR_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_valideGFR_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_valideGFR_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## lab_results_valideGFR_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## symptoms_Ankle_swelling_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## symptoms_Ascites_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## symptoms_Breathlessness_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## symptoms_Cardiac_murmur_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## symptoms_Chest_pain_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## symptoms_Cheyne_stokes_respiration_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## symptoms_Depression_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## symptoms_Dizziness_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## symptoms_Elevated_jugular_venous_pressure_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## symptoms_Fatigue_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## symptoms_Hepatojugular_reflux_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## symptoms_Hepatomegaly_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## symptoms_Intermittent_claudication_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## symptoms_Irregular_pulse_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## symptoms_Loss_of_appetite_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## symptoms_Nocturnal_cough_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## symptoms_Oliguria_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## symptoms_Orthopnoea_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## symptoms_Palpitations_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## symptoms_Paroxysmal_nocturnal_dyspnea_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## symptoms_Peripheral_edema_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## symptoms_Pleural_effusion_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## symptoms_Pulmonary_crepitations_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## symptoms_Reduced_exercise_tolerance_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## symptoms_Syncope_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## symptoms_Tachycardia_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## symptoms_Tachypnoea_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## symptoms_Third_heart_sound_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## symptoms_Weight_gain_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## symptoms_Weight_loss_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## echocardiographs_lvef

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 3960 (63.12%)
- min/max: 5.0 / 89.0
- mean/std: 38.4257 / 14.514058590295816
- quantiles: {'0.05': 15.0, '0.25': 25.0, '0.5': 40.0, '0.75': 50.0, '0.95': 60.0}

## echocardiographs_lvef_pET_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4055 (64.63%)
- min/max: 5.0 / 89.0
- mean/std: 38.0189 / 14.602042994888766
- quantiles: {'0.05': 15.0, '0.25': 25.0, '0.5': 40.0, '0.75': 50.0, '0.95': 59.0}

## echocardiographs_lvef_pET_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4055 (64.63%)
- min/max: 5.0 / 89.0
- mean/std: 39.3560 / 14.287129458433736
- quantiles: {'0.05': 15.0, '0.25': 29.0, '0.5': 40.0, '0.75': 52.0, '0.95': 60.0}

## echocardiographs_lvef_pET_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4055 (64.63%)
- min/max: 5.0 / 89.0
- mean/std: 38.6404 / 14.522347166164852
- quantiles: {'0.05': 15.0, '0.25': 25.0, '0.5': 40.0, '0.75': 50.0, '0.95': 60.0}

## echocardiographs_lvef_pET_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4055 (64.63%)
- min/max: 5.0 / 89.0
- mean/std: 38.7143 / 14.42515675805727
- quantiles: {'0.05': 15.0, '0.25': 26.0, '0.5': 40.0, '0.75': 50.0, '0.95': 60.0}

## echocardiographs_lvef_pET_stddev

- dtype: `Float64` (numeric)
- nulls: 4055 (64.63%)
- min/max: 0.0 / 20.0
- mean/std: 0.6459 / 2.1150335458235157
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 5.0}

## echocardiographs_lvef_pET_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 4055 (64.63%)
- min/max: 5.0 / 89.0
- mean/std: 38.6753 / 14.289411869850102
- quantiles: {'0.05': 15.0, '0.25': 26.583333333333332, '0.5': 40.0, '0.75': 50.0, '0.95': 59.0}

## electrocardiographs_ecg_qrs_duration_pET_stddev

- dtype: `Float64` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## electrocardiographs_ecg_qrs_duration_pET_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## electrocardiographs_ecg_qrs_duration_pET_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## electrocardiographs_ecg_qrs_duration_pET_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## electrocardiographs_ecg_qrs_duration_pET_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## electrocardiographs_ecg_qrs_duration_pET_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## electrocardiographs_ecg_qrs_axis_pET_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## electrocardiographs_ecg_qrs_axis_pET_stddev

- dtype: `Float64` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## electrocardiographs_ecg_qrs_axis_pET_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## electrocardiographs_ecg_qrs_axis_pET_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## electrocardiographs_ecg_qrs_axis_pET_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## electrocardiographs_ecg_qrs_axis_pET_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## electrocardiographs_ecg_qt_duration_corrected_pET_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## electrocardiographs_ecg_qt_duration_corrected_pET_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## electrocardiographs_ecg_qt_duration_corrected_pET_stddev

- dtype: `Float64` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## electrocardiographs_ecg_qt_duration_corrected_pET_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## electrocardiographs_ecg_qt_duration_corrected_pET_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## electrocardiographs_ecg_qt_duration_corrected_pET_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## electrocardiographs_ecg_st_pET

- dtype: `Boolean` (boolean)
- nulls: 6274 (100.00%)
- unique values: 0
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 6274

## electrocardiographs_ecg_ischemia_without_st_pET

- dtype: `Boolean` (boolean)
- nulls: 6274 (100.00%)
- unique values: 0
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 6274

## electrocardiographs_ecg_type_of_rhythms_pET_first

- dtype: `List(String)` (categorical)
- nulls: 6274 (100.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 6274

## electrocardiographs_ecg_type_of_rhythms_pET_last

- dtype: `List(String)` (categorical)
- nulls: 6274 (100.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 6274

## smoking_status_smoker_last

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## smoking_status_formerSmoker_last

- dtype: `Boolean` (boolean)
- nulls: 6274 (100.00%)
- unique values: 0
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 6274

## smoking_status_smoker_totalSmokingDuration_sum

- dtype: `Int64` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## smoking_status_smoker_startTime_count

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 0.0
- mean/std: 0.0000 / 0.0
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 0.0}
- ⚠️ constant column (single value)

## nyha_nyha

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 4
- top values (shown only where count ≥ 5):
  - `LA28406-9`: 2459
  - `LA28407-7`: 2123
  - `LA28405-1`: 1666
  - `LA28404-4`: 26

## nyha_nyha_pET

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 4
- top values (shown only where count ≥ 5):
  - `LA28406-9`: 2464
  - `LA28407-7`: 2106
  - `LA28405-1`: 1677
  - `LA28404-4`: 27

## vital_signs_systolicBpDuringEncounter_value_pET

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 0 (0.00%)
- min/max: 50.0 / 250.0
- mean/std: 133.3756 / 23.67274188581887
- quantiles: {'0.05': 100.0, '0.25': 120.0, '0.5': 130.0, '0.75': 143.48808827908186, '0.95': 180.0}

## vital_signs_bmi_value_pET

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 0 (0.00%)
- min/max: 18.0 / 25.0
- mean/std: 21.7532 / 1.8701470751402411
- quantiles: {'0.05': 18.8277678123233, '0.25': 20.108431914413185, '0.5': 21.755805692912602, '0.75': 23.380551958881806, '0.95': 24.66705846226328}

## lab_results_creatBS_value_p3a_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 0 (0.00%)
- min/max: 3.3 / 59.0
- mean/std: 10.9768 / 4.908453943141329
- quantiles: {'0.05': 6.282166666666666, '0.25': 8.0, '0.5': 9.700000000000001, '0.75': 12.34404761904762, '0.95': 20.281025641025618}

## lab_results_validSerumCreatinine_value_pET

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 0 (0.00%)
- min/max: 1.5 / 23.0
- mean/std: 10.2746 / 3.5583064671602864
- quantiles: {'0.05': 5.865000000000003, '0.25': 7.7, '0.5': 10.0, '0.75': 11.799999999999999, '0.95': 17.8}

## hyperkalemia_severity_categorizedValue

- dtype: `String` (categorical)
- nulls: 8 (0.13%)
- unique values: 5
- top values (shown only where count ≥ 5):
  - `normal`: 5969
  - `mild`: 227
  - `moderate`: 46
  - `severe`: 24
  - null (missing): 8

## ckd_severity_categorizedValue

- dtype: `String` (categorical)
- nulls: 6274 (100.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 6274

## med_requests_activeDuringEncounter_bb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_requests_activeDuringEncounter_ace_inhibitors_arb_use_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## conditions_heartFailure_timeFromEarliest_first

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 57.0
- mean/std: 1.2816 / 5.277918483874049
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 8.0}

## conditions_heart_failure_hf_within_18mo_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `True`: 6274

## conditions_heart_failure_occurred_prior_to_18_months_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 6119
  - `True`: 155

## conditions_ap_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 5248
  - `True`: 1026

## conditions_af_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3772
  - `True`: 2502

## conditions_cm_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4900
  - `True`: 1374

## conditions_dysl_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 3474
  - `False`: 2800

## conditions_hf_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `True`: 6274

## conditions_hyp_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 4506
  - `False`: 1768

## conditions_ihd_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 3418
  - `False`: 2856

## conditions_mi_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 5193
  - `True`: 1081

## conditions_pad_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 5632
  - `True`: 642

## conditions_stroke_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 6184
  - `True`: 90

## conditions_tia_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 6230
  - `True`: 44

## conditions_vd_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 3159
  - `False`: 3115

## conditions_revasc_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 5093
  - `True`: 1181

## conditions_devices_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 5333
  - `True`: 941

## conditions_aidshiv_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 6270
  - 1 other distinct value(s) suppressed, covering 4 row(s) (count below 5 and/or ranked beyond top 20)

## conditions_copd_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 5995
  - `True`: 279

## conditions_diabetes_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4411
  - `True`: 1863

## conditions_dem_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 6179
  - `True`: 95

## conditions_dep_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 6172
  - `True`: 102

## conditions_dia_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## conditions_hyperthyroid_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 6256
  - `True`: 18

## conditions_hypothyroid_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 6055
  - `True`: 219

## conditions_ibd_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 6267
  - `True`: 7

## conditions_ld_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 6011
  - `True`: 263

## conditions_mc_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 6084
  - `True`: 190

## conditions_osa_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 6009
  - `True`: 265

## conditions_rd_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 6244
  - `True`: 30

## conditions_ckd_chronic_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 5493
  - `True`: 781

## conditions_myocarditis_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 6265
  - `True`: 9

## conditions_pericardial_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 6160
  - `True`: 114

## conditions_substance_abuse_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 6249
  - `True`: 25

## conditions_ap_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 5141
  - `True`: 1133

## conditions_af_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3724
  - `True`: 2550

## conditions_cm_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4779
  - `True`: 1495

## conditions_dysl_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 3597
  - `False`: 2677

## conditions_hf_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `True`: 6274

## conditions_hyp_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 4623
  - `False`: 1651

## conditions_ihd_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 3524
  - `False`: 2750

## conditions_mi_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 5065
  - `True`: 1209

## conditions_pad_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 5555
  - `True`: 719

## conditions_stroke_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 6172
  - `True`: 102

## conditions_tia_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 6221
  - `True`: 53

## conditions_vd_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 3272
  - `False`: 3002

## conditions_revasc_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 5018
  - `True`: 1256

## conditions_devices_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 5275
  - `True`: 999

## conditions_aidshiv_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 6270
  - 1 other distinct value(s) suppressed, covering 4 row(s) (count below 5 and/or ranked beyond top 20)

## conditions_copd_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 5978
  - `True`: 296

## conditions_diabetes_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4371
  - `True`: 1903

## conditions_dem_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 6170
  - `True`: 104

## conditions_dep_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 6160
  - `True`: 114

## conditions_dia_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## conditions_hyperthyroid_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 6248
  - `True`: 26

## conditions_hypothyroid_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 6034
  - `True`: 240

## conditions_ibd_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 6267
  - `True`: 7

## conditions_ld_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 5991
  - `True`: 283

## conditions_mc_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 6070
  - `True`: 204

## conditions_osa_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 5982
  - `True`: 292

## conditions_rd_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 6244
  - `True`: 30

## conditions_ckd_chronic_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 5421
  - `True`: 853

## conditions_myocarditis_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 6264
  - `True`: 10

## conditions_pericardial_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 6151
  - `True`: 123

## conditions_substance_abuse_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 6240
  - `True`: 34

## conditions_af_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3724
  - `True`: 2550

## conditions_ckd_chronic_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 5421
  - `True`: 853

## conditions_cm_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4779
  - `True`: 1495

## conditions_copd_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 5978
  - `True`: 296

## conditions_dem_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 6170
  - `True`: 104

## conditions_dep_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 6160
  - `True`: 114

## conditions_diabetes_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4371
  - `True`: 1903

## conditions_hypothyroid_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 6034
  - `True`: 240

## conditions_hyp_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 4623
  - `False`: 1651

## conditions_ihd_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 3524
  - `False`: 2750

## conditions_mc_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 6070
  - `True`: 204

## conditions_mi_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 5065
  - `True`: 1209

## conditions_pad_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 5555
  - `True`: 719

## conditions_rd_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 6244
  - `True`: 30

## conditions_stroke_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 6172
  - `True`: 102

## conditions_vd_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 3273
  - `False`: 3001

## med_admins_rasi_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_admins_arni_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_admins_acei_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_admins_arb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_admins_mra_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_admins_diuretics_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_admins_diuretics_loop_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_admins_anti_coag_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_admins_anti_plat_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_admins_thrombolytic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_admins_bb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_admins_ccb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_admins_digitalis_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_admins_antiarrhytmic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_admins_inotropes_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_admins_vasodil_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_admins_platelet_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_admins_ll_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_admins_ivabradine_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_admins_potassium_binders_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_admins_insulins_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_admins_oral_antidiabetic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_admins_sglt2i_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_admins_ari_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_admins_rdoad_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_admins_rdoad_syst_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_admins_cortico_syst_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_admins_antiinfl_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_requests_rasi_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_requests_arni_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_requests_acei_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_requests_arb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_requests_mra_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_requests_diuretics_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_requests_diuretics_loop_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_requests_anti_coag_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_requests_anti_plat_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_requests_thrombolytic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_requests_bb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_requests_ccb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_requests_digitalis_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_requests_antiarrhytmic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_requests_inotropes_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_requests_vasodil_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_requests_platelet_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_requests_ll_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_requests_ivabradine_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_requests_potassium_binders_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_requests_insulins_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_requests_oral_antidiabetic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_requests_sglt2i_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_requests_ari_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_requests_rdoad_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_requests_rdoad_syst_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_requests_cortico_syst_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_requests_antiinfl_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_admins_history_rasi_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_admins_history_arni_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_admins_history_acei_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_admins_history_arb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_admins_history_mra_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_admins_history_diuretics_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_admins_history_diuretics_loop_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_admins_history_anti_coag_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_admins_history_anti_plat_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_admins_history_thrombolytic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_admins_history_bb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_admins_history_ccb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_admins_history_digitalis_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_admins_history_antiarrhytmic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_admins_history_inotropes_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_admins_history_vasodil_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_admins_history_platelet_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_admins_history_ll_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_admins_history_ivabradine_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_admins_history_potassium_binders_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_admins_history_insulins_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_admins_history_oral_antidiabetic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_admins_history_sglt2i_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_admins_history_ari_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_admins_history_rdoad_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_admins_history_rdoad_syst_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_admins_history_cortico_syst_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_admins_history_antiinfl_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_requests_history_rasi_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_requests_history_arni_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_requests_history_acei_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_requests_history_arb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_requests_history_mra_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_requests_history_diuretics_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_requests_history_diuretics_loop_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_requests_history_anti_coag_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_requests_history_anti_plat_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_requests_history_thrombolytic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_requests_history_bb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_requests_history_ccb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_requests_history_digitalis_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_requests_history_antiarrhytmic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_requests_history_inotropes_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_requests_history_vasodil_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_requests_history_platelet_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_requests_history_ll_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_requests_history_ivabradine_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_requests_history_potassium_binders_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_requests_history_insulins_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_requests_history_oral_antidiabetic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_requests_history_sglt2i_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_requests_history_ari_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_requests_history_rdoad_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_requests_history_rdoad_syst_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_requests_history_cortico_syst_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## med_requests_history_antiinfl_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 6274

## encounter_primary_reason_HF_Disease_f5a_w7d_first

- dtype: `Boolean` (boolean)
- nulls: 6259 (99.76%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 6259
  - `False`: 14
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## encounter_primary_reason_HF_Disease_f5a_w1mo_first

- dtype: `Boolean` (boolean)
- nulls: 6231 (99.31%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 6231
  - `False`: 42
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## encounter_primary_reason_HF_Disease_f5a_w3mo_first

- dtype: `Boolean` (boolean)
- nulls: 6201 (98.84%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 6201
  - `False`: 71
  - 1 other distinct value(s) suppressed, covering 2 row(s) (count below 5 and/or ranked beyond top 20)

## encounter_primary_reason_HF_Disease_f5a_w6mo_first

- dtype: `Boolean` (boolean)
- nulls: 6171 (98.36%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 6171
  - `False`: 100
  - 1 other distinct value(s) suppressed, covering 3 row(s) (count below 5 and/or ranked beyond top 20)

## encounter_primary_reason_HF_Disease_f5a_w1a_first

- dtype: `Boolean` (boolean)
- nulls: 6147 (97.98%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 6147
  - `False`: 124
  - 1 other distinct value(s) suppressed, covering 3 row(s) (count below 5 and/or ranked beyond top 20)

## encounter_primary_reason_HF_Disease_f5a_w3a_first

- dtype: `Boolean` (boolean)
- nulls: 6110 (97.39%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 6110
  - `False`: 161
  - 1 other distinct value(s) suppressed, covering 3 row(s) (count below 5 and/or ranked beyond top 20)

## encounter_primary_reason_HF_Disease_f5a_w5a_first

- dtype: `Boolean` (boolean)
- nulls: 6107 (97.34%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 6107
  - `False`: 164
  - 1 other distinct value(s) suppressed, covering 3 row(s) (count below 5 and/or ranked beyond top 20)

## encounter_primary_reason_number_of_days_to_rehosp_for_heart_failure_f5a_first

- dtype: `Int32` (numeric)
- nulls: 6271 (99.95%)
- min/max: 3.0 / 170.0
- mean/std: 69.6667 / 85.45369116271884
- quantiles: {'0.05': 6.7, '0.25': 21.5, '0.5': 40.0, '0.75': 103.0, '0.95': 153.39999999999998}

## encounter_primary_reason_number_of_HF_rehospitalizations_5a_f5a_count

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 2.0
- mean/std: 0.0065 / 0.0863117376825872
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 0.0}

## encounter_primary_reason_CV_Disease_f5a_w7d_first

- dtype: `Boolean` (boolean)
- nulls: 6259 (99.76%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 6259
  - `False`: 8
  - `True`: 7

## encounter_primary_reason_CV_Disease_f5a_w1mo_first

- dtype: `Boolean` (boolean)
- nulls: 6231 (99.31%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 6231
  - `True`: 27
  - `False`: 16

## encounter_primary_reason_CV_Disease_f5a_w3mo_first

- dtype: `Boolean` (boolean)
- nulls: 6201 (98.84%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 6201
  - `True`: 41
  - `False`: 32

## encounter_primary_reason_CV_Disease_f5a_w6mo_first

- dtype: `Boolean` (boolean)
- nulls: 6171 (98.36%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 6171
  - `True`: 60
  - `False`: 43

## encounter_primary_reason_CV_Disease_f5a_w1a_first

- dtype: `Boolean` (boolean)
- nulls: 6147 (97.98%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 6147
  - `True`: 73
  - `False`: 54

## encounter_primary_reason_CV_Disease_f5a_w3a_first

- dtype: `Boolean` (boolean)
- nulls: 6110 (97.39%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 6110
  - `True`: 94
  - `False`: 70

## encounter_primary_reason_CV_Disease_f5a_w5a_first

- dtype: `Boolean` (boolean)
- nulls: 6107 (97.34%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 6107
  - `True`: 96
  - `False`: 71

## encounter_primary_reason_number_of_days_to_rehosp_for_CV_f5a_first

- dtype: `Int32` (numeric)
- nulls: 6178 (98.47%)
- min/max: 3.0 / 1200.0
- mean/std: 225.7396 / 274.2551345694722
- quantiles: {'0.05': 5.75, '0.25': 26.75, '0.5': 137.0, '0.75': 278.25, '0.95': 777.0}

## encounter_primary_reason_number_of_CV_rehospitalizations_5a_f5a_count

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 14.0
- mean/std: 0.1302 / 0.8962264173980161
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 0.0}

## encounter_primary_reason_non_CV_Disease_f5a_w7d_first

- dtype: `Boolean` (boolean)
- nulls: 6259 (99.76%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 6259
  - `True`: 8
  - `False`: 7

## encounter_primary_reason_non_CV_Disease_f5a_w1mo_first

- dtype: `Boolean` (boolean)
- nulls: 6231 (99.31%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 6231
  - `False`: 27
  - `True`: 16

## encounter_primary_reason_non_CV_Disease_f5a_w3mo_first

- dtype: `Boolean` (boolean)
- nulls: 6201 (98.84%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 6201
  - `False`: 41
  - `True`: 32

## encounter_primary_reason_non_CV_Disease_f5a_w6mo_first

- dtype: `Boolean` (boolean)
- nulls: 6171 (98.36%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 6171
  - `False`: 60
  - `True`: 43

## encounter_primary_reason_non_CV_Disease_f5a_w1a_first

- dtype: `Boolean` (boolean)
- nulls: 6147 (97.98%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 6147
  - `False`: 73
  - `True`: 54

## encounter_primary_reason_non_CV_Disease_f5a_w3a_first

- dtype: `Boolean` (boolean)
- nulls: 6110 (97.39%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 6110
  - `False`: 94
  - `True`: 70

## encounter_primary_reason_non_CV_Disease_f5a_w5a_first

- dtype: `Boolean` (boolean)
- nulls: 6107 (97.34%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 6107
  - `False`: 96
  - `True`: 71

## encounter_primary_reason_number_of_days_to_rehosp_for_non_CV_f5a_first

- dtype: `Int32` (numeric)
- nulls: 6203 (98.87%)
- min/max: 1.0 / 1200.0
- mean/std: 206.0423 / 235.12102881098235
- quantiles: {'0.05': 4.0, '0.25': 34.5, '0.5': 106.0, '0.75': 332.5, '0.95': 677.5}

## encounter_primary_reason_number_of_non_CV_rehospitalizations_5a_f5a_count

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 16.0
- mean/std: 0.1111 / 0.8018756804699785
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 0.0}

## encounter_primary_reason_renal_complications_f5a_w7d_first

- dtype: `Boolean` (boolean)
- nulls: 6259 (99.76%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 6259
  - `False`: 15

## encounter_primary_reason_renal_complications_f5a_w1mo_first

- dtype: `Boolean` (boolean)
- nulls: 6231 (99.31%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 6231
  - `False`: 43

## encounter_primary_reason_renal_complications_f5a_w3mo_first

- dtype: `Boolean` (boolean)
- nulls: 6201 (98.84%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 6201
  - `False`: 73

## encounter_primary_reason_renal_complications_f5a_w6mo_first

- dtype: `Boolean` (boolean)
- nulls: 6171 (98.36%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 6171
  - `False`: 103

## encounter_primary_reason_renal_complications_f5a_w1a_first

- dtype: `Boolean` (boolean)
- nulls: 6147 (97.98%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 6147
  - `False`: 127

## encounter_primary_reason_renal_complications_f5a_w3a_first

- dtype: `Boolean` (boolean)
- nulls: 6110 (97.39%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 6110
  - `False`: 164

## encounter_primary_reason_renal_complications_f5a_w5a_first

- dtype: `Boolean` (boolean)
- nulls: 6107 (97.34%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 6107
  - `False`: 167

## encounter_primary_reason_number_of_days_to_rehosp_for_renal_complications_f5a_first

- dtype: `Int32` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## encounter_primary_reason_number_of_renal_rehospitalizations_5a_f5a_count

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 3.0
- mean/std: 0.0040 / 0.07018499674933434
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 0.0}

## cause_of_death_isCV_f5a_w7d_first

- dtype: `Boolean` (boolean)
- nulls: 6273 (99.98%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 6273
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## cause_of_death_isCV_f5a_w1mo_first

- dtype: `Boolean` (boolean)
- nulls: 6273 (99.98%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 6273
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## cause_of_death_isCV_f5a_w3mo_first

- dtype: `Boolean` (boolean)
- nulls: 6273 (99.98%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 6273
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## cause_of_death_isCV_f5a_w6mo_first

- dtype: `Boolean` (boolean)
- nulls: 6273 (99.98%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 6273
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## cause_of_death_isCV_f5a_w1a_first

- dtype: `Boolean` (boolean)
- nulls: 6273 (99.98%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 6273
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## cause_of_death_isCV_f5a_w3a_first

- dtype: `Boolean` (boolean)
- nulls: 6273 (99.98%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 6273
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## cause_of_death_isCV_f5a_w5a_first

- dtype: `Boolean` (boolean)
- nulls: 6273 (99.98%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 6273
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## cause_of_death_number_of_days_to_death_for_CV_f5a_first

- dtype: `Int32` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## cause_of_death_isRenal_f5a_w7d_first

- dtype: `Boolean` (boolean)
- nulls: 6273 (99.98%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 6273
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## cause_of_death_isRenal_f5a_w1mo_first

- dtype: `Boolean` (boolean)
- nulls: 6273 (99.98%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 6273
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## cause_of_death_isRenal_f5a_w3mo_first

- dtype: `Boolean` (boolean)
- nulls: 6273 (99.98%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 6273
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## cause_of_death_isRenal_f5a_w6mo_first

- dtype: `Boolean` (boolean)
- nulls: 6273 (99.98%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 6273
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## cause_of_death_isRenal_f5a_w1a_first

- dtype: `Boolean` (boolean)
- nulls: 6273 (99.98%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 6273
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## cause_of_death_isRenal_f5a_w3a_first

- dtype: `Boolean` (boolean)
- nulls: 6273 (99.98%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 6273
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## cause_of_death_isRenal_f5a_w5a_first

- dtype: `Boolean` (boolean)
- nulls: 6273 (99.98%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 6273
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## cause_of_death_number_of_days_to_death_for_renal_f5a_first

- dtype: `Int32` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## cause_of_death_isNonRenalAndNonCV_f5a_w7d_first

- dtype: `Boolean` (boolean)
- nulls: 6273 (99.98%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 6273
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## cause_of_death_isNonRenalAndNonCV_f5a_w1mo_first

- dtype: `Boolean` (boolean)
- nulls: 6273 (99.98%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 6273
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## cause_of_death_isNonRenalAndNonCV_f5a_w3mo_first

- dtype: `Boolean` (boolean)
- nulls: 6273 (99.98%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 6273
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## cause_of_death_isNonRenalAndNonCV_f5a_w6mo_first

- dtype: `Boolean` (boolean)
- nulls: 6273 (99.98%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 6273
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## cause_of_death_isNonRenalAndNonCV_f5a_w1a_first

- dtype: `Boolean` (boolean)
- nulls: 6273 (99.98%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 6273
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## cause_of_death_isNonRenalAndNonCV_f5a_w3a_first

- dtype: `Boolean` (boolean)
- nulls: 6273 (99.98%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 6273
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## cause_of_death_isNonRenalAndNonCV_f5a_w5a_first

- dtype: `Boolean` (boolean)
- nulls: 6273 (99.98%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 6273
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## cause_of_death_number_of_days_to_death_for_non_renal_and_non_CV_f5a_first

- dtype: `Int32` (numeric)
- nulls: 6274 (100.00%)
- All values are null.

## cause_of_death_isAllCause_f5a_w7d_first

- dtype: `Boolean` (boolean)
- nulls: 6273 (99.98%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 6273
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## cause_of_death_isAllCause_f5a_w1mo_first

- dtype: `Boolean` (boolean)
- nulls: 6273 (99.98%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 6273
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## cause_of_death_isAllCause_f5a_w3mo_first

- dtype: `Boolean` (boolean)
- nulls: 6273 (99.98%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 6273
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## cause_of_death_isAllCause_f5a_w6mo_first

- dtype: `Boolean` (boolean)
- nulls: 6273 (99.98%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 6273
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## cause_of_death_isAllCause_f5a_w1a_first

- dtype: `Boolean` (boolean)
- nulls: 6273 (99.98%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 6273
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## cause_of_death_isAllCause_f5a_w3a_first

- dtype: `Boolean` (boolean)
- nulls: 6273 (99.98%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 6273
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## cause_of_death_isAllCause_f5a_w5a_first

- dtype: `Boolean` (boolean)
- nulls: 6273 (99.98%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 6273
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## cause_of_death_number_of_days_to_death_for_all_cause_f5a_first

- dtype: `Int32` (numeric)
- nulls: 6273 (99.98%)
- min/max: 1.0 / 1.0
- mean/std: 1.0000 / None
- quantiles: {'0.05': 1.0, '0.25': 1.0, '0.5': 1.0, '0.75': 1.0, '0.95': 1.0}
- ⚠️ constant column (single value)

## eGFR_2021_ckd_epi_creatinine

- dtype: `Decimal(precision=38, scale=6)` (numeric)
- nulls: 0 (0.00%)
- min/max: 20.0 / 220.0
- mean/std: 76.8784 / 23.923231143294167
- quantiles: {'0.05': 35.52869065000001, '0.25': 57.874977, '0.5': 79.350515, '0.75': 95.771454, '0.95': 111.22690849999998}

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

- dtype: `Int32` (numeric)
- nulls: 4055 (64.63%)
- min/max: 8.0 / 44.0
- mean/std: 25.6133 / 6.503319716722066
- quantiles: {'0.05': 14.0, '0.25': 21.0, '0.5': 26.0, '0.75': 30.0, '0.95': 36.0}
