# Column Analysis

Total rows: 4445
Total columns: 540

## pid

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 4445
- ⚠️ high-cardinality column (4445 distinct values)
- top values (shown only where count ≥ 5):
  - (none met the display threshold)
  - 4445 other distinct value(s) suppressed, covering 4445 row(s) (count below 5 and/or ranked beyond top 20)

## encounterId

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 4445
- ⚠️ high-cardinality column (4445 distinct values)
- top values (shown only where count ≥ 5):
  - (none met the display threshold)
  - 4445 other distinct value(s) suppressed, covering 4445 row(s) (count below 5 and/or ranked beyond top 20)

## referenceTimePoint

- dtype: `Datetime(time_unit='ns', time_zone=None)` (categorical)
- nulls: 0 (0.00%)
- unique values: 2403
- ⚠️ high-cardinality column (2403 distinct values)
- top values (shown only where count ≥ 5):
  - `2023-06-20 00:00:00`: 7
  - `2024-01-12 00:00:00`: 7
  - `2016-04-13 00:00:00`: 6
  - `2017-04-07 00:00:00`: 6
  - `2019-02-01 00:00:00`: 6
  - `2019-03-26 00:00:00`: 6
  - `2019-05-17 00:00:00`: 6
  - `2019-10-01 00:00:00`: 6
  - `2021-07-02 00:00:00`: 6
  - `2022-10-28 00:00:00`: 6
  - `2022-12-21 00:00:00`: 6
  - `2023-02-16 00:00:00`: 6
  - `2024-01-23 00:00:00`: 6
  - `2024-02-28 00:00:00`: 6
  - `2024-03-15 00:00:00`: 6
  - `2024-11-21 00:00:00`: 6
  - `2025-06-27 00:00:00`: 6
  - `2025-12-03 00:00:00`: 6
  - `2016-07-27 00:00:00`: 5
  - `2016-08-02 00:00:00`: 5
  - 2383 other distinct value(s) suppressed, covering 4325 row(s) (count below 5 and/or ranked beyond top 20)

## eventTime

- dtype: `Datetime(time_unit='ns', time_zone=None)` (categorical)
- nulls: 0 (0.00%)
- unique values: 2502
- ⚠️ high-cardinality column (2502 distinct values)
- top values (shown only where count ≥ 5):
  - `2023-12-18 00:00:00`: 7
  - `2016-10-11 00:00:00`: 6
  - `2019-01-17 00:00:00`: 6
  - `2021-02-01 00:00:00`: 6
  - `2021-06-24 00:00:00`: 6
  - `2021-07-09 00:00:00`: 6
  - `2023-11-28 00:00:00`: 6
  - `2024-01-16 00:00:00`: 6
  - `2016-11-02 00:00:00`: 5
  - `2017-01-16 00:00:00`: 5
  - `2017-01-17 00:00:00`: 5
  - `2018-09-05 00:00:00`: 5
  - `2019-02-08 00:00:00`: 5
  - `2019-02-19 00:00:00`: 5
  - `2019-03-07 00:00:00`: 5
  - `2019-09-16 00:00:00`: 5
  - `2019-09-30 00:00:00`: 5
  - `2020-12-10 00:00:00`: 5
  - `2021-10-13 00:00:00`: 5
  - `2021-10-18 00:00:00`: 5
  - 2482 other distinct value(s) suppressed, covering 4336 row(s) (count below 5 and/or ranked beyond top 20)

## exitTime

- dtype: `Datetime(time_unit='ns', time_zone=None)` (categorical)
- nulls: 4445 (100.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 4445

## patient_demographics_sourceIdentifier

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 4445
- ⚠️ high-cardinality column (4445 distinct values)
- top values (shown only where count ≥ 5):
  - (none met the display threshold)
  - 4445 other distinct value(s) suppressed, covering 4445 row(s) (count below 5 and/or ranked beyond top 20)

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

## encounters_admissionDate

- dtype: `Datetime(time_unit='ns', time_zone=None)` (categorical)
- nulls: 0 (0.00%)
- unique values: 2502
- ⚠️ high-cardinality column (2502 distinct values)
- top values (shown only where count ≥ 5):
  - `2023-12-18 00:00:00`: 7
  - `2016-10-11 00:00:00`: 6
  - `2019-01-17 00:00:00`: 6
  - `2021-02-01 00:00:00`: 6
  - `2021-06-24 00:00:00`: 6
  - `2021-07-09 00:00:00`: 6
  - `2023-11-28 00:00:00`: 6
  - `2024-01-16 00:00:00`: 6
  - `2016-11-02 00:00:00`: 5
  - `2017-01-16 00:00:00`: 5
  - `2017-01-17 00:00:00`: 5
  - `2018-09-05 00:00:00`: 5
  - `2019-02-08 00:00:00`: 5
  - `2019-02-19 00:00:00`: 5
  - `2019-03-07 00:00:00`: 5
  - `2019-09-16 00:00:00`: 5
  - `2019-09-30 00:00:00`: 5
  - `2020-12-10 00:00:00`: 5
  - `2021-10-13 00:00:00`: 5
  - `2021-10-18 00:00:00`: 5
  - 2482 other distinct value(s) suppressed, covering 4336 row(s) (count below 5 and/or ranked beyond top 20)

## encounters_dischargeDate

- dtype: `Datetime(time_unit='ns', time_zone=None)` (categorical)
- nulls: 0 (0.00%)
- unique values: 2403
- ⚠️ high-cardinality column (2403 distinct values)
- top values (shown only where count ≥ 5):
  - `2023-06-20 00:00:00`: 7
  - `2024-01-12 00:00:00`: 7
  - `2016-04-13 00:00:00`: 6
  - `2017-04-07 00:00:00`: 6
  - `2019-02-01 00:00:00`: 6
  - `2019-03-26 00:00:00`: 6
  - `2019-05-17 00:00:00`: 6
  - `2019-10-01 00:00:00`: 6
  - `2021-07-02 00:00:00`: 6
  - `2022-10-28 00:00:00`: 6
  - `2022-12-21 00:00:00`: 6
  - `2023-02-16 00:00:00`: 6
  - `2024-01-23 00:00:00`: 6
  - `2024-02-28 00:00:00`: 6
  - `2024-03-15 00:00:00`: 6
  - `2024-11-21 00:00:00`: 6
  - `2025-06-27 00:00:00`: 6
  - `2025-12-03 00:00:00`: 6
  - `2016-07-27 00:00:00`: 5
  - `2016-08-02 00:00:00`: 5
  - 2383 other distinct value(s) suppressed, covering 4325 row(s) (count below 5 and/or ranked beyond top 20)

## encounters_numOfPreviousHFStays_count

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 790.0
- mean/std: 50.4160 / 65.15365392434441
- quantiles: {'0.05': 0.0, '0.25': 4.0, '0.5': 26.0, '0.75': 71.0, '0.95': 180.0}

## vital_signs_weight_value_p6mo_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 582 (13.09%)
- min/max: 30.0 / 210.0
- mean/std: 75.2718 / 19.62324505151762
- quantiles: {'0.05': 49.1, '0.25': 61.6, '0.5': 72.8, '0.75': 85.05, '0.95': 110.1}

## vital_signs_weight_value_p6mo_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 582 (13.09%)
- min/max: 35.0 / 250.0
- mean/std: 82.3314 / 21.298061665553785
- quantiles: {'0.05': 53.71, '0.25': 68.0, '0.5': 79.5, '0.75': 93.45, '0.95': 120.0}

## vital_signs_weight_value_p6mo_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 582 (13.09%)
- min/max: 32.0 / 220.0
- mean/std: 78.4674 / 20.193759559001073
- quantiles: {'0.05': 51.147666666666666, '0.25': 64.80000000000001, '0.5': 75.52, '0.75': 88.8343137254902, '0.95': 114.4976602564102}

## vital_signs_weight_value_p6mo_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 582 (13.09%)
- min/max: 32.0 / 210.0
- mean/std: 77.0965 / 19.999515548389795
- quantiles: {'0.05': 50.1, '0.25': 63.3, '0.5': 74.2, '0.75': 87.2, '0.95': 113.0}

## vital_signs_weight_value_p6mo_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 582 (13.09%)
- min/max: 33.0 / 250.0
- mean/std: 79.5544 / 20.527310011402754
- quantiles: {'0.05': 52.0, '0.25': 65.3, '0.5': 77.0, '0.75': 90.0, '0.95': 116.0}

## vital_signs_weight_value_p6mo_stddev

- dtype: `Float64` (numeric)
- nulls: 582 (13.09%)
- min/max: 0.0 / 24.0
- mean/std: 2.2283 / 2.039748534747475
- quantiles: {'0.05': 0.0, '0.25': 0.7753929083332103, '0.5': 1.8307344464467863, '0.75': 3.141205670788626, '0.95': 6.004728103399326}

## vital_signs_height_value_p1a_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 1040 (23.40%)
- min/max: 110.0 / 210.0
- mean/std: 171.1922 / 10.413875760083737
- quantiles: {'0.05': 154.0, '0.25': 164.0, '0.5': 171.33333333333331, '0.75': 178.0, '0.95': 187.17142857142852}

## vital_signs_weight_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 812 (18.27%)
- min/max: 32.0 / 210.0
- mean/std: 77.0821 / 20.00789714316492
- quantiles: {'0.05': 50.1, '0.25': 63.2, '0.5': 74.1, '0.75': 87.2, '0.95': 113.0}

## vital_signs_height_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2060 (46.34%)
- min/max: 110.0 / 210.0
- mean/std: 171.3765 / 10.500347252073981
- quantiles: {'0.05': 155.0, '0.25': 165.0, '0.5': 171.0, '0.75': 178.0, '0.95': 188.0}

## vital_signs_diastolicBp_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 464 (10.44%)
- min/max: 33.0 / 210.0
- mean/std: 91.5177 / 18.659121481169333
- quantiles: {'0.05': 66.0, '0.25': 80.0, '0.5': 89.0, '0.75': 100.0, '0.95': 125.0}

## vital_signs_diastolicBp_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 464 (10.44%)
- min/max: 0.0 / 160.0
- mean/std: 52.6740 / 14.352199319728058
- quantiles: {'0.05': 32.0, '0.25': 44.0, '0.5': 52.0, '0.75': 60.0, '0.95': 77.0}

## vital_signs_diastolicBp_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 464 (10.44%)
- min/max: 30.0 / 160.0
- mean/std: 70.2112 / 11.68001960929442
- quantiles: {'0.05': 53.666666666666664, '0.25': 62.81818181818183, '0.5': 69.0, '0.75': 76.2, '0.95': 90.0}

## vital_signs_diastolicBp_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 464 (10.44%)
- min/max: 6.0 / 160.0
- mean/std: 68.7940 / 13.383492210453591
- quantiles: {'0.05': 48.0, '0.25': 60.0, '0.5': 68.0, '0.75': 77.0, '0.95': 91.0}

## vital_signs_diastolicBp_value_stddev

- dtype: `Float64` (numeric)
- nulls: 464 (10.44%)
- min/max: 0.0 / 56.0
- mean/std: 9.4084 / 4.356611788249646
- quantiles: {'0.05': 2.8674417556808756, '0.25': 6.806665523112143, '0.5': 9.001405785900833, '0.75': 11.628310199573408, '0.95': 16.80594149167156}

## vital_signs_diastolicBp_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 464 (10.44%)
- min/max: 8.0 / 210.0
- mean/std: 75.3921 / 18.709918044974017
- quantiles: {'0.05': 49.0, '0.25': 63.0, '0.5': 73.0, '0.75': 85.0, '0.95': 109.0}

## vital_signs_heartRate_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 5 (0.11%)
- min/max: 39.0 / 300.0
- mean/std: 118.4399 / 32.33997964407053
- quantiles: {'0.05': 75.0, '0.25': 96.0, '0.5': 114.0, '0.75': 137.0, '0.95': 174.0}

## vital_signs_heartRate_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 5 (0.11%)
- min/max: 0.0 / 150.0
- mean/std: 50.3924 / 27.613033357871846
- quantiles: {'0.05': 0.0, '0.25': 39.0, '0.5': 57.0, '0.75': 68.0, '0.95': 87.0}

## vital_signs_heartRate_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 5 (0.11%)
- min/max: 15.0 / 180.0
- mean/std: 84.5401 / 15.259851818853548
- quantiles: {'0.05': 61.115222002262435, '0.25': 74.37318548387096, '0.5': 83.49320652173913, '0.75': 93.91258741258741, '0.95': 110.06205336951606}

## vital_signs_heartRate_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 5 (0.11%)
- min/max: 0.0 / 180.0
- mean/std: 81.3158 / 17.560818889798032
- quantiles: {'0.05': 57.0, '0.25': 69.0, '0.5': 80.0, '0.75': 91.0, '0.95': 112.0}

## vital_signs_heartRate_value_stddev

- dtype: `Float64` (numeric)
- nulls: 5 (0.11%)
- min/max: 0.0 / 62.0
- mean/std: 12.7773 / 7.159141883925378
- quantiles: {'0.05': 3.406044417351897, '0.25': 7.578082675953796, '0.5': 11.64486048058418, '0.75': 16.681418857051042, '0.95': 25.89499802992169}

## vital_signs_heartRate_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 5 (0.11%)
- min/max: 0.0 / 260.0
- mean/std: 90.9255 / 25.324905284841897
- quantiles: {'0.05': 57.0, '0.25': 73.0, '0.5': 88.0, '0.75': 105.0, '0.95': 136.0}

## vital_signs_systolicBp_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 464 (10.44%)
- min/max: 51.0 / 270.0
- mean/std: 152.3173 / 28.397635546550408
- quantiles: {'0.05': 111.0, '0.25': 132.0, '0.5': 150.0, '0.75': 170.0, '0.95': 203.0}

## vital_signs_systolicBp_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 464 (10.44%)
- min/max: 29.0 / 220.0
- mean/std: 97.7501 / 20.996268587395367
- quantiles: {'0.05': 67.0, '0.25': 84.0, '0.5': 96.0, '0.75': 109.0, '0.95': 134.0}

## vital_signs_systolicBp_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 464 (10.44%)
- min/max: 42.0 / 230.0
- mean/std: 122.8550 / 20.16780241872178
- quantiles: {'0.05': 94.88888888888889, '0.25': 108.15384615384615, '0.5': 120.19780219780219, '0.75': 135.66666666666669, '0.95': 158.11904761904762}

## vital_signs_systolicBp_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 464 (10.44%)
- min/max: 50.0 / 220.0
- mean/std: 120.7114 / 22.190673119573276
- quantiles: {'0.05': 90.0, '0.25': 105.0, '0.5': 118.0, '0.75': 134.0, '0.95': 159.0}

## vital_signs_systolicBp_value_stddev

- dtype: `Float64` (numeric)
- nulls: 464 (10.44%)
- min/max: 0.0 / 50.0
- mean/std: 13.7474 / 6.388677490018168
- quantiles: {'0.05': 4.1096093353126495, '0.25': 9.623552086760933, '0.5': 13.316020009330439, '0.75': 17.52520599080102, '0.95': 24.874815252048183}

## vital_signs_systolicBp_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 464 (10.44%)
- min/max: 46.0 / 260.0
- mean/std: 130.1874 / 28.759862660434035
- quantiles: {'0.05': 91.0, '0.25': 110.0, '0.5': 126.0, '0.75': 148.0, '0.95': 181.0}

## vital_signs_oxygenSaturation_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 2527 (56.85%)
- min/max: 0.0 / 100.0
- mean/std: 99.6621 / 5.325089656183501
- quantiles: {'0.05': 100.0, '0.25': 100.0, '0.5': 100.0, '0.75': 100.0, '0.95': 100.0}

## vital_signs_oxygenSaturation_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2527 (56.85%)
- min/max: 0.0 / 100.0
- mean/std: 99.7393 / 5.100433967009836
- quantiles: {'0.05': 100.0, '0.25': 100.0, '0.5': 100.0, '0.75': 100.0, '0.95': 100.0}

## vital_signs_oxygenSaturation_value_stddev

- dtype: `Float64` (numeric)
- nulls: 2527 (56.85%)
- min/max: 0.0 / 50.0
- mean/std: 0.1390 / 2.272644150528637
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 0.0}

## vital_signs_oxygenSaturation_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2527 (56.85%)
- min/max: 0.0 / 100.0
- mean/std: 99.6408 / 5.940651166733161
- quantiles: {'0.05': 100.0, '0.25': 100.0, '0.5': 100.0, '0.75': 100.0, '0.95': 100.0}

## vital_signs_oxygenSaturation_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2527 (56.85%)
- min/max: 0.0 / 100.0
- mean/std: 99.7393 / 5.100433967009836
- quantiles: {'0.05': 100.0, '0.25': 100.0, '0.5': 100.0, '0.75': 100.0, '0.95': 100.0}

## vital_signs_oxygenSaturation_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2527 (56.85%)
- min/max: 0.0 / 100.0
- mean/std: 99.3337 / 8.073425569357466
- quantiles: {'0.05': 100.0, '0.25': 100.0, '0.5': 100.0, '0.75': 100.0, '0.95': 100.0}

## lab_results_hemoglobin_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1269 (28.55%)
- min/max: 35.0 / 220.0
- mean/std: 115.4287 / 24.057201475803293
- quantiles: {'0.05': 78.9586, '0.25': 98.2954, '0.5': 114.4094, '0.75': 132.1348, '0.95': 156.3058}

## lab_results_hemoglobin_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1269 (28.55%)
- min/max: 40.0 / 240.0
- mean/std: 115.4718 / 22.76690392070071
- quantiles: {'0.05': 82.1814, '0.25': 99.50395, '0.5': 112.798, '0.75': 130.5234, '0.95': 156.3058}

## lab_results_hemoglobin_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1269 (28.55%)
- min/max: 0.16 / 220.0
- mean/std: 107.7290 / 25.98816516316438
- quantiles: {'0.05': 69.2902, '0.25': 88.627, '0.5': 106.3524, '0.75': 125.6892, '0.95': 151.4716}

## lab_results_hemoglobin_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1269 (28.55%)
- min/max: 51.0 / 240.0
- mean/std: 121.7779 / 22.173828847129766
- quantiles: {'0.05': 88.627, '0.25': 105.94955, '0.5': 119.2436, '0.75': 135.3576, '0.95': 161.14}

## lab_results_hemoglobin_value_stddev

- dtype: `Float64` (numeric)
- nulls: 1269 (28.55%)
- min/max: 0.0 / 48.0
- mean/std: 4.8877 / 5.335938492325409
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 3.6531871481440215, '0.75': 7.747515054188828, '0.95': 14.530546929137437}

## lab_results_hemoglobin_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 1269 (28.55%)
- min/max: 45.0 / 230.0
- mean/std: 114.5323 / 22.641151279040564
- quantiles: {'0.05': 83.6920875, '0.25': 96.684, '0.5': 111.72373333333334, '0.75': 129.71769999999998, '0.95': 154.6944}

## lab_results_ferritin_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 3794 (85.35%)
- min/max: 10.0 / 43000.0
- mean/std: 596.7989 / 2198.4794489173114
- quantiles: {'0.05': 30.015, '0.25': 83.15, '0.5': 202.0, '0.75': 470.0, '0.95': 1654.4999999999998}

## lab_results_ferritin_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 3794 (85.35%)
- min/max: 9.0 / 43000.0
- mean/std: 555.1936 / 1979.12339636844
- quantiles: {'0.05': 31.0, '0.25': 86.0, '0.5': 209.0, '0.75': 493.0, '0.95': 1584.4999999999998}

## lab_results_ferritin_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 3794 (85.35%)
- min/max: 9.0 / 43000.0
- mean/std: 527.4063 / 1927.9212968677518
- quantiles: {'0.05': 30.015, '0.25': 79.3, '0.5': 200.0, '0.75': 470.0, '0.95': 1519.5000000000002}

## lab_results_ferritin_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 3794 (85.35%)
- min/max: 10.0 / 43000.0
- mean/std: 658.0712 / 2520.4684347995903
- quantiles: {'0.05': 31.0, '0.25': 86.2, '0.5': 213.0, '0.75': 493.0, '0.95': 1679.5000000000002}

## lab_results_ferritin_value_stddev

- dtype: `Float64` (numeric)
- nulls: 3794 (85.35%)
- min/max: 0.0 / 6400.0
- mean/std: 50.4053 / 442.36012615319214
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 51.25}

## lab_results_ferritin_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 3794 (85.35%)
- min/max: 9.5 / 43000.0
- mean/std: 584.1632 / 2084.163868272495
- quantiles: {'0.05': 30.999999999999996, '0.25': 86.0, '0.5': 210.7, '0.75': 479.75, '0.95': 1679.5}

## lab_results_tfs_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4445 (100.00%)
- All values are null.

## lab_results_tfs_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4445 (100.00%)
- All values are null.

## lab_results_tfs_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4445 (100.00%)
- All values are null.

## lab_results_tfs_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4445 (100.00%)
- All values are null.

## lab_results_tfs_value_stddev

- dtype: `Float64` (numeric)
- nulls: 4445 (100.00%)
- All values are null.

## lab_results_tfs_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 4445 (100.00%)
- All values are null.

## lab_results_ntProBnp_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2771 (62.34%)
- min/max: 33.0 / 70000.0
- mean/std: 10327.7937 / 14172.968453382287
- quantiles: {'0.05': 468.25, '0.25': 1929.75, '0.5': 4708.0, '0.75': 11984.0, '0.95': 38497.55}

## lab_results_ntProBnp_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2771 (62.34%)
- min/max: 33.0 / 70000.0
- mean/std: 9930.4708 / 14230.682936458183
- quantiles: {'0.05': 433.25, '0.25': 1827.5000000000002, '0.5': 4287.5, '0.75': 11261.5, '0.95': 40407.49999999994}

## lab_results_ntProBnp_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2771 (62.34%)
- min/max: 33.0 / 70000.0
- mean/std: 8758.0139 / 12725.948397651651
- quantiles: {'0.05': 394.65, '0.25': 1657.25, '0.5': 3952.5, '0.75': 9679.0, '0.95': 33532.0}

## lab_results_ntProBnp_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2771 (62.34%)
- min/max: 33.0 / 70000.0
- mean/std: 11787.2599 / 15789.500095110614
- quantiles: {'0.05': 533.76, '0.25': 2134.75, '0.5': 5438.000000000001, '0.75': 14151.499999999998, '0.95': 48508.89999999997}

## lab_results_ntProBnp_value_stddev

- dtype: `Float64` (numeric)
- nulls: 2771 (62.34%)
- min/max: 0.0 / 29000.0
- mean/std: 1311.6208 / 3453.4802977251275
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 696.6699297940168, '0.95': 8838.844224253155}

## lab_results_ntProBnp_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 2771 (62.34%)
- min/max: 33.0 / 70000.0
- mean/std: 10177.0742 / 13740.03341044991
- quantiles: {'0.05': 519.5500000000001, '0.25': 1988.5000000000002, '0.5': 4790.5, '0.75': 12075.75, '0.95': 38834.39999999999}

## lab_results_bnp_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4445 (100.00%)
- All values are null.

## lab_results_bnp_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4445 (100.00%)
- All values are null.

## lab_results_bnp_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4445 (100.00%)
- All values are null.

## lab_results_bnp_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4445 (100.00%)
- All values are null.

## lab_results_bnp_value_stddev

- dtype: `Float64` (numeric)
- nulls: 4445 (100.00%)
- All values are null.

## lab_results_bnp_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 4445 (100.00%)
- All values are null.

## lab_results_crpNonHs_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1839 (41.37%)
- min/max: 0.3 / 550.0
- mean/std: 76.0436 / 83.75195005996497
- quantiles: {'0.05': 3.0, '0.25': 17.0, '0.5': 45.0, '0.75': 107.9, '0.95': 259.525}

## lab_results_crpNonHs_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1839 (41.37%)
- min/max: 0.3 / 630.0
- mean/std: 56.1088 / 65.43011502959291
- quantiles: {'0.05': 3.0, '0.25': 13.05, '0.5': 33.0, '0.75': 73.1, '0.95': 189.825}

## lab_results_crpNonHs_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1839 (41.37%)
- min/max: 0.3 / 550.0
- mean/std: 42.5135 / 51.28491954710767
- quantiles: {'0.05': 2.5, '0.25': 10.0, '0.5': 24.3, '0.75': 53.175, '0.95': 145.45}

## lab_results_crpNonHs_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1839 (41.37%)
- min/max: 0.6 / 630.0
- mean/std: 102.8719 / 102.21404857014114
- quantiles: {'0.05': 4.45, '0.25': 24.425, '0.5': 66.45, '0.75': 151.0, '0.95': 318.75}

## lab_results_crpNonHs_value_stddev

- dtype: `Float64` (numeric)
- nulls: 1839 (41.37%)
- min/max: 0.0 / 230.0
- mean/std: 22.1250 / 30.49380679368486
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 8.164281382128433, '0.75': 34.29585920299844, '0.95': 87.25899229535295}

## lab_results_crpNonHs_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 1839 (41.37%)
- min/max: 0.5 / 550.0
- mean/std: 68.6172 / 64.56129496415703
- quantiles: {'0.05': 4.0, '0.25': 20.0, '0.5': 48.77, '0.75': 98.97500000000001, '0.95': 201.29166666666666}

## lab_results_crpHs_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4445 (100.00%)
- All values are null.

## lab_results_crpHs_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4445 (100.00%)
- All values are null.

## lab_results_crpHs_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4445 (100.00%)
- All values are null.

## lab_results_crpHs_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4445 (100.00%)
- All values are null.

## lab_results_crpHs_value_stddev

- dtype: `Float64` (numeric)
- nulls: 4445 (100.00%)
- All values are null.

## lab_results_crpHs_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 4445 (100.00%)
- All values are null.

## lab_results_tropIHs_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4445 (100.00%)
- All values are null.

## lab_results_tropIHs_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4445 (100.00%)
- All values are null.

## lab_results_tropIHs_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4445 (100.00%)
- All values are null.

## lab_results_tropIHs_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4445 (100.00%)
- All values are null.

## lab_results_tropIHs_value_stddev

- dtype: `Float64` (numeric)
- nulls: 4445 (100.00%)
- All values are null.

## lab_results_tropIHs_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 4445 (100.00%)
- All values are null.

## lab_results_tropInHs_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4445 (100.00%)
- All values are null.

## lab_results_tropInHs_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4445 (100.00%)
- All values are null.

## lab_results_tropInHs_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4445 (100.00%)
- All values are null.

## lab_results_tropInHs_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4445 (100.00%)
- All values are null.

## lab_results_tropInHs_value_stddev

- dtype: `Float64` (numeric)
- nulls: 4445 (100.00%)
- All values are null.

## lab_results_tropInHs_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 4445 (100.00%)
- All values are null.

## lab_results_tropTHs_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4445 (100.00%)
- All values are null.

## lab_results_tropTHs_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4445 (100.00%)
- All values are null.

## lab_results_tropTHs_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4445 (100.00%)
- All values are null.

## lab_results_tropTHs_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4445 (100.00%)
- All values are null.

## lab_results_tropTHs_value_stddev

- dtype: `Float64` (numeric)
- nulls: 4445 (100.00%)
- All values are null.

## lab_results_tropTHs_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 4445 (100.00%)
- All values are null.

## lab_results_tropTnHs_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4445 (100.00%)
- All values are null.

## lab_results_tropTnHs_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4445 (100.00%)
- All values are null.

## lab_results_tropTnHs_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4445 (100.00%)
- All values are null.

## lab_results_tropTnHs_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4445 (100.00%)
- All values are null.

## lab_results_tropTnHs_value_stddev

- dtype: `Float64` (numeric)
- nulls: 4445 (100.00%)
- All values are null.

## lab_results_tropTnHs_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 4445 (100.00%)
- All values are null.

## lab_results_triGly_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4213 (94.78%)
- min/max: 0.24 / 17.0
- mean/std: 1.6504 / 1.4814515625809281
- quantiles: {'0.05': 0.621, '0.25': 0.93775, '0.5': 1.3, '0.75': 1.94, '0.95': 3.4879999999999995}

## lab_results_triGly_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4213 (94.78%)
- min/max: 0.24 / 12.0
- mean/std: 1.5953 / 1.1025300605124027
- quantiles: {'0.05': 0.6355, '0.25': 0.949, '0.5': 1.315, '0.75': 1.945, '0.95': 3.322499999999997}

## lab_results_triGly_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4213 (94.78%)
- min/max: 0.24 / 12.0
- mean/std: 1.5680 / 1.10098603568969
- quantiles: {'0.05': 0.621, '0.25': 0.93775, '0.5': 1.255, '0.75': 1.8875, '0.95': 3.322499999999997}

## lab_results_triGly_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4213 (94.78%)
- min/max: 0.24 / 17.0
- mean/std: 1.6837 / 1.5055523920804943
- quantiles: {'0.05': 0.6355, '0.25': 0.949, '0.5': 1.32, '0.75': 1.98, '0.95': 3.5324999999999993}

## lab_results_triGly_value_stddev

- dtype: `Float64` (numeric)
- nulls: 4213 (94.78%)
- min/max: 0.0 / 3.0
- mean/std: 0.0313 / 0.22081967435478136
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 0.02724999999999994}

## lab_results_triGly_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 4213 (94.78%)
- min/max: 0.24 / 12.0
- mean/std: 1.5978 / 1.1086349302330465
- quantiles: {'0.05': 0.6355000000000001, '0.25': 0.9444999999999999, '0.5': 1.315, '0.75': 1.9324999999999999, '0.95': 3.4644999999999997}

## lab_results_cholTot_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4217 (94.87%)
- min/max: 1.4 / 12.0
- mean/std: 3.8629 / 1.1780740029892227
- quantiles: {'0.05': 2.2975, '0.25': 3.1675, '0.5': 3.76, '0.75': 4.4325, '0.95': 5.853}

## lab_results_cholTot_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4217 (94.87%)
- min/max: 1.4 / 12.0
- mean/std: 3.8527 / 1.1848751565960223
- quantiles: {'0.05': 2.267, '0.25': 3.1475, '0.5': 3.75, '0.75': 4.4325, '0.95': 5.853}

## lab_results_cholTot_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4217 (94.87%)
- min/max: 1.4 / 12.0
- mean/std: 3.8457 / 1.1894773614750418
- quantiles: {'0.05': 2.221, '0.25': 3.1475, '0.5': 3.72, '0.75': 4.4325, '0.95': 5.853}

## lab_results_cholTot_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4217 (94.87%)
- min/max: 1.4 / 12.0
- mean/std: 3.8699 / 1.1733231511144024
- quantiles: {'0.05': 2.3335, '0.25': 3.1675, '0.5': 3.765, '0.75': 4.4325, '0.95': 5.853}

## lab_results_cholTot_value_stddev

- dtype: `Float64` (numeric)
- nulls: 4217 (94.87%)
- min/max: 0.0 / 1.1
- mean/std: 0.0121 / 0.08418038936053857
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 0.0}

## lab_results_cholTot_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 4217 (94.87%)
- min/max: 1.4 / 12.0
- mean/std: 3.8578 / 1.1784152899725482
- quantiles: {'0.05': 2.2975000000000003, '0.25': 3.16, '0.5': 3.75, '0.75': 4.4325, '0.95': 5.853}

## lab_results_hdl_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4232 (95.21%)
- min/max: 0.16 / 2.6
- mean/std: 1.0983 / 0.3961841496815007
- quantiles: {'0.05': 0.57, '0.25': 0.83, '0.5': 1.05, '0.75': 1.28, '0.95': 1.8979999999999995}

## lab_results_hdl_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4232 (95.21%)
- min/max: 0.22 / 2.6
- mean/std: 1.1005 / 0.39654047113451885
- quantiles: {'0.05': 0.57, '0.25': 0.83, '0.5': 1.06, '0.75': 1.29, '0.95': 1.8979999999999995}

## lab_results_hdl_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4232 (95.21%)
- min/max: 0.16 / 2.6
- mean/std: 1.0961 / 0.39920771738663446
- quantiles: {'0.05': 0.562, '0.25': 0.83, '0.5': 1.05, '0.75': 1.28, '0.95': 1.8979999999999995}

## lab_results_hdl_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4232 (95.21%)
- min/max: 0.22 / 2.6
- mean/std: 1.1026 / 0.3941848244604091
- quantiles: {'0.05': 0.576, '0.25': 0.83, '0.5': 1.06, '0.75': 1.29, '0.95': 1.8979999999999995}

## lab_results_hdl_value_stddev

- dtype: `Float64` (numeric)
- nulls: 4232 (95.21%)
- min/max: 0.0 / 0.2
- mean/std: 0.0031 / 0.02116036142870751
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 0.0}

## lab_results_hdl_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 4232 (95.21%)
- min/max: 0.21 / 2.6
- mean/std: 1.0995 / 0.39608281356407604
- quantiles: {'0.05': 0.57, '0.25': 0.83, '0.5': 1.06, '0.75': 1.2899999999999998, '0.95': 1.8979999999999995}

## lab_results_creatUS_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4051 (91.14%)
- min/max: 100.0 / 2900.0
- mean/std: 676.6500 / 433.76995938545934
- quantiles: {'0.05': 210.4032, '0.25': 369.9024, '0.5': 576.912, '0.75': 878.094, '0.95': 1537.69672}

## lab_results_creatUS_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4051 (91.14%)
- min/max: 22.0 / 3200.0
- mean/std: 650.4233 / 442.17478441840075
- quantiles: {'0.05': 170.41528, '0.25': 350.672, '0.5': 531.664, '0.75': 828.604, '0.95': 1469.315679999998}

## lab_results_creatUS_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4051 (91.14%)
- min/max: 22.0 / 2700.0
- mean/std: 563.6511 / 402.48286699513915
- quantiles: {'0.05': 147.056, '0.25': 283.6484, '0.5': 441.168, '0.75': 732.452, '0.95': 1338.0399199999995}

## lab_results_creatUS_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4051 (91.14%)
- min/max: 100.0 / 3200.0
- mean/std: 779.7987 / 480.8296625616766
- quantiles: {'0.05': 214.928, '0.25': 429.856, '0.5': 672.4984, '0.75': 1017.5144, '0.95': 1700.7591999999995}

## lab_results_creatUS_value_stddev

- dtype: `Float64` (numeric)
- nulls: 4051 (91.14%)
- min/max: 0.0 / 1400.0
- mean/std: 90.9743 / 160.8169389362776
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 146.59078736266542, '0.95': 408.60643503590467}

## lab_results_creatUS_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 4051 (91.14%)
- min/max: 100.0 / 2700.0
- mean/std: 663.3055 / 396.9817228055458
- quantiles: {'0.05': 213.40088, '0.25': 390.54679999999996, '0.5': 565.1758000000001, '0.75': 818.9888, '0.95': 1371.80624}

## lab_results_albuminUS_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4445 (100.00%)
- All values are null.

## lab_results_albuminUS_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4445 (100.00%)
- All values are null.

## lab_results_albuminUS_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4445 (100.00%)
- All values are null.

## lab_results_albuminUS_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4445 (100.00%)
- All values are null.

## lab_results_albuminUS_value_stddev

- dtype: `Float64` (numeric)
- nulls: 4445 (100.00%)
- All values are null.

## lab_results_albuminUS_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 4445 (100.00%)
- All values are null.

## lab_results_bun_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4445 (100.00%)
- All values are null.

## lab_results_bun_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4445 (100.00%)
- All values are null.

## lab_results_bun_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4445 (100.00%)
- All values are null.

## lab_results_bun_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4445 (100.00%)
- All values are null.

## lab_results_bun_value_stddev

- dtype: `Float64` (numeric)
- nulls: 4445 (100.00%)
- All values are null.

## lab_results_bun_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 4445 (100.00%)
- All values are null.

## lab_results_acr_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4445 (100.00%)
- All values are null.

## lab_results_acr_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4445 (100.00%)
- All values are null.

## lab_results_acr_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4445 (100.00%)
- All values are null.

## lab_results_acr_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4445 (100.00%)
- All values are null.

## lab_results_acr_value_stddev

- dtype: `Float64` (numeric)
- nulls: 4445 (100.00%)
- All values are null.

## lab_results_acr_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 4445 (100.00%)
- All values are null.

## lab_results_ldl_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4230 (95.16%)
- min/max: 0.1 / 8.2
- mean/std: 2.1320 / 0.9954176626632288
- quantiles: {'0.05': 0.8560000000000001, '0.25': 1.46, '0.5': 1.99, '0.75': 2.66, '0.95': 3.923}

## lab_results_ldl_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 4230 (95.16%)
- min/max: 0.1 / 8.2
- mean/std: 2.1407 / 0.9895367821898406
- quantiles: {'0.05': 0.88455, '0.25': 1.48, '0.5': 2.0, '0.75': 2.6599999999999997, '0.95': 3.9229999999999996}

## lab_results_ldl_value_stddev

- dtype: `Float64` (numeric)
- nulls: 4230 (95.16%)
- min/max: 0.0 / 1.1
- mean/std: 0.0087 / 0.08109095552971188
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 0.0}

## lab_results_ldl_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4230 (95.16%)
- min/max: 0.1 / 8.2
- mean/std: 2.1496 / 0.9903651134499382
- quantiles: {'0.05': 0.88455, '0.25': 1.48, '0.5': 2.01, '0.75': 2.68, '0.95': 3.923}

## lab_results_ldl_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4230 (95.16%)
- min/max: 0.1 / 8.2
- mean/std: 2.1480 / 0.9924543886422783
- quantiles: {'0.05': 0.88455, '0.25': 1.48, '0.5': 2.01, '0.75': 2.68, '0.95': 3.923}

## lab_results_ldl_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4230 (95.16%)
- min/max: 0.1 / 8.2
- mean/std: 2.1339 / 0.9933871315269002
- quantiles: {'0.05': 0.8560000000000001, '0.25': 1.46, '0.5': 1.99, '0.75': 2.66, '0.95': 3.923}

## lab_results_potassium_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 158 (3.55%)
- min/max: 2.7 / 6.8
- mean/std: 4.0959 / 0.4682216206567404
- quantiles: {'0.05': 3.440342857142857, '0.25': 3.7888888888888888, '0.5': 4.039000000000001, '0.75': 4.35, '0.95': 4.938714285714285}

## lab_results_potassium_value_stddev

- dtype: `Float64` (numeric)
- nulls: 158 (3.55%)
- min/max: 0.0 / 4.7
- mean/std: 0.3033 / 0.2496337254450668
- quantiles: {'0.05': 0.0, '0.25': 0.11004259714075999, '0.5': 0.2861380785564899, '0.75': 0.4425759031756751, '0.95': 0.7216527501174067}

## lab_results_potassium_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 158 (3.55%)
- min/max: 1.1 / 6.5
- mean/std: 3.6427 / 0.5570807471589699
- quantiles: {'0.05': 2.82, '0.25': 3.3, '0.5': 3.6, '0.75': 3.99, '0.95': 4.6}

## lab_results_potassium_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 158 (3.55%)
- min/max: 2.0 / 8.1
- mean/std: 4.1680 / 0.5854954035555506
- quantiles: {'0.05': 3.35, '0.25': 3.8, '0.5': 4.1, '0.75': 4.5, '0.95': 5.2}

## lab_results_potassium_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 158 (3.55%)
- min/max: 1.1 / 7.6
- mean/std: 4.0854 / 0.6294538931855286
- quantiles: {'0.05': 3.2, '0.25': 3.67, '0.5': 4.0, '0.75': 4.4, '0.95': 5.226999999999998}

## lab_results_potassium_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 158 (3.55%)
- min/max: 2.7 / 15.0
- mean/std: 4.6428 / 0.8286893511128934
- quantiles: {'0.05': 3.6, '0.25': 4.1, '0.5': 4.5, '0.75': 5.0, '0.95': 6.1}

## lab_results_sodium_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 240 (5.40%)
- min/max: 2.0 / 150.0
- mean/std: 135.4400 / 5.668347985631159
- quantiles: {'0.05': 126.0, '0.25': 133.0, '0.5': 136.0, '0.75': 139.0, '0.95': 143.0}

## lab_results_sodium_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 240 (5.40%)
- min/max: 100.0 / 160.0
- mean/std: 137.9454 / 4.2996296407048336
- quantiles: {'0.05': 130.60307692307694, '0.25': 135.6, '0.5': 138.25, '0.75': 140.66666666666666, '0.95': 144.25444444444446}

## lab_results_sodium_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 240 (5.40%)
- min/max: 110.0 / 180.0
- mean/std: 140.4708 / 4.901235924030761
- quantiles: {'0.05': 133.0, '0.25': 138.0, '0.5': 140.3, '0.75': 143.0, '0.95': 148.0}

## lab_results_sodium_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 240 (5.40%)
- min/max: 2.0 / 170.0
- mean/std: 137.9002 / 5.038196392035766
- quantiles: {'0.05': 130.0, '0.25': 135.2, '0.5': 138.0, '0.75': 141.0, '0.95': 144.8}

## lab_results_sodium_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 240 (5.40%)
- min/max: 100.0 / 160.0
- mean/std: 137.9150 / 4.795781158130408
- quantiles: {'0.05': 129.9, '0.25': 135.4, '0.5': 138.3, '0.75': 141.0, '0.95': 144.7}

## lab_results_sodium_value_stddev

- dtype: `Float64` (numeric)
- nulls: 240 (5.40%)
- min/max: 0.0 / 62.0
- mean/std: 1.6526 / 1.8200568643428259
- quantiles: {'0.05': 0.0, '0.25': 0.4642796092394701, '0.5': 1.4142135623730951, '0.75': 2.351616465327627, '0.95': 4.545023045348558}

## lab_results_albuminBS_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2932 (65.96%)
- min/max: 8.3 / 47.0
- mean/std: 29.0979 / 6.676529075601147
- quantiles: {'0.05': 18.0, '0.25': 24.4, '0.5': 29.0, '0.75': 34.0, '0.95': 40.0}

## lab_results_albuminBS_value_stddev

- dtype: `Float64` (numeric)
- nulls: 2932 (65.96%)
- min/max: 0.0 / 9.6
- mean/std: 1.1337 / 1.4846550982745461
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.5, '0.75': 1.9165942015286734, '0.95': 4.0316229369994865}

## lab_results_albuminBS_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2932 (65.96%)
- min/max: 10.0 / 49.0
- mean/std: 30.2416 / 6.32096301601697
- quantiles: {'0.05': 20.2, '0.25': 26.0, '0.5': 30.0, '0.75': 34.8, '0.95': 41.0}

## lab_results_albuminBS_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2932 (65.96%)
- min/max: 9.0 / 47.0
- mean/std: 28.5941 / 6.7601866337207985
- quantiles: {'0.05': 17.56, '0.25': 23.9, '0.5': 28.6, '0.75': 33.0, '0.95': 40.0}

## lab_results_albuminBS_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 2932 (65.96%)
- min/max: 10.0 / 47.0
- mean/std: 28.5297 / 6.59908539485201
- quantiles: {'0.05': 17.588571428571427, '0.25': 23.7, '0.5': 28.5, '0.75': 33.0, '0.95': 40.0}

## lab_results_albuminBS_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2932 (65.96%)
- min/max: 6.8 / 47.0
- mean/std: 27.0867 / 7.354039852066153
- quantiles: {'0.05': 14.86, '0.25': 21.7, '0.5': 27.1, '0.75': 32.0, '0.95': 39.0}

## lab_results_hba1c%_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4382 (98.58%)
- min/max: 26.0 / 160.0
- mean/std: 57.0206 / 23.664327030023948
- quantiles: {'0.05': 35.67, '0.25': 42.0, '0.5': 49.9, '0.75': 65.4, '0.95': 101.69999999999999}

## lab_results_hba1c%_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4382 (98.58%)
- min/max: 26.0 / 160.0
- mean/std: 57.0206 / 23.664327030023948
- quantiles: {'0.05': 35.67, '0.25': 42.0, '0.5': 49.9, '0.75': 65.4, '0.95': 101.69999999999999}

## lab_results_hba1c%_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 4382 (98.58%)
- min/max: 26.0 / 160.0
- mean/std: 57.0206 / 23.664327030023948
- quantiles: {'0.05': 35.67, '0.25': 42.0, '0.5': 49.9, '0.75': 65.4, '0.95': 101.69999999999999}

## lab_results_hba1c%_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4382 (98.58%)
- min/max: 26.0 / 160.0
- mean/std: 57.0206 / 23.664327030023948
- quantiles: {'0.05': 35.67, '0.25': 42.0, '0.5': 49.9, '0.75': 65.4, '0.95': 101.69999999999999}

## lab_results_hba1c%_value_stddev

- dtype: `Float64` (numeric)
- nulls: 4382 (98.58%)
- min/max: 0.0 / 0.0
- mean/std: 0.0000 / 0.0
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 0.0}
- ⚠️ constant column (single value)

## lab_results_hba1c%_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4382 (98.58%)
- min/max: 26.0 / 160.0
- mean/std: 57.0206 / 23.664327030023948
- quantiles: {'0.05': 35.67, '0.25': 42.0, '0.5': 49.9, '0.75': 65.4, '0.95': 101.69999999999999}

## lab_results_hba1c_value_stddev

- dtype: `Float64` (numeric)
- nulls: 4285 (96.40%)
- min/max: 0.0 / 5.0
- mean/std: 0.0669 / 0.4606767444208758
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 0.0}

## lab_results_hba1c_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4285 (96.40%)
- min/max: 4.6 / 150.0
- mean/std: 37.7821 / 31.916231731214722
- quantiles: {'0.05': 5.6385, '0.25': 7.175, '0.5': 39.0, '0.75': 54.25, '0.95': 100.24999999999991}

## lab_results_hba1c_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 4285 (96.40%)
- min/max: 4.6 / 150.0
- mean/std: 37.7796 / 31.866150103677708
- quantiles: {'0.05': 5.6385, '0.25': 7.175, '0.5': 39.0, '0.75': 54.49999999999999, '0.95': 96.44999999999985}

## lab_results_hba1c_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4285 (96.40%)
- min/max: 4.6 / 150.0
- mean/std: 37.7133 / 31.801283442354567
- quantiles: {'0.05': 5.6385, '0.25': 7.175, '0.5': 39.0, '0.75': 54.25, '0.95': 96.44999999999985}

## lab_results_hba1c_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4285 (96.40%)
- min/max: 4.6 / 150.0
- mean/std: 37.7904 / 31.832297784615353
- quantiles: {'0.05': 5.6385, '0.25': 7.235, '0.5': 39.0, '0.75': 54.5, '0.95': 96.44999999999985}

## lab_results_hba1c_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4285 (96.40%)
- min/max: 4.6 / 150.0
- mean/std: 37.8529 / 31.945068957134847
- quantiles: {'0.05': 5.6385, '0.25': 7.235, '0.5': 39.0, '0.75': 54.5, '0.95': 100.24999999999991}

## lab_results_validSerumCreatinine_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 604 (13.59%)
- min/max: 1.1 / 23.0
- mean/std: 12.6110 / 4.338528867414379
- quantiles: {'0.05': 6.30108375, '0.25': 9.274199999999999, '0.5': 12.04515, '0.75': 15.680982352941175, '0.95': 20.499375}

## lab_results_validSerumCreatinine_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 604 (13.59%)
- min/max: 1.1 / 23.0
- mean/std: 14.2637 / 4.870889513687657
- quantiles: {'0.05': 7.1253, '0.25': 10.2921, '0.5': 13.6851, '0.75': 18.6615, '0.95': 22.1676}

## lab_results_validSerumCreatinine_value_stddev

- dtype: `Float64` (numeric)
- nulls: 604 (13.59%)
- min/max: 0.0 / 6.9
- mean/std: 1.0215 / 1.0700234257577423
- quantiles: {'0.05': 0.0, '0.25': 0.1131000000000002, '0.5': 0.7475516449550492, '0.75': 1.4942280220294968, '0.95': 3.2251132604142754}

## lab_results_validSerumCreatinine_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 604 (13.59%)
- min/max: 1.1 / 23.0
- mean/std: 12.8690 / 4.604636721870002
- quantiles: {'0.05': 6.36753, '0.25': 9.2742, '0.5': 12.1017, '0.75': 16.1733, '0.95': 21.2628}

## lab_results_validSerumCreatinine_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 604 (13.59%)
- min/max: 0.22 / 23.0
- mean/std: 11.2896 / 4.335605816172391
- quantiles: {'0.05': 5.15736, '0.25': 8.1432, '0.5': 10.6314, '0.75': 13.9113, '0.95': 19.7925}

## lab_results_valideGFR_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 309 (6.95%)
- min/max: 4.0 / 93.0
- mean/std: 50.9530 / 24.008290538177285
- quantiles: {'0.05': 13.276785714285715, '0.25': 31.49404761904762, '0.5': 50.0, '0.75': 70.1456043956044, '0.95': 90.0}

## lab_results_valideGFR_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 309 (6.95%)
- min/max: 4.0 / 90.0
- mean/std: 50.3552 / 24.609615845960917
- quantiles: {'0.05': 13.0, '0.25': 31.0, '0.5': 49.0, '0.75': 69.0, '0.95': 90.0}

## lab_results_valideGFR_value_stddev

- dtype: `Float64` (numeric)
- nulls: 309 (6.95%)
- min/max: 0.0 / 31.0
- mean/std: 4.2683 / 5.1777163417275585
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 2.5495097567963922, '0.75': 6.342099196813483, '0.95': 15.000120521479257}

## lab_results_valideGFR_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 309 (6.95%)
- min/max: 4.0 / 90.0
- mean/std: 44.6025 / 24.275925613935556
- quantiles: {'0.05': 10.0, '0.25': 25.0, '0.5': 42.0, '0.75': 61.0, '0.95': 90.0}

## lab_results_valideGFR_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 309 (6.95%)
- min/max: 4.0 / 110.0
- mean/std: 56.8194 / 24.95272475623623
- quantiles: {'0.05': 15.0, '0.25': 37.0, '0.5': 58.0, '0.75': 81.0, '0.95': 90.0}

## lab_results_valideGFR_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 309 (6.95%)
- min/max: 4.0 / 95.0
- mean/std: 51.6753 / 25.01935847670144
- quantiles: {'0.05': 12.0, '0.25': 31.0, '0.5': 51.0, '0.75': 72.25, '0.95': 90.0}

## symptoms_Ankle_swelling_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4445

## symptoms_Ascites_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4445

## symptoms_Breathlessness_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4445

## symptoms_Cardiac_murmur_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4445

## symptoms_Chest_pain_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4445

## symptoms_Cheyne_stokes_respiration_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4445

## symptoms_Depression_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4445

## symptoms_Dizziness_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4445

## symptoms_Elevated_jugular_venous_pressure_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4445

## symptoms_Fatigue_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4445

## symptoms_Hepatojugular_reflux_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4445

## symptoms_Hepatomegaly_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4445

## symptoms_Intermittent_claudication_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4445

## symptoms_Irregular_pulse_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4445

## symptoms_Loss_of_appetite_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4445

## symptoms_Nocturnal_cough_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4445

## symptoms_Oliguria_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4445

## symptoms_Orthopnoea_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4445

## symptoms_Palpitations_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4445

## symptoms_Paroxysmal_nocturnal_dyspnea_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4445

## symptoms_Peripheral_edema_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4445

## symptoms_Pleural_effusion_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4445

## symptoms_Pulmonary_crepitations_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4445

## symptoms_Reduced_exercise_tolerance_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4445

## symptoms_Syncope_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4445

## symptoms_Tachycardia_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4445

## symptoms_Tachypnoea_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4445

## symptoms_Third_heart_sound_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4445

## symptoms_Weight_gain_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4445

## symptoms_Weight_loss_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4445

## echocardiographs_lvef

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 3291 (74.04%)
- min/max: -16.0 / 100.0
- mean/std: 43.1324 / 17.856909278074653
- quantiles: {'0.05': 17.6248635, '0.25': 29.459662609100498, '0.5': 42.3399879455565, '0.75': 54.97719478607175, '0.95': 75.93449615899999}

## echocardiographs_lvef_pET_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 3528 (79.37%)
- min/max: -16.0 / 97.0
- mean/std: 40.9863 / 17.50135788062598
- quantiles: {'0.05': 16.754532, '0.25': 27.782604099999997, '0.5': 39.0, '0.75': 52.0, '0.95': 73.86753488400001}

## echocardiographs_lvef_pET_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 3240 (72.89%)
- min/max: -24.0 / 97.0
- mean/std: 41.4719 / 17.397526917623093
- quantiles: {'0.05': 17.0064395013008, '0.25': 28.22803247, '0.5': 39.430878135, '0.75': 53.542140960693, '0.95': 73.134868781}

## echocardiographs_lvef_pET_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 3240 (72.89%)
- min/max: -200.0 / 97.0
- mean/std: 39.0111 / 20.098785250994506
- quantiles: {'0.05': 14.373382, '0.25': 26.13336, '0.5': 36.99039, '0.75': 51.05138, '0.95': 72.734596372}

## echocardiographs_lvef_pET_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 3240 (72.89%)
- min/max: -16.0 / 97.0
- mean/std: 43.7565 / 17.793119071548578
- quantiles: {'0.05': 18.172077257999998, '0.25': 30.14, '0.5': 42.47, '0.75': 55.376155853271, '0.95': 77.0}

## echocardiographs_lvef_pET_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 3471 (78.09%)
- min/max: -44.0 / 96.0
- mean/std: 41.3280 / 17.745499787507114
- quantiles: {'0.05': 15.381782293319649, '0.25': 27.887958645000005, '0.5': 39.614588552246005, '0.75': 53.48421066284175, '0.95': 73.226016424}

## echocardiographs_lvef_pET_stddev

- dtype: `Float64` (numeric)
- nulls: 3240 (72.89%)
- min/max: 0.0 / 94.0
- mean/std: 2.1496 / 5.462662081825407
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.048731775000000255, '0.75': 2.1743580778883844, '0.95': 11.404983427055203}

## electrocardiographs_ecg_qrs_duration_pET_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2006 (45.13%)
- min/max: 4.0 / 250.0
- mean/std: 108.4736 / 31.51393329801255
- quantiles: {'0.05': 70.0, '0.25': 86.0, '0.5': 100.0, '0.75': 128.0, '0.95': 169.0}

## electrocardiographs_ecg_qrs_duration_pET_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2006 (45.13%)
- min/max: 57.0 / 300.0
- mean/std: 132.8909 / 39.93686049233679
- quantiles: {'0.05': 86.0, '0.25': 101.0, '0.5': 122.0, '0.75': 160.5, '0.95': 208.0}

## electrocardiographs_ecg_qrs_duration_pET_stddev

- dtype: `Float64` (numeric)
- nulls: 2006 (45.13%)
- min/max: 0.0 / 82.0
- mean/std: 7.8720 / 9.12563871875336
- quantiles: {'0.05': 0.0, '0.25': 2.0, '0.5': 5.1720402163943, '0.75': 9.91028497299715, '0.95': 26.498978690448695}

## electrocardiographs_ecg_qrs_duration_pET_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 2006 (45.13%)
- min/max: 56.0 / 260.0
- mean/std: 119.3637 / 32.74665939653924
- quantiles: {'0.05': 80.66666666666667, '0.25': 94.0, '0.5': 109.375, '0.75': 141.77083333333334, '0.95': 181.0}

## electrocardiographs_ecg_qrs_duration_pET_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2012 (45.26%)
- min/max: 12.0 / 260.0
- mean/std: 119.0140 / 34.63161112110005
- quantiles: {'0.05': 78.0, '0.25': 93.0, '0.5': 109.0, '0.75': 142.0, '0.95': 185.4000000000001}

## electrocardiographs_ecg_qrs_duration_pET_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2008 (45.17%)
- min/max: 20.0 / 270.0
- mean/std: 119.6118 / 34.46952426859718
- quantiles: {'0.05': 80.0, '0.25': 94.0, '0.5': 109.0, '0.75': 143.0, '0.95': 184.0}

## electrocardiographs_ecg_qrs_axis_pET_stddev

- dtype: `Float64` (numeric)
- nulls: 4445 (100.00%)
- All values are null.

## electrocardiographs_ecg_qrs_axis_pET_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4445 (100.00%)
- All values are null.

## electrocardiographs_ecg_qrs_axis_pET_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4445 (100.00%)
- All values are null.

## electrocardiographs_ecg_qrs_axis_pET_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 4445 (100.00%)
- All values are null.

## electrocardiographs_ecg_qrs_axis_pET_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4445 (100.00%)
- All values are null.

## electrocardiographs_ecg_qrs_axis_pET_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 4445 (100.00%)
- All values are null.

## electrocardiographs_ecg_qt_duration_corrected_pET_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2006 (45.13%)
- min/max: 260.0 / 880.0
- mean/std: 508.7339 / 63.769459883942005
- quantiles: {'0.05': 422.0, '0.25': 464.0, '0.5': 500.0, '0.75': 544.0, '0.95': 623.0}

## electrocardiographs_ecg_qt_duration_corrected_pET_stddev

- dtype: `Float64` (numeric)
- nulls: 2006 (45.13%)
- min/max: 0.0 / 190.0
- mean/std: 26.2101 / 21.272264894598273
- quantiles: {'0.05': 0.0, '0.25': 10.168536255847414, '0.5': 23.763154251908563, '0.75': 37.75316869709621, '0.95': 64.33784883983736}

## electrocardiographs_ecg_qt_duration_corrected_pET_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 2006 (45.13%)
- min/max: 260.0 / 640.0
- mean/std: 465.2923 / 41.860850847881146
- quantiles: {'0.05': 404.0, '0.25': 437.1428571428571, '0.5': 462.0, '0.75': 489.29166666666674, '0.95': 541.3499999999999}

## electrocardiographs_ecg_qt_duration_corrected_pET_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2008 (45.17%)
- min/max: 150.0 / 770.0
- mean/std: 463.1793 / 51.71314175566883
- quantiles: {'0.05': 389.0, '0.25': 429.0, '0.5': 458.0, '0.75': 491.0, '0.95': 552.0}

## electrocardiographs_ecg_qt_duration_corrected_pET_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2006 (45.13%)
- min/max: 130.0 / 640.0
- mean/std: 426.4608 / 52.85082016611834
- quantiles: {'0.05': 347.0, '0.25': 397.0, '0.5': 426.0, '0.75': 457.0, '0.95': 511.0999999999999}

## electrocardiographs_ecg_qt_duration_corrected_pET_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2013 (45.29%)
- min/max: 140.0 / 810.0
- mean/std: 464.2599 / 53.900268077923556
- quantiles: {'0.05': 386.0, '0.25': 430.0, '0.5': 460.0, '0.75': 494.0, '0.95': 557.0}

## electrocardiographs_ecg_st_pET

- dtype: `Boolean` (boolean)
- nulls: 4396 (98.90%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 4396
  - `True`: 49

## electrocardiographs_ecg_ischemia_without_st_pET

- dtype: `Boolean` (boolean)
- nulls: 4445 (100.00%)
- unique values: 0
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 4445

## electrocardiographs_ecg_type_of_rhythms_pET_first

- dtype: `List(String)` (categorical)
- nulls: 2005 (45.11%)
- unique values: 2
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `[]`: 2440
  - null (missing): 2005

## electrocardiographs_ecg_type_of_rhythms_pET_last

- dtype: `List(String)` (categorical)
- nulls: 2005 (45.11%)
- unique values: 2
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `[]`: 2440
  - null (missing): 2005

## smoking_status_smoker_last

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4204
  - `True`: 241

## smoking_status_formerSmoker_last

- dtype: `Boolean` (boolean)
- nulls: 4185 (94.15%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 4185
  - `True`: 241
  - `False`: 19

## smoking_status_smoker_totalSmokingDuration_sum

- dtype: `Int64` (numeric)
- nulls: 4374 (98.40%)
- min/max: 880.0 / 340000.0
- mean/std: 76340.8732 / 76879.65860782115
- quantiles: {'0.05': 3549.0, '0.25': 16783.5, '0.5': 54712.0, '0.75': 109519.0, '0.95': 217159.5}

## smoking_status_smoker_startTime_count

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 19.0
- mean/std: 0.0785 / 0.8415939775764368
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 0.0}

## nyha_nyha

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 4
- top values (shown only where count ≥ 5):
  - `LA28406-9`: 1752
  - `LA28405-1`: 1694
  - `LA28404-4`: 662
  - `LA28407-7`: 337

## nyha_nyha_pET

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 4
- top values (shown only where count ≥ 5):
  - `LA28406-9`: 1782
  - `LA28405-1`: 1643
  - `LA28404-4`: 537
  - `LA28407-7`: 483

## vital_signs_systolicBpDuringEncounter_value_pET

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 0 (0.00%)
- min/max: 41.0 / 220.0
- mean/std: 121.3406 / 22.468568188906428
- quantiles: {'0.05': 90.0, '0.25': 106.0, '0.5': 118.0, '0.75': 135.0, '0.95': 161.0}

## vital_signs_bmi_value_pET

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 0 (0.00%)
- min/max: 13.0 / 72.0
- mean/std: 25.7383 / 6.112361118004038
- quantiles: {'0.05': 18.592000000000002, '0.25': 21.7, '0.5': 24.49, '0.75': 28.5, '0.95': 37.42400000000001}

## lab_results_creatBS_value_p3a_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 0 (0.00%)
- min/max: 1.3 / 190.0
- mean/std: 15.4279 / 10.991683455577855
- quantiles: {'0.05': 6.583676666666666, '0.25': 9.498515000000001, '0.5': 12.441, '0.75': 17.44002, '0.95': 33.83369304818483}

## lab_results_validSerumCreatinine_value_pET

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.22 / 23.0
- mean/std: 12.9473 / 4.761819027399907
- quantiles: {'0.05': 6.064422, '0.25': 9.274199999999999, '0.5': 12.2148, '0.75': 16.3995, '0.95': 21.489}

## hyperkalemia_severity_categorizedValue

- dtype: `String` (categorical)
- nulls: 345 (7.76%)
- unique values: 5
- top values (shown only where count ≥ 5):
  - `normal`: 3785
  - null (missing): 345
  - `mild`: 213
  - `moderate`: 72
  - `severe`: 30

## ckd_severity_categorizedValue

- dtype: `String` (categorical)
- nulls: 309 (6.95%)
- unique values: 7
- top values (shown only where count ≥ 5):
  - `mildly_decreased`: 1170
  - `moderate_to_severe_decrease`: 795
  - `mild_to_moderate_decrease`: 731
  - `severe_decrease`: 659
  - `normal_or_high`: 540
  - null (missing): 309
  - `kidney_failure`: 241

## med_requests_activeDuringEncounter_bb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 2854
  - `False`: 1591

## med_requests_activeDuringEncounter_ace_inhibitors_arb_use_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 2433
  - `False`: 2012

## conditions_heartFailure_timeFromEarliest_first

- dtype: `Int64` (numeric)
- nulls: 13 (0.29%)
- min/max: 0.0 / 140.0
- mean/std: 13.0099 / 24.651255485588848
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 13.0, '0.95': 73.0}

## conditions_heart_failure_hf_within_18mo_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 4432
  - `False`: 13

## conditions_heart_failure_occurred_prior_to_18_months_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3431
  - `True`: 1014

## conditions_ap_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4357
  - `True`: 88

## conditions_af_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2954
  - `True`: 1491

## conditions_cm_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3875
  - `True`: 570

## conditions_dysl_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4029
  - `True`: 416

## conditions_hf_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 4319
  - `False`: 126

## conditions_hyp_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2872
  - `True`: 1573

## conditions_ihd_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3041
  - `True`: 1404

## conditions_mi_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4159
  - `True`: 286

## conditions_pad_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4022
  - `True`: 423

## conditions_stroke_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4296
  - `True`: 149

## conditions_tia_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4354
  - `True`: 91

## conditions_vd_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3080
  - `True`: 1365

## conditions_revasc_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4039
  - `True`: 406

## conditions_devices_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3759
  - `True`: 686

## conditions_aidshiv_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4424
  - `True`: 21

## conditions_copd_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3903
  - `True`: 542

## conditions_diabetes_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3352
  - `True`: 1093

## conditions_dem_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4361
  - `True`: 84

## conditions_dep_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4394
  - `True`: 51

## conditions_dia_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4445

## conditions_hyperthyroid_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4388
  - `True`: 57

## conditions_hypothyroid_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4291
  - `True`: 154

## conditions_ibd_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4378
  - `True`: 67

## conditions_ld_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4251
  - `True`: 194

## conditions_mc_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4035
  - `True`: 410

## conditions_osa_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4270
  - `True`: 175

## conditions_rd_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4300
  - `True`: 145

## conditions_ckd_chronic_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3176
  - `True`: 1269

## conditions_myocarditis_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4432
  - `True`: 13

## conditions_pericardial_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4385
  - `True`: 60

## conditions_substance_abuse_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4075
  - `True`: 370

## conditions_ap_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3798
  - `True`: 647

## conditions_af_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2275
  - `True`: 2170

## conditions_cm_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3554
  - `True`: 891

## conditions_dysl_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3623
  - `True`: 822

## conditions_hf_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 4255
  - `False`: 190

## conditions_hyp_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 2420
  - `False`: 2025

## conditions_ihd_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2363
  - `True`: 2082

## conditions_mi_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3732
  - `True`: 713

## conditions_pad_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3619
  - `True`: 826

## conditions_stroke_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4056
  - `True`: 389

## conditions_tia_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4169
  - `True`: 276

## conditions_vd_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2545
  - `True`: 1900

## conditions_revasc_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3714
  - `True`: 731

## conditions_devices_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3390
  - `True`: 1055

## conditions_aidshiv_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4403
  - `True`: 42

## conditions_copd_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3672
  - `True`: 773

## conditions_diabetes_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3019
  - `True`: 1426

## conditions_dem_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4343
  - `True`: 102

## conditions_dep_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4341
  - `True`: 104

## conditions_dia_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4445

## conditions_hyperthyroid_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4320
  - `True`: 125

## conditions_hypothyroid_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4119
  - `True`: 326

## conditions_ibd_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4243
  - `True`: 202

## conditions_ld_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4111
  - `True`: 334

## conditions_mc_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3568
  - `True`: 877

## conditions_osa_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4128
  - `True`: 317

## conditions_rd_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4204
  - `True`: 241

## conditions_ckd_chronic_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2676
  - `True`: 1769

## conditions_myocarditis_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4431
  - `True`: 14

## conditions_pericardial_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4276
  - `True`: 169

## conditions_substance_abuse_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3799
  - `True`: 646

## conditions_af_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2247
  - `True`: 2198

## conditions_ckd_chronic_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2654
  - `True`: 1791

## conditions_cm_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3547
  - `True`: 898

## conditions_copd_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3663
  - `True`: 782

## conditions_dem_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4341
  - `True`: 104

## conditions_dep_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4332
  - `True`: 113

## conditions_diabetes_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3014
  - `True`: 1431

## conditions_hypothyroid_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4119
  - `True`: 326

## conditions_hyp_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 2427
  - `False`: 2018

## conditions_ihd_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2340
  - `True`: 2105

## conditions_mc_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3554
  - `True`: 891

## conditions_mi_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3710
  - `True`: 735

## conditions_pad_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3602
  - `True`: 843

## conditions_rd_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4202
  - `True`: 243

## conditions_stroke_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4027
  - `True`: 418

## conditions_vd_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2525
  - `True`: 1920

## med_admins_rasi_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 2403
  - `False`: 2042

## med_admins_arni_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4100
  - `True`: 345

## med_admins_acei_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2846
  - `True`: 1599

## med_admins_arb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3494
  - `True`: 951

## med_admins_mra_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 2456
  - `False`: 1989

## med_admins_diuretics_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 4222
  - `False`: 223

## med_admins_diuretics_loop_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 4160
  - `False`: 285

## med_admins_anti_coag_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 3712
  - `False`: 733

## med_admins_anti_plat_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2619
  - `True`: 1826

## med_admins_thrombolytic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4418
  - `True`: 27

## med_admins_bb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 3076
  - `False`: 1369

## med_admins_ccb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3339
  - `True`: 1106

## med_admins_digitalis_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3354
  - `True`: 1091

## med_admins_antiarrhytmic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3711
  - `True`: 734

## med_admins_inotropes_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3484
  - `True`: 961

## med_admins_vasodil_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3378
  - `True`: 1067

## med_admins_platelet_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2590
  - `True`: 1855

## med_admins_ll_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 2244
  - `False`: 2201

## med_admins_ivabradine_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4337
  - `True`: 108

## med_admins_potassium_binders_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4264
  - `True`: 181

## med_admins_insulins_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3240
  - `True`: 1205

## med_admins_oral_antidiabetic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2896
  - `True`: 1549

## med_admins_sglt2i_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3546
  - `True`: 899

## med_admins_ari_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4445

## med_admins_rdoad_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2729
  - `True`: 1716

## med_admins_rdoad_syst_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4423
  - `True`: 22

## med_admins_cortico_syst_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3421
  - `True`: 1024

## med_admins_antiinfl_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4392
  - `True`: 53

## med_requests_rasi_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 2577
  - `False`: 1868

## med_requests_arni_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4068
  - `True`: 377

## med_requests_acei_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2736
  - `True`: 1709

## med_requests_arb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3399
  - `True`: 1046

## med_requests_mra_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 2547
  - `False`: 1898

## med_requests_diuretics_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 4241
  - `False`: 204

## med_requests_diuretics_loop_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 4184
  - `False`: 261

## med_requests_anti_coag_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4445

## med_requests_anti_plat_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4445

## med_requests_thrombolytic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4445

## med_requests_bb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 3007
  - `False`: 1438

## med_requests_ccb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4443
  - 1 other distinct value(s) suppressed, covering 2 row(s) (count below 5 and/or ranked beyond top 20)

## med_requests_digitalis_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4445

## med_requests_antiarrhytmic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4445

## med_requests_inotropes_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4445

## med_requests_vasodil_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4445

## med_requests_platelet_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4445

## med_requests_ll_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4445

## med_requests_ivabradine_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4445

## med_requests_potassium_binders_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4445

## med_requests_insulins_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4445

## med_requests_oral_antidiabetic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3509
  - `True`: 936

## med_requests_sglt2i_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3509
  - `True`: 936

## med_requests_ari_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4445

## med_requests_rdoad_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4445

## med_requests_rdoad_syst_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4445

## med_requests_cortico_syst_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4445

## med_requests_antiinfl_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4445

## med_admins_history_rasi_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2612
  - `True`: 1833

## med_admins_history_arni_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4202
  - `True`: 243

## med_admins_history_acei_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3188
  - `True`: 1257

## med_admins_history_arb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3643
  - `True`: 802

## med_admins_history_mra_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3093
  - `True`: 1352

## med_admins_history_diuretics_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 3462
  - `False`: 983

## med_admins_history_diuretics_loop_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 3400
  - `False`: 1045

## med_admins_history_anti_coag_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 3142
  - `False`: 1303

## med_admins_history_anti_plat_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2840
  - `True`: 1605

## med_admins_history_thrombolytic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4401
  - `True`: 44

## med_admins_history_bb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 2243
  - `False`: 2202

## med_admins_history_ccb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3002
  - `True`: 1443

## med_admins_history_digitalis_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3578
  - `True`: 867

## med_admins_history_antiarrhytmic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3913
  - `True`: 532

## med_admins_history_inotropes_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2945
  - `True`: 1500

## med_admins_history_vasodil_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3011
  - `True`: 1434

## med_admins_history_platelet_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2819
  - `True`: 1626

## med_admins_history_ll_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2433
  - `True`: 2012

## med_admins_history_ivabradine_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4333
  - `True`: 112

## med_admins_history_potassium_binders_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4223
  - `True`: 222

## med_admins_history_insulins_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3372
  - `True`: 1073

## med_admins_history_oral_antidiabetic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3374
  - `True`: 1071

## med_admins_history_sglt2i_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4064
  - `True`: 381

## med_admins_history_ari_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4445

## med_admins_history_rdoad_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2897
  - `True`: 1548

## med_admins_history_rdoad_syst_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4403
  - `True`: 42

## med_admins_history_cortico_syst_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3068
  - `True`: 1377

## med_admins_history_antiinfl_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4260
  - `True`: 185

## med_requests_history_rasi_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 2304
  - `False`: 2141

## med_requests_history_arni_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4140
  - `True`: 305

## med_requests_history_acei_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2788
  - `True`: 1657

## med_requests_history_arb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3390
  - `True`: 1055

## med_requests_history_mra_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2789
  - `True`: 1656

## med_requests_history_diuretics_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 3561
  - `False`: 884

## med_requests_history_diuretics_loop_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 3502
  - `False`: 943

## med_requests_history_anti_coag_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4445

## med_requests_history_anti_plat_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4445

## med_requests_history_thrombolytic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4445

## med_requests_history_bb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 2445
  - `False`: 2000

## med_requests_history_ccb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 4437
  - `True`: 8

## med_requests_history_digitalis_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4445

## med_requests_history_antiarrhytmic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4445

## med_requests_history_inotropes_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4445

## med_requests_history_vasodil_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4445

## med_requests_history_platelet_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4445

## med_requests_history_ll_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4445

## med_requests_history_ivabradine_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4445

## med_requests_history_potassium_binders_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4445

## med_requests_history_insulins_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4445

## med_requests_history_oral_antidiabetic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3987
  - `True`: 458

## med_requests_history_sglt2i_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 3987
  - `True`: 458

## med_requests_history_ari_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4445

## med_requests_history_rdoad_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4445

## med_requests_history_rdoad_syst_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4445

## med_requests_history_cortico_syst_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4445

## med_requests_history_antiinfl_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 4445

## encounter_primary_reason_HF_Disease_f5a_w7d_first

- dtype: `Boolean` (boolean)
- nulls: 3983 (89.61%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 3983
  - `False`: 418
  - `True`: 44

## encounter_primary_reason_HF_Disease_f5a_w1mo_first

- dtype: `Boolean` (boolean)
- nulls: 3325 (74.80%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 3325
  - `False`: 988
  - `True`: 132

## encounter_primary_reason_HF_Disease_f5a_w3mo_first

- dtype: `Boolean` (boolean)
- nulls: 2632 (59.21%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 2632
  - `False`: 1600
  - `True`: 213

## encounter_primary_reason_HF_Disease_f5a_w6mo_first

- dtype: `Boolean` (boolean)
- nulls: 2291 (51.54%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 2291
  - `False`: 1901
  - `True`: 253

## encounter_primary_reason_HF_Disease_f5a_w1a_first

- dtype: `Boolean` (boolean)
- nulls: 2073 (46.64%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2099
  - null (missing): 2073
  - `True`: 273

## encounter_primary_reason_HF_Disease_f5a_w3a_first

- dtype: `Boolean` (boolean)
- nulls: 1864 (41.93%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2289
  - null (missing): 1864
  - `True`: 292

## encounter_primary_reason_HF_Disease_f5a_w5a_first

- dtype: `Boolean` (boolean)
- nulls: 1820 (40.94%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2331
  - null (missing): 1820
  - `True`: 294

## encounter_primary_reason_number_of_days_to_rehosp_for_heart_failure_f5a_first

- dtype: `Int32` (numeric)
- nulls: 4151 (93.39%)
- min/max: 1.0 / 1900.0
- mean/std: 105.6803 / 200.86148883800305
- quantiles: {'0.05': 1.0, '0.25': 12.0, '0.5': 38.5, '0.75': 97.0, '0.95': 448.3499999999993}

## encounter_primary_reason_number_of_HF_rehospitalizations_5a_f5a_count

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 20.0
- mean/std: 0.3638 / 1.1068854651359008
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 2.0}

## encounter_primary_reason_CV_Disease_f5a_w7d_first

- dtype: `Boolean` (boolean)
- nulls: 3983 (89.61%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 3983
  - `False`: 286
  - `True`: 176

## encounter_primary_reason_CV_Disease_f5a_w1mo_first

- dtype: `Boolean` (boolean)
- nulls: 3325 (74.80%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 3325
  - `False`: 685
  - `True`: 435

## encounter_primary_reason_CV_Disease_f5a_w3mo_first

- dtype: `Boolean` (boolean)
- nulls: 2632 (59.21%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 2632
  - `False`: 1133
  - `True`: 680

## encounter_primary_reason_CV_Disease_f5a_w6mo_first

- dtype: `Boolean` (boolean)
- nulls: 2291 (51.54%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 2291
  - `False`: 1332
  - `True`: 822

## encounter_primary_reason_CV_Disease_f5a_w1a_first

- dtype: `Boolean` (boolean)
- nulls: 2073 (46.64%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 2073
  - `False`: 1476
  - `True`: 896

## encounter_primary_reason_CV_Disease_f5a_w3a_first

- dtype: `Boolean` (boolean)
- nulls: 1864 (41.93%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 1864
  - `False`: 1607
  - `True`: 974

## encounter_primary_reason_CV_Disease_f5a_w5a_first

- dtype: `Boolean` (boolean)
- nulls: 1820 (40.94%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 1820
  - `False`: 1637
  - `True`: 988

## encounter_primary_reason_number_of_days_to_rehosp_for_CV_f5a_first

- dtype: `Int32` (numeric)
- nulls: 3457 (77.77%)
- min/max: 0.0 / 1900.0
- mean/std: 125.6215 / 234.72688741791154
- quantiles: {'0.05': 1.0, '0.25': 12.0, '0.5': 39.0, '0.75': 117.25, '0.95': 597.65}

## encounter_primary_reason_number_of_CV_rehospitalizations_5a_f5a_count

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 37.0
- mean/std: 1.7444 / 3.3288714219527487
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 2.0, '0.95': 8.0}

## encounter_primary_reason_non_CV_Disease_f5a_w7d_first

- dtype: `Boolean` (boolean)
- nulls: 3983 (89.61%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 3983
  - `True`: 286
  - `False`: 176

## encounter_primary_reason_non_CV_Disease_f5a_w1mo_first

- dtype: `Boolean` (boolean)
- nulls: 3325 (74.80%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 3325
  - `True`: 685
  - `False`: 435

## encounter_primary_reason_non_CV_Disease_f5a_w3mo_first

- dtype: `Boolean` (boolean)
- nulls: 2632 (59.21%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 2632
  - `True`: 1133
  - `False`: 680

## encounter_primary_reason_non_CV_Disease_f5a_w6mo_first

- dtype: `Boolean` (boolean)
- nulls: 2291 (51.54%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 2291
  - `True`: 1332
  - `False`: 822

## encounter_primary_reason_non_CV_Disease_f5a_w1a_first

- dtype: `Boolean` (boolean)
- nulls: 2073 (46.64%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 2073
  - `True`: 1476
  - `False`: 896

## encounter_primary_reason_non_CV_Disease_f5a_w3a_first

- dtype: `Boolean` (boolean)
- nulls: 1864 (41.93%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 1864
  - `True`: 1607
  - `False`: 974

## encounter_primary_reason_non_CV_Disease_f5a_w5a_first

- dtype: `Boolean` (boolean)
- nulls: 1820 (40.94%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 1820
  - `True`: 1637
  - `False`: 988

## encounter_primary_reason_number_of_days_to_rehosp_for_non_CV_f5a_first

- dtype: `Int32` (numeric)
- nulls: 2808 (63.17%)
- min/max: 0.0 / 1800.0
- mean/std: 131.1698 / 240.29350277336016
- quantiles: {'0.05': 2.0, '0.25': 13.0, '0.5': 42.0, '0.75': 119.0, '0.95': 627.5999999999967}

## encounter_primary_reason_number_of_non_CV_rehospitalizations_5a_f5a_count

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 230.0
- mean/std: 5.1539 / 11.107047510509771
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 1.0, '0.75': 5.0, '0.95': 24.0}

## encounter_primary_reason_renal_complications_f5a_w7d_first

- dtype: `Boolean` (boolean)
- nulls: 3983 (89.61%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 3983
  - `False`: 458
  - 1 other distinct value(s) suppressed, covering 4 row(s) (count below 5 and/or ranked beyond top 20)

## encounter_primary_reason_renal_complications_f5a_w1mo_first

- dtype: `Boolean` (boolean)
- nulls: 3325 (74.80%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 3325
  - `False`: 1096
  - `True`: 24

## encounter_primary_reason_renal_complications_f5a_w3mo_first

- dtype: `Boolean` (boolean)
- nulls: 2632 (59.21%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 2632
  - `False`: 1770
  - `True`: 43

## encounter_primary_reason_renal_complications_f5a_w6mo_first

- dtype: `Boolean` (boolean)
- nulls: 2291 (51.54%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 2291
  - `False`: 2102
  - `True`: 52

## encounter_primary_reason_renal_complications_f5a_w1a_first

- dtype: `Boolean` (boolean)
- nulls: 2073 (46.64%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2312
  - null (missing): 2073
  - `True`: 60

## encounter_primary_reason_renal_complications_f5a_w3a_first

- dtype: `Boolean` (boolean)
- nulls: 1864 (41.93%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2517
  - null (missing): 1864
  - `True`: 64

## encounter_primary_reason_renal_complications_f5a_w5a_first

- dtype: `Boolean` (boolean)
- nulls: 1820 (40.94%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2561
  - null (missing): 1820
  - `True`: 64

## encounter_primary_reason_number_of_days_to_rehosp_for_renal_complications_f5a_first

- dtype: `Int32` (numeric)
- nulls: 4381 (98.56%)
- min/max: 3.0 / 720.0
- mean/std: 109.1406 / 146.6698990106479
- quantiles: {'0.05': 6.450000000000001, '0.25': 19.0, '0.5': 46.5, '0.75': 125.25, '0.95': 367.0999999999996}

## encounter_primary_reason_number_of_renal_rehospitalizations_5a_f5a_count

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 20.0
- mean/std: 0.2065 / 1.020096313759256
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 1.0}

## cause_of_death_isCV_f5a_w7d_first

- dtype: `Boolean` (boolean)
- nulls: 4308 (96.92%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 4308
  - `False`: 124
  - `True`: 13

## cause_of_death_isCV_f5a_w1mo_first

- dtype: `Boolean` (boolean)
- nulls: 4124 (92.78%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 4124
  - `False`: 304
  - `True`: 17

## cause_of_death_isCV_f5a_w3mo_first

- dtype: `Boolean` (boolean)
- nulls: 3887 (87.45%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 3887
  - `False`: 534
  - `True`: 24

## cause_of_death_isCV_f5a_w6mo_first

- dtype: `Boolean` (boolean)
- nulls: 3675 (82.68%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 3675
  - `False`: 738
  - `True`: 32

## cause_of_death_isCV_f5a_w1a_first

- dtype: `Boolean` (boolean)
- nulls: 3459 (77.82%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 3459
  - `False`: 947
  - `True`: 39

## cause_of_death_isCV_f5a_w3a_first

- dtype: `Boolean` (boolean)
- nulls: 3086 (69.43%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 3086
  - `False`: 1302
  - `True`: 57

## cause_of_death_isCV_f5a_w5a_first

- dtype: `Boolean` (boolean)
- nulls: 2926 (65.83%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 2926
  - `False`: 1457
  - `True`: 62

## cause_of_death_number_of_days_to_death_for_CV_f5a_first

- dtype: `Int32` (numeric)
- nulls: 4383 (98.61%)
- min/max: 1.0 / 1700.0
- mean/std: 361.9839 / 433.6220493734177
- quantiles: {'0.05': 1.0, '0.25': 23.25, '0.5': 166.5, '0.75': 610.75, '0.95': 1352.2999999999995}

## cause_of_death_isRenal_f5a_w7d_first

- dtype: `Boolean` (boolean)
- nulls: 4308 (96.92%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 4308
  - `False`: 137

## cause_of_death_isRenal_f5a_w1mo_first

- dtype: `Boolean` (boolean)
- nulls: 4124 (92.78%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 4124
  - `False`: 321

## cause_of_death_isRenal_f5a_w3mo_first

- dtype: `Boolean` (boolean)
- nulls: 3887 (87.45%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 3887
  - `False`: 558

## cause_of_death_isRenal_f5a_w6mo_first

- dtype: `Boolean` (boolean)
- nulls: 3675 (82.68%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 3675
  - `False`: 770

## cause_of_death_isRenal_f5a_w1a_first

- dtype: `Boolean` (boolean)
- nulls: 3459 (77.82%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 3459
  - `False`: 986

## cause_of_death_isRenal_f5a_w3a_first

- dtype: `Boolean` (boolean)
- nulls: 3086 (69.43%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 3086
  - `False`: 1359

## cause_of_death_isRenal_f5a_w5a_first

- dtype: `Boolean` (boolean)
- nulls: 2926 (65.83%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 2926
  - `False`: 1519

## cause_of_death_number_of_days_to_death_for_renal_f5a_first

- dtype: `Int32` (numeric)
- nulls: 4445 (100.00%)
- All values are null.

## cause_of_death_isNonRenalAndNonCV_f5a_w7d_first

- dtype: `Boolean` (boolean)
- nulls: 4308 (96.92%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 4308
  - `False`: 137

## cause_of_death_isNonRenalAndNonCV_f5a_w1mo_first

- dtype: `Boolean` (boolean)
- nulls: 4124 (92.78%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 4124
  - `False`: 321

## cause_of_death_isNonRenalAndNonCV_f5a_w3mo_first

- dtype: `Boolean` (boolean)
- nulls: 3887 (87.45%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 3887
  - `False`: 558

## cause_of_death_isNonRenalAndNonCV_f5a_w6mo_first

- dtype: `Boolean` (boolean)
- nulls: 3675 (82.68%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 3675
  - `False`: 770

## cause_of_death_isNonRenalAndNonCV_f5a_w1a_first

- dtype: `Boolean` (boolean)
- nulls: 3459 (77.82%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 3459
  - `False`: 986

## cause_of_death_isNonRenalAndNonCV_f5a_w3a_first

- dtype: `Boolean` (boolean)
- nulls: 3086 (69.43%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 3086
  - `False`: 1359

## cause_of_death_isNonRenalAndNonCV_f5a_w5a_first

- dtype: `Boolean` (boolean)
- nulls: 2926 (65.83%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 2926
  - `False`: 1519

## cause_of_death_number_of_days_to_death_for_non_renal_and_non_CV_f5a_first

- dtype: `Int32` (numeric)
- nulls: 4445 (100.00%)
- All values are null.

## cause_of_death_isAllCause_f5a_w7d_first

- dtype: `Boolean` (boolean)
- nulls: 4308 (96.92%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 4308
  - `True`: 124
  - `False`: 13

## cause_of_death_isAllCause_f5a_w1mo_first

- dtype: `Boolean` (boolean)
- nulls: 4124 (92.78%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 4124
  - `True`: 304
  - `False`: 17

## cause_of_death_isAllCause_f5a_w3mo_first

- dtype: `Boolean` (boolean)
- nulls: 3887 (87.45%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 3887
  - `True`: 534
  - `False`: 24

## cause_of_death_isAllCause_f5a_w6mo_first

- dtype: `Boolean` (boolean)
- nulls: 3675 (82.68%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 3675
  - `True`: 738
  - `False`: 32

## cause_of_death_isAllCause_f5a_w1a_first

- dtype: `Boolean` (boolean)
- nulls: 3459 (77.82%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 3459
  - `True`: 947
  - `False`: 39

## cause_of_death_isAllCause_f5a_w3a_first

- dtype: `Boolean` (boolean)
- nulls: 3086 (69.43%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 3086
  - `True`: 1302
  - `False`: 57

## cause_of_death_isAllCause_f5a_w5a_first

- dtype: `Boolean` (boolean)
- nulls: 2926 (65.83%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 2926
  - `True`: 1457
  - `False`: 62

## cause_of_death_number_of_days_to_death_for_all_cause_f5a_first

- dtype: `Int32` (numeric)
- nulls: 2988 (67.22%)
- min/max: 1.0 / 1900.0
- mean/std: 377.7865 / 444.7742841991265
- quantiles: {'0.05': 4.0, '0.25': 42.0, '0.5': 174.0, '0.75': 584.0, '0.95': 1371.2000000000003}

## eGFR_2021_ckd_epi_creatinine

- dtype: `Decimal(precision=38, scale=6)` (numeric)
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

- dtype: `Int32` (numeric)
- nulls: 3528 (79.37%)
- min/max: 3.0 / 44.0
- mean/std: 23.3937 / 7.362782850577475
- quantiles: {'0.05': 10.800000000000004, '0.25': 18.0, '0.5': 24.0, '0.75': 29.0, '0.95': 35.0}
