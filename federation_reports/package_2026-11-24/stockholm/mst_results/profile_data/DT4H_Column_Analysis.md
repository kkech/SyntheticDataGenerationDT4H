# Column Analysis

Total rows: 802
Total columns: 540

## pid

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 802
- ⚠️ high-cardinality column (802 distinct values)
- top values (shown only where count ≥ 5):
  - (none met the display threshold)
  - 802 other distinct value(s) suppressed, covering 802 row(s) (count below 5 and/or ranked beyond top 20)

## encounterId

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 802
- ⚠️ high-cardinality column (802 distinct values)
- top values (shown only where count ≥ 5):
  - (none met the display threshold)
  - 802 other distinct value(s) suppressed, covering 802 row(s) (count below 5 and/or ranked beyond top 20)

## referenceTimePoint

- dtype: `Datetime(time_unit='ns', time_zone=None)` (categorical)
- nulls: 0 (0.00%)
- unique values: 704
- ⚠️ high-cardinality column (704 distinct values)
- top values (shown only where count ≥ 5):
  - (none met the display threshold)
  - 704 other distinct value(s) suppressed, covering 802 row(s) (count below 5 and/or ranked beyond top 20)

## eventTime

- dtype: `Datetime(time_unit='ns', time_zone=None)` (categorical)
- nulls: 0 (0.00%)
- unique values: 713
- ⚠️ high-cardinality column (713 distinct values)
- top values (shown only where count ≥ 5):
  - (none met the display threshold)
  - 713 other distinct value(s) suppressed, covering 802 row(s) (count below 5 and/or ranked beyond top 20)

## exitTime

- dtype: `Datetime(time_unit='ns', time_zone=None)` (categorical)
- nulls: 802 (100.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 802

## patient_demographics_sourceIdentifier

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 802
- ⚠️ high-cardinality column (802 distinct values)
- top values (shown only where count ≥ 5):
  - (none met the display threshold)
  - 802 other distinct value(s) suppressed, covering 802 row(s) (count below 5 and/or ranked beyond top 20)

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

## encounters_admissionDate

- dtype: `Datetime(time_unit='ns', time_zone=None)` (categorical)
- nulls: 0 (0.00%)
- unique values: 713
- ⚠️ high-cardinality column (713 distinct values)
- top values (shown only where count ≥ 5):
  - (none met the display threshold)
  - 713 other distinct value(s) suppressed, covering 802 row(s) (count below 5 and/or ranked beyond top 20)

## encounters_dischargeDate

- dtype: `Datetime(time_unit='ns', time_zone=None)` (categorical)
- nulls: 0 (0.00%)
- unique values: 704
- ⚠️ high-cardinality column (704 distinct values)
- top values (shown only where count ≥ 5):
  - (none met the display threshold)
  - 704 other distinct value(s) suppressed, covering 802 row(s) (count below 5 and/or ranked beyond top 20)

## encounters_numOfPreviousHFStays_count

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 6800.0
- mean/std: 91.9863 / 289.9219810871747
- quantiles: {'0.05': 0.0, '0.25': 12.25, '0.5': 41.0, '0.75': 97.0, '0.95': 299.8499999999998}

## vital_signs_weight_value_p6mo_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 223 (27.81%)
- min/max: 10.0 / 230.0
- mean/std: 78.4655 / 21.100180188460804
- quantiles: {'0.05': 49.47, '0.25': 64.75, '0.5': 75.2, '0.75': 89.05, '0.95': 117.8}

## vital_signs_weight_value_p6mo_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 223 (27.81%)
- min/max: 35.0 / 240.0
- mean/std: 81.8431 / 21.66728380342932
- quantiles: {'0.05': 52.59, '0.25': 67.35, '0.5': 79.0, '0.75': 92.0, '0.95': 122.16000000000004}

## vital_signs_weight_value_p6mo_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 223 (27.81%)
- min/max: 37.0 / 240.0
- mean/std: 83.7120 / 22.455257477187164
- quantiles: {'0.05': 52.9, '0.25': 69.0, '0.5': 81.0, '0.75': 94.65, '0.95': 126.52}

## vital_signs_weight_value_p6mo_stddev

- dtype: `Float64` (numeric)
- nulls: 223 (27.81%)
- min/max: 0.0 / 28.0
- mean/std: 1.7320 / 2.8450404835324177
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.9662150012142363, '0.75': 2.1526826991280883, '0.95': 5.998940057315023}

## vital_signs_weight_value_p6mo_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 223 (27.81%)
- min/max: 37.0 / 230.0
- mean/std: 80.9429 / 21.04515726104574
- quantiles: {'0.05': 51.682857142857145, '0.25': 66.875, '0.5': 77.93333333333332, '0.75': 91.57954545454545, '0.95': 120.8425}

## vital_signs_weight_value_p6mo_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 223 (27.81%)
- min/max: 37.0 / 230.0
- mean/std: 80.1956 / 21.06299503825875
- quantiles: {'0.05': 50.0, '0.25': 66.7, '0.5': 77.4, '0.75': 90.45, '0.95': 119.55000000000001}

## vital_signs_height_value_p1a_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 411 (51.25%)
- min/max: 51.0 / 200.0
- mean/std: 170.6547 / 18.164958830734115
- quantiles: {'0.05': 150.0, '0.25': 165.0, '0.5': 174.0, '0.75': 180.0, '0.95': 188.5}

## vital_signs_weight_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 312 (38.90%)
- min/max: 37.0 / 230.0
- mean/std: 80.3711 / 21.216395003344484
- quantiles: {'0.05': 49.78, '0.25': 67.0, '0.5': 77.1, '0.75': 90.875, '0.95': 120.38499999999996}

## vital_signs_height_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 542 (67.58%)
- min/max: 46.0 / 200.0
- mean/std: 169.8219 / 22.361453755248426
- quantiles: {'0.05': 150.0, '0.25': 165.0, '0.5': 174.5, '0.75': 180.0, '0.95': 189.04999999999995}

## vital_signs_diastolicBp_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 60 (7.48%)
- min/max: 32.0 / 130.0
- mean/std: 71.4717 / 12.65982336949446
- quantiles: {'0.05': 54.0, '0.25': 60.0, '0.5': 70.0, '0.75': 80.0, '0.95': 90.0}

## vital_signs_diastolicBp_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 60 (7.48%)
- min/max: 0.0 / 110.0
- mean/std: 61.0914 / 13.644010950409845
- quantiles: {'0.05': 40.0, '0.25': 53.0, '0.5': 60.0, '0.75': 70.0, '0.95': 83.0}

## vital_signs_diastolicBp_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 60 (7.48%)
- min/max: 32.0 / 160.0
- mean/std: 73.6706 / 11.957717186778837
- quantiles: {'0.05': 58.26964285714286, '0.25': 65.38846153846154, '0.5': 72.32467532467533, '0.75': 80.0, '0.95': 92.75317460317457}

## vital_signs_diastolicBp_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 60 (7.48%)
- min/max: 32.0 / 180.0
- mean/std: 87.9798 / 17.000781785846137
- quantiles: {'0.05': 65.0, '0.25': 80.0, '0.5': 86.5, '0.75': 97.0, '0.95': 117.89999999999986}

## vital_signs_diastolicBp_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 60 (7.48%)
- min/max: 32.0 / 170.0
- mean/std: 77.7776 / 17.741368379891334
- quantiles: {'0.05': 55.0, '0.25': 65.0, '0.5': 76.0, '0.75': 88.0, '0.95': 110.0}

## vital_signs_diastolicBp_value_stddev

- dtype: `Float64` (numeric)
- nulls: 60 (7.48%)
- min/max: 0.0 / 34.0
- mean/std: 7.8371 / 4.884368132352902
- quantiles: {'0.05': 0.0, '0.25': 5.006234452640111, '0.5': 7.786750496135033, '0.75': 10.463298117176004, '0.95': 16.402828342744492}

## vital_signs_heartRate_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 59 (7.36%)
- min/max: 37.0 / 170.0
- mean/std: 76.3863 / 16.926993226624177
- quantiles: {'0.05': 54.0, '0.25': 64.0, '0.5': 74.0, '0.75': 87.0, '0.95': 103.0}

## vital_signs_heartRate_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 59 (7.36%)
- min/max: 0.0 / 170.0
- mean/std: 65.3011 / 16.29252714859293
- quantiles: {'0.05': 44.1, '0.25': 56.0, '0.5': 63.0, '0.75': 73.0, '0.95': 90.0}

## vital_signs_heartRate_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 59 (7.36%)
- min/max: 37.0 / 210.0
- mean/std: 80.4898 / 16.999814333711157
- quantiles: {'0.05': 57.34, '0.25': 69.20833333333333, '0.5': 78.625, '0.75': 89.95454545454545, '0.95': 107.59384615384616}

## vital_signs_heartRate_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 59 (7.36%)
- min/max: 37.0 / 220.0
- mean/std: 97.5895 / 25.270232163143532
- quantiles: {'0.05': 62.1, '0.25': 80.0, '0.5': 95.0, '0.75': 112.0, '0.95': 140.0}

## vital_signs_heartRate_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 59 (7.36%)
- min/max: 25.0 / 220.0
- mean/std: 87.3674 / 25.81962864637507
- quantiles: {'0.05': 53.1, '0.25': 69.0, '0.5': 84.0, '0.75': 100.5, '0.95': 131.89999999999998}

## vital_signs_heartRate_value_stddev

- dtype: `Float64` (numeric)
- nulls: 59 (7.36%)
- min/max: 0.0 / 59.0
- mean/std: 9.6373 / 7.70364264261717
- quantiles: {'0.05': 0.0, '0.25': 4.237314331158748, '0.5': 8.5, '0.75': 13.08363015413926, '0.95': 24.354041671197944}

## vital_signs_systolicBp_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 59 (7.36%)
- min/max: 69.0 / 230.0
- mean/std: 122.7954 / 22.48566193174436
- quantiles: {'0.05': 90.0, '0.25': 110.0, '0.5': 120.0, '0.75': 135.0, '0.95': 160.89999999999998}

## vital_signs_systolicBp_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 59 (7.36%)
- min/max: 0.0 / 220.0
- mean/std: 107.8829 / 22.175871790087392
- quantiles: {'0.05': 75.0, '0.25': 93.5, '0.5': 107.0, '0.75': 120.0, '0.95': 143.0}

## vital_signs_systolicBp_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 59 (7.36%)
- min/max: 76.0 / 220.0
- mean/std: 126.4779 / 21.502536451520793
- quantiles: {'0.05': 95.0, '0.25': 111.16666666666667, '0.5': 125.45454545454545, '0.75': 137.25, '0.95': 166.09711538461536}

## vital_signs_systolicBp_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 59 (7.36%)
- min/max: 77.0 / 280.0
- mean/std: 147.6716 / 29.14472371129382
- quantiles: {'0.05': 110.0, '0.25': 129.0, '0.5': 143.0, '0.75': 161.0, '0.95': 203.5999999999999}

## vital_signs_systolicBp_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 59 (7.36%)
- min/max: 64.0 / 260.0
- mean/std: 132.4293 / 29.3190011426233
- quantiles: {'0.05': 90.0, '0.25': 113.5, '0.5': 130.0, '0.75': 147.0, '0.95': 185.0}

## vital_signs_systolicBp_value_stddev

- dtype: `Float64` (numeric)
- nulls: 59 (7.36%)
- min/max: 0.0 / 49.0
- mean/std: 11.7411 / 7.611963180694697
- quantiles: {'0.05': 0.0, '0.25': 7.038642113211075, '0.5': 11.366067921669309, '0.75': 15.737622710411998, '0.95': 24.845320448187103}

## vital_signs_oxygenSaturation_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 80 (9.98%)
- min/max: 85.0 / 100.0
- mean/std: 97.6607 / 2.043175739499103
- quantiles: {'0.05': 94.0, '0.25': 97.0, '0.5': 98.0, '0.75': 99.0, '0.95': 99.0}

## vital_signs_oxygenSaturation_value_stddev

- dtype: `Float64` (numeric)
- nulls: 80 (9.98%)
- min/max: 0.0 / 37.0
- mean/std: 2.1693 / 2.513566913059602
- quantiles: {'0.05': 0.0, '0.25': 1.0, '0.5': 1.7524103605608463, '0.75': 2.627373716819264, '0.95': 5.469284170791104}

## vital_signs_oxygenSaturation_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 80 (9.98%)
- min/max: 6.0 / 100.0
- mean/std: 94.5360 / 5.474130778809627
- quantiles: {'0.05': 86.05, '0.25': 93.0, '0.5': 96.0, '0.75': 98.0, '0.95': 99.0}

## vital_signs_oxygenSaturation_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 80 (9.98%)
- min/max: 68.0 / 100.0
- mean/std: 95.5249 / 3.3105662903829565
- quantiles: {'0.05': 90.0, '0.25': 94.0, '0.5': 96.0, '0.75': 98.0, '0.95': 99.0}

## vital_signs_oxygenSaturation_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 80 (9.98%)
- min/max: 4.0 / 99.0
- mean/std: 90.2271 / 10.159937908830281
- quantiles: {'0.05': 78.0, '0.25': 89.25, '0.5': 92.0, '0.75': 95.0, '0.95': 98.0}

## vital_signs_oxygenSaturation_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 80 (9.98%)
- min/max: 76.0 / 99.0
- mean/std: 95.0638 / 2.789730068003351
- quantiles: {'0.05': 89.75535714285715, '0.25': 93.83030303030303, '0.5': 95.71428571428572, '0.75': 97.0, '0.95': 98.41527777777777}

## lab_results_hemoglobin_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 75 (9.35%)
- min/max: 78.0 / 190.0
- mean/std: 131.9147 / 18.858714674380348
- quantiles: {'0.05': 102.0, '0.25': 118.0, '0.5': 133.0, '0.75': 146.0, '0.95': 162.0}

## lab_results_hemoglobin_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 75 (9.35%)
- min/max: 45.0 / 190.0
- mean/std: 129.8473 / 19.951734193825065
- quantiles: {'0.05': 96.0, '0.25': 116.0, '0.5': 131.0, '0.75': 144.0, '0.95': 161.0}

## lab_results_hemoglobin_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 75 (9.35%)
- min/max: 37.0 / 190.0
- mean/std: 123.6424 / 21.025074074766376
- quantiles: {'0.05': 87.0, '0.25': 110.0, '0.5': 125.0, '0.75': 138.0, '0.95': 155.0}

## lab_results_hemoglobin_value_stddev

- dtype: `Float64` (numeric)
- nulls: 75 (9.35%)
- min/max: 0.0 / 29.0
- mean/std: 3.0256 / 4.343777309385979
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 5.471469461692411, '0.95': 11.241931958101883}

## lab_results_hemoglobin_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 75 (9.35%)
- min/max: 75.0 / 190.0
- mean/std: 126.6176 / 18.91869273389749
- quantiles: {'0.05': 95.0, '0.25': 113.0, '0.5': 127.0, '0.75': 140.0, '0.95': 156.69999999999993}

## lab_results_hemoglobin_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 75 (9.35%)
- min/max: 65.0 / 190.0
- mean/std: 127.4425 / 18.92604294027919
- quantiles: {'0.05': 95.0, '0.25': 114.875, '0.5': 128.5, '0.75': 141.0, '0.95': 157.0}

## lab_results_ferritin_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 750 (93.52%)
- min/max: 12.0 / 1600.0
- mean/std: 334.5962 / 329.62678001425206
- quantiles: {'0.05': 26.65, '0.25': 98.25, '0.5': 258.0, '0.75': 427.0, '0.95': 989.7499999999995}

## lab_results_ferritin_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 750 (93.52%)
- min/max: 12.0 / 1300.0
- mean/std: 320.6923 / 305.5183020390835
- quantiles: {'0.05': 26.65, '0.25': 98.25, '0.5': 231.0, '0.75': 421.0, '0.95': 989.7499999999995}

## lab_results_ferritin_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 750 (93.52%)
- min/max: 12.0 / 1300.0
- mean/std: 315.7692 / 299.71670384565255
- quantiles: {'0.05': 26.65, '0.25': 98.25, '0.5': 221.5, '0.75': 421.0, '0.95': 989.7499999999995}

## lab_results_ferritin_value_stddev

- dtype: `Float64` (numeric)
- nulls: 750 (93.52%)
- min/max: 0.0 / 190.0
- mean/std: 8.5859 / 30.596581957226878
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 52.115448039790365}

## lab_results_ferritin_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 750 (93.52%)
- min/max: 12.0 / 1600.0
- mean/std: 329.0577 / 324.32683911005444
- quantiles: {'0.05': 26.65, '0.25': 98.25, '0.5': 258.0, '0.75': 427.0, '0.95': 989.7499999999995}

## lab_results_ferritin_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 750 (93.52%)
- min/max: 12.0 / 1400.0
- mean/std: 324.8109 / 313.554782037545
- quantiles: {'0.05': 26.650000000000002, '0.25': 98.25, '0.5': 242.5, '0.75': 427.0, '0.95': 989.7499999999995}

## lab_results_tfs_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 759 (94.64%)
- min/max: 4.0 / 87.0
- mean/std: 17.4419 / 13.865514476367336
- quantiles: {'0.05': 5.0, '0.25': 9.0, '0.5': 15.0, '0.75': 21.5, '0.95': 35.8}

## lab_results_tfs_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 759 (94.64%)
- min/max: 4.0 / 87.0
- mean/std: 17.0465 / 13.873499040860828
- quantiles: {'0.05': 5.0, '0.25': 9.0, '0.5': 14.0, '0.75': 21.0, '0.95': 35.8}

## lab_results_tfs_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 759 (94.64%)
- min/max: 4.0 / 87.0
- mean/std: 17.0465 / 13.873499040860828
- quantiles: {'0.05': 5.0, '0.25': 9.0, '0.5': 14.0, '0.75': 21.0, '0.95': 35.8}

## lab_results_tfs_value_stddev

- dtype: `Float64` (numeric)
- nulls: 759 (94.64%)
- min/max: 0.0 / 5.0
- mean/std: 0.1725 / 0.8332063930561295
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 0.0}

## lab_results_tfs_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 759 (94.64%)
- min/max: 4.0 / 87.0
- mean/std: 17.4419 / 13.865514476367336
- quantiles: {'0.05': 5.0, '0.25': 9.0, '0.5': 15.0, '0.75': 21.5, '0.95': 35.8}

## lab_results_tfs_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 759 (94.64%)
- min/max: 4.0 / 87.0
- mean/std: 17.2500 / 13.835018543364447
- quantiles: {'0.05': 5.0, '0.25': 9.0, '0.5': 14.5, '0.75': 21.0, '0.95': 35.8}

## lab_results_ntProBnp_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 357 (44.51%)
- min/max: 27.0 / 35000.0
- mean/std: 7609.0854 / 9091.796668886038
- quantiles: {'0.05': 302.4, '0.25': 1490.0, '0.5': 4200.0, '0.75': 9770.0, '0.95': 31519.999999999938}

## lab_results_ntProBnp_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 357 (44.51%)
- min/max: 27.0 / 35000.0
- mean/std: 7056.9663 / 8679.085530213026
- quantiles: {'0.05': 281.20000000000005, '0.25': 1440.0, '0.5': 3890.0, '0.75': 8570.0, '0.95': 29100.0}

## lab_results_ntProBnp_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 357 (44.51%)
- min/max: 27.0 / 35000.0
- mean/std: 6290.7371 / 8086.782447976256
- quantiles: {'0.05': 241.80000000000004, '0.25': 1300.0, '0.5': 3220.0, '0.75': 7240.0, '0.95': 26999.999999999978}

## lab_results_ntProBnp_value_stddev

- dtype: `Float64` (numeric)
- nulls: 357 (44.51%)
- min/max: 0.0 / 13000.0
- mean/std: 578.5019 / 1637.5905885459238
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 78.0, '0.95': 4030.314154124429}

## lab_results_ntProBnp_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 357 (44.51%)
- min/max: 27.0 / 35000.0
- mean/std: 6734.5169 / 8425.618403150216
- quantiles: {'0.05': 284.6000000000001, '0.25': 1350.0, '0.5': 3600.0, '0.75': 7910.000000000001, '0.95': 27579.999999999996}

## lab_results_ntProBnp_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 357 (44.51%)
- min/max: 27.0 / 35000.0
- mean/std: 6954.1851 / 8439.63010737663
- quantiles: {'0.05': 302.4, '0.25': 1460.0, '0.5': 3940.0000000000005, '0.75': 8195.0, '0.95': 27496.666666666664}

## lab_results_bnp_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 802 (100.00%)
- All values are null.

## lab_results_bnp_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 802 (100.00%)
- All values are null.

## lab_results_bnp_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 802 (100.00%)
- All values are null.

## lab_results_bnp_value_stddev

- dtype: `Float64` (numeric)
- nulls: 802 (100.00%)
- All values are null.

## lab_results_bnp_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 802 (100.00%)
- All values are null.

## lab_results_bnp_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 802 (100.00%)
- All values are null.

## lab_results_crpNonHs_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 123 (15.34%)
- min/max: 1.0 / 450.0
- mean/std: 47.6786 / 73.29469671738201
- quantiles: {'0.05': 1.0, '0.25': 5.0, '0.5': 16.0, '0.75': 58.0, '0.95': 229.10000000000002}

## lab_results_crpNonHs_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 123 (15.34%)
- min/max: 1.0 / 380.0
- mean/std: 30.1116 / 47.94900773018018
- quantiles: {'0.05': 1.0, '0.25': 3.0, '0.5': 11.0, '0.75': 34.5, '0.95': 126.20000000000005}

## lab_results_crpNonHs_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 123 (15.34%)
- min/max: 1.0 / 290.0
- mean/std: 22.0940 / 33.83397005757584
- quantiles: {'0.05': 1.0, '0.25': 3.0, '0.5': 10.0, '0.75': 26.0, '0.95': 81.10000000000002}

## lab_results_crpNonHs_value_stddev

- dtype: `Float64` (numeric)
- nulls: 123 (15.34%)
- min/max: 0.0 / 140.0
- mean/std: 9.2291 / 20.608224870370716
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 7.090629434079952, '0.95': 55.14085197969207}

## lab_results_crpNonHs_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 123 (15.34%)
- min/max: 1.0 / 350.0
- mean/std: 30.2736 / 45.08007294327075
- quantiles: {'0.05': 1.0, '0.25': 4.0, '0.5': 13.0, '0.75': 38.0, '0.95': 109.0}

## lab_results_crpNonHs_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 123 (15.34%)
- min/max: 1.0 / 320.0
- mean/std: 33.3691 / 45.08500389862265
- quantiles: {'0.05': 1.0, '0.25': 4.0, '0.5': 14.0, '0.75': 46.58333333333333, '0.95': 125.12500000000003}

## lab_results_crpHs_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 801 (99.88%)
- min/max: 14.0 / 15.0
- mean/std: 14.8000 / None
- quantiles: {'0.05': 14.8, '0.25': 14.8, '0.5': 14.8, '0.75': 14.8, '0.95': 14.8}
- ⚠️ constant column (single value)

## lab_results_crpHs_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 801 (99.88%)
- min/max: 14.0 / 15.0
- mean/std: 14.8000 / None
- quantiles: {'0.05': 14.8, '0.25': 14.8, '0.5': 14.8, '0.75': 14.8, '0.95': 14.8}
- ⚠️ constant column (single value)

## lab_results_crpHs_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 801 (99.88%)
- min/max: 14.0 / 15.0
- mean/std: 14.8000 / None
- quantiles: {'0.05': 14.8, '0.25': 14.8, '0.5': 14.8, '0.75': 14.8, '0.95': 14.8}
- ⚠️ constant column (single value)

## lab_results_crpHs_value_stddev

- dtype: `Float64` (numeric)
- nulls: 801 (99.88%)
- min/max: 0.0 / 0.0
- mean/std: 0.0000 / None
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 0.0}
- ⚠️ constant column (single value)

## lab_results_crpHs_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 801 (99.88%)
- min/max: 14.0 / 15.0
- mean/std: 14.8000 / None
- quantiles: {'0.05': 14.8, '0.25': 14.8, '0.5': 14.8, '0.75': 14.8, '0.95': 14.8}
- ⚠️ constant column (single value)

## lab_results_crpHs_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 801 (99.88%)
- min/max: 14.0 / 15.0
- mean/std: 14.8000 / None
- quantiles: {'0.05': 14.799999999999999, '0.25': 14.799999999999999, '0.5': 14.799999999999999, '0.75': 14.799999999999999, '0.95': 14.799999999999999}
- ⚠️ constant column (single value)

## lab_results_tropIHs_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 802 (100.00%)
- All values are null.

## lab_results_tropIHs_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 802 (100.00%)
- All values are null.

## lab_results_tropIHs_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 802 (100.00%)
- All values are null.

## lab_results_tropIHs_value_stddev

- dtype: `Float64` (numeric)
- nulls: 802 (100.00%)
- All values are null.

## lab_results_tropIHs_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 802 (100.00%)
- All values are null.

## lab_results_tropIHs_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 802 (100.00%)
- All values are null.

## lab_results_tropInHs_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 802 (100.00%)
- All values are null.

## lab_results_tropInHs_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 802 (100.00%)
- All values are null.

## lab_results_tropInHs_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 802 (100.00%)
- All values are null.

## lab_results_tropInHs_value_stddev

- dtype: `Float64` (numeric)
- nulls: 802 (100.00%)
- All values are null.

## lab_results_tropInHs_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 802 (100.00%)
- All values are null.

## lab_results_tropInHs_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 802 (100.00%)
- All values are null.

## lab_results_tropTHs_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 802 (100.00%)
- All values are null.

## lab_results_tropTHs_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 802 (100.00%)
- All values are null.

## lab_results_tropTHs_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 802 (100.00%)
- All values are null.

## lab_results_tropTHs_value_stddev

- dtype: `Float64` (numeric)
- nulls: 802 (100.00%)
- All values are null.

## lab_results_tropTHs_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 802 (100.00%)
- All values are null.

## lab_results_tropTHs_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 802 (100.00%)
- All values are null.

## lab_results_tropTnHs_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 268 (33.42%)
- min/max: 0.34 / 35000.0
- mean/std: 339.6392 / 1784.0062765469338
- quantiles: {'0.05': 10.0, '0.25': 25.0, '0.5': 48.5, '0.75': 124.75, '0.95': 1403.9999999999986}

## lab_results_tropTnHs_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 268 (33.42%)
- min/max: 0.34 / 35000.0
- mean/std: 225.3901 / 1608.3105499872693
- quantiles: {'0.05': 9.0, '0.25': 22.0, '0.5': 40.0, '0.75': 79.75, '0.95': 449.39999999999924}

## lab_results_tropTnHs_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 268 (33.42%)
- min/max: 0.06 / 7800.0
- mean/std: 142.7304 / 537.6754045042447
- quantiles: {'0.05': 9.0, '0.25': 21.25, '0.5': 38.0, '0.75': 70.75, '0.95': 381.7999999999997}

## lab_results_tropTnHs_value_stddev

- dtype: `Float64` (numeric)
- nulls: 268 (33.42%)
- min/max: 0.0 / 7700.0
- mean/std: 66.2693 / 414.93061859614625
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 1.0, '0.75': 10.148529270389176, '0.95': 237.00259018338448}

## lab_results_tropTnHs_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 268 (33.42%)
- min/max: 0.06 / 7800.0
- mean/std: 213.8372 / 669.405329846287
- quantiles: {'0.05': 9.0, '0.25': 24.25, '0.5': 46.0, '0.75': 99.0, '0.95': 1083.9999999999986}

## lab_results_tropTnHs_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 268 (33.42%)
- min/max: 0.15 / 11000.0
- mean/std: 222.4043 / 791.296022435709
- quantiles: {'0.05': 9.441666666666666, '0.25': 24.5, '0.5': 46.0, '0.75': 104.25, '0.95': 858.3933333333322}

## lab_results_triGly_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 702 (87.53%)
- min/max: 0.52 / 15.0
- mean/std: 1.5807 / 1.5437415226937732
- quantiles: {'0.05': 0.629, '0.25': 0.95, '0.5': 1.3, '0.75': 1.7, '0.95': 2.814999999999999}

## lab_results_triGly_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 702 (87.53%)
- min/max: 0.52 / 5.2
- mean/std: 1.4487 / 0.8268153031085875
- quantiles: {'0.05': 0.629, '0.25': 0.95, '0.5': 1.25, '0.75': 1.6, '0.95': 2.8}

## lab_results_triGly_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 702 (87.53%)
- min/max: 0.52 / 4.2
- mean/std: 1.3382 / 0.6372865140657701
- quantiles: {'0.05': 0.629, '0.25': 0.94, '0.5': 1.2, '0.75': 1.6, '0.95': 2.7049999999999996}

## lab_results_triGly_value_stddev

- dtype: `Float64` (numeric)
- nulls: 702 (87.53%)
- min/max: 0.0 / 2.5
- mean/std: 0.0722 / 0.2867646404915536
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 0.49839728179296167}

## lab_results_triGly_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 702 (87.53%)
- min/max: 0.52 / 4.2
- mean/std: 1.3885 / 0.6837144170195653
- quantiles: {'0.05': 0.629, '0.25': 0.95, '0.5': 1.2, '0.75': 1.6, '0.95': 2.8}

## lab_results_triGly_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 702 (87.53%)
- min/max: 0.52 / 4.3
- mean/std: 1.4063 / 0.7088054257203508
- quantiles: {'0.05': 0.629, '0.25': 0.9500000000000001, '0.5': 1.2483333333333333, '0.75': 1.6, '0.95': 2.8}

## lab_results_cholTot_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 714 (89.03%)
- min/max: 2.2 / 7.8
- mean/std: 3.9523 / 1.117157442955601
- quantiles: {'0.05': 2.4, '0.25': 3.1, '0.5': 3.8, '0.75': 4.65, '0.95': 5.6}

## lab_results_cholTot_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 714 (89.03%)
- min/max: 2.2 / 7.8
- mean/std: 3.9409 / 1.1217429340233653
- quantiles: {'0.05': 2.4, '0.25': 3.1, '0.5': 3.8, '0.75': 4.65, '0.95': 5.6}

## lab_results_cholTot_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 714 (89.03%)
- min/max: 2.2 / 7.8
- mean/std: 3.9307 / 1.1250908473710182
- quantiles: {'0.05': 2.4, '0.25': 3.075, '0.5': 3.8, '0.75': 4.65, '0.95': 5.6}

## lab_results_cholTot_value_stddev

- dtype: `Float64` (numeric)
- nulls: 714 (89.03%)
- min/max: 0.0 / 0.45
- mean/std: 0.0108 / 0.056451007970987
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 0.032499999999999606}

## lab_results_cholTot_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 714 (89.03%)
- min/max: 2.2 / 7.8
- mean/std: 3.9420 / 1.1206239631899957
- quantiles: {'0.05': 2.4, '0.25': 3.1, '0.5': 3.8, '0.75': 4.65, '0.95': 5.6}

## lab_results_cholTot_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 714 (89.03%)
- min/max: 2.2 / 7.8
- mean/std: 3.9415 / 1.1197090547520039
- quantiles: {'0.05': 2.4, '0.25': 3.1, '0.5': 3.8000000000000003, '0.75': 4.6499999999999995, '0.95': 5.6}

## lab_results_hdl_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 714 (89.03%)
- min/max: 0.5 / 3.0
- mean/std: 1.1693 / 0.38991833938628695
- quantiles: {'0.05': 0.6, '0.25': 0.9, '0.5': 1.1, '0.75': 1.4, '0.95': 1.7}

## lab_results_hdl_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 714 (89.03%)
- min/max: 0.5 / 3.0
- mean/std: 1.1682 / 0.3908568408131669
- quantiles: {'0.05': 0.6, '0.25': 0.9, '0.5': 1.1, '0.75': 1.4, '0.95': 1.7}

## lab_results_hdl_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 714 (89.03%)
- min/max: 0.4 / 3.0
- mean/std: 1.1636 / 0.39338840979537953
- quantiles: {'0.05': 0.6, '0.25': 0.9, '0.5': 1.1, '0.75': 1.4, '0.95': 1.7}

## lab_results_hdl_value_stddev

- dtype: `Float64` (numeric)
- nulls: 714 (89.03%)
- min/max: 0.0 / 0.1
- mean/std: 0.0028 / 0.013891936273138023
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 0.0}

## lab_results_hdl_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 714 (89.03%)
- min/max: 0.4 / 3.0
- mean/std: 1.1648 / 0.39246927479924076
- quantiles: {'0.05': 0.6, '0.25': 0.9, '0.5': 1.1, '0.75': 1.4, '0.95': 1.7}

## lab_results_hdl_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 714 (89.03%)
- min/max: 0.45 / 3.0
- mean/std: 1.1665 / 0.3914107692464933
- quantiles: {'0.05': 0.6, '0.25': 0.9, '0.5': 1.1, '0.75': 1.4, '0.95': 1.7}

## lab_results_creatUS_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 786 (98.00%)
- min/max: 1.8 / 790.0
- mean/std: 282.2250 / 230.81977529377042
- quantiles: {'0.05': 6.675, '0.25': 105.8, '0.5': 243.8, '0.75': 441.6, '0.95': 647.4499999999999}

## lab_results_creatUS_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 786 (98.00%)
- min/max: 1.8 / 790.0
- mean/std: 270.1500 / 223.79277617176714
- quantiles: {'0.05': 6.675, '0.25': 105.8, '0.5': 243.8, '0.75': 361.1, '0.95': 647.4499999999999}

## lab_results_creatUS_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 786 (98.00%)
- min/max: 1.8 / 790.0
- mean/std: 270.1500 / 223.79277617176714
- quantiles: {'0.05': 6.675, '0.25': 105.8, '0.5': 243.8, '0.75': 361.1, '0.95': 647.4499999999999}

## lab_results_creatUS_value_stddev

- dtype: `Float64` (numeric)
- nulls: 786 (98.00%)
- min/max: 0.0 / 97.0
- mean/std: 6.0375 / 24.150000000000002
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 24.15}

## lab_results_creatUS_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 786 (98.00%)
- min/max: 1.8 / 790.0
- mean/std: 282.2250 / 230.81977529377042
- quantiles: {'0.05': 6.675, '0.25': 105.8, '0.5': 243.8, '0.75': 441.6, '0.95': 647.4499999999999}

## lab_results_creatUS_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 786 (98.00%)
- min/max: 1.8 / 790.0
- mean/std: 276.1875 / 226.04704193301595
- quantiles: {'0.05': 6.675, '0.25': 105.8, '0.5': 243.79999999999998, '0.75': 405.95, '0.95': 647.45}

## lab_results_albuminUS_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 791 (98.63%)
- min/max: 3.0 / 2700.0
- mean/std: 416.0909 / 836.3718616196333
- quantiles: {'0.05': 3.0, '0.25': 12.0, '0.5': 29.0, '0.75': 213.5, '0.95': 2007.5000000000002}

## lab_results_albuminUS_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 791 (98.63%)
- min/max: 3.0 / 2700.0
- mean/std: 416.0909 / 836.3718616196333
- quantiles: {'0.05': 3.0, '0.25': 12.0, '0.5': 29.0, '0.75': 213.5, '0.95': 2007.5000000000002}

## lab_results_albuminUS_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 791 (98.63%)
- min/max: 3.0 / 2700.0
- mean/std: 405.1818 / 822.915040351289
- quantiles: {'0.05': 3.0, '0.25': 12.0, '0.5': 29.0, '0.75': 213.5, '0.95': 1947.5000000000002}

## lab_results_albuminUS_value_stddev

- dtype: `Float64` (numeric)
- nulls: 791 (98.63%)
- min/max: 0.0 / 60.0
- mean/std: 5.4545 / 18.090680674665816
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 30.0}

## lab_results_albuminUS_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 791 (98.63%)
- min/max: 3.0 / 2700.0
- mean/std: 405.1818 / 822.915040351289
- quantiles: {'0.05': 3.0, '0.25': 12.0, '0.5': 29.0, '0.75': 213.5, '0.95': 1947.5000000000002}

## lab_results_albuminUS_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 791 (98.63%)
- min/max: 3.0 / 2700.0
- mean/std: 410.6364 / 829.4734803147445
- quantiles: {'0.05': 3.0, '0.25': 12.0, '0.5': 29.0, '0.75': 213.5, '0.95': 1977.5}

## lab_results_bun_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 802 (100.00%)
- All values are null.

## lab_results_bun_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 802 (100.00%)
- All values are null.

## lab_results_bun_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 802 (100.00%)
- All values are null.

## lab_results_bun_value_stddev

- dtype: `Float64` (numeric)
- nulls: 802 (100.00%)
- All values are null.

## lab_results_bun_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 802 (100.00%)
- All values are null.

## lab_results_bun_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 802 (100.00%)
- All values are null.

## lab_results_acr_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 792 (98.75%)
- min/max: 0.5 / 240.0
- mean/std: 55.9500 / 91.72476037944534
- quantiles: {'0.05': 0.95, '0.25': 2.525, '0.5': 5.6, '0.75': 67.4, '0.95': 223.68999999999997}

## lab_results_acr_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 792 (98.75%)
- min/max: 0.5 / 240.0
- mean/std: 55.9500 / 91.72476037944534
- quantiles: {'0.05': 0.95, '0.25': 2.525, '0.5': 5.6, '0.75': 67.4, '0.95': 223.68999999999997}

## lab_results_acr_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 792 (98.75%)
- min/max: 0.5 / 240.0
- mean/std: 46.7600 / 78.10055057424371
- quantiles: {'0.05': 0.95, '0.25': 2.525, '0.5': 5.6, '0.75': 67.4, '0.95': 182.33499999999987}

## lab_results_acr_value_stddev

- dtype: `Float64` (numeric)
- nulls: 792 (98.75%)
- min/max: 0.0 / 46.0
- mean/std: 4.5950 / 14.530665848473703
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 25.272499999999948}

## lab_results_acr_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 792 (98.75%)
- min/max: 0.5 / 240.0
- mean/std: 46.7600 / 78.10055057424371
- quantiles: {'0.05': 0.95, '0.25': 2.525, '0.5': 5.6, '0.75': 67.4, '0.95': 182.33499999999987}

## lab_results_acr_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 792 (98.75%)
- min/max: 0.5 / 240.0
- mean/std: 51.3550 / 83.93702153003366
- quantiles: {'0.05': 0.9500000000000001, '0.25': 2.525, '0.5': 5.6, '0.75': 67.39999999999999, '0.95': 203.01249999999993}

## lab_results_ldl_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 716 (89.28%)
- min/max: 0.6 / 5.2
- mean/std: 2.1593 / 0.9353989909283751
- quantiles: {'0.05': 0.85, '0.25': 1.5, '0.5': 2.0, '0.75': 2.8, '0.95': 3.7750000000000004}

## lab_results_ldl_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 716 (89.28%)
- min/max: 0.6 / 5.2
- mean/std: 2.1605 / 0.9372398863957736
- quantiles: {'0.05': 0.85, '0.25': 1.425, '0.5': 2.0, '0.75': 2.8, '0.95': 3.775}

## lab_results_ldl_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 716 (89.28%)
- min/max: 0.6 / 5.2
- mean/std: 2.1686 / 0.9315719324828743
- quantiles: {'0.05': 0.85, '0.25': 1.5, '0.5': 2.0, '0.75': 2.8, '0.95': 3.775}

## lab_results_ldl_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 716 (89.28%)
- min/max: 0.6 / 5.2
- mean/std: 2.1581 / 0.9355042825417201
- quantiles: {'0.05': 0.85, '0.25': 1.5, '0.5': 1.95, '0.75': 2.8, '0.95': 3.775}

## lab_results_ldl_value_stddev

- dtype: `Float64` (numeric)
- nulls: 716 (89.28%)
- min/max: 0.0 / 0.3
- mean/std: 0.0093 / 0.0416650706488673
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 0.04999999999999999}

## lab_results_ldl_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 716 (89.28%)
- min/max: 0.6 / 5.2
- mean/std: 2.1500 / 0.9410569777086388
- quantiles: {'0.05': 0.85, '0.25': 1.4, '0.5': 1.95, '0.75': 2.8, '0.95': 3.775}

## lab_results_potassium_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 53 (6.61%)
- min/max: 2.7 / 5.9
- mean/std: 4.0952 / 0.4261454002496229
- quantiles: {'0.05': 3.4, '0.25': 3.825, '0.5': 4.1, '0.75': 4.3, '0.95': 4.835833333333333}

## lab_results_potassium_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 53 (6.61%)
- min/max: 1.0 / 5.7
- mean/std: 3.8561 / 0.5257600302189022
- quantiles: {'0.05': 3.0, '0.25': 3.5, '0.5': 3.9, '0.75': 4.2, '0.95': 4.7}

## lab_results_potassium_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 53 (6.61%)
- min/max: 2.7 / 19.0
- mean/std: 4.3798 / 0.7701010253665052
- quantiles: {'0.05': 3.6, '0.25': 4.0, '0.5': 4.3, '0.75': 4.6, '0.95': 5.4}

## lab_results_potassium_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 53 (6.61%)
- min/max: 2.6 / 6.9
- mean/std: 4.1773 / 0.5848284087242903
- quantiles: {'0.05': 3.3, '0.25': 3.8, '0.5': 4.1, '0.75': 4.4, '0.95': 5.2}

## lab_results_potassium_value_stddev

- dtype: `Float64` (numeric)
- nulls: 53 (6.61%)
- min/max: 0.0 / 1.2
- mean/std: 0.1857 / 0.22097377197965817
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.10000000000000009, '0.75': 0.31124748994971857, '0.95': 0.6171422846693988}

## lab_results_potassium_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 53 (6.61%)
- min/max: 2.7 / 6.0
- mean/std: 4.0721 / 0.46785265671371346
- quantiles: {'0.05': 3.34, '0.25': 3.8, '0.5': 4.0, '0.75': 4.3, '0.95': 4.860000000000002}

## lab_results_sodium_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 50 (6.23%)
- min/max: 110.0 / 160.0
- mean/std: 139.3418 / 3.944212987300008
- quantiles: {'0.05': 133.0, '0.25': 137.0, '0.5': 140.0, '0.75': 142.0, '0.95': 145.0}

## lab_results_sodium_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 50 (6.23%)
- min/max: 120.0 / 160.0
- mean/std: 139.8178 / 3.5816593806939903
- quantiles: {'0.05': 134.0, '0.25': 138.0, '0.5': 140.0, '0.75': 142.0, '0.95': 145.0}

## lab_results_sodium_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 50 (6.23%)
- min/max: 110.0 / 160.0
- mean/std: 138.2992 / 4.127411456098308
- quantiles: {'0.05': 131.0, '0.25': 136.0, '0.5': 139.0, '0.75': 141.0, '0.95': 144.0}

## lab_results_sodium_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 50 (6.23%)
- min/max: 120.0 / 160.0
- mean/std: 139.7510 / 3.440106083427492
- quantiles: {'0.05': 133.8875, '0.25': 138.0, '0.5': 140.0, '0.75': 142.0, '0.95': 145.0}

## lab_results_sodium_value_stddev

- dtype: `Float64` (numeric)
- nulls: 50 (6.23%)
- min/max: 0.0 / 8.2
- mean/std: 1.0467 / 1.2796427407143782
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.5, '0.75': 1.699673171197595, '0.95': 3.490782244058075}

## lab_results_sodium_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 50 (6.23%)
- min/max: 120.0 / 160.0
- mean/std: 141.1596 / 3.7669901069446183
- quantiles: {'0.05': 135.0, '0.25': 139.0, '0.5': 141.0, '0.75': 143.0, '0.95': 147.0}

## lab_results_albuminBS_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 618 (77.06%)
- min/max: 11.0 / 39.0
- mean/std: 27.3750 / 5.149738706300486
- quantiles: {'0.05': 19.0, '0.25': 24.0, '0.5': 28.0, '0.75': 31.0, '0.95': 36.0}

## lab_results_albuminBS_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 618 (77.06%)
- min/max: 16.0 / 39.0
- mean/std: 27.6810 / 4.923304392005343
- quantiles: {'0.05': 19.553571428571427, '0.25': 24.0, '0.5': 28.050000000000004, '0.75': 30.999999999999996, '0.95': 35.849999999999994}

## lab_results_albuminBS_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 618 (77.06%)
- min/max: 12.0 / 39.0
- mean/std: 28.4783 / 5.022307093716697
- quantiles: {'0.05': 19.15, '0.25': 25.0, '0.5': 29.0, '0.75': 32.0, '0.95': 36.0}

## lab_results_albuminBS_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 618 (77.06%)
- min/max: 10.0 / 39.0
- mean/std: 26.5326 / 5.758246330940769
- quantiles: {'0.05': 16.15, '0.25': 23.0, '0.5': 27.0, '0.75': 31.0, '0.95': 35.0}

## lab_results_albuminBS_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 618 (77.06%)
- min/max: 16.0 / 39.0
- mean/std: 28.9457 / 4.766376154779195
- quantiles: {'0.05': 20.15, '0.25': 26.0, '0.5': 29.5, '0.75': 32.0, '0.95': 36.0}

## lab_results_albuminBS_value_stddev

- dtype: `Float64` (numeric)
- nulls: 618 (77.06%)
- min/max: 0.0 / 7.0
- mean/std: 0.9130 / 1.3620149904517775
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 1.5, '0.95': 3.7481500305951405}

## lab_results_hba1c%_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 802 (100.00%)
- All values are null.

## lab_results_hba1c%_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 802 (100.00%)
- All values are null.

## lab_results_hba1c%_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 802 (100.00%)
- All values are null.

## lab_results_hba1c%_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 802 (100.00%)
- All values are null.

## lab_results_hba1c%_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 802 (100.00%)
- All values are null.

## lab_results_hba1c%_value_stddev

- dtype: `Float64` (numeric)
- nulls: 802 (100.00%)
- All values are null.

## lab_results_hba1c_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 764 (95.26%)
- min/max: 23.0 / 120.0
- mean/std: 49.9737 / 21.88852203486007
- quantiles: {'0.05': 34.0, '0.25': 37.25, '0.5': 42.0, '0.75': 49.75, '0.95': 98.19999999999999}

## lab_results_hba1c_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 764 (95.26%)
- min/max: 23.0 / 120.0
- mean/std: 49.9211 / 21.90073623216052
- quantiles: {'0.05': 34.0, '0.25': 37.25, '0.5': 42.0, '0.75': 49.75, '0.95': 98.19999999999999}

## lab_results_hba1c_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 764 (95.26%)
- min/max: 23.0 / 120.0
- mean/std: 49.9386 / 21.89901135387679
- quantiles: {'0.05': 34.0, '0.25': 37.16666666666667, '0.5': 42.0, '0.75': 49.75, '0.95': 98.19999999999999}

## lab_results_hba1c_value_stddev

- dtype: `Float64` (numeric)
- nulls: 764 (95.26%)
- min/max: 0.0 / 1.0
- mean/std: 0.0387 / 0.1774631834890104
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 0.07071067811865409}

## lab_results_hba1c_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 764 (95.26%)
- min/max: 23.0 / 120.0
- mean/std: 49.9737 / 21.88852203486007
- quantiles: {'0.05': 34.0, '0.25': 37.25, '0.5': 42.0, '0.75': 49.75, '0.95': 98.19999999999999}

## lab_results_hba1c_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 764 (95.26%)
- min/max: 23.0 / 120.0
- mean/std: 49.8947 / 21.916043085453882
- quantiles: {'0.05': 34.0, '0.25': 37.0, '0.5': 42.0, '0.75': 49.75, '0.95': 98.19999999999999}

## lab_results_validSerumCreatinine_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 75 (9.35%)
- min/max: 3.2 / 23.0
- mean/std: 11.7119 / 4.033189244754817
- quantiles: {'0.05': 6.4467, '0.25': 8.8218, '0.5': 10.9707, '0.75': 13.7982, '0.95': 20.097869999999993}

## lab_results_validSerumCreatinine_value_stddev

- dtype: `Float64` (numeric)
- nulls: 75 (9.35%)
- min/max: 0.0 / 4.7
- mean/std: 0.5986 / 0.831434449918406
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.28275000000000006, '0.75': 0.9048000000000003, '0.95': 2.3407777955301077}

## lab_results_validSerumCreatinine_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 75 (9.35%)
- min/max: 2.2 / 23.0
- mean/std: 10.8169 / 3.718382716958943
- quantiles: {'0.05': 5.91513, '0.25': 8.1432, '0.5': 10.179, '0.75': 12.7803, '0.95': 18.514469999999992}

## lab_results_validSerumCreatinine_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 75 (9.35%)
- min/max: 3.5 / 23.0
- mean/std: 11.5329 / 3.803645868439801
- quantiles: {'0.05': 6.5598, '0.25': 8.76525, '0.5': 10.8576, '0.75': 13.6144125, '0.95': 19.158899999999985}

## lab_results_validSerumCreatinine_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 75 (9.35%)
- min/max: 3.5 / 23.0
- mean/std: 12.3802 / 4.244030514186902
- quantiles: {'0.05': 6.786, '0.25': 9.2742, '0.5': 11.4231, '0.75': 14.98575, '0.95': 20.776469999999993}

## lab_results_valideGFR_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 356 (44.39%)
- min/max: 6.0 / 90.0
- mean/std: 50.4731 / 21.647179909749088
- quantiles: {'0.05': 13.0, '0.25': 33.0, '0.5': 52.0, '0.75': 67.0, '0.95': 84.75}

## lab_results_valideGFR_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 356 (44.39%)
- min/max: 6.1 / 90.0
- mean/std: 54.3842 / 20.841834231033722
- quantiles: {'0.05': 18.15, '0.25': 40.541666666666664, '0.5': 56.24999999999999, '0.75': 70.16071428571429, '0.95': 86.625}

## lab_results_valideGFR_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 356 (44.39%)
- min/max: 7.0 / 90.0
- mean/std: 58.1233 / 21.27192682295904
- quantiles: {'0.05': 20.25, '0.25': 44.0, '0.5': 61.0, '0.75': 75.0, '0.95': 90.0}

## lab_results_valideGFR_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 356 (44.39%)
- min/max: 7.0 / 90.0
- mean/std: 53.4260 / 21.4671454378396
- quantiles: {'0.05': 17.25, '0.25': 38.0, '0.5': 55.0, '0.75': 68.75, '0.95': 90.0}

## lab_results_valideGFR_value_stddev

- dtype: `Float64` (numeric)
- nulls: 356 (44.39%)
- min/max: 0.0 / 29.0
- mean/std: 2.9069 / 4.27573147853017
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 1.389029025725781, '0.75': 4.0175269261433275, '0.95': 11.514254525131204}

## lab_results_valideGFR_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 356 (44.39%)
- min/max: 6.0 / 90.0
- mean/std: 55.3161 / 21.437152658007847
- quantiles: {'0.05': 17.25, '0.25': 41.0, '0.5': 58.0, '0.75': 71.75, '0.95': 90.0}

## symptoms_Ankle_swelling_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## symptoms_Ascites_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## symptoms_Breathlessness_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 801
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## symptoms_Cardiac_murmur_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## symptoms_Chest_pain_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## symptoms_Cheyne_stokes_respiration_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## symptoms_Depression_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## symptoms_Dizziness_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## symptoms_Elevated_jugular_venous_pressure_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## symptoms_Fatigue_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## symptoms_Hepatojugular_reflux_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## symptoms_Hepatomegaly_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## symptoms_Intermittent_claudication_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## symptoms_Irregular_pulse_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## symptoms_Loss_of_appetite_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## symptoms_Nocturnal_cough_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## symptoms_Oliguria_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## symptoms_Orthopnoea_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## symptoms_Palpitations_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## symptoms_Paroxysmal_nocturnal_dyspnea_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## symptoms_Peripheral_edema_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 788
  - `True`: 14

## symptoms_Pleural_effusion_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## symptoms_Pulmonary_crepitations_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## symptoms_Reduced_exercise_tolerance_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## symptoms_Syncope_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## symptoms_Tachycardia_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## symptoms_Tachypnoea_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## symptoms_Third_heart_sound_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## symptoms_Weight_gain_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## symptoms_Weight_loss_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## echocardiographs_lvef

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 765 (95.39%)
- min/max: 15.0 / 60.0
- mean/std: 32.2297 / 10.432051701496821
- quantiles: {'0.05': 15.0, '0.25': 30.0, '0.5': 30.0, '0.75': 35.0, '0.95': 51.99999999999996}

## echocardiographs_lvef_pET_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 784 (97.76%)
- min/max: 15.0 / 45.0
- mean/std: 29.2778 / 7.600739558099946
- quantiles: {'0.05': 15.0, '0.25': 26.25, '0.5': 30.0, '0.75': 30.0, '0.95': 40.67499999999999}

## echocardiographs_lvef_pET_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 784 (97.76%)
- min/max: 15.0 / 45.0
- mean/std: 30.3889 / 6.592231514431914
- quantiles: {'0.05': 19.25, '0.25': 30.0, '0.5': 30.0, '0.75': 30.0, '0.95': 40.67499999999999}

## echocardiographs_lvef_pET_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 784 (97.76%)
- min/max: 15.0 / 45.0
- mean/std: 28.1667 / 8.509510250647939
- quantiles: {'0.05': 15.0, '0.25': 21.25, '0.5': 30.0, '0.75': 30.0, '0.95': 40.67499999999999}

## echocardiographs_lvef_pET_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 784 (97.76%)
- min/max: 15.0 / 45.0
- mean/std: 27.3333 / 9.037503559514391
- quantiles: {'0.05': 15.0, '0.25': 20.0, '0.5': 30.0, '0.75': 30.0, '0.95': 40.67499999999999}

## echocardiographs_lvef_pET_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 784 (97.76%)
- min/max: 15.0 / 45.0
- mean/std: 29.1597 / 7.151949234962251
- quantiles: {'0.05': 19.25, '0.25': 25.0, '0.5': 30.0, '0.75': 30.0, '0.95': 40.67499999999999}

## echocardiographs_lvef_pET_stddev

- dtype: `Float64` (numeric)
- nulls: 784 (97.76%)
- min/max: 0.0 / 7.4
- mean/std: 1.3349 / 2.6041425515879824
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 6.307436087666682}

## electrocardiographs_ecg_qrs_duration_pET_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 798 (99.50%)
- min/max: 90.0 / 120.0
- mean/std: 106.5000 / 13.102162671355696
- quantiles: {'0.05': 91.8, '0.25': 99.0, '0.5': 109.0, '0.75': 116.5, '0.95': 117.7}

## electrocardiographs_ecg_qrs_duration_pET_stddev

- dtype: `Float64` (numeric)
- nulls: 798 (99.50%)
- min/max: 0.0 / 0.0
- mean/std: 0.0000 / 0.0
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 0.0}
- ⚠️ constant column (single value)

## electrocardiographs_ecg_qrs_duration_pET_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 798 (99.50%)
- min/max: 90.0 / 120.0
- mean/std: 106.5000 / 13.102162671355696
- quantiles: {'0.05': 91.8, '0.25': 99.0, '0.5': 109.0, '0.75': 116.5, '0.95': 117.7}

## electrocardiographs_ecg_qrs_duration_pET_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 798 (99.50%)
- min/max: 90.0 / 120.0
- mean/std: 106.5000 / 13.102162671355696
- quantiles: {'0.05': 91.8, '0.25': 99.0, '0.5': 109.0, '0.75': 116.5, '0.95': 117.7}

## electrocardiographs_ecg_qrs_duration_pET_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 798 (99.50%)
- min/max: 90.0 / 120.0
- mean/std: 106.5000 / 13.102162671355696
- quantiles: {'0.05': 91.8, '0.25': 99.0, '0.5': 109.0, '0.75': 116.5, '0.95': 117.7}

## electrocardiographs_ecg_qrs_duration_pET_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 798 (99.50%)
- min/max: 90.0 / 120.0
- mean/std: 106.5000 / 13.102162671355696
- quantiles: {'0.05': 91.8, '0.25': 99.0, '0.5': 109.0, '0.75': 116.5, '0.95': 117.7}

## electrocardiographs_ecg_qrs_axis_pET_stddev

- dtype: `Float64` (numeric)
- nulls: 802 (100.00%)
- All values are null.

## electrocardiographs_ecg_qrs_axis_pET_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 802 (100.00%)
- All values are null.

## electrocardiographs_ecg_qrs_axis_pET_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 802 (100.00%)
- All values are null.

## electrocardiographs_ecg_qrs_axis_pET_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 802 (100.00%)
- All values are null.

## electrocardiographs_ecg_qrs_axis_pET_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 802 (100.00%)
- All values are null.

## electrocardiographs_ecg_qrs_axis_pET_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 802 (100.00%)
- All values are null.

## electrocardiographs_ecg_qt_duration_corrected_pET_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 802 (100.00%)
- All values are null.

## electrocardiographs_ecg_qt_duration_corrected_pET_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 802 (100.00%)
- All values are null.

## electrocardiographs_ecg_qt_duration_corrected_pET_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 802 (100.00%)
- All values are null.

## electrocardiographs_ecg_qt_duration_corrected_pET_stddev

- dtype: `Float64` (numeric)
- nulls: 802 (100.00%)
- All values are null.

## electrocardiographs_ecg_qt_duration_corrected_pET_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 802 (100.00%)
- All values are null.

## electrocardiographs_ecg_qt_duration_corrected_pET_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 802 (100.00%)
- All values are null.

## electrocardiographs_ecg_st_pET

- dtype: `Boolean` (boolean)
- nulls: 802 (100.00%)
- unique values: 0
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 802

## electrocardiographs_ecg_ischemia_without_st_pET

- dtype: `Boolean` (boolean)
- nulls: 802 (100.00%)
- unique values: 0
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 802

## electrocardiographs_ecg_type_of_rhythms_pET_first

- dtype: `List(String)` (categorical)
- nulls: 738 (92.02%)
- unique values: 6
- top values (shown only where count ≥ 5):
  - null (missing): 738
  - `['LA17718-0']`: 32
  - `['LA17093-8']`: 16
  - `['LA17084-7']`: 13
  - 2 other distinct value(s) suppressed, covering 3 row(s) (count below 5 and/or ranked beyond top 20)

## electrocardiographs_ecg_type_of_rhythms_pET_last

- dtype: `List(String)` (categorical)
- nulls: 738 (92.02%)
- unique values: 6
- top values (shown only where count ≥ 5):
  - null (missing): 738
  - `['LA17718-0']`: 31
  - `['LA17093-8']`: 17
  - `['LA17084-7']`: 13
  - 2 other distinct value(s) suppressed, covering 3 row(s) (count below 5 and/or ranked beyond top 20)

## smoking_status_smoker_last

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## smoking_status_formerSmoker_last

- dtype: `Boolean` (boolean)
- nulls: 802 (100.00%)
- unique values: 0
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 802

## smoking_status_smoker_totalSmokingDuration_sum

- dtype: `Int64` (numeric)
- nulls: 802 (100.00%)
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
  - `LA28406-9`: 505
  - `LA28405-1`: 239
  - `LA28404-4`: 41
  - `LA28407-7`: 17

## nyha_nyha_pET

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `LA28406-9`: 542
  - `LA28405-1`: 207
  - `LA28404-4`: 53

## vital_signs_systolicBpDuringEncounter_value_pET

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 0 (0.00%)
- min/max: 69.0 / 230.0
- mean/std: 122.7915 / 22.12843958945778
- quantiles: {'0.05': 90.0, '0.25': 110.0, '0.5': 120.0, '0.75': 135.0, '0.95': 160.0}

## vital_signs_bmi_value_pET

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 0 (0.00%)
- min/max: 18.0 / 25.0
- mean/std: 21.7927 / 1.8698867100680474
- quantiles: {'0.05': 18.905743680004385, '0.25': 20.1862384176704, '0.5': 21.743702857855464, '0.75': 23.398181056379606, '0.95': 24.718109418765835}

## lab_results_creatBS_value_p3a_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 0 (0.00%)
- min/max: 3.5 / 190.0
- mean/std: 18.1150 / 20.889083960680995
- quantiles: {'0.05': 6.72945, '0.25': 9.048, '0.5': 11.639875, '0.75': 16.074974464285713, '0.95': 70.12938958333307}

## lab_results_validSerumCreatinine_value_pET

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 0 (0.00%)
- min/max: 3.5 / 23.0
- mean/std: 11.5930 / 3.9545920974917363
- quantiles: {'0.05': 6.4467, '0.25': 8.8218, '0.5': 10.8576, '0.75': 13.572000000000001, '0.95': 20.017765}

## hyperkalemia_severity_categorizedValue

- dtype: `String` (categorical)
- nulls: 53 (6.61%)
- unique values: 5
- top values (shown only where count ≥ 5):
  - `normal`: 722
  - null (missing): 53
  - `mild`: 19
  - `moderate`: 7
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## ckd_severity_categorizedValue

- dtype: `String` (categorical)
- nulls: 356 (44.39%)
- unique values: 7
- top values (shown only where count ≥ 5):
  - null (missing): 356
  - `mildly_decreased`: 187
  - `mild_to_moderate_decrease`: 105
  - `moderate_to_severe_decrease`: 63
  - `severe_decrease`: 48
  - `normal_or_high`: 27
  - `kidney_failure`: 16

## med_requests_activeDuringEncounter_bb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 673
  - `False`: 129

## med_requests_activeDuringEncounter_ace_inhibitors_arb_use_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 651
  - `False`: 151

## conditions_heartFailure_timeFromEarliest_first

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 280.0
- mean/std: 44.4800 / 65.21534653911789
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 5.0, '0.75': 77.0, '0.95': 189.94999999999993}

## conditions_heart_failure_hf_within_18mo_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `True`: 802

## conditions_heart_failure_occurred_prior_to_18_months_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 473
  - `True`: 329

## conditions_ap_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 769
  - `True`: 33

## conditions_af_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 534
  - `True`: 268

## conditions_cm_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 770
  - `True`: 32

## conditions_dysl_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## conditions_hf_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 744
  - `False`: 58

## conditions_hyp_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 425
  - `False`: 377

## conditions_ihd_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 433
  - `True`: 369

## conditions_mi_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 759
  - `True`: 43

## conditions_pad_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 772
  - `True`: 30

## conditions_stroke_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 793
  - `True`: 9

## conditions_tia_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 796
  - `True`: 6

## conditions_vd_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 745
  - `True`: 57

## conditions_revasc_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 716
  - `True`: 86

## conditions_devices_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 648
  - `True`: 154

## conditions_aidshiv_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 799
  - 1 other distinct value(s) suppressed, covering 3 row(s) (count below 5 and/or ranked beyond top 20)

## conditions_copd_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 719
  - `True`: 83

## conditions_diabetes_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 606
  - `True`: 196

## conditions_dem_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 777
  - `True`: 25

## conditions_dep_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 797
  - `True`: 5

## conditions_dia_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## conditions_hyperthyroid_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## conditions_hypothyroid_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 767
  - `True`: 35

## conditions_ibd_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## conditions_ld_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 785
  - `True`: 17

## conditions_mc_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 775
  - `True`: 27

## conditions_osa_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 780
  - `True`: 22

## conditions_rd_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 794
  - `True`: 8

## conditions_ckd_chronic_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 682
  - `True`: 120

## conditions_myocarditis_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## conditions_pericardial_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## conditions_substance_abuse_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## conditions_ap_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 528
  - `True`: 274

## conditions_af_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 441
  - `True`: 361

## conditions_cm_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 714
  - `True`: 88

## conditions_dysl_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## conditions_hf_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 801
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## conditions_hyp_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 606
  - `False`: 196

## conditions_ihd_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 462
  - `False`: 340

## conditions_mi_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 521
  - `True`: 281

## conditions_pad_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 659
  - `True`: 143

## conditions_stroke_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 688
  - `True`: 114

## conditions_tia_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 695
  - `True`: 107

## conditions_vd_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 666
  - `True`: 136

## conditions_revasc_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 606
  - `True`: 196

## conditions_devices_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 599
  - `True`: 203

## conditions_aidshiv_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 798
  - 1 other distinct value(s) suppressed, covering 4 row(s) (count below 5 and/or ranked beyond top 20)

## conditions_copd_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 668
  - `True`: 134

## conditions_diabetes_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 550
  - `True`: 252

## conditions_dem_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 760
  - `True`: 42

## conditions_dep_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 712
  - `True`: 90

## conditions_dia_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## conditions_hyperthyroid_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## conditions_hypothyroid_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 728
  - `True`: 74

## conditions_ibd_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## conditions_ld_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 765
  - `True`: 37

## conditions_mc_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 613
  - `True`: 189

## conditions_osa_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 721
  - `True`: 81

## conditions_rd_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 771
  - `True`: 31

## conditions_ckd_chronic_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 630
  - `True`: 172

## conditions_myocarditis_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 799
  - 1 other distinct value(s) suppressed, covering 3 row(s) (count below 5 and/or ranked beyond top 20)

## conditions_pericardial_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 777
  - `True`: 25

## conditions_substance_abuse_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## conditions_af_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 439
  - `True`: 363

## conditions_ckd_chronic_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 623
  - `True`: 179

## conditions_cm_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 714
  - `True`: 88

## conditions_copd_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 666
  - `True`: 136

## conditions_dem_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 759
  - `True`: 43

## conditions_dep_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 712
  - `True`: 90

## conditions_diabetes_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 549
  - `True`: 253

## conditions_hypothyroid_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 727
  - `True`: 75

## conditions_hyp_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 610
  - `False`: 192

## conditions_ihd_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 463
  - `False`: 339

## conditions_mc_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 611
  - `True`: 191

## conditions_mi_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 519
  - `True`: 283

## conditions_pad_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 657
  - `True`: 145

## conditions_rd_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 771
  - `True`: 31

## conditions_stroke_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 688
  - `True`: 114

## conditions_vd_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 666
  - `True`: 136

## med_admins_rasi_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_admins_arni_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_admins_acei_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_admins_arb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_admins_mra_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_admins_diuretics_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_admins_diuretics_loop_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_admins_anti_coag_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_admins_anti_plat_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_admins_thrombolytic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_admins_bb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_admins_ccb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_admins_digitalis_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_admins_antiarrhytmic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_admins_inotropes_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_admins_vasodil_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_admins_platelet_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_admins_ll_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_admins_ivabradine_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_admins_potassium_binders_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_admins_insulins_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_admins_oral_antidiabetic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_admins_sglt2i_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_admins_ari_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_admins_rdoad_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_admins_rdoad_syst_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_admins_cortico_syst_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_admins_antiinfl_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_requests_rasi_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_requests_arni_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_requests_acei_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_requests_arb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_requests_mra_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_requests_diuretics_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_requests_diuretics_loop_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_requests_anti_coag_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_requests_anti_plat_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_requests_thrombolytic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_requests_bb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_requests_ccb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_requests_digitalis_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_requests_antiarrhytmic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_requests_inotropes_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_requests_vasodil_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_requests_platelet_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_requests_ll_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_requests_ivabradine_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_requests_potassium_binders_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_requests_insulins_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_requests_oral_antidiabetic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_requests_sglt2i_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_requests_ari_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_requests_rdoad_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_requests_rdoad_syst_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_requests_cortico_syst_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_requests_antiinfl_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_admins_history_rasi_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_admins_history_arni_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_admins_history_acei_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_admins_history_arb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_admins_history_mra_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_admins_history_diuretics_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_admins_history_diuretics_loop_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_admins_history_anti_coag_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_admins_history_anti_plat_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_admins_history_thrombolytic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_admins_history_bb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_admins_history_ccb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_admins_history_digitalis_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_admins_history_antiarrhytmic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_admins_history_inotropes_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_admins_history_vasodil_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_admins_history_platelet_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_admins_history_ll_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_admins_history_ivabradine_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_admins_history_potassium_binders_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_admins_history_insulins_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_admins_history_oral_antidiabetic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_admins_history_sglt2i_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_admins_history_ari_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_admins_history_rdoad_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_admins_history_rdoad_syst_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_admins_history_cortico_syst_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_admins_history_antiinfl_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_requests_history_rasi_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 612
  - `False`: 190

## med_requests_history_arni_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 759
  - `True`: 43

## med_requests_history_acei_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 502
  - `False`: 300

## med_requests_history_arb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 465
  - `True`: 337

## med_requests_history_mra_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 465
  - `True`: 337

## med_requests_history_diuretics_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 618
  - `False`: 184

## med_requests_history_diuretics_loop_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 550
  - `False`: 252

## med_requests_history_anti_coag_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 563
  - `False`: 239

## med_requests_history_anti_plat_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 540
  - `False`: 262

## med_requests_history_thrombolytic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 794
  - `True`: 8

## med_requests_history_bb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 623
  - `False`: 179

## med_requests_history_ccb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 423
  - `True`: 379

## med_requests_history_digitalis_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 643
  - `True`: 159

## med_requests_history_antiarrhytmic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 716
  - `True`: 86

## med_requests_history_inotropes_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 756
  - `True`: 46

## med_requests_history_vasodil_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 425
  - `False`: 377

## med_requests_history_platelet_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 540
  - `False`: 262

## med_requests_history_ll_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 483
  - `False`: 319

## med_requests_history_ivabradine_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 800
  - 1 other distinct value(s) suppressed, covering 2 row(s) (count below 5 and/or ranked beyond top 20)

## med_requests_history_potassium_binders_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 781
  - `True`: 21

## med_requests_history_insulins_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 628
  - `True`: 174

## med_requests_history_oral_antidiabetic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 587
  - `True`: 215

## med_requests_history_sglt2i_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 754
  - `True`: 48

## med_requests_history_ari_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 802

## med_requests_history_rdoad_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 458
  - `True`: 344

## med_requests_history_rdoad_syst_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 766
  - `True`: 36

## med_requests_history_cortico_syst_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 485
  - `True`: 317

## med_requests_history_antiinfl_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 410
  - `True`: 392

## encounter_primary_reason_HF_Disease_f5a_w7d_first

- dtype: `Boolean` (boolean)
- nulls: 105 (13.09%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 536
  - `True`: 161
  - null (missing): 105

## encounter_primary_reason_HF_Disease_f5a_w1mo_first

- dtype: `Boolean` (boolean)
- nulls: 71 (8.85%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 558
  - `True`: 173
  - null (missing): 71

## encounter_primary_reason_HF_Disease_f5a_w3mo_first

- dtype: `Boolean` (boolean)
- nulls: 52 (6.48%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 566
  - `True`: 184
  - null (missing): 52

## encounter_primary_reason_HF_Disease_f5a_w6mo_first

- dtype: `Boolean` (boolean)
- nulls: 42 (5.24%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 572
  - `True`: 188
  - null (missing): 42

## encounter_primary_reason_HF_Disease_f5a_w1a_first

- dtype: `Boolean` (boolean)
- nulls: 32 (3.99%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 578
  - `True`: 192
  - null (missing): 32

## encounter_primary_reason_HF_Disease_f5a_w3a_first

- dtype: `Boolean` (boolean)
- nulls: 20 (2.49%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 587
  - `True`: 195
  - null (missing): 20

## encounter_primary_reason_HF_Disease_f5a_w5a_first

- dtype: `Boolean` (boolean)
- nulls: 19 (2.37%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 588
  - `True`: 195
  - null (missing): 19

## encounter_primary_reason_number_of_days_to_rehosp_for_heart_failure_f5a_first

- dtype: `Int32` (numeric)
- nulls: 607 (75.69%)
- min/max: 0.0 / 620.0
- mean/std: 19.7282 / 75.1041405804556
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 127.69999999999968}

## encounter_primary_reason_number_of_HF_rehospitalizations_5a_f5a_count

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 130.0
- mean/std: 4.9950 / 9.634518417177398
- quantiles: {'0.05': 0.0, '0.25': 1.0, '0.5': 2.0, '0.75': 6.0, '0.95': 19.949999999999932}

## encounter_primary_reason_CV_Disease_f5a_w7d_first

- dtype: `Boolean` (boolean)
- nulls: 105 (13.09%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 513
  - `False`: 184
  - null (missing): 105

## encounter_primary_reason_CV_Disease_f5a_w1mo_first

- dtype: `Boolean` (boolean)
- nulls: 71 (8.85%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 542
  - `False`: 189
  - null (missing): 71

## encounter_primary_reason_CV_Disease_f5a_w3mo_first

- dtype: `Boolean` (boolean)
- nulls: 52 (6.48%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 559
  - `False`: 191
  - null (missing): 52

## encounter_primary_reason_CV_Disease_f5a_w6mo_first

- dtype: `Boolean` (boolean)
- nulls: 42 (5.24%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 567
  - `False`: 193
  - null (missing): 42

## encounter_primary_reason_CV_Disease_f5a_w1a_first

- dtype: `Boolean` (boolean)
- nulls: 32 (3.99%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 575
  - `False`: 195
  - null (missing): 32

## encounter_primary_reason_CV_Disease_f5a_w3a_first

- dtype: `Boolean` (boolean)
- nulls: 20 (2.49%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 581
  - `False`: 201
  - null (missing): 20

## encounter_primary_reason_CV_Disease_f5a_w5a_first

- dtype: `Boolean` (boolean)
- nulls: 19 (2.37%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 582
  - `False`: 201
  - null (missing): 19

## encounter_primary_reason_number_of_days_to_rehosp_for_CV_f5a_first

- dtype: `Int32` (numeric)
- nulls: 220 (27.43%)
- min/max: 0.0 / 1700.0
- mean/std: 16.0344 / 93.16555058251869
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 50.899999999999864}

## encounter_primary_reason_number_of_CV_rehospitalizations_5a_f5a_count

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 170.0
- mean/std: 17.8603 / 21.596236988483135
- quantiles: {'0.05': 1.0, '0.25': 4.0, '0.5': 10.0, '0.75': 24.0, '0.95': 62.0}

## encounter_primary_reason_non_CV_Disease_f5a_w7d_first

- dtype: `Boolean` (boolean)
- nulls: 105 (13.09%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 513
  - `True`: 184
  - null (missing): 105

## encounter_primary_reason_non_CV_Disease_f5a_w1mo_first

- dtype: `Boolean` (boolean)
- nulls: 71 (8.85%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 542
  - `True`: 189
  - null (missing): 71

## encounter_primary_reason_non_CV_Disease_f5a_w3mo_first

- dtype: `Boolean` (boolean)
- nulls: 52 (6.48%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 559
  - `True`: 191
  - null (missing): 52

## encounter_primary_reason_non_CV_Disease_f5a_w6mo_first

- dtype: `Boolean` (boolean)
- nulls: 42 (5.24%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 567
  - `True`: 193
  - null (missing): 42

## encounter_primary_reason_non_CV_Disease_f5a_w1a_first

- dtype: `Boolean` (boolean)
- nulls: 32 (3.99%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 575
  - `True`: 195
  - null (missing): 32

## encounter_primary_reason_non_CV_Disease_f5a_w3a_first

- dtype: `Boolean` (boolean)
- nulls: 20 (2.49%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 581
  - `True`: 201
  - null (missing): 20

## encounter_primary_reason_non_CV_Disease_f5a_w5a_first

- dtype: `Boolean` (boolean)
- nulls: 19 (2.37%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 582
  - `True`: 201
  - null (missing): 19

## encounter_primary_reason_number_of_days_to_rehosp_for_non_CV_f5a_first

- dtype: `Int32` (numeric)
- nulls: 601 (74.94%)
- min/max: 0.0 / 820.0
- mean/std: 20.2985 / 97.24603049873652
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 62.0}

## encounter_primary_reason_number_of_non_CV_rehospitalizations_5a_f5a_count

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 870.0
- mean/std: 15.0973 / 54.7316049621167
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 4.0, '0.75': 12.75, '0.95': 52.94999999999993}

## encounter_primary_reason_renal_complications_f5a_w7d_first

- dtype: `Boolean` (boolean)
- nulls: 105 (13.09%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 673
  - null (missing): 105
  - `True`: 24

## encounter_primary_reason_renal_complications_f5a_w1mo_first

- dtype: `Boolean` (boolean)
- nulls: 71 (8.85%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 707
  - null (missing): 71
  - `True`: 24

## encounter_primary_reason_renal_complications_f5a_w3mo_first

- dtype: `Boolean` (boolean)
- nulls: 52 (6.48%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 726
  - null (missing): 52
  - `True`: 24

## encounter_primary_reason_renal_complications_f5a_w6mo_first

- dtype: `Boolean` (boolean)
- nulls: 42 (5.24%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 736
  - null (missing): 42
  - `True`: 24

## encounter_primary_reason_renal_complications_f5a_w1a_first

- dtype: `Boolean` (boolean)
- nulls: 32 (3.99%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 746
  - null (missing): 32
  - `True`: 24

## encounter_primary_reason_renal_complications_f5a_w3a_first

- dtype: `Boolean` (boolean)
- nulls: 20 (2.49%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 757
  - `True`: 25
  - null (missing): 20

## encounter_primary_reason_renal_complications_f5a_w5a_first

- dtype: `Boolean` (boolean)
- nulls: 19 (2.37%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 758
  - `True`: 25
  - null (missing): 19

## encounter_primary_reason_number_of_days_to_rehosp_for_renal_complications_f5a_first

- dtype: `Int32` (numeric)
- nulls: 777 (96.88%)
- min/max: 0.0 / 540.0
- mean/std: 21.8800 / 106.9081849064888
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 5.199999999999989}

## encounter_primary_reason_number_of_renal_rehospitalizations_5a_f5a_count

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 850.0
- mean/std: 5.1097 / 48.703315226296695
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 7.949999999999932}

## cause_of_death_isCV_f5a_w7d_first

- dtype: `Boolean` (boolean)
- nulls: 781 (97.38%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 781
  - `True`: 15
  - `False`: 6

## cause_of_death_isCV_f5a_w1mo_first

- dtype: `Boolean` (boolean)
- nulls: 757 (94.39%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 757
  - `True`: 29
  - `False`: 16

## cause_of_death_isCV_f5a_w3mo_first

- dtype: `Boolean` (boolean)
- nulls: 703 (87.66%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 703
  - `True`: 59
  - `False`: 40

## cause_of_death_isCV_f5a_w6mo_first

- dtype: `Boolean` (boolean)
- nulls: 668 (83.29%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 668
  - `True`: 82
  - `False`: 52

## cause_of_death_isCV_f5a_w1a_first

- dtype: `Boolean` (boolean)
- nulls: 606 (75.56%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 606
  - `True`: 112
  - `False`: 84

## cause_of_death_isCV_f5a_w3a_first

- dtype: `Boolean` (boolean)
- nulls: 500 (62.34%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 500
  - `True`: 168
  - `False`: 134

## cause_of_death_isCV_f5a_w5a_first

- dtype: `Boolean` (boolean)
- nulls: 413 (51.50%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 413
  - `True`: 212
  - `False`: 177

## cause_of_death_number_of_days_to_death_for_CV_f5a_first

- dtype: `Int32` (numeric)
- nulls: 590 (73.57%)
- min/max: 1.0 / 1800.0
- mean/std: 516.1462 / 527.2651179272615
- quantiles: {'0.05': 6.0, '0.25': 65.25, '0.5': 305.0, '0.75': 817.5, '0.95': 1606.9499999999998}

## cause_of_death_isRenal_f5a_w7d_first

- dtype: `Boolean` (boolean)
- nulls: 781 (97.38%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 781
  - `False`: 21

## cause_of_death_isRenal_f5a_w1mo_first

- dtype: `Boolean` (boolean)
- nulls: 757 (94.39%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 757
  - `False`: 45

## cause_of_death_isRenal_f5a_w3mo_first

- dtype: `Boolean` (boolean)
- nulls: 703 (87.66%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 703
  - `False`: 96
  - 1 other distinct value(s) suppressed, covering 3 row(s) (count below 5 and/or ranked beyond top 20)

## cause_of_death_isRenal_f5a_w6mo_first

- dtype: `Boolean` (boolean)
- nulls: 668 (83.29%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 668
  - `False`: 130
  - 1 other distinct value(s) suppressed, covering 4 row(s) (count below 5 and/or ranked beyond top 20)

## cause_of_death_isRenal_f5a_w1a_first

- dtype: `Boolean` (boolean)
- nulls: 606 (75.56%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 606
  - `False`: 191
  - `True`: 5

## cause_of_death_isRenal_f5a_w3a_first

- dtype: `Boolean` (boolean)
- nulls: 500 (62.34%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 500
  - `False`: 295
  - `True`: 7

## cause_of_death_isRenal_f5a_w5a_first

- dtype: `Boolean` (boolean)
- nulls: 413 (51.50%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 413
  - `False`: 382
  - `True`: 7

## cause_of_death_number_of_days_to_death_for_renal_f5a_first

- dtype: `Int32` (numeric)
- nulls: 795 (99.13%)
- min/max: 34.0 / 980.0
- mean/std: 291.7143 / 344.8075377587301
- quantiles: {'0.05': 35.5, '0.25': 54.5, '0.5': 114.0, '0.75': 403.0, '0.95': 821.4999999999995}

## cause_of_death_isNonRenalAndNonCV_f5a_w7d_first

- dtype: `Boolean` (boolean)
- nulls: 781 (97.38%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 781
  - `False`: 15
  - `True`: 6

## cause_of_death_isNonRenalAndNonCV_f5a_w1mo_first

- dtype: `Boolean` (boolean)
- nulls: 757 (94.39%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 757
  - `False`: 29
  - `True`: 16

## cause_of_death_isNonRenalAndNonCV_f5a_w3mo_first

- dtype: `Boolean` (boolean)
- nulls: 703 (87.66%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 703
  - `False`: 63
  - `True`: 36

## cause_of_death_isNonRenalAndNonCV_f5a_w6mo_first

- dtype: `Boolean` (boolean)
- nulls: 668 (83.29%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 668
  - `False`: 87
  - `True`: 47

## cause_of_death_isNonRenalAndNonCV_f5a_w1a_first

- dtype: `Boolean` (boolean)
- nulls: 606 (75.56%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 606
  - `False`: 119
  - `True`: 77

## cause_of_death_isNonRenalAndNonCV_f5a_w3a_first

- dtype: `Boolean` (boolean)
- nulls: 500 (62.34%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 500
  - `False`: 185
  - `True`: 117

## cause_of_death_isNonRenalAndNonCV_f5a_w5a_first

- dtype: `Boolean` (boolean)
- nulls: 413 (51.50%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 413
  - `False`: 240
  - `True`: 149

## cause_of_death_number_of_days_to_death_for_non_renal_and_non_CV_f5a_first

- dtype: `Int32` (numeric)
- nulls: 653 (81.42%)
- min/max: 1.0 / 1900.0
- mean/std: 557.9060 / 531.6250627351324
- quantiles: {'0.05': 12.4, '0.25': 95.0, '0.5': 348.0, '0.75': 883.0, '0.95': 1608.9999999999998}

## cause_of_death_isAllCause_f5a_w7d_first

- dtype: `Boolean` (boolean)
- nulls: 781 (97.38%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 781
  - `False`: 21

## cause_of_death_isAllCause_f5a_w1mo_first

- dtype: `Boolean` (boolean)
- nulls: 757 (94.39%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 757
  - `False`: 45

## cause_of_death_isAllCause_f5a_w3mo_first

- dtype: `Boolean` (boolean)
- nulls: 703 (87.66%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 703
  - `False`: 98
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## cause_of_death_isAllCause_f5a_w6mo_first

- dtype: `Boolean` (boolean)
- nulls: 668 (83.29%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 668
  - `False`: 133
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## cause_of_death_isAllCause_f5a_w1a_first

- dtype: `Boolean` (boolean)
- nulls: 606 (75.56%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 606
  - `False`: 194
  - 1 other distinct value(s) suppressed, covering 2 row(s) (count below 5 and/or ranked beyond top 20)

## cause_of_death_isAllCause_f5a_w3a_first

- dtype: `Boolean` (boolean)
- nulls: 500 (62.34%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 500
  - `False`: 292
  - `True`: 10

## cause_of_death_isAllCause_f5a_w5a_first

- dtype: `Boolean` (boolean)
- nulls: 413 (51.50%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 413
  - `False`: 368
  - `True`: 21

## cause_of_death_number_of_days_to_death_for_all_cause_f5a_first

- dtype: `Int32` (numeric)
- nulls: 781 (97.38%)
- min/max: 71.0 / 1800.0
- mean/std: 1125.3333 / 452.44561367454247
- quantiles: {'0.05': 298.0, '0.25': 1006.0, '0.5': 1114.0, '0.75': 1404.0, '0.95': 1746.0}

## eGFR_2021_ckd_epi_creatinine

- dtype: `Decimal(precision=38, scale=6)` (numeric)
- nulls: 0 (0.00%)
- min/max: 22.0 / 130.0
- mean/std: 67.0761 / 22.972531291210956
- quantiles: {'0.05': 30.99572615, '0.25': 48.25591075, '0.5': 66.639739, '0.75': 85.830665, '0.95': 102.43982679999999}

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

## maggic_total_score

- dtype: `Int32` (numeric)
- nulls: 784 (97.76%)
- min/max: 16.0 / 41.0
- mean/std: 25.7778 / 7.1583398959087905
- quantiles: {'0.05': 16.0, '0.25': 20.25, '0.5': 24.0, '0.75': 28.75, '0.95': 38.449999999999996}
