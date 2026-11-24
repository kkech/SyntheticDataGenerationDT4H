# Column Analysis

Total rows: 2065
Total columns: 540

## pid

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2065
- ⚠️ high-cardinality column (2065 distinct values)
- top values (shown only where count ≥ 5):
  - (none met the display threshold)
  - 2065 other distinct value(s) suppressed, covering 2065 row(s) (count below 5 and/or ranked beyond top 20)

## encounterId

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2065
- ⚠️ high-cardinality column (2065 distinct values)
- top values (shown only where count ≥ 5):
  - (none met the display threshold)
  - 2065 other distinct value(s) suppressed, covering 2065 row(s) (count below 5 and/or ranked beyond top 20)

## referenceTimePoint

- dtype: `Datetime(time_unit='ns', time_zone=None)` (categorical)
- nulls: 0 (0.00%)
- unique values: 1565
- ⚠️ high-cardinality column (1565 distinct values)
- top values (shown only where count ≥ 5):
  - `2017-05-23 00:00:00`: 5
  - 1564 other distinct value(s) suppressed, covering 2060 row(s) (count below 5 and/or ranked beyond top 20)

## eventTime

- dtype: `Datetime(time_unit='ns', time_zone=None)` (categorical)
- nulls: 0 (0.00%)
- unique values: 1610
- ⚠️ high-cardinality column (1610 distinct values)
- top values (shown only where count ≥ 5):
  - `2026-03-24 00:00:00`: 5
  - 1609 other distinct value(s) suppressed, covering 2060 row(s) (count below 5 and/or ranked beyond top 20)

## exitTime

- dtype: `Datetime(time_unit='ns', time_zone=None)` (categorical)
- nulls: 2065 (100.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 2065

## patient_demographics_sourceIdentifier

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 2065
- ⚠️ high-cardinality column (2065 distinct values)
- top values (shown only where count ≥ 5):
  - (none met the display threshold)
  - 2065 other distinct value(s) suppressed, covering 2065 row(s) (count below 5 and/or ranked beyond top 20)

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

## encounters_admissionDate

- dtype: `Datetime(time_unit='ns', time_zone=None)` (categorical)
- nulls: 0 (0.00%)
- unique values: 1610
- ⚠️ high-cardinality column (1610 distinct values)
- top values (shown only where count ≥ 5):
  - `2026-03-24 00:00:00`: 5
  - 1609 other distinct value(s) suppressed, covering 2060 row(s) (count below 5 and/or ranked beyond top 20)

## encounters_dischargeDate

- dtype: `Datetime(time_unit='ns', time_zone=None)` (categorical)
- nulls: 0 (0.00%)
- unique values: 1565
- ⚠️ high-cardinality column (1565 distinct values)
- top values (shown only where count ≥ 5):
  - `2017-05-23 00:00:00`: 5
  - 1564 other distinct value(s) suppressed, covering 2060 row(s) (count below 5 and/or ranked beyond top 20)

## encounters_numOfPreviousHFStays_count

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 450.0
- mean/std: 10.6475 / 22.56793522169907
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 4.0, '0.75': 12.0, '0.95': 41.799999999999955}

## vital_signs_weight_value_p6mo_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 257 (12.45%)
- min/max: 41.0 / 190.0
- mean/std: 84.4099 / 19.28784692509805
- quantiles: {'0.05': 56.0, '0.25': 71.74999999999999, '0.5': 82.0, '0.75': 95.0, '0.95': 120.0}

## vital_signs_weight_value_p6mo_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 257 (12.45%)
- min/max: 41.0 / 190.0
- mean/std: 84.1372 / 19.280413806718993
- quantiles: {'0.05': 56.0, '0.25': 71.0, '0.5': 82.0, '0.75': 95.0, '0.95': 119.64999999999986}

## vital_signs_weight_value_p6mo_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 257 (12.45%)
- min/max: 41.0 / 190.0
- mean/std: 84.5697 / 19.53976858064446
- quantiles: {'0.05': 56.0, '0.25': 71.0, '0.5': 83.0, '0.75': 95.0, '0.95': 120.0}

## vital_signs_weight_value_p6mo_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 257 (12.45%)
- min/max: 41.0 / 190.0
- mean/std: 84.2893 / 19.31891944437421
- quantiles: {'0.05': 56.0, '0.25': 71.0, '0.5': 82.0, '0.75': 95.0, '0.95': 119.64999999999986}

## vital_signs_weight_value_p6mo_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 257 (12.45%)
- min/max: 41.0 / 190.0
- mean/std: 84.7135 / 19.59009370275624
- quantiles: {'0.05': 56.35000000000001, '0.25': 72.0, '0.5': 83.0, '0.75': 95.0, '0.95': 120.0}

## vital_signs_weight_value_p6mo_stddev

- dtype: `Float64` (numeric)
- nulls: 257 (12.45%)
- min/max: 0.0 / 52.0
- mean/std: 0.2718 / 1.914728039807845
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 1.0}

## vital_signs_height_value_p1a_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 18 (0.87%)
- min/max: 65.0 / 200.0
- mean/std: 170.7699 / 10.531563721336529
- quantiles: {'0.05': 155.0, '0.25': 165.0, '0.5': 172.0, '0.75': 178.0, '0.95': 185.0}

## vital_signs_weight_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 320 (15.50%)
- min/max: 41.0 / 190.0
- mean/std: 84.4115 / 19.25404417743762
- quantiles: {'0.05': 56.0, '0.25': 71.0, '0.5': 82.0, '0.75': 95.0, '0.95': 118.79999999999995}

## vital_signs_height_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 29 (1.40%)
- min/max: 65.0 / 200.0
- mean/std: 170.8251 / 10.427136974355111
- quantiles: {'0.05': 155.0, '0.25': 165.0, '0.5': 172.0, '0.75': 178.0, '0.95': 186.0}

## vital_signs_diastolicBp_value_stddev

- dtype: `Float64` (numeric)
- nulls: 1047 (50.70%)
- min/max: 0.0 / 0.0
- mean/std: 0.0000 / 0.0
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 0.0}
- ⚠️ constant column (single value)

## vital_signs_diastolicBp_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 1047 (50.70%)
- min/max: 6.0 / 810.0
- mean/std: 76.0717 / 26.19756579371753
- quantiles: {'0.05': 55.0, '0.25': 67.0, '0.5': 76.0, '0.75': 80.0, '0.95': 100.0}

## vital_signs_diastolicBp_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1047 (50.70%)
- min/max: 6.0 / 810.0
- mean/std: 76.0717 / 26.19756579371753
- quantiles: {'0.05': 55.0, '0.25': 67.0, '0.5': 76.0, '0.75': 80.0, '0.95': 100.0}

## vital_signs_diastolicBp_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1047 (50.70%)
- min/max: 6.0 / 810.0
- mean/std: 76.0717 / 26.19756579371753
- quantiles: {'0.05': 55.0, '0.25': 67.0, '0.5': 76.0, '0.75': 80.0, '0.95': 100.0}

## vital_signs_diastolicBp_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1047 (50.70%)
- min/max: 6.0 / 810.0
- mean/std: 76.0717 / 26.19756579371753
- quantiles: {'0.05': 55.0, '0.25': 67.0, '0.5': 76.0, '0.75': 80.0, '0.95': 100.0}

## vital_signs_diastolicBp_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1047 (50.70%)
- min/max: 6.0 / 810.0
- mean/std: 76.0717 / 26.19756579371753
- quantiles: {'0.05': 55.0, '0.25': 67.0, '0.5': 76.0, '0.75': 80.0, '0.95': 100.0}

## vital_signs_heartRate_value_stddev

- dtype: `Float64` (numeric)
- nulls: 1201 (58.16%)
- min/max: 0.0 / 0.0
- mean/std: 0.0000 / 0.0
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 0.0}
- ⚠️ constant column (single value)

## vital_signs_heartRate_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 1201 (58.16%)
- min/max: 25.0 / 180.0
- mean/std: 83.5313 / 22.02329842513595
- quantiles: {'0.05': 55.0, '0.25': 70.0, '0.5': 80.0, '0.75': 95.0, '0.95': 128.8499999999999}

## vital_signs_heartRate_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1201 (58.16%)
- min/max: 25.0 / 180.0
- mean/std: 83.5313 / 22.02329842513595
- quantiles: {'0.05': 55.0, '0.25': 70.0, '0.5': 80.0, '0.75': 95.0, '0.95': 128.8499999999999}

## vital_signs_heartRate_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1201 (58.16%)
- min/max: 25.0 / 180.0
- mean/std: 83.5313 / 22.02329842513595
- quantiles: {'0.05': 55.0, '0.25': 70.0, '0.5': 80.0, '0.75': 95.0, '0.95': 128.8499999999999}

## vital_signs_heartRate_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1201 (58.16%)
- min/max: 25.0 / 180.0
- mean/std: 83.5313 / 22.02329842513595
- quantiles: {'0.05': 55.0, '0.25': 70.0, '0.5': 80.0, '0.75': 95.0, '0.95': 128.8499999999999}

## vital_signs_heartRate_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1201 (58.16%)
- min/max: 25.0 / 180.0
- mean/std: 83.5313 / 22.02329842513595
- quantiles: {'0.05': 55.0, '0.25': 70.0, '0.5': 80.0, '0.75': 95.0, '0.95': 128.8499999999999}

## vital_signs_systolicBp_value_stddev

- dtype: `Float64` (numeric)
- nulls: 1047 (50.70%)
- min/max: 0.0 / 0.0
- mean/std: 0.0000 / 0.0
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 0.0}
- ⚠️ constant column (single value)

## vital_signs_systolicBp_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 1047 (50.70%)
- min/max: 60.0 / 230.0
- mean/std: 129.0481 / 21.89818761675394
- quantiles: {'0.05': 100.0, '0.25': 116.0, '0.5': 126.00000000000001, '0.75': 140.0, '0.95': 170.0}

## vital_signs_systolicBp_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1047 (50.70%)
- min/max: 60.0 / 230.0
- mean/std: 129.0481 / 21.89818761675394
- quantiles: {'0.05': 100.0, '0.25': 116.0, '0.5': 126.0, '0.75': 140.0, '0.95': 170.0}

## vital_signs_systolicBp_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1047 (50.70%)
- min/max: 60.0 / 230.0
- mean/std: 129.0481 / 21.89818761675394
- quantiles: {'0.05': 100.0, '0.25': 116.0, '0.5': 126.0, '0.75': 140.0, '0.95': 170.0}

## vital_signs_systolicBp_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1047 (50.70%)
- min/max: 60.0 / 230.0
- mean/std: 129.0481 / 21.89818761675394
- quantiles: {'0.05': 100.0, '0.25': 116.0, '0.5': 126.0, '0.75': 140.0, '0.95': 170.0}

## vital_signs_systolicBp_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1047 (50.70%)
- min/max: 60.0 / 230.0
- mean/std: 129.0481 / 21.89818761675394
- quantiles: {'0.05': 100.0, '0.25': 116.0, '0.5': 126.0, '0.75': 140.0, '0.95': 170.0}

## vital_signs_oxygenSaturation_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2035 (98.55%)
- min/max: 70.0 / 100.0
- mean/std: 92.8333 / 7.149069548011044
- quantiles: {'0.05': 77.25, '0.25': 91.25, '0.5': 95.5, '0.75': 97.75, '0.95': 98.55}

## vital_signs_oxygenSaturation_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2035 (98.55%)
- min/max: 70.0 / 100.0
- mean/std: 92.8333 / 7.149069548011044
- quantiles: {'0.05': 77.25, '0.25': 91.25, '0.5': 95.5, '0.75': 97.75, '0.95': 98.55}

## vital_signs_oxygenSaturation_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 2035 (98.55%)
- min/max: 70.0 / 100.0
- mean/std: 92.8333 / 7.149069548011044
- quantiles: {'0.05': 77.25, '0.25': 91.25000000000001, '0.5': 95.5, '0.75': 97.75, '0.95': 98.55000000000001}

## vital_signs_oxygenSaturation_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2035 (98.55%)
- min/max: 70.0 / 100.0
- mean/std: 92.8333 / 7.149069548011044
- quantiles: {'0.05': 77.25, '0.25': 91.25, '0.5': 95.5, '0.75': 97.75, '0.95': 98.55}

## vital_signs_oxygenSaturation_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2035 (98.55%)
- min/max: 70.0 / 100.0
- mean/std: 92.8333 / 7.149069548011044
- quantiles: {'0.05': 77.25, '0.25': 91.25, '0.5': 95.5, '0.75': 97.75, '0.95': 98.55}

## vital_signs_oxygenSaturation_value_stddev

- dtype: `Float64` (numeric)
- nulls: 2035 (98.55%)
- min/max: 0.0 / 0.0
- mean/std: 0.0000 / 0.0
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 0.0}
- ⚠️ constant column (single value)

## lab_results_hemoglobin_value_stddev

- dtype: `Float64` (numeric)
- nulls: 1711 (82.86%)
- min/max: 0.0 / 21.0
- mean/std: 1.3207 / 2.8867039993634247
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 1.5, '0.95': 7.2542008497129435}

## lab_results_hemoglobin_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 1711 (82.86%)
- min/max: 50.0 / 200.0
- mean/std: 124.6349 / 26.33520793720501
- quantiles: {'0.05': 80.65, '0.25': 108.25, '0.5': 125.25000000000001, '0.75': 142.0, '0.95': 167.67499999999998}

## lab_results_hemoglobin_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1711 (82.86%)
- min/max: 50.0 / 200.0
- mean/std: 124.4379 / 26.62674180118927
- quantiles: {'0.05': 80.0, '0.25': 108.25, '0.5': 125.0, '0.75': 143.0, '0.95': 167.34999999999997}

## lab_results_hemoglobin_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1711 (82.86%)
- min/max: 50.0 / 200.0
- mean/std: 126.2627 / 26.535270653583925
- quantiles: {'0.05': 83.3, '0.25': 110.0, '0.5': 127.0, '0.75': 144.75, '0.95': 169.69999999999993}

## lab_results_hemoglobin_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1711 (82.86%)
- min/max: 50.0 / 200.0
- mean/std: 123.1271 / 26.69188796043471
- quantiles: {'0.05': 77.65, '0.25': 107.25, '0.5': 124.0, '0.75': 142.0, '0.95': 167.0}

## lab_results_hemoglobin_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1711 (82.86%)
- min/max: 50.0 / 200.0
- mean/std: 124.5989 / 26.730055419738168
- quantiles: {'0.05': 79.65, '0.25': 108.25, '0.5': 125.0, '0.75': 143.0, '0.95': 168.34999999999997}

## lab_results_ferritin_value_stddev

- dtype: `Float64` (numeric)
- nulls: 1979 (95.84%)
- min/max: 0.0 / 0.46
- mean/std: 0.0052 / 0.0485247479415473
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 0.0}

## lab_results_ferritin_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 1979 (95.84%)
- min/max: 3.7 / 1100.0
- mean/std: 174.9517 / 203.14655915599957
- quantiles: {'0.05': 15.15, '0.25': 48.875, '0.5': 108.5, '0.75': 228.325, '0.95': 549.8000000000001}

## lab_results_ferritin_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1979 (95.84%)
- min/max: 3.7 / 1100.0
- mean/std: 174.9465 / 203.15096136945834
- quantiles: {'0.05': 15.15, '0.25': 48.875, '0.5': 108.5, '0.75': 228.325, '0.95': 549.8}

## lab_results_ferritin_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1979 (95.84%)
- min/max: 3.7 / 1100.0
- mean/std: 174.9570 / 203.14216843829158
- quantiles: {'0.05': 15.15, '0.25': 48.875, '0.5': 108.5, '0.75': 228.325, '0.95': 549.8}

## lab_results_ferritin_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1979 (95.84%)
- min/max: 3.7 / 1100.0
- mean/std: 174.9465 / 203.15096136945834
- quantiles: {'0.05': 15.15, '0.25': 48.875, '0.5': 108.5, '0.75': 228.325, '0.95': 549.8}

## lab_results_ferritin_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1979 (95.84%)
- min/max: 3.7 / 1100.0
- mean/std: 174.9570 / 203.14216843829158
- quantiles: {'0.05': 15.15, '0.25': 48.875, '0.5': 108.5, '0.75': 228.325, '0.95': 549.8}

## lab_results_tfs_value_stddev

- dtype: `Float64` (numeric)
- nulls: 1979 (95.84%)
- min/max: 0.0 / 0.0005
- mean/std: 0.0000 / 5.391638660171915e-05
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 0.0}

## lab_results_tfs_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 1979 (95.84%)
- min/max: 0.012 / 0.51
- mean/std: 0.1878 / 0.115593784505984
- quantiles: {'0.05': 0.03575, '0.25': 0.1, '0.5': 0.1655, '0.75': 0.26475000000000004, '0.95': 0.41275}

## lab_results_tfs_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1979 (95.84%)
- min/max: 0.012 / 0.51
- mean/std: 0.1878 / 0.1156027184321023
- quantiles: {'0.05': 0.03575, '0.25': 0.1, '0.5': 0.1655, '0.75': 0.26475, '0.95': 0.41275}

## lab_results_tfs_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1979 (95.84%)
- min/max: 0.013 / 0.51
- mean/std: 0.1878 / 0.1155848750394847
- quantiles: {'0.05': 0.03575, '0.25': 0.1, '0.5': 0.1655, '0.75': 0.26475, '0.95': 0.41275}

## lab_results_tfs_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1979 (95.84%)
- min/max: 0.012 / 0.51
- mean/std: 0.1878 / 0.1156027184321023
- quantiles: {'0.05': 0.03575, '0.25': 0.1, '0.5': 0.1655, '0.75': 0.26475, '0.95': 0.41275}

## lab_results_tfs_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1979 (95.84%)
- min/max: 0.013 / 0.51
- mean/std: 0.1878 / 0.1155848750394847
- quantiles: {'0.05': 0.03575, '0.25': 0.1, '0.5': 0.1655, '0.75': 0.26475, '0.95': 0.41275}

## lab_results_ntProBnp_value_stddev

- dtype: `Float64` (numeric)
- nulls: 828 (40.10%)
- min/max: 0.0 / 2800.0
- mean/std: 17.1002 / 148.80064097037265
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 0.0}

## lab_results_ntProBnp_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 828 (40.10%)
- min/max: 0.0 / 35000.0
- mean/std: 6830.2307 / 8669.383960939464
- quantiles: {'0.05': 197.80000000000004, '0.25': 1331.0, '0.5': 3528.0, '0.75': 8397.0, '0.95': 32945.80000000005}

## lab_results_ntProBnp_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 828 (40.10%)
- min/max: 0.0 / 35000.0
- mean/std: 6826.7090 / 8672.951271060296
- quantiles: {'0.05': 193.0, '0.25': 1322.0, '0.5': 3512.9999999999995, '0.75': 8361.0, '0.95': 32945.80000000004}

## lab_results_ntProBnp_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 828 (40.10%)
- min/max: 0.0 / 35000.0
- mean/std: 6847.5190 / 8673.304723920604
- quantiles: {'0.05': 197.80000000000004, '0.25': 1336.0, '0.5': 3579.0000000000005, '0.75': 8441.0, '0.95': 32945.80000000004}

## lab_results_ntProBnp_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 828 (40.10%)
- min/max: 0.0 / 35000.0
- mean/std: 6813.0655 / 8668.190296570181
- quantiles: {'0.05': 193.0, '0.25': 1322.0, '0.5': 3512.9999999999995, '0.75': 8361.0, '0.95': 32945.80000000004}

## lab_results_ntProBnp_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 828 (40.10%)
- min/max: 0.0 / 35000.0
- mean/std: 6833.8715 / 8668.579543281001
- quantiles: {'0.05': 197.80000000000004, '0.25': 1336.0, '0.5': 3579.0000000000005, '0.75': 8441.0, '0.95': 32945.80000000004}

## lab_results_bnp_value_stddev

- dtype: `Float64` (numeric)
- nulls: 2065 (100.00%)
- All values are null.

## lab_results_bnp_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 2065 (100.00%)
- All values are null.

## lab_results_bnp_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2065 (100.00%)
- All values are null.

## lab_results_bnp_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2065 (100.00%)
- All values are null.

## lab_results_bnp_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2065 (100.00%)
- All values are null.

## lab_results_bnp_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2065 (100.00%)
- All values are null.

## lab_results_crpNonHs_value_stddev

- dtype: `Float64` (numeric)
- nulls: 59 (2.86%)
- min/max: 0.0 / 53.0
- mean/std: 0.3343 / 2.7915277390012787
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 0.0}

## lab_results_crpNonHs_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 59 (2.86%)
- min/max: 0.2 / 400.0
- mean/std: 39.7130 / 62.85465752015965
- quantiles: {'0.05': 0.8, '0.25': 3.1, '0.5': 10.85, '0.75': 46.6, '0.95': 180.625}

## lab_results_crpNonHs_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 59 (2.86%)
- min/max: 0.0 / 400.0
- mean/std: 39.7158 / 62.85879512222906
- quantiles: {'0.05': 0.8, '0.25': 3.1, '0.5': 10.7, '0.75': 46.575, '0.95': 180.625}

## lab_results_crpNonHs_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 59 (2.86%)
- min/max: 0.2 / 400.0
- mean/std: 40.0712 / 63.27313634419474
- quantiles: {'0.05': 0.8, '0.25': 3.1, '0.5': 10.95, '0.75': 47.25, '0.95': 180.625}

## lab_results_crpNonHs_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 59 (2.86%)
- min/max: 0.0 / 400.0
- mean/std: 39.3780 / 62.58979762018335
- quantiles: {'0.05': 0.8, '0.25': 3.025, '0.5': 10.5, '0.75': 46.175, '0.95': 180.625}

## lab_results_crpNonHs_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 59 (2.86%)
- min/max: 0.0 / 400.0
- mean/std: 39.7216 / 63.01155632059555
- quantiles: {'0.05': 0.8, '0.25': 3.1, '0.5': 10.7, '0.75': 46.575, '0.95': 180.625}

## lab_results_crpHs_value_stddev

- dtype: `Float64` (numeric)
- nulls: 2065 (100.00%)
- All values are null.

## lab_results_crpHs_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 2065 (100.00%)
- All values are null.

## lab_results_crpHs_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2065 (100.00%)
- All values are null.

## lab_results_crpHs_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2065 (100.00%)
- All values are null.

## lab_results_crpHs_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2065 (100.00%)
- All values are null.

## lab_results_crpHs_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2065 (100.00%)
- All values are null.

## lab_results_tropIHs_value_stddev

- dtype: `Float64` (numeric)
- nulls: 2065 (100.00%)
- All values are null.

## lab_results_tropIHs_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 2065 (100.00%)
- All values are null.

## lab_results_tropIHs_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2065 (100.00%)
- All values are null.

## lab_results_tropIHs_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2065 (100.00%)
- All values are null.

## lab_results_tropIHs_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2065 (100.00%)
- All values are null.

## lab_results_tropIHs_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2065 (100.00%)
- All values are null.

## lab_results_tropInHs_value_stddev

- dtype: `Float64` (numeric)
- nulls: 2065 (100.00%)
- All values are null.

## lab_results_tropInHs_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 2065 (100.00%)
- All values are null.

## lab_results_tropInHs_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2065 (100.00%)
- All values are null.

## lab_results_tropInHs_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2065 (100.00%)
- All values are null.

## lab_results_tropInHs_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2065 (100.00%)
- All values are null.

## lab_results_tropInHs_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2065 (100.00%)
- All values are null.

## lab_results_tropTHs_value_stddev

- dtype: `Float64` (numeric)
- nulls: 1165 (56.42%)
- min/max: 0.0 / 5.0
- mean/std: 0.0759 / 0.4286403147221687
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 0.2941749999999989}

## lab_results_tropTHs_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 1165 (56.42%)
- min/max: 0.0 / 10.0
- mean/std: 1.0718 / 2.027310047889024
- quantiles: {'0.05': 0.017, '0.25': 0.045, '0.5': 0.167, '0.75': 0.96355, '0.95': 5.744174999999999}

## lab_results_tropTHs_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1165 (56.42%)
- min/max: 0.0 / 10.0
- mean/std: 1.0592 / 2.054948896347228
- quantiles: {'0.05': 0.015, '0.25': 0.042, '0.5': 0.1475, '0.75': 0.927, '0.95': 5.761049999999996}

## lab_results_tropTHs_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1165 (56.42%)
- min/max: 0.0 / 10.0
- mean/std: 1.1510 / 2.1647640656334506
- quantiles: {'0.05': 0.017, '0.25': 0.047, '0.5': 0.1755, '0.75': 1.0695, '0.95': 6.301749999999997}

## lab_results_tropTHs_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1165 (56.42%)
- min/max: 0.0 / 10.0
- mean/std: 0.9947 / 1.9786692742792786
- quantiles: {'0.05': 0.014, '0.25': 0.03975, '0.5': 0.132, '0.75': 0.82125, '0.95': 5.550099999999997}

## lab_results_tropTHs_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1165 (56.42%)
- min/max: 0.0 / 10.0
- mean/std: 1.0853 / 2.096166481424657
- quantiles: {'0.05': 0.016, '0.25': 0.043, '0.5': 0.152, '0.75': 0.97825, '0.95': 5.761049999999996}

## lab_results_tropTnHs_value_stddev

- dtype: `Float64` (numeric)
- nulls: 2065 (100.00%)
- All values are null.

## lab_results_tropTnHs_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 2065 (100.00%)
- All values are null.

## lab_results_tropTnHs_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2065 (100.00%)
- All values are null.

## lab_results_tropTnHs_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2065 (100.00%)
- All values are null.

## lab_results_tropTnHs_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2065 (100.00%)
- All values are null.

## lab_results_tropTnHs_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2065 (100.00%)
- All values are null.

## lab_results_triGly_value_stddev

- dtype: `Float64` (numeric)
- nulls: 1104 (53.46%)
- min/max: 0.0 / 0.35
- mean/std: 0.0008 / 0.014295178517906923
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 0.0}

## lab_results_triGly_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 1104 (53.46%)
- min/max: 0.33 / 14.0
- mean/std: 1.3516 / 1.102358585579577
- quantiles: {'0.05': 0.5, '0.25': 0.79, '0.5': 1.0899999999999999, '0.75': 1.54, '0.95': 2.85}

## lab_results_triGly_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1104 (53.46%)
- min/max: 0.0 / 14.0
- mean/std: 1.3515 / 1.1027600044167865
- quantiles: {'0.05': 0.5, '0.25': 0.79, '0.5': 1.09, '0.75': 1.54, '0.95': 2.85}

## lab_results_triGly_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1104 (53.46%)
- min/max: 0.33 / 14.0
- mean/std: 1.3524 / 1.1020313086899967
- quantiles: {'0.05': 0.5, '0.25': 0.79, '0.5': 1.09, '0.75': 1.54, '0.95': 2.85}

## lab_results_triGly_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1104 (53.46%)
- min/max: 0.0 / 14.0
- mean/std: 1.3508 / 1.1028710719421118
- quantiles: {'0.05': 0.5, '0.25': 0.79, '0.5': 1.09, '0.75': 1.54, '0.95': 2.85}

## lab_results_triGly_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1104 (53.46%)
- min/max: 0.33 / 14.0
- mean/std: 1.3517 / 1.1021429577171817
- quantiles: {'0.05': 0.5, '0.25': 0.79, '0.5': 1.09, '0.75': 1.54, '0.95': 2.85}

## lab_results_cholTot_value_stddev

- dtype: `Float64` (numeric)
- nulls: 1123 (54.38%)
- min/max: 0.0 / 1.6
- mean/std: 0.0025 / 0.05311884440387786
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 0.0}

## lab_results_cholTot_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 1123 (54.38%)
- min/max: 1.2 / 8.9
- mean/std: 4.1263 / 1.2194436789568868
- quantiles: {'0.05': 2.4505, '0.25': 3.2199999999999998, '0.5': 3.9600000000000004, '0.75': 4.930000000000001, '0.95': 6.379499999999999}

## lab_results_cholTot_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1123 (54.38%)
- min/max: 0.0 / 8.9
- mean/std: 4.1244 / 1.223805737654829
- quantiles: {'0.05': 2.4505, '0.25': 3.22, '0.5': 3.96, '0.75': 4.93, '0.95': 6.379499999999999}

## lab_results_cholTot_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1123 (54.38%)
- min/max: 1.2 / 8.9
- mean/std: 4.1288 / 1.2177304616493254
- quantiles: {'0.05': 2.46, '0.25': 3.22, '0.5': 3.96, '0.75': 4.93, '0.95': 6.379499999999999}

## lab_results_cholTot_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1123 (54.38%)
- min/max: 0.0 / 8.9
- mean/std: 4.1238 / 1.2234629207003953
- quantiles: {'0.05': 2.4505, '0.25': 3.22, '0.5': 3.96, '0.75': 4.93, '0.95': 6.379499999999999}

## lab_results_cholTot_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1123 (54.38%)
- min/max: 1.2 / 8.9
- mean/std: 4.1283 / 1.2173877887402764
- quantiles: {'0.05': 2.46, '0.25': 3.22, '0.5': 3.96, '0.75': 4.93, '0.95': 6.379499999999999}

## lab_results_hdl_value_stddev

- dtype: `Float64` (numeric)
- nulls: 1127 (54.58%)
- min/max: 0.0 / 0.48
- mean/std: 0.0007 / 0.015880093386144583
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 0.0}

## lab_results_hdl_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 1127 (54.58%)
- min/max: 0.13 / 3.0
- mean/std: 1.0763 / 0.34248587305159633
- quantiles: {'0.05': 0.5785, '0.25': 0.85, '0.5': 1.03, '0.75': 1.26, '0.95': 1.6814999999999998}

## lab_results_hdl_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1127 (54.58%)
- min/max: 0.0 / 3.0
- mean/std: 1.0756 / 0.3436747646662467
- quantiles: {'0.05': 0.5785, '0.25': 0.85, '0.5': 1.03, '0.75': 1.26, '0.95': 1.6814999999999998}

## lab_results_hdl_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1127 (54.58%)
- min/max: 0.13 / 3.0
- mean/std: 1.0770 / 0.34203683909783594
- quantiles: {'0.05': 0.58, '0.25': 0.8525, '0.5': 1.03, '0.75': 1.26, '0.95': 1.6814999999999998}

## lab_results_hdl_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1127 (54.58%)
- min/max: 0.0 / 3.0
- mean/std: 1.0756 / 0.3436688841191399
- quantiles: {'0.05': 0.5785, '0.25': 0.85, '0.5': 1.03, '0.75': 1.26, '0.95': 1.6814999999999998}

## lab_results_hdl_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1127 (54.58%)
- min/max: 0.13 / 3.0
- mean/std: 1.0769 / 0.342030971306218
- quantiles: {'0.05': 0.58, '0.25': 0.8525, '0.5': 1.03, '0.75': 1.26, '0.95': 1.6814999999999998}

## lab_results_creatUS_value_stddev

- dtype: `Float64` (numeric)
- nulls: 2034 (98.50%)
- min/max: 0.0 / 0.0
- mean/std: 0.0000 / 0.0
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 0.0}
- ⚠️ constant column (single value)

## lab_results_creatUS_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 2034 (98.50%)
- min/max: 79.0 / 3500.0
- mean/std: 871.0240 / 695.56076668503
- quantiles: {'0.05': 152.712, '0.25': 384.608, '0.5': 701.344, '0.75': 1040.704, '0.95': 1934.3519999999999}

## lab_results_creatUS_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2034 (98.50%)
- min/max: 79.0 / 3500.0
- mean/std: 871.0240 / 695.56076668503
- quantiles: {'0.05': 152.712, '0.25': 384.608, '0.5': 701.344, '0.75': 1040.704, '0.95': 1934.352}

## lab_results_creatUS_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2034 (98.50%)
- min/max: 79.0 / 3500.0
- mean/std: 871.0240 / 695.56076668503
- quantiles: {'0.05': 152.712, '0.25': 384.608, '0.5': 701.344, '0.75': 1040.704, '0.95': 1934.352}

## lab_results_creatUS_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2034 (98.50%)
- min/max: 79.0 / 3500.0
- mean/std: 871.0240 / 695.56076668503
- quantiles: {'0.05': 152.712, '0.25': 384.608, '0.5': 701.344, '0.75': 1040.704, '0.95': 1934.352}

## lab_results_creatUS_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2034 (98.50%)
- min/max: 79.0 / 3500.0
- mean/std: 871.0240 / 695.56076668503
- quantiles: {'0.05': 152.712, '0.25': 384.608, '0.5': 701.344, '0.75': 1040.704, '0.95': 1934.352}

## lab_results_albuminUS_value_stddev

- dtype: `Float64` (numeric)
- nulls: 2039 (98.74%)
- min/max: 0.0 / 0.0
- mean/std: 0.0000 / 0.0
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 0.0}
- ⚠️ constant column (single value)

## lab_results_albuminUS_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 2039 (98.74%)
- min/max: 3.0 / 500.0
- mean/std: 95.5615 / 114.27949274274216
- quantiles: {'0.05': 3.35, '0.25': 18.15, '0.5': 46.699999999999996, '0.75': 133.0, '0.95': 301.625}

## lab_results_albuminUS_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2039 (98.74%)
- min/max: 3.0 / 500.0
- mean/std: 95.5615 / 114.27949274274216
- quantiles: {'0.05': 3.35, '0.25': 18.15, '0.5': 46.7, '0.75': 133.0, '0.95': 301.62499999999994}

## lab_results_albuminUS_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2039 (98.74%)
- min/max: 3.0 / 500.0
- mean/std: 95.5615 / 114.27949274274216
- quantiles: {'0.05': 3.35, '0.25': 18.15, '0.5': 46.7, '0.75': 133.0, '0.95': 301.62499999999994}

## lab_results_albuminUS_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2039 (98.74%)
- min/max: 3.0 / 500.0
- mean/std: 95.5615 / 114.27949274274216
- quantiles: {'0.05': 3.35, '0.25': 18.15, '0.5': 46.7, '0.75': 133.0, '0.95': 301.62499999999994}

## lab_results_albuminUS_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2039 (98.74%)
- min/max: 3.0 / 500.0
- mean/std: 95.5615 / 114.27949274274216
- quantiles: {'0.05': 3.35, '0.25': 18.15, '0.5': 46.7, '0.75': 133.0, '0.95': 301.62499999999994}

## lab_results_bun_value_stddev

- dtype: `Float64` (numeric)
- nulls: 2064 (99.95%)
- min/max: 0.0 / 0.0
- mean/std: 0.0000 / None
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 0.0}
- ⚠️ constant column (single value)

## lab_results_bun_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 2064 (99.95%)
- min/max: 5.9 / 6.0
- mean/std: 5.9619 / None
- quantiles: {'0.05': 5.961899999999999, '0.25': 5.961899999999999, '0.5': 5.961899999999999, '0.75': 5.961899999999999, '0.95': 5.961899999999999}
- ⚠️ constant column (single value)

## lab_results_bun_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2064 (99.95%)
- min/max: 5.9 / 6.0
- mean/std: 5.9619 / None
- quantiles: {'0.05': 5.961899999999999, '0.25': 5.961899999999999, '0.5': 5.961899999999999, '0.75': 5.961899999999999, '0.95': 5.961899999999999}
- ⚠️ constant column (single value)

## lab_results_bun_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2064 (99.95%)
- min/max: 5.9 / 6.0
- mean/std: 5.9619 / None
- quantiles: {'0.05': 5.961899999999999, '0.25': 5.961899999999999, '0.5': 5.961899999999999, '0.75': 5.961899999999999, '0.95': 5.961899999999999}
- ⚠️ constant column (single value)

## lab_results_bun_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2064 (99.95%)
- min/max: 5.9 / 6.0
- mean/std: 5.9619 / None
- quantiles: {'0.05': 5.961899999999999, '0.25': 5.961899999999999, '0.5': 5.961899999999999, '0.75': 5.961899999999999, '0.95': 5.961899999999999}
- ⚠️ constant column (single value)

## lab_results_bun_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2064 (99.95%)
- min/max: 5.9 / 6.0
- mean/std: 5.9619 / None
- quantiles: {'0.05': 5.961899999999999, '0.25': 5.961899999999999, '0.5': 5.961899999999999, '0.75': 5.961899999999999, '0.95': 5.961899999999999}
- ⚠️ constant column (single value)

## lab_results_acr_value_stddev

- dtype: `Float64` (numeric)
- nulls: 2039 (98.74%)
- min/max: 0.0 / 0.0
- mean/std: 0.0000 / 0.0
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 0.0}
- ⚠️ constant column (single value)

## lab_results_acr_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 2039 (98.74%)
- min/max: 0.0 / 140.0
- mean/std: 16.8962 / 26.8341346910122
- quantiles: {'0.05': 0.175, '0.25': 3.8000000000000003, '0.5': 7.0, '0.75': 18.525, '0.95': 46.425}

## lab_results_acr_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2039 (98.74%)
- min/max: 0.0 / 140.0
- mean/std: 16.8962 / 26.8341346910122
- quantiles: {'0.05': 0.175, '0.25': 3.8, '0.5': 7.0, '0.75': 18.525, '0.95': 46.425}

## lab_results_acr_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2039 (98.74%)
- min/max: 0.0 / 140.0
- mean/std: 16.8962 / 26.8341346910122
- quantiles: {'0.05': 0.175, '0.25': 3.8, '0.5': 7.0, '0.75': 18.525, '0.95': 46.425}

## lab_results_acr_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2039 (98.74%)
- min/max: 0.0 / 140.0
- mean/std: 16.8962 / 26.8341346910122
- quantiles: {'0.05': 0.175, '0.25': 3.8, '0.5': 7.0, '0.75': 18.525, '0.95': 46.425}

## lab_results_acr_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2039 (98.74%)
- min/max: 0.0 / 140.0
- mean/std: 16.8962 / 26.8341346910122
- quantiles: {'0.05': 0.175, '0.25': 3.8, '0.5': 7.0, '0.75': 18.525, '0.95': 46.425}

## lab_results_ldl_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 2049 (99.23%)
- min/max: 0.71 / 4.9
- mean/std: 2.6956 / 1.1659157702567255
- quantiles: {'0.05': 1.325, '0.25': 1.89, '0.5': 2.48, '0.75': 3.3375, '0.95': 4.7325}

## lab_results_ldl_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2049 (99.23%)
- min/max: 0.71 / 4.9
- mean/std: 2.6956 / 1.1659157702567255
- quantiles: {'0.05': 1.325, '0.25': 1.89, '0.5': 2.48, '0.75': 3.3375, '0.95': 4.7325}

## lab_results_ldl_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2049 (99.23%)
- min/max: 0.71 / 4.9
- mean/std: 2.6956 / 1.1659157702567255
- quantiles: {'0.05': 1.325, '0.25': 1.89, '0.5': 2.48, '0.75': 3.3375, '0.95': 4.7325}

## lab_results_ldl_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2049 (99.23%)
- min/max: 0.71 / 4.9
- mean/std: 2.6956 / 1.1659157702567255
- quantiles: {'0.05': 1.325, '0.25': 1.89, '0.5': 2.48, '0.75': 3.3375, '0.95': 4.7325}

## lab_results_ldl_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2049 (99.23%)
- min/max: 0.71 / 4.9
- mean/std: 2.6956 / 1.1659157702567255
- quantiles: {'0.05': 1.325, '0.25': 1.89, '0.5': 2.48, '0.75': 3.3375, '0.95': 4.7325}

## lab_results_ldl_value_stddev

- dtype: `Float64` (numeric)
- nulls: 2049 (99.23%)
- min/max: 0.0 / 0.0
- mean/std: 0.0000 / 0.0
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 0.0}
- ⚠️ constant column (single value)

## lab_results_potassium_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 61 (2.95%)
- min/max: 0.0 / 7.7
- mean/std: 4.1289 / 0.7034105421253806
- quantiles: {'0.05': 3.2, '0.25': 3.775, '0.5': 4.1, '0.75': 4.5, '0.95': 5.2}

## lab_results_potassium_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 61 (2.95%)
- min/max: 0.0 / 7.1
- mean/std: 4.0748 / 0.7335897741174667
- quantiles: {'0.05': 3.1, '0.25': 3.7, '0.5': 4.1, '0.75': 4.5, '0.95': 5.184999999999991}

## lab_results_potassium_value_stddev

- dtype: `Float64` (numeric)
- nulls: 61 (2.95%)
- min/max: 0.0 / 2.7
- mean/std: 0.0540 / 0.22824646184557426
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 0.3484901951359266}

## lab_results_potassium_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 61 (2.95%)
- min/max: 0.0 / 7.7
- mean/std: 4.1870 / 0.6651776795766972
- quantiles: {'0.05': 3.3, '0.25': 3.8, '0.5': 4.2, '0.75': 4.5, '0.95': 5.3}

## lab_results_potassium_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 61 (2.95%)
- min/max: 0.0 / 7.1
- mean/std: 4.1307 / 0.6561319985374061
- quantiles: {'0.05': 3.2, '0.25': 3.75, '0.5': 4.1, '0.75': 4.5, '0.95': 5.2}

## lab_results_potassium_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 61 (2.95%)
- min/max: 0.0 / 7.1
- mean/std: 4.1351 / 0.6907721725151444
- quantiles: {'0.05': 3.2, '0.25': 3.8, '0.5': 4.1, '0.75': 4.5, '0.95': 5.2}

## lab_results_sodium_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 61 (2.95%)
- min/max: 110.0 / 170.0
- mean/std: 137.8977 / 4.703809730589292
- quantiles: {'0.05': 130.0, '0.25': 136.0, '0.5': 138.0, '0.75': 141.0, '0.95': 144.0}

## lab_results_sodium_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 61 (2.95%)
- min/max: 0.0 / 170.0
- mean/std: 137.1677 / 8.367843278505454
- quantiles: {'0.05': 129.0, '0.25': 135.0, '0.5': 138.0, '0.75': 141.0, '0.95': 144.0}

## lab_results_sodium_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 61 (2.95%)
- min/max: 0.0 / 170.0
- mean/std: 137.4117 / 7.760949354358149
- quantiles: {'0.05': 129.0, '0.25': 135.0, '0.5': 138.0, '0.75': 141.0, '0.95': 144.0}

## lab_results_sodium_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 61 (2.95%)
- min/max: 67.0 / 170.0
- mean/std: 137.5485 / 5.711988199523823
- quantiles: {'0.05': 129.0, '0.25': 135.0, '0.5': 138.0, '0.75': 141.0, '0.95': 144.0}

## lab_results_sodium_value_stddev

- dtype: `Float64` (numeric)
- nulls: 61 (2.95%)
- min/max: 0.0 / 70.0
- mean/std: 0.3562 / 3.438174060032299
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 1.5}

## lab_results_sodium_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 61 (2.95%)
- min/max: 0.0 / 170.0
- mean/std: 137.6612 / 5.662982055572544
- quantiles: {'0.05': 129.0, '0.25': 135.0, '0.5': 138.0, '0.75': 141.0, '0.95': 144.0}

## lab_results_albuminBS_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1343 (65.04%)
- min/max: 18.0 / 56.0
- mean/std: 38.6546 / 5.4110010093858705
- quantiles: {'0.05': 29.305, '0.25': 35.3, '0.5': 39.1, '0.75': 42.6, '0.95': 46.9}

## lab_results_albuminBS_value_stddev

- dtype: `Float64` (numeric)
- nulls: 1343 (65.04%)
- min/max: 0.0 / 4.6
- mean/std: 0.0226 / 0.24541437160707033
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 0.0}

## lab_results_albuminBS_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1343 (65.04%)
- min/max: 18.0 / 56.0
- mean/std: 38.6780 / 5.401245534637768
- quantiles: {'0.05': 29.405, '0.25': 35.3, '0.5': 39.1, '0.75': 42.6, '0.95': 46.9}

## lab_results_albuminBS_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1343 (65.04%)
- min/max: 18.0 / 56.0
- mean/std: 38.6327 / 5.409247307315748
- quantiles: {'0.05': 29.305, '0.25': 35.3, '0.5': 39.1, '0.75': 42.575, '0.95': 46.9}

## lab_results_albuminBS_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 1343 (65.04%)
- min/max: 18.0 / 56.0
- mean/std: 38.6552 / 5.3996590455934275
- quantiles: {'0.05': 29.405, '0.25': 35.300000000000004, '0.5': 39.1, '0.75': 42.575, '0.95': 46.9}

## lab_results_albuminBS_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1343 (65.04%)
- min/max: 18.0 / 56.0
- mean/std: 38.6551 / 5.399494767393373
- quantiles: {'0.05': 29.405, '0.25': 35.3, '0.5': 39.1, '0.75': 42.575, '0.95': 46.9}

## lab_results_hba1c%_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2065 (100.00%)
- All values are null.

## lab_results_hba1c%_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 2065 (100.00%)
- All values are null.

## lab_results_hba1c%_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2065 (100.00%)
- All values are null.

## lab_results_hba1c%_value_stddev

- dtype: `Float64` (numeric)
- nulls: 2065 (100.00%)
- All values are null.

## lab_results_hba1c%_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2065 (100.00%)
- All values are null.

## lab_results_hba1c%_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2065 (100.00%)
- All values are null.

## lab_results_hba1c_value_stddev

- dtype: `Float64` (numeric)
- nulls: 1531 (74.14%)
- min/max: 0.0 / 0.0
- mean/std: 0.0000 / 0.0
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 0.0}
- ⚠️ constant column (single value)

## lab_results_hba1c_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 1531 (74.14%)
- min/max: 24.0 / 130.0
- mean/std: 50.0206 / 15.151936786536199
- quantiles: {'0.05': 36.0, '0.25': 40.0, '0.5': 45.0, '0.75': 55.0, '0.95': 82.0}

## lab_results_hba1c_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1531 (74.14%)
- min/max: 24.0 / 130.0
- mean/std: 50.0206 / 15.151936786536199
- quantiles: {'0.05': 36.0, '0.25': 40.0, '0.5': 45.0, '0.75': 55.0, '0.95': 82.0}

## lab_results_hba1c_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1531 (74.14%)
- min/max: 24.0 / 130.0
- mean/std: 50.0206 / 15.151936786536199
- quantiles: {'0.05': 36.0, '0.25': 40.0, '0.5': 45.0, '0.75': 55.0, '0.95': 82.0}

## lab_results_hba1c_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1531 (74.14%)
- min/max: 24.0 / 130.0
- mean/std: 50.0206 / 15.151936786536199
- quantiles: {'0.05': 36.0, '0.25': 40.0, '0.5': 45.0, '0.75': 55.0, '0.95': 82.0}

## lab_results_hba1c_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1531 (74.14%)
- min/max: 24.0 / 130.0
- mean/std: 50.0206 / 15.151936786536199
- quantiles: {'0.05': 36.0, '0.25': 40.0, '0.5': 45.0, '0.75': 55.0, '0.95': 82.0}

## lab_results_validSerumCreatinine_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 190 (9.20%)
- min/max: 3.3 / 23.0
- mean/std: 12.1938 / 3.897419074496976
- quantiles: {'0.05': 7.006, '0.25': 9.266, '0.5': 11.413, '0.75': 14.577, '0.95': 20.034899999999997}

## lab_results_validSerumCreatinine_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 190 (9.20%)
- min/max: 3.3 / 23.0
- mean/std: 12.3043 / 3.9224168129882924
- quantiles: {'0.05': 7.006, '0.25': 9.379, '0.5': 11.526, '0.75': 14.69, '0.95': 20.34}

## lab_results_validSerumCreatinine_value_stddev

- dtype: `Float64` (numeric)
- nulls: 190 (9.20%)
- min/max: 0.0 / 6.0
- mean/std: 0.0530 / 0.24084668954093413
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 0.3955000000000002}

## lab_results_validSerumCreatinine_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 190 (9.20%)
- min/max: 3.3 / 23.0
- mean/std: 12.2553 / 3.915043015023447
- quantiles: {'0.05': 7.006, '0.25': 9.266, '0.5': 11.413, '0.75': 14.577, '0.95': 20.34}

## lab_results_validSerumCreatinine_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 190 (9.20%)
- min/max: 3.3 / 23.0
- mean/std: 12.2506 / 3.9027816085739695
- quantiles: {'0.05': 7.005999999999999, '0.25': 9.379000000000001, '0.5': 11.413, '0.75': 14.577, '0.95': 20.18745}

## lab_results_valideGFR_value_stddev

- dtype: `Float64` (numeric)
- nulls: 78 (3.78%)
- min/max: 0.0 / 24.0
- mean/std: 0.2624 / 1.215743214073765
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 1.789905651979663}

## lab_results_valideGFR_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 78 (3.78%)
- min/max: 1.0 / 150.0
- mean/std: 54.4169 / 24.652104057633803
- quantiles: {'0.05': 16.380000000000006, '0.25': 36.0, '0.5': 52.2, '0.75': 72.9, '0.95': 96.0}

## lab_results_valideGFR_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 78 (3.78%)
- min/max: 1.0 / 150.0
- mean/std: 54.4371 / 24.6387525951972
- quantiles: {'0.05': 16.200000000000003, '0.25': 36.0, '0.5': 52.2, '0.75': 72.6, '0.95': 96.0}

## lab_results_valideGFR_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 78 (3.78%)
- min/max: 1.0 / 150.0
- mean/std: 54.1710 / 24.591152793272496
- quantiles: {'0.05': 16.2, '0.25': 36.0, '0.5': 52.2, '0.75': 72.0, '0.95': 96.0}

## lab_results_valideGFR_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 78 (3.78%)
- min/max: 1.0 / 150.0
- mean/std: 54.4229 / 24.69798328597717
- quantiles: {'0.05': 16.2, '0.25': 36.0, '0.5': 52.2, '0.75': 72.6, '0.95': 96.6}

## lab_results_valideGFR_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 78 (3.78%)
- min/max: 1.0 / 150.0
- mean/std: 54.7204 / 24.758179812947688
- quantiles: {'0.05': 16.8, '0.25': 36.6, '0.5': 52.2, '0.75': 73.8, '0.95': 96.6}

## symptoms_Ankle_swelling_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1633
  - `True`: 432

## symptoms_Ascites_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2052
  - `True`: 13

## symptoms_Breathlessness_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1953
  - `True`: 112

## symptoms_Cardiac_murmur_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2063
  - 1 other distinct value(s) suppressed, covering 2 row(s) (count below 5 and/or ranked beyond top 20)

## symptoms_Chest_pain_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2019
  - `True`: 46

## symptoms_Cheyne_stokes_respiration_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 2065

## symptoms_Depression_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1745
  - `True`: 320

## symptoms_Dizziness_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2036
  - `True`: 29

## symptoms_Elevated_jugular_venous_pressure_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 2065

## symptoms_Fatigue_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2052
  - `True`: 13

## symptoms_Hepatojugular_reflux_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 2065

## symptoms_Hepatomegaly_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2051
  - `True`: 14

## symptoms_Intermittent_claudication_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 2065

## symptoms_Irregular_pulse_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2058
  - `True`: 7

## symptoms_Loss_of_appetite_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 2065

## symptoms_Nocturnal_cough_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 2065

## symptoms_Oliguria_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 2065

## symptoms_Orthopnoea_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2062
  - 1 other distinct value(s) suppressed, covering 3 row(s) (count below 5 and/or ranked beyond top 20)

## symptoms_Palpitations_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2063
  - 1 other distinct value(s) suppressed, covering 2 row(s) (count below 5 and/or ranked beyond top 20)

## symptoms_Paroxysmal_nocturnal_dyspnea_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2064
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## symptoms_Peripheral_edema_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1795
  - `True`: 270

## symptoms_Pleural_effusion_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1942
  - `True`: 123

## symptoms_Pulmonary_crepitations_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2043
  - `True`: 22

## symptoms_Reduced_exercise_tolerance_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 2065

## symptoms_Syncope_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2052
  - `True`: 13

## symptoms_Tachycardia_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2001
  - `True`: 64

## symptoms_Tachypnoea_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2063
  - 1 other distinct value(s) suppressed, covering 2 row(s) (count below 5 and/or ranked beyond top 20)

## symptoms_Third_heart_sound_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2060
  - `True`: 5

## symptoms_Weight_gain_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 2065

## symptoms_Weight_loss_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 2065

## echocardiographs_lvef

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 820 (39.71%)
- min/max: 8.0 / 120.0
- mean/std: 39.9896 / 16.03580359667641
- quantiles: {'0.05': 15.0, '0.25': 25.0, '0.5': 40.0, '0.75': 55.0, '0.95': 65.0}

## echocardiographs_lvef_pET_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1077 (52.15%)
- min/max: 0.8 / 75.0
- mean/std: 38.2852 / 15.628490345754694
- quantiles: {'0.05': 15.0, '0.25': 25.0, '0.5': 36.5, '0.75': 50.0, '0.95': 65.0}

## echocardiographs_lvef_pET_stddev

- dtype: `Float64` (numeric)
- nulls: 1072 (51.91%)
- min/max: 0.0 / 22.0
- mean/std: 0.7203 / 2.221192116364401
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 5.0}

## echocardiographs_lvef_pET_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1072 (51.91%)
- min/max: 10.0 / 75.0
- mean/std: 39.2115 / 15.594327608213359
- quantiles: {'0.05': 15.0, '0.25': 25.0, '0.5': 38.0, '0.75': 50.0, '0.95': 65.0}

## echocardiographs_lvef_pET_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1072 (51.91%)
- min/max: 0.8 / 75.0
- mean/std: 37.6937 / 15.588510619584756
- quantiles: {'0.05': 15.0, '0.25': 25.0, '0.5': 35.0, '0.75': 50.0, '0.95': 65.0}

## echocardiographs_lvef_pET_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1085 (52.54%)
- min/max: 8.0 / 75.0
- mean/std: 38.6163 / 15.609697451278146
- quantiles: {'0.05': 15.0, '0.25': 25.0, '0.5': 37.0, '0.75': 50.0, '0.95': 65.0}

## echocardiographs_lvef_pET_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 1072 (51.91%)
- min/max: 10.0 / 75.0
- mean/std: 38.4406 / 15.424216735470711
- quantiles: {'0.05': 15.0, '0.25': 25.0, '0.5': 37.0, '0.75': 50.0, '0.95': 65.0}

## electrocardiographs_ecg_qrs_duration_pET_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1142 (55.30%)
- min/max: 25.0 / 250.0
- mean/std: 123.8559 / 37.24926057761084
- quantiles: {'0.05': 84.0, '0.25': 96.0, '0.5': 110.0, '0.75': 145.0, '0.95': 200.89999999999998}

## electrocardiographs_ecg_qrs_duration_pET_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1142 (55.30%)
- min/max: 9.0 / 240.0
- mean/std: 116.2774 / 35.43352513698167
- quantiles: {'0.05': 79.1, '0.25': 92.0, '0.5': 106.0, '0.75': 134.0, '0.95': 188.0}

## electrocardiographs_ecg_qrs_duration_pET_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1142 (55.30%)
- min/max: 62.0 / 330.0
- mean/std: 130.7302 / 39.367833013711426
- quantiles: {'0.05': 86.1, '0.25': 101.0, '0.5': 118.0, '0.75': 153.0, '0.95': 210.0}

## electrocardiographs_ecg_qrs_duration_pET_stddev

- dtype: `Float64` (numeric)
- nulls: 1142 (55.30%)
- min/max: 0.0 / 90.0
- mean/std: 5.8316 / 9.001504209964711
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 3.1972210155418126, '0.75': 6.815758934574301, '0.95': 24.95090056514883}

## electrocardiographs_ecg_qrs_duration_pET_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 1142 (55.30%)
- min/max: 62.0 / 260.0
- mean/std: 123.1818 / 35.471070040569415
- quantiles: {'0.05': 84.0, '0.25': 96.0, '0.5': 111.8, '0.75': 141.83333333333334, '0.95': 198.0}

## electrocardiographs_ecg_qrs_duration_pET_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1143 (55.35%)
- min/max: 21.0 / 280.0
- mean/std: 123.2570 / 36.63980971565873
- quantiles: {'0.05': 84.0, '0.25': 97.0, '0.5': 112.0, '0.75': 142.0, '0.95': 200.0}

## electrocardiographs_ecg_qrs_axis_pET_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 1144 (55.40%)
- min/max: -87.0 / 270.0
- mean/std: 10.5042 / 64.79327953952486
- quantiles: {'0.05': -70.0, '0.25': -39.0, '0.5': 1.625, '0.75': 43.666666666666664, '0.95': 133.5}

## electrocardiographs_ecg_qrs_axis_pET_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1151 (55.74%)
- min/max: -88.0 / 270.0
- mean/std: 11.4136 / 75.95610128792399
- quantiles: {'0.05': -73.35, '0.25': -44.75, '0.5': -3.5, '0.75': 42.0, '0.95': 177.3499999999999}

## electrocardiographs_ecg_qrs_axis_pET_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1144 (55.40%)
- min/max: -87.0 / 270.0
- mean/std: 29.9555 / 83.5164730874537
- quantiles: {'0.05': -66.0, '0.25': -32.0, '0.5': 14.0, '0.75': 64.0, '0.95': 235.0}

## electrocardiographs_ecg_qrs_axis_pET_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1148 (55.59%)
- min/max: -89.0 / 270.0
- mean/std: 10.2236 / 72.2889426879791
- quantiles: {'0.05': -73.0, '0.25': -45.0, '0.5': -3.0, '0.75': 44.0, '0.95': 158.7999999999994}

## electrocardiographs_ecg_qrs_axis_pET_stddev

- dtype: `Float64` (numeric)
- nulls: 1144 (55.40%)
- min/max: 0.0 / 180.0
- mean/std: 15.0982 / 31.199937697777077
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 4.5, '0.75': 11.440668201153676, '0.95': 97.37241783535028}

## electrocardiographs_ecg_qrs_axis_pET_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1144 (55.40%)
- min/max: -89.0 / 270.0
- mean/std: -6.0945 / 63.23392373249949
- quantiles: {'0.05': -79.0, '0.25': -52.0, '0.5': -18.0, '0.75': 23.0, '0.95': 107.0}

## electrocardiographs_ecg_qt_duration_corrected_pET_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1142 (55.30%)
- min/max: 200.0 / 710.0
- mean/std: 487.6024 / 50.11911056443754
- quantiles: {'0.05': 420.0, '0.25': 454.0, '0.5': 484.0, '0.75': 516.0, '0.95': 572.0}

## electrocardiographs_ecg_qt_duration_corrected_pET_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 1142 (55.30%)
- min/max: 200.0 / 640.0
- mean/std: 467.9697 / 44.06385933558639
- quantiles: {'0.05': 408.0, '0.25': 440.9166666666667, '0.5': 463.0, '0.75': 493.0, '0.95': 547.0}

## electrocardiographs_ecg_qt_duration_corrected_pET_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1142 (55.30%)
- min/max: 160.0 / 640.0
- mean/std: 447.7963 / 54.178660296609245
- quantiles: {'0.05': 376.1, '0.25': 422.0, '0.5': 446.0, '0.75': 476.0, '0.95': 532.8}

## electrocardiographs_ecg_qt_duration_corrected_pET_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1143 (55.35%)
- min/max: 180.0 / 690.0
- mean/std: 466.5510 / 50.51079054067504
- quantiles: {'0.05': 397.05, '0.25': 437.0, '0.5': 463.0, '0.75': 495.0, '0.95': 551.9499999999999}

## electrocardiographs_ecg_qt_duration_corrected_pET_stddev

- dtype: `Float64` (numeric)
- nulls: 1142 (55.30%)
- min/max: 0.0 / 160.0
- mean/std: 15.7028 / 19.330869501181116
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 10.498677165349088, '0.75': 22.481756040352728, '0.95': 52.58797821048013}

## electrocardiographs_ecg_qt_duration_corrected_pET_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1142 (55.30%)
- min/max: 200.0 / 670.0
- mean/std: 468.2546 / 50.061168310307224
- quantiles: {'0.05': 403.1, '0.25': 437.5, '0.5': 464.0, '0.75': 498.0, '0.95': 553.0}

## electrocardiographs_ecg_st_pET

- dtype: `Boolean` (boolean)
- nulls: 1148 (55.59%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 1148
  - `False`: 896
  - `True`: 21

## electrocardiographs_ecg_ischemia_without_st_pET

- dtype: `Boolean` (boolean)
- nulls: 1148 (55.59%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 1148
  - `False`: 898
  - `True`: 19

## electrocardiographs_ecg_type_of_rhythms_pET_first

- dtype: `List(String)` (categorical)
- nulls: 1141 (55.25%)
- unique values: 22
- top values (shown only where count ≥ 5):
  - null (missing): 1141
  - `['LA17059-9', 'LA17718-0']`: 270
  - `['LA17059-9', 'LA17084-7']`: 250
  - `['LA17059-9']`: 182
  - `['LA17059-9', 'LA17099-5']`: 51
  - `['LA17059-9', 'LA17095-3', 'LA17718-0']`: 44
  - `['LA17059-9', 'LA17098-7']`: 37
  - `['LA17059-9', 'LA17094-6', 'LA17718-0']`: 29
  - `['LA17059-9', 'LA17097-9']`: 14
  - `['LA17059-9', 'LA17094-6', 'LA17095-3', 'LA17718-0']`: 13
  - `['LA17059-9', 'LA17095-3', 'LA17099-5']`: 8
  - `['LA17059-9', 'LA17085-4', 'LA17099-5']`: 6
  - `['LA17059-9', 'LA17094-6', 'LA17099-5']`: 5
  - 9 other distinct value(s) suppressed, covering 15 row(s) (count below 5 and/or ranked beyond top 20)

## electrocardiographs_ecg_type_of_rhythms_pET_last

- dtype: `List(String)` (categorical)
- nulls: 1141 (55.25%)
- unique values: 18
- top values (shown only where count ≥ 5):
  - null (missing): 1141
  - `['LA17059-9', 'LA17718-0']`: 317
  - `['LA17059-9']`: 219
  - `['LA17059-9', 'LA17084-7']`: 207
  - `['LA17059-9', 'LA17094-6', 'LA17718-0']`: 39
  - `['LA17059-9', 'LA17098-7']`: 36
  - `['LA17059-9', 'LA17095-3', 'LA17718-0']`: 34
  - `['LA17059-9', 'LA17099-5']`: 26
  - `['LA17059-9', 'LA17097-9']`: 15
  - `['LA17059-9', 'LA17094-6', 'LA17095-3', 'LA17718-0']`: 9
  - `['LA17059-9', 'LA17094-6', 'LA17099-5']`: 6
  - `['LA17059-9', 'LA17095-3', 'LA17099-5']`: 6
  - 6 other distinct value(s) suppressed, covering 10 row(s) (count below 5 and/or ranked beyond top 20)

## smoking_status_smoker_last

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1667
  - `True`: 398

## smoking_status_formerSmoker_last

- dtype: `Boolean` (boolean)
- nulls: 640 (30.99%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 743
  - `True`: 682
  - null (missing): 640

## smoking_status_smoker_totalSmokingDuration_sum

- dtype: `Int64` (numeric)
- nulls: 1589 (76.95%)
- min/max: 0.0 / 31000.0
- mean/std: 872.9853 / 2228.8412296867855
- quantiles: {'0.05': 1.0, '0.25': 4.0, '0.5': 13.5, '0.75': 812.0, '0.95': 4384.75}

## smoking_status_smoker_startTime_count

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 15.0
- mean/std: 0.4029 / 1.053046191893137
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 2.0}

## nyha_nyha

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `LA28405-1`: 1208
  - `LA28406-9`: 803
  - `LA28404-4`: 54

## nyha_nyha_pET

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 3
- top values (shown only where count ≥ 5):
  - `LA28405-1`: 1138
  - `LA28406-9`: 855
  - `LA28404-4`: 72

## vital_signs_systolicBpDuringEncounter_value_pET

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 0 (0.00%)
- min/max: 60.0 / 230.0
- mean/std: 129.3101 / 17.60202276907315
- quantiles: {'0.05': 100.2, '0.25': 120.0, '0.5': 128.29355177418137, '0.75': 137.04476072292678, '0.95': 160.0}

## vital_signs_bmi_value_pET

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 0 (0.00%)
- min/max: 15.0 / 440.0
- mean/std: 28.6618 / 15.118781428294032
- quantiles: {'0.05': 19.84517695061474, '0.25': 23.875114784205692, '0.5': 27.379664683612763, '0.75': 31.141868512110726, '0.95': 38.62184763290578}

## lab_results_creatBS_value_p3a_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 0 (0.00%)
- min/max: 3.7 / 87.0
- mean/std: 13.9429 / 7.587868038994555
- quantiles: {'0.05': 7.29528, '0.25': 9.680333333333333, '0.5': 12.091, '0.75': 15.661799999999998, '0.95': 26.540471428571426}

## lab_results_validSerumCreatinine_value_pET

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 0 (0.00%)
- min/max: 3.3 / 23.0
- mean/std: 12.5272 / 3.927252612022608
- quantiles: {'0.05': 7.232, '0.25': 9.605, '0.5': 11.751999999999999, '0.75': 14.802999999999999, '0.95': 20.34}

## hyperkalemia_severity_categorizedValue

- dtype: `String` (categorical)
- nulls: 61 (2.95%)
- unique values: 5
- top values (shown only where count ≥ 5):
  - `normal`: 1827
  - `mild`: 123
  - null (missing): 61
  - `moderate`: 37
  - `severe`: 17

## ckd_severity_categorizedValue

- dtype: `String` (categorical)
- nulls: 78 (3.78%)
- unique values: 7
- top values (shown only where count ≥ 5):
  - `mildly_decreased`: 607
  - `mild_to_moderate_decrease`: 462
  - `moderate_to_severe_decrease`: 388
  - `severe_decrease`: 250
  - `normal_or_high`: 197
  - `kidney_failure`: 83
  - null (missing): 78

## med_requests_activeDuringEncounter_bb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 1619
  - `False`: 446

## med_requests_activeDuringEncounter_ace_inhibitors_arb_use_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 1555
  - `False`: 510

## conditions_heartFailure_timeFromEarliest_first

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 150.0
- mean/std: 5.7351 / 18.33675229660521
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 41.799999999999955}

## conditions_heart_failure_hf_within_18mo_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `True`: 2065

## conditions_heart_failure_occurred_prior_to_18_months_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1875
  - `True`: 190

## conditions_ap_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2044
  - `True`: 21

## conditions_af_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1104
  - `True`: 961

## conditions_cm_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1926
  - `True`: 139

## conditions_dysl_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1177
  - `True`: 888

## conditions_hf_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `True`: 2065

## conditions_hyp_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 1586
  - `False`: 479

## conditions_ihd_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 1085
  - `False`: 980

## conditions_mi_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1438
  - `True`: 627

## conditions_pad_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1699
  - `True`: 366

## conditions_stroke_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1882
  - `True`: 183

## conditions_tia_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2023
  - `True`: 42

## conditions_vd_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1872
  - `True`: 193

## conditions_revasc_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2045
  - `True`: 20

## conditions_devices_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1634
  - `True`: 431

## conditions_aidshiv_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 2065

## conditions_copd_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1704
  - `True`: 361

## conditions_diabetes_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1208
  - `True`: 857

## conditions_dem_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2048
  - `True`: 17

## conditions_dep_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2011
  - `True`: 54

## conditions_dia_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 2065

## conditions_hyperthyroid_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2053
  - `True`: 12

## conditions_hypothyroid_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1907
  - `True`: 158

## conditions_ibd_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2054
  - `True`: 11

## conditions_ld_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2053
  - `True`: 12

## conditions_mc_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2038
  - `True`: 27

## conditions_osa_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2057
  - `True`: 8

## conditions_rd_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2039
  - `True`: 26

## conditions_ckd_chronic_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1924
  - `True`: 141

## conditions_myocarditis_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2045
  - `True`: 20

## conditions_pericardial_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1967
  - `True`: 98

## conditions_substance_abuse_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2056
  - `True`: 9

## conditions_ap_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1866
  - `True`: 199

## conditions_af_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 1037
  - `False`: 1028

## conditions_cm_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1766
  - `True`: 299

## conditions_dysl_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1095
  - `True`: 970

## conditions_hf_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `True`: 2065

## conditions_hyp_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 1664
  - `False`: 401

## conditions_ihd_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 1461
  - `False`: 604

## conditions_mi_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1316
  - `True`: 749

## conditions_pad_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1596
  - `True`: 469

## conditions_stroke_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1817
  - `True`: 248

## conditions_tia_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1958
  - `True`: 107

## conditions_vd_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1766
  - `True`: 299

## conditions_revasc_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2034
  - `True`: 31

## conditions_devices_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1526
  - `True`: 539

## conditions_aidshiv_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 2065

## conditions_copd_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1669
  - `True`: 396

## conditions_diabetes_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1191
  - `True`: 874

## conditions_dem_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2040
  - `True`: 25

## conditions_dep_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1987
  - `True`: 78

## conditions_dia_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 2065

## conditions_hyperthyroid_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2049
  - `True`: 16

## conditions_hypothyroid_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1883
  - `True`: 182

## conditions_ibd_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2047
  - `True`: 18

## conditions_ld_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2041
  - `True`: 24

## conditions_mc_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1985
  - `True`: 80

## conditions_osa_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2038
  - `True`: 27

## conditions_rd_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2034
  - `True`: 31

## conditions_ckd_chronic_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1845
  - `True`: 220

## conditions_myocarditis_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2039
  - `True`: 26

## conditions_pericardial_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1923
  - `True`: 142

## conditions_substance_abuse_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2050
  - `True`: 15

## conditions_af_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 1052
  - `False`: 1013

## conditions_ckd_chronic_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1844
  - `True`: 221

## conditions_cm_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1731
  - `True`: 334

## conditions_copd_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1669
  - `True`: 396

## conditions_dem_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2040
  - `True`: 25

## conditions_dep_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1987
  - `True`: 78

## conditions_diabetes_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1191
  - `True`: 874

## conditions_hypothyroid_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1883
  - `True`: 182

## conditions_hyp_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 1666
  - `False`: 399

## conditions_ihd_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 1555
  - `False`: 510

## conditions_mc_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1982
  - `True`: 83

## conditions_mi_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1287
  - `True`: 778

## conditions_pad_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1590
  - `True`: 475

## conditions_rd_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2034
  - `True`: 31

## conditions_stroke_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1815
  - `True`: 250

## conditions_vd_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1756
  - `True`: 309

## med_admins_rasi_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1928
  - `True`: 137

## med_admins_arni_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 2065

## med_admins_acei_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1965
  - `True`: 100

## med_admins_arb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2024
  - `True`: 41

## med_admins_mra_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1894
  - `True`: 171

## med_admins_diuretics_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1165
  - `True`: 900

## med_admins_diuretics_loop_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1185
  - `True`: 880

## med_admins_anti_coag_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1227
  - `True`: 838

## med_admins_anti_plat_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2031
  - `True`: 34

## med_admins_thrombolytic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 2065

## med_admins_bb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1900
  - `True`: 165

## med_admins_ccb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2011
  - `True`: 54

## med_admins_digitalis_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1980
  - `True`: 85

## med_admins_antiarrhytmic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2027
  - `True`: 38

## med_admins_inotropes_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2064
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## med_admins_vasodil_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2055
  - `True`: 10

## med_admins_platelet_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2031
  - `True`: 34

## med_admins_ll_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1983
  - `True`: 82

## med_admins_ivabradine_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2063
  - 1 other distinct value(s) suppressed, covering 2 row(s) (count below 5 and/or ranked beyond top 20)

## med_admins_potassium_binders_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 2065

## med_admins_insulins_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2054
  - `True`: 11

## med_admins_oral_antidiabetic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2026
  - `True`: 39

## med_admins_sglt2i_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2064
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## med_admins_ari_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 2065

## med_admins_rdoad_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2060
  - `True`: 5

## med_admins_rdoad_syst_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2046
  - `True`: 19

## med_admins_cortico_syst_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 2065

## med_admins_antiinfl_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2060
  - `True`: 5

## med_requests_rasi_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 1319
  - `False`: 746

## med_requests_arni_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1967
  - `True`: 98

## med_requests_acei_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1108
  - `True`: 957

## med_requests_arb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1693
  - `True`: 372

## med_requests_mra_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 1245
  - `False`: 820

## med_requests_diuretics_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 1624
  - `False`: 441

## med_requests_diuretics_loop_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 1410
  - `False`: 655

## med_requests_anti_coag_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1266
  - `True`: 799

## med_requests_anti_plat_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1493
  - `True`: 572

## med_requests_thrombolytic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2064
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## med_requests_bb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 1430
  - `False`: 635

## med_requests_ccb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1711
  - `True`: 354

## med_requests_digitalis_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1755
  - `True`: 310

## med_requests_antiarrhytmic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1761
  - `True`: 304

## med_requests_inotropes_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2057
  - `True`: 8

## med_requests_vasodil_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2024
  - `True`: 41

## med_requests_platelet_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1493
  - `True`: 572

## med_requests_ll_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 1079
  - `False`: 986

## med_requests_ivabradine_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2045
  - `True`: 20

## med_requests_potassium_binders_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2063
  - 1 other distinct value(s) suppressed, covering 2 row(s) (count below 5 and/or ranked beyond top 20)

## med_requests_insulins_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1970
  - `True`: 95

## med_requests_oral_antidiabetic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1432
  - `True`: 633

## med_requests_sglt2i_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1689
  - `True`: 376

## med_requests_ari_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 2065

## med_requests_rdoad_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1937
  - `True`: 128

## med_requests_rdoad_syst_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1972
  - `True`: 93

## med_requests_cortico_syst_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 2065

## med_requests_antiinfl_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2059
  - `True`: 6

## med_admins_history_rasi_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1896
  - `True`: 169

## med_admins_history_arni_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 2065

## med_admins_history_acei_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1942
  - `True`: 123

## med_admins_history_arb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2014
  - `True`: 51

## med_admins_history_mra_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1945
  - `True`: 120

## med_admins_history_diuretics_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1632
  - `True`: 433

## med_admins_history_diuretics_loop_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1649
  - `True`: 416

## med_admins_history_anti_coag_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1604
  - `True`: 461

## med_admins_history_anti_plat_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2033
  - `True`: 32

## med_admins_history_thrombolytic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 2065

## med_admins_history_bb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1904
  - `True`: 161

## med_admins_history_ccb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1994
  - `True`: 71

## med_admins_history_digitalis_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2007
  - `True`: 58

## med_admins_history_antiarrhytmic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2020
  - `True`: 45

## med_admins_history_inotropes_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2064
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## med_admins_history_vasodil_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2048
  - `True`: 17

## med_admins_history_platelet_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2033
  - `True`: 32

## med_admins_history_ll_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1956
  - `True`: 109

## med_admins_history_ivabradine_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2063
  - 1 other distinct value(s) suppressed, covering 2 row(s) (count below 5 and/or ranked beyond top 20)

## med_admins_history_potassium_binders_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 2065

## med_admins_history_insulins_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2047
  - `True`: 18

## med_admins_history_oral_antidiabetic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2006
  - `True`: 59

## med_admins_history_sglt2i_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2063
  - 1 other distinct value(s) suppressed, covering 2 row(s) (count below 5 and/or ranked beyond top 20)

## med_admins_history_ari_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 2065

## med_admins_history_rdoad_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2054
  - `True`: 11

## med_admins_history_rdoad_syst_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2042
  - `True`: 23

## med_admins_history_cortico_syst_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 2065

## med_admins_history_antiinfl_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2061
  - 1 other distinct value(s) suppressed, covering 4 row(s) (count below 5 and/or ranked beyond top 20)

## med_requests_history_rasi_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1373
  - `True`: 692

## med_requests_history_arni_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2018
  - `True`: 47

## med_requests_history_acei_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1543
  - `True`: 522

## med_requests_history_arb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1829
  - `True`: 236

## med_requests_history_mra_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1531
  - `True`: 534

## med_requests_history_diuretics_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1337
  - `True`: 728

## med_requests_history_diuretics_loop_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1453
  - `True`: 612

## med_requests_history_anti_coag_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1632
  - `True`: 433

## med_requests_history_anti_plat_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1792
  - `True`: 273

## med_requests_history_thrombolytic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 2065

## med_requests_history_bb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1377
  - `True`: 688

## med_requests_history_ccb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1765
  - `True`: 300

## med_requests_history_digitalis_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1889
  - `True`: 176

## med_requests_history_antiarrhytmic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1861
  - `True`: 204

## med_requests_history_inotropes_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2060
  - `True`: 5

## med_requests_history_vasodil_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2014
  - `True`: 51

## med_requests_history_platelet_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1792
  - `True`: 273

## med_requests_history_ll_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1539
  - `True`: 526

## med_requests_history_ivabradine_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2050
  - `True`: 15

## med_requests_history_potassium_binders_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2064
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## med_requests_history_insulins_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1979
  - `True`: 86

## med_requests_history_oral_antidiabetic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1743
  - `True`: 322

## med_requests_history_sglt2i_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1941
  - `True`: 124

## med_requests_history_ari_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 2065

## med_requests_history_rdoad_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1963
  - `True`: 102

## med_requests_history_rdoad_syst_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1970
  - `True`: 95

## med_requests_history_cortico_syst_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 2065

## med_requests_history_antiinfl_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 2049
  - `True`: 16

## encounter_primary_reason_HF_Disease_f5a_w7d_first

- dtype: `Boolean` (boolean)
- nulls: 1823 (88.28%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 1823
  - `False`: 229
  - `True`: 13

## encounter_primary_reason_HF_Disease_f5a_w1mo_first

- dtype: `Boolean` (boolean)
- nulls: 1470 (71.19%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 1470
  - `False`: 565
  - `True`: 30

## encounter_primary_reason_HF_Disease_f5a_w3mo_first

- dtype: `Boolean` (boolean)
- nulls: 1030 (49.88%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 1030
  - `False`: 990
  - `True`: 45

## encounter_primary_reason_HF_Disease_f5a_w6mo_first

- dtype: `Boolean` (boolean)
- nulls: 906 (43.87%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1114
  - null (missing): 906
  - `True`: 45

## encounter_primary_reason_HF_Disease_f5a_w1a_first

- dtype: `Boolean` (boolean)
- nulls: 817 (39.56%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1202
  - null (missing): 817
  - `True`: 46

## encounter_primary_reason_HF_Disease_f5a_w3a_first

- dtype: `Boolean` (boolean)
- nulls: 706 (34.19%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1312
  - null (missing): 706
  - `True`: 47

## encounter_primary_reason_HF_Disease_f5a_w5a_first

- dtype: `Boolean` (boolean)
- nulls: 677 (32.78%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1341
  - null (missing): 677
  - `True`: 47

## encounter_primary_reason_number_of_days_to_rehosp_for_heart_failure_f5a_first

- dtype: `Int32` (numeric)
- nulls: 2018 (97.72%)
- min/max: 1.0 / 850.0
- mean/std: 50.7447 / 128.02436386419078
- quantiles: {'0.05': 3.0, '0.25': 6.5, '0.5': 21.0, '0.75': 48.0, '0.95': 84.0}

## encounter_primary_reason_number_of_HF_rehospitalizations_5a_f5a_count

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 87.0
- mean/std: 0.3763 / 2.4704295055569103
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 2.0}

## encounter_primary_reason_CV_Disease_f5a_w7d_first

- dtype: `Boolean` (boolean)
- nulls: 1823 (88.28%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 1823
  - `True`: 129
  - `False`: 113

## encounter_primary_reason_CV_Disease_f5a_w1mo_first

- dtype: `Boolean` (boolean)
- nulls: 1470 (71.19%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 1470
  - `True`: 305
  - `False`: 290

## encounter_primary_reason_CV_Disease_f5a_w3mo_first

- dtype: `Boolean` (boolean)
- nulls: 1030 (49.88%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 1030
  - `False`: 536
  - `True`: 499

## encounter_primary_reason_CV_Disease_f5a_w6mo_first

- dtype: `Boolean` (boolean)
- nulls: 906 (43.87%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 906
  - `False`: 590
  - `True`: 569

## encounter_primary_reason_CV_Disease_f5a_w1a_first

- dtype: `Boolean` (boolean)
- nulls: 817 (39.56%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 817
  - `False`: 636
  - `True`: 612

## encounter_primary_reason_CV_Disease_f5a_w3a_first

- dtype: `Boolean` (boolean)
- nulls: 706 (34.19%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 706
  - `False`: 688
  - `True`: 671

## encounter_primary_reason_CV_Disease_f5a_w5a_first

- dtype: `Boolean` (boolean)
- nulls: 677 (32.78%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 704
  - `True`: 684
  - null (missing): 677

## encounter_primary_reason_number_of_days_to_rehosp_for_CV_f5a_first

- dtype: `Int32` (numeric)
- nulls: 1381 (66.88%)
- min/max: 1.0 / 1700.0
- mean/std: 132.0395 / 261.9789199834246
- quantiles: {'0.05': 2.0, '0.25': 14.0, '0.5': 35.0, '0.75': 96.25, '0.95': 726.85}

## encounter_primary_reason_number_of_CV_rehospitalizations_5a_f5a_count

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 120.0
- mean/std: 5.3932 / 9.874716782435181
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 1.0, '0.75': 7.0, '0.95': 25.0}

## encounter_primary_reason_non_CV_Disease_f5a_w7d_first

- dtype: `Boolean` (boolean)
- nulls: 1823 (88.28%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 1823
  - `False`: 129
  - `True`: 113

## encounter_primary_reason_non_CV_Disease_f5a_w1mo_first

- dtype: `Boolean` (boolean)
- nulls: 1470 (71.19%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 1470
  - `False`: 305
  - `True`: 290

## encounter_primary_reason_non_CV_Disease_f5a_w3mo_first

- dtype: `Boolean` (boolean)
- nulls: 1030 (49.88%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 1030
  - `True`: 536
  - `False`: 499

## encounter_primary_reason_non_CV_Disease_f5a_w6mo_first

- dtype: `Boolean` (boolean)
- nulls: 906 (43.87%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 906
  - `True`: 590
  - `False`: 569

## encounter_primary_reason_non_CV_Disease_f5a_w1a_first

- dtype: `Boolean` (boolean)
- nulls: 817 (39.56%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 817
  - `True`: 636
  - `False`: 612

## encounter_primary_reason_non_CV_Disease_f5a_w3a_first

- dtype: `Boolean` (boolean)
- nulls: 706 (34.19%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 706
  - `True`: 688
  - `False`: 671

## encounter_primary_reason_non_CV_Disease_f5a_w5a_first

- dtype: `Boolean` (boolean)
- nulls: 677 (32.78%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 704
  - `False`: 684
  - null (missing): 677

## encounter_primary_reason_number_of_days_to_rehosp_for_non_CV_f5a_first

- dtype: `Int32` (numeric)
- nulls: 1360 (65.86%)
- min/max: 1.0 / 1900.0
- mean/std: 130.5929 / 269.98027117667186
- quantiles: {'0.05': 3.0, '0.25': 15.0, '0.5': 38.0, '0.75': 81.0, '0.95': 713.7999999999986}

## encounter_primary_reason_number_of_non_CV_rehospitalizations_5a_f5a_count

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 220.0
- mean/std: 6.1768 / 13.241797304865019
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 1.0, '0.75': 8.0, '0.95': 24.799999999999955}

## encounter_primary_reason_renal_complications_f5a_w7d_first

- dtype: `Boolean` (boolean)
- nulls: 1823 (88.28%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 1823
  - `False`: 242

## encounter_primary_reason_renal_complications_f5a_w1mo_first

- dtype: `Boolean` (boolean)
- nulls: 1470 (71.19%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 1470
  - `False`: 594
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## encounter_primary_reason_renal_complications_f5a_w3mo_first

- dtype: `Boolean` (boolean)
- nulls: 1030 (49.88%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1034
  - null (missing): 1030
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## encounter_primary_reason_renal_complications_f5a_w6mo_first

- dtype: `Boolean` (boolean)
- nulls: 906 (43.87%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1158
  - null (missing): 906
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## encounter_primary_reason_renal_complications_f5a_w1a_first

- dtype: `Boolean` (boolean)
- nulls: 817 (39.56%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1246
  - null (missing): 817
  - 1 other distinct value(s) suppressed, covering 2 row(s) (count below 5 and/or ranked beyond top 20)

## encounter_primary_reason_renal_complications_f5a_w3a_first

- dtype: `Boolean` (boolean)
- nulls: 706 (34.19%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1357
  - null (missing): 706
  - 1 other distinct value(s) suppressed, covering 2 row(s) (count below 5 and/or ranked beyond top 20)

## encounter_primary_reason_renal_complications_f5a_w5a_first

- dtype: `Boolean` (boolean)
- nulls: 677 (32.78%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1386
  - null (missing): 677
  - 1 other distinct value(s) suppressed, covering 2 row(s) (count below 5 and/or ranked beyond top 20)

## encounter_primary_reason_number_of_days_to_rehosp_for_renal_complications_f5a_first

- dtype: `Int32` (numeric)
- nulls: 2063 (99.90%)
- min/max: 16.0 / 230.0
- mean/std: 121.0000 / 148.49242404917499
- quantiles: {'0.05': 26.5, '0.25': 68.5, '0.5': 121.0, '0.75': 173.5, '0.95': 215.5}

## encounter_primary_reason_number_of_renal_rehospitalizations_5a_f5a_count

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 8.0
- mean/std: 0.0576 / 0.3710335272370817
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 0.0}

## cause_of_death_isCV_f5a_w7d_first

- dtype: `Boolean` (boolean)
- nulls: 1873 (90.70%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 1873
  - `False`: 192

## cause_of_death_isCV_f5a_w1mo_first

- dtype: `Boolean` (boolean)
- nulls: 1787 (86.54%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 1787
  - `False`: 278

## cause_of_death_isCV_f5a_w3mo_first

- dtype: `Boolean` (boolean)
- nulls: 1700 (82.32%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 1700
  - `False`: 365

## cause_of_death_isCV_f5a_w6mo_first

- dtype: `Boolean` (boolean)
- nulls: 1606 (77.77%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 1606
  - `False`: 459

## cause_of_death_isCV_f5a_w1a_first

- dtype: `Boolean` (boolean)
- nulls: 1502 (72.74%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 1502
  - `False`: 563

## cause_of_death_isCV_f5a_w3a_first

- dtype: `Boolean` (boolean)
- nulls: 1244 (60.24%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 1244
  - `False`: 821

## cause_of_death_isCV_f5a_w5a_first

- dtype: `Boolean` (boolean)
- nulls: 1088 (52.69%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 1088
  - `False`: 977

## cause_of_death_number_of_days_to_death_for_CV_f5a_first

- dtype: `Int32` (numeric)
- nulls: 2065 (100.00%)
- All values are null.

## cause_of_death_isRenal_f5a_w7d_first

- dtype: `Boolean` (boolean)
- nulls: 1873 (90.70%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 1873
  - `False`: 192

## cause_of_death_isRenal_f5a_w1mo_first

- dtype: `Boolean` (boolean)
- nulls: 1787 (86.54%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 1787
  - `False`: 278

## cause_of_death_isRenal_f5a_w3mo_first

- dtype: `Boolean` (boolean)
- nulls: 1700 (82.32%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 1700
  - `False`: 365

## cause_of_death_isRenal_f5a_w6mo_first

- dtype: `Boolean` (boolean)
- nulls: 1606 (77.77%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 1606
  - `False`: 459

## cause_of_death_isRenal_f5a_w1a_first

- dtype: `Boolean` (boolean)
- nulls: 1502 (72.74%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 1502
  - `False`: 563

## cause_of_death_isRenal_f5a_w3a_first

- dtype: `Boolean` (boolean)
- nulls: 1244 (60.24%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 1244
  - `False`: 821

## cause_of_death_isRenal_f5a_w5a_first

- dtype: `Boolean` (boolean)
- nulls: 1088 (52.69%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 1088
  - `False`: 977

## cause_of_death_number_of_days_to_death_for_renal_f5a_first

- dtype: `Int32` (numeric)
- nulls: 2065 (100.00%)
- All values are null.

## cause_of_death_isNonRenalAndNonCV_f5a_w7d_first

- dtype: `Boolean` (boolean)
- nulls: 1873 (90.70%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 1873
  - `False`: 192

## cause_of_death_isNonRenalAndNonCV_f5a_w1mo_first

- dtype: `Boolean` (boolean)
- nulls: 1787 (86.54%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 1787
  - `False`: 278

## cause_of_death_isNonRenalAndNonCV_f5a_w3mo_first

- dtype: `Boolean` (boolean)
- nulls: 1700 (82.32%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 1700
  - `False`: 365

## cause_of_death_isNonRenalAndNonCV_f5a_w6mo_first

- dtype: `Boolean` (boolean)
- nulls: 1606 (77.77%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 1606
  - `False`: 459

## cause_of_death_isNonRenalAndNonCV_f5a_w1a_first

- dtype: `Boolean` (boolean)
- nulls: 1502 (72.74%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 1502
  - `False`: 563

## cause_of_death_isNonRenalAndNonCV_f5a_w3a_first

- dtype: `Boolean` (boolean)
- nulls: 1244 (60.24%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 1244
  - `False`: 821

## cause_of_death_isNonRenalAndNonCV_f5a_w5a_first

- dtype: `Boolean` (boolean)
- nulls: 1088 (52.69%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 1088
  - `False`: 977

## cause_of_death_number_of_days_to_death_for_non_renal_and_non_CV_f5a_first

- dtype: `Int32` (numeric)
- nulls: 2065 (100.00%)
- All values are null.

## cause_of_death_isAllCause_f5a_w7d_first

- dtype: `Boolean` (boolean)
- nulls: 1873 (90.70%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 1873
  - `True`: 192

## cause_of_death_isAllCause_f5a_w1mo_first

- dtype: `Boolean` (boolean)
- nulls: 1787 (86.54%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 1787
  - `True`: 278

## cause_of_death_isAllCause_f5a_w3mo_first

- dtype: `Boolean` (boolean)
- nulls: 1700 (82.32%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 1700
  - `True`: 365

## cause_of_death_isAllCause_f5a_w6mo_first

- dtype: `Boolean` (boolean)
- nulls: 1606 (77.77%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 1606
  - `True`: 459

## cause_of_death_isAllCause_f5a_w1a_first

- dtype: `Boolean` (boolean)
- nulls: 1502 (72.74%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 1502
  - `True`: 563

## cause_of_death_isAllCause_f5a_w3a_first

- dtype: `Boolean` (boolean)
- nulls: 1244 (60.24%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 1244
  - `True`: 821

## cause_of_death_isAllCause_f5a_w5a_first

- dtype: `Boolean` (boolean)
- nulls: 1088 (52.69%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 1088
  - `True`: 977

## cause_of_death_number_of_days_to_death_for_all_cause_f5a_first

- dtype: `Int32` (numeric)
- nulls: 1088 (52.69%)
- min/max: 1.0 / 1900.0
- mean/std: 457.4862 / 521.8397311273384
- quantiles: {'0.05': 1.0, '0.25': 19.0, '0.5': 225.0, '0.75': 782.0, '0.95': 1552.0}

## eGFR_2021_ckd_epi_creatinine

- dtype: `Decimal(precision=38, scale=6)` (numeric)
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

- dtype: `Int32` (numeric)
- nulls: 1085 (52.54%)
- min/max: 3.0 / 40.0
- mean/std: 21.6041 / 6.202769132953484
- quantiles: {'0.05': 11.0, '0.25': 17.0, '0.5': 22.0, '0.75': 26.0, '0.95': 31.0}
