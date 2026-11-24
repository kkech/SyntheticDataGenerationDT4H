# Column Analysis

Total rows: 1005
Total columns: 540

## pid

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1005
- ⚠️ high-cardinality column (1005 distinct values)
- top values (shown only where count ≥ 5):
  - (none met the display threshold)
  - 1005 other distinct value(s) suppressed, covering 1005 row(s) (count below 5 and/or ranked beyond top 20)

## encounterId

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1005
- ⚠️ high-cardinality column (1005 distinct values)
- top values (shown only where count ≥ 5):
  - (none met the display threshold)
  - 1005 other distinct value(s) suppressed, covering 1005 row(s) (count below 5 and/or ranked beyond top 20)

## referenceTimePoint

- dtype: `Datetime(time_unit='ns', time_zone=None)` (categorical)
- nulls: 0 (0.00%)
- unique values: 1005
- ⚠️ high-cardinality column (1005 distinct values)
- top values (shown only where count ≥ 5):
  - (none met the display threshold)
  - 1005 other distinct value(s) suppressed, covering 1005 row(s) (count below 5 and/or ranked beyond top 20)

## eventTime

- dtype: `Datetime(time_unit='ns', time_zone=None)` (categorical)
- nulls: 0 (0.00%)
- unique values: 1005
- ⚠️ high-cardinality column (1005 distinct values)
- top values (shown only where count ≥ 5):
  - (none met the display threshold)
  - 1005 other distinct value(s) suppressed, covering 1005 row(s) (count below 5 and/or ranked beyond top 20)

## exitTime

- dtype: `Datetime(time_unit='ns', time_zone=None)` (categorical)
- nulls: 1005 (100.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 1005

## patient_demographics_sourceIdentifier

- dtype: `String` (categorical)
- nulls: 0 (0.00%)
- unique values: 1005
- ⚠️ high-cardinality column (1005 distinct values)
- top values (shown only where count ≥ 5):
  - (none met the display threshold)
  - 1005 other distinct value(s) suppressed, covering 1005 row(s) (count below 5 and/or ranked beyond top 20)

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

## encounters_admissionDate

- dtype: `Datetime(time_unit='ns', time_zone=None)` (categorical)
- nulls: 0 (0.00%)
- unique values: 1005
- ⚠️ high-cardinality column (1005 distinct values)
- top values (shown only where count ≥ 5):
  - (none met the display threshold)
  - 1005 other distinct value(s) suppressed, covering 1005 row(s) (count below 5 and/or ranked beyond top 20)

## encounters_dischargeDate

- dtype: `Datetime(time_unit='ns', time_zone=None)` (categorical)
- nulls: 0 (0.00%)
- unique values: 1005
- ⚠️ high-cardinality column (1005 distinct values)
- top values (shown only where count ≥ 5):
  - (none met the display threshold)
  - 1005 other distinct value(s) suppressed, covering 1005 row(s) (count below 5 and/or ranked beyond top 20)

## encounters_numOfPreviousHFStays_count

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 320.0
- mean/std: 5.7841 / 15.856557746853968
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 6.0, '0.95': 28.799999999999955}

## vital_signs_weight_value_p6mo_stddev

- dtype: `Float64` (numeric)
- nulls: 70 (6.97%)
- min/max: 0.0 / 22.0
- mean/std: 2.8248 / 2.2139654859891427
- quantiles: {'0.05': 0.0, '0.25': 1.3750632261063396, '0.5': 2.6375907125489158, '0.75': 3.7455151621166323, '0.95': 6.634662277629678}

## vital_signs_weight_value_p6mo_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 70 (6.97%)
- min/max: 31.0 / 230.0
- mean/std: 81.3195 / 23.606429439368448
- quantiles: {'0.05': 48.897429194421655, '0.25': 65.58433310442213, '0.5': 78.1163564948701, '0.75': 93.94422980619882, '0.95': 124.210522693913}

## vital_signs_weight_value_p6mo_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 70 (6.97%)
- min/max: 30.0 / 230.0
- mean/std: 75.3031 / 22.394680300652823
- quantiles: {'0.05': 44.50005314361999, '0.25': 59.99698800330949, '0.5': 72.65923026406006, '0.75': 88.35113386499185, '0.95': 112.85921925183634}

## vital_signs_weight_value_p6mo_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 70 (6.97%)
- min/max: 32.0 / 230.0
- mean/std: 77.9317 / 22.333327283894512
- quantiles: {'0.05': 46.79811862585039, '0.25': 62.20726991322577, '0.5': 75.2760118634934, '0.75': 91.20363684525694, '0.95': 114.64015655239606}

## vital_signs_weight_value_p6mo_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 70 (6.97%)
- min/max: 33.0 / 240.0
- mean/std: 83.9265 / 23.552392123842573
- quantiles: {'0.05': 51.06656911695106, '0.25': 68.33105932310107, '0.5': 81.29880206880776, '0.75': 96.14146343522383, '0.95': 125.18898915847315}

## vital_signs_weight_value_p6mo_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 70 (6.97%)
- min/max: 32.0 / 230.0
- mean/std: 79.5711 / 22.566784979569544
- quantiles: {'0.05': 48.182313584447876, '0.25': 64.82265657824497, '0.5': 76.79933398892994, '0.75': 92.45207760691508, '0.95': 118.43396852680833}

## vital_signs_height_value_p1a_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 152 (15.12%)
- min/max: 97.0 / 200.0
- mean/std: 166.3464 / 12.535450213269446
- quantiles: {'0.05': 146.20162394290045, '0.25': 158.10054313614498, '0.5': 166.68383705571293, '0.75': 174.59979174930604, '0.95': 186.98066816964743}

## vital_signs_weight_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 92 (9.15%)
- min/max: 32.0 / 230.0
- mean/std: 78.0214 / 22.17255925528033
- quantiles: {'0.05': 46.90697375038797, '0.25': 62.49552938303604, '0.5': 75.4278456353409, '0.75': 91.25620305764876, '0.95': 114.60414433716245}

## vital_signs_height_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 206 (20.50%)
- min/max: 97.0 / 200.0
- mean/std: 166.4438 / 12.987033388193165
- quantiles: {'0.05': 145.50496419470386, '0.25': 157.16704814696686, '0.5': 166.62777331683785, '0.75': 175.4245292296633, '0.95': 187.60808945305644}

## vital_signs_diastolicBp_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 2 (0.20%)
- min/max: 45.0 / 130.0
- mean/std: 71.9355 / 10.242644197657539
- quantiles: {'0.05': 58.31192307692308, '0.25': 64.96969544171037, '0.5': 70.60000000000001, '0.75': 77.43083003952569, '0.95': 90.68809523809524}

## vital_signs_diastolicBp_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2 (0.20%)
- min/max: 2.0 / 110.0
- mean/std: 53.1256 / 10.996015041457495
- quantiles: {'0.05': 36.1, '0.25': 48.0, '0.5': 53.0, '0.75': 58.0, '0.95': 71.0}

## vital_signs_diastolicBp_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2 (0.20%)
- min/max: 56.0 / 200.0
- mean/std: 97.0907 / 16.57303270203762
- quantiles: {'0.05': 75.0, '0.25': 87.0, '0.5': 95.0, '0.75': 106.0, '0.95': 127.0}

## vital_signs_diastolicBp_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2 (0.20%)
- min/max: 31.0 / 130.0
- mean/std: 69.4018 / 12.72349410206804
- quantiles: {'0.05': 52.0, '0.25': 60.0, '0.5': 68.0, '0.75': 77.0, '0.95': 90.0}

## vital_signs_diastolicBp_value_stddev

- dtype: `Float64` (numeric)
- nulls: 2 (0.20%)
- min/max: 1.0 / 23.0
- mean/std: 9.8705 / 3.153592969272864
- quantiles: {'0.05': 5.432325287235293, '0.25': 7.763956253438983, '0.5': 9.437978797760794, '0.75': 11.446797485882868, '0.95': 15.681480600374364}

## vital_signs_diastolicBp_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2 (0.20%)
- min/max: 38.0 / 200.0
- mean/std: 79.4317 / 17.33847819540058
- quantiles: {'0.05': 55.1, '0.25': 67.0, '0.5': 78.0, '0.75': 89.0, '0.95': 110.0}

## vital_signs_heartRate_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 0 (0.00%)
- min/max: 46.0 / 150.0
- mean/std: 80.7209 / 14.086098741358368
- quantiles: {'0.05': 60.40180995475113, '0.25': 70.76923076923076, '0.5': 79.20588235294117, '0.75': 88.9325259515571, '0.95': 107.43518518518518}

## vital_signs_heartRate_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 120.0
- mean/std: 58.9751 / 11.564576102380137
- quantiles: {'0.05': 43.0, '0.25': 52.0, '0.5': 58.0, '0.75': 65.0, '0.95': 78.0}

## vital_signs_heartRate_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 0 (0.00%)
- min/max: 49.0 / 230.0
- mean/std: 110.0100 / 24.43003467293954
- quantiles: {'0.05': 79.0, '0.25': 93.0, '0.5': 106.0, '0.75': 123.0, '0.95': 155.0}

## vital_signs_heartRate_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 0 (0.00%)
- min/max: 35.0 / 160.0
- mean/std: 76.3542 / 15.322492597802887
- quantiles: {'0.05': 54.2, '0.25': 65.0, '0.5': 75.0, '0.75': 86.0, '0.95': 102.0}

## vital_signs_heartRate_value_stddev

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 1.2 / 45.0
- mean/std: 10.9845 / 5.674367479606872
- quantiles: {'0.05': 4.435387067583342, '0.25': 7.204279796074206, '0.5': 9.461535877278324, '0.75': 13.205984932974156, '0.95': 22.721773513042315}

## vital_signs_heartRate_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 0 (0.00%)
- min/max: 30.0 / 230.0
- mean/std: 90.4925 / 25.1058554548014
- quantiles: {'0.05': 55.0, '0.25': 73.0, '0.5': 88.0, '0.75': 104.0, '0.95': 137.79999999999995}

## vital_signs_systolicBp_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 2 (0.20%)
- min/max: 78.0 / 210.0
- mean/std: 128.8840 / 19.911182367116993
- quantiles: {'0.05': 101.5992213570634, '0.25': 114.4543469760167, '0.5': 126.74074074074073, '0.75': 141.24859943977592, '0.95': 165.8240847092606}

## vital_signs_systolicBp_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2 (0.20%)
- min/max: 45.0 / 180.0
- mean/std: 98.9641 / 19.18937840029196
- quantiles: {'0.05': 68.0, '0.25': 87.0, '0.5': 97.0, '0.75': 110.0, '0.95': 131.0}

## vital_signs_systolicBp_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2 (0.20%)
- min/max: 100.0 / 300.0
- mean/std: 166.9501 / 28.829381557632765
- quantiles: {'0.05': 124.0, '0.25': 145.0, '0.5': 165.0, '0.75': 185.0, '0.95': 218.0}

## vital_signs_systolicBp_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2 (0.20%)
- min/max: 60.0 / 210.0
- mean/std: 123.3180 / 21.93725272807376
- quantiles: {'0.05': 94.0, '0.25': 108.0, '0.5': 119.0, '0.75': 136.0, '0.95': 165.0}

## vital_signs_systolicBp_value_stddev

- dtype: `Float64` (numeric)
- nulls: 2 (0.20%)
- min/max: 0.5 / 56.0
- mean/std: 15.6092 / 5.867975200378621
- quantiles: {'0.05': 7.808336485463953, '0.25': 11.518100625998326, '0.5': 14.689680567800464, '0.75': 19.095130899113542, '0.95': 26.460540263015076}

## vital_signs_systolicBp_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2 (0.20%)
- min/max: 64.0 / 280.0
- mean/std: 141.7827 / 30.453370955289703
- quantiles: {'0.05': 98.1, '0.25': 120.0, '0.5': 137.0, '0.75': 160.0, '0.95': 198.0}

## vital_signs_oxygenSaturation_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 934 (92.94%)
- min/max: 75.0 / 100.0
- mean/std: 95.5211 / 4.269021818133776
- quantiles: {'0.05': 86.5, '0.25': 94.0, '0.5': 96.0, '0.75': 98.0, '0.95': 100.0}

## vital_signs_oxygenSaturation_value_stddev

- dtype: `Float64` (numeric)
- nulls: 934 (92.94%)
- min/max: 0.0 / 4.8
- mean/std: 0.2736 / 0.836695024306899
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 2.2896471525334583}

## vital_signs_oxygenSaturation_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 934 (92.94%)
- min/max: 75.0 / 100.0
- mean/std: 95.8310 / 4.355574147795173
- quantiles: {'0.05': 86.5, '0.25': 94.5, '0.5': 97.0, '0.75': 99.0, '0.95': 100.0}

## vital_signs_oxygenSaturation_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 934 (92.94%)
- min/max: 75.0 / 100.0
- mean/std: 95.5007 / 4.2031161170894125
- quantiles: {'0.05': 86.50000000000001, '0.25': 94.0, '0.5': 96.0, '0.75': 98.0, '0.95': 100.0}

## vital_signs_oxygenSaturation_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 934 (92.94%)
- min/max: 73.0 / 100.0
- mean/std: 94.3239 / 5.518215674051905
- quantiles: {'0.05': 84.0, '0.25': 93.0, '0.5': 96.0, '0.75': 98.0, '0.95': 100.0}

## vital_signs_oxygenSaturation_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 934 (92.94%)
- min/max: 75.0 / 100.0
- mean/std: 95.7465 / 4.32176653649591
- quantiles: {'0.05': 86.5, '0.25': 94.0, '0.5': 96.0, '0.75': 99.0, '0.95': 100.0}

## lab_results_hemoglobin_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 0 (0.00%)
- min/max: 56.0 / 230.0
- mean/std: 126.8119 / 22.807034297833805
- quantiles: {'0.05': 93.0, '0.25': 110.0, '0.5': 126.0, '0.75': 142.0, '0.95': 165.0}

## lab_results_hemoglobin_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 0 (0.00%)
- min/max: 56.0 / 210.0
- mean/std: 119.6209 / 23.056887926833223
- quantiles: {'0.05': 84.0, '0.25': 102.0, '0.5': 119.0, '0.75': 136.0, '0.95': 160.0}

## lab_results_hemoglobin_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 0 (0.00%)
- min/max: 54.0 / 210.0
- mean/std: 118.4872 / 22.073349255873215
- quantiles: {'0.05': 84.84, '0.25': 101.8, '0.5': 117.5, '0.75': 134.0, '0.95': 156.39999999999998}

## lab_results_hemoglobin_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 0 (0.00%)
- min/max: 52.0 / 200.0
- mean/std: 120.0517 / 23.03257068162216
- quantiles: {'0.05': 84.0, '0.25': 104.0, '0.5': 120.0, '0.75': 136.0, '0.95': 157.0}

## lab_results_hemoglobin_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 200.0
- mean/std: 111.0517 / 23.509964604607482
- quantiles: {'0.05': 74.0, '0.25': 94.0, '0.5': 111.0, '0.75': 127.0, '0.95': 150.0}

## lab_results_hemoglobin_value_stddev

- dtype: `Float64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 33.0
- mean/std: 5.1300 / 3.8080140400074187
- quantiles: {'0.05': 0.0, '0.25': 2.712931993250107, '0.5': 4.5814811072187664, '0.75': 6.858571279792897, '0.95': 11.902561834109871}

## lab_results_ferritin_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 806 (80.20%)
- min/max: 9.0 / 15000.0
- mean/std: 542.5377 / 1380.804010162738
- quantiles: {'0.05': 25.9, '0.25': 80.0, '0.5': 178.0, '0.75': 488.0, '0.95': 1931.5999999999995}

## lab_results_ferritin_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 806 (80.20%)
- min/max: 9.0 / 15000.0
- mean/std: 503.6834 / 1206.519958131685
- quantiles: {'0.05': 25.9, '0.25': 77.0, '0.5': 178.0, '0.75': 482.0, '0.95': 1850.1999999999994}

## lab_results_ferritin_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 806 (80.20%)
- min/max: 9.0 / 7100.0
- mean/std: 462.0276 / 857.2920854347038
- quantiles: {'0.05': 25.9, '0.25': 79.5, '0.5': 178.0, '0.75': 465.25, '0.95': 1931.5999999999995}

## lab_results_ferritin_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 806 (80.20%)
- min/max: 9.0 / 5100.0
- mean/std: 419.6683 / 685.1516167477712
- quantiles: {'0.05': 25.9, '0.25': 74.5, '0.5': 170.0, '0.75': 441.0, '0.95': 1818.6999999999994}

## lab_results_ferritin_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 806 (80.20%)
- min/max: 9.0 / 5100.0
- mean/std: 412.8492 / 669.9606104710651
- quantiles: {'0.05': 25.9, '0.25': 73.5, '0.5': 170.0, '0.75': 425.5, '0.95': 1808.1}

## lab_results_ferritin_value_stddev

- dtype: `Float64` (numeric)
- nulls: 806 (80.20%)
- min/max: 0.0 / 4700.0
- mean/std: 49.3172 / 384.4124550299454
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 57.24999999999996}

## lab_results_tfs_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 629 (62.59%)
- min/max: 3.0 / 93.0
- mean/std: 19.3830 / 15.073584051917207
- quantiles: {'0.05': 5.0, '0.25': 10.0, '0.5': 16.0, '0.75': 24.0, '0.95': 44.25}

## lab_results_tfs_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 629 (62.59%)
- min/max: 3.0 / 93.0
- mean/std: 18.9335 / 14.871479439155333
- quantiles: {'0.05': 5.0, '0.25': 9.75, '0.5': 15.0, '0.75': 24.0, '0.95': 44.0}

## lab_results_tfs_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 629 (62.59%)
- min/max: 3.0 / 93.0
- mean/std: 18.7531 / 14.548035045327017
- quantiles: {'0.05': 5.0, '0.25': 9.875, '0.5': 15.583333333333332, '0.75': 23.0, '0.95': 44.0}

## lab_results_tfs_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 629 (62.59%)
- min/max: 0.0 / 93.0
- mean/std: 18.6037 / 14.67823375271326
- quantiles: {'0.05': 5.0, '0.25': 9.0, '0.5': 15.0, '0.75': 23.0, '0.95': 44.0}

## lab_results_tfs_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 629 (62.59%)
- min/max: 0.0 / 93.0
- mean/std: 18.1489 / 14.472010648095225
- quantiles: {'0.05': 5.0, '0.25': 9.0, '0.5': 15.0, '0.75': 23.0, '0.95': 43.25}

## lab_results_tfs_value_stddev

- dtype: `Float64` (numeric)
- nulls: 629 (62.59%)
- min/max: 0.0 / 41.0
- mean/std: 0.6004 / 2.562309251073659
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 4.027402333828163}

## lab_results_ntProBnp_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 271 (26.97%)
- min/max: 50.0 / 190000.0
- mean/std: 10305.9591 / 15692.474084518333
- quantiles: {'0.05': 333.25, '0.25': 1897.25, '0.5': 4538.5, '0.75': 11250.749999999998, '0.95': 40092.70000000004}

## lab_results_ntProBnp_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 271 (26.97%)
- min/max: 50.0 / 190000.0
- mean/std: 9480.3924 / 14916.37572248711
- quantiles: {'0.05': 284.3, '0.25': 1751.75, '0.5': 4020.5, '0.75': 9915.25, '0.95': 35785.450000000004}

## lab_results_ntProBnp_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 271 (26.97%)
- min/max: 50.0 / 190000.0
- mean/std: 9506.4511 / 14535.77010551081
- quantiles: {'0.05': 320.8999999999999, '0.25': 1861.5, '0.5': 4238.583333333334, '0.75': 10162.75, '0.95': 35973.525000000016}

## lab_results_ntProBnp_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 271 (26.97%)
- min/max: 50.0 / 190000.0
- mean/std: 9531.4782 / 14647.564841595566
- quantiles: {'0.05': 327.9, '0.25': 1846.25, '0.5': 4177.0, '0.75': 10453.25, '0.95': 35726.100000000006}

## lab_results_ntProBnp_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 271 (26.97%)
- min/max: 50.0 / 190000.0
- mean/std: 8810.7752 / 14034.235933115207
- quantiles: {'0.05': 283.0, '0.25': 1655.5000000000002, '0.5': 3748.0, '0.75': 9512.499999999998, '0.95': 33556.30000000003}

## lab_results_ntProBnp_value_stddev

- dtype: `Float64` (numeric)
- nulls: 271 (26.97%)
- min/max: 0.0 / 33000.0
- mean/std: 677.0120 / 2585.872811874172
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 38.75, '0.95': 3152.650000000002}

## lab_results_bnp_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1005 (100.00%)
- All values are null.

## lab_results_bnp_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1005 (100.00%)
- All values are null.

## lab_results_bnp_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 1005 (100.00%)
- All values are null.

## lab_results_bnp_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1005 (100.00%)
- All values are null.

## lab_results_bnp_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1005 (100.00%)
- All values are null.

## lab_results_bnp_value_stddev

- dtype: `Float64` (numeric)
- nulls: 1005 (100.00%)
- All values are null.

## lab_results_crpNonHs_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 10 (1.00%)
- min/max: 0.6 / 570.0
- mean/std: 54.2484 / 75.68258044611935
- quantiles: {'0.05': 1.6, '0.25': 7.55, '0.5': 23.1, '0.75': 67.05, '0.95': 227.33999999999997}

## lab_results_crpNonHs_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 10 (1.00%)
- min/max: 0.6 / 420.0
- mean/std: 26.7618 / 44.73665812535698
- quantiles: {'0.05': 1.1, '0.25': 4.4, '0.5': 11.6, '0.75': 29.6, '0.95': 101.17999999999996}

## lab_results_crpNonHs_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 10 (1.00%)
- min/max: 0.6 / 390.0
- mean/std: 30.4770 / 39.43074712546604
- quantiles: {'0.05': 1.3, '0.25': 5.9399999999999995, '0.5': 15.600000000000001, '0.75': 39.02782608695652, '0.95': 111.8885}

## lab_results_crpNonHs_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 10 (1.00%)
- min/max: 0.6 / 320.0
- mean/std: 28.2446 / 44.05254680507984
- quantiles: {'0.05': 1.2, '0.25': 4.5, '0.5': 12.4, '0.75': 31.55, '0.95': 108.17999999999994}

## lab_results_crpNonHs_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 10 (1.00%)
- min/max: 0.0 / 250.0
- mean/std: 15.5008 / 24.961550701777494
- quantiles: {'0.05': 0.8700000000000002, '0.25': 3.1, '0.5': 7.2, '0.75': 17.55, '0.95': 56.42999999999999}

## lab_results_crpNonHs_value_stddev

- dtype: `Float64` (numeric)
- nulls: 10 (1.00%)
- min/max: 0.0 / 180.0
- mean/std: 12.9387 / 21.31423782986395
- quantiles: {'0.05': 0.0, '0.25': 0.2999999999999998, '0.5': 3.498888712466036, '0.75': 15.888151631935282, '0.95': 57.2733656241696}

## lab_results_crpHs_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1005 (100.00%)
- All values are null.

## lab_results_crpHs_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1005 (100.00%)
- All values are null.

## lab_results_crpHs_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 1005 (100.00%)
- All values are null.

## lab_results_crpHs_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1005 (100.00%)
- All values are null.

## lab_results_crpHs_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1005 (100.00%)
- All values are null.

## lab_results_crpHs_value_stddev

- dtype: `Float64` (numeric)
- nulls: 1005 (100.00%)
- All values are null.

## lab_results_tropIHs_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1005 (100.00%)
- All values are null.

## lab_results_tropIHs_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1005 (100.00%)
- All values are null.

## lab_results_tropIHs_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 1005 (100.00%)
- All values are null.

## lab_results_tropIHs_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1005 (100.00%)
- All values are null.

## lab_results_tropIHs_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1005 (100.00%)
- All values are null.

## lab_results_tropIHs_value_stddev

- dtype: `Float64` (numeric)
- nulls: 1005 (100.00%)
- All values are null.

## lab_results_tropInHs_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1005 (100.00%)
- All values are null.

## lab_results_tropInHs_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1005 (100.00%)
- All values are null.

## lab_results_tropInHs_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 1005 (100.00%)
- All values are null.

## lab_results_tropInHs_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1005 (100.00%)
- All values are null.

## lab_results_tropInHs_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1005 (100.00%)
- All values are null.

## lab_results_tropInHs_value_stddev

- dtype: `Float64` (numeric)
- nulls: 1005 (100.00%)
- All values are null.

## lab_results_tropTHs_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1005 (100.00%)
- All values are null.

## lab_results_tropTHs_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1005 (100.00%)
- All values are null.

## lab_results_tropTHs_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 1005 (100.00%)
- All values are null.

## lab_results_tropTHs_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1005 (100.00%)
- All values are null.

## lab_results_tropTHs_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1005 (100.00%)
- All values are null.

## lab_results_tropTHs_value_stddev

- dtype: `Float64` (numeric)
- nulls: 1005 (100.00%)
- All values are null.

## lab_results_tropTnHs_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 271 (26.97%)
- min/max: 3.0 / 7400.0
- mean/std: 127.8297 / 454.1836352242378
- quantiles: {'0.05': 13.0, '0.25': 26.0, '0.5': 42.0, '0.75': 89.75, '0.95': 340.8000000000002}

## lab_results_tropTnHs_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 271 (26.97%)
- min/max: 3.0 / 4500.0
- mean/std: 97.8297 / 301.63958443548046
- quantiles: {'0.05': 12.0, '0.25': 24.0, '0.5': 38.0, '0.75': 75.0, '0.95': 288.9000000000003}

## lab_results_tropTnHs_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 271 (26.97%)
- min/max: 3.0 / 6000.0
- mean/std: 93.3130 / 297.71051261323925
- quantiles: {'0.05': 12.0, '0.25': 24.0, '0.5': 40.0, '0.75': 80.375, '0.95': 289.5500000000003}

## lab_results_tropTnHs_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 271 (26.97%)
- min/max: 3.0 / 7400.0
- mean/std: 80.9891 / 296.1362742443717
- quantiles: {'0.05': 12.0, '0.25': 24.0, '0.5': 39.0, '0.75': 73.0, '0.95': 204.8000000000002}

## lab_results_tropTnHs_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 271 (26.97%)
- min/max: 3.0 / 4500.0
- mean/std: 66.7330 / 188.2211829930938
- quantiles: {'0.05': 11.0, '0.25': 22.25, '0.5': 35.5, '0.75': 64.75, '0.95': 174.0}

## lab_results_tropTnHs_value_stddev

- dtype: `Float64` (numeric)
- nulls: 271 (26.97%)
- min/max: 0.0 / 1900.0
- mean/std: 23.5367 / 121.7268307115583
- quantiles: {'0.05': 0.0, '0.25': 0.5, '0.5': 1.5405694150420945, '0.75': 5.0, '0.95': 56.25714624943282}

## lab_results_triGly_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 884 (87.96%)
- min/max: 0.4 / 3.2
- mean/std: 1.2851 / 0.5173105381078812
- quantiles: {'0.05': 0.7, '0.25': 0.9, '0.5': 1.2, '0.75': 1.5, '0.95': 2.2}

## lab_results_triGly_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 884 (87.96%)
- min/max: 0.4 / 3.2
- mean/std: 1.2636 / 0.4959838707054898
- quantiles: {'0.05': 0.7, '0.25': 0.9, '0.5': 1.2, '0.75': 1.4, '0.95': 2.2}

## lab_results_triGly_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 884 (87.96%)
- min/max: 0.4 / 3.2
- mean/std: 1.2639 / 0.49605082581696663
- quantiles: {'0.05': 0.7, '0.25': 0.9, '0.5': 1.2, '0.75': 1.4, '0.95': 2.2}

## lab_results_triGly_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 884 (87.96%)
- min/max: 0.4 / 3.2
- mean/std: 1.2636 / 0.5112077203381551
- quantiles: {'0.05': 0.6, '0.25': 0.9, '0.5': 1.2, '0.75': 1.4, '0.95': 2.2}

## lab_results_triGly_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 884 (87.96%)
- min/max: 0.4 / 3.2
- mean/std: 1.2430 / 0.4890171173352032
- quantiles: {'0.05': 0.6, '0.25': 0.9, '0.5': 1.2, '0.75': 1.4, '0.95': 2.2}

## lab_results_triGly_value_stddev

- dtype: `Float64` (numeric)
- nulls: 884 (87.96%)
- min/max: 0.0 / 0.55
- mean/std: 0.0196 / 0.07348251337640771
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 0.15000000000000002}

## lab_results_cholTot_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 885 (88.06%)
- min/max: 1.7 / 6.7
- mean/std: 4.0125 / 1.1642837809954563
- quantiles: {'0.05': 2.19, '0.25': 3.275, '0.5': 3.85, '0.75': 4.825, '0.95': 6.005}

## lab_results_cholTot_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 885 (88.06%)
- min/max: 1.7 / 6.7
- mean/std: 3.9725 / 1.1640961076728844
- quantiles: {'0.05': 2.19, '0.25': 3.2, '0.5': 3.8, '0.75': 4.8, '0.95': 6.005}

## lab_results_cholTot_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 885 (88.06%)
- min/max: 1.1 / 6.7
- mean/std: 3.9751 / 1.1722054874270653
- quantiles: {'0.05': 1.9900000000000002, '0.25': 3.2, '0.5': 3.8000000000000003, '0.75': 4.8, '0.95': 6.005}

## lab_results_cholTot_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 885 (88.06%)
- min/max: 0.5 / 6.7
- mean/std: 3.9825 / 1.1948532837749992
- quantiles: {'0.05': 1.99, '0.25': 3.2, '0.5': 3.8, '0.75': 4.825, '0.95': 6.005}

## lab_results_cholTot_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 885 (88.06%)
- min/max: 0.5 / 6.7
- mean/std: 3.9475 / 1.1915643626141648
- quantiles: {'0.05': 1.99, '0.25': 3.2, '0.5': 3.75, '0.75': 4.8, '0.95': 6.005}

## lab_results_cholTot_value_stddev

- dtype: `Float64` (numeric)
- nulls: 885 (88.06%)
- min/max: 0.0 / 0.97
- mean/std: 0.0295 / 0.13278611251409334
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 0.10249999999999994}

## lab_results_hdl_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 887 (88.26%)
- min/max: 0.5 / 2.7
- mean/std: 1.2076 / 0.36635342789455155
- quantiles: {'0.05': 0.7, '0.25': 1.0, '0.5': 1.2, '0.75': 1.4, '0.95': 1.8}

## lab_results_hdl_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 887 (88.26%)
- min/max: 0.4 / 2.7
- mean/std: 1.2000 / 0.3723338268711005
- quantiles: {'0.05': 0.7, '0.25': 1.0, '0.5': 1.2, '0.75': 1.4, '0.95': 1.8}

## lab_results_hdl_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 887 (88.26%)
- min/max: 0.5 / 2.7
- mean/std: 1.2029 / 0.3693716186626624
- quantiles: {'0.05': 0.7, '0.25': 1.0, '0.5': 1.2, '0.75': 1.4, '0.95': 1.8}

## lab_results_hdl_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 887 (88.26%)
- min/max: 0.5 / 2.7
- mean/std: 1.2042 / 0.37012219460549983
- quantiles: {'0.05': 0.7, '0.25': 1.0, '0.5': 1.2, '0.75': 1.4, '0.95': 1.8}

## lab_results_hdl_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 887 (88.26%)
- min/max: 0.4 / 2.7
- mean/std: 1.1983 / 0.3732470254740236
- quantiles: {'0.05': 0.7, '0.25': 0.925, '0.5': 1.2, '0.75': 1.4, '0.95': 1.8}

## lab_results_hdl_value_stddev

- dtype: `Float64` (numeric)
- nulls: 887 (88.26%)
- min/max: 0.0 / 0.11
- mean/std: 0.0042 / 0.019110955274963672
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 0.007499999999999572}

## lab_results_creatUS_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 849 (84.48%)
- min/max: 45.0 / 2400.0
- mean/std: 677.4574 / 476.84853509566864
- quantiles: {'0.05': 138.37785, '0.25': 309.5547, '0.5': 589.7034, '0.75': 869.710725, '0.95': 1585.83165}

## lab_results_creatUS_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 849 (84.48%)
- min/max: 45.0 / 2400.0
- mean/std: 627.3410 / 460.79724964064894
- quantiles: {'0.05': 129.32985, '0.25': 291.68489999999997, '0.5': 504.82185000000004, '0.75': 817.4867999999999, '0.95': 1451.8647}

## lab_results_creatUS_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 849 (84.48%)
- min/max: 45.0 / 2400.0
- mean/std: 624.7270 / 453.221861355337
- quantiles: {'0.05': 135.5362125, '0.25': 295.5303, '0.5': 517.6587, '0.75': 803.490675, '0.95': 1448.1324}

## lab_results_creatUS_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 849 (84.48%)
- min/max: 45.0 / 2400.0
- mean/std: 629.8075 / 479.80412705824136
- quantiles: {'0.05': 130.262925, '0.25': 287.8395, '0.5': 482.54115, '0.75': 837.7599749999999, '0.95': 1585.83165}

## lab_results_creatUS_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 849 (84.48%)
- min/max: 45.0 / 2400.0
- mean/std: 577.9939 / 462.8142836849975
- quantiles: {'0.05': 111.544875, '0.25': 261.11962500000004, '0.5': 444.82230000000004, '0.75': 774.9612, '0.95': 1448.1324}

## lab_results_creatUS_value_stddev

- dtype: `Float64` (numeric)
- nulls: 849 (84.48%)
- min/max: 0.0 / 790.0
- mean/std: 45.8708 / 112.7157230008213
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 5.556037499999981, '0.95': 290.2852875}

## lab_results_albuminUS_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 974 (96.92%)
- min/max: 4.0 / 4300.0
- mean/std: 493.4194 / 872.9300000264835
- quantiles: {'0.05': 6.5, '0.25': 33.5, '0.5': 149.0, '0.75': 536.5, '0.95': 1825.0}

## lab_results_albuminUS_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 974 (96.92%)
- min/max: 4.0 / 1600.0
- mean/std: 299.8065 / 417.7093422748119
- quantiles: {'0.05': 6.5, '0.25': 29.5, '0.5': 96.0, '0.75': 347.5, '0.95': 1197.5000000000002}

## lab_results_albuminUS_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 974 (96.92%)
- min/max: 4.0 / 1900.0
- mean/std: 374.6290 / 510.0228747769053
- quantiles: {'0.05': 6.500000000000001, '0.25': 33.5, '0.5': 146.0, '0.75': 504.99999999999994, '0.95': 1428.5}

## lab_results_albuminUS_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 974 (96.92%)
- min/max: 4.0 / 2100.0
- mean/std: 371.0645 / 525.7732677041104
- quantiles: {'0.05': 6.5, '0.25': 33.5, '0.5': 149.0, '0.75': 485.5, '0.95': 1428.4999999999998}

## lab_results_albuminUS_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 974 (96.92%)
- min/max: 4.0 / 1600.0
- mean/std: 290.2581 / 410.3463551474482
- quantiles: {'0.05': 6.5, '0.25': 29.5, '0.5': 96.0, '0.75': 347.5, '0.95': 1197.5000000000002}

## lab_results_albuminUS_value_stddev

- dtype: `Float64` (numeric)
- nulls: 974 (96.92%)
- min/max: 0.0 / 1800.0
- mean/std: 95.9682 / 353.09537419534576
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 566.25}

## lab_results_bun_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 160 (15.92%)
- min/max: 1.6 / 62.0
- mean/std: 12.4001 / 8.722407407432822
- quantiles: {'0.05': 4.0, '0.25': 6.2, '0.5': 9.4, '0.75': 15.9, '0.95': 29.9}

## lab_results_bun_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 160 (15.92%)
- min/max: 1.5 / 62.0
- mean/std: 11.1798 / 8.093579161322614
- quantiles: {'0.05': 3.5, '0.25': 5.6, '0.5': 8.5, '0.75': 14.4, '0.95': 28.719999999999985}

## lab_results_bun_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 160 (15.92%)
- min/max: 1.6 / 50.0
- mean/std: 10.9584 / 7.3490620557466615
- quantiles: {'0.05': 3.8000000000000003, '0.25': 5.8, '0.5': 8.6, '0.75': 13.9, '0.95': 26.246341463414627}

## lab_results_bun_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 160 (15.92%)
- min/max: 1.6 / 50.0
- mean/std: 10.7310 / 7.460295189782059
- quantiles: {'0.05': 3.7, '0.25': 5.7, '0.5': 8.3, '0.75': 13.4, '0.95': 26.479999999999997}

## lab_results_bun_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 160 (15.92%)
- min/max: 1.5 / 50.0
- mean/std: 9.6663 / 6.756838624383993
- quantiles: {'0.05': 3.1200000000000006, '0.25': 5.2, '0.5': 7.7, '0.75': 11.9, '0.95': 23.579999999999995}

## lab_results_bun_value_stddev

- dtype: `Float64` (numeric)
- nulls: 160 (15.92%)
- min/max: 0.0 / 17.0
- mean/std: 1.0384 / 1.8660494094262075
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.15000000000000036, '0.75': 1.202081528017131, '0.95': 5.145100266098665}

## lab_results_acr_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 974 (96.92%)
- min/max: 1.3 / 710.0
- mean/std: 116.2639 / 172.45331083469947
- quantiles: {'0.05': 3.07, '0.25': 9.385, '0.5': 33.53, '0.75': 118.95, '0.95': 438.36}

## lab_results_acr_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 974 (96.92%)
- min/max: 1.3 / 480.0
- mean/std: 87.1677 / 134.83947579028128
- quantiles: {'0.05': 3.07, '0.25': 6.36, '0.5': 21.13, '0.75': 92.23, '0.95': 390.37500000000006}

## lab_results_acr_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 974 (96.92%)
- min/max: 1.3 / 480.0
- mean/std: 96.9504 / 137.83614118442557
- quantiles: {'0.05': 3.0700000000000003, '0.25': 9.385, '0.5': 33.53, '0.75': 102.78999999999999, '0.95': 412.9933333333333}

## lab_results_acr_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 974 (96.92%)
- min/max: 1.3 / 480.0
- mean/std: 97.8965 / 134.1186946339793
- quantiles: {'0.05': 3.07, '0.25': 9.385, '0.5': 33.53, '0.75': 118.95, '0.95': 393.6}

## lab_results_acr_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 974 (96.92%)
- min/max: 1.3 / 480.0
- mean/std: 79.0961 / 123.50274403638215
- quantiles: {'0.05': 3.07, '0.25': 6.36, '0.5': 21.13, '0.75': 92.23, '0.95': 387.915}

## lab_results_acr_value_stddev

- dtype: `Float64` (numeric)
- nulls: 974 (96.92%)
- min/max: 0.0 / 210.0
- mean/std: 17.1762 / 48.67375162448748
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 126.8775}

## lab_results_ldl_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1005 (100.00%)
- All values are null.

## lab_results_ldl_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1005 (100.00%)
- All values are null.

## lab_results_ldl_value_stddev

- dtype: `Float64` (numeric)
- nulls: 1005 (100.00%)
- All values are null.

## lab_results_ldl_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 1005 (100.00%)
- All values are null.

## lab_results_ldl_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1005 (100.00%)
- All values are null.

## lab_results_ldl_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1005 (100.00%)
- All values are null.

## lab_results_potassium_value_stddev

- dtype: `Float64` (numeric)
- nulls: 10 (1.00%)
- min/max: 0.0 / 1.9
- mean/std: 0.3257 / 0.21426788691337206
- quantiles: {'0.05': 0.0, '0.25': 0.19938079899999045, '0.5': 0.3059411708155669, '0.75': 0.4405975440700327, '0.95': 0.7048907077427937}

## lab_results_potassium_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 10 (1.00%)
- min/max: 2.8 / 7.4
- mean/std: 4.3480 / 0.5890377502497404
- quantiles: {'0.05': 3.5, '0.25': 4.0, '0.5': 4.3, '0.75': 4.7, '0.95': 5.3}

## lab_results_potassium_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 10 (1.00%)
- min/max: 3.4 / 7.7
- mean/std: 4.8801 / 0.6642843003866566
- quantiles: {'0.05': 3.9, '0.25': 4.5, '0.5': 4.8, '0.75': 5.2, '0.95': 6.1}

## lab_results_potassium_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 10 (1.00%)
- min/max: 0.0 / 7.3
- mean/std: 3.8698 / 0.5706217067199729
- quantiles: {'0.05': 3.0, '0.25': 3.5, '0.5': 3.9, '0.75': 4.2, '0.95': 4.729999999999996}

## lab_results_potassium_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 10 (1.00%)
- min/max: 3.1 / 7.3
- mean/std: 4.3456 / 0.47926635629810155
- quantiles: {'0.05': 3.6293333333333333, '0.25': 4.0, '0.5': 4.307142857142857, '0.75': 4.6000000000000005, '0.95': 5.165581395348838}

## lab_results_potassium_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 10 (1.00%)
- min/max: 2.6 / 7.3
- mean/std: 4.4850 / 0.6329221254412725
- quantiles: {'0.05': 3.5, '0.25': 4.1, '0.5': 4.5, '0.75': 4.9, '0.95': 5.6}

## lab_results_sodium_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 16 (1.59%)
- min/max: 10.0 / 160.0
- mean/std: 136.7614 / 6.600491066401029
- quantiles: {'0.05': 127.0, '0.25': 135.0, '0.5': 138.0, '0.75': 140.0, '0.95': 143.0}

## lab_results_sodium_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 16 (1.59%)
- min/max: 100.0 / 170.0
- mean/std: 137.4092 / 4.563262123247401
- quantiles: {'0.05': 129.0, '0.25': 135.5, '0.5': 138.0909090909091, '0.75': 140.27272727272728, '0.95': 143.0}

## lab_results_sodium_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 16 (1.59%)
- min/max: 110.0 / 170.0
- mean/std: 137.9555 / 4.853109782591065
- quantiles: {'0.05': 130.0, '0.25': 135.0, '0.5': 138.0, '0.75': 141.0, '0.95': 144.0}

## lab_results_sodium_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 16 (1.59%)
- min/max: 0.0 / 160.0
- mean/std: 133.9636 / 10.039934915260751
- quantiles: {'0.05': 124.0, '0.25': 132.0, '0.5': 136.0, '0.75': 138.0, '0.95': 141.0}

## lab_results_sodium_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 16 (1.59%)
- min/max: 120.0 / 170.0
- mean/std: 140.4297 / 4.65797453269726
- quantiles: {'0.05': 133.0, '0.25': 138.0, '0.5': 141.0, '0.75': 143.0, '0.95': 148.0}

## lab_results_sodium_value_stddev

- dtype: `Float64` (numeric)
- nulls: 16 (1.59%)
- min/max: 0.0 / 58.0
- mean/std: 2.0688 / 3.2487609684173053
- quantiles: {'0.05': 0.0, '0.25': 1.0, '0.5': 1.6583123951777, '0.75': 2.5000000000000013, '0.95': 4.636343469614227}

## lab_results_albuminBS_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 64 (6.37%)
- min/max: 18.0 / 52.0
- mean/std: 39.6557 / 4.467428467189825
- quantiles: {'0.05': 32.0, '0.25': 37.0, '0.5': 40.0, '0.75': 43.0, '0.95': 46.0}

## lab_results_albuminBS_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 64 (6.37%)
- min/max: 16.0 / 50.0
- mean/std: 36.9773 / 4.804483312315526
- quantiles: {'0.05': 28.5, '0.25': 33.9375, '0.5': 37.666666666666664, '0.75': 40.2, '0.95': 44.0}

## lab_results_albuminBS_value_stddev

- dtype: `Float64` (numeric)
- nulls: 64 (6.37%)
- min/max: 0.0 / 19.0
- mean/std: 1.6906 / 1.3380834744366419
- quantiles: {'0.05': 0.0, '0.25': 0.82915619758885, '0.5': 1.632993161855452, '0.75': 2.4159335036125364, '0.95': 3.6660605559646724}

## lab_results_albuminBS_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 64 (6.37%)
- min/max: 17.0 / 50.0
- mean/std: 36.9511 / 5.093983865888256
- quantiles: {'0.05': 28.0, '0.25': 34.0, '0.5': 38.0, '0.75': 40.0, '0.95': 45.0}

## lab_results_albuminBS_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 64 (6.37%)
- min/max: 0.0 / 50.0
- mean/std: 34.5866 / 5.888881634050523
- quantiles: {'0.05': 24.0, '0.25': 31.0, '0.5': 35.0, '0.75': 39.0, '0.95': 43.0}

## lab_results_albuminBS_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 64 (6.37%)
- min/max: 0.0 / 52.0
- mean/std: 38.1190 / 5.13484855486667
- quantiles: {'0.05': 29.0, '0.25': 35.0, '0.5': 39.0, '0.75': 41.0, '0.95': 45.0}

## lab_results_hba1c%_value_stddev

- dtype: `Float64` (numeric)
- nulls: 948 (94.33%)
- min/max: 0.0 / 0.11
- mean/std: 0.0096 / 0.025771309648570537
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 0.05999999999999993}

## lab_results_hba1c%_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 948 (94.33%)
- min/max: 4.5 / 13.0
- mean/std: 7.0263 / 1.6232161984984224
- quantiles: {'0.05': 5.26, '0.25': 6.1, '0.5': 6.7, '0.75': 7.6, '0.95': 9.759999999999996}

## lab_results_hba1c%_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 948 (94.33%)
- min/max: 4.5 / 13.0
- mean/std: 7.0421 / 1.6289728762563689
- quantiles: {'0.05': 5.26, '0.25': 6.1, '0.5': 6.7, '0.75': 7.7, '0.95': 9.759999999999996}

## lab_results_hba1c%_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 948 (94.33%)
- min/max: 4.5 / 13.0
- mean/std: 7.0386 / 1.6302666725290367
- quantiles: {'0.05': 5.26, '0.25': 6.1, '0.5': 6.7, '0.75': 7.7, '0.95': 9.759999999999996}

## lab_results_hba1c%_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 948 (94.33%)
- min/max: 4.5 / 13.0
- mean/std: 7.0228 / 1.6244798666320037
- quantiles: {'0.05': 5.26, '0.25': 6.1, '0.5': 6.7, '0.75': 7.6, '0.95': 9.759999999999996}

## lab_results_hba1c%_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 948 (94.33%)
- min/max: 4.5 / 13.0
- mean/std: 7.0325 / 1.626523769864688
- quantiles: {'0.05': 5.26, '0.25': 6.1, '0.5': 6.7, '0.75': 7.65, '0.95': 9.759999999999996}

## lab_results_hba1c_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 777 (77.31%)
- min/max: 25.0 / 140.0
- mean/std: 52.6360 / 20.43414102254264
- quantiles: {'0.05': 34.0, '0.25': 40.0, '0.5': 46.0, '0.75': 58.25, '0.95': 98.54999999999984}

## lab_results_hba1c_value_stddev

- dtype: `Float64` (numeric)
- nulls: 777 (77.31%)
- min/max: 0.0 / 5.5
- mean/std: 0.1157 / 0.5342842709652484
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 0.5}

## lab_results_hba1c_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 777 (77.31%)
- min/max: 25.0 / 140.0
- mean/std: 52.4649 / 20.361784982104826
- quantiles: {'0.05': 34.0, '0.25': 40.0, '0.5': 46.0, '0.75': 58.0, '0.95': 97.89999999999986}

## lab_results_hba1c_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 777 (77.31%)
- min/max: 25.0 / 140.0
- mean/std: 52.3816 / 20.31555063235901
- quantiles: {'0.05': 34.0, '0.25': 40.0, '0.5': 46.0, '0.75': 57.25, '0.95': 97.89999999999986}

## lab_results_hba1c_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 777 (77.31%)
- min/max: 25.0 / 140.0
- mean/std: 52.4978 / 20.35868692234221
- quantiles: {'0.05': 34.0, '0.25': 40.0, '0.5': 46.0, '0.75': 58.0, '0.95': 98.11666666666652}

## lab_results_hba1c_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 777 (77.31%)
- min/max: 25.0 / 140.0
- mean/std: 52.5219 / 20.37369928832486
- quantiles: {'0.05': 34.0, '0.25': 40.0, '0.5': 46.0, '0.75': 58.0, '0.95': 98.54999999999984}

## lab_results_validSerumCreatinine_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 75 (7.46%)
- min/max: 1.8 / 23.0
- mean/std: 10.6731 / 4.013726451057291
- quantiles: {'0.05': 5.379036, '0.25': 7.6908, '0.5': 9.867975, '0.75': 12.8934, '0.95': 18.6615}

## lab_results_validSerumCreatinine_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 75 (7.46%)
- min/max: 2.8 / 23.0
- mean/std: 12.2758 / 4.157832224384833
- quantiles: {'0.05': 6.4566905000000006, '0.25': 9.107054357142857, '0.5': 11.593494078947367, '0.75': 15.193502286585368, '0.95': 20.1256926}

## lab_results_validSerumCreatinine_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 75 (7.46%)
- min/max: 2.1 / 23.0
- mean/std: 12.2496 / 4.400026588350347
- quantiles: {'0.05': 6.2205, '0.25': 8.9349, '0.5': 11.5362, '0.75': 15.127125, '0.95': 20.985704999999996}

## lab_results_validSerumCreatinine_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 75 (7.46%)
- min/max: 4.2 / 23.0
- mean/std: 14.2374 / 4.854834804375291
- quantiles: {'0.05': 7.326618, '0.25': 10.094175, '0.5': 13.4589, '0.75': 18.4353, '0.95': 22.1676}

## lab_results_validSerumCreatinine_value_stddev

- dtype: `Float64` (numeric)
- nulls: 75 (7.46%)
- min/max: 0.0 / 6.9
- mean/std: 1.1410 / 0.9526276078943477
- quantiles: {'0.05': 0.03901950000000028, '0.25': 0.48524669393932746, '0.5': 0.880604682076079, '0.75': 1.5685159522928727, '0.95': 2.9389541075052685}

## lab_results_valideGFR_value_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 2 (0.20%)
- min/max: 2.5 / 90.0
- mean/std: 54.3946 / 23.101423944869712
- quantiles: {'0.05': 16.90142857142857, '0.25': 35.61904761904762, '0.5': 54.47727272727273, '0.75': 73.14835164835166, '0.95': 90.0}

## lab_results_valideGFR_value_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2 (0.20%)
- min/max: 3.0 / 90.0
- mean/std: 61.5025 / 23.261353413836588
- quantiles: {'0.05': 21.0, '0.25': 43.0, '0.5': 64.0, '0.75': 84.0, '0.95': 90.0}

## lab_results_valideGFR_value_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2 (0.20%)
- min/max: 2.0 / 90.0
- mean/std: 54.2403 / 23.861775375560114
- quantiles: {'0.05': 16.0, '0.25': 35.0, '0.5': 54.0, '0.75': 75.0, '0.95': 90.0}

## lab_results_valideGFR_value_stddev

- dtype: `Float64` (numeric)
- nulls: 2 (0.20%)
- min/max: 0.0 / 32.0
- mean/std: 4.7708 / 4.566400100911346
- quantiles: {'0.05': 0.0, '0.25': 1.4790199457749038, '0.5': 3.7416573867739413, '0.75': 6.724595897161239, '0.95': 13.939959221316286}

## lab_results_valideGFR_value_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2 (0.20%)
- min/max: 3.0 / 90.0
- mean/std: 54.5145 / 23.670824331438137
- quantiles: {'0.05': 17.0, '0.25': 36.0, '0.5': 55.0, '0.75': 74.0, '0.95': 90.0}

## lab_results_valideGFR_value_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 2 (0.20%)
- min/max: 2.0 / 90.0
- mean/std: 46.9751 / 23.91256480092996
- quantiles: {'0.05': 12.0, '0.25': 27.0, '0.5': 44.0, '0.75': 64.0, '0.95': 90.0}

## symptoms_Ankle_swelling_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1004
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## symptoms_Ascites_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 962
  - `True`: 43

## symptoms_Breathlessness_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 838
  - `True`: 167

## symptoms_Cardiac_murmur_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## symptoms_Chest_pain_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 964
  - `True`: 41

## symptoms_Cheyne_stokes_respiration_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## symptoms_Depression_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 996
  - `True`: 9

## symptoms_Dizziness_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1001
  - 1 other distinct value(s) suppressed, covering 4 row(s) (count below 5 and/or ranked beyond top 20)

## symptoms_Elevated_jugular_venous_pressure_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## symptoms_Fatigue_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## symptoms_Hepatojugular_reflux_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## symptoms_Hepatomegaly_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 996
  - `True`: 9

## symptoms_Intermittent_claudication_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## symptoms_Irregular_pulse_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## symptoms_Loss_of_appetite_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## symptoms_Nocturnal_cough_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## symptoms_Oliguria_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1004
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## symptoms_Orthopnoea_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## symptoms_Palpitations_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1000
  - `True`: 5

## symptoms_Paroxysmal_nocturnal_dyspnea_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## symptoms_Peripheral_edema_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1001
  - 1 other distinct value(s) suppressed, covering 4 row(s) (count below 5 and/or ranked beyond top 20)

## symptoms_Pleural_effusion_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 625
  - `True`: 380

## symptoms_Pulmonary_crepitations_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1004
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## symptoms_Reduced_exercise_tolerance_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1004
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## symptoms_Syncope_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## symptoms_Tachycardia_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 974
  - `True`: 31

## symptoms_Tachypnoea_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1004
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## symptoms_Third_heart_sound_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## symptoms_Weight_gain_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## symptoms_Weight_loss_display_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## echocardiographs_lvef

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1005 (100.00%)
- All values are null.

## echocardiographs_lvef_pET_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 1000 (99.50%)
- min/max: 16.0 / 33.0
- mean/std: 23.5700 / 6.130118269658424
- quantiles: {'0.05': 17.008, '0.25': 21.0, '0.5': 22.369999999999997, '0.75': 26.0, '0.95': 31.176}

## echocardiographs_lvef_pET_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1005 (100.00%)
- All values are null.

## echocardiographs_lvef_pET_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1005 (100.00%)
- All values are null.

## echocardiographs_lvef_pET_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1000 (99.50%)
- min/max: 16.0 / 33.0
- mean/std: 23.5700 / 6.130118269658424
- quantiles: {'0.05': 17.008, '0.25': 21.0, '0.5': 22.37, '0.75': 26.0, '0.95': 31.176}

## echocardiographs_lvef_pET_stddev

- dtype: `Float64` (numeric)
- nulls: 1000 (99.50%)
- min/max: 0.0 / 0.0
- mean/std: 0.0000 / 0.0
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 0.0}
- ⚠️ constant column (single value)

## echocardiographs_lvef_pET_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1000 (99.50%)
- min/max: 16.0 / 33.0
- mean/std: 23.5700 / 6.130118269658424
- quantiles: {'0.05': 17.008, '0.25': 21.0, '0.5': 22.37, '0.75': 26.0, '0.95': 31.176}

## electrocardiographs_ecg_qrs_duration_pET_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1005 (100.00%)
- All values are null.

## electrocardiographs_ecg_qrs_duration_pET_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1005 (100.00%)
- All values are null.

## electrocardiographs_ecg_qrs_duration_pET_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 1005 (100.00%)
- All values are null.

## electrocardiographs_ecg_qrs_duration_pET_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1005 (100.00%)
- All values are null.

## electrocardiographs_ecg_qrs_duration_pET_stddev

- dtype: `Float64` (numeric)
- nulls: 1005 (100.00%)
- All values are null.

## electrocardiographs_ecg_qrs_duration_pET_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1005 (100.00%)
- All values are null.

## electrocardiographs_ecg_qrs_axis_pET_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1005 (100.00%)
- All values are null.

## electrocardiographs_ecg_qrs_axis_pET_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 1005 (100.00%)
- All values are null.

## electrocardiographs_ecg_qrs_axis_pET_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1005 (100.00%)
- All values are null.

## electrocardiographs_ecg_qrs_axis_pET_stddev

- dtype: `Float64` (numeric)
- nulls: 1005 (100.00%)
- All values are null.

## electrocardiographs_ecg_qrs_axis_pET_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1005 (100.00%)
- All values are null.

## electrocardiographs_ecg_qrs_axis_pET_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1005 (100.00%)
- All values are null.

## electrocardiographs_ecg_qt_duration_corrected_pET_last

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1005 (100.00%)
- All values are null.

## electrocardiographs_ecg_qt_duration_corrected_pET_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 1005 (100.00%)
- All values are null.

## electrocardiographs_ecg_qt_duration_corrected_pET_first

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1005 (100.00%)
- All values are null.

## electrocardiographs_ecg_qt_duration_corrected_pET_min

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1005 (100.00%)
- All values are null.

## electrocardiographs_ecg_qt_duration_corrected_pET_stddev

- dtype: `Float64` (numeric)
- nulls: 1005 (100.00%)
- All values are null.

## electrocardiographs_ecg_qt_duration_corrected_pET_max

- dtype: `Decimal(precision=38, scale=18)` (numeric)
- nulls: 1005 (100.00%)
- All values are null.

## electrocardiographs_ecg_st_pET

- dtype: `Boolean` (boolean)
- nulls: 1005 (100.00%)
- unique values: 0
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 1005

## electrocardiographs_ecg_ischemia_without_st_pET

- dtype: `Boolean` (boolean)
- nulls: 1005 (100.00%)
- unique values: 0
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 1005

## electrocardiographs_ecg_type_of_rhythms_pET_first

- dtype: `List(String)` (categorical)
- nulls: 1005 (100.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 1005

## electrocardiographs_ecg_type_of_rhythms_pET_last

- dtype: `List(String)` (categorical)
- nulls: 1005 (100.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 1005

## smoking_status_smoker_last

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 971
  - `True`: 34

## smoking_status_formerSmoker_last

- dtype: `Boolean` (boolean)
- nulls: 785 (78.11%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 785
  - `False`: 111
  - `True`: 109

## smoking_status_smoker_totalSmokingDuration_sum

- dtype: `Int64` (numeric)
- nulls: 970 (96.52%)
- min/max: 1.0 / 2000.0
- mean/std: 241.8857 / 386.097044447497
- quantiles: {'0.05': 1.0, '0.25': 7.5, '0.5': 66.0, '0.75': 308.0, '0.95': 769.0999999999998}

## smoking_status_smoker_startTime_count

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 3.0
- mean/std: 0.0507 / 0.29337291902973484
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 0.0}

## nyha_nyha

- dtype: `String` (categorical)
- nulls: 1005 (100.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 1005

## nyha_nyha_pET

- dtype: `String` (categorical)
- nulls: 1005 (100.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 1005

## vital_signs_systolicBpDuringEncounter_value_pET

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 0 (0.00%)
- min/max: 60.0 / 210.0
- mean/std: 123.3262 / 21.91681126158149
- quantiles: {'0.05': 94.0, '0.25': 108.0, '0.5': 119.0, '0.75': 136.0, '0.95': 165.0}

## vital_signs_bmi_value_pET

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 0 (0.00%)
- min/max: 14.0 / 94.0
- mean/std: 27.2290 / 7.382153593815175
- quantiles: {'0.05': 18.599999999999998, '0.25': 22.1, '0.5': 25.5, '0.75': 30.8, '0.95': 40.66}

## lab_results_creatBS_value_p3a_avg

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 0 (0.00%)
- min/max: 3.9 / 170.0
- mean/std: 15.1085 / 12.165783354758137
- quantiles: {'0.05': 6.50325, '0.25': 9.33075, '0.5': 12.04515, '0.75': 16.851899999999997, '0.95': 30.073289999999986}

## lab_results_validSerumCreatinine_value_pET

- dtype: `Decimal(precision=38, scale=22)` (numeric)
- nulls: 0 (0.00%)
- min/max: 3.9 / 23.0
- mean/std: 12.6543 / 4.504124153913538
- quantiles: {'0.05': 6.5598, '0.25': 9.161100000000001, '0.5': 11.706554367197109, '0.75': 15.834000000000001, '0.95': 21.353279999999994}

## hyperkalemia_severity_categorizedValue

- dtype: `String` (categorical)
- nulls: 10 (1.00%)
- unique values: 5
- top values (shown only where count ≥ 5):
  - `normal`: 869
  - `mild`: 88
  - `severe`: 20
  - `moderate`: 18
  - null (missing): 10

## ckd_severity_categorizedValue

- dtype: `String` (categorical)
- nulls: 2 (0.20%)
- unique values: 7
- top values (shown only where count ≥ 5):
  - `mildly_decreased`: 325
  - `mild_to_moderate_decrease`: 197
  - `moderate_to_severe_decrease`: 195
  - `severe_decrease`: 136
  - `normal_or_high`: 118
  - `kidney_failure`: 32
  - null (missing): 2

## med_requests_activeDuringEncounter_bb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## med_requests_activeDuringEncounter_ace_inhibitors_arb_use_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## conditions_heartFailure_timeFromEarliest_first

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 67.0
- mean/std: 7.4547 / 13.803804186155077
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 8.0, '0.95': 40.799999999999955}

## conditions_heart_failure_hf_within_18mo_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `True`: 1005

## conditions_heart_failure_occurred_prior_to_18_months_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 830
  - `True`: 175

## conditions_ap_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 970
  - `True`: 35

## conditions_af_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 609
  - `True`: 396

## conditions_cm_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 933
  - `True`: 72

## conditions_dysl_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 835
  - `True`: 170

## conditions_hf_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 780
  - `False`: 225

## conditions_hyp_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 541
  - `True`: 464

## conditions_ihd_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 718
  - `True`: 287

## conditions_mi_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 971
  - `True`: 34

## conditions_pad_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 832
  - `True`: 173

## conditions_stroke_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 984
  - `True`: 21

## conditions_tia_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 999
  - `True`: 6

## conditions_vd_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 759
  - `False`: 246

## conditions_revasc_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## conditions_devices_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## conditions_aidshiv_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 988
  - `True`: 17

## conditions_copd_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 852
  - `True`: 153

## conditions_diabetes_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 703
  - `True`: 302

## conditions_dem_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 936
  - `True`: 69

## conditions_dep_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 998
  - `True`: 7

## conditions_dia_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## conditions_hyperthyroid_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 992
  - `True`: 13

## conditions_hypothyroid_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 920
  - `True`: 85

## conditions_ibd_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 961
  - `True`: 44

## conditions_ld_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 923
  - `True`: 82

## conditions_mc_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 914
  - `True`: 91

## conditions_osa_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 937
  - `True`: 68

## conditions_rd_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 933
  - `True`: 72

## conditions_ckd_chronic_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 734
  - `True`: 271

## conditions_myocarditis_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## conditions_pericardial_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 910
  - `True`: 95

## conditions_substance_abuse_during_pET_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 906
  - `True`: 99

## conditions_ap_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 935
  - `True`: 70

## conditions_af_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 524
  - `True`: 481

## conditions_cm_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 903
  - `True`: 102

## conditions_dysl_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 724
  - `True`: 281

## conditions_hf_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 840
  - `False`: 165

## conditions_hyp_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 620
  - `False`: 385

## conditions_ihd_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 617
  - `True`: 388

## conditions_mi_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 918
  - `True`: 87

## conditions_pad_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 770
  - `True`: 235

## conditions_stroke_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 916
  - `True`: 89

## conditions_tia_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 954
  - `True`: 51

## conditions_vd_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 703
  - `False`: 302

## conditions_revasc_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## conditions_devices_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## conditions_aidshiv_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 956
  - `True`: 49

## conditions_copd_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 798
  - `True`: 207

## conditions_diabetes_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 644
  - `True`: 361

## conditions_dem_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 934
  - `True`: 71

## conditions_dep_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 997
  - `True`: 8

## conditions_dia_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## conditions_hyperthyroid_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 982
  - `True`: 23

## conditions_hypothyroid_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 880
  - `True`: 125

## conditions_ibd_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 869
  - `True`: 136

## conditions_ld_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 882
  - `True`: 123

## conditions_mc_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 839
  - `True`: 166

## conditions_osa_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 925
  - `True`: 80

## conditions_rd_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 875
  - `True`: 130

## conditions_ckd_chronic_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 677
  - `True`: 328

## conditions_myocarditis_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1001
  - 1 other distinct value(s) suppressed, covering 4 row(s) (count below 5 and/or ranked beyond top 20)

## conditions_pericardial_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 907
  - `True`: 98

## conditions_substance_abuse_pre_adm_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 848
  - `True`: 157

## conditions_af_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 544
  - `False`: 461

## conditions_ckd_chronic_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 609
  - `True`: 396

## conditions_cm_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 882
  - `True`: 123

## conditions_copd_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 768
  - `True`: 237

## conditions_dem_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 913
  - `True`: 92

## conditions_dep_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 994
  - `True`: 11

## conditions_diabetes_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 593
  - `True`: 412

## conditions_hypothyroid_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 864
  - `True`: 141

## conditions_hyp_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 712
  - `False`: 293

## conditions_ihd_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 557
  - `True`: 448

## conditions_mc_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 822
  - `True`: 183

## conditions_mi_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 895
  - `True`: 110

## conditions_pad_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 701
  - `True`: 304

## conditions_rd_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 855
  - `True`: 150

## conditions_stroke_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 906
  - `True`: 99

## conditions_vd_pre_dc_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 917
  - `False`: 88

## med_admins_rasi_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 607
  - `True`: 398

## med_admins_arni_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 909
  - `True`: 96

## med_admins_acei_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 687
  - `True`: 318

## med_admins_arb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 698
  - `True`: 307

## med_admins_mra_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 910
  - `False`: 95

## med_admins_diuretics_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 974
  - `False`: 31

## med_admins_diuretics_loop_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 883
  - `False`: 122

## med_admins_anti_coag_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 806
  - `False`: 199

## med_admins_anti_plat_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 794
  - `True`: 211

## med_admins_thrombolytic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 982
  - `True`: 23

## med_admins_bb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 718
  - `True`: 287

## med_admins_ccb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 765
  - `True`: 240

## med_admins_digitalis_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 817
  - `True`: 188

## med_admins_antiarrhytmic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 932
  - `True`: 73

## med_admins_inotropes_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 830
  - `True`: 175

## med_admins_vasodil_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 875
  - `True`: 130

## med_admins_platelet_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 794
  - `True`: 211

## med_admins_ll_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `True`: 571
  - `False`: 434

## med_admins_ivabradine_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 997
  - `True`: 8

## med_admins_potassium_binders_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## med_admins_insulins_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 814
  - `True`: 191

## med_admins_oral_antidiabetic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 594
  - `True`: 411

## med_admins_sglt2i_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 726
  - `True`: 279

## med_admins_ari_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## med_admins_rdoad_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 622
  - `True`: 383

## med_admins_rdoad_syst_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 699
  - `True`: 306

## med_admins_cortico_syst_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 817
  - `True`: 188

## med_admins_antiinfl_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 976
  - `True`: 29

## med_requests_rasi_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## med_requests_arni_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## med_requests_acei_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## med_requests_arb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## med_requests_mra_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## med_requests_diuretics_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## med_requests_diuretics_loop_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## med_requests_anti_coag_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## med_requests_anti_plat_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## med_requests_thrombolytic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## med_requests_bb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## med_requests_ccb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## med_requests_digitalis_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## med_requests_antiarrhytmic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## med_requests_inotropes_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## med_requests_vasodil_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## med_requests_platelet_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## med_requests_ll_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## med_requests_ivabradine_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## med_requests_potassium_binders_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## med_requests_insulins_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## med_requests_oral_antidiabetic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## med_requests_sglt2i_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## med_requests_ari_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## med_requests_rdoad_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## med_requests_rdoad_syst_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## med_requests_cortico_syst_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## med_requests_antiinfl_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## med_admins_history_rasi_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 822
  - `True`: 183

## med_admins_history_arni_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 969
  - `True`: 36

## med_admins_history_acei_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 845
  - `True`: 160

## med_admins_history_arb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 874
  - `True`: 131

## med_admins_history_mra_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 682
  - `True`: 323

## med_admins_history_diuretics_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 656
  - `True`: 349

## med_admins_history_diuretics_loop_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 696
  - `True`: 309

## med_admins_history_anti_coag_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 698
  - `True`: 307

## med_admins_history_anti_plat_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 905
  - `True`: 100

## med_admins_history_thrombolytic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1002
  - 1 other distinct value(s) suppressed, covering 3 row(s) (count below 5 and/or ranked beyond top 20)

## med_admins_history_bb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 877
  - `True`: 128

## med_admins_history_ccb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 891
  - `True`: 114

## med_admins_history_digitalis_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 921
  - `True`: 84

## med_admins_history_antiarrhytmic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 940
  - `True`: 65

## med_admins_history_inotropes_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 895
  - `True`: 110

## med_admins_history_vasodil_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 943
  - `True`: 62

## med_admins_history_platelet_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 905
  - `True`: 100

## med_admins_history_ll_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 764
  - `True`: 241

## med_admins_history_ivabradine_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 1001
  - 1 other distinct value(s) suppressed, covering 4 row(s) (count below 5 and/or ranked beyond top 20)

## med_admins_history_potassium_binders_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## med_admins_history_insulins_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 909
  - `True`: 96

## med_admins_history_oral_antidiabetic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 836
  - `True`: 169

## med_admins_history_sglt2i_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 917
  - `True`: 88

## med_admins_history_ari_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## med_admins_history_rdoad_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 816
  - `True`: 189

## med_admins_history_rdoad_syst_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 836
  - `True`: 169

## med_admins_history_cortico_syst_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 882
  - `True`: 123

## med_admins_history_antiinfl_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - `False`: 982
  - `True`: 23

## med_requests_history_rasi_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## med_requests_history_arni_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## med_requests_history_acei_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## med_requests_history_arb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## med_requests_history_mra_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## med_requests_history_diuretics_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## med_requests_history_diuretics_loop_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## med_requests_history_anti_coag_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## med_requests_history_anti_plat_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## med_requests_history_thrombolytic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## med_requests_history_bb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## med_requests_history_ccb_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## med_requests_history_digitalis_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## med_requests_history_antiarrhytmic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## med_requests_history_inotropes_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## med_requests_history_vasodil_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## med_requests_history_platelet_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## med_requests_history_ll_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## med_requests_history_ivabradine_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## med_requests_history_potassium_binders_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## med_requests_history_insulins_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## med_requests_history_oral_antidiabetic_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## med_requests_history_sglt2i_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## med_requests_history_ari_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## med_requests_history_rdoad_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## med_requests_history_rdoad_syst_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## med_requests_history_cortico_syst_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## med_requests_history_antiinfl_any

- dtype: `Boolean` (boolean)
- nulls: 0 (0.00%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - `False`: 1005

## encounter_primary_reason_HF_Disease_f5a_w7d_first

- dtype: `Boolean` (boolean)
- nulls: 974 (96.92%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 974
  - `False`: 30
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## encounter_primary_reason_HF_Disease_f5a_w1mo_first

- dtype: `Boolean` (boolean)
- nulls: 901 (89.65%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 901
  - `False`: 102
  - 1 other distinct value(s) suppressed, covering 2 row(s) (count below 5 and/or ranked beyond top 20)

## encounter_primary_reason_HF_Disease_f5a_w3mo_first

- dtype: `Boolean` (boolean)
- nulls: 823 (81.89%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 823
  - `False`: 177
  - `True`: 5

## encounter_primary_reason_HF_Disease_f5a_w6mo_first

- dtype: `Boolean` (boolean)
- nulls: 778 (77.41%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 778
  - `False`: 220
  - `True`: 7

## encounter_primary_reason_HF_Disease_f5a_w1a_first

- dtype: `Boolean` (boolean)
- nulls: 731 (72.74%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 731
  - `False`: 267
  - `True`: 7

## encounter_primary_reason_HF_Disease_f5a_w3a_first

- dtype: `Boolean` (boolean)
- nulls: 688 (68.46%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 688
  - `False`: 310
  - `True`: 7

## encounter_primary_reason_HF_Disease_f5a_w5a_first

- dtype: `Boolean` (boolean)
- nulls: 679 (67.56%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 679
  - `False`: 319
  - `True`: 7

## encounter_primary_reason_number_of_days_to_rehosp_for_heart_failure_f5a_first

- dtype: `Int32` (numeric)
- nulls: 998 (99.30%)
- min/max: 1.0 / 150.0
- mean/std: 66.1429 / 53.86225818829784
- quantiles: {'0.05': 4.0, '0.25': 24.5, '0.5': 68.0, '0.75': 102.0, '0.95': 136.5}

## encounter_primary_reason_number_of_HF_rehospitalizations_5a_f5a_count

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 1.0
- mean/std: 0.0080 / 0.0889083776115733
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 0.0}

## encounter_primary_reason_CV_Disease_f5a_w7d_first

- dtype: `Boolean` (boolean)
- nulls: 974 (96.92%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 974
  - `False`: 27
  - 1 other distinct value(s) suppressed, covering 4 row(s) (count below 5 and/or ranked beyond top 20)

## encounter_primary_reason_CV_Disease_f5a_w1mo_first

- dtype: `Boolean` (boolean)
- nulls: 901 (89.65%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 901
  - `False`: 91
  - `True`: 13

## encounter_primary_reason_CV_Disease_f5a_w3mo_first

- dtype: `Boolean` (boolean)
- nulls: 823 (81.89%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 823
  - `False`: 156
  - `True`: 26

## encounter_primary_reason_CV_Disease_f5a_w6mo_first

- dtype: `Boolean` (boolean)
- nulls: 778 (77.41%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 778
  - `False`: 192
  - `True`: 35

## encounter_primary_reason_CV_Disease_f5a_w1a_first

- dtype: `Boolean` (boolean)
- nulls: 731 (72.74%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 731
  - `False`: 229
  - `True`: 45

## encounter_primary_reason_CV_Disease_f5a_w3a_first

- dtype: `Boolean` (boolean)
- nulls: 688 (68.46%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 688
  - `False`: 262
  - `True`: 55

## encounter_primary_reason_CV_Disease_f5a_w5a_first

- dtype: `Boolean` (boolean)
- nulls: 679 (67.56%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 679
  - `False`: 269
  - `True`: 57

## encounter_primary_reason_number_of_days_to_rehosp_for_CV_f5a_first

- dtype: `Int32` (numeric)
- nulls: 948 (94.33%)
- min/max: 1.0 / 1500.0
- mean/std: 236.2456 / 308.5802327192438
- quantiles: {'0.05': 2.0, '0.25': 38.0, '0.5': 107.0, '0.75': 298.0, '0.95': 847.5999999999997}

## encounter_primary_reason_number_of_CV_rehospitalizations_5a_f5a_count

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 6.0
- mean/std: 0.1453 / 0.5269528749533089
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 1.0}

## encounter_primary_reason_non_CV_Disease_f5a_w7d_first

- dtype: `Boolean` (boolean)
- nulls: 974 (96.92%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 974
  - `True`: 27
  - 1 other distinct value(s) suppressed, covering 4 row(s) (count below 5 and/or ranked beyond top 20)

## encounter_primary_reason_non_CV_Disease_f5a_w1mo_first

- dtype: `Boolean` (boolean)
- nulls: 901 (89.65%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 901
  - `True`: 91
  - `False`: 13

## encounter_primary_reason_non_CV_Disease_f5a_w3mo_first

- dtype: `Boolean` (boolean)
- nulls: 823 (81.89%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 823
  - `True`: 159
  - `False`: 23

## encounter_primary_reason_non_CV_Disease_f5a_w6mo_first

- dtype: `Boolean` (boolean)
- nulls: 778 (77.41%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 778
  - `True`: 198
  - `False`: 29

## encounter_primary_reason_non_CV_Disease_f5a_w1a_first

- dtype: `Boolean` (boolean)
- nulls: 731 (72.74%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 731
  - `True`: 238
  - `False`: 36

## encounter_primary_reason_non_CV_Disease_f5a_w3a_first

- dtype: `Boolean` (boolean)
- nulls: 688 (68.46%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 688
  - `True`: 274
  - `False`: 43

## encounter_primary_reason_non_CV_Disease_f5a_w5a_first

- dtype: `Boolean` (boolean)
- nulls: 679 (67.56%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 679
  - `True`: 281
  - `False`: 45

## encounter_primary_reason_number_of_days_to_rehosp_for_non_CV_f5a_first

- dtype: `Int32` (numeric)
- nulls: 724 (72.04%)
- min/max: 1.0 / 1800.0
- mean/std: 189.4199 / 289.9698780439997
- quantiles: {'0.05': 4.0, '0.25': 22.0, '0.5': 71.0, '0.75': 219.0, '0.95': 818.0}

## encounter_primary_reason_number_of_non_CV_rehospitalizations_5a_f5a_count

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 84.0
- mean/std: 1.3423 / 4.30905799602026
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 1.0, '0.95': 6.0}

## encounter_primary_reason_renal_complications_f5a_w7d_first

- dtype: `Boolean` (boolean)
- nulls: 974 (96.92%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 974
  - `False`: 31

## encounter_primary_reason_renal_complications_f5a_w1mo_first

- dtype: `Boolean` (boolean)
- nulls: 901 (89.65%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 901
  - `False`: 104

## encounter_primary_reason_renal_complications_f5a_w3mo_first

- dtype: `Boolean` (boolean)
- nulls: 823 (81.89%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 823
  - `False`: 182

## encounter_primary_reason_renal_complications_f5a_w6mo_first

- dtype: `Boolean` (boolean)
- nulls: 778 (77.41%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 778
  - `False`: 226
  - 1 other distinct value(s) suppressed, covering 1 row(s) (count below 5 and/or ranked beyond top 20)

## encounter_primary_reason_renal_complications_f5a_w1a_first

- dtype: `Boolean` (boolean)
- nulls: 731 (72.74%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 731
  - `False`: 272
  - 1 other distinct value(s) suppressed, covering 2 row(s) (count below 5 and/or ranked beyond top 20)

## encounter_primary_reason_renal_complications_f5a_w3a_first

- dtype: `Boolean` (boolean)
- nulls: 688 (68.46%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 688
  - `False`: 315
  - 1 other distinct value(s) suppressed, covering 2 row(s) (count below 5 and/or ranked beyond top 20)

## encounter_primary_reason_renal_complications_f5a_w5a_first

- dtype: `Boolean` (boolean)
- nulls: 679 (67.56%)
- unique values: 2
- top values (shown only where count ≥ 5):
  - null (missing): 679
  - `False`: 324
  - 1 other distinct value(s) suppressed, covering 2 row(s) (count below 5 and/or ranked beyond top 20)

## encounter_primary_reason_number_of_days_to_rehosp_for_renal_complications_f5a_first

- dtype: `Int32` (numeric)
- nulls: 1003 (99.80%)
- min/max: 140.0 / 240.0
- mean/std: 189.5000 / 62.932503525602726
- quantiles: {'0.05': 149.45, '0.25': 167.25, '0.5': 189.5, '0.75': 211.75, '0.95': 229.55}

## encounter_primary_reason_number_of_renal_rehospitalizations_5a_f5a_count

- dtype: `Int64` (numeric)
- nulls: 0 (0.00%)
- min/max: 0.0 / 2.0
- mean/std: 0.0030 / 0.07050645456677775
- quantiles: {'0.05': 0.0, '0.25': 0.0, '0.5': 0.0, '0.75': 0.0, '0.95': 0.0}

## cause_of_death_isCV_f5a_w7d_first

- dtype: `Boolean` (boolean)
- nulls: 989 (98.41%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 989
  - `False`: 16

## cause_of_death_isCV_f5a_w1mo_first

- dtype: `Boolean` (boolean)
- nulls: 950 (94.53%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 950
  - `False`: 55

## cause_of_death_isCV_f5a_w3mo_first

- dtype: `Boolean` (boolean)
- nulls: 894 (88.96%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 894
  - `False`: 111

## cause_of_death_isCV_f5a_w6mo_first

- dtype: `Boolean` (boolean)
- nulls: 845 (84.08%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 845
  - `False`: 160

## cause_of_death_isCV_f5a_w1a_first

- dtype: `Boolean` (boolean)
- nulls: 761 (75.72%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 761
  - `False`: 244

## cause_of_death_isCV_f5a_w3a_first

- dtype: `Boolean` (boolean)
- nulls: 653 (64.98%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 653
  - `False`: 352

## cause_of_death_isCV_f5a_w5a_first

- dtype: `Boolean` (boolean)
- nulls: 620 (61.69%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 620
  - `False`: 385

## cause_of_death_number_of_days_to_death_for_CV_f5a_first

- dtype: `Int32` (numeric)
- nulls: 1005 (100.00%)
- All values are null.

## cause_of_death_isRenal_f5a_w7d_first

- dtype: `Boolean` (boolean)
- nulls: 989 (98.41%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 989
  - `False`: 16

## cause_of_death_isRenal_f5a_w1mo_first

- dtype: `Boolean` (boolean)
- nulls: 950 (94.53%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 950
  - `False`: 55

## cause_of_death_isRenal_f5a_w3mo_first

- dtype: `Boolean` (boolean)
- nulls: 894 (88.96%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 894
  - `False`: 111

## cause_of_death_isRenal_f5a_w6mo_first

- dtype: `Boolean` (boolean)
- nulls: 845 (84.08%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 845
  - `False`: 160

## cause_of_death_isRenal_f5a_w1a_first

- dtype: `Boolean` (boolean)
- nulls: 761 (75.72%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 761
  - `False`: 244

## cause_of_death_isRenal_f5a_w3a_first

- dtype: `Boolean` (boolean)
- nulls: 653 (64.98%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 653
  - `False`: 352

## cause_of_death_isRenal_f5a_w5a_first

- dtype: `Boolean` (boolean)
- nulls: 620 (61.69%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 620
  - `False`: 385

## cause_of_death_number_of_days_to_death_for_renal_f5a_first

- dtype: `Int32` (numeric)
- nulls: 1005 (100.00%)
- All values are null.

## cause_of_death_isNonRenalAndNonCV_f5a_w7d_first

- dtype: `Boolean` (boolean)
- nulls: 989 (98.41%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 989
  - `False`: 16

## cause_of_death_isNonRenalAndNonCV_f5a_w1mo_first

- dtype: `Boolean` (boolean)
- nulls: 950 (94.53%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 950
  - `False`: 55

## cause_of_death_isNonRenalAndNonCV_f5a_w3mo_first

- dtype: `Boolean` (boolean)
- nulls: 894 (88.96%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 894
  - `False`: 111

## cause_of_death_isNonRenalAndNonCV_f5a_w6mo_first

- dtype: `Boolean` (boolean)
- nulls: 845 (84.08%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 845
  - `False`: 160

## cause_of_death_isNonRenalAndNonCV_f5a_w1a_first

- dtype: `Boolean` (boolean)
- nulls: 761 (75.72%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 761
  - `False`: 244

## cause_of_death_isNonRenalAndNonCV_f5a_w3a_first

- dtype: `Boolean` (boolean)
- nulls: 653 (64.98%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 653
  - `False`: 352

## cause_of_death_isNonRenalAndNonCV_f5a_w5a_first

- dtype: `Boolean` (boolean)
- nulls: 620 (61.69%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 620
  - `False`: 385

## cause_of_death_number_of_days_to_death_for_non_renal_and_non_CV_f5a_first

- dtype: `Int32` (numeric)
- nulls: 1005 (100.00%)
- All values are null.

## cause_of_death_isAllCause_f5a_w7d_first

- dtype: `Boolean` (boolean)
- nulls: 989 (98.41%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 989
  - `True`: 16

## cause_of_death_isAllCause_f5a_w1mo_first

- dtype: `Boolean` (boolean)
- nulls: 950 (94.53%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 950
  - `True`: 55

## cause_of_death_isAllCause_f5a_w3mo_first

- dtype: `Boolean` (boolean)
- nulls: 894 (88.96%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 894
  - `True`: 111

## cause_of_death_isAllCause_f5a_w6mo_first

- dtype: `Boolean` (boolean)
- nulls: 845 (84.08%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 845
  - `True`: 160

## cause_of_death_isAllCause_f5a_w1a_first

- dtype: `Boolean` (boolean)
- nulls: 761 (75.72%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 761
  - `True`: 244

## cause_of_death_isAllCause_f5a_w3a_first

- dtype: `Boolean` (boolean)
- nulls: 653 (64.98%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 653
  - `True`: 352

## cause_of_death_isAllCause_f5a_w5a_first

- dtype: `Boolean` (boolean)
- nulls: 620 (61.69%)
- unique values: 1
- ⚠️ constant column (single value)
- top values (shown only where count ≥ 5):
  - null (missing): 620
  - `True`: 385

## cause_of_death_number_of_days_to_death_for_all_cause_f5a_first

- dtype: `Int32` (numeric)
- nulls: 620 (61.69%)
- min/max: 0.0 / 1800.0
- mean/std: 385.7247 / 408.8724422101809
- quantiles: {'0.05': 9.0, '0.25': 67.0, '0.5': 243.0, '0.75': 570.0, '0.95': 1261.9999999999998}

## eGFR_2021_ckd_epi_creatinine

- dtype: `Decimal(precision=38, scale=6)` (numeric)
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

## maggic_total_score

- dtype: `Int32` (numeric)
- nulls: 1005 (100.00%)
- All values are null.
