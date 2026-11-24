# Row-Coherence Audit

233 rules ({'implication': 221, 'category_range': 4, 'days_bounds': 1, 'flag_days_consistency': 7}) mined/learned from the TRAIN split and validated on real data. The holdout row is the fair baseline: real, unseen patients violating the same rules. A synthetic dataset far above it produces rows that are individually implausible patients even when every column's distribution is correct.

Two measures, answering different questions: the violation RATE is per applicable rule-check, while the row SHARE is the fraction of patients carrying at least one violation -- the one a release decision turns on, read against the real holdout's own share.

The 'consequent Missing' column is the evasion check for implication rules: the share of antecedent-true checks whose consequent was Missing and thus undecidable. A generator can push its violation rate toward zero by emitting Missing consequents; a share far above the real frames' reveals exactly that.

| frame | applicable checks | violations | violation rate | rules violated | consequent Missing | rows with >=1 violation |
|---|---|---|---|---|---|---|
| train (real) | 116221 | 2 | 2e-05 | 1/233 | 23.8% | 0.1% (2/3520) |
| holdout (real, unseen) | 37312 | 9 | 0.00024 | 6/233 | 24.9% | 0.8% (9/1174) |
| synthetic[aim40_eps1_seed0] | 7336 | 272 | 0.03708 | 4/233 | 3.1% | 6.5% (228/3520) |
| synthetic[aim50_eps1_seed0] | 7667 | 405 | 0.05282 | 4/233 | 3.4% | 11.0% (388/3520) |
| synthetic[ctgan_qt_seed0] | 103606 | 40127 | 0.3873 | 232/233 | 36.0% | 97.2% (3423/3520) |
| synthetic[ctgan_seed0] | 127346 | 38761 | 0.30438 | 232/233 | 32.6% | 99.7% (3509/3520) |
| synthetic[ctgan_seed1] | 111343 | 38952 | 0.34984 | 230/233 | 37.9% | 99.6% (3506/3520) |
| synthetic[ctgan_seed2] | 112002 | 33602 | 0.30001 | 231/233 | 33.7% | 97.2% (3421/3520) |
| synthetic[ddpm_g_seed0] | 184906 | 47924 | 0.25918 | 232/233 | 26.5% | 99.5% (3504/3520) |
| synthetic[ddpm_seed0] | 224076 | 85917 | 0.38343 | 232/233 | 20.3% | 100.0% (3519/3520) |
| synthetic[ddpm_seed1] | 246783 | 95453 | 0.38679 | 232/233 | 17.9% | 100.0% (3520/3520) |
| synthetic[ddpm_seed2] | 229672 | 88419 | 0.38498 | 232/233 | 20.2% | 100.0% (3519/3520) |
| synthetic[dpctgan_eps10_seed0] | 51229 | 37605 | 0.73406 | 91/233 | 55.5% | 100.0% (3520/3520) |
| synthetic[dpctgan_eps15_seed0] | 27082 | 8109 | 0.29942 | 53/233 | 52.0% | 100.0% (3519/3520) |
| synthetic[dpctgan_eps15_seed1] | 59835 | 45583 | 0.76181 | 92/233 | 56.3% | 100.0% (3520/3520) |
| synthetic[dpctgan_eps15_seed2] | 63938 | 8560 | 0.13388 | 61/233 | 56.6% | 96.5% (3397/3520) |
| synthetic[dpctgan_eps1_seed0] | 90832 | 53796 | 0.59226 | 100/233 | 50.5% | 100.0% (3520/3520) |
| synthetic[dpctgan_eps20_seed0] | 84975 | 40517 | 0.47681 | 70/233 | 47.6% | 100.0% (3520/3520) |
| synthetic[dpctgan_eps5_seed0] | 47236 | 22344 | 0.47303 | 78/233 | 65.4% | 100.0% (3520/3520) |
| synthetic[dpctgan_eps8_seed0] | 64972 | 31840 | 0.49006 | 65/233 | 56.9% | 100.0% (3520/3520) |
| synthetic[gaussian_copula_seed0] | 82420 | 22827 | 0.27696 | 230/233 | 48.6% | 96.2% (3388/3520) |
| synthetic[gaussian_copula_seed1] | 83650 | 23255 | 0.278 | 232/233 | 48.1% | 96.2% (3388/3520) |
| synthetic[gaussian_copula_seed2] | 81488 | 22884 | 0.28083 | 230/233 | 48.5% | 96.3% (3390/3520) |
| synthetic[mst_eps0p5_seed0] | 95201 | 24683 | 0.25927 | 164/233 | 34.6% | 83.8% (2948/3520) |
| synthetic[mst_eps10_seed0] | 115942 | 4731 | 0.0408 | 194/233 | 23.5% | 24.7% (871/3520) |
| synthetic[mst_eps15_seed0] | 115885 | 3426 | 0.02956 | 191/233 | 23.4% | 21.0% (738/3520) |
| synthetic[mst_eps15_seed1] | 116107 | 3837 | 0.03305 | 199/233 | 23.3% | 20.9% (737/3520) |
| synthetic[mst_eps15_seed2] | 116042 | 4275 | 0.03684 | 189/233 | 23.3% | 23.2% (816/3520) |
| synthetic[mst_eps1_seed0] | 110073 | 25246 | 0.22936 | 198/233 | 28.0% | 67.1% (2361/3520) |
| synthetic[mst_eps20_seed0] | 115617 | 3699 | 0.03199 | 211/233 | 23.4% | 21.3% (751/3520) |
| synthetic[mst_eps5_seed0] | 115907 | 7155 | 0.06173 | 205/233 | 23.3% | 34.9% (1229/3520) |
| synthetic[mst_eps8_seed0] | 115944 | 5236 | 0.04516 | 191/233 | 23.2% | 26.7% (939/3520) |
| synthetic[patectgan_eps15_seed0] | 65650 | 2792 | 0.04253 | 140/233 | 27.4% | 47.6% (1676/3520) |
| synthetic[patectgan_eps1_seed0] | 102027 | 45143 | 0.44246 | 232/233 | 46.7% | 100.0% (3519/3520) |
| synthetic[patectgan_eps5_seed0] | 111608 | 6509 | 0.05832 | 170/233 | 24.5% | 77.8% (2737/3520) |
| synthetic[tvae_cap256_seed0] | 102914 | 2226 | 0.02163 | 157/233 | 32.2% | 27.0% (952/3520) |
| synthetic[tvae_ep1000_seed0] | 99291 | 2186 | 0.02202 | 178/233 | 30.2% | 25.0% (880/3520) |
| synthetic[tvae_ind_seed0] | 101277 | 1866 | 0.01842 | 162/233 | 31.5% | 21.2% (747/3520) |
| synthetic[tvae_qt_seed0] | 98820 | 2100 | 0.02125 | 155/233 | 30.7% | 28.2% (992/3520) |
| synthetic[tvae_qt_seed1] | 103749 | 1983 | 0.01911 | 139/233 | 31.0% | 25.5% (899/3520) |
| synthetic[tvae_qt_seed2] | 102015 | 2521 | 0.02471 | 148/233 | 30.4% | 31.0% (1091/3520) |
| synthetic[tvae_seed0] | 101847 | 2164 | 0.02125 | 165/233 | 30.2% | 26.1% (919/3520) |
| synthetic[tvae_seed1] | 99205 | 2255 | 0.02273 | 155/233 | 32.1% | 24.7% (868/3520) |
| synthetic[tvae_seed2] | 98644 | 1938 | 0.01965 | 137/233 | 33.2% | 24.8% (872/3520) |

## Worst rules per synthetic dataset

**synthetic[aim40_eps1_seed0]**
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w1mo_first` (implication): rate 0.46296 over 54 rows
- `ckd_severity_from_calculated_egfr vs lab_results_valideGFR_value_last` (category_range): rate 0.07739 over 2946 rows
- `ckd_severity_from_calculated_egfr vs lab_results_valideGFR_value_first` (category_range): rate 0.00601 over 2997 rows
- `med_acei => med_rasi` (implication): rate 0.00075 over 1339 rows

**synthetic[aim50_eps1_seed0]**
- `med_arni => med_rasi` (implication): rate 0.39118 over 363 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w1mo_first` (implication): rate 0.35294 over 68 rows
- `ckd_severity_from_calculated_egfr vs lab_results_valideGFR_value_last` (category_range): rate 0.08389 over 2837 rows
- `med_acei => med_rasi` (implication): rate 0.00075 over 1325 rows

**synthetic[ctgan_qt_seed0]**
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w5a_first` (implication): rate 0.95652 over 161 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 0.95376 over 173 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w1a_first` (implication): rate 0.95031 over 161 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w6mo_first` (implication): rate 0.94483 over 145 rows
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first => encounter_primary_reason_HF_Disease_f5a_w6mo_first` (implication): rate 0.93701 over 127 rows

**synthetic[ctgan_seed0]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 1.0 over 146 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w3a_first` (implication): rate 0.97561 over 123 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w5a_first` (implication): rate 0.96212 over 132 rows
- `encounter_primary_reason_HF_Disease_f5a_w3a_first => encounter_primary_reason_HF_Disease_f5a_w1a_first` (implication): rate 0.95699 over 93 rows
- `encounter_primary_reason_HF_Disease_f5a_w1a_first => encounter_primary_reason_HF_Disease_f5a_w3a_first` (implication): rate 0.95652 over 92 rows

**synthetic[ctgan_seed1]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 0.98913 over 184 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w3mo_first` (implication): rate 0.95302 over 149 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w5a_first` (implication): rate 0.94767 over 172 rows
- `encounter_primary_reason_HF_Disease_f5a_w1a_first => encounter_primary_reason_HF_Disease_f5a_w3mo_first` (implication): rate 0.94652 over 187 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w6mo_first` (implication): rate 0.93671 over 158 rows

**synthetic[ctgan_seed2]**
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w3mo_first` (implication): rate 0.96 over 50 rows
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first => encounter_primary_reason_HF_Disease_f5a_w3a_first` (implication): rate 0.9434 over 106 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w3a_first` (implication): rate 0.9403 over 67 rows
- `cause_of_death_isCV_f5a_w1a_first => cause_of_death_isCV_f5a_w3a_first` (implication): rate 0.93814 over 97 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 0.92982 over 114 rows

**synthetic[ddpm_g_seed0]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 1.0 over 1709 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 1.0 over 1515 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w3mo_first` (flag_days_consistency): rate 1.0 over 1685 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w6mo_first` (flag_days_consistency): rate 1.0 over 1658 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1a_first` (flag_days_consistency): rate 1.0 over 1194 rows

**synthetic[ddpm_seed0]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 1.0 over 2014 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 1.0 over 1382 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w3mo_first` (flag_days_consistency): rate 1.0 over 1637 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w6mo_first` (flag_days_consistency): rate 1.0 over 1742 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1a_first` (flag_days_consistency): rate 1.0 over 1119 rows

**synthetic[ddpm_seed1]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 1.0 over 1665 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 1.0 over 1502 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w3mo_first` (flag_days_consistency): rate 1.0 over 1684 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w6mo_first` (flag_days_consistency): rate 1.0 over 1619 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1a_first` (flag_days_consistency): rate 1.0 over 1390 rows

**synthetic[ddpm_seed2]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 1.0 over 1146 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 1.0 over 2037 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w3mo_first` (flag_days_consistency): rate 1.0 over 1696 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w6mo_first` (flag_days_consistency): rate 1.0 over 1434 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1a_first` (flag_days_consistency): rate 1.0 over 1178 rows

**synthetic[dpctgan_eps10_seed0]**
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w3mo_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w5a_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w3mo_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w1a_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w3a_first` (implication): rate 1.0 over 1 rows

**synthetic[dpctgan_eps15_seed0]**
- `smoking_status_formerSmoker_last => smoking_status_smoker_last` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w1a_first => encounter_primary_reason_HF_Disease_f5a_w5a_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w1a_first => encounter_primary_reason_CV_Disease_f5a_w1mo_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w1a_first => encounter_primary_reason_CV_Disease_f5a_w3mo_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w1a_first => encounter_primary_reason_CV_Disease_f5a_w5a_first` (implication): rate 1.0 over 1 rows

**synthetic[dpctgan_eps15_seed1]**
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w1a_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w5a_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w3mo_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w3a_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w5a_first` (implication): rate 1.0 over 1 rows

**synthetic[dpctgan_eps15_seed2]**
- `encounter_primary_reason_HF_Disease_f5a_w6mo_first => encounter_primary_reason_CV_Disease_f5a_w1mo_first` (implication): rate 1.0 over 2 rows
- `encounter_primary_reason_HF_Disease_f5a_w6mo_first => encounter_primary_reason_CV_Disease_f5a_w6mo_first` (implication): rate 1.0 over 2 rows
- `encounter_primary_reason_HF_Disease_f5a_w6mo_first => encounter_primary_reason_CV_Disease_f5a_w3a_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w6mo_first => encounter_primary_reason_CV_Disease_f5a_w5a_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w1a_first => encounter_primary_reason_CV_Disease_f5a_w1mo_first` (implication): rate 1.0 over 3 rows

**synthetic[dpctgan_eps1_seed0]**
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w3mo_first` (implication): rate 1.0 over 2 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w1a_first` (implication): rate 1.0 over 2 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w3a_first` (implication): rate 1.0 over 2 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w5a_first` (implication): rate 1.0 over 2 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w3a_first` (implication): rate 1.0 over 2 rows

**synthetic[dpctgan_eps20_seed0]**
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first => encounter_primary_reason_HF_Disease_f5a_w1mo_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first => encounter_primary_reason_HF_Disease_f5a_w3a_first` (implication): rate 1.0 over 3518 rows
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first => encounter_primary_reason_CV_Disease_f5a_w1mo_first` (implication): rate 1.0 over 3369 rows
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first => encounter_primary_reason_CV_Disease_f5a_w6mo_first` (implication): rate 1.0 over 3518 rows
- `encounter_primary_reason_HF_Disease_f5a_w6mo_first => encounter_primary_reason_HF_Disease_f5a_w1a_first` (implication): rate 1.0 over 16 rows

**synthetic[dpctgan_eps5_seed0]**
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w3mo_first` (implication): rate 1.0 over 2 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w3a_first` (implication): rate 1.0 over 2 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w5a_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w3mo_first` (implication): rate 1.0 over 2 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w6mo_first` (implication): rate 1.0 over 2 rows

**synthetic[dpctgan_eps8_seed0]**
- `encounter_primary_reason_HF_Disease_f5a_w6mo_first => encounter_primary_reason_HF_Disease_f5a_w1mo_first` (implication): rate 1.0 over 3513 rows
- `encounter_primary_reason_HF_Disease_f5a_w6mo_first => encounter_primary_reason_HF_Disease_f5a_w3mo_first` (implication): rate 1.0 over 3505 rows
- `encounter_primary_reason_HF_Disease_f5a_w6mo_first => encounter_primary_reason_HF_Disease_f5a_w1a_first` (implication): rate 1.0 over 3518 rows
- `encounter_primary_reason_HF_Disease_f5a_w6mo_first => encounter_primary_reason_HF_Disease_f5a_w3a_first` (implication): rate 1.0 over 3 rows
- `encounter_primary_reason_HF_Disease_f5a_w6mo_first => encounter_primary_reason_CV_Disease_f5a_w3mo_first` (implication): rate 1.0 over 1 rows

**synthetic[gaussian_copula_seed0]**
- `cause_of_death_isCV_f5a_w1a_first => cause_of_death_isCV_f5a_w5a_first` (implication): rate 1.0 over 35 rows
- `cause_of_death_isCV_f5a_w5a_first => cause_of_death_isCV_f5a_w1a_first` (implication): rate 1.0 over 4 rows
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first => encounter_primary_reason_HF_Disease_f5a_w5a_first` (implication): rate 0.99038 over 104 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 0.98413 over 126 rows
- `encounter_primary_reason_HF_Disease_f5a_w5a_first => encounter_primary_reason_HF_Disease_f5a_w3mo_first` (implication): rate 0.98246 over 57 rows

**synthetic[gaussian_copula_seed1]**
- `cause_of_death_isCV_f5a_w1a_first => cause_of_death_isCV_f5a_w5a_first` (implication): rate 1.0 over 34 rows
- `cause_of_death_isCV_f5a_w3a_first => cause_of_death_isCV_f5a_w5a_first` (implication): rate 1.0 over 58 rows
- `cause_of_death_isCV_f5a_w5a_first => cause_of_death_isCV_f5a_w1a_first` (implication): rate 1.0 over 6 rows
- `cause_of_death_isCV_f5a_w5a_first => cause_of_death_isCV_f5a_w3a_first` (implication): rate 1.0 over 9 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 1.0 over 119 rows

**synthetic[gaussian_copula_seed2]**
- `cause_of_death_isCV_f5a_w1a_first => cause_of_death_isCV_f5a_w5a_first` (implication): rate 1.0 over 37 rows
- `cause_of_death_isCV_f5a_w3a_first => cause_of_death_isCV_f5a_w5a_first` (implication): rate 1.0 over 45 rows
- `cause_of_death_isCV_f5a_w5a_first => cause_of_death_isCV_f5a_w1a_first` (implication): rate 1.0 over 6 rows
- `cause_of_death_isCV_f5a_w5a_first => cause_of_death_isCV_f5a_w3a_first` (implication): rate 1.0 over 13 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 1.0 over 223 rows

**synthetic[mst_eps0p5_seed0]**
- `smoking_status_formerSmoker_last => smoking_status_smoker_last` (implication): rate 1.0 over 35 rows
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first => encounter_primary_reason_HF_Disease_f5a_w1mo_first` (implication): rate 1.0 over 15 rows
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first => encounter_primary_reason_HF_Disease_f5a_w6mo_first` (implication): rate 1.0 over 101 rows
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first => encounter_primary_reason_CV_Disease_f5a_w1mo_first` (implication): rate 1.0 over 123 rows
- `encounter_primary_reason_HF_Disease_f5a_w1a_first => encounter_primary_reason_HF_Disease_f5a_w1mo_first` (implication): rate 1.0 over 7 rows

**synthetic[mst_eps10_seed0]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 1.0 over 120 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 1.0 over 275 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w3mo_first` (flag_days_consistency): rate 1.0 over 466 rows
- `cause_of_death_isCV_f5a_w1a_first => cause_of_death_isCV_f5a_w3a_first` (implication): rate 0.73077 over 52 rows
- `encounter_primary_reason_HF_Disease_f5a_w6mo_first => encounter_primary_reason_HF_Disease_f5a_w1mo_first` (implication): rate 0.63158 over 171 rows

**synthetic[mst_eps15_seed0]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 1.0 over 107 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 1.0 over 259 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w3mo_first` (flag_days_consistency): rate 1.0 over 453 rows
- `cause_of_death_isCV_f5a_w1a_first => cause_of_death_isCV_f5a_w5a_first` (implication): rate 0.69643 over 56 rows
- `encounter_primary_reason_HF_Disease_f5a_w1a_first => encounter_primary_reason_HF_Disease_f5a_w1mo_first` (implication): rate 0.63684 over 190 rows

**synthetic[mst_eps15_seed1]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 1.0 over 90 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 1.0 over 261 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w3mo_first` (flag_days_consistency): rate 1.0 over 460 rows
- `cause_of_death_isCV_f5a_w1a_first => cause_of_death_isCV_f5a_w5a_first` (implication): rate 0.52542 over 59 rows
- `encounter_primary_reason_HF_Disease_f5a_w6mo_first => encounter_primary_reason_HF_Disease_f5a_w1mo_first` (implication): rate 0.46667 over 180 rows

**synthetic[mst_eps15_seed2]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 1.0 over 108 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 1.0 over 267 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w3mo_first` (flag_days_consistency): rate 1.0 over 461 rows
- `encounter_primary_reason_HF_Disease_f5a_w3a_first => encounter_primary_reason_HF_Disease_f5a_w1mo_first` (implication): rate 0.64171 over 187 rows
- `encounter_primary_reason_HF_Disease_f5a_w5a_first => encounter_primary_reason_HF_Disease_f5a_w1mo_first` (implication): rate 0.60556 over 180 rows

**synthetic[mst_eps1_seed0]**
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w3mo_first` (implication): rate 1.0 over 65 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w1a_first` (implication): rate 1.0 over 68 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w3a_first` (implication): rate 1.0 over 68 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w5a_first` (implication): rate 1.0 over 68 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w3mo_first` (implication): rate 1.0 over 50 rows

**synthetic[mst_eps20_seed0]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 1.0 over 115 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 1.0 over 267 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w3mo_first` (flag_days_consistency): rate 1.0 over 444 rows
- `encounter_primary_reason_HF_Disease_f5a_w3a_first => encounter_primary_reason_HF_Disease_f5a_w1mo_first` (implication): rate 0.5988 over 167 rows
- `encounter_primary_reason_HF_Disease_f5a_w5a_first => encounter_primary_reason_HF_Disease_f5a_w1mo_first` (implication): rate 0.59036 over 166 rows

**synthetic[mst_eps5_seed0]**
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => encounter_primary_reason_CV_Disease_f5a_w1mo_first` (implication): rate 1.0 over 109 rows
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => encounter_primary_reason_CV_Disease_f5a_w3mo_first` (implication): rate 1.0 over 109 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 1.0 over 90 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 1.0 over 270 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w3mo_first` (flag_days_consistency): rate 1.0 over 440 rows

**synthetic[mst_eps8_seed0]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 1.0 over 88 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 1.0 over 271 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w3mo_first` (flag_days_consistency): rate 1.0 over 462 rows
- `encounter_primary_reason_HF_Disease_f5a_w5a_first => encounter_primary_reason_HF_Disease_f5a_w1mo_first` (implication): rate 0.60588 over 170 rows
- `encounter_primary_reason_HF_Disease_f5a_w3a_first => encounter_primary_reason_HF_Disease_f5a_w1mo_first` (implication): rate 0.58824 over 204 rows

**synthetic[patectgan_eps15_seed0]**
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first => encounter_primary_reason_HF_Disease_f5a_w1mo_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first => encounter_primary_reason_HF_Disease_f5a_w1a_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w1a_first => encounter_primary_reason_HF_Disease_f5a_w1mo_first` (implication): rate 1.0 over 2 rows
- `encounter_primary_reason_HF_Disease_f5a_w1a_first => encounter_primary_reason_HF_Disease_f5a_w3mo_first` (implication): rate 1.0 over 2 rows
- `encounter_primary_reason_HF_Disease_f5a_w1a_first => encounter_primary_reason_HF_Disease_f5a_w6mo_first` (implication): rate 1.0 over 5 rows

**synthetic[patectgan_eps1_seed0]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 1.0 over 283 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 0.97872 over 376 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w3mo_first` (flag_days_consistency): rate 0.95324 over 556 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w6mo_first` (flag_days_consistency): rate 0.94534 over 805 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1a_first` (flag_days_consistency): rate 0.93445 over 1373 rows

**synthetic[patectgan_eps5_seed0]**
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w3mo_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w6mo_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w1a_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w3a_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w5a_first` (implication): rate 1.0 over 1 rows

**synthetic[tvae_cap256_seed0]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 0.9322 over 59 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 0.46094 over 256 rows
- `cause_of_death_isCV_f5a_w3a_first => cause_of_death_isCV_f5a_w5a_first` (implication): rate 0.24468 over 94 rows
- `encounter_primary_reason_HF_Disease_f5a_w3a_first => encounter_primary_reason_HF_Disease_f5a_w1mo_first` (implication): rate 0.21311 over 61 rows
- `encounter_primary_reason_HF_Disease_f5a_w1a_first => encounter_primary_reason_HF_Disease_f5a_w1mo_first` (implication): rate 0.18644 over 59 rows

**synthetic[tvae_ep1000_seed0]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 0.89474 over 38 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 0.48603 over 179 rows
- `cause_of_death_isCV_f5a_w1a_first => cause_of_death_isCV_f5a_w5a_first` (implication): rate 0.29762 over 84 rows
- `cause_of_death_isCV_f5a_w3a_first => cause_of_death_isCV_f5a_w5a_first` (implication): rate 0.29333 over 150 rows
- `encounter_primary_reason_HF_Disease_f5a_w1a_first => encounter_primary_reason_HF_Disease_f5a_w1mo_first` (implication): rate 0.16667 over 54 rows

**synthetic[tvae_ind_seed0]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 0.29464 over 224 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 0.28947 over 76 rows
- `encounter_primary_reason_HF_Disease_f5a_w6mo_first => encounter_primary_reason_HF_Disease_f5a_w3a_first` (implication): rate 0.15464 over 97 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1a_first` (flag_days_consistency): rate 0.14851 over 1515 rows
- `encounter_primary_reason_HF_Disease_f5a_w3a_first => encounter_primary_reason_HF_Disease_f5a_w1mo_first` (implication): rate 0.13333 over 45 rows

**synthetic[tvae_qt_seed0]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 0.80952 over 63 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 0.45669 over 254 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w3mo_first` (flag_days_consistency): rate 0.22222 over 612 rows
- `cause_of_death_isCV_f5a_w3a_first => cause_of_death_isCV_f5a_w5a_first` (implication): rate 0.13571 over 140 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w6mo_first` (implication): rate 0.125 over 32 rows

**synthetic[tvae_qt_seed1]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 0.70833 over 48 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 0.43697 over 238 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w3mo_first` (flag_days_consistency): rate 0.25234 over 535 rows
- `cause_of_death_isCV_f5a_w1a_first => cause_of_death_isCV_f5a_w5a_first` (implication): rate 0.22449 over 49 rows
- `cause_of_death_isCV_f5a_w1a_first => cause_of_death_isCV_f5a_w3a_first` (implication): rate 0.16327 over 49 rows

**synthetic[tvae_qt_seed2]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 0.72727 over 44 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 0.48571 over 210 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w3mo_first` (flag_days_consistency): rate 0.25714 over 490 rows
- `encounter_primary_reason_HF_Disease_f5a_w1a_first => encounter_primary_reason_HF_Disease_f5a_w6mo_first` (implication): rate 0.15476 over 84 rows
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first => encounter_primary_reason_HF_Disease_f5a_w3mo_first` (implication): rate 0.15094 over 53 rows

**synthetic[tvae_seed0]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 0.91304 over 46 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 0.48718 over 195 rows
- `cause_of_death_isCV_f5a_w3a_first => cause_of_death_isCV_f5a_w5a_first` (implication): rate 0.17341 over 173 rows
- `cause_of_death_isCV_f5a_w1a_first => cause_of_death_isCV_f5a_w5a_first` (implication): rate 0.17117 over 111 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w3a_first` (flag_days_consistency): rate 0.14613 over 2094 rows

**synthetic[tvae_seed1]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 0.81818 over 33 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 0.48066 over 181 rows
- `cause_of_death_isCV_f5a_w3a_first => cause_of_death_isCV_f5a_w5a_first` (implication): rate 0.24684 over 158 rows
- `cause_of_death_isCV_f5a_w3a_first => cause_of_death_isCV_f5a_w1a_first` (implication): rate 0.15942 over 69 rows
- `cause_of_death_isCV_f5a_w1a_first => cause_of_death_isCV_f5a_w5a_first` (implication): rate 0.15873 over 63 rows

**synthetic[tvae_seed2]**
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w7d_first` (flag_days_consistency): rate 0.92 over 50 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w1mo_first` (flag_days_consistency): rate 0.46667 over 225 rows
- `cause_of_death_isCV_f5a_w3a_first => cause_of_death_isCV_f5a_w5a_first` (implication): rate 0.31522 over 92 rows
- `cause_of_death_isCV_f5a_w1a_first => cause_of_death_isCV_f5a_w5a_first` (implication): rate 0.16667 over 36 rows
- `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first / cause_of_death_isAllCause_f5a_w3a_first` (flag_days_consistency): rate 0.15845 over 2272 rows

