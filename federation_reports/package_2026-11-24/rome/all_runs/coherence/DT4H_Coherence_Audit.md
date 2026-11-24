# Row-Coherence Audit

138 rules ({'implication': 134, 'category_range': 4}) mined/learned from the TRAIN split and validated on real data. The holdout row is the fair baseline: real, unseen patients violating the same rules. A synthetic dataset far above it produces rows that are individually implausible patients even when every column's distribution is correct.

Two measures, answering different questions: the violation RATE is per applicable rule-check, while the row SHARE is the fraction of patients carrying at least one violation -- the one a release decision turns on, read against the real holdout's own share.

The 'consequent Missing' column is the evasion check for implication rules: the share of antecedent-true checks whose consequent was Missing and thus undecidable. A generator can push its violation rate toward zero by emitting Missing consequents; a share far above the real frames' reveals exactly that.

| frame | applicable checks | violations | violation rate | rules violated | consequent Missing | rows with >=1 violation |
|---|---|---|---|---|---|---|
| train (real) | 54820 | 0 | 0.0 | 0/138 | 63.0% | 0.0% (0/3773) |
| holdout (real, unseen) | 17511 | 0 | 0.0 | 0/138 | 63.8% | 0.0% (0/1258) |
| synthetic[aim40_eps1_seed0] | 13788 | 35 | 0.00254 | 1/138 | 36.6% | 0.9% (35/3773) |
| synthetic[aim40_eps5_seed0] | 12824 | 4 | 0.00031 | 1/138 | 40.5% | 0.1% (4/3773) |
| synthetic[aim50_eps1_seed0] | 13422 | 19 | 0.00142 | 2/138 | 41.6% | 0.5% (19/3773) |
| synthetic[ctgan_qt_seed0] | 75371 | 11771 | 0.15617 | 79/138 | 53.3% | 68.2% (2573/3773) |
| synthetic[ctgan_seed0] | 65901 | 12569 | 0.19073 | 79/138 | 57.5% | 77.5% (2924/3773) |
| synthetic[ctgan_seed1] | 63719 | 10561 | 0.16574 | 79/138 | 61.0% | 68.8% (2596/3773) |
| synthetic[ctgan_seed2] | 57964 | 10623 | 0.18327 | 79/138 | 65.8% | 69.7% (2630/3773) |
| synthetic[ddpm_g_seed0] | 13482 | 97 | 0.00719 | 2/138 | 99.3% | 2.6% (97/3773) |
| synthetic[ddpm_seed0] | 50892 | 764 | 0.01501 | 70/138 | 64.3% | 10.6% (399/3773) |
| synthetic[ddpm_seed1] | 47258 | 529 | 0.01119 | 66/138 | 65.9% | 9.8% (368/3773) |
| synthetic[ddpm_seed2] | 50717 | 671 | 0.01323 | 62/138 | 63.9% | 9.1% (343/3773) |
| synthetic[dpctgan_eps10_seed0] | 84180 | 19174 | 0.22777 | 19/138 | 48.9% | 100.0% (3773/3773) |
| synthetic[dpctgan_eps15_seed0] | 46961 | 1293 | 0.02753 | 16/138 | 66.1% | 31.4% (1186/3773) |
| synthetic[dpctgan_eps15_seed1] | 45324 | 9510 | 0.20982 | 13/138 | 58.0% | 100.0% (3773/3773) |
| synthetic[dpctgan_eps15_seed2] | 22619 | 18346 | 0.81109 | 19/138 | 93.1% | 100.0% (3773/3773) |
| synthetic[dpctgan_eps1_seed0] | 30545 | 10125 | 0.33148 | 18/138 | 84.5% | 100.0% (3773/3773) |
| synthetic[dpctgan_eps20_seed0] | 14391 | 7138 | 0.496 | 9/138 | 97.1% | 100.0% (3773/3773) |
| synthetic[dpctgan_eps5_seed0] | 11228 | 7313 | 0.65132 | 16/138 | 96.6% | 100.0% (3773/3773) |
| synthetic[dpctgan_eps8_seed0] | 25081 | 10949 | 0.43655 | 26/138 | 86.5% | 98.1% (3701/3773) |
| synthetic[gaussian_copula_seed0] | 41043 | 6498 | 0.15832 | 79/138 | 73.3% | 58.2% (2197/3773) |
| synthetic[gaussian_copula_seed1] | 40701 | 6684 | 0.16422 | 79/138 | 73.9% | 60.0% (2265/3773) |
| synthetic[gaussian_copula_seed2] | 40433 | 6419 | 0.15876 | 79/138 | 74.1% | 59.2% (2233/3773) |
| synthetic[mst_eps0p5_seed0] | 54429 | 6386 | 0.11733 | 55/138 | 62.8% | 46.1% (1739/3773) |
| synthetic[mst_eps10_seed0] | 56770 | 732 | 0.01289 | 63/138 | 61.1% | 3.5% (133/3773) |
| synthetic[mst_eps15_seed0] | 57394 | 596 | 0.01038 | 74/138 | 60.6% | 3.0% (112/3773) |
| synthetic[mst_eps15_seed1] | 57251 | 582 | 0.01017 | 65/138 | 60.7% | 3.2% (122/3773) |
| synthetic[mst_eps15_seed2] | 56887 | 430 | 0.00756 | 67/138 | 61.1% | 2.8% (107/3773) |
| synthetic[mst_eps1_seed0] | 58970 | 3429 | 0.05815 | 40/138 | 58.4% | 17.9% (675/3773) |
| synthetic[mst_eps20_seed0] | 57158 | 505 | 0.00884 | 68/138 | 60.9% | 3.0% (112/3773) |
| synthetic[mst_eps5_seed0] | 57270 | 1008 | 0.0176 | 47/138 | 60.8% | 4.7% (177/3773) |
| synthetic[mst_eps8_seed0] | 57361 | 759 | 0.01323 | 65/138 | 60.7% | 3.9% (148/3773) |
| synthetic[patectgan_eps15_seed0] | 45353 | 1648 | 0.03634 | 40/138 | 75.2% | 36.0% (1360/3773) |
| synthetic[patectgan_eps1_seed0] | 64916 | 15030 | 0.23153 | 79/138 | 65.2% | 95.3% (3595/3773) |
| synthetic[patectgan_eps5_seed0] | 24089 | 1302 | 0.05405 | 35/138 | 91.4% | 28.0% (1056/3773) |
| synthetic[tvae_cap256_seed0] | 54268 | 507 | 0.00934 | 60/138 | 65.0% | 4.8% (182/3773) |
| synthetic[tvae_ep1000_seed0] | 51414 | 595 | 0.01157 | 61/138 | 66.0% | 4.5% (171/3773) |
| synthetic[tvae_ind_seed0] | 52394 | 828 | 0.0158 | 69/138 | 65.8% | 5.5% (207/3773) |
| synthetic[tvae_qt_seed0] | 53206 | 440 | 0.00827 | 55/138 | 65.1% | 4.5% (171/3773) |
| synthetic[tvae_qt_seed1] | 52385 | 355 | 0.00678 | 46/138 | 66.0% | 4.0% (151/3773) |
| synthetic[tvae_qt_seed2] | 51708 | 403 | 0.00779 | 45/138 | 66.2% | 4.8% (182/3773) |
| synthetic[tvae_seed0] | 51143 | 572 | 0.01118 | 61/138 | 66.3% | 4.1% (153/3773) |
| synthetic[tvae_seed1] | 50299 | 589 | 0.01171 | 59/138 | 67.2% | 4.4% (167/3773) |
| synthetic[tvae_seed2] | 48370 | 301 | 0.00622 | 60/138 | 68.0% | 3.1% (116/3773) |

## Worst rules per synthetic dataset

**synthetic[aim40_eps1_seed0]**
- `smoking_status_formerSmoker_last => smoking_status_smoker_last` (implication): rate 0.0203 over 1724 rows

**synthetic[aim40_eps5_seed0]**
- `smoking_status_formerSmoker_last => smoking_status_smoker_last` (implication): rate 0.00235 over 1701 rows

**synthetic[aim50_eps1_seed0]**
- `conditions_mi => conditions_ihd` (implication): rate 0.00911 over 1976 rows
- `smoking_status_formerSmoker_last => smoking_status_smoker_last` (implication): rate 0.00059 over 1685 rows

**synthetic[ctgan_qt_seed0]**
- `smoking_status_formerSmoker_last => smoking_status_smoker_last` (implication): rate 0.61507 over 1686 rows
- `encounter_primary_reason_CV_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w1a_first` (implication): rate 0.51488 over 336 rows
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => encounter_primary_reason_CV_Disease_f5a_w1a_first` (implication): rate 0.51261 over 119 rows
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => encounter_primary_reason_CV_Disease_f5a_w5a_first` (implication): rate 0.48438 over 128 rows
- `encounter_primary_reason_CV_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w5a_first` (implication): rate 0.47564 over 349 rows

**synthetic[ctgan_seed0]**
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => encounter_primary_reason_CV_Disease_f5a_w1a_first` (implication): rate 0.72174 over 230 rows
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => encounter_primary_reason_CV_Disease_f5a_w5a_first` (implication): rate 0.676 over 250 rows
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => encounter_primary_reason_CV_Disease_f5a_w3mo_first` (implication): rate 0.6738 over 187 rows
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => encounter_primary_reason_CV_Disease_f5a_w6mo_first` (implication): rate 0.67188 over 192 rows
- `encounter_primary_reason_CV_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w1a_first` (implication): rate 0.65044 over 226 rows

**synthetic[ctgan_seed1]**
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => encounter_primary_reason_CV_Disease_f5a_w6mo_first` (implication): rate 0.63636 over 143 rows
- `conditions_ap => conditions_ihd` (implication): rate 0.62222 over 315 rows
- `conditions_mi => conditions_ihd` (implication): rate 0.61093 over 2470 rows
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => encounter_primary_reason_CV_Disease_f5a_w5a_first` (implication): rate 0.53125 over 160 rows
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => encounter_primary_reason_CV_Disease_f5a_w3mo_first` (implication): rate 0.52308 over 130 rows

**synthetic[ctgan_seed2]**
- `smoking_status_formerSmoker_last => smoking_status_smoker_last` (implication): rate 0.72042 over 862 rows
- `conditions_ap => conditions_ihd` (implication): rate 0.55479 over 146 rows
- `conditions_mi => conditions_ihd` (implication): rate 0.5464 over 1918 rows
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => encounter_primary_reason_CV_Disease_f5a_w1a_first` (implication): rate 0.51256 over 199 rows
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => encounter_primary_reason_CV_Disease_f5a_w6mo_first` (implication): rate 0.46951 over 164 rows

**synthetic[ddpm_g_seed0]**
- `hyperkalemia_severity_categorizedValue vs lab_results_potassium_value_last` (category_range): rate 0.02002 over 3347 rows
- `ckd_severity_from_calculated_egfr vs lab_results_valideGFR_value_last` (category_range): rate 0.00961 over 3121 rows

**synthetic[ddpm_seed0]**
- `conditions_ap => conditions_ihd` (implication): rate 0.24359 over 156 rows
- `encounter_primary_reason_CV_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w6mo_first` (implication): rate 0.17391 over 46 rows
- `encounter_primary_reason_CV_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w5a_first` (implication): rate 0.16667 over 48 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w5a_first => encounter_primary_reason_non_CV_Disease_f5a_w1mo_first` (implication): rate 0.16279 over 43 rows
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => encounter_primary_reason_CV_Disease_f5a_w6mo_first` (implication): rate 0.15385 over 13 rows

**synthetic[ddpm_seed1]**
- `conditions_ap => conditions_ihd` (implication): rate 0.2303 over 165 rows
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => encounter_primary_reason_CV_Disease_f5a_w1mo_first` (implication): rate 0.11111 over 18 rows
- `encounter_primary_reason_CV_Disease_f5a_w1a_first => encounter_primary_reason_CV_Disease_f5a_w1mo_first` (implication): rate 0.09434 over 53 rows
- `conditions_mi => conditions_ihd` (implication): rate 0.08513 over 2185 rows
- `encounter_primary_reason_CV_Disease_f5a_w6mo_first => encounter_primary_reason_CV_Disease_f5a_w1mo_first` (implication): rate 0.07843 over 51 rows

**synthetic[ddpm_seed2]**
- `conditions_ap => conditions_ihd` (implication): rate 0.26582 over 158 rows
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => encounter_primary_reason_CV_Disease_f5a_w3a_first` (implication): rate 0.1875 over 16 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w3a_first => encounter_primary_reason_non_CV_Disease_f5a_w1mo_first` (implication): rate 0.13953 over 43 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w5a_first => encounter_primary_reason_non_CV_Disease_f5a_w1mo_first` (implication): rate 0.13953 over 43 rows
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => encounter_primary_reason_CV_Disease_f5a_w6mo_first` (implication): rate 0.125 over 16 rows

**synthetic[dpctgan_eps10_seed0]**
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => encounter_primary_reason_CV_Disease_f5a_w3a_first` (implication): rate 1.0 over 4 rows
- `encounter_primary_reason_CV_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w3a_first` (implication): rate 1.0 over 2 rows
- `encounter_primary_reason_CV_Disease_f5a_w3mo_first => encounter_primary_reason_CV_Disease_f5a_w3a_first` (implication): rate 1.0 over 227 rows
- `encounter_primary_reason_CV_Disease_f5a_w5a_first => encounter_primary_reason_CV_Disease_f5a_w3a_first` (implication): rate 1.0 over 76 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w1mo_first => encounter_primary_reason_non_CV_Disease_f5a_w3a_first` (implication): rate 1.0 over 5 rows

**synthetic[dpctgan_eps15_seed0]**
- `encounter_primary_reason_CV_Disease_f5a_w1a_first => encounter_primary_reason_CV_Disease_f5a_w3a_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w3mo_first => encounter_primary_reason_non_CV_Disease_f5a_w1mo_first` (implication): rate 1.0 over 3 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w3mo_first => encounter_primary_reason_non_CV_Disease_f5a_w6mo_first` (implication): rate 1.0 over 4 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w3a_first => encounter_primary_reason_non_CV_Disease_f5a_w6mo_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_CV_Disease_f5a_w5a_first => encounter_primary_reason_CV_Disease_f5a_w3mo_first` (implication): rate 0.99301 over 143 rows

**synthetic[dpctgan_eps15_seed1]**
- `encounter_primary_reason_CV_Disease_f5a_w3mo_first => encounter_primary_reason_CV_Disease_f5a_w1a_first` (implication): rate 1.0 over 2 rows
- `encounter_primary_reason_CV_Disease_f5a_w5a_first => encounter_primary_reason_CV_Disease_f5a_w1a_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w1mo_first => encounter_primary_reason_non_CV_Disease_f5a_w5a_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w3mo_first => encounter_primary_reason_non_CV_Disease_f5a_w1mo_first` (implication): rate 1.0 over 2 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w3mo_first => encounter_primary_reason_non_CV_Disease_f5a_w5a_first` (implication): rate 1.0 over 2 rows

**synthetic[dpctgan_eps15_seed2]**
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => encounter_primary_reason_CV_Disease_f5a_w3mo_first` (implication): rate 1.0 over 2 rows
- `encounter_primary_reason_CV_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w3mo_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_CV_Disease_f5a_w6mo_first => encounter_primary_reason_CV_Disease_f5a_w3mo_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_CV_Disease_f5a_w5a_first => encounter_primary_reason_CV_Disease_f5a_w3mo_first` (implication): rate 1.0 over 3 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w1a_first => encounter_primary_reason_non_CV_Disease_f5a_w3a_first` (implication): rate 1.0 over 1 rows

**synthetic[dpctgan_eps1_seed0]**
- `encounter_primary_reason_non_CV_Disease_f5a_w6mo_first => encounter_primary_reason_non_CV_Disease_f5a_w3mo_first` (implication): rate 1.0 over 31 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w1a_first => encounter_primary_reason_non_CV_Disease_f5a_w3mo_first` (implication): rate 1.0 over 8 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w3a_first => encounter_primary_reason_non_CV_Disease_f5a_w1mo_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w3a_first => encounter_primary_reason_non_CV_Disease_f5a_w3mo_first` (implication): rate 1.0 over 149 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w5a_first => encounter_primary_reason_non_CV_Disease_f5a_w3mo_first` (implication): rate 1.0 over 2 rows

**synthetic[dpctgan_eps20_seed0]**
- `encounter_primary_reason_non_CV_Disease_f5a_w3a_first => encounter_primary_reason_non_CV_Disease_f5a_w6mo_first` (implication): rate 1.0 over 2 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w5a_first => encounter_primary_reason_non_CV_Disease_f5a_w1a_first` (implication): rate 1.0 over 3 rows
- `ckd_severity_from_calculated_egfr vs lab_results_valideGFR_value_first` (category_range): rate 1.0 over 3773 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w3a_first => encounter_primary_reason_non_CV_Disease_f5a_w1a_first` (implication): rate 0.89277 over 3749 rows
- `encounter_primary_reason_CV_Disease_f5a_w3a_first => encounter_primary_reason_CV_Disease_f5a_w5a_first` (implication): rate 0.66667 over 3 rows

**synthetic[dpctgan_eps5_seed0]**
- `encounter_primary_reason_CV_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w1a_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_CV_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w5a_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_CV_Disease_f5a_w1a_first => encounter_primary_reason_CV_Disease_f5a_w3mo_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_CV_Disease_f5a_w3a_first => encounter_primary_reason_CV_Disease_f5a_w5a_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w3mo_first => encounter_primary_reason_non_CV_Disease_f5a_w1mo_first` (implication): rate 1.0 over 1 rows

**synthetic[dpctgan_eps8_seed0]**
- `encounter_primary_reason_CV_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w3a_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_CV_Disease_f5a_w3mo_first => encounter_primary_reason_CV_Disease_f5a_w3a_first` (implication): rate 1.0 over 30 rows
- `encounter_primary_reason_CV_Disease_f5a_w6mo_first => encounter_primary_reason_CV_Disease_f5a_w3a_first` (implication): rate 1.0 over 64 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w1mo_first => encounter_primary_reason_non_CV_Disease_f5a_w1a_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w1mo_first => encounter_primary_reason_non_CV_Disease_f5a_w5a_first` (implication): rate 1.0 over 1 rows

**synthetic[gaussian_copula_seed0]**
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => encounter_primary_reason_CV_Disease_f5a_w1mo_first` (implication): rate 0.75 over 4 rows
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => encounter_primary_reason_CV_Disease_f5a_w1a_first` (implication): rate 0.72727 over 11 rows
- `encounter_primary_reason_CV_Disease_f5a_w3mo_first => encounter_primary_reason_CV_Disease_f5a_w1a_first` (implication): rate 0.67626 over 139 rows
- `encounter_primary_reason_CV_Disease_f5a_w3mo_first => encounter_primary_reason_CV_Disease_f5a_w1mo_first` (implication): rate 0.675 over 40 rows
- `encounter_primary_reason_CV_Disease_f5a_w3mo_first => encounter_primary_reason_CV_Disease_f5a_w6mo_first` (implication): rate 0.66929 over 127 rows

**synthetic[gaussian_copula_seed1]**
- `encounter_primary_reason_CV_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w6mo_first` (implication): rate 0.69767 over 43 rows
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => encounter_primary_reason_CV_Disease_f5a_w1mo_first` (implication): rate 0.66667 over 6 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w1a_first => encounter_primary_reason_non_CV_Disease_f5a_w5a_first` (implication): rate 0.65205 over 342 rows
- `encounter_primary_reason_CV_Disease_f5a_w3mo_first => encounter_primary_reason_CV_Disease_f5a_w6mo_first` (implication): rate 0.65152 over 132 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w6mo_first => encounter_primary_reason_non_CV_Disease_f5a_w3mo_first` (implication): rate 0.63704 over 135 rows

**synthetic[gaussian_copula_seed2]**
- `encounter_primary_reason_CV_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w1a_first` (implication): rate 0.68627 over 51 rows
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => encounter_primary_reason_CV_Disease_f5a_w3mo_first` (implication): rate 0.63636 over 11 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w6mo_first => encounter_primary_reason_non_CV_Disease_f5a_w1mo_first` (implication): rate 0.62264 over 53 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w1a_first => encounter_primary_reason_non_CV_Disease_f5a_w3a_first` (implication): rate 0.62258 over 310 rows
- `encounter_primary_reason_CV_Disease_f5a_w3a_first => encounter_primary_reason_CV_Disease_f5a_w1mo_first` (implication): rate 0.62162 over 74 rows

**synthetic[mst_eps0p5_seed0]**
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => encounter_primary_reason_CV_Disease_f5a_w3mo_first` (implication): rate 1.0 over 31 rows
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => encounter_primary_reason_CV_Disease_f5a_w1a_first` (implication): rate 1.0 over 31 rows
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => encounter_primary_reason_CV_Disease_f5a_w3a_first` (implication): rate 1.0 over 31 rows
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => encounter_primary_reason_CV_Disease_f5a_w5a_first` (implication): rate 1.0 over 31 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w1mo_first => encounter_primary_reason_non_CV_Disease_f5a_w3a_first` (implication): rate 1.0 over 167 rows

**synthetic[mst_eps10_seed0]**
- `encounter_primary_reason_CV_Disease_f5a_w1a_first => encounter_primary_reason_CV_Disease_f5a_w7d_first` (implication): rate 0.32353 over 68 rows
- `encounter_primary_reason_CV_Disease_f5a_w5a_first => encounter_primary_reason_CV_Disease_f5a_w7d_first` (implication): rate 0.29825 over 57 rows
- `encounter_primary_reason_CV_Disease_f5a_w1a_first => encounter_primary_reason_CV_Disease_f5a_w1mo_first` (implication): rate 0.25974 over 154 rows
- `encounter_primary_reason_CV_Disease_f5a_w5a_first => encounter_primary_reason_CV_Disease_f5a_w1mo_first` (implication): rate 0.24476 over 143 rows
- `encounter_primary_reason_CV_Disease_f5a_w3a_first => encounter_primary_reason_CV_Disease_f5a_w7d_first` (implication): rate 0.23729 over 59 rows

**synthetic[mst_eps15_seed0]**
- `encounter_primary_reason_CV_Disease_f5a_w3a_first => encounter_primary_reason_CV_Disease_f5a_w7d_first` (implication): rate 0.34694 over 49 rows
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => encounter_primary_reason_CV_Disease_f5a_w3a_first` (implication): rate 0.25581 over 43 rows
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => encounter_primary_reason_CV_Disease_f5a_w1mo_first` (implication): rate 0.23529 over 51 rows
- `encounter_primary_reason_CV_Disease_f5a_w3mo_first => encounter_primary_reason_CV_Disease_f5a_w7d_first` (implication): rate 0.23077 over 52 rows
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => encounter_primary_reason_CV_Disease_f5a_w6mo_first` (implication): rate 0.22449 over 49 rows

**synthetic[mst_eps15_seed1]**
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => encounter_primary_reason_CV_Disease_f5a_w5a_first` (implication): rate 0.41176 over 34 rows
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => encounter_primary_reason_CV_Disease_f5a_w3a_first` (implication): rate 0.39583 over 48 rows
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => encounter_primary_reason_CV_Disease_f5a_w6mo_first` (implication): rate 0.33333 over 48 rows
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => encounter_primary_reason_CV_Disease_f5a_w1a_first` (implication): rate 0.33333 over 51 rows
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => encounter_primary_reason_CV_Disease_f5a_w3mo_first` (implication): rate 0.22449 over 49 rows

**synthetic[mst_eps15_seed2]**
- `encounter_primary_reason_CV_Disease_f5a_w5a_first => encounter_primary_reason_CV_Disease_f5a_w7d_first` (implication): rate 0.17391 over 46 rows
- `encounter_primary_reason_CV_Disease_f5a_w3a_first => encounter_primary_reason_CV_Disease_f5a_w7d_first` (implication): rate 0.16327 over 49 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w1mo_first => encounter_primary_reason_non_CV_Disease_f5a_w5a_first` (implication): rate 0.11644 over 146 rows
- `encounter_primary_reason_CV_Disease_f5a_w3mo_first => encounter_primary_reason_CV_Disease_f5a_w7d_first` (implication): rate 0.1087 over 46 rows
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => encounter_primary_reason_CV_Disease_f5a_w3mo_first` (implication): rate 0.08889 over 45 rows

**synthetic[mst_eps1_seed0]**
- `encounter_primary_reason_non_CV_Disease_f5a_w1mo_first => encounter_primary_reason_non_CV_Disease_f5a_w6mo_first` (implication): rate 1.0 over 150 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w1mo_first => encounter_primary_reason_non_CV_Disease_f5a_w5a_first` (implication): rate 1.0 over 150 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w1mo_first => encounter_primary_reason_non_CV_Disease_f5a_w3mo_first` (implication): rate 0.99333 over 150 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w3mo_first => encounter_primary_reason_non_CV_Disease_f5a_w1mo_first` (implication): rate 0.92308 over 13 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w1mo_first => encounter_primary_reason_non_CV_Disease_f5a_w1a_first` (implication): rate 0.75333 over 150 rows

**synthetic[mst_eps20_seed0]**
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => encounter_primary_reason_CV_Disease_f5a_w1mo_first` (implication): rate 0.57143 over 42 rows
- `encounter_primary_reason_CV_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w7d_first` (implication): rate 0.55 over 40 rows
- `encounter_primary_reason_CV_Disease_f5a_w5a_first => encounter_primary_reason_CV_Disease_f5a_w7d_first` (implication): rate 0.37143 over 35 rows
- `encounter_primary_reason_CV_Disease_f5a_w3mo_first => encounter_primary_reason_CV_Disease_f5a_w7d_first` (implication): rate 0.36364 over 44 rows
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => encounter_primary_reason_CV_Disease_f5a_w6mo_first` (implication): rate 0.35714 over 42 rows

**synthetic[mst_eps5_seed0]**
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => encounter_primary_reason_CV_Disease_f5a_w3a_first` (implication): rate 0.62222 over 45 rows
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => encounter_primary_reason_CV_Disease_f5a_w5a_first` (implication): rate 0.52273 over 44 rows
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => encounter_primary_reason_CV_Disease_f5a_w6mo_first` (implication): rate 0.40816 over 49 rows
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => encounter_primary_reason_CV_Disease_f5a_w1a_first` (implication): rate 0.38298 over 47 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w5a_first => encounter_primary_reason_non_CV_Disease_f5a_w1mo_first` (implication): rate 0.25397 over 189 rows

**synthetic[mst_eps8_seed0]**
- `encounter_primary_reason_CV_Disease_f5a_w5a_first => encounter_primary_reason_CV_Disease_f5a_w7d_first` (implication): rate 0.34375 over 32 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w1mo_first => encounter_primary_reason_non_CV_Disease_f5a_w5a_first` (implication): rate 0.26712 over 146 rows
- `encounter_primary_reason_CV_Disease_f5a_w6mo_first => encounter_primary_reason_CV_Disease_f5a_w7d_first` (implication): rate 0.23333 over 60 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w1mo_first => encounter_primary_reason_non_CV_Disease_f5a_w3a_first` (implication): rate 0.23288 over 146 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w1mo_first => encounter_primary_reason_non_CV_Disease_f5a_w1a_first` (implication): rate 0.21622 over 148 rows

**synthetic[patectgan_eps15_seed0]**
- `encounter_primary_reason_CV_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w1a_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_CV_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w3a_first` (implication): rate 1.0 over 3 rows
- `encounter_primary_reason_CV_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w5a_first` (implication): rate 1.0 over 3 rows
- `encounter_primary_reason_CV_Disease_f5a_w3mo_first => encounter_primary_reason_CV_Disease_f5a_w1a_first` (implication): rate 1.0 over 5 rows
- `encounter_primary_reason_CV_Disease_f5a_w3mo_first => encounter_primary_reason_CV_Disease_f5a_w3a_first` (implication): rate 1.0 over 11 rows

**synthetic[patectgan_eps1_seed0]**
- `encounter_primary_reason_CV_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w5a_first` (implication): rate 0.68159 over 201 rows
- `encounter_primary_reason_CV_Disease_f5a_w3mo_first => encounter_primary_reason_CV_Disease_f5a_w5a_first` (implication): rate 0.66168 over 334 rows
- `encounter_primary_reason_CV_Disease_f5a_w1a_first => encounter_primary_reason_CV_Disease_f5a_w5a_first` (implication): rate 0.6327 over 471 rows
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => encounter_primary_reason_CV_Disease_f5a_w1a_first` (implication): rate 0.625 over 144 rows
- `encounter_primary_reason_CV_Disease_f5a_w3mo_first => encounter_primary_reason_CV_Disease_f5a_w1a_first` (implication): rate 0.6229 over 297 rows

**synthetic[patectgan_eps5_seed0]**
- `encounter_primary_reason_CV_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w6mo_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_CV_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w3a_first` (implication): rate 1.0 over 2 rows
- `encounter_primary_reason_CV_Disease_f5a_w3mo_first => encounter_primary_reason_CV_Disease_f5a_w3a_first` (implication): rate 1.0 over 2 rows
- `encounter_primary_reason_CV_Disease_f5a_w3mo_first => encounter_primary_reason_CV_Disease_f5a_w5a_first` (implication): rate 1.0 over 2 rows
- `encounter_primary_reason_CV_Disease_f5a_w6mo_first => encounter_primary_reason_CV_Disease_f5a_w1a_first` (implication): rate 1.0 over 1 rows

**synthetic[tvae_cap256_seed0]**
- `encounter_primary_reason_CV_Disease_f5a_w1a_first => encounter_primary_reason_CV_Disease_f5a_w7d_first` (implication): rate 0.13793 over 29 rows
- `encounter_primary_reason_CV_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w3mo_first` (implication): rate 0.12 over 100 rows
- `encounter_primary_reason_CV_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w3a_first` (implication): rate 0.12 over 100 rows
- `encounter_primary_reason_CV_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w5a_first` (implication): rate 0.12 over 100 rows
- `encounter_primary_reason_CV_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w6mo_first` (implication): rate 0.11 over 100 rows

**synthetic[tvae_ep1000_seed0]**
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => encounter_primary_reason_CV_Disease_f5a_w3a_first` (implication): rate 0.29412 over 17 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w1mo_first => encounter_primary_reason_non_CV_Disease_f5a_w1a_first` (implication): rate 0.26316 over 38 rows
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => encounter_primary_reason_CV_Disease_f5a_w5a_first` (implication): rate 0.23529 over 17 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w3a_first => encounter_primary_reason_non_CV_Disease_f5a_w1mo_first` (implication): rate 0.19565 over 46 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w3mo_first => encounter_primary_reason_non_CV_Disease_f5a_w1mo_first` (implication): rate 0.17778 over 45 rows

**synthetic[tvae_ind_seed0]**
- `encounter_primary_reason_CV_Disease_f5a_w6mo_first => encounter_primary_reason_CV_Disease_f5a_w7d_first` (implication): rate 0.2963 over 27 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w1mo_first => encounter_primary_reason_non_CV_Disease_f5a_w1a_first` (implication): rate 0.24286 over 70 rows
- `encounter_primary_reason_CV_Disease_f5a_w1a_first => encounter_primary_reason_CV_Disease_f5a_w7d_first` (implication): rate 0.24 over 25 rows
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => encounter_primary_reason_CV_Disease_f5a_w3a_first` (implication): rate 0.21053 over 19 rows
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => encounter_primary_reason_CV_Disease_f5a_w5a_first` (implication): rate 0.21053 over 19 rows

**synthetic[tvae_qt_seed0]**
- `encounter_primary_reason_non_CV_Disease_f5a_w3mo_first => encounter_primary_reason_non_CV_Disease_f5a_w1mo_first` (implication): rate 0.10811 over 37 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w6mo_first => encounter_primary_reason_non_CV_Disease_f5a_w1mo_first` (implication): rate 0.10811 over 37 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w1a_first => encounter_primary_reason_non_CV_Disease_f5a_w1mo_first` (implication): rate 0.10811 over 37 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w5a_first => encounter_primary_reason_non_CV_Disease_f5a_w1mo_first` (implication): rate 0.10811 over 37 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w3a_first => encounter_primary_reason_non_CV_Disease_f5a_w1mo_first` (implication): rate 0.08333 over 36 rows

**synthetic[tvae_qt_seed1]**
- `encounter_primary_reason_CV_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w3a_first` (implication): rate 0.13333 over 15 rows
- `encounter_primary_reason_CV_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w5a_first` (implication): rate 0.06667 over 15 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w3a_first => encounter_primary_reason_non_CV_Disease_f5a_w1mo_first` (implication): rate 0.06667 over 30 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w5a_first => encounter_primary_reason_non_CV_Disease_f5a_w1mo_first` (implication): rate 0.06667 over 30 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w5a_first => encounter_primary_reason_non_CV_Disease_f5a_w6mo_first` (implication): rate 0.04368 over 435 rows

**synthetic[tvae_qt_seed2]**
- `encounter_primary_reason_CV_Disease_f5a_w3a_first => encounter_primary_reason_CV_Disease_f5a_w1a_first` (implication): rate 0.0574 over 453 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w3a_first => encounter_primary_reason_non_CV_Disease_f5a_w3mo_first` (implication): rate 0.05392 over 204 rows
- `encounter_primary_reason_CV_Disease_f5a_w5a_first => encounter_primary_reason_CV_Disease_f5a_w1a_first` (implication): rate 0.05122 over 449 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w1a_first => encounter_primary_reason_non_CV_Disease_f5a_w3mo_first` (implication): rate 0.04926 over 203 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w5a_first => encounter_primary_reason_non_CV_Disease_f5a_w3mo_first` (implication): rate 0.04926 over 203 rows

**synthetic[tvae_seed0]**
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => encounter_primary_reason_CV_Disease_f5a_w3a_first` (implication): rate 0.23077 over 13 rows
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => encounter_primary_reason_CV_Disease_f5a_w5a_first` (implication): rate 0.23077 over 13 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w1mo_first => encounter_primary_reason_non_CV_Disease_f5a_w1a_first` (implication): rate 0.23077 over 39 rows
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => encounter_primary_reason_CV_Disease_f5a_w3mo_first` (implication): rate 0.15385 over 13 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w3mo_first => encounter_primary_reason_non_CV_Disease_f5a_w1mo_first` (implication): rate 0.13953 over 43 rows

**synthetic[tvae_seed1]**
- `encounter_primary_reason_CV_Disease_f5a_w1a_first => encounter_primary_reason_CV_Disease_f5a_w7d_first` (implication): rate 0.18519 over 27 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w3a_first => encounter_primary_reason_non_CV_Disease_f5a_w1mo_first` (implication): rate 0.14286 over 91 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w1mo_first => encounter_primary_reason_non_CV_Disease_f5a_w1a_first` (implication): rate 0.14103 over 78 rows
- `encounter_primary_reason_non_CV_Disease_f5a_w5a_first => encounter_primary_reason_non_CV_Disease_f5a_w1mo_first` (implication): rate 0.13483 over 89 rows
- `encounter_primary_reason_CV_Disease_f5a_w1mo_first => encounter_primary_reason_CV_Disease_f5a_w3a_first` (implication): rate 0.12088 over 91 rows

**synthetic[tvae_seed2]**
- `encounter_primary_reason_CV_Disease_f5a_w7d_first => encounter_primary_reason_CV_Disease_f5a_w3a_first` (implication): rate 0.09091 over 11 rows
- `encounter_primary_reason_CV_Disease_f5a_w6mo_first => encounter_primary_reason_CV_Disease_f5a_w1mo_first` (implication): rate 0.08333 over 60 rows
- `encounter_primary_reason_CV_Disease_f5a_w3a_first => encounter_primary_reason_CV_Disease_f5a_w1mo_first` (implication): rate 0.06897 over 58 rows
- `encounter_primary_reason_CV_Disease_f5a_w3mo_first => encounter_primary_reason_CV_Disease_f5a_w1mo_first` (implication): rate 0.05172 over 58 rows
- `encounter_primary_reason_CV_Disease_f5a_w1a_first => encounter_primary_reason_CV_Disease_f5a_w1mo_first` (implication): rate 0.05172 over 58 rows

