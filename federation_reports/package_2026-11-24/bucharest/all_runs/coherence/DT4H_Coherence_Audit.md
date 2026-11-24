# Row-Coherence Audit

10 rules ({'implication': 8, 'category_range': 2}) mined/learned from the TRAIN split and validated on real data. The holdout row is the fair baseline: real, unseen patients violating the same rules. A synthetic dataset far above it produces rows that are individually implausible patients even when every column's distribution is correct.

Two measures, answering different questions: the violation RATE is per applicable rule-check, while the row SHARE is the fraction of patients carrying at least one violation -- the one a release decision turns on, read against the real holdout's own share.

The 'consequent Missing' column is the evasion check for implication rules: the share of antecedent-true checks whose consequent was Missing and thus undecidable. A generator can push its violation rate toward zero by emitting Missing consequents; a share far above the real frames' reveals exactly that.

| frame | applicable checks | violations | violation rate | rules violated | consequent Missing | rows with >=1 violation |
|---|---|---|---|---|---|---|
| train (real) | 11444 | 0 | 0.0 | 0/10 | 1.3% | 0.0% (0/4706) |
| holdout (real, unseen) | 3836 | 0 | 0.0 | 0/10 | 2.5% | 0.0% (0/1568) |
| synthetic[ctgan_qt_seed0] | 13132 | 1762 | 0.13418 | 10/10 | 12.3% | 26.5% (1246/4706) |
| synthetic[ctgan_seed0] | 13509 | 1702 | 0.12599 | 10/10 | 7.1% | 22.9% (1076/4706) |
| synthetic[ctgan_seed1] | 13500 | 1829 | 0.13548 | 10/10 | 5.6% | 24.6% (1159/4706) |
| synthetic[ctgan_seed2] | 13667 | 1610 | 0.1178 | 10/10 | 7.4% | 22.5% (1060/4706) |
| synthetic[ddpm_g_seed0] | 9308 | 37 | 0.00398 | 1/10 | n/a | 0.8% (37/4706) |
| synthetic[ddpm_seed0] | 10514 | 121 | 0.01151 | 9/10 | 1.0% | 2.4% (113/4706) |
| synthetic[ddpm_seed1] | 10520 | 148 | 0.01407 | 9/10 | 1.7% | 2.9% (136/4706) |
| synthetic[ddpm_seed2] | 10637 | 132 | 0.01241 | 9/10 | 0.8% | 2.6% (122/4706) |
| synthetic[dpctgan_eps10_seed0] | 12357 | 2254 | 0.18241 | 3/10 | 0.7% | 47.4% (2231/4706) |
| synthetic[dpctgan_eps15_seed0] | 8230 | 132 | 0.01604 | 2/10 | 0.0% | 2.8% (132/4706) |
| synthetic[dpctgan_eps15_seed1] | 8391 | 0 | 0.0 | 0/10 | 0.2% | 0.0% (0/4706) |
| synthetic[dpctgan_eps15_seed2] | 4855 | 9 | 0.00185 | 4/10 | 77.8% | 0.2% (9/4706) |
| synthetic[dpctgan_eps1_seed0] | 4705 | 2670 | 0.56748 | 2/10 | 85.7% | 56.7% (2667/4706) |
| synthetic[dpctgan_eps20_seed0] | 4897 | 1 | 0.0002 | 1/10 | 85.7% | 0.0% (1/4706) |
| synthetic[dpctgan_eps5_seed0] | 13426 | 1599 | 0.1191 | 2/10 | 1.2% | 34.0% (1598/4706) |
| synthetic[dpctgan_eps8_seed0] | 6462 | 162 | 0.02507 | 1/10 | 0.2% | 3.4% (162/4706) |
| synthetic[gaussian_copula_seed0] | 11137 | 673 | 0.06043 | 6/10 | 15.6% | 13.2% (623/4706) |
| synthetic[gaussian_copula_seed1] | 11191 | 733 | 0.0655 | 7/10 | 15.1% | 14.3% (674/4706) |
| synthetic[gaussian_copula_seed2] | 11203 | 655 | 0.05847 | 9/10 | 17.0% | 13.1% (618/4706) |
| synthetic[mst_eps15_seed0] | 11400 | 199 | 0.01746 | 9/10 | 1.6% | 2.4% (114/4706) |
| synthetic[patectgan_eps15_seed0] | 10934 | 178 | 0.01628 | 4/10 | 0.0% | 3.7% (173/4706) |
| synthetic[patectgan_eps1_seed0] | 9779 | 2092 | 0.21393 | 10/10 | 55.8% | 35.8% (1686/4706) |
| synthetic[patectgan_eps5_seed0] | 10491 | 332 | 0.03165 | 4/10 | 0.2% | 6.7% (315/4706) |
| synthetic[tvae_cap256_seed0] | 10996 | 42 | 0.00382 | 4/10 | 3.7% | 0.9% (42/4706) |
| synthetic[tvae_ep1000_seed0] | 11152 | 64 | 0.00574 | 5/10 | 1.9% | 1.3% (63/4706) |
| synthetic[tvae_ind_seed0] | 10729 | 18 | 0.00168 | 5/10 | 4.2% | 0.4% (17/4706) |
| synthetic[tvae_qt_seed0] | 10351 | 140 | 0.01353 | 8/10 | 6.7% | 2.9% (136/4706) |
| synthetic[tvae_qt_seed1] | 11911 | 104 | 0.00873 | 8/10 | 1.7% | 2.1% (100/4706) |
| synthetic[tvae_qt_seed2] | 10653 | 137 | 0.01286 | 6/10 | 4.0% | 2.9% (135/4706) |
| synthetic[tvae_seed0] | 10804 | 52 | 0.00481 | 5/10 | 3.0% | 1.1% (51/4706) |
| synthetic[tvae_seed1] | 11268 | 37 | 0.00328 | 3/10 | 3.8% | 0.8% (37/4706) |
| synthetic[tvae_seed2] | 10965 | 54 | 0.00492 | 5/10 | 3.2% | 1.1% (53/4706) |

## Worst rules per synthetic dataset

**synthetic[ctgan_qt_seed0]**
- `encounter_primary_reason_CV_Disease_f5a_w5a_first => encounter_primary_reason_CV_Disease_f5a_w3a_first` (implication): rate 0.36811 over 508 rows
- `encounter_primary_reason_CV_Disease_f5a_w1a_first => encounter_primary_reason_CV_Disease_f5a_w3a_first` (implication): rate 0.34454 over 476 rows
- `encounter_primary_reason_CV_Disease_f5a_w3a_first => encounter_primary_reason_CV_Disease_f5a_w5a_first` (implication): rate 0.24648 over 426 rows
- `conditions_ap => conditions_ihd` (implication): rate 0.24312 over 436 rows
- `encounter_primary_reason_CV_Disease_f5a_w5a_first => encounter_primary_reason_CV_Disease_f5a_w1a_first` (implication): rate 0.23 over 500 rows

**synthetic[ctgan_seed0]**
- `conditions_ap => conditions_ihd` (implication): rate 0.55987 over 309 rows
- `conditions_mi => conditions_ihd` (implication): rate 0.54464 over 448 rows
- `encounter_primary_reason_CV_Disease_f5a_w1a_first => encounter_primary_reason_CV_Disease_f5a_w3a_first` (implication): rate 0.43234 over 569 rows
- `encounter_primary_reason_CV_Disease_f5a_w5a_first => encounter_primary_reason_CV_Disease_f5a_w3a_first` (implication): rate 0.42632 over 699 rows
- `encounter_primary_reason_CV_Disease_f5a_w5a_first => encounter_primary_reason_CV_Disease_f5a_w1a_first` (implication): rate 0.28846 over 676 rows

**synthetic[ctgan_seed1]**
- `encounter_primary_reason_CV_Disease_f5a_w1a_first => encounter_primary_reason_CV_Disease_f5a_w3a_first` (implication): rate 0.67877 over 358 rows
- `encounter_primary_reason_CV_Disease_f5a_w5a_first => encounter_primary_reason_CV_Disease_f5a_w3a_first` (implication): rate 0.65714 over 420 rows
- `encounter_primary_reason_CV_Disease_f5a_w3a_first => encounter_primary_reason_CV_Disease_f5a_w1a_first` (implication): rate 0.48889 over 225 rows
- `encounter_primary_reason_CV_Disease_f5a_w5a_first => encounter_primary_reason_CV_Disease_f5a_w1a_first` (implication): rate 0.48086 over 418 rows
- `encounter_primary_reason_CV_Disease_f5a_w1a_first => encounter_primary_reason_CV_Disease_f5a_w5a_first` (implication): rate 0.41979 over 374 rows

**synthetic[ctgan_seed2]**
- `conditions_ap => conditions_ihd` (implication): rate 0.36803 over 269 rows
- `conditions_mi => conditions_ihd` (implication): rate 0.3557 over 790 rows
- `encounter_primary_reason_CV_Disease_f5a_w1a_first => encounter_primary_reason_CV_Disease_f5a_w5a_first` (implication): rate 0.34043 over 564 rows
- `encounter_primary_reason_CV_Disease_f5a_w3a_first => encounter_primary_reason_CV_Disease_f5a_w5a_first` (implication): rate 0.33059 over 608 rows
- `encounter_primary_reason_CV_Disease_f5a_w3a_first => encounter_primary_reason_CV_Disease_f5a_w1a_first` (implication): rate 0.25714 over 595 rows

**synthetic[ddpm_g_seed0]**
- `hyperkalemia_severity_categorizedValue vs lab_results_potassium_value_last` (category_range): rate 0.00795 over 4653 rows

**synthetic[ddpm_seed0]**
- `encounter_primary_reason_CV_Disease_f5a_w1a_first => encounter_primary_reason_CV_Disease_f5a_w3a_first` (implication): rate 0.17857 over 28 rows
- `encounter_primary_reason_CV_Disease_f5a_w1a_first => encounter_primary_reason_CV_Disease_f5a_w5a_first` (implication): rate 0.14286 over 28 rows
- `conditions_ap => conditions_ihd` (implication): rate 0.08015 over 524 rows
- `conditions_mi => conditions_ihd` (implication): rate 0.07692 over 507 rows
- `encounter_primary_reason_CV_Disease_f5a_w5a_first => encounter_primary_reason_CV_Disease_f5a_w3a_first` (implication): rate 0.06667 over 30 rows

**synthetic[ddpm_seed1]**
- `encounter_primary_reason_CV_Disease_f5a_w1a_first => encounter_primary_reason_CV_Disease_f5a_w5a_first` (implication): rate 0.18182 over 33 rows
- `encounter_primary_reason_CV_Disease_f5a_w1a_first => encounter_primary_reason_CV_Disease_f5a_w3a_first` (implication): rate 0.15152 over 33 rows
- `encounter_primary_reason_CV_Disease_f5a_w3a_first => encounter_primary_reason_CV_Disease_f5a_w1a_first` (implication): rate 0.15152 over 33 rows
- `encounter_primary_reason_CV_Disease_f5a_w5a_first => encounter_primary_reason_CV_Disease_f5a_w1a_first` (implication): rate 0.12903 over 31 rows
- `conditions_mi => conditions_ihd` (implication): rate 0.10536 over 541 rows

**synthetic[ddpm_seed2]**
- `encounter_primary_reason_CV_Disease_f5a_w1a_first => encounter_primary_reason_CV_Disease_f5a_w3a_first` (implication): rate 0.12195 over 41 rows
- `encounter_primary_reason_CV_Disease_f5a_w1a_first => encounter_primary_reason_CV_Disease_f5a_w5a_first` (implication): rate 0.09756 over 41 rows
- `conditions_ap => conditions_ihd` (implication): rate 0.09392 over 543 rows
- `conditions_mi => conditions_ihd` (implication): rate 0.07509 over 546 rows
- `encounter_primary_reason_CV_Disease_f5a_w5a_first => encounter_primary_reason_CV_Disease_f5a_w3a_first` (implication): rate 0.06977 over 43 rows

**synthetic[dpctgan_eps10_seed0]**
- `hyperkalemia_severity_categorizedValue vs lab_results_potassium_value_last` (category_range): rate 0.47626 over 4655 rows
- `conditions_mi => conditions_ihd` (implication): rate 0.00759 over 4086 rows
- `hyperkalemia_severity_categorizedValue vs lab_results_potassium_value_first` (category_range): rate 0.00166 over 3614 rows

**synthetic[dpctgan_eps15_seed0]**
- `hyperkalemia_severity_categorizedValue vs lab_results_potassium_value_last` (category_range): rate 0.0321 over 3988 rows
- `conditions_ap => conditions_ihd` (implication): rate 0.00096 over 4181 rows

**synthetic[dpctgan_eps15_seed2]**
- `conditions_ap => conditions_ihd` (implication): rate 1.0 over 2 rows
- `conditions_mi => conditions_ihd` (implication): rate 1.0 over 2 rows
- `hyperkalemia_severity_categorizedValue vs lab_results_potassium_value_last` (category_range): rate 0.0033 over 1211 rows
- `hyperkalemia_severity_categorizedValue vs lab_results_potassium_value_first` (category_range): rate 0.00027 over 3640 rows

**synthetic[dpctgan_eps1_seed0]**
- `conditions_mi => conditions_ihd` (implication): rate 0.66667 over 6 rows
- `hyperkalemia_severity_categorizedValue vs lab_results_potassium_value_last` (category_range): rate 0.57149 over 4665 rows

**synthetic[dpctgan_eps20_seed0]**
- `conditions_mi => conditions_ihd` (implication): rate 1.0 over 1 rows

**synthetic[dpctgan_eps5_seed0]**
- `hyperkalemia_severity_categorizedValue vs lab_results_potassium_value_last` (category_range): rate 0.34645 over 4601 rows
- `conditions_ap => conditions_ihd` (implication): rate 0.00107 over 4679 rows

**synthetic[dpctgan_eps8_seed0]**
- `conditions_ap => conditions_ihd` (implication): rate 0.0922 over 1757 rows

**synthetic[gaussian_copula_seed0]**
- `encounter_primary_reason_CV_Disease_f5a_w1a_first => encounter_primary_reason_CV_Disease_f5a_w5a_first` (implication): rate 0.5 over 4 rows
- `encounter_primary_reason_CV_Disease_f5a_w5a_first => encounter_primary_reason_CV_Disease_f5a_w1a_first` (implication): rate 0.5 over 4 rows
- `conditions_mi => conditions_ihd` (implication): rate 0.32717 over 920 rows
- `conditions_ap => conditions_ihd` (implication): rate 0.32423 over 842 rows
- `hyperkalemia_severity_categorizedValue vs lab_results_potassium_value_last` (category_range): rate 0.01944 over 4681 rows

**synthetic[gaussian_copula_seed1]**
- `encounter_primary_reason_CV_Disease_f5a_w1a_first => encounter_primary_reason_CV_Disease_f5a_w3a_first` (implication): rate 1.0 over 1 rows
- `encounter_primary_reason_CV_Disease_f5a_w1a_first => encounter_primary_reason_CV_Disease_f5a_w5a_first` (implication): rate 0.5 over 4 rows
- `conditions_mi => conditions_ihd` (implication): rate 0.33743 over 895 rows
- `encounter_primary_reason_CV_Disease_f5a_w5a_first => encounter_primary_reason_CV_Disease_f5a_w3a_first` (implication): rate 0.33333 over 3 rows
- `conditions_ap => conditions_ihd` (implication): rate 0.32868 over 931 rows

**synthetic[gaussian_copula_seed2]**
- `encounter_primary_reason_CV_Disease_f5a_w1a_first => encounter_primary_reason_CV_Disease_f5a_w5a_first` (implication): rate 0.66667 over 3 rows
- `encounter_primary_reason_CV_Disease_f5a_w3a_first => encounter_primary_reason_CV_Disease_f5a_w1a_first` (implication): rate 0.5 over 2 rows
- `encounter_primary_reason_CV_Disease_f5a_w3a_first => encounter_primary_reason_CV_Disease_f5a_w5a_first` (implication): rate 0.5 over 2 rows
- `encounter_primary_reason_CV_Disease_f5a_w5a_first => encounter_primary_reason_CV_Disease_f5a_w1a_first` (implication): rate 0.5 over 2 rows
- `encounter_primary_reason_CV_Disease_f5a_w5a_first => encounter_primary_reason_CV_Disease_f5a_w3a_first` (implication): rate 0.5 over 2 rows

**synthetic[mst_eps15_seed0]**
- `encounter_primary_reason_CV_Disease_f5a_w1a_first => encounter_primary_reason_CV_Disease_f5a_w5a_first` (implication): rate 0.85106 over 47 rows
- `encounter_primary_reason_CV_Disease_f5a_w5a_first => encounter_primary_reason_CV_Disease_f5a_w1a_first` (implication): rate 0.84783 over 46 rows
- `encounter_primary_reason_CV_Disease_f5a_w3a_first => encounter_primary_reason_CV_Disease_f5a_w5a_first` (implication): rate 0.64516 over 62 rows
- `encounter_primary_reason_CV_Disease_f5a_w5a_first => encounter_primary_reason_CV_Disease_f5a_w3a_first` (implication): rate 0.63333 over 60 rows
- `encounter_primary_reason_CV_Disease_f5a_w3a_first => encounter_primary_reason_CV_Disease_f5a_w1a_first` (implication): rate 0.28788 over 66 rows

**synthetic[patectgan_eps15_seed0]**
- `conditions_mi => conditions_ihd` (implication): rate 0.07123 over 730 rows
- `conditions_ap => conditions_ihd` (implication): rate 0.07053 over 794 rows
- `hyperkalemia_severity_categorizedValue vs lab_results_potassium_value_last` (category_range): rate 0.01232 over 4706 rows
- `hyperkalemia_severity_categorizedValue vs lab_results_potassium_value_first` (category_range): rate 0.00255 over 4704 rows

**synthetic[patectgan_eps1_seed0]**
- `encounter_primary_reason_CV_Disease_f5a_w3a_first => encounter_primary_reason_CV_Disease_f5a_w1a_first` (implication): rate 0.60645 over 155 rows
- `encounter_primary_reason_CV_Disease_f5a_w5a_first => encounter_primary_reason_CV_Disease_f5a_w1a_first` (implication): rate 0.56774 over 155 rows
- `conditions_ap => conditions_ihd` (implication): rate 0.49178 over 791 rows
- `conditions_mi => conditions_ihd` (implication): rate 0.47425 over 563 rows
- `encounter_primary_reason_CV_Disease_f5a_w1a_first => encounter_primary_reason_CV_Disease_f5a_w3a_first` (implication): rate 0.46491 over 114 rows

**synthetic[patectgan_eps5_seed0]**
- `conditions_mi => conditions_ihd` (implication): rate 0.26449 over 552 rows
- `conditions_ap => conditions_ihd` (implication): rate 0.23452 over 533 rows
- `hyperkalemia_severity_categorizedValue vs lab_results_potassium_value_last` (category_range): rate 0.01212 over 4703 rows
- `hyperkalemia_severity_categorizedValue vs lab_results_potassium_value_first` (category_range): rate 0.00085 over 4703 rows

**synthetic[tvae_cap256_seed0]**
- `encounter_primary_reason_CV_Disease_f5a_w1a_first => encounter_primary_reason_CV_Disease_f5a_w5a_first` (implication): rate 0.07143 over 14 rows
- `conditions_mi => conditions_ihd` (implication): rate 0.01065 over 845 rows
- `hyperkalemia_severity_categorizedValue vs lab_results_potassium_value_last` (category_range): rate 0.0064 over 4688 rows
- `conditions_ap => conditions_ihd` (implication): rate 0.00303 over 661 rows

**synthetic[tvae_ep1000_seed0]**
- `encounter_primary_reason_CV_Disease_f5a_w1a_first => encounter_primary_reason_CV_Disease_f5a_w5a_first` (implication): rate 0.04762 over 21 rows
- `encounter_primary_reason_CV_Disease_f5a_w3a_first => encounter_primary_reason_CV_Disease_f5a_w5a_first` (implication): rate 0.03571 over 28 rows
- `conditions_ap => conditions_ihd` (implication): rate 0.03251 over 646 rows
- `conditions_mi => conditions_ihd` (implication): rate 0.01508 over 1061 rows
- `hyperkalemia_severity_categorizedValue vs lab_results_potassium_value_last` (category_range): rate 0.00539 over 4638 rows

**synthetic[tvae_ind_seed0]**
- `encounter_primary_reason_CV_Disease_f5a_w1a_first => encounter_primary_reason_CV_Disease_f5a_w3a_first` (implication): rate 0.08333 over 12 rows
- `encounter_primary_reason_CV_Disease_f5a_w5a_first => encounter_primary_reason_CV_Disease_f5a_w3a_first` (implication): rate 0.06452 over 31 rows
- `encounter_primary_reason_CV_Disease_f5a_w3a_first => encounter_primary_reason_CV_Disease_f5a_w5a_first` (implication): rate 0.03333 over 30 rows
- `conditions_mi => conditions_ihd` (implication): rate 0.00851 over 470 rows
- `hyperkalemia_severity_categorizedValue vs lab_results_potassium_value_last` (category_range): rate 0.00213 over 4702 rows

**synthetic[tvae_qt_seed0]**
- `encounter_primary_reason_CV_Disease_f5a_w1a_first => encounter_primary_reason_CV_Disease_f5a_w3a_first` (implication): rate 0.125 over 16 rows
- `encounter_primary_reason_CV_Disease_f5a_w3a_first => encounter_primary_reason_CV_Disease_f5a_w5a_first` (implication): rate 0.07692 over 39 rows
- `encounter_primary_reason_CV_Disease_f5a_w3a_first => encounter_primary_reason_CV_Disease_f5a_w1a_first` (implication): rate 0.06667 over 15 rows
- `encounter_primary_reason_CV_Disease_f5a_w1a_first => encounter_primary_reason_CV_Disease_f5a_w5a_first` (implication): rate 0.05882 over 17 rows
- `encounter_primary_reason_CV_Disease_f5a_w5a_first => encounter_primary_reason_CV_Disease_f5a_w1a_first` (implication): rate 0.05882 over 17 rows

**synthetic[tvae_qt_seed1]**
- `encounter_primary_reason_CV_Disease_f5a_w3a_first => encounter_primary_reason_CV_Disease_f5a_w5a_first` (implication): rate 0.05556 over 36 rows
- `encounter_primary_reason_CV_Disease_f5a_w1a_first => encounter_primary_reason_CV_Disease_f5a_w3a_first` (implication): rate 0.04348 over 23 rows
- `encounter_primary_reason_CV_Disease_f5a_w1a_first => encounter_primary_reason_CV_Disease_f5a_w5a_first` (implication): rate 0.04348 over 23 rows
- `encounter_primary_reason_CV_Disease_f5a_w3a_first => encounter_primary_reason_CV_Disease_f5a_w1a_first` (implication): rate 0.04348 over 23 rows
- `encounter_primary_reason_CV_Disease_f5a_w5a_first => encounter_primary_reason_CV_Disease_f5a_w1a_first` (implication): rate 0.04348 over 23 rows

**synthetic[tvae_qt_seed2]**
- `encounter_primary_reason_CV_Disease_f5a_w3a_first => encounter_primary_reason_CV_Disease_f5a_w5a_first` (implication): rate 0.04878 over 41 rows
- `encounter_primary_reason_CV_Disease_f5a_w1a_first => encounter_primary_reason_CV_Disease_f5a_w3a_first` (implication): rate 0.04545 over 22 rows
- `encounter_primary_reason_CV_Disease_f5a_w1a_first => encounter_primary_reason_CV_Disease_f5a_w5a_first` (implication): rate 0.04545 over 22 rows
- `hyperkalemia_severity_categorizedValue vs lab_results_potassium_value_last` (category_range): rate 0.02565 over 4639 rows
- `conditions_mi => conditions_ihd` (implication): rate 0.01682 over 535 rows

**synthetic[tvae_seed0]**
- `encounter_primary_reason_CV_Disease_f5a_w1a_first => encounter_primary_reason_CV_Disease_f5a_w5a_first` (implication): rate 0.04545 over 22 rows
- `encounter_primary_reason_CV_Disease_f5a_w3a_first => encounter_primary_reason_CV_Disease_f5a_w5a_first` (implication): rate 0.02857 over 35 rows
- `conditions_ap => conditions_ihd` (implication): rate 0.026 over 500 rows
- `hyperkalemia_severity_categorizedValue vs lab_results_potassium_value_last` (category_range): rate 0.00687 over 4655 rows
- `conditions_mi => conditions_ihd` (implication): rate 0.00605 over 826 rows

**synthetic[tvae_seed1]**
- `hyperkalemia_severity_categorizedValue vs lab_results_potassium_value_last` (category_range): rate 0.00682 over 4691 rows
- `conditions_ap => conditions_ihd` (implication): rate 0.00572 over 699 rows
- `conditions_mi => conditions_ihd` (implication): rate 0.00097 over 1032 rows

**synthetic[tvae_seed2]**
- `encounter_primary_reason_CV_Disease_f5a_w3a_first => encounter_primary_reason_CV_Disease_f5a_w1a_first` (implication): rate 0.05882 over 17 rows
- `encounter_primary_reason_CV_Disease_f5a_w5a_first => encounter_primary_reason_CV_Disease_f5a_w1a_first` (implication): rate 0.05556 over 18 rows
- `conditions_mi => conditions_ihd` (implication): rate 0.01601 over 937 rows
- `conditions_ap => conditions_ihd` (implication): rate 0.01093 over 549 rows
- `hyperkalemia_severity_categorizedValue vs lab_results_potassium_value_last` (category_range): rate 0.00664 over 4666 rows

