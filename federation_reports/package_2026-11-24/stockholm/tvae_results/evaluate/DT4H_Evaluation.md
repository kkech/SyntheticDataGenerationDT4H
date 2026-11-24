# Evaluation: fidelity against the sampling-noise floor

Numeric metrics (KS, `W/std`) are computed over observed values only; the missing-rate MAD compares numeric missingness separately, and covers numeric columns alone. Categorical TVD instead treats nulls as an explicit 'Missing' category, so categorical missingness differences are already inside the TVD. KS and TVD are in [0,1], lower is closer; `W/std` is the Wasserstein distance in units of the reference standard deviation. The `train vs holdout` row is the sampling-noise floor: two disjoint samples of real patients differ by this much purely by chance, so read every synthetic row against it. To keep that reading fair, each synthetic frame is subsampled to the holdout's row count (averaged over seeded draws) before train-vs-synthetic scoring -- the floor is only exchangeable with comparisons at the same sample sizes. 71 constant columns (re-attached verbatim, trivially perfect) are excluded from all aggregates.

| comparison | cols | KS mean | KS median | KS<0.1 | W/std mean | TVD mean | TVD<0.05 | missing-rate MAD |
|---|---|---|---|---|---|---|---|---|
| original vs preprocessed | 164 | 0.0032 | 0.0 | 1.0 | 0.0127 | 0.0 | 1.0 | 0.0028 |
| train vs holdout | 182 | 0.1082 | 0.064 | 0.6935 | 0.2045 | 0.0216 | 0.9083 | 0.0156 |
| train vs synthetic[tvae_seed0] | 182 | 0.2751 | 0.2452 | 0.0484 | 0.3195 | 0.0729 | 0.375 | 0.0594 |

## Per (model, ε) across seeds (train vs synthetic)

| model | ε | runs | KS mean ± sd | TVD mean ± sd | missing-MAD ± sd |
|---|---|---|---|---|---|
| tvae | - | 1 | 0.2751 | 0.0729 | 0.0594 |

## Full-joint distinguishability (C2ST)

Out-of-fold AUC (5-fold stratified CV) of a classifier separating real from synthetic rows, over the columns present in BOTH frames; 0.5 = joints indistinguishable. `coverage` is the fraction of modelled columns the synthetic file actually contains -- width-limited runs are scored on that intersection only, never on schema width itself. Floor (train vs holdout): **0.4342**.

| run | C2ST AUC | ± sd | coverage |
|---|---|---|---|
| tvae_seed0 | 0.9989 | 0.0016 | 1.0 |

## Subgroup fidelity (KS mean per stratum, train vs synthetic)

Does the synthetic cohort represent every subgroup as faithfully as the majority? Each cell is read against its stratum's own noise floor.

| run | female | male | age_65_79 | age_80_plus |
|---|---|---|---|---|
| *noise floor* | 0.2152 | 0.1276 | 0.1853 | 0.2369 |
| tvae_seed0 | 0.3666 | 0.2907 | 0.2974 | 0.3365 |

## Generalization (holdout vs synthetic)

Distance to real records the generator NEVER saw. A model that is much closer to train than to holdout is fitting its training sample, not the population.

| run | KS mean (train) | KS mean (holdout) | TVD mean (train) | TVD mean (holdout) |
|---|---|---|---|---|
| tvae_seed0 | 0.2751 | 0.2917 | 0.0729 | 0.065 |

## Association structure (train vs synthetic)

Absolute change in pairwise association; 0 = relationship perfectly preserved. `fabricated` counts pairs nearly independent in real data (|assoc|<0.1) rendered strongly associated (>0.5) in the synthetic data. Noise floor rows show how much two real samples differ.

| run | pair type | pairs | mean \|Δ\| | median \|Δ\| | \|Δ\|<0.1 | fabricated | worst pair |
|---|---|---|---|---|---|---|---|
| *noise floor* | Spearman (num-num) | 986 | 0.0811 | 0.0664 | 0.6684 | 0 | - |
| *noise floor* | Cramer's V (cat-cat) | 6903 | 0.0544 | 0.0415 | 0.855 | 2 | - |
| *noise floor* | corr-ratio (num-cat) | 8740 | 0.0406 | 0.0176 | 0.865 | 0 | - |
| tvae_seed0 | Spearman (num-num) | 845 | 0.1075 | 0.0826 | 0.5964 | 0 | `lab_results_tropTnHs_value_first|lab_results_tropTnHs_value_last` (0.8718 -> -0.0576) |
| tvae_seed0 | Cramer's V (cat-cat) | 4656 | 0.0928 | 0.0684 | 0.6619 | 6 | `med_oral_antidiabetic_history|conditions_revasc` (0.1374 -> 0.8126) |
| tvae_seed0 | corr-ratio (num-cat) | 7980 | 0.0459 | 0.0192 | 0.8494 | 0 | `encounter_primary_reason_number_of_days_to_rehosp_for_CV_f5a_first|encounter_primary_reason_CV_Disease_f5a_w3a_first` (0.8511 -> 0.0) |

## original vs preprocessed

Worst numeric columns (by KS):
- `encounter_primary_reason_number_of_CV_rehospitalizations_5a_f5a_count`: KS=0.0761, W/std=0.225, mean 17.8603 -> 13.004, missing 0% -> 8%
- `encounter_primary_reason_number_of_non_CV_rehospitalizations_5a_f5a_count`: KS=0.0561, W/std=0.1384, mean 15.0973 -> 7.5271, missing 0% -> 6%
- `vital_signs_height_value_last`: KS=0.0308, W/std=0.1466, mean 169.8219 -> 173.0944, missing 68% -> 69%
- `vital_signs_height_value_p1a_avg`: KS=0.0128, W/std=0.0746, mean 170.6547 -> 172.0087, missing 51% -> 52%
- `encounter_primary_reason_number_of_renal_rehospitalizations_5a_f5a_count`: KS=0.0125, W/std=0.0818, mean 5.1097 -> 1.1301, missing 0% -> 1%
Worst categorical columns (by TVD):
- `patient_demographics_gender`: TVD=0.0, 2 -> 2 categories, missing 0% -> 0%
- `encounters_encounterClass`: TVD=0.0, 1 -> 1 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.0, 13 -> 13 categories, missing 0% -> 0%
- `symptoms_Ankle_swelling_display_pET_any`: TVD=0.0, 1 -> 1 categories, missing 0% -> 0%
- `symptoms_Ascites_display_pET_any`: TVD=0.0, 1 -> 1 categories, missing 0% -> 0%

## train vs holdout

Worst numeric columns (by KS):
- `lab_results_ldl_value_last`: KS=0.4134, W/std=0.5952, mean 2.0515 -> 2.5722, missing 89% -> 91%
- `lab_results_ldl_value_first`: KS=0.4134, W/std=0.5917, mean 2.05 -> 2.5667, missing 89% -> 91%
- `lab_results_cholTot_value_first`: KS=0.3707, W/std=0.5244, mean 3.8246 -> 4.3632, missing 89% -> 90%
- `lab_results_cholTot_value_last`: KS=0.3707, W/std=0.5099, mean 3.829 -> 4.3526, missing 89% -> 90%
- `lab_results_ferritin_value_first`: KS=0.3304, W/std=0.5702, mean 287.7073 -> 443.6364, missing 93% -> 94%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.1111, 13 -> 13 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.1034, 7 -> 7 categories, missing 0% -> 0%
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first`: TVD=0.0676, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first`: TVD=0.061, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_HF_Disease_f5a_w7d_first`: TVD=0.0577, 3 -> 3 categories, missing 0% -> 0%

## train vs synthetic[tvae_seed0]

Worst numeric columns (by KS):
- `encounter_primary_reason_number_of_days_to_rehosp_for_non_CV_f5a_first`: KS=0.8903, W/std=0.2064, mean 20.2774 -> 1.2895, missing 74% -> 94%
- `encounter_primary_reason_number_of_days_to_rehosp_for_heart_failure_f5a_first`: KS=0.7899, W/std=0.2571, mean 14.7826 -> 5.4303, missing 77% -> 82%
- `encounter_primary_reason_number_of_days_to_rehosp_for_CV_f5a_first`: KS=0.7019, W/std=0.1208, mean 13.662 -> 6.4555, missing 28% -> 48%
- `encounter_primary_reason_number_of_renal_rehospitalizations_5a_f5a_count`: KS=0.6906, W/std=0.16, mean 1.0571 -> 0.611, missing 1% -> 43%
- `lab_results_tfs_value_first`: KS=0.5093, W/std=0.6839, mean 15.5556 -> 16.0022, missing 94% -> 98%
Worst categorical columns (by TVD):
- `conditions_mc`: TVD=0.2358, 2 -> 2 categories, missing 0% -> 0%
- `med_anti_coag_history`: TVD=0.2091, 2 -> 2 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.203, 13 -> 13 categories, missing 0% -> 0%
- `conditions_copd`: TVD=0.1761, 2 -> 1 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.1737, 5 -> 4 categories, missing 0% -> 0%
