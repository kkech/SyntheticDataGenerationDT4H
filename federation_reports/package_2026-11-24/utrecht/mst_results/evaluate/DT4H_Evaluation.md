# Evaluation: fidelity against the sampling-noise floor

Numeric metrics (KS, `W/std`) are computed over observed values only; the missing-rate MAD compares numeric missingness separately, and covers numeric columns alone. Categorical TVD instead treats nulls as an explicit 'Missing' category, so categorical missingness differences are already inside the TVD. KS and TVD are in [0,1], lower is closer; `W/std` is the Wasserstein distance in units of the reference standard deviation. The `train vs holdout` row is the sampling-noise floor: two disjoint samples of real patients differ by this much purely by chance, so read every synthetic row against it. To keep that reading fair, each synthetic frame is subsampled to the holdout's row count (averaged over seeded draws) before train-vs-synthetic scoring -- the floor is only exchangeable with comparisons at the same sample sizes. 74 constant columns (re-attached verbatim, trivially perfect) are excluded from all aggregates.

| comparison | cols | KS mean | KS median | KS<0.1 | W/std mean | TVD mean | TVD<0.05 | missing-rate MAD |
|---|---|---|---|---|---|---|---|---|
| original vs preprocessed | 183 | 0.0022 | 0.0 | 1.0 | 0.0118 | 0.0 | 1.0 | 0.0006 |
| train vs holdout | 198 | 0.0586 | 0.0493 | 0.878 | 0.0984 | 0.0137 | 0.9741 | 0.0104 |
| train vs synthetic[mst_eps15_seed0] | 198 | 0.4867 | 0.4164 | 0.0244 | 0.8645 | 0.0118 | 0.9914 | 0.005 |

## Per (model, ε) across seeds (train vs synthetic)

| model | ε | runs | KS mean ± sd | TVD mean ± sd | missing-MAD ± sd |
|---|---|---|---|---|---|
| mst | 15 | 1 | 0.4867 | 0.0118 | 0.005 |

## Full-joint distinguishability (C2ST)

Out-of-fold AUC (5-fold stratified CV) of a classifier separating real from synthetic rows, over the columns present in BOTH frames; 0.5 = joints indistinguishable. `coverage` is the fraction of modelled columns the synthetic file actually contains -- width-limited runs are scored on that intersection only, never on schema width itself. Floor (train vs holdout): **0.4951**.

| run | C2ST AUC | ± sd | coverage |
|---|---|---|---|
| mst_eps15_seed0 | 1.0 | 0.0 | 1.0 |

## Subgroup fidelity (KS mean per stratum, train vs synthetic)

Does the synthetic cohort represent every subgroup as faithfully as the majority? Each cell is read against its stratum's own noise floor.

| run | female | male | age_under_65 | age_65_79 | age_80_plus |
|---|---|---|---|---|---|
| *noise floor* | 0.1151 | 0.0746 | 0.0839 | 0.1013 | 0.153 |
| mst_eps15_seed0 | 0.504 | 0.5002 | 0.6786 | 0.5228 | 0.6454 |

## Generalization (holdout vs synthetic)

Distance to real records the generator NEVER saw. A model that is much closer to train than to holdout is fitting its training sample, not the population.

| run | KS mean (train) | KS mean (holdout) | TVD mean (train) | TVD mean (holdout) |
|---|---|---|---|---|
| mst_eps15_seed0 | 0.4867 | 0.4825 | 0.0118 | 0.0149 |

## Association structure (train vs synthetic)

Absolute change in pairwise association; 0 = relationship perfectly preserved. `fabricated` counts pairs nearly independent in real data (|assoc|<0.1) rendered strongly associated (>0.5) in the synthetic data. Noise floor rows show how much two real samples differ.

| run | pair type | pairs | mean \|Δ\| | median \|Δ\| | \|Δ\|<0.1 | fabricated | worst pair |
|---|---|---|---|---|---|---|---|
| *noise floor* | Spearman (num-num) | 2685 | 0.0693 | 0.0493 | 0.7616 | 0 | - |
| *noise floor* | Cramer's V (cat-cat) | 6555 | 0.0311 | 0.0235 | 0.9744 | 0 | - |
| *noise floor* | corr-ratio (num-cat) | 14820 | 0.0277 | 0.0105 | 0.9385 | 5 | - |
| mst_eps15_seed0 | Spearman (num-num) | 2618 | 0.2843 | 0.2454 | 0.2277 | 244 | `encounter_primary_reason_number_of_days_to_rehosp_for_heart_failure_f5a_first|encounter_primary_reason_number_of_CV_rehospitalizations_5a_f5a_count` (-0.1914 -> 0.9934) |
| mst_eps15_seed0 | Cramer's V (cat-cat) | 6670 | 0.2317 | 0.1909 | 0.3504 | 843 | `med_arni_history|conditions_mi` (0.0257 -> 0.9735) |
| mst_eps15_seed0 | corr-ratio (num-cat) | 14820 | 0.1278 | 0.0384 | 0.6182 | 858 | `lab_results_ferritin_value_last|encounter_primary_reason_renal_complications_f5a_w6mo_first` (0.0101 -> 1.0) |

## original vs preprocessed

Worst numeric columns (by KS):
- `lab_results_bnp_value_last`: KS=0.0494, W/std=0.1809, mean 2517.6457 -> 1878.9763, missing 89% -> 90%
- `lab_results_bnp_value_first`: KS=0.0456, W/std=0.169, mean 2551.4415 -> 1954.1061, missing 89% -> 90%
- `lab_results_ntProBnp_value_last`: KS=0.0176, W/std=0.1082, mean 10046.6132 -> 7911.5324, missing 82% -> 82%
- `vital_signs_height_value_last`: KS=0.014, W/std=0.0967, mean 172.435 -> 174.178, missing 34% -> 35%
- `lab_results_tfs_value_last`: KS=0.0121, W/std=0.0735, mean 17.6634 -> 16.201, missing 83% -> 84%
Worst categorical columns (by TVD):
- `patient_demographics_gender`: TVD=0.0, 2 -> 2 categories, missing 0% -> 0%
- `encounters_encounterClass`: TVD=0.0, 1 -> 1 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.0, 13 -> 13 categories, missing 0% -> 0%
- `symptoms_Ankle_swelling_display_pET_any`: TVD=0.0, 1 -> 1 categories, missing 0% -> 0%
- `symptoms_Ascites_display_pET_any`: TVD=0.0, 1 -> 1 categories, missing 0% -> 0%

## train vs holdout

Worst numeric columns (by KS):
- `echocardiographs_lvef`: KS=0.2192, W/std=0.4006, mean 42.0451 -> 50.0552, missing 91% -> 93%
- `echocardiographs_lvef_pET_last`: KS=0.2005, W/std=0.2969, mean 41.0622 -> 46.8099, missing 93% -> 95%
- `maggic_total_score`: KS=0.1459, W/std=0.2376, mean 24.2632 -> 23.1667, missing 93% -> 95%
- `lab_results_hba1c%_value_last`: KS=0.1262, W/std=0.1527, mean 6.3125 -> 6.4602, missing 88% -> 86%
- `lab_results_hba1c_value_last`: KS=0.1262, W/std=0.1567, mean 45.4688 -> 47.1364, missing 88% -> 86%
Worst categorical columns (by TVD):
- `ckd_severity_categorizedValue`: TVD=0.0621, 7 -> 6 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.0595, 13 -> 12 categories, missing 0% -> 0%
- `conditions_heart_failure_occurred_prior_to_18_months_any`: TVD=0.0509, 2 -> 2 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.0387, 5 -> 5 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.0387, 5 -> 5 categories, missing 0% -> 0%

## train vs synthetic[mst_eps15_seed0]

Worst numeric columns (by KS):
- `encounter_primary_reason_number_of_renal_rehospitalizations_5a_f5a_count`: KS=0.9984, W/std=5.0465, mean 0.0494 -> 3.1724, missing 0% -> 1%
- `encounters_numOfPreviousHFStays_count`: KS=0.9968, W/std=7.8795, mean 2.7336 -> 53.9785, missing 0% -> 0%
- `lab_results_creatBS_value_p3a_avg`: KS=0.9909, W/std=5.0891, mean 13.3257 -> 58.4946, missing 0% -> 0%
- `smoking_status_smoker_startTime_count`: KS=0.9903, W/std=4.4696, mean 0.0988 -> 2.8468, missing 0% -> 0%
- `lab_results_ferritin_value_first`: KS=0.9894, W/std=5.452, mean 453.6144 -> 6828.5895, missing 80% -> 80%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.0523, 13 -> 13 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.0265, 7 -> 7 categories, missing 0% -> 0%
- `encounter_primary_reason_HF_Disease_f5a_w3a_first`: TVD=0.0212, 3 -> 3 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.0202, 5 -> 5 categories, missing 0% -> 0%
- `encounter_primary_reason_HF_Disease_f5a_w1a_first`: TVD=0.0197, 3 -> 3 categories, missing 0% -> 0%
