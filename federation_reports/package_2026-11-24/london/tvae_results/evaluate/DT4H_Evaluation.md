# Evaluation: fidelity against the sampling-noise floor

Numeric metrics (KS, `W/std`) are computed over observed values only; the missing-rate MAD compares numeric missingness separately, and covers numeric columns alone. Categorical TVD instead treats nulls as an explicit 'Missing' category, so categorical missingness differences are already inside the TVD. KS and TVD are in [0,1], lower is closer; `W/std` is the Wasserstein distance in units of the reference standard deviation. The `train vs holdout` row is the sampling-noise floor: two disjoint samples of real patients differ by this much purely by chance, so read every synthetic row against it. To keep that reading fair, each synthetic frame is subsampled to the holdout's row count (averaged over seeded draws) before train-vs-synthetic scoring -- the floor is only exchangeable with comparisons at the same sample sizes. 37 constant columns (re-attached verbatim, trivially perfect) are excluded from all aggregates.

| comparison | cols | KS mean | KS median | KS<0.1 | W/std mean | TVD mean | TVD<0.05 | missing-rate MAD |
|---|---|---|---|---|---|---|---|---|
| original vs preprocessed | 168 | 0.0001 | 0.0 | 1.0 | 0.0014 | 0.0 | 1.0 | 0.0001 |
| train vs holdout | 220 | 0.0882 | 0.0739 | 0.6923 | 0.1436 | 0.0206 | 0.9226 | 0.0118 |
| 🚨 synthetic[tvae_seed0] IS STALE (1508 undecoded cells; excluded from group aggregates) -- rerun generate | | | | | | | | |
| train vs synthetic[tvae_seed0] | 220 | 0.2257 | 0.1775 | 0.2154 | 0.2784 | 0.0588 | 0.5613 | 0.0355 |

## Per (model, ε) across seeds (train vs synthetic)

| model | ε | runs | KS mean ± sd | TVD mean ± sd | missing-MAD ± sd |
|---|---|---|---|---|---|
| tvae | - | 0 | - | - | - |

Excluded from these group aggregates as stale (scores describe outdated files; see the per-run table): `tvae_seed0`.

## Full-joint distinguishability (C2ST)

Out-of-fold AUC (5-fold stratified CV) of a classifier separating real from synthetic rows, over the columns present in BOTH frames; 0.5 = joints indistinguishable. `coverage` is the fraction of modelled columns the synthetic file actually contains -- width-limited runs are scored on that intersection only, never on schema width itself. Floor (train vs holdout): **0.5563**.

| run | C2ST AUC | ± sd | coverage |
|---|---|---|---|
| tvae_seed0 | 0.9952 | 0.0049 | 1.0 |

## Subgroup fidelity (KS mean per stratum, train vs synthetic)

Does the synthetic cohort represent every subgroup as faithfully as the majority? Each cell is read against its stratum's own noise floor.

| run | female | male | age_under_65 | age_65_79 | age_80_plus |
|---|---|---|---|---|---|
| *noise floor* | 0.144 | 0.1035 | 0.1721 | 0.1332 | 0.138 |
| tvae_seed0 | 0.2828 | 0.2356 | 0.2857 | 0.2575 | 0.2807 |

## Generalization (holdout vs synthetic)

Distance to real records the generator NEVER saw. A model that is much closer to train than to holdout is fitting its training sample, not the population.

| run | KS mean (train) | KS mean (holdout) | TVD mean (train) | TVD mean (holdout) |
|---|---|---|---|---|
| tvae_seed0 | 0.2257 | 0.2279 | 0.0588 | 0.0587 |

## Association structure (train vs synthetic)

Absolute change in pairwise association; 0 = relationship perfectly preserved. `fabricated` counts pairs nearly independent in real data (|assoc|<0.1) rendered strongly associated (>0.5) in the synthetic data. Noise floor rows show how much two real samples differ.

| run | pair type | pairs | mean \|Δ\| | median \|Δ\| | \|Δ\|<0.1 | fabricated | worst pair |
|---|---|---|---|---|---|---|---|
| *noise floor* | Spearman (num-num) | 1183 | 0.0741 | 0.06 | 0.7354 | 0 | - |
| *noise floor* | Cramer's V (cat-cat) | 11628 | 0.05 | 0.0367 | 0.8767 | 4 | - |
| *noise floor* | corr-ratio (num-cat) | 9500 | 0.0424 | 0.0269 | 0.8893 | 3 | - |
| tvae_seed0 | Spearman (num-num) | 1018 | 0.0951 | 0.0734 | 0.6395 | 0 | `lab_results_tfs_value_last|lab_results_tfs_value_first` (0.9445 -> 0.1295) |
| tvae_seed0 | Cramer's V (cat-cat) | 7750 | 0.0905 | 0.0686 | 0.6277 | 7 | `med_vasodil_history|conditions_ibd` (0.0628 -> 0.6726) |
| tvae_seed0 | corr-ratio (num-cat) | 8740 | 0.0598 | 0.0371 | 0.7803 | 2 | `lab_results_potassium_value_last|hyperkalemia_severity_categorizedValue` (0.7537 -> 0.0) |

## original vs preprocessed

Worst numeric columns (by KS):
- `lab_results_ntProBnp_value_last`: KS=0.0027, W/std=0.0231, mean 9480.3924 -> 9135.6025, missing 27% -> 27%
- `lab_results_ntProBnp_value_first`: KS=0.0014, W/std=0.0161, mean 9531.4782 -> 9295.8076, missing 27% -> 27%
- `lab_results_bun_value_last`: KS=0.0012, W/std=0.0073, mean 11.1798 -> 11.1206, missing 16% -> 16%
- `lab_results_albuminBS_value_first`: KS=0.0011, W/std=0.0079, mean 38.119 -> 38.1596, missing 6% -> 6%
- `lab_results_sodium_value_first`: KS=0.001, W/std=0.0194, mean 136.7614 -> 136.8897, missing 2% -> 2%
Worst categorical columns (by TVD):
- `patient_demographics_gender`: TVD=0.0, 2 -> 2 categories, missing 0% -> 0%
- `encounters_encounterClass`: TVD=0.0, 1 -> 1 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.0, 64 -> 64 categories, missing 0% -> 0%
- `symptoms_Ankle_swelling_display_pET_any`: TVD=0.0, 2 -> 2 categories, missing 0% -> 0%
- `symptoms_Ascites_display_pET_any`: TVD=0.0, 2 -> 2 categories, missing 0% -> 0%

## train vs holdout

Worst numeric columns (by KS):
- `vital_signs_oxygenSaturation_value_first`: KS=0.2814, W/std=0.3586, mean 95.4808 -> 96.4737, missing 93% -> 92%
- `vital_signs_oxygenSaturation_value_last`: KS=0.2672, W/std=0.335, mean 95.3462 -> 96.0, missing 93% -> 92%
- `lab_results_cholTot_value_last`: KS=0.2396, W/std=0.3376, mean 4.0427 -> 3.6917, missing 87% -> 90%
- `lab_results_cholTot_value_first`: KS=0.2396, W/std=0.3281, mean 4.051 -> 3.7083, missing 87% -> 90%
- `lab_results_triGly_value_last`: KS=0.1929, W/std=0.3807, mean 1.3021 -> 1.1083, missing 87% -> 90%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.2317, 58 -> 53 categories, missing 0% -> 0%
- `conditions_cm`: TVD=0.1077, 2 -> 2 categories, missing 0% -> 0%
- `med_rasi`: TVD=0.0775, 2 -> 2 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.0735, 7 -> 6 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.0711, 5 -> 5 categories, missing 0% -> 0%

## train vs synthetic[tvae_seed0]

Worst numeric columns (by KS):
- `vital_signs_oxygenSaturation_value_last`: KS=1.0, W/std=None, mean 95.3462 -> None, missing 93% -> 100%
- `vital_signs_oxygenSaturation_value_first`: KS=0.8654, W/std=0.6483, mean 95.4808 -> 95.76, missing 93% -> 100%
- `encounter_primary_reason_number_of_non_CV_rehospitalizations_5a_f5a_count`: KS=0.6499, W/std=0.2394, mean 1.2218 -> 1.165, missing 0% -> 27%
- `lab_results_hba1c%_value_last`: KS=0.5648, W/std=0.5752, mean 7.0583 -> 7.1184, missing 95% -> 99%
- `lab_results_hba1c%_value_first`: KS=0.5, W/std=0.5369, mean 7.0417 -> 7.0635, missing 95% -> 98%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.2436, 58 -> 52 categories, missing 0% -> 0%
- `conditions_copd`: TVD=0.2109, 2 -> 2 categories, missing 0% -> 0%
- `conditions_pad`: TVD=0.2041, 2 -> 2 categories, missing 0% -> 0%
- `conditions_dysl`: TVD=0.1921, 2 -> 2 categories, missing 0% -> 0%
- `med_digitalis`: TVD=0.1897, 2 -> 1 categories, missing 0% -> 0%
