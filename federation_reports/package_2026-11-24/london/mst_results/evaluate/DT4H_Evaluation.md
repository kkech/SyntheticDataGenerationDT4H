# Evaluation: fidelity against the sampling-noise floor

Numeric metrics (KS, `W/std`) are computed over observed values only; the missing-rate MAD compares numeric missingness separately, and covers numeric columns alone. Categorical TVD instead treats nulls as an explicit 'Missing' category, so categorical missingness differences are already inside the TVD. KS and TVD are in [0,1], lower is closer; `W/std` is the Wasserstein distance in units of the reference standard deviation. The `train vs holdout` row is the sampling-noise floor: two disjoint samples of real patients differ by this much purely by chance, so read every synthetic row against it. To keep that reading fair, each synthetic frame is subsampled to the holdout's row count (averaged over seeded draws) before train-vs-synthetic scoring -- the floor is only exchangeable with comparisons at the same sample sizes. 37 constant columns (re-attached verbatim, trivially perfect) are excluded from all aggregates.

| comparison | cols | KS mean | KS median | KS<0.1 | W/std mean | TVD mean | TVD<0.05 | missing-rate MAD |
|---|---|---|---|---|---|---|---|---|
| original vs preprocessed | 168 | 0.0001 | 0.0 | 1.0 | 0.0014 | 0.0 | 1.0 | 0.0001 |
| train vs holdout | 220 | 0.0882 | 0.0739 | 0.6923 | 0.1436 | 0.0206 | 0.9226 | 0.0118 |
| 🚨 synthetic[mst_eps15_seed0] IS STALE (1508 undecoded cells; excluded from group aggregates) -- rerun generate | | | | | | | | |
| train vs synthetic[mst_eps15_seed0] | 220 | 0.5331 | 0.4326 | 0.0 | 3.0401 | 0.0157 | 0.9871 | 0.0072 |

## Per (model, ε) across seeds (train vs synthetic)

| model | ε | runs | KS mean ± sd | TVD mean ± sd | missing-MAD ± sd |
|---|---|---|---|---|---|
| mst | 15 | 0 | - | - | - |

Excluded from these group aggregates as stale (scores describe outdated files; see the per-run table): `mst_eps15_seed0`.

## Full-joint distinguishability (C2ST)

Out-of-fold AUC (5-fold stratified CV) of a classifier separating real from synthetic rows, over the columns present in BOTH frames; 0.5 = joints indistinguishable. `coverage` is the fraction of modelled columns the synthetic file actually contains -- width-limited runs are scored on that intersection only, never on schema width itself. Floor (train vs holdout): **0.5563**.

| run | C2ST AUC | ± sd | coverage |
|---|---|---|---|
| mst_eps15_seed0 | 1.0 | 0.0 | 1.0 |

## Subgroup fidelity (KS mean per stratum, train vs synthetic)

Does the synthetic cohort represent every subgroup as faithfully as the majority? Each cell is read against its stratum's own noise floor.

| run | female | male | age_under_65 | age_65_79 | age_80_plus |
|---|---|---|---|---|---|
| *noise floor* | 0.144 | 0.1035 | 0.1721 | 0.1332 | 0.138 |
| mst_eps15_seed0 | 0.5578 | 0.5435 | 0.7395 | 0.6188 | 0.6226 |

## Generalization (holdout vs synthetic)

Distance to real records the generator NEVER saw. A model that is much closer to train than to holdout is fitting its training sample, not the population.

| run | KS mean (train) | KS mean (holdout) | TVD mean (train) | TVD mean (holdout) |
|---|---|---|---|---|
| mst_eps15_seed0 | 0.5331 | 0.533 | 0.0157 | 0.0234 |

## Association structure (train vs synthetic)

Absolute change in pairwise association; 0 = relationship perfectly preserved. `fabricated` counts pairs nearly independent in real data (|assoc|<0.1) rendered strongly associated (>0.5) in the synthetic data. Noise floor rows show how much two real samples differ.

| run | pair type | pairs | mean \|Δ\| | median \|Δ\| | \|Δ\|<0.1 | fabricated | worst pair |
|---|---|---|---|---|---|---|---|
| *noise floor* | Spearman (num-num) | 1183 | 0.0741 | 0.06 | 0.7354 | 0 | - |
| *noise floor* | Cramer's V (cat-cat) | 11628 | 0.05 | 0.0367 | 0.8767 | 4 | - |
| *noise floor* | corr-ratio (num-cat) | 9500 | 0.0424 | 0.0269 | 0.8893 | 3 | - |
| mst_eps15_seed0 | Spearman (num-num) | 1187 | 0.346 | 0.3139 | 0.1643 | 177 | `vital_signs_weight_value_p6mo_first|lab_results_ntProBnp_value_last` (-0.34 -> 0.7532) |
| mst_eps15_seed0 | Cramer's V (cat-cat) | 11325 | 0.3544 | 0.3632 | 0.1372 | 3270 | `cause_of_death_isRenal_f5a_w7d_first|cause_of_death_isNonRenalAndNonCV_f5a_w7d_first` (1.0 -> 0.0099) |
| mst_eps15_seed0 | corr-ratio (num-cat) | 9500 | 0.2319 | 0.2222 | 0.3531 | 1253 | `encounter_primary_reason_number_of_renal_rehospitalizations_5a_f5a_count|encounter_primary_reason_renal_complications_f5a_w3a_first` (1.0 -> 0.0591) |

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

## train vs synthetic[mst_eps15_seed0]

Worst numeric columns (by KS):
- `lab_results_ferritin_value_first`: KS=1.0, W/std=14.8666, mean 454.268 -> 11648.5046, missing 80% -> 78%
- `lab_results_tropTnHs_value_last`: KS=1.0, W/std=22.4799, mean 88.3696 -> 6250.0, missing 26% -> 26%
- `encounter_primary_reason_number_of_HF_rehospitalizations_5a_f5a_count`: KS=1.0, W/std=36.2179, mean 0.0093 -> 3.4827, missing 0% -> 0%
- `encounter_primary_reason_number_of_renal_rehospitalizations_5a_f5a_count`: KS=1.0, W/std=36.4996, mean 0.0027 -> 2.6594, missing 0% -> 0%
- `lab_results_tropTnHs_value_first`: KS=0.9982, W/std=23.0327, mean 75.8304 -> 7474.4969, missing 26% -> 25%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.2904, 58 -> 56 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.052, 7 -> 6 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.0437, 5 -> 5 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.0363, 5 -> 5 categories, missing 0% -> 0%
- `med_ll`: TVD=0.033, 2 -> 2 categories, missing 0% -> 0%
