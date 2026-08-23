# Evaluation: fidelity against the sampling-noise floor

Metrics are computed per column over observed values (nulls excluded); missingness rates are compared separately. KS and TVD are in [0,1], lower is closer; `W/std` is the Wasserstein distance in units of the reference standard deviation. The `train vs holdout` row is the sampling-noise floor: two disjoint samples of real patients differ by this much purely by chance, so read every synthetic row against it. 38 constant columns (re-attached verbatim, trivially perfect) are excluded from all aggregates.

| comparison | cols | KS mean | KS median | KS<0.1 | W/std mean | TVD mean | TVD<0.05 | missing-rate MAD |
|---|---|---|---|---|---|---|---|---|
| original vs preprocessed | 164 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 1.0 | 0.0 |
| train vs holdout | 211 | 0.0486 | 0.0405 | 0.9344 | 0.0782 | 0.01 | 1.0 | 0.0075 |
| train vs synthetic[aim50_eps1_seed0] | 50 | 0.3256 | 0.2432 | 0.0476 | 0.3177 | 0.0175 | 0.931 | 0.016 |
| train vs synthetic[aim50_eps5_seed0] | 50 | 0.3148 | 0.2521 | 0.0476 | 0.2537 | 0.0045 | 1.0 | 0.0024 |
| train vs synthetic[ctgan_seed0] | 211 | 0.2957 | 0.2678 | 0.0656 | 0.5578 | 0.1053 | 0.3 | 0.0792 |
| train vs synthetic[ctgan_seed1] | 211 | 0.3318 | 0.3409 | 0.0492 | 0.6602 | 0.0794 | 0.4333 | 0.0673 |
| train vs synthetic[ctgan_seed2] | 211 | 0.3009 | 0.2561 | 0.082 | 0.5863 | 0.0759 | 0.4133 | 0.0902 |
| train vs synthetic[dpctgan_eps10_seed0] | 176 | 0.8244 | 0.8809 | 0.0 | 3.2231 | 0.324 | 0.1533 | 0.162 |
| train vs synthetic[dpctgan_eps15_seed0] | 180 | 0.8419 | 0.9724 | 0.0 | 2.8545 | 0.3093 | 0.1533 | 0.3029 |
| train vs synthetic[dpctgan_eps15_seed1] | 181 | 0.8623 | 0.9529 | 0.0 | 2.8222 | 0.3084 | 0.1333 | 0.2254 |
| train vs synthetic[dpctgan_eps15_seed2] | 174 | 0.8281 | 0.9728 | 0.0 | 3.1901 | 0.327 | 0.1467 | 0.2655 |
| train vs synthetic[dpctgan_eps1_seed0] | 178 | 0.7261 | 0.8325 | 0.0 | 3.0404 | 0.3054 | 0.1533 | 0.2129 |
| train vs synthetic[dpctgan_eps20_seed0] | 173 | 0.8974 | 0.9954 | 0.0 | 3.018 | 0.3468 | 0.16 | 0.2369 |
| train vs synthetic[dpctgan_eps5_seed0] | 177 | 0.8934 | 0.9834 | 0.0 | 3.028 | 0.3361 | 0.16 | 0.2666 |
| train vs synthetic[dpctgan_eps8_seed0] | 179 | 0.8343 | 0.9109 | 0.0 | 2.7334 | 0.3406 | 0.1467 | 0.2317 |
| train vs synthetic[gaussian_copula_seed0] | 202 | 0.424 | 0.3641 | 0.0769 | 1.3393 | 0.0093 | 0.9933 | 0.1232 |
| train vs synthetic[gaussian_copula_seed1] | 204 | 0.4396 | 0.3897 | 0.0741 | 1.3379 | 0.0088 | 0.9933 | 0.1229 |
| train vs synthetic[gaussian_copula_seed2] | 202 | 0.4356 | 0.3711 | 0.0769 | 1.3367 | 0.009 | 0.9933 | 0.1258 |
| train vs synthetic[mst_eps10_seed0] | 211 | 0.409 | 0.284 | 0.0164 | 0.3914 | 0.0027 | 1.0 | 0.0046 |
| train vs synthetic[mst_eps15_seed0] | 211 | 0.4048 | 0.2889 | 0.0164 | 0.3803 | 0.0018 | 1.0 | 0.0028 |
| train vs synthetic[mst_eps15_seed1] | 211 | 0.4061 | 0.2851 | 0.0164 | 0.378 | 0.0017 | 1.0 | 0.0027 |
| train vs synthetic[mst_eps15_seed2] | 211 | 0.4074 | 0.2863 | 0.0164 | 0.3791 | 0.0018 | 1.0 | 0.003 |
| train vs synthetic[mst_eps1_seed0] | 211 | 0.4863 | 0.4338 | 0.0 | 0.9166 | 0.0242 | 0.92 | 0.0316 |
| train vs synthetic[mst_eps20_seed0] | 211 | 0.407 | 0.2871 | 0.0164 | 0.3748 | 0.0014 | 1.0 | 0.0019 |
| train vs synthetic[mst_eps5_seed0] | 211 | 0.4238 | 0.2996 | 0.0164 | 0.4645 | 0.0051 | 1.0 | 0.0079 |
| train vs synthetic[mst_eps8_seed0] | 211 | 0.4189 | 0.2968 | 0.0164 | 0.408 | 0.0032 | 1.0 | 0.0049 |
| train vs synthetic[tvae_seed0] | 211 | 0.1977 | 0.163 | 0.1639 | 0.2678 | 0.055 | 0.6067 | 0.0402 |
| train vs synthetic[tvae_seed1] | 211 | 0.2032 | 0.1963 | 0.2459 | 0.2626 | 0.0589 | 0.5067 | 0.0356 |
| train vs synthetic[tvae_seed2] | 211 | 0.204 | 0.1699 | 0.2131 | 0.2673 | 0.0625 | 0.52 | 0.0352 |

## Per (model, ε) across seeds (train vs synthetic)

| model | ε | runs | KS mean ± sd | TVD mean ± sd | missing-MAD ± sd |
|---|---|---|---|---|---|
| aim | 1 | 1 | 0.3256 | 0.0175 | 0.016 |
| aim | 5 | 1 | 0.3148 | 0.0045 | 0.0024 |
| ctgan | - | 3 | 0.3095 ± 0.0195 | 0.0869 ± 0.0161 | 0.0789 ± 0.0115 |
| dpctgan | 1 | 1 | 0.7261 | 0.3054 | 0.2129 |
| dpctgan | 5 | 1 | 0.8934 | 0.3361 | 0.2666 |
| dpctgan | 8 | 1 | 0.8343 | 0.3406 | 0.2317 |
| dpctgan | 10 | 1 | 0.8244 | 0.324 | 0.162 |
| dpctgan | 15 | 3 | 0.8441 ± 0.0172 | 0.3149 ± 0.0105 | 0.2646 ± 0.0388 |
| dpctgan | 20 | 1 | 0.8974 | 0.3468 | 0.2369 |
| gaussian_copula | - | 3 | 0.4331 ± 0.0081 | 0.009 ± 0.0003 | 0.124 ± 0.0016 |
| mst | 1 | 1 | 0.4863 | 0.0242 | 0.0316 |
| mst | 5 | 1 | 0.4238 | 0.0051 | 0.0079 |
| mst | 8 | 1 | 0.4189 | 0.0032 | 0.0049 |
| mst | 10 | 1 | 0.409 | 0.0027 | 0.0046 |
| mst | 15 | 3 | 0.4061 ± 0.0013 | 0.0018 ± 0.0001 | 0.0028 ± 0.0002 |
| mst | 20 | 1 | 0.407 | 0.0014 | 0.0019 |
| tvae | - | 3 | 0.2016 ± 0.0034 | 0.0588 ± 0.0038 | 0.037 ± 0.0028 |

## Full-joint distinguishability (C2ST)

AUC of a classifier separating real from synthetic rows; 0.5 = joints indistinguishable. Floor (train vs holdout): **0.4572**.

| run | C2ST AUC |
|---|---|
| aim50_eps1_seed0 | 1.0 |
| aim50_eps5_seed0 | 1.0 |
| ctgan_seed0 | 1.0 |
| ctgan_seed1 | 1.0 |
| ctgan_seed2 | 1.0 |
| dpctgan_eps10_seed0 | 1.0 |
| dpctgan_eps15_seed0 | 1.0 |
| dpctgan_eps15_seed1 | 0.9995 |
| dpctgan_eps15_seed2 | 1.0 |
| dpctgan_eps1_seed0 | 1.0 |
| dpctgan_eps20_seed0 | 1.0 |
| dpctgan_eps5_seed0 | 1.0 |
| dpctgan_eps8_seed0 | 1.0 |
| gaussian_copula_seed0 | 1.0 |
| gaussian_copula_seed1 | 1.0 |
| gaussian_copula_seed2 | 1.0 |
| mst_eps10_seed0 | 1.0 |
| mst_eps15_seed0 | 1.0 |
| mst_eps15_seed1 | 1.0 |
| mst_eps15_seed2 | 1.0 |
| mst_eps1_seed0 | 1.0 |
| mst_eps20_seed0 | 1.0 |
| mst_eps5_seed0 | 1.0 |
| mst_eps8_seed0 | 1.0 |
| tvae_seed0 | 1.0 |
| tvae_seed1 | 1.0 |
| tvae_seed2 | 1.0 |

## Subgroup fidelity (KS mean per stratum, train vs synthetic)

Does the synthetic cohort represent every subgroup as faithfully as the majority? Each cell is read against its stratum's own noise floor.

| run | female | male | age_under_65 | age_65_79 | age_80_plus |
|---|---|---|---|---|---|
| *noise floor* | 0.0745 | 0.0684 | 0.0997 | 0.0754 | 0.0961 |
| aim50_eps1_seed0 | 0.3449 | 0.3395 | 0.3915 | 0.3374 | 0.3724 |
| aim50_eps5_seed0 | 0.336 | 0.3252 | 0.5021 | 0.3473 | 0.4246 |
| ctgan_seed0 | 0.3192 | 0.2976 | 0.3149 | 0.3018 | 0.3294 |
| ctgan_seed1 | 0.3421 | 0.339 | 0.3396 | 0.3439 | 0.3507 |
| ctgan_seed2 | 0.3016 | 0.325 | 0.3131 | 0.3191 | 0.3178 |
| dpctgan_eps10_seed0 | - | 0.8246 | - | - | 0.8341 |
| dpctgan_eps15_seed0 | 0.8492 | - | - | - | 0.8451 |
| dpctgan_eps15_seed1 | 0.8629 | - | - | - | 0.8673 |
| dpctgan_eps15_seed2 | 0.8324 | - | 0.8231 | - | - |
| dpctgan_eps1_seed0 | 0.7355 | - | 0.7274 | 0.7219 | 0.7358 |
| dpctgan_eps20_seed0 | 0.9025 | 0.8968 | - | - | 0.9048 |
| dpctgan_eps5_seed0 | - | 0.8899 | 0.8773 | - | - |
| dpctgan_eps8_seed0 | - | 0.8293 | 0.8296 | 0.8393 | 0.8477 |
| gaussian_copula_seed0 | 0.4419 | 0.4376 | 0.4462 | 0.4391 | 0.4285 |
| gaussian_copula_seed1 | 0.4396 | 0.4476 | 0.4497 | 0.4458 | 0.438 |
| gaussian_copula_seed2 | 0.4346 | 0.4248 | 0.432 | 0.4212 | 0.4436 |
| mst_eps10_seed0 | 0.4353 | 0.4272 | 0.4906 | 0.511 | 0.4893 |
| mst_eps15_seed0 | 0.4295 | 0.4237 | 0.4958 | 0.473 | 0.4744 |
| mst_eps15_seed1 | 0.4328 | 0.4263 | 0.567 | 0.4918 | 0.4821 |
| mst_eps15_seed2 | 0.4281 | 0.4272 | 0.5194 | 0.4634 | 0.4787 |
| mst_eps1_seed0 | 0.4995 | 0.4951 | 0.5703 | 0.5555 | 0.5737 |
| mst_eps20_seed0 | 0.4332 | 0.4242 | 0.5989 | 0.4754 | 0.4784 |
| mst_eps5_seed0 | 0.4518 | 0.4417 | 0.4896 | 0.4633 | 0.4851 |
| mst_eps8_seed0 | 0.4461 | 0.4294 | 0.5291 | 0.4634 | 0.4759 |
| tvae_seed0 | 0.2423 | 0.2058 | 0.2162 | 0.2083 | 0.2239 |
| tvae_seed1 | 0.232 | 0.2186 | 0.2184 | 0.2187 | 0.2235 |
| tvae_seed2 | 0.2481 | 0.2108 | 0.2202 | 0.2113 | 0.2211 |

## Generalization (holdout vs synthetic)

Distance to real records the generator NEVER saw. A model that is much closer to train than to holdout is fitting its training sample, not the population.

| run | KS mean (train) | KS mean (holdout) | TVD mean (train) | TVD mean (holdout) |
|---|---|---|---|---|
| aim50_eps1_seed0 | 0.3256 | 0.3223 | 0.0175 | 0.0255 |
| aim50_eps5_seed0 | 0.3148 | 0.3106 | 0.0045 | 0.0174 |
| ctgan_seed0 | 0.2957 | 0.2997 | 0.1053 | 0.1066 |
| ctgan_seed1 | 0.3318 | 0.339 | 0.0794 | 0.0801 |
| ctgan_seed2 | 0.3009 | 0.3054 | 0.0759 | 0.0793 |
| dpctgan_eps10_seed0 | 0.8244 | 0.8242 | 0.324 | 0.324 |
| dpctgan_eps15_seed0 | 0.8419 | 0.8457 | 0.3093 | 0.3103 |
| dpctgan_eps15_seed1 | 0.8623 | 0.862 | 0.3084 | 0.3082 |
| dpctgan_eps15_seed2 | 0.8281 | 0.831 | 0.327 | 0.3275 |
| dpctgan_eps1_seed0 | 0.7261 | 0.7232 | 0.3054 | 0.3062 |
| dpctgan_eps20_seed0 | 0.8974 | 0.8997 | 0.3468 | 0.3468 |
| dpctgan_eps5_seed0 | 0.8934 | 0.8954 | 0.3361 | 0.3363 |
| dpctgan_eps8_seed0 | 0.8343 | 0.8353 | 0.3406 | 0.3399 |
| gaussian_copula_seed0 | 0.424 | 0.4267 | 0.0093 | 0.0145 |
| gaussian_copula_seed1 | 0.4396 | 0.4423 | 0.0088 | 0.0144 |
| gaussian_copula_seed2 | 0.4356 | 0.4387 | 0.009 | 0.0142 |
| mst_eps10_seed0 | 0.409 | 0.408 | 0.0027 | 0.0103 |
| mst_eps15_seed0 | 0.4048 | 0.4049 | 0.0018 | 0.0102 |
| mst_eps15_seed1 | 0.4061 | 0.4026 | 0.0017 | 0.0102 |
| mst_eps15_seed2 | 0.4074 | 0.4036 | 0.0018 | 0.0103 |
| mst_eps1_seed0 | 0.4863 | 0.4875 | 0.0242 | 0.0255 |
| mst_eps20_seed0 | 0.407 | 0.4055 | 0.0014 | 0.01 |
| mst_eps5_seed0 | 0.4238 | 0.4247 | 0.0051 | 0.0117 |
| mst_eps8_seed0 | 0.4189 | 0.4173 | 0.0032 | 0.0107 |
| tvae_seed0 | 0.1977 | 0.2054 | 0.055 | 0.056 |
| tvae_seed1 | 0.2032 | 0.2096 | 0.0589 | 0.0599 |
| tvae_seed2 | 0.204 | 0.206 | 0.0625 | 0.0634 |

## Association structure (train vs synthetic)

Absolute change in pairwise association; 0 = relationship perfectly preserved. `fabricated` counts pairs nearly independent in real data (|assoc|<0.1) rendered strongly associated (>0.5) in the synthetic data. Noise floor rows show how much two real samples differ.

| run | pair type | pairs | mean \|Δ\| | median \|Δ\| | \|Δ\|<0.1 | fabricated | worst pair |
|---|---|---|---|---|---|---|---|
| *noise floor* | Spearman (num-num) | 1621 | 0.0649 | 0.0453 | 0.7964 | 0 | - |
| *noise floor* | Cramer's V (cat-cat) | 11175 | 0.0222 | 0.0177 | 0.9975 | 0 | - |
| *noise floor* | corr-ratio (num-cat) | 11468 | 0.0278 | 0.0169 | 0.9613 | 0 | - |
| aim50_eps1_seed0 | Spearman (num-num) | 210 | 0.1498 | 0.1062 | 0.4667 | 0 | `vital_signs_weight_value_p6mo_first|vital_signs_bmi_value_last` (0.8066 -> -0.0314) |
| aim50_eps1_seed0 | Cramer's V (cat-cat) | 378 | 0.0997 | 0.054 | 0.7196 | 8 | `med_rasi|med_anti_coag_history` (0.0759 -> 0.812) |
| aim50_eps1_seed0 | corr-ratio (num-cat) | 1386 | 0.0394 | 0.0 | 0.8629 | 3 | `vital_signs_weight_value_p6mo_last|med_ll_history` (0.0114 -> 0.5705) |
| aim50_eps5_seed0 | Spearman (num-num) | 210 | 0.3247 | 0.2695 | 0.181 | 22 | `lab_results_crpNonHs_value_last|lab_results_albuminBS_value_last` (-0.441 -> 0.4541) |
| aim50_eps5_seed0 | Cramer's V (cat-cat) | 378 | 0.1496 | 0.0844 | 0.5476 | 22 | `conditions_mc|conditions_pad` (0.0197 -> 0.9298) |
| aim50_eps5_seed0 | corr-ratio (num-cat) | 1386 | 0.0765 | 0.0 | 0.7605 | 42 | `lab_results_ntProBnp_value_first|conditions_mc` (0.0135 -> 0.694) |
| ctgan_seed0 | Spearman (num-num) | 1601 | 0.1261 | 0.0905 | 0.5465 | 0 | `lab_results_ldl_value_last|lab_results_ldl_value_first` (0.9836 -> -0.1046) |
| ctgan_seed0 | Cramer's V (cat-cat) | 11026 | 0.0743 | 0.0309 | 0.7734 | 0 | `med_anti_plat_history|med_platelet_history` (0.9936 -> 0.0032) |
| ctgan_seed0 | corr-ratio (num-cat) | 11220 | 0.0424 | 0.0222 | 0.8953 | 0 | `lab_results_valideGFR_value_last|ckd_severity_from_calculated_egfr` (0.958 -> 0.1141) |
| ctgan_seed1 | Spearman (num-num) | 1637 | 0.1266 | 0.0917 | 0.5357 | 0 | `lab_results_tropTHs_value_last|lab_results_tropTnHs_value_last` (0.9994 -> -0.0712) |
| ctgan_seed1 | Cramer's V (cat-cat) | 11026 | 0.0666 | 0.0248 | 0.8262 | 0 | `smoking_status_smoker_last|smoking_status_formerSmoker_last` (1.0 -> 0.012) |
| ctgan_seed1 | corr-ratio (num-cat) | 11407 | 0.0402 | 0.0199 | 0.9077 | 0 | `lab_results_valideGFR_value_last|ckd_severity_from_calculated_egfr` (0.958 -> 0.0625) |
| ctgan_seed2 | Spearman (num-num) | 1603 | 0.1196 | 0.082 | 0.582 | 0 | `lab_results_tropTHs_value_first|lab_results_tropTnHs_value_first` (1.0 -> -0.1228) |
| ctgan_seed2 | Cramer's V (cat-cat) | 11026 | 0.0734 | 0.0352 | 0.7568 | 0 | `smoking_status_smoker_last|smoking_status_formerSmoker_last` (1.0 -> 0.0327) |
| ctgan_seed2 | corr-ratio (num-cat) | 11220 | 0.0424 | 0.0226 | 0.8946 | 1 | `lab_results_valideGFR_value_last|ckd_severity_calculated_or_measured` (0.9616 -> 0.1461) |
| dpctgan_eps10_seed0 | Spearman (num-num) | 325 | 0.2953 | 0.2154 | 0.2646 | 31 | `vital_signs_weight_value_p6mo_last|vital_signs_weight_value_last` (1.0 -> -0.344) |
| dpctgan_eps10_seed0 | Cramer's V (cat-cat) | 8778 | 0.0934 | 0.0406 | 0.8058 | 1 | `cause_of_death_isRenal_f5a_w1a_first|cause_of_death_isNonRenalAndNonCV_f5a_w1a_first` (1.0 -> 0.0006) |
| dpctgan_eps10_seed0 | corr-ratio (num-cat) | 4862 | 0.0444 | 0.0228 | 0.8778 | 0 | `lab_results_valideGFR_value_last|ckd_severity_from_calculated_egfr` (0.958 -> 0.0) |
| dpctgan_eps15_seed0 | Spearman (num-num) | 405 | 0.3973 | 0.3559 | 0.2198 | 76 | `lab_results_validSerumCreatinine_value_first|lab_results_valideGFR_value_first` (-0.9057 -> 0.8801) |
| dpctgan_eps15_seed0 | Cramer's V (cat-cat) | 7626 | 0.1046 | 0.0481 | 0.7714 | 2 | `cause_of_death_isCV_f5a_w3a_first|cause_of_death_isNonRenalAndNonCV_f5a_w3a_first` (1.0 -> 0.0003) |
| dpctgan_eps15_seed0 | corr-ratio (num-cat) | 5423 | 0.0438 | 0.0249 | 0.88 | 0 | `lab_results_valideGFR_value_last|ckd_severity_categorizedValue` (0.9716 -> 0.0269) |
| dpctgan_eps15_seed1 | Spearman (num-num) | 375 | 0.314 | 0.1909 | 0.2747 | 58 | `lab_results_validSerumCreatinine_value_last|eGFR_2021_ckd_epi_creatinine` (-0.9249 -> 0.9028) |
| dpctgan_eps15_seed1 | Cramer's V (cat-cat) | 7875 | 0.0958 | 0.0412 | 0.7947 | 0 | `cause_of_death_isNonRenalAndNonCV_f5a_w3mo_first|cause_of_death_isAllCause_f5a_w3mo_first` (1.0 -> 0.0003) |
| dpctgan_eps15_seed1 | corr-ratio (num-cat) | 5236 | 0.0483 | 0.0261 | 0.8665 | 0 | `lab_results_valideGFR_value_last|ckd_severity_from_calculated_egfr` (0.958 -> 0.0178) |
| dpctgan_eps15_seed2 | Spearman (num-num) | 252 | 0.2577 | 0.1608 | 0.3175 | 31 | `lab_results_validSerumCreatinine_value_last|lab_results_valideGFR_value_last` (-0.9109 -> 0.1003) |
| dpctgan_eps15_seed2 | Cramer's V (cat-cat) | 8128 | 0.0911 | 0.0402 | 0.8031 | 2 | `cause_of_death_isCV_f5a_w1a_first|cause_of_death_isNonRenalAndNonCV_f5a_w1a_first` (1.0 -> 0.0003) |
| dpctgan_eps15_seed2 | corr-ratio (num-cat) | 4301 | 0.0454 | 0.022 | 0.8761 | 0 | `lab_results_valideGFR_value_last|ckd_severity_calculated_or_measured` (0.9616 -> 0.0) |
| dpctgan_eps1_seed0 | Spearman (num-num) | 378 | 0.2462 | 0.1586 | 0.3439 | 38 | `lab_results_validSerumCreatinine_value_last|eGFR_2021_ckd_epi_creatinine` (-0.9249 -> 0.0928) |
| dpctgan_eps1_seed0 | Cramer's V (cat-cat) | 9730 | 0.0935 | 0.0384 | 0.8051 | 0 | `cause_of_death_isRenal_f5a_w7d_first|cause_of_death_isNonRenalAndNonCV_f5a_w7d_first` (1.0 -> 0.0003) |
| dpctgan_eps1_seed0 | corr-ratio (num-cat) | 5236 | 0.0391 | 0.0205 | 0.9089 | 0 | `lab_results_valideGFR_value_last|ckd_severity_categorizedValue` (0.9716 -> 0.035) |
| dpctgan_eps20_seed0 | Spearman (num-num) | 252 | 0.3793 | 0.3098 | 0.2183 | 57 | `lab_results_validSerumCreatinine_value_first|lab_results_valideGFR_value_last` (-0.7136 -> 0.7629) |
| dpctgan_eps20_seed0 | Cramer's V (cat-cat) | 8385 | 0.089 | 0.0401 | 0.8038 | 1 | `cause_of_death_isCV_f5a_w7d_first|cause_of_death_isNonRenalAndNonCV_f5a_w7d_first` (1.0 -> 0.0003) |
| dpctgan_eps20_seed0 | corr-ratio (num-cat) | 4301 | 0.0454 | 0.0235 | 0.8675 | 0 | `lab_results_valideGFR_value_last|ckd_severity_categorizedValue` (0.9716 -> 0.0187) |
| dpctgan_eps5_seed0 | Spearman (num-num) | 346 | 0.2891 | 0.1888 | 0.2977 | 46 | `lab_results_validSerumCreatinine_value_last|lab_results_valideGFR_value_first` (-0.7271 -> 0.7637) |
| dpctgan_eps5_seed0 | Cramer's V (cat-cat) | 6903 | 0.103 | 0.0441 | 0.7821 | 2 | `cause_of_death_isCV_f5a_w1a_first|cause_of_death_isNonRenalAndNonCV_f5a_w1a_first` (1.0 -> 0.0003) |
| dpctgan_eps5_seed0 | corr-ratio (num-cat) | 5049 | 0.0462 | 0.0249 | 0.879 | 0 | `lab_results_valideGFR_value_last|ckd_severity_calculated_or_measured` (0.9616 -> 0.0) |
| dpctgan_eps8_seed0 | Spearman (num-num) | 378 | 0.3812 | 0.3452 | 0.1746 | 71 | `lab_results_validSerumCreatinine_value_first|lab_results_valideGFR_value_first` (-0.9057 -> 0.703) |
| dpctgan_eps8_seed0 | Cramer's V (cat-cat) | 7626 | 0.1004 | 0.0436 | 0.7857 | 0 | `cause_of_death_isCV_f5a_w6mo_first|cause_of_death_isAllCause_f5a_w6mo_first` (1.0 -> 0.0004) |
| dpctgan_eps8_seed0 | corr-ratio (num-cat) | 5236 | 0.0454 | 0.0241 | 0.8673 | 0 | `eGFR_2021_ckd_epi_creatinine|ckd_severity_from_calculated_egfr` (0.9497 -> 0.0333) |
| gaussian_copula_seed0 | Spearman (num-num) | 973 | 0.1068 | 0.0745 | 0.6177 | 1 | `lab_results_validSerumCreatinine_value_last|eGFR_2021_ckd_epi_creatinine` (-0.9249 -> -0.0004) |
| gaussian_copula_seed0 | Cramer's V (cat-cat) | 11026 | 0.0671 | 0.0253 | 0.8507 | 0 | `cause_of_death_isCV_f5a_w7d_first|cause_of_death_isRenal_f5a_w7d_first` (1.0 -> 0.0115) |
| gaussian_copula_seed0 | corr-ratio (num-cat) | 8415 | 0.0376 | 0.0194 | 0.9181 | 0 | `lab_results_valideGFR_value_last|ckd_severity_from_calculated_egfr` (0.958 -> 0.2435) |
| gaussian_copula_seed1 | Spearman (num-num) | 996 | 0.1109 | 0.0724 | 0.6235 | 1 | `lab_results_validSerumCreatinine_value_last|eGFR_2021_ckd_epi_creatinine` (-0.9249 -> 0.0078) |
| gaussian_copula_seed1 | Cramer's V (cat-cat) | 11026 | 0.0671 | 0.0251 | 0.8531 | 0 | `cause_of_death_isRenal_f5a_w7d_first|cause_of_death_isNonRenalAndNonCV_f5a_w7d_first` (1.0 -> 0.0428) |
| gaussian_copula_seed1 | corr-ratio (num-cat) | 8602 | 0.0385 | 0.0196 | 0.9129 | 1 | `lab_results_valideGFR_value_last|ckd_severity_from_calculated_egfr` (0.958 -> 0.2272) |
| gaussian_copula_seed2 | Spearman (num-num) | 976 | 0.1106 | 0.0733 | 0.6291 | 1 | `lab_results_validSerumCreatinine_value_last|eGFR_2021_ckd_epi_creatinine` (-0.9249 -> 0.0042) |
| gaussian_copula_seed2 | Cramer's V (cat-cat) | 11026 | 0.067 | 0.0249 | 0.8534 | 0 | `cause_of_death_isCV_f5a_w7d_first|cause_of_death_isNonRenalAndNonCV_f5a_w7d_first` (1.0 -> 0.0202) |
| gaussian_copula_seed2 | corr-ratio (num-cat) | 8415 | 0.0381 | 0.02 | 0.9155 | 0 | `encounter_primary_reason_number_of_days_to_rehosp_for_non_CV_f5a_first|encounter_primary_reason_HF_Disease_f5a_w1a_first` (0.8476 -> 0.0454) |
| mst_eps10_seed0 | Spearman (num-num) | 1727 | 0.2803 | 0.2438 | 0.2003 | 138 | `vital_signs_weight_value_p6mo_first|lab_results_ntProBnp_value_first` (-0.2346 -> 0.8241) |
| mst_eps10_seed0 | Cramer's V (cat-cat) | 11026 | 0.2437 | 0.1985 | 0.3026 | 1639 | `conditions_heart_failure_hf_within_18mo_any|conditions_hf` (0.9533 -> 0.0055) |
| mst_eps10_seed0 | corr-ratio (num-cat) | 11220 | 0.1659 | 0.1007 | 0.4988 | 960 | `electrocardiographs_ecg_qrs_duration_pET_last|conditions_mi` (0.0328 -> 0.8458) |
| mst_eps15_seed0 | Spearman (num-num) | 1753 | 0.2829 | 0.2473 | 0.2196 | 175 | `lab_results_tropTHs_value_last|lab_results_tropTnHs_value_last` (0.9994 -> -0.0041) |
| mst_eps15_seed0 | Cramer's V (cat-cat) | 11026 | 0.2431 | 0.209 | 0.2889 | 1413 | `med_arni|med_antiarrhytmic_history` (0.033 -> 0.9833) |
| mst_eps15_seed0 | corr-ratio (num-cat) | 11407 | 0.1669 | 0.1063 | 0.4886 | 988 | `encounter_primary_reason_number_of_days_to_rehosp_for_non_CV_f5a_first|encounter_primary_reason_non_CV_Disease_f5a_w1a_first` (0.8476 -> 0.0082) |
| mst_eps15_seed1 | Spearman (num-num) | 1737 | 0.2943 | 0.2579 | 0.1865 | 186 | `lab_results_tropTnHs_value_first|encounter_primary_reason_number_of_days_to_rehosp_for_CV_f5a_first` (-0.0518 -> 0.9998) |
| mst_eps15_seed1 | Cramer's V (cat-cat) | 11026 | 0.2477 | 0.2098 | 0.2784 | 1557 | `conditions_heart_failure_hf_within_18mo_any|conditions_hf` (0.9533 -> 0.0027) |
| mst_eps15_seed1 | corr-ratio (num-cat) | 11407 | 0.175 | 0.1119 | 0.4767 | 1088 | `smoking_status_smoker_startTime_count|encounter_primary_reason_renal_complications_f5a_w1a_first` (0.0343 -> 0.8611) |
| mst_eps15_seed2 | Spearman (num-num) | 1708 | 0.2961 | 0.2667 | 0.1915 | 178 | `vital_signs_weight_value_p6mo_first|lab_results_hdl_value_first` (-0.2677 -> 0.6839) |
| mst_eps15_seed2 | Cramer's V (cat-cat) | 11026 | 0.2536 | 0.2059 | 0.2774 | 1837 | `med_arb|med_inotropes_history` (0.0077 -> 0.9676) |
| mst_eps15_seed2 | corr-ratio (num-cat) | 11220 | 0.1748 | 0.112 | 0.4784 | 1078 | `encounter_primary_reason_number_of_days_to_rehosp_for_non_CV_f5a_first|med_digitalis_history` (0.0335 -> 0.8418) |
| mst_eps1_seed0 | Spearman (num-num) | 1500 | 0.2581 | 0.2032 | 0.298 | 86 | `lab_results_cholTot_value_first|lab_results_ldl_value_last` (0.8631 -> -0.457) |
| mst_eps1_seed0 | Cramer's V (cat-cat) | 11026 | 0.2911 | 0.2661 | 0.2479 | 2387 | `med_rdoad_syst_history|conditions_aidshiv` (0.0069 -> 1.0) |
| mst_eps1_seed0 | corr-ratio (num-cat) | 11033 | 0.1874 | 0.1048 | 0.4913 | 1074 | `lab_results_tropTHs_value_last|conditions_stroke` (0.0001 -> 1.9189) |
| mst_eps20_seed0 | Spearman (num-num) | 1759 | 0.3016 | 0.2707 | 0.1956 | 195 | `lab_results_potassium_value_last|lab_results_valideGFR_value_first` (-0.1994 -> 0.792) |
| mst_eps20_seed0 | Cramer's V (cat-cat) | 11026 | 0.2411 | 0.1839 | 0.3209 | 1712 | `med_arb|med_inotropes_history` (0.0077 -> 0.9613) |
| mst_eps20_seed0 | corr-ratio (num-cat) | 11407 | 0.1792 | 0.113 | 0.474 | 1126 | `encounter_primary_reason_number_of_days_to_rehosp_for_heart_failure_f5a_first|conditions_mi` (0.012 -> 0.8637) |
| mst_eps5_seed0 | Spearman (num-num) | 1758 | 0.2517 | 0.2043 | 0.2435 | 110 | `lab_results_hdl_value_last|lab_results_hdl_value_first` (0.9946 -> -0.0559) |
| mst_eps5_seed0 | Cramer's V (cat-cat) | 11026 | 0.2615 | 0.2266 | 0.2724 | 1784 | `conditions_heart_failure_hf_within_18mo_any|conditions_hf` (0.9533 -> 0.01) |
| mst_eps5_seed0 | corr-ratio (num-cat) | 11407 | 0.1552 | 0.0947 | 0.5111 | 745 | `encounter_primary_reason_number_of_days_to_rehosp_for_heart_failure_f5a_first|encounter_primary_reason_HF_Disease_f5a_w5a_first` (0.0 -> 0.9426) |
| mst_eps8_seed0 | Spearman (num-num) | 1768 | 0.2692 | 0.2214 | 0.2364 | 161 | `encounter_primary_reason_number_of_days_to_rehosp_for_heart_failure_f5a_first|encounter_primary_reason_number_of_days_to_rehosp_for_CV_f5a_first` (1.0 -> -0.1765) |
| mst_eps8_seed0 | Cramer's V (cat-cat) | 11026 | 0.2534 | 0.2034 | 0.2897 | 1821 | `conditions_heart_failure_hf_within_18mo_any|conditions_hf` (0.9533 -> 0.0015) |
| mst_eps8_seed0 | corr-ratio (num-cat) | 11407 | 0.1691 | 0.0984 | 0.5026 | 1038 | `encounter_primary_reason_number_of_days_to_rehosp_for_heart_failure_f5a_first|conditions_mi` (0.012 -> 0.8517) |
| tvae_seed0 | Spearman (num-num) | 1592 | 0.0861 | 0.0607 | 0.7067 | 0 | `echocardiographs_lvef_pET_last|echocardiographs_lvef_pET_first` (0.8998 -> -0.0174) |
| tvae_seed0 | Cramer's V (cat-cat) | 8001 | 0.0585 | 0.0343 | 0.8245 | 1 | `conditions_ap|conditions_dysl` (0.1435 -> 0.6727) |
| tvae_seed0 | corr-ratio (num-cat) | 11033 | 0.0412 | 0.0226 | 0.8892 | 10 | `lab_results_creatUS_value_last|encounter_primary_reason_HF_Disease_f5a_w5a_first` (0.1061 -> 0.8403) |
| tvae_seed1 | Spearman (num-num) | 1610 | 0.0817 | 0.0569 | 0.7093 | 1 | `vital_signs_height_value_p1a_avg|vital_signs_height_value_last` (0.989 -> 0.1661) |
| tvae_seed1 | Cramer's V (cat-cat) | 8385 | 0.0617 | 0.0369 | 0.8109 | 3 | `med_acei_history|med_arb_history` (0.007 -> 0.6323) |
| tvae_seed1 | corr-ratio (num-cat) | 11033 | 0.0408 | 0.0224 | 0.8966 | 13 | `encounter_primary_reason_number_of_days_to_rehosp_for_heart_failure_f5a_first|encounter_primary_reason_CV_Disease_f5a_w3a_first` (0.8082 -> 0.0) |
| tvae_seed2 | Spearman (num-num) | 1549 | 0.0827 | 0.0588 | 0.7069 | 0 | `echocardiographs_lvef_pET_last|echocardiographs_lvef_pET_first` (0.8998 -> 0.0484) |
| tvae_seed2 | Cramer's V (cat-cat) | 8128 | 0.0556 | 0.0304 | 0.8343 | 1 | `conditions_ap|conditions_dysl` (0.1435 -> 0.7761) |
| tvae_seed2 | corr-ratio (num-cat) | 10846 | 0.0392 | 0.0211 | 0.9015 | 8 | `nyha_nyha_pET|med_arb` (0.0696 -> 0.7599) |

## original vs preprocessed

Worst numeric columns (by KS):
- `patient_demographics_age`: KS=0.0, W/std=0.0, mean 70.9842 -> 70.9842, missing 0% -> 0%
- `encounters_lengthOfStay`: KS=0.0, W/std=0.0, mean 10.5254 -> 10.5254, missing 0% -> 0%
- `encounters_numOfPreviousHFStays_count`: KS=0.0, W/std=0.0, mean 52.5614 -> 52.5614, missing 0% -> 0%
- `vital_signs_weight_value_p6mo_last`: KS=0.0, W/std=0.0, mean 77.4701 -> 77.4701, missing 8% -> 8%
- `vital_signs_weight_value_p6mo_first`: KS=0.0, W/std=0.0, mean 79.7398 -> 79.7398, missing 8% -> 8%
Worst categorical columns (by TVD):
- `patient_demographics_gender`: TVD=0.0, 2 -> 2 categories, missing 0% -> 0%
- `encounters_encounterClass`: TVD=0.0, 1 -> 1 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.0, 10 -> 10 categories, missing 0% -> 0%
- `symptoms_Ankle_swelling_display_pET_any`: TVD=0.0, 1 -> 1 categories, missing 0% -> 0%
- `symptoms_Ascites_display_pET_any`: TVD=0.0, 1 -> 1 categories, missing 0% -> 0%

## train vs holdout

Worst numeric columns (by KS):
- `encounter_primary_reason_number_of_days_to_rehosp_for_heart_failure_f5a_first`: KS=0.1802, W/std=0.2533, mean 82.6105 -> 120.1538, missing 95% -> 96%
- `lab_results_creatUS_value_first`: KS=0.1225, W/std=0.1885, mean 754.8314 -> 672.9652, missing 90% -> 89%
- `lab_results_triGly_value_last`: KS=0.102, W/std=0.1875, mean 1.5467 -> 1.3626, missing 90% -> 92%
- `lab_results_triGly_value_first`: KS=0.102, W/std=0.1905, mean 1.5982 -> 1.3738, missing 90% -> 92%
- `lab_results_cholTot_value_last`: KS=0.0936, W/std=0.1611, mean 3.9492 -> 3.8205, missing 89% -> 90%
Worst categorical columns (by TVD):
- `med_mra_history`: TVD=0.0489, 2 -> 2 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.042, 7 -> 7 categories, missing 0% -> 0%
- `conditions_ckd_chronic`: TVD=0.0333, 2 -> 2 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.0328, 6 -> 6 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.0322, 7 -> 7 categories, missing 0% -> 0%

## train vs synthetic[aim50_eps1_seed0]

Worst numeric columns (by KS):
- `encounters_lengthOfStay`: KS=0.7531, W/std=0.9907, mean 10.5818 -> 23.4364, missing 0% -> 0%
- `lab_results_crpNonHs_value_last`: KS=0.6775, W/std=0.5972, mean 43.6592 -> 77.1504, missing 11% -> 8%
- `lab_results_crpNonHs_value_first`: KS=0.6207, W/std=0.3345, mean 45.8171 -> 67.6754, missing 11% -> 12%
- `encounters_numOfPreviousHFStays_count`: KS=0.5443, W/std=0.2855, mean 51.846 -> 60.7628, missing 0% -> 0%
- `lab_results_ntProBnp_value_last`: KS=0.4938, W/std=0.1753, mean 10078.8465 -> 11150.0828, missing 19% -> 20%
Worst categorical columns (by TVD):
- `ckd_severity_from_calculated_egfr`: TVD=0.1406, 6 -> 6 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.0642, 10 -> 9 categories, missing 0% -> 0%
- `hyperkalemia_severity_categorizedValue`: TVD=0.0386, 5 -> 5 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.0312, 7 -> 7 categories, missing 0% -> 0%
- `conditions_vd`: TVD=0.0253, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[aim50_eps5_seed0]

Worst numeric columns (by KS):
- `encounters_lengthOfStay`: KS=0.7531, W/std=0.6178, mean 10.5818 -> 17.48, missing 0% -> 0%
- `lab_results_crpNonHs_value_last`: KS=0.6775, W/std=0.3968, mean 43.6592 -> 62.0155, missing 11% -> 11%
- `lab_results_crpNonHs_value_first`: KS=0.6207, W/std=0.2708, mean 45.8171 -> 58.1664, missing 11% -> 11%
- `encounters_numOfPreviousHFStays_count`: KS=0.5443, W/std=0.27, mean 51.846 -> 63.3586, missing 0% -> 0%
- `lab_results_ntProBnp_value_last`: KS=0.4938, W/std=0.1641, mean 10078.8465 -> 10912.3524, missing 19% -> 19%
Worst categorical columns (by TVD):
- `ace_inhibitors_arb_use_pre_dc`: TVD=0.0227, 2 -> 2 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.0168, 7 -> 7 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.0099, 10 -> 10 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.0099, 7 -> 7 categories, missing 0% -> 0%
- `hyperkalemia_severity_categorizedValue`: TVD=0.0088, 5 -> 5 categories, missing 0% -> 0%

## train vs synthetic[ctgan_seed0]

Worst numeric columns (by KS):
- `lab_results_tropTHs_value_last`: KS=0.6847, W/std=0.3316, mean 0.4652 -> 1.0607, missing 63% -> 56%
- `echocardiographs_lvef_pET_last`: KS=0.6492, W/std=1.818, mean 40.6716 -> 76.0297, missing 83% -> 90%
- `lab_results_tropTnHs_value_last`: KS=0.6124, W/std=0.5346, mean 281.7666 -> 694.2468, missing 88% -> 92%
- `lab_results_ferritin_value_first`: KS=0.602, W/std=0.4291, mean 561.1328 -> 1092.2969, missing 80% -> 74%
- `echocardiographs_lvef_pET_first`: KS=0.5964, W/std=1.7063, mean 40.9911 -> 9.912, missing 82% -> 92%
Worst categorical columns (by TVD):
- `med_acei_history`: TVD=0.3281, 2 -> 2 categories, missing 0% -> 0%
- `conditions_heart_failure_occurred_prior_to_18_months_any`: TVD=0.2869, 2 -> 2 categories, missing 0% -> 0%
- `cause_of_death_isAllCause_f5a_w5a_first`: TVD=0.2443, 3 -> 3 categories, missing 0% -> 0%
- `cause_of_death_isRenal_f5a_w1a_first`: TVD=0.2409, 2 -> 2 categories, missing 0% -> 0%
- `med_insulins`: TVD=0.2392, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[ctgan_seed1]

Worst numeric columns (by KS):
- `lab_results_tropTHs_value_last`: KS=0.7126, W/std=0.4801, mean 0.4652 -> 1.517, missing 63% -> 81%
- `lab_results_sodium_value_first`: KS=0.6322, W/std=1.4343, mean 137.0846 -> 144.2849, missing 4% -> 7%
- `lab_results_tropTnHs_value_last`: KS=0.6255, W/std=0.4769, mean 281.7666 -> 629.3572, missing 88% -> 85%
- `electrocardiographs_ecg_qt_duration_corrected_pET_first`: KS=0.6235, W/std=1.8271, mean 471.831 -> 380.6963, missing 49% -> 38%
- `encounter_primary_reason_number_of_days_to_rehosp_for_heart_failure_f5a_first`: KS=0.5724, W/std=0.5173, mean 82.6105 -> 129.1957, missing 95% -> 97%
Worst categorical columns (by TVD):
- `med_rdoad`: TVD=0.2972, 2 -> 2 categories, missing 0% -> 0%
- `patient_demographics_gender`: TVD=0.2966, 2 -> 2 categories, missing 0% -> 0%
- `med_rasi_history`: TVD=0.2719, 2 -> 2 categories, missing 0% -> 0%
- `med_anti_plat_history`: TVD=0.2585, 2 -> 2 categories, missing 0% -> 0%
- `conditions_diabetes`: TVD=0.2335, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[ctgan_seed2]

Worst numeric columns (by KS):
- `lab_results_tropTnHs_value_last`: KS=0.7336, W/std=0.7409, mean 281.7666 -> 880.8121, missing 88% -> 97%
- `lab_results_tropTHs_value_first`: KS=0.6502, W/std=0.4395, mean 0.2171 -> 0.5141, missing 63% -> 41%
- `lab_results_tropTnHs_value_first`: KS=0.5756, W/std=0.586, mean 212.254 -> 475.604, missing 88% -> 97%
- `lab_results_potassium_value_last`: KS=0.5726, W/std=1.4326, mean 4.1751 -> 3.34, missing 4% -> 8%
- `lab_results_tropTHs_value_last`: KS=0.5679, W/std=0.2706, mean 0.4652 -> 0.9749, missing 63% -> 98%
Worst categorical columns (by TVD):
- `med_anti_plat`: TVD=0.3386, 2 -> 2 categories, missing 0% -> 0%
- `patient_demographics_gender`: TVD=0.3125, 2 -> 2 categories, missing 0% -> 0%
- `conditions_mi`: TVD=0.231, 2 -> 2 categories, missing 0% -> 0%
- `med_vasodil_history`: TVD=0.229, 2 -> 2 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.2179, 6 -> 6 categories, missing 0% -> 0%

## train vs synthetic[dpctgan_eps10_seed0]

Worst numeric columns (by KS):
- `vital_signs_weight_value_p6mo_first`: KS=0.9997, W/std=7.9541, mean 79.6287 -> 239.9866, missing 8% -> 0%
- `vital_signs_weight_value_last`: KS=0.9997, W/std=6.5285, mean 77.3464 -> 206.8234, missing 15% -> 0%
- `lab_results_hemoglobin_value_first`: KS=0.9997, W/std=3.6299, mean 121.9202 -> 209.481, missing 3% -> 0%
- `eGFR_2021_ckd_epi_creatinine`: KS=0.9997, W/std=4.9797, mean 64.4886 -> 193.7614, missing 10% -> 0%
- `electrocardiographs_ecg_qrs_duration_pET_last`: KS=0.9994, W/std=3.8078, mean 118.0915 -> 249.9691, missing 49% -> 0%
Worst categorical columns (by TVD):
- `cause_of_death_isAllCause_f5a_w1a_first`: TVD=0.9818, 3 -> 2 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.9787, 10 -> 9 categories, missing 0% -> 0%
- `cause_of_death_isCV_f5a_w3a_first`: TVD=0.9753, 3 -> 3 categories, missing 0% -> 0%
- `cause_of_death_isCV_f5a_w7d_first`: TVD=0.9651, 3 -> 3 categories, missing 0% -> 0%
- `conditions_substance_abuse`: TVD=0.8531, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[dpctgan_eps15_seed0]

Worst numeric columns (by KS):
- `vital_signs_diastolicBp_value_first`: KS=0.9997, W/std=6.795, mean 75.7013 -> 200.9986, missing 5% -> 0%
- `vital_signs_diastolicBp_value_last`: KS=0.9997, W/std=5.478, mean 69.0778 -> 143.3624, missing 5% -> 0%
- `lab_results_hemoglobin_value_last`: KS=0.9997, W/std=3.9152, mean 120.0203 -> 209.4806, missing 3% -> 0%
- `lab_results_sodium_value_last`: KS=0.9997, W/std=4.7537, mean 137.8251 -> 159.6881, missing 4% -> 0%
- `lab_results_potassium_value_last`: KS=0.9994, W/std=3.7433, mean 4.1751 -> 1.9931, missing 4% -> 99%
Worst categorical columns (by TVD):
- `cause_of_death_isAllCause_f5a_w1a_first`: TVD=0.9798, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_HF_Disease_f5a_w1a_first`: TVD=0.948, 3 -> 2 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.9361, 6 -> 6 categories, missing 0% -> 0%
- `cause_of_death_isRenal_f5a_w1mo_first`: TVD=0.8776, 2 -> 2 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.8724, 7 -> 3 categories, missing 0% -> 0%

## train vs synthetic[dpctgan_eps15_seed1]

Worst numeric columns (by KS):
- `patient_demographics_age`: KS=0.9997, W/std=2.3963, mean 70.9054 -> 103.9997, missing 0% -> 0%
- `vital_signs_systolicBp_value_last`: KS=0.9997, W/std=4.4644, mean 121.0177 -> 220.9907, missing 5% -> 0%
- `lab_results_hemoglobin_value_last`: KS=0.9997, W/std=3.9147, mean 120.0203 -> 209.4687, missing 3% -> 0%
- `lab_results_sodium_value_last`: KS=0.9997, W/std=4.8134, mean 137.8251 -> 159.9627, missing 4% -> 0%
- `lab_results_validSerumCreatinine_value_last`: KS=0.9997, W/std=2.3005, mean 12.1376 -> 22.5738, missing 10% -> 0%
Worst categorical columns (by TVD):
- `conditions_aidshiv`: TVD=0.9903, 2 -> 2 categories, missing 0% -> 0%
- `conditions_pericardial`: TVD=0.9622, 2 -> 2 categories, missing 0% -> 0%
- `med_potassium_binders`: TVD=0.9616, 2 -> 2 categories, missing 0% -> 0%
- `conditions_dep`: TVD=0.9585, 2 -> 2 categories, missing 0% -> 0%
- `conditions_osa`: TVD=0.9293, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[dpctgan_eps15_seed2]

Worst numeric columns (by KS):
- `vital_signs_systolicBp_value_first`: KS=0.9997, W/std=4.3818, mean 130.3643 -> 256.9641, missing 5% -> 0%
- `vital_signs_diastolicBp_value_last`: KS=0.9997, W/std=5.5222, mean 69.0778 -> 143.9619, missing 5% -> 0%
- `lab_results_potassium_value_last`: KS=0.9997, W/std=5.3592, mean 4.1751 -> 7.299, missing 4% -> 0%
- `vital_signs_height_value_p1a_avg`: KS=0.9996, W/std=3.4543, mean 171.0332 -> 206.9075, missing 21% -> 0%
- `vital_signs_heartRate_value_first`: KS=0.9995, W/std=6.6802, mean 112.7017 -> 222.9731, missing 48% -> 0%
Worst categorical columns (by TVD):
- `conditions_ibd`: TVD=0.921, 2 -> 2 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.9009, 10 -> 2 categories, missing 0% -> 0%
- `encounter_primary_reason_renal_complications_f5a_w1mo_first`: TVD=0.8952, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_CV_Disease_f5a_w3mo_first`: TVD=0.8523, 3 -> 3 categories, missing 0% -> 0%
- `conditions_osa`: TVD=0.8511, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[dpctgan_eps1_seed0]

Worst numeric columns (by KS):
- `lab_results_hemoglobin_value_last`: KS=0.9997, W/std=3.9151, mean 120.0203 -> 209.4784, missing 3% -> 0%
- `lab_results_hemoglobin_value_first`: KS=0.9997, W/std=3.5884, mean 121.9202 -> 208.4801, missing 3% -> 0%
- `lab_results_potassium_value_last`: KS=0.9997, W/std=5.3507, mean 4.1751 -> 7.294, missing 4% -> 0%
- `lab_results_sodium_value_first`: KS=0.9997, W/std=4.3477, mean 137.0846 -> 158.9109, missing 4% -> 0%
- `vital_signs_heartRate_value_last`: KS=0.9995, W/std=9.6918, mean 109.7305 -> 250.303, missing 48% -> 0%
Worst categorical columns (by TVD):
- `encounter_primary_reason_renal_complications_f5a_w6mo_first`: TVD=0.9517, 3 -> 3 categories, missing 0% -> 0%
- `cause_of_death_isAllCause_f5a_w3a_first`: TVD=0.8872, 3 -> 3 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.8869, 10 -> 10 categories, missing 0% -> 0%
- `conditions_revasc`: TVD=0.808, 2 -> 2 categories, missing 0% -> 0%
- `med_potassium_binders`: TVD=0.7747, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[dpctgan_eps20_seed0]

Worst numeric columns (by KS):
- `patient_demographics_age`: KS=0.9997, W/std=2.3963, mean 70.9054 -> 103.999, missing 0% -> 0%
- `vital_signs_diastolicBp_value_first`: KS=0.9997, W/std=6.795, mean 75.7013 -> 200.9994, missing 5% -> 0%
- `lab_results_hemoglobin_value_last`: KS=0.9997, W/std=3.9005, mean 120.0203 -> 209.1447, missing 3% -> 0%
- `lab_results_potassium_value_last`: KS=0.9997, W/std=5.3415, mean 4.1751 -> 7.2886, missing 4% -> 0%
- `lab_results_sodium_value_last`: KS=0.9997, W/std=4.8214, mean 137.8251 -> 159.9997, missing 4% -> 0%
Worst categorical columns (by TVD):
- `cause_of_death_isAllCause_f5a_w1a_first`: TVD=0.9849, 3 -> 2 categories, missing 0% -> 0%
- `cause_of_death_isAllCause_f5a_w5a_first`: TVD=0.9693, 3 -> 3 categories, missing 0% -> 0%
- `smoking_status_formerSmoker_last`: TVD=0.944, 3 -> 2 categories, missing 0% -> 0%
- `med_antiarrhytmic_history`: TVD=0.9159, 2 -> 2 categories, missing 0% -> 0%
- `conditions_osa`: TVD=0.877, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[dpctgan_eps5_seed0]

Worst numeric columns (by KS):
- `vital_signs_diastolicBp_value_last`: KS=0.9997, W/std=5.5246, mean 69.0778 -> 143.9947, missing 5% -> 0%
- `lab_results_hemoglobin_value_first`: KS=0.9997, W/std=3.6124, mean 121.9202 -> 209.0587, missing 3% -> 0%
- `lab_results_sodium_value_last`: KS=0.9997, W/std=4.8162, mean 137.8251 -> 159.9757, missing 4% -> 0%
- `lab_results_sodium_value_first`: KS=0.9997, W/std=4.3616, mean 137.0846 -> 158.9807, missing 4% -> 0%
- `vital_signs_height_value_p1a_avg`: KS=0.9996, W/std=3.4631, mean 171.0332 -> 206.9997, missing 21% -> 0%
Worst categorical columns (by TVD):
- `encounter_primary_reason_non_CV_Disease_f5a_w7d_first`: TVD=0.9349, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first`: TVD=0.9102, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_non_CV_Disease_f5a_w1mo_first`: TVD=0.9074, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_HF_Disease_f5a_w5a_first`: TVD=0.9031, 3 -> 3 categories, missing 0% -> 0%
- `cause_of_death_isCV_f5a_w3mo_first`: TVD=0.871, 3 -> 1 categories, missing 0% -> 0%

## train vs synthetic[dpctgan_eps8_seed0]

Worst numeric columns (by KS):
- `vital_signs_weight_value_p6mo_last`: KS=0.9997, W/std=6.5384, mean 77.4079 -> 206.9955, missing 8% -> 0%
- `lab_results_hemoglobin_value_last`: KS=0.9997, W/std=3.9151, mean 120.0203 -> 209.4763, missing 3% -> 0%
- `lab_results_sodium_value_last`: KS=0.9997, W/std=4.8168, mean 137.8251 -> 159.9784, missing 4% -> 0%
- `lab_results_sodium_value_first`: KS=0.9997, W/std=4.3569, mean 137.0846 -> 158.9569, missing 4% -> 0%
- `eGFR_2021_ckd_epi_creatinine`: KS=0.9997, W/std=4.9736, mean 64.4886 -> 193.6024, missing 10% -> 0%
Worst categorical columns (by TVD):
- `hyperkalemia_severity_categorizedValue`: TVD=0.9497, 5 -> 1 categories, missing 0% -> 0%
- `conditions_pericardial`: TVD=0.948, 2 -> 2 categories, missing 0% -> 0%
- `cause_of_death_isNonRenalAndNonCV_f5a_w1mo_first`: TVD=0.9185, 2 -> 2 categories, missing 0% -> 0%
- `med_antiarrhytmic_history`: TVD=0.8739, 2 -> 2 categories, missing 0% -> 0%
- `cause_of_death_isAllCause_f5a_w3mo_first`: TVD=0.8707, 3 -> 2 categories, missing 0% -> 0%

## train vs synthetic[gaussian_copula_seed0]

Worst numeric columns (by KS):
- `lab_results_hdl_value_first`: KS=0.9906, W/std=2.1127, mean 1.1873 -> 0.2425, missing 91% -> 100%
- `smoking_status_smoker_startTime_count`: KS=0.9815, W/std=3.5091, mean 0.0804 -> 3.0497, missing 0% -> 0%
- `lab_results_creatUS_value_first`: KS=0.9288, W/std=1.1478, mean 754.8314 -> 147.4408, missing 90% -> 100%
- `echocardiographs_lvef_pET_first`: KS=0.8986, W/std=3.5865, mean 40.9911 -> -24.4573, missing 82% -> 97%
- `lab_results_tropTHs_value_last`: KS=0.8585, W/std=1.591, mean 0.4652 -> 4.2487, missing 63% -> 100%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.4401, 10 -> 9 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.0267, 6 -> 6 categories, missing 0% -> 0%
- `encounter_primary_reason_HF_Disease_f5a_w3mo_first`: TVD=0.0241, 3 -> 3 categories, missing 0% -> 0%
- `conditions_dysl`: TVD=0.0207, 2 -> 2 categories, missing 0% -> 0%
- `conditions_hyp`: TVD=0.0207, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[gaussian_copula_seed1]

Worst numeric columns (by KS):
- `lab_results_ldl_value_first`: KS=0.9937, W/std=1.933, mean 2.1313 -> 0.1964, missing 91% -> 100%
- `smoking_status_smoker_startTime_count`: KS=0.9815, W/std=3.5218, mean 0.0804 -> 3.0611, missing 0% -> 0%
- `lab_results_hdl_value_first`: KS=0.9781, W/std=1.7686, mean 1.1873 -> 0.3987, missing 91% -> 100%
- `echocardiographs_lvef_pET_first`: KS=0.9569, W/std=3.6837, mean 40.9911 -> -26.2312, missing 82% -> 97%
- `lab_results_creatUS_value_first`: KS=0.9402, W/std=1.2155, mean 754.8314 -> 111.2266, missing 90% -> 100%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.4435, 10 -> 9 categories, missing 0% -> 0%
- `med_ll_history`: TVD=0.0216, 2 -> 2 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.0205, 7 -> 7 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.0196, 6 -> 6 categories, missing 0% -> 0%
- `med_platelet_history`: TVD=0.0193, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[gaussian_copula_seed2]

Worst numeric columns (by KS):
- `lab_results_hdl_value_first`: KS=0.9938, W/std=2.2037, mean 1.1873 -> 0.2015, missing 91% -> 100%
- `smoking_status_smoker_startTime_count`: KS=0.9815, W/std=3.4054, mean 0.0804 -> 2.9619, missing 0% -> 0%
- `lab_results_ferritin_value_last`: KS=0.9341, W/std=0.6011, mean 523.1277 -> 1295.371, missing 80% -> 100%
- `echocardiographs_lvef_pET_first`: KS=0.9092, W/std=3.5778, mean 40.9911 -> -24.2982, missing 82% -> 97%
- `lab_results_cholTot_value_first`: KS=0.8622, W/std=1.9996, mean 3.9591 -> 1.6192, missing 89% -> 99%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.4287, 10 -> 9 categories, missing 0% -> 0%
- `med_mra`: TVD=0.0233, 2 -> 2 categories, missing 0% -> 0%
- `encounter_primary_reason_non_CV_Disease_f5a_w6mo_first`: TVD=0.0224, 3 -> 3 categories, missing 0% -> 0%
- `cause_of_death_isAllCause_f5a_w1a_first`: TVD=0.0224, 3 -> 3 categories, missing 0% -> 0%
- `patient_demographics_gender`: TVD=0.0219, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[mst_eps10_seed0]

Worst numeric columns (by KS):
- `smoking_status_smoker_startTime_count`: KS=0.9815, W/std=1.1679, mean 0.0804 -> 1.0687, missing 0% -> 0%
- `lab_results_tropTHs_value_last`: KS=0.9809, W/std=1.5874, mean 0.4652 -> 4.3833, missing 63% -> 63%
- `lab_results_ferritin_value_last`: KS=0.9742, W/std=1.4376, mean 523.1277 -> 3263.9866, missing 80% -> 80%
- `lab_results_ferritin_value_first`: KS=0.9713, W/std=1.1205, mean 561.1328 -> 2930.3177, missing 80% -> 80%
- `lab_results_tropTHs_value_first`: KS=0.9503, W/std=0.8484, mean 0.2171 -> 0.9009, missing 63% -> 63%
Worst categorical columns (by TVD):
- `ckd_severity_categorizedValue`: TVD=0.0156, 7 -> 7 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.0105, 10 -> 10 categories, missing 0% -> 0%
- `hyperkalemia_severity_categorizedValue`: TVD=0.0091, 5 -> 5 categories, missing 0% -> 0%
- `med_ivabradine`: TVD=0.0077, 2 -> 2 categories, missing 0% -> 0%
- `encounter_primary_reason_HF_Disease_f5a_w6mo_first`: TVD=0.0068, 3 -> 3 categories, missing 0% -> 0%

## train vs synthetic[mst_eps15_seed0]

Worst numeric columns (by KS):
- `smoking_status_smoker_startTime_count`: KS=0.9815, W/std=1.1362, mean 0.0804 -> 1.0412, missing 0% -> 0%
- `lab_results_tropTHs_value_last`: KS=0.9809, W/std=1.554, mean 0.4652 -> 4.2707, missing 63% -> 63%
- `lab_results_ferritin_value_last`: KS=0.9742, W/std=1.2946, mean 523.1277 -> 2973.5216, missing 80% -> 80%
- `lab_results_ferritin_value_first`: KS=0.9713, W/std=1.1089, mean 561.1328 -> 2679.3634, missing 80% -> 80%
- `lab_results_tropTHs_value_first`: KS=0.9503, W/std=0.9139, mean 0.2171 -> 0.9721, missing 63% -> 63%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.006, 10 -> 10 categories, missing 0% -> 0%
- `encounter_primary_reason_renal_complications_f5a_w1a_first`: TVD=0.0054, 3 -> 3 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.0054, 7 -> 7 categories, missing 0% -> 0%
- `hyperkalemia_severity_categorizedValue`: TVD=0.0048, 5 -> 5 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.0048, 7 -> 7 categories, missing 0% -> 0%

## train vs synthetic[mst_eps15_seed1]

Worst numeric columns (by KS):
- `smoking_status_smoker_startTime_count`: KS=0.9815, W/std=1.2044, mean 0.0804 -> 1.0995, missing 0% -> 0%
- `lab_results_tropTHs_value_last`: KS=0.9809, W/std=1.5953, mean 0.4652 -> 4.3916, missing 63% -> 63%
- `lab_results_ferritin_value_last`: KS=0.9742, W/std=1.2335, mean 523.1277 -> 2775.0312, missing 80% -> 80%
- `lab_results_ferritin_value_first`: KS=0.9713, W/std=1.1126, mean 561.1328 -> 2670.7979, missing 80% -> 80%
- `lab_results_tropTHs_value_first`: KS=0.9503, W/std=0.8607, mean 0.2171 -> 0.9334, missing 63% -> 62%
Worst categorical columns (by TVD):
- `ckd_severity_categorizedValue`: TVD=0.008, 7 -> 7 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.0068, 10 -> 10 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.0063, 7 -> 7 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.006, 6 -> 6 categories, missing 0% -> 0%
- `conditions_ibd`: TVD=0.006, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[mst_eps15_seed2]

Worst numeric columns (by KS):
- `smoking_status_smoker_startTime_count`: KS=0.9815, W/std=1.2076, mean 0.0804 -> 1.1022, missing 0% -> 0%
- `lab_results_tropTHs_value_last`: KS=0.9809, W/std=1.5769, mean 0.4652 -> 4.3036, missing 63% -> 63%
- `lab_results_ferritin_value_last`: KS=0.9742, W/std=1.2448, mean 523.1277 -> 2633.5, missing 80% -> 80%
- `lab_results_ferritin_value_first`: KS=0.9713, W/std=1.1162, mean 561.1328 -> 2926.0835, missing 80% -> 80%
- `lab_results_tropTHs_value_first`: KS=0.9503, W/std=0.8825, mean 0.2171 -> 0.9278, missing 63% -> 63%
Worst categorical columns (by TVD):
- `ckd_severity_categorizedValue`: TVD=0.0094, 7 -> 7 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.0074, 7 -> 7 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.0068, 10 -> 10 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.0065, 6 -> 6 categories, missing 0% -> 0%
- `cause_of_death_isNonRenalAndNonCV_f5a_w5a_first`: TVD=0.0057, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[mst_eps1_seed0]

Worst numeric columns (by KS):
- `smoking_status_smoker_startTime_count`: KS=0.9815, W/std=1.6413, mean 0.0804 -> 1.4671, missing 0% -> 0%
- `lab_results_tropTHs_value_last`: KS=0.9809, W/std=1.5922, mean 0.4652 -> 4.1069, missing 63% -> 62%
- `lab_results_ferritin_value_last`: KS=0.9742, W/std=2.6356, mean 523.1277 -> 5554.3471, missing 80% -> 80%
- `lab_results_ferritin_value_first`: KS=0.9713, W/std=1.13, mean 561.1328 -> 2633.5, missing 80% -> 83%
- `lab_results_tropTHs_value_first`: KS=0.9503, W/std=0.8859, mean 0.2171 -> 0.7933, missing 63% -> 62%
Worst categorical columns (by TVD):
- `ckd_severity_categorizedValue`: TVD=0.2406, 7 -> 7 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.1077, 10 -> 10 categories, missing 0% -> 0%
- `hyperkalemia_severity_categorizedValue`: TVD=0.0807, 5 -> 5 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.0776, 7 -> 7 categories, missing 0% -> 0%
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first`: TVD=0.0699, 3 -> 3 categories, missing 0% -> 0%

## train vs synthetic[mst_eps20_seed0]

Worst numeric columns (by KS):
- `smoking_status_smoker_startTime_count`: KS=0.9815, W/std=1.142, mean 0.0804 -> 1.0439, missing 0% -> 0%
- `lab_results_tropTHs_value_last`: KS=0.9809, W/std=1.5633, mean 0.4652 -> 4.2395, missing 63% -> 63%
- `lab_results_ferritin_value_last`: KS=0.9742, W/std=1.3098, mean 523.1277 -> 3010.2192, missing 80% -> 80%
- `lab_results_ferritin_value_first`: KS=0.9713, W/std=1.0994, mean 561.1328 -> 2852.9403, missing 80% -> 80%
- `lab_results_tropTHs_value_first`: KS=0.9503, W/std=0.8484, mean 0.2171 -> 0.9223, missing 63% -> 63%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.0099, 10 -> 10 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.0068, 7 -> 7 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.006, 7 -> 7 categories, missing 0% -> 0%
- `med_vasodil`: TVD=0.0045, 2 -> 2 categories, missing 0% -> 0%
- `hyperkalemia_severity_categorizedValue`: TVD=0.0043, 5 -> 5 categories, missing 0% -> 0%

## train vs synthetic[mst_eps5_seed0]

Worst numeric columns (by KS):
- `smoking_status_smoker_startTime_count`: KS=0.9815, W/std=1.3354, mean 0.0804 -> 1.2107, missing 0% -> 0%
- `lab_results_tropTHs_value_last`: KS=0.9809, W/std=1.5838, mean 0.4652 -> 4.3477, missing 63% -> 62%
- `lab_results_ferritin_value_last`: KS=0.9742, W/std=1.2248, mean 523.1277 -> 2687.2416, missing 80% -> 81%
- `lab_results_ferritin_value_first`: KS=0.9713, W/std=1.3311, mean 561.1328 -> 3413.1773, missing 80% -> 80%
- `lab_results_tropTHs_value_first`: KS=0.9503, W/std=1.0712, mean 0.2171 -> 1.1222, missing 63% -> 62%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.0253, 10 -> 10 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.0188, 7 -> 7 categories, missing 0% -> 0%
- `ckd_severity_categorizedValue`: TVD=0.0173, 7 -> 7 categories, missing 0% -> 0%
- `cause_of_death_isCV_f5a_w1mo_first`: TVD=0.0128, 3 -> 3 categories, missing 0% -> 0%
- `cause_of_death_isAllCause_f5a_w7d_first`: TVD=0.0125, 3 -> 3 categories, missing 0% -> 0%

## train vs synthetic[mst_eps8_seed0]

Worst numeric columns (by KS):
- `smoking_status_smoker_startTime_count`: KS=0.9815, W/std=1.3341, mean 0.0804 -> 1.2096, missing 0% -> 0%
- `lab_results_tropTHs_value_last`: KS=0.9809, W/std=1.6053, mean 0.4652 -> 4.4391, missing 63% -> 63%
- `lab_results_ferritin_value_last`: KS=0.9742, W/std=1.2481, mean 523.1277 -> 2877.3957, missing 80% -> 80%
- `lab_results_ferritin_value_first`: KS=0.9713, W/std=1.0818, mean 561.1328 -> 2737.0527, missing 80% -> 80%
- `lab_results_tropTHs_value_first`: KS=0.9503, W/std=0.8357, mean 0.2171 -> 0.8886, missing 63% -> 63%
Worst categorical columns (by TVD):
- `ckd_severity_categorizedValue`: TVD=0.0162, 7 -> 7 categories, missing 0% -> 0%
- `hyperkalemia_severity_categorizedValue`: TVD=0.0134, 5 -> 5 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.0128, 7 -> 7 categories, missing 0% -> 0%
- `cause_of_death_isAllCause_f5a_w5a_first`: TVD=0.0091, 3 -> 3 categories, missing 0% -> 0%
- `med_ivabradine_history`: TVD=0.0088, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[tvae_seed0]

Worst numeric columns (by KS):
- `conditions_heartFailure_timeFromEarliest_first`: KS=0.5485, W/std=0.1227, mean 11.0621 -> 11.356, missing 0% -> 36%
- `lab_results_creatUS_value_last`: KS=0.4674, W/std=0.6229, mean 694.4878 -> 441.2814, missing 90% -> 98%
- `lab_results_tropTHs_value_last`: KS=0.4297, W/std=0.1664, mean 0.4652 -> 0.2265, missing 63% -> 67%
- `lab_results_tropTnHs_value_last`: KS=0.3853, W/std=0.2239, mean 281.7666 -> 165.6781, missing 88% -> 91%
- `encounter_primary_reason_number_of_days_to_rehosp_for_heart_failure_f5a_first`: KS=0.3823, W/std=0.2874, mean 82.6105 -> 44.9845, missing 95% -> 96%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.306, 10 -> 10 categories, missing 0% -> 0%
- `med_bb`: TVD=0.1926, 2 -> 2 categories, missing 0% -> 0%
- `med_digitalis`: TVD=0.1801, 2 -> 2 categories, missing 0% -> 0%
- `conditions_copd`: TVD=0.1733, 2 -> 2 categories, missing 0% -> 0%
- `patient_demographics_gender`: TVD=0.1716, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[tvae_seed1]

Worst numeric columns (by KS):
- `conditions_heartFailure_timeFromEarliest_first`: KS=0.5452, W/std=0.1271, mean 11.0621 -> 12.4017, missing 0% -> 35%
- `lab_results_creatUS_value_last`: KS=0.4689, W/std=0.6365, mean 694.4878 -> 428.2142, missing 90% -> 98%
- `lab_results_tropTHs_value_last`: KS=0.4571, W/std=0.1706, mean 0.4652 -> 0.2562, missing 63% -> 63%
- `lab_results_ferritin_value_last`: KS=0.3818, W/std=0.1372, mean 523.1277 -> 569.0884, missing 80% -> 88%
- `lab_results_tropTnHs_value_last`: KS=0.38, W/std=0.2272, mean 281.7666 -> 147.5155, missing 88% -> 94%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.269, 10 -> 10 categories, missing 0% -> 0%
- `med_bb`: TVD=0.2011, 2 -> 2 categories, missing 0% -> 0%
- `med_digitalis`: TVD=0.1866, 2 -> 2 categories, missing 0% -> 0%
- `conditions_copd`: TVD=0.1761, 2 -> 2 categories, missing 0% -> 0%
- `conditions_cm`: TVD=0.1707, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[tvae_seed2]

Worst numeric columns (by KS):
- `conditions_heartFailure_timeFromEarliest_first`: KS=0.5457, W/std=0.0868, mean 11.0621 -> 10.6546, missing 0% -> 35%
- `lab_results_ferritin_value_first`: KS=0.511, W/std=0.2413, mean 561.1328 -> 589.0632, missing 80% -> 90%
- `lab_results_creatUS_value_last`: KS=0.4985, W/std=0.6796, mean 694.4878 -> 393.8098, missing 90% -> 99%
- `lab_results_tropTHs_value_last`: KS=0.4625, W/std=0.1702, mean 0.4652 -> 0.2517, missing 63% -> 67%
- `lab_results_ferritin_value_last`: KS=0.4509, W/std=0.1668, mean 523.1277 -> 566.38, missing 80% -> 91%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.2849, 10 -> 10 categories, missing 0% -> 0%
- `med_bb`: TVD=0.2, 2 -> 2 categories, missing 0% -> 0%
- `med_digitalis`: TVD=0.1991, 2 -> 2 categories, missing 0% -> 0%
- `med_anti_coag`: TVD=0.1835, 2 -> 2 categories, missing 0% -> 0%
- `conditions_cm`: TVD=0.1776, 2 -> 2 categories, missing 0% -> 0%
