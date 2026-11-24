# Evaluation: fidelity against the sampling-noise floor

Numeric metrics (KS, `W/std`) are computed over observed values only; the missing-rate MAD compares numeric missingness separately, and covers numeric columns alone. Categorical TVD instead treats nulls as an explicit 'Missing' category, so categorical missingness differences are already inside the TVD. KS and TVD are in [0,1], lower is closer; `W/std` is the Wasserstein distance in units of the reference standard deviation. The `train vs holdout` row is the sampling-noise floor: two disjoint samples of real patients differ by this much purely by chance, so read every synthetic row against it. To keep that reading fair, each synthetic frame is subsampled to the holdout's row count (averaged over seeded draws) before train-vs-synthetic scoring -- the floor is only exchangeable with comparisons at the same sample sizes. 128 constant columns (re-attached verbatim, trivially perfect) are excluded from all aggregates.

| comparison | cols | KS mean | KS median | KS<0.1 | W/std mean | TVD mean | TVD<0.05 | missing-rate MAD |
|---|---|---|---|---|---|---|---|---|
| original vs preprocessed | 140 | 0.0001 | 0.0 | 1.0 | 0.0009 | 0.0 | 1.0 | 0.0001 |
| train vs holdout | 101 | 0.0255 | 0.0229 | 1.0 | 0.0497 | 0.0065 | 0.9841 | 0.0041 |
| train vs synthetic[ctgan_qt_seed0] | 101 | 0.181 | 0.1562 | 0.2632 | 0.5804 | 0.082 | 0.4127 | 0.0833 |
| train vs synthetic[ctgan_seed0] | 101 | 0.2367 | 0.2021 | 0.1579 | 0.461 | 0.098 | 0.3651 | 0.0402 |
| train vs synthetic[ctgan_seed1] | 101 | 0.2677 | 0.2376 | 0.0789 | 0.6061 | 0.0955 | 0.3333 | 0.0396 |
| train vs synthetic[ctgan_seed2] | 101 | 0.2603 | 0.1984 | 0.1316 | 0.6366 | 0.0909 | 0.3651 | 0.0392 |
| train vs synthetic[ddpm_g_seed0] | 101 | 0.1442 | 0.073 | 0.7105 | 0.3761 | 0.0403 | 0.8095 | 0.0523 |
| train vs synthetic[ddpm_seed0] | 101 | 0.1371 | 0.0655 | 0.7632 | 0.3589 | 0.0209 | 0.873 | 0.0498 |
| train vs synthetic[ddpm_seed1] | 101 | 0.1119 | 0.0605 | 0.7632 | 0.2908 | 0.0208 | 0.8413 | 0.0471 |
| train vs synthetic[ddpm_seed2] | 101 | 0.1368 | 0.0688 | 0.7368 | 0.3546 | 0.0202 | 0.8413 | 0.0552 |
| train vs synthetic[dpctgan_eps10_seed0] | 101 | 0.8539 | 0.9761 | 0.0 | 2.7839 | 0.1644 | 0.6984 | 0.3023 |
| train vs synthetic[dpctgan_eps15_seed0] | 101 | 0.833 | 0.9652 | 0.0 | 2.8907 | 0.1294 | 0.7143 | 0.317 |
| train vs synthetic[dpctgan_eps15_seed1] | 101 | 0.7561 | 0.8968 | 0.0 | 2.7971 | 0.1347 | 0.7143 | 0.1716 |
| train vs synthetic[dpctgan_eps15_seed2] | 101 | 0.9024 | 0.9949 | 0.0 | 3.8963 | 0.1206 | 0.7143 | 0.3012 |
| train vs synthetic[dpctgan_eps1_seed0] | 101 | 0.8282 | 0.9711 | 0.0 | 2.5695 | 0.1169 | 0.7143 | 0.2812 |
| train vs synthetic[dpctgan_eps20_seed0] | 101 | 0.784 | 0.9313 | 0.0 | 3.1537 | 0.1313 | 0.7302 | 0.255 |
| train vs synthetic[dpctgan_eps5_seed0] | 101 | 0.7574 | 0.8542 | 0.0 | 2.1247 | 0.1302 | 0.7302 | 0.2628 |
| train vs synthetic[dpctgan_eps8_seed0] | 101 | 0.7479 | 0.9045 | 0.0 | 2.3437 | 0.1143 | 0.7143 | 0.2771 |
| train vs synthetic[gaussian_copula_seed0] | 101 | 0.3424 | 0.2032 | 0.3947 | 1.8047 | 0.0091 | 0.9841 | 0.0514 |
| train vs synthetic[gaussian_copula_seed1] | 101 | 0.3424 | 0.1904 | 0.4211 | 1.8326 | 0.0092 | 0.9841 | 0.0481 |
| train vs synthetic[gaussian_copula_seed2] | 101 | 0.3428 | 0.1915 | 0.3947 | 1.8315 | 0.0089 | 0.9841 | 0.0503 |
| train vs synthetic[mst_eps15_seed0] | 101 | 0.5294 | 0.4243 | 0.0526 | 4.8483 | 0.004 | 1.0 | 0.0026 |
| train vs synthetic[patectgan_eps15_seed0] | 101 | 0.3279 | 0.2272 | 0.1053 | 0.488 | 0.0353 | 0.9524 | 0.0685 |
| train vs synthetic[patectgan_eps1_seed0] | 101 | 0.5425 | 0.5264 | 0.0 | 6.7639 | 0.111 | 0.2698 | 0.0949 |
| train vs synthetic[patectgan_eps5_seed0] | 101 | 0.3269 | 0.2417 | 0.1053 | 0.6636 | 0.0381 | 0.8889 | 0.0676 |
| train vs synthetic[tvae_cap256_seed0] | 101 | 0.1236 | 0.1147 | 0.4737 | 0.19 | 0.0324 | 0.8413 | 0.0161 |
| train vs synthetic[tvae_ep1000_seed0] | 101 | 0.1375 | 0.1568 | 0.3684 | 0.2214 | 0.0306 | 0.8571 | 0.0131 |
| 🚨 synthetic[tvae_ind_seed0] IS STALE (4 undecoded cells; excluded from group aggregates) -- rerun generate | | | | | | | | |
| train vs synthetic[tvae_ind_seed0] | 101 | 0.1185 | 0.1084 | 0.4474 | 0.1789 | 0.035 | 0.8254 | 0.0188 |
| train vs synthetic[tvae_qt_seed0] | 101 | 0.1025 | 0.0983 | 0.5 | 0.1896 | 0.0347 | 0.8095 | 0.0284 |
| train vs synthetic[tvae_qt_seed1] | 101 | 0.112 | 0.0959 | 0.5 | 0.2003 | 0.0372 | 0.7778 | 0.0219 |
| train vs synthetic[tvae_qt_seed2] | 101 | 0.1129 | 0.1002 | 0.5 | 0.1912 | 0.0305 | 0.8254 | 0.0145 |
| train vs synthetic[tvae_seed0] | 101 | 0.1266 | 0.1268 | 0.4737 | 0.1984 | 0.0329 | 0.8254 | 0.0182 |
| train vs synthetic[tvae_seed1] | 101 | 0.1319 | 0.1191 | 0.4474 | 0.2003 | 0.0348 | 0.7937 | 0.0242 |
| train vs synthetic[tvae_seed2] | 101 | 0.1276 | 0.1474 | 0.4211 | 0.1896 | 0.0311 | 0.8254 | 0.0138 |

## Per (model, ε) across seeds (train vs synthetic)

| model | ε | runs | KS mean ± sd | TVD mean ± sd | missing-MAD ± sd |
|---|---|---|---|---|---|
| ctgan | - | 3 | 0.2549 ± 0.0162 | 0.0948 ± 0.0036 | 0.0397 ± 0.0005 |
| ctgan_qt | - | 1 | 0.181 | 0.082 | 0.0833 |
| ddpm | - | 3 | 0.1286 ± 0.0145 | 0.0206 ± 0.0004 | 0.0507 ± 0.0041 |
| ddpm_g | - | 1 | 0.1442 | 0.0403 | 0.0523 |
| dpctgan | 1 | 1 | 0.8282 | 0.1169 | 0.2812 |
| dpctgan | 5 | 1 | 0.7574 | 0.1302 | 0.2628 |
| dpctgan | 8 | 1 | 0.7479 | 0.1143 | 0.2771 |
| dpctgan | 10 | 1 | 0.8539 | 0.1644 | 0.3023 |
| dpctgan | 15 | 3 | 0.8305 ± 0.0732 | 0.1282 ± 0.0071 | 0.2633 ± 0.0798 |
| dpctgan | 20 | 1 | 0.784 | 0.1313 | 0.255 |
| gaussian_copula | - | 3 | 0.3425 ± 0.0002 | 0.0091 ± 0.0002 | 0.0499 ± 0.0017 |
| mst | 15 | 1 | 0.5294 | 0.004 | 0.0026 |
| patectgan | 1 | 1 | 0.5425 | 0.111 | 0.0949 |
| patectgan | 5 | 1 | 0.3269 | 0.0381 | 0.0676 |
| patectgan | 15 | 1 | 0.3279 | 0.0353 | 0.0685 |
| tvae | - | 3 | 0.1287 ± 0.0028 | 0.0329 ± 0.0019 | 0.0187 ± 0.0052 |
| tvae_cap256 | - | 1 | 0.1236 | 0.0324 | 0.0161 |
| tvae_ep1000 | - | 1 | 0.1375 | 0.0306 | 0.0131 |
| tvae_ind | - | 0 | - | - | - |
| tvae_qt | - | 3 | 0.1091 ± 0.0058 | 0.0341 ± 0.0034 | 0.0216 ± 0.007 |

Excluded from these group aggregates as stale (scores describe outdated files; see the per-run table): `tvae_ind_seed0`.

## Full-joint distinguishability (C2ST)

Out-of-fold AUC (5-fold stratified CV) of a classifier separating real from synthetic rows, over the columns present in BOTH frames; 0.5 = joints indistinguishable. `coverage` is the fraction of modelled columns the synthetic file actually contains -- width-limited runs are scored on that intersection only, never on schema width itself. Floor (train vs holdout): **0.5122**.

| run | C2ST AUC | ± sd | coverage |
|---|---|---|---|
| ctgan_qt_seed0 | 1.0 | 0.0 | 1.0 |
| ctgan_seed0 | 0.9999 | 0.0002 | 1.0 |
| ctgan_seed1 | 1.0 | 0.0 | 1.0 |
| ctgan_seed2 | 1.0 | 0.0001 | 1.0 |
| ddpm_g_seed0 | 0.9887 | 0.0029 | 1.0 |
| ddpm_seed0 | 0.9448 | 0.0107 | 1.0 |
| ddpm_seed1 | 0.9485 | 0.005 | 1.0 |
| ddpm_seed2 | 0.9519 | 0.0075 | 1.0 |
| dpctgan_eps10_seed0 | 1.0 | 0.0 | 1.0 |
| dpctgan_eps15_seed0 | 1.0 | 0.0 | 1.0 |
| dpctgan_eps15_seed1 | 0.9999 | 0.0002 | 1.0 |
| dpctgan_eps15_seed2 | 1.0 | 0.0 | 1.0 |
| dpctgan_eps1_seed0 | 1.0 | 0.0 | 1.0 |
| dpctgan_eps20_seed0 | 1.0 | 0.0 | 1.0 |
| dpctgan_eps5_seed0 | 0.9999 | 0.0002 | 1.0 |
| dpctgan_eps8_seed0 | 1.0 | 0.0 | 1.0 |
| gaussian_copula_seed0 | 1.0 | 0.0 | 1.0 |
| gaussian_copula_seed1 | 1.0 | 0.0 | 1.0 |
| gaussian_copula_seed2 | 1.0 | 0.0 | 1.0 |
| mst_eps15_seed0 | 1.0 | 0.0 | 1.0 |
| patectgan_eps15_seed0 | 1.0 | 0.0 | 1.0 |
| patectgan_eps1_seed0 | 1.0 | 0.0 | 1.0 |
| patectgan_eps5_seed0 | 1.0 | 0.0 | 1.0 |
| tvae_cap256_seed0 | 0.9988 | 0.0003 | 1.0 |
| tvae_ep1000_seed0 | 0.9977 | 0.0011 | 1.0 |
| tvae_ind_seed0 | 0.9973 | 0.0006 | 1.0 |
| tvae_qt_seed0 | 0.997 | 0.0009 | 1.0 |
| tvae_qt_seed1 | 0.9947 | 0.001 | 1.0 |
| tvae_qt_seed2 | 0.9963 | 0.0019 | 1.0 |
| tvae_seed0 | 0.9988 | 0.0004 | 1.0 |
| tvae_seed1 | 0.9987 | 0.0005 | 1.0 |
| tvae_seed2 | 0.9986 | 0.0008 | 1.0 |

## Subgroup fidelity (KS mean per stratum, train vs synthetic)

Does the synthetic cohort represent every subgroup as faithfully as the majority? Each cell is read against its stratum's own noise floor.

| run | female | male | age_under_65 | age_65_79 | age_80_plus |
|---|---|---|---|---|---|
| *noise floor* | 0.036 | 0.0321 | 0.0341 | 0.0355 | 0.0525 |
| ctgan_qt_seed0 | 0.1968 | 0.1853 | 0.2143 | 0.1868 | 0.2097 |
| ctgan_seed0 | 0.2351 | 0.2493 | 0.2417 | 0.2402 | 0.2487 |
| ctgan_seed1 | 0.2666 | 0.277 | 0.2834 | 0.2646 | 0.2786 |
| ctgan_seed2 | 0.2635 | 0.2695 | 0.2855 | 0.2646 | 0.2567 |
| ddpm_g_seed0 | 0.1498 | 0.1457 | 0.1458 | 0.1498 | 0.164 |
| ddpm_seed0 | 0.147 | 0.1439 | 0.1441 | 0.1505 | 0.1643 |
| ddpm_seed1 | 0.1125 | 0.1272 | 0.123 | 0.1268 | 0.1414 |
| ddpm_seed2 | 0.1462 | 0.1418 | 0.1419 | 0.1502 | 0.172 |
| dpctgan_eps10_seed0 | 0.86 | - | 0.8513 | 0.8482 | 0.8572 |
| dpctgan_eps15_seed0 | - | 0.8405 | 0.8673 | 0.8394 | 0.8407 |
| dpctgan_eps15_seed1 | 0.7601 | - | 0.7721 | 0.7603 | 0.776 |
| dpctgan_eps15_seed2 | 0.903 | - | 0.9127 | - | - |
| dpctgan_eps1_seed0 | 0.8281 | - | - | 0.8345 | 0.8231 |
| dpctgan_eps20_seed0 | - | 0.7897 | 0.8078 | 0.7883 | 0.7717 |
| dpctgan_eps5_seed0 | 0.7495 | - | 0.7795 | 0.7593 | 0.7538 |
| dpctgan_eps8_seed0 | - | 0.7521 | 0.7409 | 0.7531 | 0.7652 |
| gaussian_copula_seed0 | 0.3506 | 0.3537 | 0.3593 | 0.3505 | 0.3414 |
| gaussian_copula_seed1 | 0.3549 | 0.3497 | 0.3613 | 0.346 | 0.3616 |
| gaussian_copula_seed2 | 0.3522 | 0.3484 | 0.3563 | 0.3498 | 0.3491 |
| mst_eps15_seed0 | 0.5694 | 0.542 | 0.6668 | 0.5745 | 0.7234 |
| patectgan_eps15_seed0 | 0.3414 | 0.3377 | 0.3456 | 0.3435 | 0.3575 |
| patectgan_eps1_seed0 | 0.5463 | 0.5493 | 0.5422 | 0.5478 | 0.5631 |
| patectgan_eps5_seed0 | 0.3325 | 0.3335 | 0.3551 | 0.3326 | 0.3353 |
| tvae_cap256_seed0 | 0.1333 | 0.1291 | 0.1387 | 0.1371 | 0.1393 |
| tvae_ep1000_seed0 | 0.1599 | 0.1315 | 0.1378 | 0.1492 | 0.1591 |
| tvae_ind_seed0 | 0.1709 | 0.1345 | 0.1454 | 0.1323 | 0.1576 |
| tvae_qt_seed0 | 0.1276 | 0.1119 | 0.1202 | 0.11 | 0.1394 |
| tvae_qt_seed1 | 0.1282 | 0.1176 | 0.1365 | 0.124 | 0.1282 |
| tvae_qt_seed2 | 0.1321 | 0.1215 | 0.1157 | 0.1294 | 0.1452 |
| tvae_seed0 | 0.1444 | 0.1289 | 0.1395 | 0.1422 | 0.1539 |
| tvae_seed1 | 0.1489 | 0.1374 | 0.1406 | 0.1428 | 0.1453 |
| tvae_seed2 | 0.1509 | 0.1237 | 0.1341 | 0.1445 | 0.1527 |

## Generalization (holdout vs synthetic)

Distance to real records the generator NEVER saw. A model that is much closer to train than to holdout is fitting its training sample, not the population.

| run | KS mean (train) | KS mean (holdout) | TVD mean (train) | TVD mean (holdout) |
|---|---|---|---|---|
| ctgan_qt_seed0 | 0.181 | 0.1882 | 0.082 | 0.0783 |
| ctgan_seed0 | 0.2367 | 0.2354 | 0.098 | 0.0958 |
| ctgan_seed1 | 0.2677 | 0.2638 | 0.0955 | 0.0942 |
| ctgan_seed2 | 0.2603 | 0.2582 | 0.0909 | 0.0851 |
| ddpm_g_seed0 | 0.1442 | 0.143 | 0.0403 | 0.043 |
| ddpm_seed0 | 0.1371 | 0.1396 | 0.0209 | 0.0238 |
| ddpm_seed1 | 0.1119 | 0.116 | 0.0208 | 0.0241 |
| ddpm_seed2 | 0.1368 | 0.1394 | 0.0202 | 0.0236 |
| dpctgan_eps10_seed0 | 0.8539 | 0.8487 | 0.1644 | 0.1672 |
| dpctgan_eps15_seed0 | 0.833 | 0.826 | 0.1294 | 0.1309 |
| dpctgan_eps15_seed1 | 0.7561 | 0.7517 | 0.1347 | 0.1379 |
| dpctgan_eps15_seed2 | 0.9024 | 0.8992 | 0.1206 | 0.1241 |
| dpctgan_eps1_seed0 | 0.8282 | 0.8227 | 0.1169 | 0.1205 |
| dpctgan_eps20_seed0 | 0.784 | 0.7799 | 0.1313 | 0.1334 |
| dpctgan_eps5_seed0 | 0.7574 | 0.7533 | 0.1302 | 0.1332 |
| dpctgan_eps8_seed0 | 0.7479 | 0.7475 | 0.1143 | 0.1183 |
| gaussian_copula_seed0 | 0.3424 | 0.3393 | 0.0091 | 0.0086 |
| gaussian_copula_seed1 | 0.3424 | 0.34 | 0.0092 | 0.0103 |
| gaussian_copula_seed2 | 0.3428 | 0.3392 | 0.0089 | 0.0102 |
| mst_eps15_seed0 | 0.5294 | 0.5239 | 0.004 | 0.0066 |
| patectgan_eps15_seed0 | 0.3279 | 0.3228 | 0.0353 | 0.0396 |
| patectgan_eps1_seed0 | 0.5425 | 0.5397 | 0.111 | 0.1084 |
| patectgan_eps5_seed0 | 0.3269 | 0.3178 | 0.0381 | 0.0411 |
| tvae_cap256_seed0 | 0.1236 | 0.1212 | 0.0324 | 0.0348 |
| tvae_ep1000_seed0 | 0.1375 | 0.1399 | 0.0306 | 0.0328 |
| tvae_ind_seed0 | 0.1185 | 0.1291 | 0.035 | 0.0376 |
| tvae_qt_seed0 | 0.1025 | 0.1039 | 0.0347 | 0.0367 |
| tvae_qt_seed1 | 0.112 | 0.1131 | 0.0372 | 0.0406 |
| tvae_qt_seed2 | 0.1129 | 0.1137 | 0.0305 | 0.0337 |
| tvae_seed0 | 0.1266 | 0.1299 | 0.0329 | 0.0353 |
| tvae_seed1 | 0.1319 | 0.1283 | 0.0348 | 0.0366 |
| tvae_seed2 | 0.1276 | 0.1225 | 0.0311 | 0.0335 |

## Association structure (train vs synthetic)

Absolute change in pairwise association; 0 = relationship perfectly preserved. `fabricated` counts pairs nearly independent in real data (|assoc|<0.1) rendered strongly associated (>0.5) in the synthetic data. Noise floor rows show how much two real samples differ.

| run | pair type | pairs | mean \|Δ\| | median \|Δ\| | \|Δ\|<0.1 | fabricated | worst pair |
|---|---|---|---|---|---|---|---|
| *noise floor* | Spearman (num-num) | 703 | 0.0285 | 0.0221 | 0.9829 | 0 | - |
| *noise floor* | Cramer's V (cat-cat) | 1953 | 0.0231 | 0.0133 | 0.9744 | 0 | - |
| *noise floor* | corr-ratio (num-cat) | 7220 | 0.0113 | 0.0 | 0.9885 | 0 | - |
| ctgan_qt_seed0 | Spearman (num-num) | 659 | 0.1195 | 0.0721 | 0.6085 | 0 | `echocardiographs_lvef_pET_last|echocardiographs_lvef_pET_first` (0.9625 -> -0.1308) |
| ctgan_qt_seed0 | Cramer's V (cat-cat) | 1953 | 0.0509 | 0.0279 | 0.8889 | 0 | `ckd_severity_from_calculated_egfr|ckd_severity_calculated_or_measured` (1.0 -> 0.0638) |
| ctgan_qt_seed0 | corr-ratio (num-cat) | 7030 | 0.0152 | 0.0 | 0.9694 | 0 | `eGFR_2021_ckd_epi_creatinine|ckd_severity_calculated_or_measured` (0.9418 -> 0.0578) |
| ctgan_seed0 | Spearman (num-num) | 703 | 0.1216 | 0.0688 | 0.6401 | 0 | `echocardiographs_lvef|echocardiographs_lvef_pET_last` (1.0 -> -0.1172) |
| ctgan_seed0 | Cramer's V (cat-cat) | 1953 | 0.0571 | 0.0349 | 0.8479 | 0 | `ckd_severity_from_calculated_egfr|ckd_severity_calculated_or_measured` (1.0 -> 0.0783) |
| ctgan_seed0 | corr-ratio (num-cat) | 7220 | 0.0156 | 0.0 | 0.9708 | 0 | `eGFR_2021_ckd_epi_creatinine|ckd_severity_from_calculated_egfr` (0.9418 -> 0.0515) |
| ctgan_seed1 | Spearman (num-num) | 703 | 0.1172 | 0.0709 | 0.633 | 0 | `echocardiographs_lvef_pET_last|echocardiographs_lvef_pET_first` (0.9625 -> -0.079) |
| ctgan_seed1 | Cramer's V (cat-cat) | 1953 | 0.0557 | 0.0318 | 0.8505 | 0 | `ckd_severity_from_calculated_egfr|ckd_severity_calculated_or_measured` (1.0 -> 0.0669) |
| ctgan_seed1 | corr-ratio (num-cat) | 7220 | 0.0148 | 0.0 | 0.973 | 0 | `eGFR_2021_ckd_epi_creatinine|ckd_severity_calculated_or_measured` (0.9418 -> 0.0383) |
| ctgan_seed2 | Spearman (num-num) | 703 | 0.1192 | 0.075 | 0.6216 | 0 | `echocardiographs_lvef|echocardiographs_lvef_pET_last` (1.0 -> -0.1954) |
| ctgan_seed2 | Cramer's V (cat-cat) | 1953 | 0.0596 | 0.0364 | 0.8285 | 0 | `ckd_severity_from_calculated_egfr|ckd_severity_calculated_or_measured` (1.0 -> 0.0477) |
| ctgan_seed2 | corr-ratio (num-cat) | 7220 | 0.0169 | 0.0 | 0.9622 | 0 | `eGFR_2021_ckd_epi_creatinine|ckd_severity_from_calculated_egfr` (0.9418 -> 0.0227) |
| ddpm_g_seed0 | Spearman (num-num) | 585 | 0.0669 | 0.0362 | 0.8256 | 0 | `echocardiographs_lvef_pET_last|maggic_total_score` (-0.3627 -> 0.8895) |
| ddpm_g_seed0 | Cramer's V (cat-cat) | 300 | 0.026 | 0.0178 | 0.98 | 0 | `conditions_hypothyroid|conditions_pericardial` (0.0279 -> 0.175) |
| ddpm_g_seed0 | corr-ratio (num-cat) | 6650 | 0.0111 | 0.0 | 0.9893 | 0 | `encounter_primary_reason_number_of_non_CV_rehospitalizations_5a_f5a_count|encounter_primary_reason_non_CV_Disease_f5a_w5a_first` (0.8461 -> 0.0) |
| ddpm_seed0 | Spearman (num-num) | 630 | 0.0651 | 0.0379 | 0.8365 | 0 | `echocardiographs_lvef_pET_last|maggic_total_score` (-0.3627 -> 0.9077) |
| ddpm_seed0 | Cramer's V (cat-cat) | 1275 | 0.0429 | 0.0168 | 0.8816 | 0 | `encounter_primary_reason_HF_Disease_f5a_w1mo_first|encounter_primary_reason_CV_Disease_f5a_w1a_first` (0.4073 -> 0.7703) |
| ddpm_seed0 | corr-ratio (num-cat) | 6840 | 0.0105 | 0.0 | 0.9873 | 0 | `maggic_total_score|ckd_severity_from_calculated_egfr` (0.5594 -> 0.1185) |
| ddpm_seed1 | Spearman (num-num) | 630 | 0.062 | 0.0364 | 0.8365 | 0 | `echocardiographs_lvef_pET_last|maggic_total_score` (-0.3627 -> 0.8775) |
| ddpm_seed1 | Cramer's V (cat-cat) | 1275 | 0.0396 | 0.0167 | 0.8894 | 0 | `encounter_primary_reason_HF_Disease_f5a_w7d_first|encounter_primary_reason_HF_Disease_f5a_w1mo_first` (0.8161 -> 0.3772) |
| ddpm_seed1 | corr-ratio (num-cat) | 6840 | 0.0088 | 0.0 | 0.9892 | 0 | `maggic_total_score|ckd_severity_calculated_or_measured` (0.5594 -> 0.0875) |
| ddpm_seed2 | Spearman (num-num) | 661 | 0.0656 | 0.037 | 0.8215 | 0 | `echocardiographs_lvef|maggic_total_score` (-0.3627 -> 0.9326) |
| ddpm_seed2 | Cramer's V (cat-cat) | 1378 | 0.045 | 0.0174 | 0.8636 | 0 | `encounter_primary_reason_HF_Disease_f5a_w7d_first|encounter_primary_reason_non_CV_Disease_f5a_w1mo_first` (0.4527 -> 1.0) |
| ddpm_seed2 | corr-ratio (num-cat) | 7030 | 0.0096 | 0.0 | 0.9869 | 0 | `echocardiographs_lvef|conditions_cm` (0.4033 -> 0.0372) |
| dpctgan_eps10_seed0 | Spearman (num-num) | 292 | 0.2457 | 0.1354 | 0.4041 | 27 | `lab_results_creatBS_value_p3a_avg|eGFR_2021_ckd_epi_creatinine` (-0.7891 -> 0.4357) |
| dpctgan_eps10_seed0 | Cramer's V (cat-cat) | 630 | 0.1172 | 0.0276 | 0.8175 | 2 | `encounter_primary_reason_non_CV_Disease_f5a_w6mo_first|encounter_primary_reason_renal_complications_f5a_w6mo_first` (1.0 -> 0.0006) |
| dpctgan_eps10_seed0 | corr-ratio (num-cat) | 4750 | 0.0224 | 0.0 | 0.9587 | 0 | `eGFR_2021_ckd_epi_creatinine|ckd_severity_calculated_or_measured` (0.9418 -> 0.0148) |
| dpctgan_eps15_seed0 | Spearman (num-num) | 405 | 0.2371 | 0.1341 | 0.3877 | 39 | `lab_results_validSerumCreatinine_value_first|lab_results_creatBS_value_p3a_avg` (0.8915 -> -0.5387) |
| dpctgan_eps15_seed0 | Cramer's V (cat-cat) | 378 | 0.1267 | 0.0261 | 0.7989 | 0 | `encounter_primary_reason_CV_Disease_f5a_w1a_first|encounter_primary_reason_non_CV_Disease_f5a_w1a_first` (1.0 -> 0.0009) |
| dpctgan_eps15_seed0 | corr-ratio (num-cat) | 5510 | 0.0205 | 0.0 | 0.9633 | 0 | `eGFR_2021_ckd_epi_creatinine|ckd_severity_calculated_or_measured` (0.9418 -> 0.0218) |
| dpctgan_eps15_seed1 | Spearman (num-num) | 561 | 0.1978 | 0.1245 | 0.4296 | 23 | `encounters_lengthOfStay|encounters_numOfPreviousHFStays_count` (-0.1033 -> 0.9886) |
| dpctgan_eps15_seed1 | Cramer's V (cat-cat) | 780 | 0.1163 | 0.0201 | 0.8179 | 0 | `encounter_primary_reason_non_CV_Disease_f5a_w3a_first|encounter_primary_reason_renal_complications_f5a_w3a_first` (1.0 -> 0.0006) |
| dpctgan_eps15_seed1 | corr-ratio (num-cat) | 6460 | 0.0187 | 0.0 | 0.9655 | 0 | `eGFR_2021_ckd_epi_creatinine|ckd_severity_from_calculated_egfr` (0.9418 -> 0.0133) |
| dpctgan_eps15_seed2 | Spearman (num-num) | 377 | 0.2815 | 0.1809 | 0.2918 | 43 | `lab_results_creatBS_value_p3a_avg|lab_results_validSerumCreatinine_value_pET` (0.9049 -> -0.264) |
| dpctgan_eps15_seed2 | Cramer's V (cat-cat) | 528 | 0.123 | 0.0243 | 0.8068 | 0 | `encounter_primary_reason_CV_Disease_f5a_w1a_first|encounter_primary_reason_non_CV_Disease_f5a_w1a_first` (1.0 -> 0.0006) |
| dpctgan_eps15_seed2 | corr-ratio (num-cat) | 5320 | 0.0217 | 0.0 | 0.9583 | 0 | `eGFR_2021_ckd_epi_creatinine|ckd_severity_from_calculated_egfr` (0.9418 -> 0.0) |
| dpctgan_eps1_seed0 | Spearman (num-num) | 349 | 0.2255 | 0.1319 | 0.4126 | 19 | `echocardiographs_lvef|echocardiographs_lvef_pET_last` (1.0 -> -0.0804) |
| dpctgan_eps1_seed0 | Cramer's V (cat-cat) | 1035 | 0.1447 | 0.0211 | 0.7787 | 1 | `encounter_primary_reason_CV_Disease_f5a_w7d_first|encounter_primary_reason_renal_complications_f5a_w7d_first` (1.0 -> 0.0006) |
| dpctgan_eps1_seed0 | corr-ratio (num-cat) | 5130 | 0.0209 | 0.0 | 0.9626 | 0 | `eGFR_2021_ckd_epi_creatinine|ckd_severity_calculated_or_measured` (0.9418 -> 0.036) |
| dpctgan_eps20_seed0 | Spearman (num-num) | 525 | 0.2111 | 0.1207 | 0.4229 | 19 | `lab_results_validSerumCreatinine_value_pET|eGFR_2021_ckd_epi_creatinine` (-0.865 -> 0.2987) |
| dpctgan_eps20_seed0 | Cramer's V (cat-cat) | 666 | 0.1118 | 0.0227 | 0.8183 | 0 | `encounter_primary_reason_non_CV_Disease_f5a_w3mo_first|encounter_primary_reason_renal_complications_f5a_w3mo_first` (1.0 -> 0.0006) |
| dpctgan_eps20_seed0 | corr-ratio (num-cat) | 6270 | 0.0199 | 0.0 | 0.9641 | 0 | `eGFR_2021_ckd_epi_creatinine|ckd_severity_calculated_or_measured` (0.9418 -> 0.0294) |
| dpctgan_eps5_seed0 | Spearman (num-num) | 491 | 0.2096 | 0.1105 | 0.4562 | 33 | `vital_signs_systolicBp_value_last|vital_signs_systolicBpDuringEncounter_value_pET` (0.9977 -> -0.1119) |
| dpctgan_eps5_seed0 | Cramer's V (cat-cat) | 946 | 0.126 | 0.0224 | 0.7992 | 0 | `encounter_primary_reason_non_CV_Disease_f5a_w7d_first|encounter_primary_reason_renal_complications_f5a_w7d_first` (1.0 -> 0.0006) |
| dpctgan_eps5_seed0 | corr-ratio (num-cat) | 6080 | 0.0196 | 0.0 | 0.964 | 0 | `eGFR_2021_ckd_epi_creatinine|ckd_severity_calculated_or_measured` (0.9418 -> 0.0115) |
| dpctgan_eps8_seed0 | Spearman (num-num) | 460 | 0.2027 | 0.1135 | 0.45 | 33 | `lab_results_validSerumCreatinine_value_pET|eGFR_2021_ckd_epi_creatinine` (-0.865 -> 0.3656) |
| dpctgan_eps8_seed0 | Cramer's V (cat-cat) | 465 | 0.0674 | 0.0212 | 0.8753 | 0 | `encounter_primary_reason_non_CV_Disease_f5a_w1mo_first|encounter_primary_reason_renal_complications_f5a_w1mo_first` (1.0 -> 0.0009) |
| dpctgan_eps8_seed0 | corr-ratio (num-cat) | 5890 | 0.0209 | 0.0 | 0.961 | 0 | `eGFR_2021_ckd_epi_creatinine|ckd_severity_calculated_or_measured` (0.9418 -> 0.0) |
| gaussian_copula_seed0 | Spearman (num-num) | 703 | 0.0794 | 0.0461 | 0.7696 | 0 | `echocardiographs_lvef_pET_last|maggic_total_score` (-0.3627 -> 0.9689) |
| gaussian_copula_seed0 | Cramer's V (cat-cat) | 1830 | 0.1424 | 0.0188 | 0.7732 | 0 | `encounter_primary_reason_non_CV_Disease_f5a_w7d_first|encounter_primary_reason_renal_complications_f5a_w7d_first` (1.0 -> 0.0016) |
| gaussian_copula_seed0 | corr-ratio (num-cat) | 7220 | 0.0164 | 0.0 | 0.973 | 0 | `eGFR_2021_ckd_epi_creatinine|ckd_severity_calculated_or_measured` (0.9418 -> 0.0752) |
| gaussian_copula_seed1 | Spearman (num-num) | 701 | 0.0813 | 0.0494 | 0.7703 | 0 | `echocardiographs_lvef_pET_last|maggic_total_score` (-0.3627 -> 0.964) |
| gaussian_copula_seed1 | Cramer's V (cat-cat) | 1953 | 0.136 | 0.0178 | 0.7793 | 0 | `encounter_primary_reason_HF_Disease_f5a_w7d_first|encounter_primary_reason_renal_complications_f5a_w7d_first` (1.0 -> 0.0009) |
| gaussian_copula_seed1 | corr-ratio (num-cat) | 7220 | 0.0163 | 0.0 | 0.9733 | 0 | `eGFR_2021_ckd_epi_creatinine|ckd_severity_from_calculated_egfr` (0.9418 -> 0.1218) |
| gaussian_copula_seed2 | Spearman (num-num) | 703 | 0.0848 | 0.0516 | 0.7482 | 0 | `echocardiographs_lvef_pET_last|maggic_total_score` (-0.3627 -> 0.967) |
| gaussian_copula_seed2 | Cramer's V (cat-cat) | 1891 | 0.1386 | 0.0178 | 0.78 | 0 | `encounter_primary_reason_HF_Disease_f5a_w7d_first|encounter_primary_reason_renal_complications_f5a_w7d_first` (1.0 -> 0.0009) |
| gaussian_copula_seed2 | corr-ratio (num-cat) | 7220 | 0.0164 | 0.0 | 0.972 | 0 | `eGFR_2021_ckd_epi_creatinine|ckd_severity_calculated_or_measured` (0.9418 -> 0.1204) |
| mst_eps15_seed0 | Spearman (num-num) | 694 | 0.234 | 0.1559 | 0.3444 | 63 | `patient_demographics_age|lab_results_hemoglobin_value_last` (-0.3572 -> 0.8296) |
| mst_eps15_seed0 | Cramer's V (cat-cat) | 1891 | 0.1806 | 0.1316 | 0.4125 | 110 | `encounter_primary_reason_CV_Disease_f5a_w7d_first|encounter_primary_reason_non_CV_Disease_f5a_w7d_first` (1.0 -> 0.0016) |
| mst_eps15_seed0 | corr-ratio (num-cat) | 7220 | 0.06 | 0.0 | 0.7974 | 146 | `encounter_primary_reason_number_of_CV_rehospitalizations_5a_f5a_count|hyperkalemia_severity_categorizedValue` (0.0518 -> 0.9387) |
| patectgan_eps15_seed0 | Spearman (num-num) | 703 | 0.1098 | 0.0714 | 0.6444 | 8 | `echocardiographs_lvef_pET_first|maggic_total_score` (-0.344 -> 0.6852) |
| patectgan_eps15_seed0 | Cramer's V (cat-cat) | 496 | 0.0324 | 0.0167 | 0.9637 | 0 | `ckd_severity_from_calculated_egfr|ckd_severity_calculated_or_measured` (1.0 -> 0.0016) |
| patectgan_eps15_seed0 | corr-ratio (num-cat) | 7220 | 0.018 | 0.0 | 0.9695 | 0 | `eGFR_2021_ckd_epi_creatinine|ckd_severity_calculated_or_measured` (0.9418 -> 0.0148) |
| patectgan_eps1_seed0 | Spearman (num-num) | 703 | 0.1274 | 0.0849 | 0.5932 | 0 | `echocardiographs_lvef_pET_last|echocardiographs_lvef_pET_first` (0.9625 -> -0.0127) |
| patectgan_eps1_seed0 | Cramer's V (cat-cat) | 1953 | 0.1372 | 0.022 | 0.7839 | 0 | `encounter_primary_reason_non_CV_Disease_f5a_w3a_first|encounter_primary_reason_renal_complications_f5a_w3a_first` (1.0 -> 0.0046) |
| patectgan_eps1_seed0 | corr-ratio (num-cat) | 7220 | 0.0179 | 0.0 | 0.9695 | 0 | `eGFR_2021_ckd_epi_creatinine|ckd_severity_from_calculated_egfr` (0.9418 -> 0.0491) |
| patectgan_eps5_seed0 | Spearman (num-num) | 703 | 0.1183 | 0.069 | 0.6373 | 8 | `encounters_numOfPreviousHFStays_count|encounter_primary_reason_number_of_renal_rehospitalizations_5a_f5a_count` (0.0348 -> 0.9833) |
| patectgan_eps5_seed0 | Cramer's V (cat-cat) | 630 | 0.0447 | 0.0189 | 0.9349 | 0 | `encounter_primary_reason_non_CV_Disease_f5a_w5a_first|encounter_primary_reason_renal_complications_f5a_w5a_first` (1.0 -> 0.0006) |
| patectgan_eps5_seed0 | corr-ratio (num-cat) | 7220 | 0.0177 | 0.0 | 0.9701 | 0 | `eGFR_2021_ckd_epi_creatinine|ckd_severity_from_calculated_egfr` (0.9418 -> 0.0434) |
| tvae_cap256_seed0 | Spearman (num-num) | 656 | 0.0555 | 0.0378 | 0.8338 | 0 | `vital_signs_heartRate_value_first|vital_signs_heartRate_value_last` (0.7827 -> 0.2769) |
| tvae_cap256_seed0 | Cramer's V (cat-cat) | 741 | 0.105 | 0.023 | 0.7233 | 0 | `encounter_primary_reason_HF_Disease_f5a_w6mo_first|encounter_primary_reason_renal_complications_f5a_w3mo_first` (0.843 -> 0.0011) |
| tvae_cap256_seed0 | corr-ratio (num-cat) | 7030 | 0.0139 | 0.0 | 0.9725 | 0 | `encounter_primary_reason_number_of_non_CV_rehospitalizations_5a_f5a_count|encounter_primary_reason_CV_Disease_f5a_w3mo_first` (0.547 -> 0.0017) |
| tvae_ep1000_seed0 | Spearman (num-num) | 659 | 0.0736 | 0.0483 | 0.7284 | 0 | `lab_results_potassium_value_last|nyha_nyha` (-0.1213 -> -0.5225) |
| tvae_ep1000_seed0 | Cramer's V (cat-cat) | 1225 | 0.073 | 0.0208 | 0.7739 | 2 | `encounter_primary_reason_HF_Disease_f5a_w7d_first|encounter_primary_reason_non_CV_Disease_f5a_w6mo_first` (0.2779 -> 1.0) |
| tvae_ep1000_seed0 | corr-ratio (num-cat) | 7030 | 0.0117 | 0.0 | 0.9761 | 0 | `lab_results_potassium_value_last|hyperkalemia_severity_categorizedValue` (0.5223 -> 0.1001) |
| tvae_ind_seed0 | Spearman (num-num) | 630 | 0.0695 | 0.0473 | 0.781 | 0 | `lab_results_sodium_value_first|lab_results_sodium_value_last` (0.4951 -> 0.0593) |
| tvae_ind_seed0 | Cramer's V (cat-cat) | 666 | 0.0872 | 0.0382 | 0.7267 | 0 | `encounter_primary_reason_HF_Disease_f5a_w3mo_first|encounter_primary_reason_HF_Disease_f5a_w5a_first` (0.8386 -> 0.3758) |
| tvae_ind_seed0 | corr-ratio (num-cat) | 6840 | 0.014 | 0.0 | 0.9671 | 0 | `lab_results_potassium_value_last|hyperkalemia_severity_categorizedValue` (0.5223 -> 0.0) |
| tvae_qt_seed0 | Spearman (num-num) | 630 | 0.0912 | 0.0607 | 0.6556 | 1 | `vital_signs_oxygenSaturation_value_last|nyha_nyha` (-0.1437 -> -0.6001) |
| tvae_qt_seed0 | Cramer's V (cat-cat) | 561 | 0.1017 | 0.0341 | 0.7112 | 0 | `encounter_primary_reason_non_CV_Disease_f5a_w6mo_first|encounter_primary_reason_renal_complications_f5a_w3mo_first` (0.8411 -> 0.0009) |
| tvae_qt_seed0 | corr-ratio (num-cat) | 6840 | 0.0153 | 0.0 | 0.963 | 1 | `conditions_heartFailure_timeFromEarliest_first|conditions_heart_failure_occurred_prior_to_18_months_any` (0.8642 -> 0.0) |
| tvae_qt_seed1 | Spearman (num-num) | 630 | 0.0949 | 0.0758 | 0.6079 | 0 | `vital_signs_oxygenSaturation_value_first|nyha_nyha_pET` (-0.1451 -> -0.6279) |
| tvae_qt_seed1 | Cramer's V (cat-cat) | 820 | 0.0784 | 0.0369 | 0.7061 | 0 | `encounter_primary_reason_HF_Disease_f5a_w1mo_first|encounter_primary_reason_non_CV_Disease_f5a_w3mo_first` (0.5444 -> 1.0) |
| tvae_qt_seed1 | corr-ratio (num-cat) | 6840 | 0.0129 | 0.0 | 0.97 | 0 | `lab_results_potassium_value_last|hyperkalemia_severity_categorizedValue` (0.5223 -> 0.0) |
| tvae_qt_seed2 | Spearman (num-num) | 630 | 0.091 | 0.0673 | 0.6667 | 0 | `vital_signs_oxygenSaturation_value_first|nyha_nyha` (-0.1449 -> -0.6639) |
| tvae_qt_seed2 | Cramer's V (cat-cat) | 820 | 0.0972 | 0.0454 | 0.6817 | 0 | `encounter_primary_reason_HF_Disease_f5a_w3mo_first|encounter_primary_reason_HF_Disease_f5a_w3a_first` (0.8419 -> 0.3167) |
| tvae_qt_seed2 | corr-ratio (num-cat) | 6840 | 0.0147 | 0.0 | 0.9671 | 0 | `lab_results_potassium_value_last|hyperkalemia_severity_categorizedValue` (0.5223 -> 0.0) |
| tvae_seed0 | Spearman (num-num) | 630 | 0.0679 | 0.0471 | 0.7635 | 0 | `vital_signs_heartRate_value_first|vital_signs_heartRate_value_last` (0.7827 -> 0.2565) |
| tvae_seed0 | Cramer's V (cat-cat) | 903 | 0.0776 | 0.031 | 0.7386 | 0 | `encounter_primary_reason_HF_Disease_f5a_w1mo_first|encounter_primary_reason_CV_Disease_f5a_w3a_first` (0.3632 -> 1.0) |
| tvae_seed0 | corr-ratio (num-cat) | 6840 | 0.0117 | 0.0 | 0.9756 | 0 | `lab_results_potassium_value_last|hyperkalemia_severity_categorizedValue` (0.5223 -> 0.0478) |
| tvae_seed1 | Spearman (num-num) | 630 | 0.0615 | 0.0448 | 0.8111 | 0 | `vital_signs_heartRate_value_first|vital_signs_heartRate_value_last` (0.7827 -> 0.196) |
| tvae_seed1 | Cramer's V (cat-cat) | 703 | 0.0687 | 0.0285 | 0.7553 | 0 | `encounter_primary_reason_HF_Disease_f5a_w3mo_first|encounter_primary_reason_HF_Disease_f5a_w5a_first` (0.8386 -> 0.3227) |
| tvae_seed1 | corr-ratio (num-cat) | 6840 | 0.0129 | 0.0 | 0.9709 | 1 | `lab_results_potassium_value_last|hyperkalemia_severity_categorizedValue` (0.5223 -> 0.0102) |
| tvae_seed2 | Spearman (num-num) | 666 | 0.0717 | 0.0575 | 0.7583 | 0 | `lab_results_sodium_value_first|lab_results_sodium_value_last` (0.4951 -> 0.0229) |
| tvae_seed2 | Cramer's V (cat-cat) | 946 | 0.0666 | 0.0297 | 0.7674 | 0 | `encounter_primary_reason_HF_Disease_f5a_w6mo_first|encounter_primary_reason_HF_Disease_f5a_w5a_first` (0.8893 -> 0.4869) |
| tvae_seed2 | corr-ratio (num-cat) | 7030 | 0.0129 | 0.0 | 0.9696 | 0 | `lab_results_potassium_value_last|hyperkalemia_severity_categorizedValue` (0.5223 -> 0.0519) |

## original vs preprocessed

Worst numeric columns (by KS):
- `vital_signs_systolicBp_value_first`: KS=0.0014, W/std=0.0065, mean 133.5775 -> 133.7505, missing 20% -> 20%
- `vital_signs_systolicBp_value_last`: KS=0.001, W/std=0.0048, mean 133.4869 -> 133.6138, missing 20% -> 20%
- `lab_results_potassium_value_last`: KS=0.0005, W/std=0.0072, mean 4.0495 -> 4.0451, missing 0% -> 0%
- `lab_results_triGly_value_first`: KS=0.0002, W/std=0.0065, mean 1.4861 -> 1.4787, missing 9% -> 9%
- `lab_results_triGly_value_last`: KS=0.0002, W/std=0.0081, mean 1.4222 -> 1.4121, missing 9% -> 9%
Worst categorical columns (by TVD):
- `patient_demographics_gender`: TVD=0.0, 2 -> 2 categories, missing 0% -> 0%
- `encounters_encounterClass`: TVD=0.0, 1 -> 1 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.0, 6 -> 6 categories, missing 0% -> 0%
- `symptoms_Ankle_swelling_display_pET_any`: TVD=0.0, 1 -> 1 categories, missing 0% -> 0%
- `symptoms_Ascites_display_pET_any`: TVD=0.0, 1 -> 1 categories, missing 0% -> 0%

## train vs holdout

Worst numeric columns (by KS):
- `echocardiographs_lvef`: KS=0.0707, W/std=0.111, mean 38.816 -> 37.2285, missing 63% -> 64%
- `echocardiographs_lvef_pET_last`: KS=0.0702, W/std=0.1105, mean 39.0281 -> 37.4495, missing 64% -> 65%
- `echocardiographs_lvef_pET_first`: KS=0.0517, W/std=0.0792, mean 38.9898 -> 37.8679, missing 64% -> 65%
- `vital_signs_oxygenSaturation_value_last`: KS=0.0501, W/std=0.088, mean 95.624 -> 95.3634, missing 47% -> 49%
- `vital_signs_oxygenSaturation_value_first`: KS=0.0449, W/std=0.0906, mean 95.7072 -> 95.4043, missing 47% -> 49%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.0521, 6 -> 6 categories, missing 0% -> 0%
- `conditions_dysl`: TVD=0.0238, 2 -> 2 categories, missing 0% -> 0%
- `patient_demographics_gender`: TVD=0.0117, 2 -> 2 categories, missing 0% -> 0%
- `conditions_osa`: TVD=0.0111, 2 -> 2 categories, missing 0% -> 0%
- `conditions_revasc`: TVD=0.0111, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[ctgan_qt_seed0]

Worst numeric columns (by KS):
- `echocardiographs_lvef_pET_first`: KS=0.5823, W/std=1.4006, mean 38.9898 -> 59.0298, missing 64% -> 85%
- `vital_signs_oxygenSaturation_value_first`: KS=0.4339, W/std=0.5181, mean 95.7072 -> 98.3225, missing 47% -> 69%
- `lab_results_validSerumCreatinine_value_pET`: KS=0.3328, W/std=1.0068, mean 10.2439 -> 13.8163, missing 0% -> 0%
- `echocardiographs_lvef_pET_last`: KS=0.3244, W/std=0.7524, mean 39.0281 -> 49.8398, missing 64% -> 76%
- `vital_signs_systolicBp_value_last`: KS=0.3117, W/std=1.2238, mean 133.4109 -> 165.2645, missing 20% -> 14%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.2418, 6 -> 6 categories, missing 0% -> 0%
- `conditions_ihd`: TVD=0.2171, 2 -> 2 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.1855, 5 -> 5 categories, missing 0% -> 0%
- `encounter_primary_reason_non_CV_Disease_f5a_w3a_first`: TVD=0.1592, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_HF_Disease_f5a_w3a_first`: TVD=0.1584, 3 -> 3 categories, missing 0% -> 0%

## train vs synthetic[ctgan_seed0]

Worst numeric columns (by KS):
- `vital_signs_oxygenSaturation_value_first`: KS=0.6244, W/std=0.4181, mean 95.7072 -> 97.802, missing 47% -> 28%
- `vital_signs_oxygenSaturation_value_last`: KS=0.5475, W/std=0.3277, mean 95.624 -> 97.1593, missing 47% -> 38%
- `vital_signs_systolicBp_value_first`: KS=0.5091, W/std=1.2395, mean 133.625 -> 165.5539, missing 20% -> 16%
- `vital_signs_systolicBp_value_last`: KS=0.4231, W/std=0.8827, mean 133.4109 -> 110.4488, missing 20% -> 16%
- `vital_signs_diastolicBp_value_last`: KS=0.4215, W/std=0.7921, mean 77.4992 -> 66.5721, missing 21% -> 43%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.3555, 6 -> 6 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.2935, 5 -> 5 categories, missing 0% -> 0%
- `conditions_hyp`: TVD=0.2299, 2 -> 2 categories, missing 0% -> 0%
- `patient_demographics_gender`: TVD=0.2108, 2 -> 2 categories, missing 0% -> 0%
- `conditions_cm`: TVD=0.1908, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[ctgan_seed1]

Worst numeric columns (by KS):
- `vital_signs_oxygenSaturation_value_first`: KS=0.5904, W/std=0.3679, mean 95.7072 -> 97.4861, missing 47% -> 49%
- `vital_signs_oxygenSaturation_value_last`: KS=0.533, W/std=0.94, mean 95.624 -> 91.071, missing 47% -> 50%
- `lab_results_sodium_value_last`: KS=0.5192, W/std=1.2414, mean 137.8027 -> 132.9021, missing 1% -> 1%
- `lab_results_cholTot_value_first`: KS=0.4977, W/std=1.1189, mean 4.2516 -> 2.7547, missing 7% -> 18%
- `lab_results_validSerumCreatinine_value_first`: KS=0.4892, W/std=0.9061, mean 10.6222 -> 7.0265, missing 0% -> 0%
Worst categorical columns (by TVD):
- `patient_demographics_gender`: TVD=0.2831, 2 -> 2 categories, missing 0% -> 0%
- `conditions_ihd`: TVD=0.2449, 2 -> 2 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.2427, 5 -> 5 categories, missing 0% -> 0%
- `conditions_ckd_chronic`: TVD=0.2207, 2 -> 2 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.193, 6 -> 6 categories, missing 0% -> 0%

## train vs synthetic[ctgan_seed2]

Worst numeric columns (by KS):
- `vital_signs_oxygenSaturation_value_last`: KS=0.7552, W/std=0.6089, mean 95.624 -> 98.8042, missing 47% -> 64%
- `vital_signs_systolicBp_value_first`: KS=0.5807, W/std=1.4176, mean 133.625 -> 96.9239, missing 20% -> 23%
- `vital_signs_heartRate_value_first`: KS=0.5074, W/std=1.4628, mean 83.1901 -> 121.0592, missing 40% -> 18%
- `vital_signs_systolicBp_value_last`: KS=0.4858, W/std=1.041, mean 133.4109 -> 106.4425, missing 20% -> 23%
- `vital_signs_oxygenSaturation_value_first`: KS=0.4826, W/std=0.6406, mean 95.7072 -> 93.0752, missing 47% -> 24%
Worst categorical columns (by TVD):
- `conditions_dysl`: TVD=0.2674, 2 -> 2 categories, missing 0% -> 0%
- `conditions_af`: TVD=0.2453, 2 -> 2 categories, missing 0% -> 0%
- `conditions_vd`: TVD=0.1866, 2 -> 2 categories, missing 0% -> 0%
- `encounter_primary_reason_non_CV_Disease_f5a_w5a_first`: TVD=0.179, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_HF_Disease_f5a_w3a_first`: TVD=0.1777, 3 -> 3 categories, missing 0% -> 0%

## train vs synthetic[ddpm_g_seed0]

Worst numeric columns (by KS):
- `maggic_total_score`: KS=0.7688, W/std=2.2694, mean 25.5179 -> 40.3287, missing 64% -> 81%
- `echocardiographs_lvef_pET_last`: KS=0.7208, W/std=2.4056, mean 39.0281 -> 73.5996, missing 64% -> 81%
- `echocardiographs_lvef_pET_first`: KS=0.7116, W/std=2.3973, mean 38.9898 -> 73.2937, missing 64% -> 81%
- `echocardiographs_lvef`: KS=0.7059, W/std=2.3538, mean 38.816 -> 72.609, missing 63% -> 80%
- `vital_signs_oxygenSaturation_value_last`: KS=0.2695, W/std=0.3055, mean 95.624 -> 97.1988, missing 47% -> 66%
Worst categorical columns (by TVD):
- `conditions_ihd`: TVD=0.4373, 2 -> 1 categories, missing 0% -> 0%
- `conditions_revasc`: TVD=0.2788, 2 -> 2 categories, missing 0% -> 0%
- `conditions_mi`: TVD=0.1921, 2 -> 1 categories, missing 0% -> 0%
- `conditions_ap`: TVD=0.1827, 2 -> 1 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.1337, 6 -> 5 categories, missing 0% -> 0%

## train vs synthetic[ddpm_seed0]

Worst numeric columns (by KS):
- `maggic_total_score`: KS=0.7618, W/std=2.2591, mean 25.5179 -> 40.2609, missing 64% -> 82%
- `echocardiographs_lvef_pET_last`: KS=0.6984, W/std=2.3724, mean 39.0281 -> 73.1226, missing 64% -> 81%
- `echocardiographs_lvef`: KS=0.6982, W/std=2.3705, mean 38.816 -> 72.8485, missing 63% -> 81%
- `echocardiographs_lvef_pET_first`: KS=0.6981, W/std=2.4016, mean 38.9898 -> 73.3546, missing 64% -> 81%
- `vital_signs_oxygenSaturation_value_last`: KS=0.2876, W/std=0.3227, mean 95.624 -> 97.3093, missing 47% -> 64%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.128, 6 -> 5 categories, missing 0% -> 0%
- `conditions_cm`: TVD=0.0935, 2 -> 2 categories, missing 0% -> 0%
- `conditions_hyp`: TVD=0.0918, 2 -> 2 categories, missing 0% -> 0%
- `conditions_mi`: TVD=0.0915, 2 -> 2 categories, missing 0% -> 0%
- `conditions_revasc`: TVD=0.0852, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[ddpm_seed1]

Worst numeric columns (by KS):
- `maggic_total_score`: KS=0.6212, W/std=1.8792, mean 25.5179 -> 37.7821, missing 64% -> 83%
- `echocardiographs_lvef_pET_last`: KS=0.5137, W/std=1.6414, mean 39.0281 -> 62.6168, missing 64% -> 81%
- `echocardiographs_lvef_pET_first`: KS=0.5118, W/std=1.6229, mean 38.9898 -> 62.2114, missing 64% -> 81%
- `echocardiographs_lvef`: KS=0.5107, W/std=1.6772, mean 38.816 -> 62.8961, missing 63% -> 80%
- `vital_signs_heartRate_value_last`: KS=0.1823, W/std=0.2534, mean 82.9585 -> 88.3622, missing 40% -> 49%
Worst categorical columns (by TVD):
- `ckd_severity_calculated_or_measured`: TVD=0.1205, 5 -> 5 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.1171, 5 -> 5 categories, missing 0% -> 0%
- `conditions_diabetes`: TVD=0.0955, 2 -> 2 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.0941, 6 -> 5 categories, missing 0% -> 0%
- `conditions_revasc`: TVD=0.0894, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[ddpm_seed2]

Worst numeric columns (by KS):
- `echocardiographs_lvef_pET_last`: KS=0.6863, W/std=2.3475, mean 39.0281 -> 72.7644, missing 64% -> 85%
- `maggic_total_score`: KS=0.6819, W/std=2.0291, mean 25.5179 -> 38.7603, missing 64% -> 85%
- `echocardiographs_lvef_pET_first`: KS=0.6791, W/std=2.3375, mean 38.9898 -> 72.4388, missing 64% -> 85%
- `echocardiographs_lvef`: KS=0.6755, W/std=2.3201, mean 38.816 -> 72.126, missing 63% -> 85%
- `vital_signs_oxygenSaturation_value_first`: KS=0.3506, W/std=0.3652, mean 95.7072 -> 97.5103, missing 47% -> 68%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.1265, 6 -> 5 categories, missing 0% -> 0%
- `conditions_hyp`: TVD=0.0837, 2 -> 2 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.0828, 5 -> 5 categories, missing 0% -> 0%
- `conditions_mi`: TVD=0.0824, 2 -> 2 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.0822, 5 -> 5 categories, missing 0% -> 0%

## train vs synthetic[dpctgan_eps10_seed0]

Worst numeric columns (by KS):
- `vital_signs_diastolicBp_value_last`: KS=1.0, W/std=None, mean 77.4992 -> None, missing 21% -> 100%
- `vital_signs_heartRate_value_last`: KS=1.0, W/std=None, mean 82.9585 -> None, missing 40% -> 100%
- `vital_signs_systolicBp_value_first`: KS=1.0, W/std=None, mean 133.625 -> None, missing 20% -> 100%
- `vital_signs_systolicBp_value_last`: KS=1.0, W/std=None, mean 133.4109 -> None, missing 20% -> 100%
- `lab_results_triGly_value_last`: KS=1.0, W/std=None, mean 1.4155 -> None, missing 9% -> 100%
Worst categorical columns (by TVD):
- `encounter_primary_reason_HF_Disease_f5a_w1mo_first`: TVD=0.9817, 3 -> 2 categories, missing 0% -> 0%
- `encounter_primary_reason_renal_complications_f5a_w3a_first`: TVD=0.9739, 2 -> 2 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.8172, 6 -> 3 categories, missing 0% -> 0%
- `conditions_cm`: TVD=0.7565, 2 -> 2 categories, missing 0% -> 0%
- `conditions_mi`: TVD=0.6782, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[dpctgan_eps15_seed0]

Worst numeric columns (by KS):
- `vital_signs_diastolicBp_value_first`: KS=1.0, W/std=None, mean 77.5775 -> None, missing 21% -> 100%
- `lab_results_hemoglobin_value_last`: KS=1.0, W/std=None, mean 129.3242 -> None, missing 1% -> 100%
- `lab_results_validSerumCreatinine_value_first`: KS=1.0, W/std=4.8778, mean 10.6222 -> 29.9782, missing 0% -> 0%
- `vital_signs_bmi_value_pET`: KS=1.0, W/std=6.2574, mean 21.7717 -> 10.0301, missing 0% -> 0%
- `lab_results_creatBS_value_p3a_avg`: KS=1.0, W/std=2.2244, mean 10.956 -> 0.0088, missing 0% -> 0%
Worst categorical columns (by TVD):
- `encounter_primary_reason_HF_Disease_f5a_w3a_first`: TVD=0.8929, 3 -> 2 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.7263, 5 -> 5 categories, missing 0% -> 0%
- `conditions_ap`: TVD=0.7008, 2 -> 2 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.5903, 5 -> 5 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.5901, 6 -> 6 categories, missing 0% -> 0%

## train vs synthetic[dpctgan_eps15_seed1]

Worst numeric columns (by KS):
- `vital_signs_heartRate_value_first`: KS=1.0, W/std=None, mean 83.1901 -> None, missing 40% -> 100%
- `lab_results_potassium_value_last`: KS=1.0, W/std=None, mean 4.0491 -> None, missing 1% -> 100%
- `echocardiographs_lvef_pET_last`: KS=1.0, W/std=4.162, mean 39.0281 -> 98.8401, missing 64% -> 0%
- `vital_signs_bmi_value_pET`: KS=1.0, W/std=6.1798, mean 21.7717 -> 10.1757, missing 0% -> 0%
- `lab_results_creatBS_value_p3a_avg`: KS=1.0, W/std=2.2256, mean 10.956 -> 0.0028, missing 0% -> 0%
Worst categorical columns (by TVD):
- `encounter_primary_reason_renal_complications_f5a_w5a_first`: TVD=0.9369, 2 -> 2 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.8245, 5 -> 3 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.8177, 6 -> 3 categories, missing 0% -> 0%
- `patient_demographics_gender`: TVD=0.6256, 2 -> 2 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.6228, 5 -> 3 categories, missing 0% -> 0%

## train vs synthetic[dpctgan_eps15_seed2]

Worst numeric columns (by KS):
- `patient_demographics_age`: KS=1.0, W/std=3.7784, mean 67.404 -> 18.1187, missing 0% -> 0%
- `vital_signs_heartRate_value_first`: KS=1.0, W/std=None, mean 83.1901 -> None, missing 40% -> 100%
- `vital_signs_heartRate_value_last`: KS=1.0, W/std=None, mean 82.9585 -> None, missing 40% -> 100%
- `vital_signs_systolicBp_value_first`: KS=1.0, W/std=3.8323, mean 133.625 -> 34.247, missing 20% -> 100%
- `vital_signs_systolicBp_value_last`: KS=1.0, W/std=None, mean 133.4109 -> None, missing 20% -> 100%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.8209, 6 -> 2 categories, missing 0% -> 0%
- `conditions_heart_failure_occurred_prior_to_18_months_any`: TVD=0.7111, 2 -> 2 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.6379, 5 -> 1 categories, missing 0% -> 0%
- `patient_demographics_gender`: TVD=0.6279, 2 -> 1 categories, missing 0% -> 0%
- `conditions_ihd`: TVD=0.5618, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[dpctgan_eps1_seed0]

Worst numeric columns (by KS):
- `vital_signs_diastolicBp_value_last`: KS=1.0, W/std=None, mean 77.4992 -> None, missing 21% -> 100%
- `vital_signs_heartRate_value_first`: KS=1.0, W/std=None, mean 83.1901 -> None, missing 40% -> 100%
- `vital_signs_heartRate_value_last`: KS=1.0, W/std=None, mean 82.9585 -> None, missing 40% -> 100%
- `lab_results_triGly_value_first`: KS=1.0, W/std=None, mean 1.4815 -> None, missing 9% -> 100%
- `lab_results_triGly_value_last`: KS=1.0, W/std=None, mean 1.4155 -> None, missing 9% -> 100%
Worst categorical columns (by TVD):
- `conditions_mc`: TVD=0.9443, 2 -> 2 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.745, 6 -> 6 categories, missing 0% -> 0%
- `conditions_hyp`: TVD=0.6425, 2 -> 2 categories, missing 0% -> 0%
- `patient_demographics_gender`: TVD=0.622, 2 -> 2 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.5588, 5 -> 5 categories, missing 0% -> 0%

## train vs synthetic[dpctgan_eps20_seed0]

Worst numeric columns (by KS):
- `vital_signs_diastolicBp_value_first`: KS=1.0, W/std=11.8856, mean 77.5775 -> 242.3233, missing 21% -> 0%
- `vital_signs_heartRate_value_first`: KS=1.0, W/std=None, mean 83.1901 -> None, missing 40% -> 100%
- `vital_signs_heartRate_value_last`: KS=1.0, W/std=None, mean 82.9585 -> None, missing 40% -> 100%
- `lab_results_hemoglobin_value_last`: KS=1.0, W/std=7.7997, mean 129.3242 -> 297.5702, missing 1% -> 0%
- `lab_results_triGly_value_first`: KS=1.0, W/std=None, mean 1.4815 -> None, missing 9% -> 100%
Worst categorical columns (by TVD):
- `conditions_osa`: TVD=0.9165, 2 -> 2 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.7794, 6 -> 2 categories, missing 0% -> 0%
- `conditions_cm`: TVD=0.752, 2 -> 2 categories, missing 0% -> 0%
- `conditions_hyp`: TVD=0.7371, 2 -> 2 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.6366, 5 -> 3 categories, missing 0% -> 0%

## train vs synthetic[dpctgan_eps5_seed0]

Worst numeric columns (by KS):
- `vital_signs_heartRate_value_last`: KS=1.0, W/std=None, mean 82.9585 -> None, missing 40% -> 100%
- `lab_results_triGly_value_first`: KS=1.0, W/std=None, mean 1.4815 -> None, missing 9% -> 100%
- `lab_results_cholTot_value_last`: KS=1.0, W/std=None, mean 4.1602 -> None, missing 7% -> 100%
- `echocardiographs_lvef_pET_first`: KS=1.0, W/std=4.2449, mean 38.9898 -> 99.7315, missing 64% -> 0%
- `vital_signs_bmi_value_pET`: KS=1.0, W/std=6.2732, mean 21.7717 -> 10.0005, missing 0% -> 0%
Worst categorical columns (by TVD):
- `ckd_severity_from_calculated_egfr`: TVD=0.8965, 5 -> 5 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.8702, 5 -> 4 categories, missing 0% -> 0%
- `conditions_ap`: TVD=0.8117, 2 -> 2 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.7539, 6 -> 3 categories, missing 0% -> 0%
- `conditions_hyp`: TVD=0.6795, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[dpctgan_eps8_seed0]

Worst numeric columns (by KS):
- `vital_signs_heartRate_value_first`: KS=1.0, W/std=None, mean 83.1901 -> None, missing 40% -> 100%
- `vital_signs_systolicBp_value_first`: KS=1.0, W/std=3.8558, mean 133.625 -> 33.6375, missing 20% -> 100%
- `lab_results_hemoglobin_value_last`: KS=1.0, W/std=None, mean 129.3242 -> None, missing 1% -> 100%
- `lab_results_triGly_value_first`: KS=1.0, W/std=None, mean 1.4815 -> None, missing 9% -> 100%
- `lab_results_potassium_value_last`: KS=1.0, W/std=None, mean 4.0491 -> None, missing 1% -> 100%
Worst categorical columns (by TVD):
- `conditions_cm`: TVD=0.7544, 2 -> 2 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.6235, 5 -> 2 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.5903, 5 -> 5 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.5416, 6 -> 6 categories, missing 0% -> 0%
- `conditions_vd`: TVD=0.476, 2 -> 1 categories, missing 0% -> 0%

## train vs synthetic[gaussian_copula_seed0]

Worst numeric columns (by KS):
- `encounter_primary_reason_number_of_non_CV_rehospitalizations_5a_f5a_count`: KS=0.9768, W/std=4.6422, mean 0.0958 -> 3.456, missing 0% -> 0%
- `encounter_primary_reason_number_of_CV_rehospitalizations_5a_f5a_count`: KS=0.9754, W/std=5.8133, mean 0.1177 -> 5.0697, missing 0% -> 0%
- `vital_signs_diastolicBp_value_last`: KS=0.8239, W/std=6.8739, mean 77.4992 -> 168.3509, missing 21% -> 17%
- `vital_signs_diastolicBp_value_first`: KS=0.8159, W/std=6.7864, mean 77.5775 -> 167.0834, missing 21% -> 17%
- `vital_signs_oxygenSaturation_value_last`: KS=0.7979, W/std=9.1382, mean 95.624 -> 48.0894, missing 47% -> 66%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.2566, 6 -> 5 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.0254, 5 -> 5 categories, missing 0% -> 0%
- `conditions_cm`: TVD=0.0193, 2 -> 2 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.0155, 5 -> 5 categories, missing 0% -> 0%
- `conditions_revasc`: TVD=0.015, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[gaussian_copula_seed1]

Worst numeric columns (by KS):
- `encounter_primary_reason_number_of_non_CV_rehospitalizations_5a_f5a_count`: KS=0.9768, W/std=4.6987, mean 0.0958 -> 3.4966, missing 0% -> 0%
- `encounter_primary_reason_number_of_CV_rehospitalizations_5a_f5a_count`: KS=0.9754, W/std=5.9052, mean 0.1177 -> 5.148, missing 0% -> 0%
- `vital_signs_diastolicBp_value_last`: KS=0.8498, W/std=7.1143, mean 77.4992 -> 172.5044, missing 21% -> 17%
- `vital_signs_diastolicBp_value_first`: KS=0.8462, W/std=7.0398, mean 77.5775 -> 171.6956, missing 21% -> 18%
- `vital_signs_oxygenSaturation_value_first`: KS=0.8207, W/std=9.7058, mean 95.7072 -> 46.839, missing 47% -> 65%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.2611, 6 -> 5 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.0218, 5 -> 5 categories, missing 0% -> 0%
- `conditions_ap`: TVD=0.0198, 2 -> 2 categories, missing 0% -> 0%
- `conditions_ckd_chronic`: TVD=0.0161, 2 -> 2 categories, missing 0% -> 0%
- `conditions_vd`: TVD=0.0145, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[gaussian_copula_seed2]

Worst numeric columns (by KS):
- `encounter_primary_reason_number_of_non_CV_rehospitalizations_5a_f5a_count`: KS=0.9768, W/std=4.8574, mean 0.0958 -> 3.6116, missing 0% -> 0%
- `encounter_primary_reason_number_of_CV_rehospitalizations_5a_f5a_count`: KS=0.9754, W/std=6.0379, mean 0.1177 -> 5.2611, missing 0% -> 0%
- `vital_signs_diastolicBp_value_last`: KS=0.8309, W/std=6.9549, mean 77.4992 -> 170.043, missing 21% -> 17%
- `vital_signs_diastolicBp_value_first`: KS=0.825, W/std=6.8633, mean 77.5775 -> 168.8091, missing 21% -> 17%
- `vital_signs_oxygenSaturation_value_first`: KS=0.8126, W/std=9.7279, mean 95.7072 -> 46.7712, missing 47% -> 63%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.2525, 6 -> 5 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.0296, 5 -> 5 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.0199, 5 -> 5 categories, missing 0% -> 0%
- `conditions_mi`: TVD=0.0194, 2 -> 2 categories, missing 0% -> 0%
- `conditions_af`: TVD=0.0145, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[mst_eps15_seed0]

Worst numeric columns (by KS):
- `encounters_numOfPreviousHFStays_count`: KS=1.0, W/std=79.5193, mean 0.2214 -> 51.5306, missing 0% -> 0%
- `encounter_primary_reason_number_of_HF_rehospitalizations_5a_f5a_count`: KS=1.0, W/std=30.0909, mean 0.0062 -> 2.5191, missing 0% -> 0%
- `encounter_primary_reason_number_of_renal_rehospitalizations_5a_f5a_count`: KS=0.9998, W/std=38.6042, mean 0.003 -> 2.517, missing 0% -> 0%
- `lab_results_creatBS_value_p3a_avg`: KS=0.9989, W/std=8.0499, mean 10.956 -> 50.574, missing 0% -> 0%
- `conditions_heartFailure_timeFromEarliest_first`: KS=0.9894, W/std=5.6639, mean 1.2333 -> 30.4464, missing 0% -> 0%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.0219, 6 -> 6 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.0169, 5 -> 5 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.0166, 5 -> 5 categories, missing 0% -> 0%
- `conditions_ihd`: TVD=0.0143, 2 -> 2 categories, missing 0% -> 0%
- `conditions_vd`: TVD=0.0104, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[patectgan_eps15_seed0]

Worst numeric columns (by KS):
- `encounter_primary_reason_number_of_renal_rehospitalizations_5a_f5a_count`: KS=0.9732, W/std=0.0459, mean 0.003 -> 0.0, missing 0% -> 0%
- `encounter_primary_reason_number_of_HF_rehospitalizations_5a_f5a_count`: KS=0.9639, W/std=0.0739, mean 0.0062 -> 0.0, missing 0% -> 0%
- `encounter_primary_reason_number_of_CV_rehospitalizations_5a_f5a_count`: KS=0.9632, W/std=0.1382, mean 0.1177 -> 0.0, missing 0% -> 0%
- `encounter_primary_reason_number_of_non_CV_rehospitalizations_5a_f5a_count`: KS=0.9565, W/std=0.1324, mean 0.0958 -> 0.0, missing 0% -> 0%
- `conditions_heartFailure_timeFromEarliest_first`: KS=0.8663, W/std=0.1536, mean 1.2333 -> 0.7261, missing 0% -> 0%
Worst categorical columns (by TVD):
- `ckd_severity_calculated_or_measured`: TVD=0.6224, 5 -> 3 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.6218, 5 -> 3 categories, missing 0% -> 0%
- `patient_demographics_gender`: TVD=0.093, 2 -> 2 categories, missing 0% -> 0%
- `conditions_ckd_chronic`: TVD=0.0497, 2 -> 2 categories, missing 0% -> 0%
- `conditions_mi`: TVD=0.0416, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[patectgan_eps1_seed0]

Worst numeric columns (by KS):
- `encounters_numOfPreviousHFStays_count`: KS=0.9987, W/std=101.6742, mean 0.2214 -> 65.8259, missing 0% -> 0%
- `encounter_primary_reason_number_of_renal_rehospitalizations_5a_f5a_count`: KS=0.9975, W/std=39.0794, mean 0.003 -> 2.5479, missing 0% -> 0%
- `encounter_primary_reason_number_of_HF_rehospitalizations_5a_f5a_count`: KS=0.9943, W/std=37.424, mean 0.0062 -> 3.1315, missing 0% -> 0%
- `encounter_primary_reason_number_of_non_CV_rehospitalizations_5a_f5a_count`: KS=0.9768, W/std=4.6385, mean 0.0958 -> 3.4532, missing 0% -> 0%
- `encounter_primary_reason_number_of_CV_rehospitalizations_5a_f5a_count`: KS=0.9754, W/std=2.542, mean 0.1177 -> 2.2822, missing 0% -> 0%
Worst categorical columns (by TVD):
- `encounter_primary_reason_HF_Disease_f5a_w5a_first`: TVD=0.2817, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_HF_Disease_f5a_w6mo_first`: TVD=0.2432, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_HF_Disease_f5a_w1a_first`: TVD=0.2428, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_HF_Disease_f5a_w3a_first`: TVD=0.2419, 3 -> 3 categories, missing 0% -> 0%
- `encounter_primary_reason_renal_complications_f5a_w1a_first`: TVD=0.2409, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[patectgan_eps5_seed0]

Worst numeric columns (by KS):
- `encounter_primary_reason_number_of_CV_rehospitalizations_5a_f5a_count`: KS=0.8725, W/std=0.1382, mean 0.1177 -> 0.0, missing 0% -> 0%
- `encounter_primary_reason_number_of_HF_rehospitalizations_5a_f5a_count`: KS=0.8529, W/std=0.0738, mean 0.0062 -> 0.0, missing 0% -> 0%
- `encounter_primary_reason_number_of_renal_rehospitalizations_5a_f5a_count`: KS=0.8316, W/std=0.0457, mean 0.003 -> 0.0, missing 0% -> 0%
- `conditions_heartFailure_timeFromEarliest_first`: KS=0.8219, W/std=0.238, mean 1.2333 -> 0.0001, missing 0% -> 0%
- `encounter_primary_reason_number_of_non_CV_rehospitalizations_5a_f5a_count`: KS=0.7832, W/std=0.1324, mean 0.0958 -> 0.0, missing 0% -> 0%
Worst categorical columns (by TVD):
- `ckd_severity_calculated_or_measured`: TVD=0.6177, 5 -> 5 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.615, 5 -> 5 categories, missing 0% -> 0%
- `patient_demographics_gender`: TVD=0.0876, 2 -> 2 categories, missing 0% -> 0%
- `conditions_mi`: TVD=0.0773, 2 -> 2 categories, missing 0% -> 0%
- `conditions_ap`: TVD=0.069, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[tvae_cap256_seed0]

Worst numeric columns (by KS):
- `lab_results_sodium_value_last`: KS=0.3164, W/std=0.4999, mean 137.8027 -> 139.7416, missing 1% -> 0%
- `vital_signs_oxygenSaturation_value_first`: KS=0.2312, W/std=0.1687, mean 95.7072 -> 96.1774, missing 47% -> 35%
- `vital_signs_heartRate_value_first`: KS=0.2261, W/std=0.3508, mean 83.1901 -> 76.967, missing 40% -> 36%
- `vital_signs_heartRate_value_last`: KS=0.225, W/std=0.3032, mean 82.9585 -> 77.8195, missing 40% -> 42%
- `echocardiographs_lvef_pET_last`: KS=0.2157, W/std=0.2753, mean 39.0281 -> 35.1278, missing 64% -> 64%
Worst categorical columns (by TVD):
- `conditions_hyp`: TVD=0.1574, 2 -> 2 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.1498, 6 -> 6 categories, missing 0% -> 0%
- `conditions_dysl`: TVD=0.1486, 2 -> 2 categories, missing 0% -> 0%
- `patient_demographics_gender`: TVD=0.1327, 2 -> 2 categories, missing 0% -> 0%
- `conditions_ihd`: TVD=0.1295, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[tvae_ep1000_seed0]

Worst numeric columns (by KS):
- `lab_results_sodium_value_last`: KS=0.2693, W/std=0.3635, mean 137.8027 -> 139.1128, missing 1% -> 1%
- `lab_results_hemoglobin_value_first`: KS=0.269, W/std=0.4582, mean 133.4874 -> 123.0211, missing 1% -> 1%
- `vital_signs_bmi_value_pET`: KS=0.2602, W/std=0.4154, mean 21.7717 -> 22.0829, missing 0% -> 0%
- `lab_results_hemoglobin_value_last`: KS=0.2214, W/std=0.3806, mean 129.3242 -> 121.1173, missing 1% -> 1%
- `vital_signs_diastolicBp_value_last`: KS=0.2071, W/std=0.2312, mean 77.4992 -> 80.0726, missing 21% -> 23%
Worst categorical columns (by TVD):
- `conditions_hyp`: TVD=0.1587, 2 -> 2 categories, missing 0% -> 0%
- `conditions_diabetes`: TVD=0.1519, 2 -> 2 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.1391, 6 -> 6 categories, missing 0% -> 0%
- `conditions_vd`: TVD=0.1201, 2 -> 2 categories, missing 0% -> 0%
- `conditions_pad`: TVD=0.1122, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[tvae_ind_seed0]

Worst numeric columns (by KS):
- `vital_signs_bmi_value_pET`: KS=0.2876, W/std=0.4737, mean 21.7717 -> 21.2763, missing 0% -> 0%
- `vital_signs_heartRate_value_last`: KS=0.2764, W/std=0.261, mean 82.9585 -> 79.6018, missing 40% -> 34%
- `lab_results_sodium_value_last`: KS=0.2695, W/std=0.3983, mean 137.8027 -> 139.2632, missing 1% -> 0%
- `vital_signs_heartRate_value_first`: KS=0.2465, W/std=0.2584, mean 83.1901 -> 79.6347, missing 40% -> 33%
- `vital_signs_oxygenSaturation_value_first`: KS=0.2112, W/std=0.2595, mean 95.7072 -> 96.6722, missing 47% -> 38%
Worst categorical columns (by TVD):
- `conditions_diabetes`: TVD=0.2186, 2 -> 2 categories, missing 0% -> 0%
- `conditions_hyp`: TVD=0.1808, 2 -> 2 categories, missing 0% -> 0%
- `conditions_dysl`: TVD=0.161, 2 -> 2 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.1405, 6 -> 6 categories, missing 0% -> 0%
- `conditions_cm`: TVD=0.1343, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[tvae_qt_seed0]

Worst numeric columns (by KS):
- `lab_results_sodium_value_first`: KS=0.2711, W/std=0.3351, mean 139.6512 -> 140.9253, missing 1% -> 0%
- `vital_signs_heartRate_value_last`: KS=0.2239, W/std=0.2954, mean 82.9585 -> 76.4569, missing 40% -> 46%
- `echocardiographs_lvef`: KS=0.1983, W/std=0.3436, mean 38.816 -> 38.1916, missing 63% -> 61%
- `lab_results_sodium_value_last`: KS=0.1829, W/std=0.204, mean 137.8027 -> 137.1915, missing 1% -> 1%
- `lab_results_triGly_value_first`: KS=0.1549, W/std=0.3068, mean 1.4815 -> 1.7387, missing 9% -> 20%
Worst categorical columns (by TVD):
- `conditions_diabetes`: TVD=0.1629, 2 -> 2 categories, missing 0% -> 0%
- `conditions_hyp`: TVD=0.1512, 2 -> 2 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.1349, 6 -> 6 categories, missing 0% -> 0%
- `conditions_revasc`: TVD=0.1279, 2 -> 2 categories, missing 0% -> 0%
- `conditions_vd`: TVD=0.1193, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[tvae_qt_seed1]

Worst numeric columns (by KS):
- `lab_results_sodium_value_first`: KS=0.3107, W/std=0.3995, mean 139.6512 -> 141.1571, missing 1% -> 2%
- `echocardiographs_lvef_pET_first`: KS=0.2827, W/std=0.3816, mean 38.9898 -> 34.3008, missing 64% -> 62%
- `vital_signs_heartRate_value_last`: KS=0.2763, W/std=0.3395, mean 82.9585 -> 75.2282, missing 40% -> 52%
- `echocardiographs_lvef`: KS=0.2289, W/std=0.4108, mean 38.816 -> 33.1902, missing 63% -> 60%
- `lab_results_triGly_value_last`: KS=0.2027, W/std=0.3734, mean 1.4155 -> 1.7899, missing 9% -> 15%
Worst categorical columns (by TVD):
- `conditions_dysl`: TVD=0.2201, 2 -> 2 categories, missing 0% -> 0%
- `conditions_hyp`: TVD=0.1729, 2 -> 2 categories, missing 0% -> 0%
- `conditions_diabetes`: TVD=0.1649, 2 -> 2 categories, missing 0% -> 0%
- `conditions_ihd`: TVD=0.1648, 2 -> 2 categories, missing 0% -> 0%
- `conditions_af`: TVD=0.1399, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[tvae_qt_seed2]

Worst numeric columns (by KS):
- `vital_signs_heartRate_value_last`: KS=0.3025, W/std=0.4116, mean 82.9585 -> 73.3633, missing 40% -> 44%
- `lab_results_sodium_value_first`: KS=0.2697, W/std=0.37, mean 139.6512 -> 141.0824, missing 1% -> 2%
- `echocardiographs_lvef_pET_first`: KS=0.227, W/std=0.312, mean 38.9898 -> 36.5803, missing 64% -> 65%
- `echocardiographs_lvef`: KS=0.223, W/std=0.351, mean 38.816 -> 37.2189, missing 63% -> 65%
- `vital_signs_diastolicBp_value_first`: KS=0.183, W/std=0.2086, mean 77.5775 -> 76.7593, missing 21% -> 16%
Worst categorical columns (by TVD):
- `conditions_diabetes`: TVD=0.2501, 2 -> 2 categories, missing 0% -> 0%
- `conditions_hyp`: TVD=0.1415, 2 -> 2 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.1142, 6 -> 6 categories, missing 0% -> 0%
- `conditions_pad`: TVD=0.1135, 2 -> 1 categories, missing 0% -> 0%
- `conditions_devices`: TVD=0.1026, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[tvae_seed0]

Worst numeric columns (by KS):
- `vital_signs_bmi_value_pET`: KS=0.3087, W/std=0.4516, mean 21.7717 -> 22.2107, missing 0% -> 0%
- `lab_results_sodium_value_last`: KS=0.2791, W/std=0.3948, mean 137.8027 -> 139.2891, missing 1% -> 1%
- `vital_signs_heartRate_value_first`: KS=0.2681, W/std=0.4091, mean 83.1901 -> 73.9253, missing 40% -> 53%
- `vital_signs_heartRate_value_last`: KS=0.2666, W/std=0.4018, mean 82.9585 -> 74.5903, missing 40% -> 66%
- `lab_results_potassium_value_first`: KS=0.2407, W/std=0.3549, mean 4.3642 -> 4.1615, missing 0% -> 1%
Worst categorical columns (by TVD):
- `conditions_diabetes`: TVD=0.1802, 2 -> 2 categories, missing 0% -> 0%
- `conditions_hyp`: TVD=0.1644, 2 -> 2 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.1518, 6 -> 6 categories, missing 0% -> 0%
- `conditions_vd`: TVD=0.1312, 2 -> 2 categories, missing 0% -> 0%
- `conditions_pad`: TVD=0.113, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[tvae_seed1]

Worst numeric columns (by KS):
- `lab_results_sodium_value_last`: KS=0.2959, W/std=0.4634, mean 137.8027 -> 139.5916, missing 1% -> 0%
- `vital_signs_bmi_value_pET`: KS=0.2934, W/std=0.4709, mean 21.7717 -> 22.1992, missing 0% -> 0%
- `vital_signs_heartRate_value_last`: KS=0.248, W/std=0.3439, mean 82.9585 -> 76.699, missing 40% -> 59%
- `vital_signs_heartRate_value_first`: KS=0.2438, W/std=0.3655, mean 83.1901 -> 76.9537, missing 40% -> 45%
- `echocardiographs_lvef_pET_first`: KS=0.229, W/std=0.2875, mean 38.9898 -> 35.0153, missing 64% -> 66%
Worst categorical columns (by TVD):
- `encounters_admissionYear`: TVD=0.1961, 6 -> 6 categories, missing 0% -> 0%
- `conditions_hyp`: TVD=0.1742, 2 -> 2 categories, missing 0% -> 0%
- `conditions_dysl`: TVD=0.1588, 2 -> 2 categories, missing 0% -> 0%
- `conditions_ihd`: TVD=0.1374, 2 -> 2 categories, missing 0% -> 0%
- `patient_demographics_gender`: TVD=0.1225, 2 -> 2 categories, missing 0% -> 0%

## train vs synthetic[tvae_seed2]

Worst numeric columns (by KS):
- `lab_results_sodium_value_last`: KS=0.2884, W/std=0.4671, mean 137.8027 -> 139.6, missing 1% -> 1%
- `vital_signs_bmi_value_pET`: KS=0.222, W/std=0.4177, mean 21.7717 -> 21.8191, missing 0% -> 0%
- `vital_signs_oxygenSaturation_value_first`: KS=0.2084, W/std=0.1051, mean 95.7072 -> 95.7964, missing 47% -> 50%
- `echocardiographs_lvef`: KS=0.2079, W/std=0.2944, mean 38.816 -> 34.7814, missing 63% -> 62%
- `echocardiographs_lvef_pET_first`: KS=0.2011, W/std=0.2636, mean 38.9898 -> 35.9338, missing 64% -> 65%
Worst categorical columns (by TVD):
- `conditions_hyp`: TVD=0.1689, 2 -> 2 categories, missing 0% -> 0%
- `ckd_severity_from_calculated_egfr`: TVD=0.1512, 5 -> 4 categories, missing 0% -> 0%
- `ckd_severity_calculated_or_measured`: TVD=0.1384, 5 -> 4 categories, missing 0% -> 0%
- `conditions_vd`: TVD=0.1324, 2 -> 2 categories, missing 0% -> 0%
- `encounters_admissionYear`: TVD=0.1202, 6 -> 6 categories, missing 0% -> 0%
