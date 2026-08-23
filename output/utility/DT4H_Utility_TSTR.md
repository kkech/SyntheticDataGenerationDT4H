# Utility: Train-Synthetic, Test-Real (TSTR)

A gradient-boosting classifier is trained on the real TRAINING split (baseline) and on each synthetic dataset, then both are scored on the HOLDOUT split -- real patients that neither the generators nor either classifier ever saw. The closer the TSTR AUC is to the baseline, the more useful the synthetic data is for actual modelling work.

Real train: 3520 rows | holdout test: 1174 rows

## `encounter_primary_reason_non_CV_Disease_f5a_w1mo_first`
train 853 labelled (531 positive), holdout 264 labelled (168 positive) | baseline AUC **0.6861**

| run | TSTR AUC | gap vs baseline |
|---|---|---|
| aim50_eps1_seed0 | 0.5146 | +0.1715 |
| aim50_eps5_seed0 | 0.4906 | +0.1955 |
| ctgan_seed0 | 0.3633 | +0.3228 |
| ctgan_seed1 | 0.4407 | +0.2454 |
| ctgan_seed2 | 0.6207 | +0.0654 |
| dpctgan_eps10_seed0 | - | target missing or single-class in synthetic data |
| dpctgan_eps15_seed0 | 0.5114 | +0.1747 |
| dpctgan_eps15_seed1 | - | target missing or single-class in synthetic data |
| dpctgan_eps15_seed2 | - | target missing or single-class in synthetic data |
| dpctgan_eps1_seed0 | - | target missing or single-class in synthetic data |
| dpctgan_eps20_seed0 | - | target missing or single-class in synthetic data |
| dpctgan_eps5_seed0 | - | target missing or single-class in synthetic data |
| dpctgan_eps8_seed0 | - | target missing or single-class in synthetic data |
| gaussian_copula_seed0 | 0.5195 | +0.1666 |
| gaussian_copula_seed1 | 0.5101 | +0.1760 |
| gaussian_copula_seed2 | 0.5385 | +0.1476 |
| mst_eps10_seed0 | 0.5595 | +0.1266 |
| mst_eps15_seed0 | 0.5097 | +0.1764 |
| mst_eps15_seed1 | 0.522 | +0.1641 |
| mst_eps15_seed2 | 0.5321 | +0.1540 |
| mst_eps1_seed0 | 0.5767 | +0.1094 |
| mst_eps20_seed0 | 0.5172 | +0.1689 |
| mst_eps5_seed0 | 0.4684 | +0.2177 |
| mst_eps8_seed0 | 0.4669 | +0.2192 |
| tvae_seed0 | 0.5811 | +0.1050 |
| tvae_seed1 | 0.5763 | +0.1098 |
| tvae_seed2 | 0.5937 | +0.0924 |

## `encounter_primary_reason_CV_Disease_f5a_w1mo_first`
train 853 labelled (322 positive), holdout 264 labelled (96 positive) | baseline AUC **0.6861**

| run | TSTR AUC | gap vs baseline |
|---|---|---|
| aim50_eps1_seed0 | 0.5347 | +0.1514 |
| aim50_eps5_seed0 | 0.4905 | +0.1956 |
| ctgan_seed0 | 0.427 | +0.2591 |
| ctgan_seed1 | 0.5111 | +0.1750 |
| ctgan_seed2 | 0.5861 | +0.1000 |
| dpctgan_eps10_seed0 | - | target missing or single-class in synthetic data |
| dpctgan_eps15_seed0 | - | target missing or single-class in synthetic data |
| dpctgan_eps15_seed1 | - | target missing or single-class in synthetic data |
| dpctgan_eps15_seed2 | - | target missing or single-class in synthetic data |
| dpctgan_eps1_seed0 | - | target missing or single-class in synthetic data |
| dpctgan_eps20_seed0 | - | target missing or single-class in synthetic data |
| dpctgan_eps5_seed0 | - | target missing or single-class in synthetic data |
| dpctgan_eps8_seed0 | - | target missing or single-class in synthetic data |
| gaussian_copula_seed0 | 0.5154 | +0.1707 |
| gaussian_copula_seed1 | 0.5429 | +0.1432 |
| gaussian_copula_seed2 | 0.4956 | +0.1905 |
| mst_eps10_seed0 | 0.508 | +0.1781 |
| mst_eps15_seed0 | 0.523 | +0.1631 |
| mst_eps15_seed1 | 0.537 | +0.1491 |
| mst_eps15_seed2 | 0.4581 | +0.2280 |
| mst_eps1_seed0 | 0.5146 | +0.1715 |
| mst_eps20_seed0 | 0.5085 | +0.1776 |
| mst_eps5_seed0 | 0.5391 | +0.1470 |
| mst_eps8_seed0 | 0.4537 | +0.2324 |
| tvae_seed0 | 0.5823 | +0.1038 |
| tvae_seed1 | 0.6001 | +0.0860 |
| tvae_seed2 | 0.557 | +0.1291 |

## `encounter_primary_reason_HF_Disease_f5a_w1mo_first`
train 853 labelled (98 positive), holdout 264 labelled (18 positive) | baseline AUC **0.5729**

| run | TSTR AUC | gap vs baseline |
|---|---|---|
| aim50_eps1_seed0 | 0.5411 | +0.0318 |
| aim50_eps5_seed0 | 0.5312 | +0.0417 |
| ctgan_seed0 | 0.5479 | +0.0250 |
| ctgan_seed1 | 0.5176 | +0.0553 |
| ctgan_seed2 | 0.568 | +0.0049 |
| dpctgan_eps10_seed0 | - | target missing or single-class in synthetic data |
| dpctgan_eps15_seed0 | - | target missing or single-class in synthetic data |
| dpctgan_eps15_seed1 | - | target missing or single-class in synthetic data |
| dpctgan_eps15_seed2 | - | target missing or single-class in synthetic data |
| dpctgan_eps1_seed0 | - | target missing or single-class in synthetic data |
| dpctgan_eps20_seed0 | - | target missing or single-class in synthetic data |
| dpctgan_eps5_seed0 | 0.6383 | -0.0654 |
| dpctgan_eps8_seed0 | - | target missing or single-class in synthetic data |
| gaussian_copula_seed0 | 0.371 | +0.2019 |
| gaussian_copula_seed1 | 0.516 | +0.0569 |
| gaussian_copula_seed2 | 0.3835 | +0.1894 |
| mst_eps10_seed0 | 0.5589 | +0.0140 |
| mst_eps15_seed0 | 0.5567 | +0.0162 |
| mst_eps15_seed1 | 0.4713 | +0.1016 |
| mst_eps15_seed2 | 0.6716 | -0.0987 |
| mst_eps1_seed0 | 0.5637 | +0.0092 |
| mst_eps20_seed0 | 0.6493 | -0.0764 |
| mst_eps5_seed0 | 0.6064 | -0.0335 |
| mst_eps8_seed0 | 0.5906 | -0.0177 |
| tvae_seed0 | 0.4756 | +0.0973 |
| tvae_seed1 | 0.3568 | +0.2161 |
| tvae_seed2 | 0.4636 | +0.1093 |

## `cause_of_death_isAllCause_f5a_w3a_first`
train 1291 labelled (1208 positive), holdout 438 labelled (417 positive) | baseline AUC **0.5435**

| run | TSTR AUC | gap vs baseline |
|---|---|---|
| aim50_eps1_seed0 | 0.5025 | +0.0410 |
| aim50_eps5_seed0 | 0.5108 | +0.0327 |
| ctgan_seed0 | 0.5657 | -0.0222 |
| ctgan_seed1 | 0.5244 | +0.0191 |
| ctgan_seed2 | 0.444 | +0.0995 |
| dpctgan_eps10_seed0 | - | target missing or single-class in synthetic data |
| dpctgan_eps15_seed0 | - | target missing or single-class in synthetic data |
| dpctgan_eps15_seed1 | - | target missing or single-class in synthetic data |
| dpctgan_eps15_seed2 | - | target missing or single-class in synthetic data |
| dpctgan_eps1_seed0 | - | target missing or single-class in synthetic data |
| dpctgan_eps20_seed0 | - | target missing or single-class in synthetic data |
| dpctgan_eps5_seed0 | - | target missing or single-class in synthetic data |
| dpctgan_eps8_seed0 | - | target missing or single-class in synthetic data |
| gaussian_copula_seed0 | 0.6282 | -0.0847 |
| gaussian_copula_seed1 | 0.4716 | +0.0719 |
| gaussian_copula_seed2 | 0.5487 | -0.0052 |
| mst_eps10_seed0 | 0.4969 | +0.0466 |
| mst_eps15_seed0 | 0.4704 | +0.0731 |
| mst_eps15_seed1 | 0.4172 | +0.1263 |
| mst_eps15_seed2 | 0.3669 | +0.1766 |
| mst_eps1_seed0 | 0.4836 | +0.0599 |
| mst_eps20_seed0 | 0.5035 | +0.0400 |
| mst_eps5_seed0 | 0.5067 | +0.0368 |
| mst_eps8_seed0 | 0.4259 | +0.1176 |
| tvae_seed0 | 0.5567 | -0.0132 |
| tvae_seed1 | 0.5372 | +0.0063 |
| tvae_seed2 | 0.6072 | -0.0637 |

## `cause_of_death_isCV_f5a_w3a_first`
train 1291 labelled (83 positive), holdout 438 labelled (21 positive) | baseline AUC **0.5435**

| run | TSTR AUC | gap vs baseline |
|---|---|---|
| aim50_eps1_seed0 | 0.4587 | +0.0848 |
| aim50_eps5_seed0 | 0.5761 | -0.0326 |
| ctgan_seed0 | 0.5369 | +0.0066 |
| ctgan_seed1 | 0.4502 | +0.0933 |
| ctgan_seed2 | 0.4056 | +0.1379 |
| dpctgan_eps10_seed0 | - | target missing or single-class in synthetic data |
| dpctgan_eps15_seed0 | - | target missing or single-class in synthetic data |
| dpctgan_eps15_seed1 | - | target missing or single-class in synthetic data |
| dpctgan_eps15_seed2 | - | target missing or single-class in synthetic data |
| dpctgan_eps1_seed0 | - | target missing or single-class in synthetic data |
| dpctgan_eps20_seed0 | - | target missing or single-class in synthetic data |
| dpctgan_eps5_seed0 | - | target missing or single-class in synthetic data |
| dpctgan_eps8_seed0 | - | target missing or single-class in synthetic data |
| gaussian_copula_seed0 | 0.523 | +0.0205 |
| gaussian_copula_seed1 | 0.4837 | +0.0598 |
| gaussian_copula_seed2 | 0.5255 | +0.0180 |
| mst_eps10_seed0 | 0.4361 | +0.1074 |
| mst_eps15_seed0 | 0.4985 | +0.0450 |
| mst_eps15_seed1 | 0.6033 | -0.0598 |
| mst_eps15_seed2 | 0.5009 | +0.0426 |
| mst_eps1_seed0 | 0.4428 | +0.1007 |
| mst_eps20_seed0 | 0.4636 | +0.0799 |
| mst_eps5_seed0 | 0.5063 | +0.0372 |
| mst_eps8_seed0 | 0.5087 | +0.0348 |
| tvae_seed0 | 0.5487 | -0.0052 |
| tvae_seed1 | 0.543 | +0.0005 |
| tvae_seed2 | 0.5942 | -0.0507 |

## Mean AUC gap per (model, ε) across seeds and targets (lower is better)

| model | ε | runs | mean gap ± sd |
|---|---|---|---|
| aim | 1 | 1 | +0.0961 |
| aim | 5 | 1 | +0.0866 |
| ctgan | - | 3 | +0.1058 ± 0.021 |
| dpctgan | 5 | 1 | -0.0654 |
| dpctgan | 15 | 1 | +0.1747 |
| gaussian_copula | - | 3 | +0.1016 ± 0.0066 |
| mst | 1 | 1 | +0.0901 |
| mst | 5 | 1 | +0.0810 |
| mst | 8 | 1 | +0.1173 |
| mst | 10 | 1 | +0.0945 |
| mst | 15 | 3 | +0.0972 ± 0.003 |
| mst | 20 | 1 | +0.0780 |
| tvae | - | 3 | +0.0615 ± 0.0205 |
