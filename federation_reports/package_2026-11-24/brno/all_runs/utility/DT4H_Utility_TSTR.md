# Utility: Train-Synthetic, Test-Real (TSTR)

A gradient-boosting classifier is trained on the real TRAINING split (baseline) and on each synthetic dataset, then both are scored on the HOLDOUT split -- real patients that neither the generators nor either classifier ever saw. The closer the TSTR AUC is to the baseline, the more useful the synthetic data is for actual modelling work.

Real train: 1549 rows | holdout test: 516 rows

## `encounter_primary_reason_CV_Disease_f5a_w1mo_first`
train 441 labelled (223 positive), holdout 154 labelled (82 positive) | baseline AUC **0.751** (95% CI 0.6702-0.8245) (HistGB) / 0.6214 (LogReg)

CIs are bootstrap over holdout predictions (1000 resamples). 'aug Δ vs bootstrap' is the size-matched control: the augmented AUC minus the AUC of real + bootstrap-resampled REAL rows of the same added size (positive = synthetic beats the pure row-count effect).

| run | TSTR AUC | 95% CI | gap | LogReg gap | augmentation Δ | aug Δ vs bootstrap |
|---|---|---|---|---|---|---|
| aim40_eps1_seed0 | 0.4106 | 0.3259-0.4985 | +0.3404 | +0.2134 | -0.0547 | -0.0368 |
| aim40_eps5_seed0 | 0.5271 | 0.4383-0.6226 | +0.2239 | +0.0437 | -0.0064 | +0.0198 |
| aim40_eps8_seed0 | 0.5418 | 0.4533-0.6331 | +0.2092 | +0.2173 | +0.0095 | +0.0195 |
| aim50_eps1_seed0 | 0.4695 | 0.3774-0.5573 | +0.2815 | +0.1031 | -0.0582 | -0.0218 |
| aim50_eps5_seed0 | 0.5208 | 0.4328-0.618 | +0.2302 | +0.1585 | +0.0303 | +0.0559 |
| aim50_eps8_seed0 | 0.4956 | 0.4007-0.5917 | +0.2554 | +0.0691 | -0.0467 | -0.0374 |
| ctgan_qt_seed0 | 0.6489 | 0.5711-0.7276 | +0.1021 | -0.0182 | +0.0090 | -0.0007 |
| ctgan_seed0 | 0.5046 | 0.4219-0.6011 | +0.2464 | +0.1114 | -0.0472 | -0.0269 |
| ctgan_seed1 | 0.4919 | 0.3972-0.5842 | +0.2591 | +0.1128 | -0.0252 | -0.0105 |
| ctgan_seed2 | 0.53 | 0.4402-0.6243 | +0.2210 | +0.0570 | +0.0126 | +0.0214 |
| ddpm_g_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| ddpm_seed0 | 0.4492 | 0.3572-0.5407 | +0.3018 | +0.1897 | -0.0252 | -0.0142 |
| ddpm_seed1 | 0.3874 | 0.3053-0.4757 | +0.3636 | +0.2078 | -0.0605 | -0.0241 |
| ddpm_seed2 | 0.5584 | 0.4678-0.65 | +0.1926 | +0.1429 | -0.0435 | -0.0032 |
| dpctgan_eps10_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps15_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps15_seed1 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps15_seed2 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps1_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps20_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps5_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps8_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| gaussian_copula_seed0 | 0.5564 | 0.4691-0.6453 | +0.1946 | +0.0841 | -0.0215 | +0.0076 |
| gaussian_copula_seed1 | 0.544 | 0.4524-0.6328 | +0.2070 | +0.0894 | -0.0262 | -0.0064 |
| gaussian_copula_seed2 | 0.5539 | 0.4625-0.6388 | +0.1971 | +0.0970 | +0.0058 | +0.0027 |
| mst_eps10_seed0 | 0.5249 | 0.4292-0.615 | +0.2261 | +0.1305 | -0.0024 | +0.0076 |
| mst_eps15_seed0 | 0.6114 | 0.5257-0.6961 | +0.1396 | +0.0533 | -0.0201 | -0.0089 |
| mst_eps15_seed1 | 0.3994 | 0.3079-0.4953 | +0.3516 | +0.1211 | -0.0220 | +0.0036 |
| mst_eps15_seed2 | 0.5176 | 0.4288-0.615 | +0.2334 | +0.0979 | -0.0034 | -0.0014 |
| mst_eps1_seed0 | 0.4783 | 0.3902-0.5702 | +0.2727 | +0.1236 | -0.0164 | -0.0190 |
| mst_eps20_seed0 | 0.5528 | 0.4533-0.65 | +0.1982 | +0.0799 | -0.0027 | +0.0034 |
| mst_eps5_seed0 | 0.5384 | 0.437-0.6275 | +0.2126 | +0.0699 | -0.0564 | -0.0337 |
| mst_eps8_seed0 | 0.6428 | 0.556-0.7252 | +0.1082 | +0.0777 | -0.0161 | -0.0112 |
| patectgan_eps15_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| patectgan_eps1_seed0 | 0.3933 | 0.3074-0.4914 | +0.3577 | +0.1234 | +0.0115 | +0.0476 |
| patectgan_eps5_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| tvae_cap256_seed0 | 0.7405 | 0.651-0.8183 | +0.0105 | +0.0982 | -0.0091 | -0.0018 |
| tvae_ep1000_seed0 | 0.7154 | 0.6292-0.7955 | +0.0356 | +0.1319 | -0.0030 | +0.0031 |
| tvae_ind_seed0 | 0.6739 | 0.5918-0.7522 | +0.0771 | +0.0232 | -0.0279 | -0.0191 |
| tvae_qt_seed0 | 0.6734 | 0.5877-0.7525 | +0.0776 | +0.1011 | -0.0130 | +0.0049 |
| tvae_qt_seed1 | 0.6463 | 0.5542-0.7309 | +0.1047 | -0.0033 | -0.0222 | -0.0287 |
| tvae_qt_seed2 | 0.6601 | 0.5657-0.7452 | +0.0909 | -0.0119 | -0.0318 | -0.0262 |
| tvae_seed0 | 0.6966 | 0.6069-0.7783 | +0.0544 | -0.0790 | +0.0136 | +0.0199 |
| tvae_seed1 | 0.7038 | 0.616-0.7819 | +0.0472 | +0.0386 | +0.0034 | +0.0051 |
| tvae_seed2 | 0.6736 | 0.5824-0.7556 | +0.0774 | -0.0237 | -0.0320 | -0.0090 |

## `encounter_primary_reason_non_CV_Disease_f5a_w1mo_first`
train 441 labelled (218 positive), holdout 154 labelled (72 positive) | baseline AUC **0.751** (95% CI 0.6702-0.8245) (HistGB) / 0.6214 (LogReg)

CIs are bootstrap over holdout predictions (1000 resamples). 'aug Δ vs bootstrap' is the size-matched control: the augmented AUC minus the AUC of real + bootstrap-resampled REAL rows of the same added size (positive = synthetic beats the pure row-count effect).

| run | TSTR AUC | 95% CI | gap | LogReg gap | augmentation Δ | aug Δ vs bootstrap |
|---|---|---|---|---|---|---|
| aim40_eps1_seed0 | 0.4252 | 0.3372-0.5124 | +0.3258 | +0.2149 | -0.0410 | -0.0237 |
| aim40_eps5_seed0 | 0.487 | 0.4004-0.5719 | +0.2640 | +0.0913 | -0.0305 | -0.0043 |
| aim40_eps8_seed0 | 0.5278 | 0.4375-0.6128 | +0.2232 | +0.1971 | +0.0105 | +0.0166 |
| aim50_eps1_seed0 | 0.5627 | 0.4708-0.6536 | +0.1883 | +0.0831 | -0.0032 | +0.0080 |
| aim50_eps5_seed0 | 0.492 | 0.4002-0.5857 | +0.2590 | +0.1614 | -0.0203 | -0.0183 |
| aim50_eps8_seed0 | 0.5105 | 0.4164-0.6053 | +0.2405 | +0.0411 | -0.0337 | -0.0046 |
| ctgan_qt_seed0 | 0.4385 | 0.3431-0.527 | +0.3125 | +0.1902 | -0.0117 | -0.0071 |
| ctgan_seed0 | 0.4458 | 0.3596-0.533 | +0.3052 | +0.1898 | -0.0303 | -0.0398 |
| ctgan_seed1 | 0.4182 | 0.3256-0.5175 | +0.3328 | +0.1441 | -0.0522 | -0.0244 |
| ctgan_seed2 | 0.5423 | 0.4499-0.638 | +0.2087 | +0.0511 | -0.0188 | -0.0266 |
| ddpm_g_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| ddpm_seed0 | 0.5041 | 0.4129-0.5951 | +0.2469 | +0.1177 | -0.0447 | -0.0042 |
| ddpm_seed1 | 0.5689 | 0.4747-0.6636 | +0.1821 | +0.0640 | -0.0381 | -0.0105 |
| ddpm_seed2 | 0.5525 | 0.4599-0.6417 | +0.1985 | +0.2068 | +0.0139 | +0.0398 |
| dpctgan_eps10_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps15_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps15_seed1 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps15_seed2 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps1_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps20_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps5_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps8_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| gaussian_copula_seed0 | 0.4959 | 0.4103-0.5854 | +0.2551 | +0.2127 | +0.0020 | +0.0233 |
| gaussian_copula_seed1 | 0.5027 | 0.4136-0.593 | +0.2483 | +0.1173 | -0.0278 | -0.0229 |
| gaussian_copula_seed2 | 0.544 | 0.457-0.6236 | +0.2070 | +0.0913 | -0.0682 | -0.0452 |
| mst_eps10_seed0 | 0.5029 | 0.4118-0.5982 | +0.2481 | +0.1592 | -0.0179 | -0.0205 |
| mst_eps15_seed0 | 0.5566 | 0.4689-0.6441 | +0.1944 | +0.1217 | -0.0027 | +0.0235 |
| mst_eps15_seed1 | 0.4482 | 0.3519-0.5438 | +0.3028 | +0.1139 | -0.0046 | +0.0015 |
| mst_eps15_seed2 | 0.531 | 0.4439-0.6237 | +0.2200 | +0.1085 | -0.0293 | -0.0273 |
| mst_eps1_seed0 | 0.5185 | 0.4254-0.6061 | +0.2325 | +0.1575 | -0.0003 | +0.0241 |
| mst_eps20_seed0 | 0.5407 | 0.4391-0.6384 | +0.2103 | +0.0855 | +0.0056 | +0.0176 |
| mst_eps5_seed0 | 0.6375 | 0.5464-0.7306 | +0.1135 | +0.0572 | -0.0276 | -0.0005 |
| mst_eps8_seed0 | 0.6438 | 0.5559-0.7273 | +0.1072 | -0.0114 | +0.0078 | +0.0198 |
| patectgan_eps15_seed0 | 0.4765 | 0.393-0.5659 | +0.2745 | +0.1656 | -0.0156 | -0.0075 |
| patectgan_eps1_seed0 | 0.4741 | 0.3825-0.567 | +0.2769 | +0.0753 | -0.0264 | -0.0329 |
| patectgan_eps5_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| tvae_cap256_seed0 | 0.7315 | 0.6431-0.8099 | +0.0195 | +0.1089 | -0.0120 | +0.0093 |
| tvae_ep1000_seed0 | 0.732 | 0.6485-0.8117 | +0.0190 | +0.1228 | -0.0117 | -0.0099 |
| tvae_ind_seed0 | 0.6845 | 0.6012-0.7623 | +0.0665 | +0.0308 | -0.0410 | -0.0322 |
| tvae_qt_seed0 | 0.6775 | 0.5935-0.764 | +0.0735 | +0.0919 | -0.0154 | +0.0025 |
| tvae_qt_seed1 | 0.6492 | 0.5549-0.7325 | +0.1018 | -0.0021 | -0.0201 | -0.0281 |
| tvae_qt_seed2 | 0.6397 | 0.5461-0.7296 | +0.1113 | +0.0144 | -0.0230 | +0.0110 |
| tvae_seed0 | 0.7119 | 0.6257-0.7938 | +0.0391 | -0.0754 | -0.0030 | +0.0183 |
| tvae_seed1 | 0.7095 | 0.6206-0.7856 | +0.0415 | +0.0238 | +0.0124 | +0.0280 |
| tvae_seed2 | 0.6624 | 0.5769-0.7483 | +0.0886 | -0.0326 | -0.0425 | -0.0263 |

## `encounter_primary_reason_HF_Disease_f5a_w1mo_first`
train 441 labelled (19 positive), holdout 154 labelled (11 positive) | baseline AUC **0.5868** (95% CI 0.3585-0.7851) (HistGB) / 0.5779 (LogReg)

CIs are bootstrap over holdout predictions (1000 resamples). 'aug Δ vs bootstrap' is the size-matched control: the augmented AUC minus the AUC of real + bootstrap-resampled REAL rows of the same added size (positive = synthetic beats the pure row-count effect).

| run | TSTR AUC | 95% CI | gap | LogReg gap | augmentation Δ | aug Δ vs bootstrap |
|---|---|---|---|---|---|---|
| aim40_eps1_seed0 | 0.6128 | 0.4126-0.8204 | -0.0260 | +0.0242 | +0.0896 | +0.0216 |
| aim40_eps5_seed0 | 0.4132 | 0.2326-0.5917 | +0.1736 | +0.0839 | +0.0190 | +0.0476 |
| aim40_eps8_seed0 | 0.5378 | 0.3454-0.7459 | +0.0490 | -0.0146 | +0.1322 | +0.1189 |
| aim50_eps1_seed0 | 0.6167 | 0.4306-0.7953 | -0.0299 | +0.1036 | +0.1182 | +0.0858 |
| aim50_eps5_seed0 | 0.5931 | 0.4178-0.7493 | -0.0063 | +0.0235 | +0.0705 | +0.0953 |
| aim50_eps8_seed0 | 0.4584 | 0.2209-0.6934 | +0.1284 | +0.1017 | +0.0661 | +0.0820 |
| ctgan_qt_seed0 | 0.3408 | 0.1364-0.5816 | +0.2460 | +0.0617 | +0.0616 | +0.0292 |
| ctgan_seed0 | 0.5073 | 0.3279-0.696 | +0.0795 | -0.0101 | +0.1176 | +0.0846 |
| ctgan_seed1 | 0.4882 | 0.2731-0.6942 | +0.0986 | +0.0700 | +0.0222 | +0.0076 |
| ctgan_seed2 | 0.5315 | 0.3414-0.7188 | +0.0553 | -0.1456 | +0.0693 | +0.0414 |
| ddpm_g_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| ddpm_seed0 | 0.5944 | 0.4046-0.7875 | -0.0076 | +0.0471 | +0.1150 | +0.1258 |
| ddpm_seed1 | 0.4743 | 0.2727-0.7019 | +0.1125 | +0.1977 | +0.0470 | +0.0597 |
| ddpm_seed2 | 0.4876 | 0.2865-0.6867 | +0.0992 | +0.0255 | +0.0731 | +0.0452 |
| dpctgan_eps10_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps15_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps15_seed1 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps15_seed2 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps1_seed0 | 0.4838 | 0.3098-0.6562 | +0.1030 | -0.0852 | -0.0007 | -0.0293 |
| dpctgan_eps20_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps5_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps8_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| gaussian_copula_seed0 | 0.5563 | 0.3637-0.7364 | +0.0305 | +0.0553 | +0.1081 | +0.1240 |
| gaussian_copula_seed1 | 0.4234 | 0.2721-0.5822 | +0.1634 | -0.0165 | -0.0490 | -0.1183 |
| gaussian_copula_seed2 | 0.3185 | 0.1289-0.5282 | +0.2683 | +0.0000 | +0.0686 | +0.0801 |
| mst_eps10_seed0 | 0.5486 | 0.3325-0.7448 | +0.0382 | -0.0896 | +0.0534 | +0.0636 |
| mst_eps15_seed0 | 0.6256 | 0.4383-0.7908 | -0.0388 | -0.0400 | +0.0801 | +0.1087 |
| mst_eps15_seed1 | 0.5442 | 0.3333-0.7611 | +0.0426 | -0.0210 | -0.0191 | +0.0089 |
| mst_eps15_seed2 | 0.6402 | 0.4613-0.8253 | -0.0534 | +0.0782 | +0.0489 | +0.0591 |
| mst_eps1_seed0 | 0.6262 | 0.4584-0.7952 | -0.0394 | -0.0667 | +0.0559 | +0.0229 |
| mst_eps20_seed0 | 0.5925 | 0.3833-0.7863 | -0.0057 | -0.0585 | +0.0680 | +0.0547 |
| mst_eps5_seed0 | 0.4603 | 0.2571-0.6828 | +0.1265 | +0.1488 | +0.0089 | +0.0204 |
| mst_eps8_seed0 | 0.5779 | 0.426-0.7264 | +0.0089 | +0.0553 | -0.0172 | +0.0280 |
| patectgan_eps15_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| patectgan_eps1_seed0 | 0.5416 | 0.3916-0.6901 | +0.0452 | +0.1259 | +0.1100 | +0.0350 |
| patectgan_eps5_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| tvae_cap256_seed0 | 0.6319 | 0.4376-0.8013 | -0.0451 | +0.1125 | +0.1405 | +0.1443 |
| tvae_ep1000_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| tvae_ind_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| tvae_qt_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| tvae_qt_seed1 | 0.5677 | 0.4192-0.7042 | +0.0191 | +0.0134 | +0.0292 | -0.0280 |
| tvae_qt_seed2 | 0.6243 | 0.3993-0.8417 | -0.0375 | +0.0089 | +0.0222 | -0.0693 |
| tvae_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| tvae_seed1 | 0.4704 | 0.2562-0.6868 | +0.1164 | +0.0623 | +0.1119 | +0.0719 |
| tvae_seed2 | - | - | target missing or single-class in synthetic data | - | - | - |

## Per (model, ε) across seeds and targets

Gaps vs baseline, lower is better; augmentation Δ is the AUC change from training on real+synthetic vs real alone (positive = synthetic data adds value), and 'vs bootstrap' subtracts the size-matched real-resample control. 'sd n/a (single run)' means no spread can be estimated -- it is NOT the same statement as an observed sd of 0.

| model | ε | runs | HistGB gap ± sd | LogReg gap | augmentation Δ | aug Δ vs bootstrap | Brier gap | worst-stratum gap |
|---|---|---|---|---|---|---|---|---|
| aim | 1 | 1 | +0.1466 (sd n/a: single run) | 0.0966 | 0.0189 | 0.024 | 0.0665 | 0.1683 |
| aim | 5 | 1 | +0.1610 (sd n/a: single run) | 0.1145 | 0.0268 | 0.0443 | 0.1632 | 0.2118 |
| aim | 8 | 1 | +0.2081 (sd n/a: single run) | 0.0706 | -0.0048 | 0.0133 | 0.1234 | 0.2597 |
| aim40 | 1 | 1 | +0.2134 (sd n/a: single run) | 0.1508 | -0.002 | -0.013 | 0.1865 | 0.2327 |
| aim40 | 5 | 1 | +0.2205 (sd n/a: single run) | 0.073 | -0.006 | 0.021 | 0.0557 | 0.3035 |
| aim40 | 8 | 1 | +0.1605 (sd n/a: single run) | 0.1333 | 0.0507 | 0.0517 | 0.1571 | 0.1621 |
| ctgan | - | 3 | +0.2007 ± 0.0353 | 0.0645 | 0.0053 | 0.003 | 0.0933 | 0.2169 |
| ctgan_qt | - | 1 | +0.2202 (sd n/a: single run) | 0.0779 | 0.0196 | 0.0071 | 0.0594 | 0.332 |
| ddpm | - | 3 | +0.1877 ± 0.0287 | 0.1332 | 0.0041 | 0.0238 | 0.0754 | 0.2 |
| dpctgan | 1 | 1 | +0.1030 (sd n/a: single run) | -0.0852 | -0.0007 | -0.0293 | 0.0004 | 0.0884 |
| gaussian_copula | - | 3 | +0.1968 ± 0.0331 | 0.0812 | -0.0009 | 0.005 | 0.0702 | 0.2593 |
| mst | 1 | 1 | +0.1553 (sd n/a: single run) | 0.0715 | 0.0131 | 0.0093 | 0.1564 | 0.174 |
| mst | 5 | 1 | +0.1509 (sd n/a: single run) | 0.092 | -0.025 | -0.0046 | 0.103 | 0.2168 |
| mst | 8 | 1 | +0.0748 (sd n/a: single run) | 0.0405 | -0.0085 | 0.0122 | 0.0751 | 0.0936 |
| mst | 10 | 1 | +0.1708 (sd n/a: single run) | 0.0667 | 0.011 | 0.0169 | 0.1354 | 0.238 |
| mst | 15 | 3 | +0.1547 ± 0.0695 | 0.0704 | 0.0031 | 0.0186 | 0.1481 | 0.2046 |
| mst | 20 | 1 | +0.1343 (sd n/a: single run) | 0.0356 | 0.0236 | 0.0252 | 0.1335 | 0.1459 |
| patectgan | 1 | 1 | +0.2266 (sd n/a: single run) | 0.1082 | 0.0317 | 0.0166 | 0.1122 | 0.2437 |
| patectgan | 15 | 1 | +0.2745 (sd n/a: single run) | 0.1656 | -0.0156 | -0.0075 | 0.3141 | 0.5833 |
| tvae | - | 3 | +0.0660 ± 0.0182 | -0.0213 | 0.0035 | 0.0122 | 0.0652 | 0.1212 |
| tvae_cap256 | - | 1 | -0.0050 (sd n/a: single run) | 0.1065 | 0.0398 | 0.0506 | 0.0191 | 0.0047 |
| tvae_ep1000 | - | 1 | +0.0273 (sd n/a: single run) | 0.1273 | -0.0074 | -0.0034 | 0.0298 | 0.0552 |
| tvae_ind | - | 1 | +0.0718 (sd n/a: single run) | 0.027 | -0.0345 | -0.0256 | 0.0621 | 0.1444 |
| tvae_qt | - | 3 | +0.0685 ± 0.0118 | 0.0343 | -0.0098 | -0.0176 | 0.0604 | 0.1416 |
