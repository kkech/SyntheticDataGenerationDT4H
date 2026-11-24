# Utility: Train-Synthetic, Test-Real (TSTR)

A gradient-boosting classifier is trained on the real TRAINING split (baseline) and on each synthetic dataset, then both are scored on the HOLDOUT split -- real patients that neither the generators nor either classifier ever saw. The closer the TSTR AUC is to the baseline, the more useful the synthetic data is for actual modelling work.

Real train: 3773 rows | holdout test: 1258 rows

## `encounter_primary_reason_non_CV_Disease_f5a_w6mo_first`
train 782 labelled (393 positive), holdout 238 labelled (132 positive) | baseline AUC **0.5725** (95% CI 0.4978-0.6403) (HistGB) / 0.5549 (LogReg)

CIs are bootstrap over holdout predictions (1000 resamples). 'aug Δ vs bootstrap' is the size-matched control: the augmented AUC minus the AUC of real + bootstrap-resampled REAL rows of the same added size (positive = synthetic beats the pure row-count effect).

| run | TSTR AUC | 95% CI | gap | LogReg gap | augmentation Δ | aug Δ vs bootstrap |
|---|---|---|---|---|---|---|
| aim40_eps1_seed0 | 0.5399 | 0.4653-0.6185 | +0.0326 | +0.0523 | +0.0088 | +0.0053 |
| aim40_eps5_seed0 | 0.4892 | 0.415-0.5647 | +0.0833 | +0.1497 | +0.0144 | +0.0267 |
| aim50_eps1_seed0 | 0.5068 | 0.4324-0.5795 | +0.0657 | -0.0193 | -0.0132 | -0.0030 |
| ctgan_qt_seed0 | 0.5594 | 0.488-0.6317 | +0.0131 | -0.0162 | -0.0087 | -0.0290 |
| ctgan_seed0 | 0.5432 | 0.4679-0.614 | +0.0293 | -0.0148 | -0.0111 | -0.0131 |
| ctgan_seed1 | 0.5369 | 0.4588-0.6071 | +0.0356 | +0.0187 | +0.0136 | +0.0133 |
| ctgan_seed2 | 0.5531 | 0.4765-0.6237 | +0.0194 | -0.0034 | +0.0332 | +0.0240 |
| ddpm_g_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| ddpm_seed0 | 0.5111 | 0.4349-0.5874 | +0.0614 | +0.0282 | +0.0076 | +0.0254 |
| ddpm_seed1 | 0.6328 | 0.5596-0.7021 | -0.0603 | +0.0022 | +0.0099 | +0.0132 |
| ddpm_seed2 | 0.5448 | 0.473-0.6241 | +0.0277 | +0.0042 | +0.0164 | +0.0051 |
| dpctgan_eps10_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps15_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps15_seed1 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps15_seed2 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps1_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps20_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps5_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps8_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| gaussian_copula_seed0 | 0.5078 | 0.4323-0.5891 | +0.0647 | +0.0505 | +0.0077 | +0.0107 |
| gaussian_copula_seed1 | 0.469 | 0.3965-0.5404 | +0.1035 | +0.0539 | -0.0009 | -0.0044 |
| gaussian_copula_seed2 | 0.4903 | 0.4178-0.5635 | +0.0822 | +0.0159 | +0.0089 | +0.0174 |
| mst_eps0p5_seed0 | 0.5392 | 0.4701-0.6126 | +0.0333 | +0.0218 | -0.0163 | -0.0191 |
| mst_eps10_seed0 | 0.5255 | 0.4548-0.6001 | +0.0470 | +0.0159 | -0.0060 | -0.0085 |
| mst_eps15_seed0 | 0.4716 | 0.3968-0.5425 | +0.1009 | +0.1158 | -0.0130 | +0.0048 |
| mst_eps15_seed1 | 0.4307 | 0.3489-0.5032 | +0.1418 | +0.1331 | -0.0229 | -0.0114 |
| mst_eps15_seed2 | 0.5462 | 0.479-0.6229 | +0.0263 | +0.0124 | -0.0097 | +0.0018 |
| mst_eps1_seed0 | 0.5437 | 0.473-0.6187 | +0.0288 | +0.0991 | +0.0128 | +0.0091 |
| mst_eps20_seed0 | 0.5187 | 0.4521-0.5905 | +0.0538 | +0.0072 | +0.0194 | +0.0191 |
| mst_eps5_seed0 | 0.5333 | 0.4602-0.5996 | +0.0392 | +0.0862 | +0.0115 | +0.0170 |
| mst_eps8_seed0 | 0.5581 | 0.484-0.6316 | +0.0144 | +0.0676 | -0.0173 | -0.0023 |
| patectgan_eps15_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| patectgan_eps1_seed0 | 0.4774 | 0.4036-0.5565 | +0.0951 | +0.0808 | +0.0296 | +0.0289 |
| patectgan_eps5_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| tvae_cap256_seed0 | 0.6266 | 0.5569-0.6947 | -0.0541 | -0.0236 | +0.0116 | +0.0156 |
| tvae_ep1000_seed0 | 0.5838 | 0.516-0.6609 | -0.0113 | -0.0161 | -0.0164 | -0.0127 |
| tvae_ind_seed0 | 0.5263 | 0.45-0.5986 | +0.0462 | +0.0552 | +0.0175 | +0.0200 |
| tvae_qt_seed0 | 0.5798 | 0.5128-0.6494 | -0.0073 | +0.0290 | -0.0130 | -0.0184 |
| tvae_qt_seed1 | 0.5722 | 0.498-0.6483 | +0.0003 | +0.0455 | +0.0226 | +0.0286 |
| tvae_qt_seed2 | 0.5543 | 0.4768-0.6294 | +0.0182 | -0.0142 | +0.0329 | +0.0281 |
| tvae_seed0 | 0.5481 | 0.4725-0.6242 | +0.0244 | +0.0778 | -0.0077 | +0.0034 |
| tvae_seed1 | 0.554 | 0.4799-0.6258 | +0.0185 | +0.0452 | -0.0418 | -0.0537 |
| tvae_seed2 | 0.4588 | 0.3884-0.5331 | +0.1137 | -0.0101 | -0.0331 | -0.0168 |

## `encounter_primary_reason_CV_Disease_f5a_w6mo_first`
train 782 labelled (395 positive), holdout 238 labelled (108 positive) | baseline AUC **0.5855** (95% CI 0.5142-0.6538) (HistGB) / 0.5791 (LogReg)

CIs are bootstrap over holdout predictions (1000 resamples). 'aug Δ vs bootstrap' is the size-matched control: the augmented AUC minus the AUC of real + bootstrap-resampled REAL rows of the same added size (positive = synthetic beats the pure row-count effect).

| run | TSTR AUC | 95% CI | gap | LogReg gap | augmentation Δ | aug Δ vs bootstrap |
|---|---|---|---|---|---|---|
| aim40_eps1_seed0 | 0.4563 | 0.3838-0.5405 | +0.1292 | +0.1436 | -0.0362 | -0.0230 |
| aim40_eps5_seed0 | 0.5162 | 0.4459-0.5889 | +0.0693 | +0.1565 | -0.0001 | +0.0130 |
| aim50_eps1_seed0 | 0.5318 | 0.4562-0.6017 | +0.0537 | +0.0744 | +0.0119 | +0.0134 |
| ctgan_qt_seed0 | 0.5803 | 0.5075-0.6499 | +0.0052 | +0.0187 | +0.0101 | +0.0017 |
| ctgan_seed0 | 0.5146 | 0.4416-0.5925 | +0.0709 | +0.0557 | -0.0356 | -0.0213 |
| ctgan_seed1 | 0.536 | 0.4646-0.6161 | +0.0495 | +0.0180 | -0.0064 | -0.0334 |
| ctgan_seed2 | 0.5915 | 0.5212-0.6598 | -0.0060 | +0.0097 | -0.0166 | -0.0305 |
| ddpm_g_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| ddpm_seed0 | 0.5474 | 0.4717-0.6238 | +0.0381 | +0.0576 | +0.0172 | +0.0333 |
| ddpm_seed1 | 0.5843 | 0.5132-0.652 | +0.0012 | +0.0258 | +0.0224 | -0.0065 |
| ddpm_seed2 | 0.5344 | 0.4604-0.6123 | +0.0511 | +0.0539 | -0.0467 | -0.0185 |
| dpctgan_eps10_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps15_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps15_seed1 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps15_seed2 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps1_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps20_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps5_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps8_seed0 | 0.5256 | 0.4529-0.5946 | +0.0599 | +0.0791 | +0.0038 | +0.0137 |
| gaussian_copula_seed0 | 0.5226 | 0.4487-0.5956 | +0.0629 | +0.1086 | +0.0038 | -0.0023 |
| gaussian_copula_seed1 | 0.5484 | 0.4757-0.6235 | +0.0371 | +0.0539 | -0.0033 | +0.0143 |
| gaussian_copula_seed2 | 0.5806 | 0.5054-0.656 | +0.0049 | +0.0914 | +0.0227 | +0.0341 |
| mst_eps0p5_seed0 | 0.4428 | 0.3742-0.5125 | +0.1427 | +0.1045 | -0.0119 | -0.0022 |
| mst_eps10_seed0 | 0.5341 | 0.4619-0.6077 | +0.0514 | +0.0437 | +0.0009 | +0.0147 |
| mst_eps15_seed0 | 0.5087 | 0.4344-0.584 | +0.0768 | +0.1543 | +0.0100 | +0.0232 |
| mst_eps15_seed1 | 0.4701 | 0.3908-0.5427 | +0.1154 | +0.1551 | +0.0032 | +0.0073 |
| mst_eps15_seed2 | 0.5761 | 0.5031-0.6468 | +0.0094 | +0.0346 | -0.0270 | -0.0304 |
| mst_eps1_seed0 | 0.5312 | 0.4607-0.6067 | +0.0543 | +0.1032 | -0.0053 | -0.0172 |
| mst_eps20_seed0 | 0.4885 | 0.4195-0.5677 | +0.0970 | +0.0242 | -0.0038 | +0.0083 |
| mst_eps5_seed0 | 0.5478 | 0.473-0.6197 | +0.0377 | +0.0909 | +0.0098 | +0.0079 |
| mst_eps8_seed0 | 0.5514 | 0.4795-0.627 | +0.0341 | +0.0941 | -0.0136 | -0.0015 |
| patectgan_eps15_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| patectgan_eps1_seed0 | 0.5132 | 0.4356-0.5844 | +0.0723 | +0.1113 | -0.0275 | -0.0160 |
| patectgan_eps5_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| tvae_cap256_seed0 | 0.5637 | 0.4893-0.6311 | +0.0218 | +0.0125 | -0.0064 | +0.0065 |
| tvae_ep1000_seed0 | 0.5925 | 0.5195-0.6637 | -0.0070 | +0.0659 | +0.0152 | +0.0456 |
| tvae_ind_seed0 | 0.557 | 0.4852-0.631 | +0.0285 | +0.0775 | +0.0029 | +0.0101 |
| tvae_qt_seed0 | 0.574 | 0.5049-0.6421 | +0.0115 | +0.0434 | -0.0277 | -0.0294 |
| tvae_qt_seed1 | 0.5378 | 0.4623-0.6164 | +0.0477 | +0.0829 | -0.0104 | -0.0057 |
| tvae_qt_seed2 | 0.5815 | 0.5057-0.6575 | +0.0040 | +0.0079 | +0.0119 | +0.0089 |
| tvae_seed0 | 0.5755 | 0.4987-0.645 | +0.0100 | +0.0896 | +0.0000 | -0.0002 |
| tvae_seed1 | 0.5771 | 0.5063-0.6489 | +0.0084 | +0.0638 | +0.0040 | +0.0027 |
| tvae_seed2 | 0.4963 | 0.4204-0.5684 | +0.0892 | +0.0096 | -0.0413 | -0.0500 |

## `encounter_primary_reason_HF_Disease_f5a_w6mo_first`
_holdout too imbalanced for this target; skipped_

## `encounter_primary_reason_renal_complications_f5a_w5a_first`
train 1290 labelled (42 positive), holdout 390 labelled (17 positive) | baseline AUC **0.6393** (95% CI 0.4737-0.7789) (HistGB) / 0.6485 (LogReg)

CIs are bootstrap over holdout predictions (1000 resamples). 'aug Δ vs bootstrap' is the size-matched control: the augmented AUC minus the AUC of real + bootstrap-resampled REAL rows of the same added size (positive = synthetic beats the pure row-count effect).

| run | TSTR AUC | 95% CI | gap | LogReg gap | augmentation Δ | aug Δ vs bootstrap |
|---|---|---|---|---|---|---|
| aim40_eps1_seed0 | 0.461 | 0.3184-0.607 | +0.1783 | +0.2943 | +0.0171 | +0.0100 |
| aim40_eps5_seed0 | 0.3498 | 0.2552-0.4492 | +0.2895 | +0.2435 | -0.0943 | -0.0503 |
| aim50_eps1_seed0 | 0.5272 | 0.3909-0.664 | +0.1121 | +0.1713 | -0.0080 | +0.0230 |
| ctgan_qt_seed0 | 0.4409 | 0.2814-0.5862 | +0.1984 | +0.1443 | -0.0345 | -0.0344 |
| ctgan_seed0 | 0.4883 | 0.3526-0.6297 | +0.1510 | +0.0959 | +0.0502 | +0.0822 |
| ctgan_seed1 | 0.6053 | 0.4861-0.7187 | +0.0340 | +0.0704 | +0.0483 | +0.0899 |
| ctgan_seed2 | 0.4908 | 0.319-0.6699 | +0.1485 | +0.2063 | +0.0145 | +0.0539 |
| ddpm_g_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| ddpm_seed0 | 0.5346 | 0.4008-0.6662 | +0.1047 | +0.0182 | +0.0432 | +0.0787 |
| ddpm_seed1 | - | - | target missing or single-class in synthetic data | - | - | - |
| ddpm_seed2 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps10_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps15_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps15_seed1 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps15_seed2 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps1_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps20_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps5_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps8_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| gaussian_copula_seed0 | 0.5876 | 0.448-0.7201 | +0.0517 | +0.2147 | +0.0516 | +0.0856 |
| gaussian_copula_seed1 | 0.4873 | 0.3524-0.6215 | +0.1520 | +0.1732 | -0.0219 | -0.0442 |
| gaussian_copula_seed2 | 0.5613 | 0.3867-0.7154 | +0.0780 | +0.2189 | +0.0246 | +0.0231 |
| mst_eps0p5_seed0 | 0.5184 | 0.3821-0.6622 | +0.1209 | +0.1230 | -0.0683 | -0.0518 |
| mst_eps10_seed0 | 0.5578 | 0.4012-0.7178 | +0.0815 | +0.0921 | -0.0515 | -0.0115 |
| mst_eps15_seed0 | 0.5354 | 0.3852-0.6735 | +0.1039 | +0.1678 | -0.0583 | -0.0224 |
| mst_eps15_seed1 | 0.5676 | 0.4045-0.7206 | +0.0717 | +0.1823 | -0.1203 | -0.0962 |
| mst_eps15_seed2 | 0.3682 | 0.2361-0.5059 | +0.2711 | +0.1432 | -0.0555 | -0.0155 |
| mst_eps1_seed0 | 0.5381 | 0.4273-0.6621 | +0.1012 | +0.1309 | -0.0870 | -0.1123 |
| mst_eps20_seed0 | 0.4044 | 0.2741-0.55 | +0.2349 | +0.2714 | -0.0698 | -0.0339 |
| mst_eps5_seed0 | 0.4703 | 0.2939-0.6513 | +0.1690 | +0.1970 | -0.0597 | -0.0238 |
| mst_eps8_seed0 | 0.4547 | 0.3119-0.6133 | +0.1846 | +0.1662 | -0.0511 | -0.0071 |
| patectgan_eps15_seed0 | 0.4414 | 0.3262-0.5659 | +0.1979 | +0.1558 | -0.0156 | -0.0038 |
| patectgan_eps1_seed0 | 0.5182 | 0.3611-0.6672 | +0.1211 | +0.1036 | +0.0182 | +0.0477 |
| patectgan_eps5_seed0 | 0.5734 | 0.4115-0.7339 | +0.0659 | +0.1364 | +0.1525 | +0.1466 |
| tvae_cap256_seed0 | 0.5663 | 0.421-0.7122 | +0.0730 | +0.1300 | -0.0232 | -0.0422 |
| tvae_ep1000_seed0 | 0.645 | 0.5098-0.7748 | -0.0057 | +0.1481 | +0.0559 | +0.1219 |
| tvae_ind_seed0 | 0.5753 | 0.4491-0.6945 | +0.0640 | +0.1776 | +0.0131 | +0.0616 |
| tvae_qt_seed0 | 0.6319 | 0.4934-0.7652 | +0.0074 | +0.0940 | -0.0102 | +0.0418 |
| tvae_qt_seed1 | - | - | target missing or single-class in synthetic data | - | - | - |
| tvae_qt_seed2 | 0.7084 | 0.5837-0.8286 | -0.0691 | -0.0544 | +0.0257 | +0.0790 |
| tvae_seed0 | 0.566 | 0.4108-0.7034 | +0.0733 | +0.0300 | -0.0004 | +0.0303 |
| tvae_seed1 | 0.569 | 0.4373-0.6965 | +0.0703 | +0.0721 | +0.0038 | +0.0225 |
| tvae_seed2 | - | - | target missing or single-class in synthetic data | - | - | - |

## Per (model, ε) across seeds and targets

Gaps vs baseline, lower is better; augmentation Δ is the AUC change from training on real+synthetic vs real alone (positive = synthetic data adds value), and 'vs bootstrap' subtracts the size-matched real-resample control. 'sd n/a (single run)' means no spread can be estimated -- it is NOT the same statement as an observed sd of 0.

| model | ε | runs | HistGB gap ± sd | LogReg gap | augmentation Δ | aug Δ vs bootstrap | Brier gap | worst-stratum gap |
|---|---|---|---|---|---|---|---|---|
| aim | 1 | 1 | +0.0772 (sd n/a: single run) | 0.0755 | -0.0031 | 0.0111 | 0.0845 | 0.1473 |
| aim40 | 1 | 1 | +0.1134 (sd n/a: single run) | 0.1634 | -0.0034 | -0.0026 | 0.1062 | 0.2139 |
| aim40 | 5 | 1 | +0.1474 (sd n/a: single run) | 0.1832 | -0.0267 | -0.0035 | 0.1008 | 0.2297 |
| ctgan | - | 3 | +0.0591 ± 0.0225 | 0.0507 | 0.01 | 0.0183 | 0.0349 | 0.1506 |
| ctgan_qt | - | 1 | +0.0722 (sd n/a: single run) | 0.0489 | -0.011 | -0.0206 | 0.0379 | 0.2255 |
| ddpm | - | 3 | +0.0260 ± 0.0502 | 0.0259 | 0.0079 | 0.0142 | 0.0148 | 0.1034 |
| dpctgan | 8 | 1 | +0.0599 (sd n/a: single run) | 0.0791 | 0.0038 | 0.0137 | -0.0169 | 0.1388 |
| gaussian_copula | - | 3 | +0.0708 ± 0.0233 | 0.109 | 0.0104 | 0.0149 | 0.007 | 0.1768 |
| mst | 0.5 | 1 | +0.0990 (sd n/a: single run) | 0.0831 | -0.0322 | -0.0244 | 0.1139 | 0.1749 |
| mst | 1 | 1 | +0.0614 (sd n/a: single run) | 0.1111 | -0.0265 | -0.0401 | 0.0957 | 0.166 |
| mst | 5 | 1 | +0.0820 (sd n/a: single run) | 0.1247 | -0.0128 | 0.0004 | 0.0749 | 0.1552 |
| mst | 8 | 1 | +0.0777 (sd n/a: single run) | 0.1093 | -0.0273 | -0.0036 | 0.0782 | 0.1746 |
| mst | 10 | 1 | +0.0600 (sd n/a: single run) | 0.0506 | -0.0189 | -0.0018 | 0.0759 | 0.1173 |
| mst | 15 | 3 | +0.1019 ± 0.0079 | 0.1221 | -0.0326 | -0.0154 | 0.1054 | 0.1904 |
| mst | 20 | 1 | +0.1286 (sd n/a: single run) | 0.1009 | -0.0181 | -0.0022 | 0.0821 | 0.2073 |
| patectgan | 1 | 1 | +0.0962 (sd n/a: single run) | 0.0986 | 0.0068 | 0.0202 | 0.0114 | 0.1663 |
| patectgan | 5 | 1 | +0.0659 (sd n/a: single run) | 0.1364 | 0.1525 | 0.1466 | 0.0002 | 0.1171 |
| patectgan | 15 | 1 | +0.1979 (sd n/a: single run) | 0.1558 | -0.0156 | -0.0038 | 0.0003 | 0.2083 |
| tvae | - | 3 | +0.0566 ± 0.0389 | 0.042 | -0.0171 | -0.0106 | 0.0314 | 0.1786 |
| tvae_cap256 | - | 1 | +0.0136 (sd n/a: single run) | 0.0396 | -0.006 | -0.0067 | 0.003 | 0.0733 |
| tvae_ep1000 | - | 1 | -0.0080 (sd n/a: single run) | 0.066 | 0.0182 | 0.0516 | 0.0069 | 0.0803 |
| tvae_ind | - | 1 | +0.0462 (sd n/a: single run) | 0.1034 | 0.0112 | 0.0306 | 0.0204 | 0.1085 |
| tvae_qt | - | 3 | +0.0041 ± 0.0198 | 0.0331 | 0.0042 | 0.016 | 0.0207 | 0.08 |
