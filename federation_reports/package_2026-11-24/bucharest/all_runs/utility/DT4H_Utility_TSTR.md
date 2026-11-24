# Utility: Train-Synthetic, Test-Real (TSTR)

A gradient-boosting classifier is trained on the real TRAINING split (baseline) and on each synthetic dataset, then both are scored on the HOLDOUT split -- real patients that neither the generators nor either classifier ever saw. The closer the TSTR AUC is to the baseline, the more useful the synthetic data is for actual modelling work.

Real train: 4706 rows | holdout test: 1568 rows

## `encounter_primary_reason_non_CV_Disease_f5a_w3a_first`
train 113 labelled (47 positive), holdout 51 labelled (23 positive) | baseline AUC **0.5963** (95% CI 0.4268-0.7544) (HistGB) / 0.441 (LogReg)

CIs are bootstrap over holdout predictions (1000 resamples). 'aug Δ vs bootstrap' is the size-matched control: the augmented AUC minus the AUC of real + bootstrap-resampled REAL rows of the same added size (positive = synthetic beats the pure row-count effect).

| run | TSTR AUC | 95% CI | gap | LogReg gap | augmentation Δ | aug Δ vs bootstrap |
|---|---|---|---|---|---|---|
| ctgan_qt_seed0 | 0.486 | 0.3091-0.6585 | +0.1103 | -0.0994 | -0.0932 | -0.0124 |
| ctgan_seed0 | 0.5171 | 0.3511-0.6785 | +0.0792 | -0.1444 | -0.0249 | -0.0016 |
| ctgan_seed1 | 0.5963 | 0.4425-0.7462 | +0.0000 | +0.0839 | -0.0435 | +0.0326 |
| ctgan_seed2 | 0.472 | 0.3209-0.6333 | +0.1243 | +0.0388 | -0.0699 | -0.0870 |
| ddpm_g_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| ddpm_seed0 | 0.4705 | 0.304-0.632 | +0.1258 | -0.1071 | -0.0357 | -0.0512 |
| ddpm_seed1 | 0.5512 | 0.3991-0.7123 | +0.0451 | +0.0295 | +0.0279 | -0.0171 |
| ddpm_seed2 | 0.5885 | 0.4237-0.7414 | +0.0078 | -0.0932 | +0.0279 | -0.0062 |
| dpctgan_eps10_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps15_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps15_seed1 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps15_seed2 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps1_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps20_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps5_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps8_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| gaussian_copula_seed0 | 0.4115 | 0.2507-0.5708 | +0.1848 | -0.1087 | -0.0295 | -0.0372 |
| gaussian_copula_seed1 | 0.3991 | 0.2419-0.5606 | +0.1972 | -0.1165 | -0.0093 | +0.0218 |
| gaussian_copula_seed2 | 0.4612 | 0.2984-0.6281 | +0.1351 | +0.0808 | -0.0451 | -0.0451 |
| mst_eps15_seed0 | 0.528 | 0.3584-0.6852 | +0.0683 | -0.2174 | -0.0156 | -0.0187 |
| patectgan_eps15_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| patectgan_eps1_seed0 | 0.6071 | 0.444-0.7565 | -0.0108 | -0.1351 | +0.0124 | +0.0590 |
| patectgan_eps5_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| tvae_cap256_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| tvae_ep1000_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| tvae_ind_seed0 | 0.5916 | 0.4099-0.7509 | +0.0047 | -0.0435 | -0.0575 | -0.0916 |
| tvae_qt_seed0 | 0.4309 | 0.2678-0.5957 | +0.1654 | -0.0388 | -0.1258 | -0.1568 |
| tvae_qt_seed1 | 0.4689 | 0.3031-0.6277 | +0.1274 | -0.1429 | -0.0652 | -0.1071 |
| tvae_qt_seed2 | 0.4612 | 0.3088-0.6277 | +0.1351 | -0.0404 | +0.0248 | +0.0217 |
| tvae_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| tvae_seed1 | - | - | target missing or single-class in synthetic data | - | - | - |
| tvae_seed2 | 0.5528 | 0.3923-0.705 | +0.0435 | -0.0699 | -0.0683 | -0.0838 |

## `encounter_primary_reason_CV_Disease_f5a_w3a_first`
train 113 labelled (66 positive), holdout 51 labelled (28 positive) | baseline AUC **0.5963** (95% CI 0.4268-0.7544) (HistGB) / 0.441 (LogReg)

CIs are bootstrap over holdout predictions (1000 resamples). 'aug Δ vs bootstrap' is the size-matched control: the augmented AUC minus the AUC of real + bootstrap-resampled REAL rows of the same added size (positive = synthetic beats the pure row-count effect).

| run | TSTR AUC | 95% CI | gap | LogReg gap | augmentation Δ | aug Δ vs bootstrap |
|---|---|---|---|---|---|---|
| ctgan_qt_seed0 | 0.4457 | 0.2856-0.6091 | +0.1506 | -0.0155 | -0.0606 | -0.0062 |
| ctgan_seed0 | 0.5652 | 0.4005-0.7205 | +0.0311 | +0.1196 | +0.0295 | +0.0699 |
| ctgan_seed1 | 0.6444 | 0.4815-0.7952 | -0.0481 | -0.1273 | +0.0652 | +0.1072 |
| ctgan_seed2 | 0.4845 | 0.3272-0.6382 | +0.1118 | -0.1692 | +0.0341 | +0.0279 |
| ddpm_g_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| ddpm_seed0 | 0.4938 | 0.3451-0.6537 | +0.1025 | -0.0854 | -0.0528 | -0.0947 |
| ddpm_seed1 | 0.5186 | 0.3531-0.6775 | +0.0777 | +0.0155 | +0.0419 | +0.0373 |
| ddpm_seed2 | 0.5683 | 0.4059-0.7277 | +0.0280 | -0.0838 | +0.0326 | -0.0015 |
| dpctgan_eps10_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps15_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps15_seed1 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps15_seed2 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps1_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps20_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps5_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps8_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| gaussian_copula_seed0 | 0.4565 | 0.2984-0.6349 | +0.1398 | -0.0885 | -0.0683 | -0.0465 |
| gaussian_copula_seed1 | 0.5062 | 0.3463-0.6739 | +0.0901 | -0.1165 | -0.0621 | -0.0652 |
| gaussian_copula_seed2 | 0.4969 | 0.3306-0.6652 | +0.0994 | -0.0326 | +0.0015 | +0.0171 |
| mst_eps15_seed0 | 0.4728 | 0.3111-0.64 | +0.1235 | +0.1025 | -0.0109 | +0.0140 |
| patectgan_eps15_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| patectgan_eps1_seed0 | 0.4394 | 0.2921-0.6145 | +0.1569 | +0.0668 | -0.0730 | -0.0823 |
| patectgan_eps5_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| tvae_cap256_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| tvae_ep1000_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| tvae_ind_seed0 | 0.573 | 0.3969-0.7382 | +0.0233 | -0.0714 | -0.0746 | -0.1087 |
| tvae_qt_seed0 | 0.3929 | 0.2269-0.5492 | +0.2034 | -0.0326 | -0.0870 | -0.1398 |
| tvae_qt_seed1 | 0.4876 | 0.3197-0.6426 | +0.1087 | -0.1708 | -0.0420 | -0.0575 |
| tvae_qt_seed2 | 0.4612 | 0.3088-0.6277 | +0.1351 | -0.0404 | +0.0248 | +0.0217 |
| tvae_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| tvae_seed1 | - | - | target missing or single-class in synthetic data | - | - | - |
| tvae_seed2 | 0.4837 | 0.3229-0.6443 | +0.1126 | -0.1009 | -0.0295 | -0.0652 |

## Per (model, ε) across seeds and targets

Gaps vs baseline, lower is better; augmentation Δ is the AUC change from training on real+synthetic vs real alone (positive = synthetic data adds value), and 'vs bootstrap' subtracts the size-matched real-resample control. 'sd n/a (single run)' means no spread can be estimated -- it is NOT the same statement as an observed sd of 0.

| model | ε | runs | HistGB gap ± sd | LogReg gap | augmentation Δ | aug Δ vs bootstrap | Brier gap | worst-stratum gap |
|---|---|---|---|---|---|---|---|---|
| ctgan | - | 3 | +0.0497 ± 0.0712 | -0.0331 | -0.0016 | 0.0248 | 0.0522 | 0.0998 |
| ctgan_qt | - | 1 | +0.1305 (sd n/a: single run) | -0.0575 | -0.0769 | -0.0093 | 0.0876 | 0.1666 |
| ddpm | - | 3 | +0.0645 ± 0.0482 | -0.0541 | 0.007 | -0.0222 | 0.0345 | 0.0852 |
| gaussian_copula | - | 3 | +0.1411 ± 0.0226 | -0.0637 | -0.0355 | -0.0258 | 0.065 | 0.1992 |
| mst | 15 | 1 | +0.0959 (sd n/a: single run) | -0.0575 | -0.0132 | -0.0024 | 0.1999 | 0.1553 |
| patectgan | 1 | 1 | +0.0731 (sd n/a: single run) | -0.0341 | -0.0303 | -0.0117 | 0.0571 | 0.0902 |
| tvae | - | 1 | +0.0781 (sd n/a: single run) | -0.0854 | -0.0489 | -0.0745 | 0.0628 | 0.0852 |
| tvae_ind | - | 1 | +0.0140 (sd n/a: single run) | -0.0575 | -0.066 | -0.1002 | 0.0269 | 0.0927 |
| tvae_qt | - | 3 | +0.1459 ± 0.0345 | -0.0776 | -0.0451 | -0.0696 | 0.1256 | 0.1773 |
