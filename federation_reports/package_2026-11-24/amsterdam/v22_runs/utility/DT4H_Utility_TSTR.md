# Utility: Train-Synthetic, Test-Real (TSTR)

A gradient-boosting classifier is trained on the real TRAINING split (baseline) and on each synthetic dataset, then both are scored on the HOLDOUT split -- real patients that neither the generators nor either classifier ever saw. The closer the TSTR AUC is to the baseline, the more useful the synthetic data is for actual modelling work.

Real train: 3334 rows | holdout test: 1111 rows

## `encounter_primary_reason_CV_Disease_f5a_w7d_first`
train 335 labelled (132 positive), holdout 127 labelled (44 positive) | baseline AUC **0.6024** (95% CI 0.492-0.7101) (HistGB) / 0.5712 (LogReg)

CIs are bootstrap over holdout predictions (1000 resamples). 'aug Δ vs bootstrap' is the size-matched control: the augmented AUC minus the AUC of real + bootstrap-resampled REAL rows of the same added size (positive = synthetic beats the pure row-count effect).

| run | TSTR AUC | 95% CI | gap | LogReg gap | augmentation Δ | aug Δ vs bootstrap |
|---|---|---|---|---|---|---|
| aim40_eps10_seed0 | 0.4673 | 0.3623-0.5792 | +0.1351 | +0.0690 | -0.0285 | -0.0392 |
| aim40_eps15_seed0 | 0.6276 | 0.5171-0.7365 | -0.0252 | -0.0383 | +0.0159 | +0.0153 |
| aim40_eps1_seed0 | 0.5449 | 0.4391-0.6509 | +0.0575 | +0.1043 | +0.0030 | +0.0063 |
| aim40_eps20_seed0 | 0.578 | 0.4678-0.6874 | +0.0244 | +0.0293 | +0.0318 | +0.0312 |
| aim40_eps8_seed0 | 0.5627 | 0.4606-0.6622 | +0.0397 | +0.0665 | -0.0126 | +0.0035 |
| aim50_eps1_seed0 | 0.5756 | 0.4717-0.6762 | +0.0268 | -0.0649 | +0.0082 | +0.0249 |
| aim50_eps5_seed0 | 0.5526 | 0.453-0.6511 | +0.0498 | +0.1638 | +0.0107 | +0.0263 |
| ctgan_qt_seed0 | 0.5101 | 0.4016-0.6146 | +0.0923 | +0.0288 | +0.0077 | +0.0028 |
| ctgan_seed0 | 0.5307 | 0.4199-0.6468 | +0.0717 | +0.0937 | -0.0287 | -0.0361 |
| ctgan_seed1 | 0.5318 | 0.4225-0.6393 | +0.0706 | +0.1161 | -0.0115 | -0.0104 |
| ctgan_seed2 | 0.4852 | 0.3908-0.5892 | +0.1172 | +0.0641 | -0.0906 | -0.1021 |
| ddpm_g_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| ddpm_seed0 | 0.5307 | 0.4173-0.6333 | +0.0717 | +0.0230 | +0.0082 | +0.0145 |
| ddpm_seed1 | 0.4499 | 0.3407-0.5663 | +0.1525 | +0.0416 | -0.0257 | -0.0167 |
| ddpm_seed2 | 0.4819 | 0.3747-0.5885 | +0.1205 | +0.0214 | -0.0534 | -0.0378 |
| dpctgan_eps10_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps15_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps15_seed1 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps15_seed2 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps1_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps20_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps5_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps8_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| gaussian_copula_seed0 | 0.6194 | 0.5086-0.7362 | -0.0170 | +0.0600 | -0.0693 | -0.0811 |
| gaussian_copula_seed1 | 0.5539 | 0.4423-0.6667 | +0.0485 | +0.0980 | -0.0411 | -0.0217 |
| gaussian_copula_seed2 | 0.6093 | 0.4938-0.7134 | -0.0069 | +0.0753 | -0.0257 | +0.0017 |
| mst_eps0p5_seed0 | 0.5728 | 0.4654-0.6845 | +0.0296 | -0.0145 | +0.0074 | -0.0022 |
| mst_eps10_seed0 | 0.4997 | 0.3999-0.5993 | +0.1027 | +0.0950 | -0.0183 | -0.0290 |
| mst_eps15_seed0 | 0.4904 | 0.3828-0.6062 | +0.1120 | +0.1758 | -0.0246 | -0.0123 |
| mst_eps15_seed1 | 0.5082 | 0.4016-0.6125 | +0.0942 | +0.0630 | +0.0233 | +0.0189 |
| mst_eps15_seed2 | 0.4233 | 0.3192-0.5292 | +0.1791 | +0.1843 | -0.0112 | -0.0219 |
| mst_eps1_seed0 | 0.5572 | 0.4518-0.6557 | +0.0452 | -0.0016 | -0.0449 | -0.0457 |
| mst_eps20_seed0 | 0.5096 | 0.4095-0.6112 | +0.0928 | -0.0156 | -0.0438 | -0.0274 |
| mst_eps5_seed0 | 0.4625 | 0.3648-0.5665 | +0.1399 | +0.0849 | -0.0246 | -0.0416 |
| mst_eps8_seed0 | 0.555 | 0.4583-0.6583 | +0.0474 | +0.1153 | -0.0148 | -0.0156 |
| patectgan_eps15_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| patectgan_eps1_seed0 | 0.5016 | 0.3953-0.6071 | +0.1008 | +0.1076 | -0.0375 | -0.0315 |
| patectgan_eps5_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| tvae_cap256_seed0 | 0.5361 | 0.4269-0.6568 | +0.0663 | +0.0178 | +0.0175 | +0.0205 |
| tvae_ep1000_seed0 | 0.5082 | 0.3992-0.6222 | +0.0942 | +0.1539 | -0.0307 | -0.0159 |
| tvae_ind_seed0 | 0.5438 | 0.4324-0.6493 | +0.0586 | +0.1334 | +0.0142 | +0.0454 |
| tvae_qt_seed0 | 0.4912 | 0.3851-0.6029 | +0.1112 | +0.0523 | -0.0134 | -0.0044 |
| tvae_qt_seed1 | 0.5334 | 0.4314-0.6433 | +0.0690 | +0.0416 | -0.0222 | -0.0285 |
| tvae_qt_seed2 | 0.5679 | 0.4698-0.6717 | +0.0345 | -0.0586 | +0.0296 | +0.0378 |
| tvae_seed0 | 0.5148 | 0.4078-0.6283 | +0.0876 | +0.0923 | +0.0011 | +0.0263 |
| tvae_seed1 | 0.6145 | 0.5204-0.7154 | -0.0121 | -0.0446 | +0.0364 | +0.0186 |
| tvae_seed2 | 0.4814 | 0.3716-0.588 | +0.1210 | +0.1210 | -0.0315 | -0.0162 |

## `encounter_primary_reason_non_CV_Disease_f5a_w7d_first`
train 335 labelled (203 positive), holdout 127 labelled (83 positive) | baseline AUC **0.6024** (95% CI 0.492-0.7101) (HistGB) / 0.5712 (LogReg)

CIs are bootstrap over holdout predictions (1000 resamples). 'aug Δ vs bootstrap' is the size-matched control: the augmented AUC minus the AUC of real + bootstrap-resampled REAL rows of the same added size (positive = synthetic beats the pure row-count effect).

| run | TSTR AUC | 95% CI | gap | LogReg gap | augmentation Δ | aug Δ vs bootstrap |
|---|---|---|---|---|---|---|
| aim40_eps10_seed0 | 0.5653 | 0.4562-0.6727 | +0.0371 | -0.0698 | +0.0014 | -0.0030 |
| aim40_eps15_seed0 | 0.6372 | 0.527-0.7378 | -0.0348 | -0.0701 | +0.0164 | +0.0158 |
| aim40_eps1_seed0 | 0.5 | 0.3919-0.6038 | +0.1024 | +0.1199 | +0.0096 | +0.0260 |
| aim40_eps20_seed0 | 0.5802 | 0.4693-0.6812 | +0.0222 | -0.0014 | -0.0044 | -0.0039 |
| aim40_eps8_seed0 | 0.5119 | 0.4097-0.6063 | +0.0905 | -0.0016 | -0.0140 | -0.0118 |
| aim50_eps1_seed0 | 0.5441 | 0.4488-0.6537 | +0.0583 | -0.0556 | +0.0263 | +0.0153 |
| aim50_eps5_seed0 | 0.5208 | 0.4184-0.6199 | +0.0816 | +0.1906 | -0.0252 | -0.0219 |
| ctgan_qt_seed0 | 0.5315 | 0.4321-0.6373 | +0.0709 | +0.1147 | -0.0287 | -0.0506 |
| ctgan_seed0 | 0.5778 | 0.4628-0.6856 | +0.0246 | +0.1087 | +0.0307 | +0.0080 |
| ctgan_seed1 | 0.4959 | 0.388-0.6045 | +0.1065 | +0.1473 | -0.0476 | -0.0646 |
| ctgan_seed2 | 0.6051 | 0.5095-0.7036 | -0.0027 | +0.0331 | -0.0276 | -0.0361 |
| ddpm_g_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| ddpm_seed0 | 0.5687 | 0.4614-0.6699 | +0.0337 | +0.0717 | -0.0668 | -0.0682 |
| ddpm_seed1 | 0.5416 | 0.4367-0.6468 | +0.0608 | +0.1027 | -0.0580 | -0.0605 |
| ddpm_seed2 | 0.5019 | 0.3965-0.6022 | +0.1005 | +0.0463 | -0.0745 | -0.0841 |
| dpctgan_eps10_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps15_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps15_seed1 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps15_seed2 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps1_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps20_seed0 | 0.4755 | 0.3728-0.5769 | +0.1269 | +0.0668 | +0.0257 | +0.0076 |
| dpctgan_eps5_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps8_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| gaussian_copula_seed0 | 0.5761 | 0.4721-0.6788 | +0.0263 | -0.0446 | -0.0052 | -0.0159 |
| gaussian_copula_seed1 | 0.4124 | 0.3115-0.5151 | +0.1900 | +0.1583 | -0.0279 | -0.0285 |
| gaussian_copula_seed2 | 0.491 | 0.3766-0.5911 | +0.1114 | +0.0723 | -0.0569 | -0.0460 |
| mst_eps0p5_seed0 | 0.5728 | 0.4673-0.6693 | +0.0296 | +0.0874 | -0.0167 | -0.0123 |
| mst_eps10_seed0 | 0.4915 | 0.3889-0.5928 | +0.1109 | +0.0682 | -0.0301 | -0.0320 |
| mst_eps15_seed0 | 0.5756 | 0.4743-0.6747 | +0.0268 | +0.1205 | +0.0167 | -0.0003 |
| mst_eps15_seed1 | 0.4942 | 0.3826-0.6012 | +0.1082 | +0.0605 | +0.0093 | +0.0098 |
| mst_eps15_seed2 | 0.3847 | 0.2817-0.4841 | +0.2177 | +0.1881 | +0.0134 | +0.0139 |
| mst_eps1_seed0 | 0.6413 | 0.5477-0.7413 | -0.0389 | +0.0986 | -0.0627 | -0.0430 |
| mst_eps20_seed0 | 0.5041 | 0.4022-0.605 | +0.0983 | +0.1309 | -0.0443 | -0.0449 |
| mst_eps5_seed0 | 0.5498 | 0.4441-0.6543 | +0.0526 | +0.0414 | -0.0183 | -0.0353 |
| mst_eps8_seed0 | 0.5479 | 0.451-0.6467 | +0.0545 | +0.0958 | -0.0345 | -0.0304 |
| patectgan_eps15_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| patectgan_eps1_seed0 | 0.4762 | 0.3661-0.5867 | +0.1262 | +0.1668 | -0.0142 | -0.0200 |
| patectgan_eps5_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| tvae_cap256_seed0 | 0.5561 | 0.4463-0.6772 | +0.0463 | +0.0052 | +0.0080 | +0.0170 |
| tvae_ep1000_seed0 | 0.5082 | 0.3992-0.6222 | +0.0942 | +0.1539 | -0.0307 | -0.0159 |
| tvae_ind_seed0 | 0.5077 | 0.3948-0.6177 | +0.0947 | +0.1325 | +0.0011 | +0.0323 |
| tvae_qt_seed0 | 0.4869 | 0.3811-0.594 | +0.1155 | +0.0539 | -0.0350 | -0.0260 |
| tvae_qt_seed1 | 0.5277 | 0.4278-0.6371 | +0.0747 | +0.0323 | -0.0255 | -0.0365 |
| tvae_qt_seed2 | 0.5852 | 0.4863-0.6871 | +0.0172 | -0.0728 | +0.0156 | +0.0167 |
| tvae_seed0 | 0.5233 | 0.4188-0.6291 | +0.0791 | +0.0808 | -0.0071 | -0.0068 |
| tvae_seed1 | 0.6054 | 0.5111-0.7063 | -0.0030 | -0.0419 | +0.0446 | +0.0336 |
| tvae_seed2 | 0.4932 | 0.3847-0.5989 | +0.1092 | +0.1188 | -0.0274 | -0.0184 |

## `encounter_primary_reason_HF_Disease_f5a_w1mo_first`
train 816 labelled (106 positive), holdout 304 labelled (26 positive) | baseline AUC **0.4661** (95% CI 0.3507-0.5788) (HistGB) / 0.4112 (LogReg)

CIs are bootstrap over holdout predictions (1000 resamples). 'aug Δ vs bootstrap' is the size-matched control: the augmented AUC minus the AUC of real + bootstrap-resampled REAL rows of the same added size (positive = synthetic beats the pure row-count effect).

| run | TSTR AUC | 95% CI | gap | LogReg gap | augmentation Δ | aug Δ vs bootstrap |
|---|---|---|---|---|---|---|
| aim40_eps10_seed0 | 0.4991 | 0.3732-0.622 | -0.0330 | -0.1411 | +0.0227 | -0.0050 |
| aim40_eps15_seed0 | 0.5931 | 0.4816-0.7007 | -0.1270 | -0.1818 | +0.0184 | -0.0217 |
| aim40_eps1_seed0 | 0.5748 | 0.4543-0.6962 | -0.1087 | -0.1096 | +0.0805 | +0.0606 |
| aim40_eps20_seed0 | 0.5448 | 0.4252-0.6728 | -0.0787 | -0.1606 | +0.0296 | +0.0213 |
| aim40_eps8_seed0 | 0.3983 | 0.279-0.5207 | +0.0678 | -0.1284 | +0.0195 | -0.0379 |
| aim50_eps1_seed0 | 0.5325 | 0.3915-0.6501 | -0.0664 | -0.1349 | +0.0466 | +0.0169 |
| aim50_eps5_seed0 | 0.5051 | 0.3749-0.6202 | -0.0390 | -0.1318 | +0.0010 | -0.0441 |
| ctgan_qt_seed0 | 0.4862 | 0.3591-0.6274 | -0.0201 | -0.1360 | -0.0732 | -0.1269 |
| ctgan_seed0 | 0.5228 | 0.4099-0.6392 | -0.0567 | -0.1262 | +0.0505 | +0.0095 |
| ctgan_seed1 | 0.5728 | 0.4477-0.6912 | -0.1067 | -0.0253 | +0.0237 | -0.0304 |
| ctgan_seed2 | 0.4353 | 0.3409-0.5267 | +0.0308 | +0.0133 | +0.0137 | -0.0389 |
| ddpm_g_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| ddpm_seed0 | 0.518 | 0.4139-0.6263 | -0.0519 | -0.0196 | -0.0264 | -0.0639 |
| ddpm_seed1 | 0.5203 | 0.3965-0.6401 | -0.0542 | -0.1224 | +0.0398 | +0.0190 |
| ddpm_seed2 | 0.5643 | 0.4695-0.6599 | -0.0982 | -0.0451 | +0.0475 | +0.0168 |
| dpctgan_eps10_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps15_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps15_seed1 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps15_seed2 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps1_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps20_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| dpctgan_eps5_seed0 | 0.5457 | 0.4169-0.6663 | -0.0796 | -0.0756 | +0.0361 | -0.0031 |
| dpctgan_eps8_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| gaussian_copula_seed0 | 0.558 | 0.425-0.6824 | -0.0919 | +0.0706 | +0.0733 | +0.0051 |
| gaussian_copula_seed1 | 0.4725 | 0.3556-0.5963 | -0.0064 | +0.0251 | +0.0551 | +0.0183 |
| gaussian_copula_seed2 | 0.5048 | 0.3989-0.6196 | -0.0387 | -0.0032 | +0.0256 | -0.0205 |
| mst_eps0p5_seed0 | 0.5899 | 0.4715-0.6978 | -0.1238 | -0.0744 | +0.0185 | -0.0078 |
| mst_eps10_seed0 | 0.6111 | 0.5227-0.6917 | -0.1450 | -0.0957 | +0.0616 | -0.0137 |
| mst_eps15_seed0 | 0.4757 | 0.366-0.5867 | -0.0096 | -0.0477 | +0.0519 | +0.0242 |
| mst_eps15_seed1 | 0.4086 | 0.2926-0.5228 | +0.0575 | -0.1296 | +0.0656 | +0.0119 |
| mst_eps15_seed2 | 0.5718 | 0.4644-0.6885 | -0.1057 | -0.0806 | +0.0089 | -0.0372 |
| mst_eps1_seed0 | 0.4834 | 0.3544-0.6043 | -0.0173 | -0.1403 | +0.0353 | -0.0032 |
| mst_eps20_seed0 | 0.4395 | 0.3328-0.5566 | +0.0266 | -0.2370 | +0.0699 | +0.0422 |
| mst_eps5_seed0 | 0.4127 | 0.3117-0.5253 | +0.0534 | -0.0243 | +0.0372 | -0.0053 |
| mst_eps8_seed0 | 0.5411 | 0.4188-0.6486 | -0.0750 | -0.1728 | +0.0585 | +0.0548 |
| patectgan_eps15_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| patectgan_eps1_seed0 | 0.4935 | 0.3738-0.603 | -0.0274 | -0.0030 | -0.0661 | -0.0934 |
| patectgan_eps5_seed0 | - | - | target missing or single-class in synthetic data | - | - | - |
| tvae_cap256_seed0 | 0.6057 | 0.5113-0.6918 | -0.1396 | +0.0020 | -0.0618 | -0.0904 |
| tvae_ep1000_seed0 | 0.5706 | 0.4589-0.682 | -0.1045 | -0.1530 | +0.0645 | -0.0037 |
| tvae_ind_seed0 | 0.4644 | 0.3581-0.5695 | +0.0017 | -0.0294 | -0.0439 | -0.0921 |
| tvae_qt_seed0 | 0.517 | 0.4185-0.6169 | -0.0509 | -0.1440 | -0.0028 | -0.0208 |
| tvae_qt_seed1 | 0.5957 | 0.4869-0.7106 | -0.1296 | -0.1162 | +0.0073 | -0.0388 |
| tvae_qt_seed2 | 0.5001 | 0.3817-0.6338 | -0.0340 | -0.0918 | +0.0140 | +0.0151 |
| tvae_seed0 | 0.5724 | 0.464-0.6796 | -0.1063 | -0.0586 | -0.0180 | -0.0215 |
| tvae_seed1 | 0.5448 | 0.4489-0.6473 | -0.0787 | -0.0325 | +0.0376 | +0.0087 |
| tvae_seed2 | 0.4733 | 0.3707-0.5788 | -0.0072 | -0.0484 | -0.0001 | -0.0615 |

## `cause_of_death_isCV_f5a_w1mo_first`
_holdout too imbalanced for this target; skipped_

## `cause_of_death_isAllCause_f5a_w1mo_first`
_holdout too imbalanced for this target; skipped_

## Per (model, ε) across seeds and targets

Gaps vs baseline, lower is better; augmentation Δ is the AUC change from training on real+synthetic vs real alone (positive = synthetic data adds value), and 'vs bootstrap' subtracts the size-matched real-resample control. 'sd n/a (single run)' means no spread can be estimated -- it is NOT the same statement as an observed sd of 0.

| model | ε | runs | HistGB gap ± sd | LogReg gap | augmentation Δ | aug Δ vs bootstrap | Brier gap | worst-stratum gap |
|---|---|---|---|---|---|---|---|---|
| aim | 1 | 1 | +0.0062 (sd n/a: single run) | -0.0851 | 0.027 | 0.019 | -0.0097 | 0.1662 |
| aim | 5 | 1 | +0.0308 (sd n/a: single run) | 0.0742 | -0.0045 | -0.0132 | -0.0026 | 0.1702 |
| aim40 | 1 | 1 | +0.0171 (sd n/a: single run) | 0.0382 | 0.031 | 0.031 | 0.0196 | 0.1524 |
| aim40 | 8 | 1 | +0.0660 (sd n/a: single run) | -0.0212 | -0.0024 | -0.0154 | 0.0537 | 0.3169 |
| aim40 | 10 | 1 | +0.0464 (sd n/a: single run) | -0.0473 | -0.0015 | -0.0157 | 0.037 | 0.1916 |
| aim40 | 15 | 1 | -0.0623 (sd n/a: single run) | -0.0967 | 0.0169 | 0.0031 | 0.0255 | 0.1305 |
| aim40 | 20 | 1 | -0.0107 (sd n/a: single run) | -0.0442 | 0.019 | 0.0162 | 0.0459 | 0.1186 |
| ctgan | - | 3 | +0.0284 ± 0.0181 | 0.0472 | -0.0097 | -0.0335 | 0.0574 | 0.2432 |
| ctgan_qt | - | 1 | +0.0477 (sd n/a: single run) | 0.0025 | -0.0314 | -0.0582 | 0.1008 | 0.1422 |
| ddpm | - | 3 | +0.0373 ± 0.0179 | 0.0133 | -0.0233 | -0.0312 | 0.0214 | 0.1738 |
| dpctgan | 5 | 1 | -0.0796 (sd n/a: single run) | -0.0756 | 0.0361 | -0.0031 | -0.0103 | 0.1996 |
| dpctgan | 20 | 1 | +0.1269 (sd n/a: single run) | 0.0668 | 0.0257 | 0.0076 | 0.2407 | 0.2114 |
| gaussian_copula | - | 3 | +0.0239 ± 0.0525 | 0.0569 | -0.008 | -0.021 | 0.0111 | 0.2113 |
| mst | 0.5 | 1 | -0.0215 (sd n/a: single run) | -0.0005 | 0.0031 | -0.0074 | 0.1078 | 0.0882 |
| mst | 1 | 1 | -0.0037 (sd n/a: single run) | -0.0144 | -0.0241 | -0.0306 | 0.0287 | 0.1116 |
| mst | 5 | 1 | +0.0820 (sd n/a: single run) | 0.034 | -0.0019 | -0.0274 | 0.0401 | 0.3386 |
| mst | 8 | 1 | +0.0090 (sd n/a: single run) | 0.0128 | 0.0031 | 0.0029 | 0.1425 | 0.1388 |
| mst | 10 | 1 | +0.0229 (sd n/a: single run) | 0.0225 | 0.0044 | -0.0249 | 0.0492 | 0.0965 |
| mst | 15 | 3 | +0.0756 ± 0.0286 | 0.0594 | 0.017 | 0.0008 | 0.1692 | 0.2871 |
| mst | 20 | 1 | +0.0726 (sd n/a: single run) | -0.0406 | -0.0061 | -0.01 | 0.1464 | 0.2201 |
| patectgan | 1 | 1 | +0.0665 (sd n/a: single run) | 0.0905 | -0.0393 | -0.0483 | 0.0498 | 0.1799 |
| tvae | - | 3 | +0.0211 ± 0.0528 | 0.0208 | 0.004 | -0.0041 | 0.036 | 0.1495 |
| tvae_cap256 | - | 1 | -0.0090 (sd n/a: single run) | 0.0083 | -0.0121 | -0.0176 | 0.0303 | 0.1233 |
| tvae_ep1000 | - | 1 | +0.0280 (sd n/a: single run) | 0.0516 | 0.001 | -0.0118 | 0.0321 | 0.1259 |
| tvae_ind | - | 1 | +0.0517 (sd n/a: single run) | 0.0788 | -0.0095 | -0.0048 | 0.0303 | 0.1571 |
| tvae_qt | - | 3 | +0.0231 ± 0.0308 | -0.0337 | -0.0036 | -0.0095 | 0.073 | 0.0961 |
