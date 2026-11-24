# Utility: Train-Synthetic, Test-Real (TSTR)

A gradient-boosting classifier is trained on the real TRAINING split (baseline) and on each synthetic dataset, then both are scored on the HOLDOUT split -- real patients that neither the generators nor either classifier ever saw. The closer the TSTR AUC is to the baseline, the more useful the synthetic data is for actual modelling work.

Real train: 602 rows | holdout test: 200 rows

## `cause_of_death_isCV_f5a_w5a_first`
train 292 labelled (157 positive), holdout 97 labelled (55 positive) | baseline AUC **0.5831** (95% CI 0.4589-0.6977) (HistGB) / 0.6126 (LogReg)

CIs are bootstrap over holdout predictions (1000 resamples). 'aug Δ vs bootstrap' is the size-matched control: the augmented AUC minus the AUC of real + bootstrap-resampled REAL rows of the same added size (positive = synthetic beats the pure row-count effect).

| run | TSTR AUC | 95% CI | gap | LogReg gap | augmentation Δ | aug Δ vs bootstrap |
|---|---|---|---|---|---|---|
| tvae_seed0 | 0.6658 | 0.5575-0.7705 | -0.0827 | +0.0442 | +0.0256 | +0.0217 |

## `cause_of_death_isNonRenalAndNonCV_f5a_w3a_first`
train 227 labelled (88 positive), holdout 75 labelled (29 positive) | baseline AUC **0.533** (95% CI 0.3981-0.6695) (HistGB) / 0.5787 (LogReg)

CIs are bootstrap over holdout predictions (1000 resamples). 'aug Δ vs bootstrap' is the size-matched control: the augmented AUC minus the AUC of real + bootstrap-resampled REAL rows of the same added size (positive = synthetic beats the pure row-count effect).

| run | TSTR AUC | 95% CI | gap | LogReg gap | augmentation Δ | aug Δ vs bootstrap |
|---|---|---|---|---|---|---|
| tvae_seed0 | 0.5952 | 0.4576-0.739 | -0.0622 | -0.0457 | +0.0585 | +0.0413 |

## `encounter_primary_reason_non_CV_Disease_f5a_w7d_first`
train 525 labelled (142 positive), holdout 172 labelled (42 positive) | baseline AUC **0.6718** (95% CI 0.5827-0.762) (HistGB) / 0.6106 (LogReg)

CIs are bootstrap over holdout predictions (1000 resamples). 'aug Δ vs bootstrap' is the size-matched control: the augmented AUC minus the AUC of real + bootstrap-resampled REAL rows of the same added size (positive = synthetic beats the pure row-count effect).

| run | TSTR AUC | 95% CI | gap | LogReg gap | augmentation Δ | aug Δ vs bootstrap |
|---|---|---|---|---|---|---|
| tvae_seed0 | 0.5995 | 0.4845-0.7013 | +0.0723 | +0.0577 | +0.0386 | +0.0430 |

## `encounter_primary_reason_CV_Disease_f5a_w7d_first`
train 525 labelled (383 positive), holdout 172 labelled (130 positive) | baseline AUC **0.6718** (95% CI 0.5827-0.762) (HistGB) / 0.6106 (LogReg)

CIs are bootstrap over holdout predictions (1000 resamples). 'aug Δ vs bootstrap' is the size-matched control: the augmented AUC minus the AUC of real + bootstrap-resampled REAL rows of the same added size (positive = synthetic beats the pure row-count effect).

| run | TSTR AUC | 95% CI | gap | LogReg gap | augmentation Δ | aug Δ vs bootstrap |
|---|---|---|---|---|---|---|
| tvae_seed0 | 0.5971 | 0.4886-0.697 | +0.0747 | +0.0522 | -0.0020 | -0.0348 |

## `encounter_primary_reason_HF_Disease_f5a_w1a_first`
train 579 labelled (137 positive), holdout 191 labelled (55 positive) | baseline AUC **0.7293** (95% CI 0.648-0.799) (HistGB) / 0.6758 (LogReg)

CIs are bootstrap over holdout predictions (1000 resamples). 'aug Δ vs bootstrap' is the size-matched control: the augmented AUC minus the AUC of real + bootstrap-resampled REAL rows of the same added size (positive = synthetic beats the pure row-count effect).

| run | TSTR AUC | 95% CI | gap | LogReg gap | augmentation Δ | aug Δ vs bootstrap |
|---|---|---|---|---|---|---|
| tvae_seed0 | 0.7369 | 0.6588-0.8082 | -0.0076 | +0.0311 | -0.0253 | -0.0080 |

## Per (model, ε) across seeds and targets

Gaps vs baseline, lower is better; augmentation Δ is the AUC change from training on real+synthetic vs real alone (positive = synthetic data adds value), and 'vs bootstrap' subtracts the size-matched real-resample control. 'sd n/a (single run)' means no spread can be estimated -- it is NOT the same statement as an observed sd of 0.

| model | ε | runs | HistGB gap ± sd | LogReg gap | augmentation Δ | aug Δ vs bootstrap | Brier gap | worst-stratum gap |
|---|---|---|---|---|---|---|---|---|
| tvae | - | 1 | -0.0011 (sd n/a: single run) | 0.0279 | 0.0191 | 0.0126 | -0.0041 | 0.0878 |
