# Utility: Train-Synthetic, Test-Real (TSTR)

A gradient-boosting classifier is trained on the real TRAINING split (baseline) and on each synthetic dataset, then both are scored on the HOLDOUT split -- real patients that neither the generators nor either classifier ever saw. The closer the TSTR AUC is to the baseline, the more useful the synthetic data is for actual modelling work.

Real train: 602 rows | holdout test: 200 rows

## `cause_of_death_isCV_f5a_w5a_first`
train 292 labelled (157 positive), holdout 97 labelled (55 positive) | baseline AUC **0.5831** (95% CI 0.4589-0.6977) (HistGB) / 0.6126 (LogReg)

CIs are bootstrap over holdout predictions (1000 resamples). 'aug Δ vs bootstrap' is the size-matched control: the augmented AUC minus the AUC of real + bootstrap-resampled REAL rows of the same added size (positive = synthetic beats the pure row-count effect).

| run | TSTR AUC | 95% CI | gap | LogReg gap | augmentation Δ | aug Δ vs bootstrap |
|---|---|---|---|---|---|---|
| mst_eps15_seed0 | 0.5052 | 0.3807-0.6172 | +0.0779 | +0.1169 | -0.0346 | -0.0402 |

## `cause_of_death_isNonRenalAndNonCV_f5a_w3a_first`
train 227 labelled (88 positive), holdout 75 labelled (29 positive) | baseline AUC **0.533** (95% CI 0.3981-0.6695) (HistGB) / 0.5787 (LogReg)

CIs are bootstrap over holdout predictions (1000 resamples). 'aug Δ vs bootstrap' is the size-matched control: the augmented AUC minus the AUC of real + bootstrap-resampled REAL rows of the same added size (positive = synthetic beats the pure row-count effect).

| run | TSTR AUC | 95% CI | gap | LogReg gap | augmentation Δ | aug Δ vs bootstrap |
|---|---|---|---|---|---|---|
| mst_eps15_seed0 | 0.3793 | 0.2431-0.5067 | +0.1537 | +0.1019 | -0.0315 | -0.0420 |

## `encounter_primary_reason_non_CV_Disease_f5a_w7d_first`
train 525 labelled (142 positive), holdout 172 labelled (42 positive) | baseline AUC **0.6718** (95% CI 0.5827-0.762) (HistGB) / 0.6106 (LogReg)

CIs are bootstrap over holdout predictions (1000 resamples). 'aug Δ vs bootstrap' is the size-matched control: the augmented AUC minus the AUC of real + bootstrap-resampled REAL rows of the same added size (positive = synthetic beats the pure row-count effect).

| run | TSTR AUC | 95% CI | gap | LogReg gap | augmentation Δ | aug Δ vs bootstrap |
|---|---|---|---|---|---|---|
| mst_eps15_seed0 | 0.3597 | 0.2591-0.4649 | +0.3121 | +0.1699 | +0.0326 | +0.0258 |

## `encounter_primary_reason_CV_Disease_f5a_w7d_first`
train 525 labelled (383 positive), holdout 172 labelled (130 positive) | baseline AUC **0.6718** (95% CI 0.5827-0.762) (HistGB) / 0.6106 (LogReg)

CIs are bootstrap over holdout predictions (1000 resamples). 'aug Δ vs bootstrap' is the size-matched control: the augmented AUC minus the AUC of real + bootstrap-resampled REAL rows of the same added size (positive = synthetic beats the pure row-count effect).

| run | TSTR AUC | 95% CI | gap | LogReg gap | augmentation Δ | aug Δ vs bootstrap |
|---|---|---|---|---|---|---|
| mst_eps15_seed0 | 0.3738 | 0.2711-0.4767 | +0.2980 | +0.1941 | +0.0236 | +0.0082 |

## `encounter_primary_reason_HF_Disease_f5a_w1a_first`
train 579 labelled (137 positive), holdout 191 labelled (55 positive) | baseline AUC **0.7293** (95% CI 0.648-0.799) (HistGB) / 0.6758 (LogReg)

CIs are bootstrap over holdout predictions (1000 resamples). 'aug Δ vs bootstrap' is the size-matched control: the augmented AUC minus the AUC of real + bootstrap-resampled REAL rows of the same added size (positive = synthetic beats the pure row-count effect).

| run | TSTR AUC | 95% CI | gap | LogReg gap | augmentation Δ | aug Δ vs bootstrap |
|---|---|---|---|---|---|---|
| mst_eps15_seed0 | 0.6092 | 0.5204-0.6955 | +0.1201 | +0.1397 | -0.0193 | -0.0181 |

## Per (model, ε) across seeds and targets

Gaps vs baseline, lower is better; augmentation Δ is the AUC change from training on real+synthetic vs real alone (positive = synthetic data adds value), and 'vs bootstrap' subtracts the size-matched real-resample control. 'sd n/a (single run)' means no spread can be estimated -- it is NOT the same statement as an observed sd of 0.

| model | ε | runs | HistGB gap ± sd | LogReg gap | augmentation Δ | aug Δ vs bootstrap | Brier gap | worst-stratum gap |
|---|---|---|---|---|---|---|---|---|
| mst | 15 | 1 | +0.1924 (sd n/a: single run) | 0.1445 | -0.0058 | -0.0133 | 0.1659 | 0.2657 |
