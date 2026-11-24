# Privacy Assessment: distance to closest training record

Distances are Gower-style mixed-type distances in [0,1] over 66 numeric and 168 categorical columns, computed in sentinel space (a synthetic record is only close to a real one if it matches its values AND its missingness pattern). The baseline is the HOLDOUT distribution: real patients the generators never saw, measured against the training records -- exactly what an innocent 'new' record's distance profile looks like.

**Holdout-to-train baseline**: DCR p5 = `0.054094`, median = `0.095201`, NNDR median = `0.9403`.

| run | DCR min | DCR p5 | DCR median | exact matches | NNDR median | closer than holdout p5 |
|---|---|---|---|---|---|---|
| aim40_eps1_seed0 | 0.431093 | 0.440057 | 0.453898 | 0 | 0.9915 | 0.0% |
| aim40_eps5_seed0 | 0.426526 | 0.437924 | 0.4508 | 0 | 0.9926 | 0.0% |
| aim40_eps8_seed0 | 0.42241 | 0.432413 | 0.447162 | 0 | 0.9907 | 0.0% |
| aim50_eps1_seed0 | 0.405273 | 0.42071 | 0.437104 | 0 | 0.9897 | 0.0% |
| aim50_eps5_seed0 | 0.399689 | 0.415879 | 0.432064 | 0 | 0.991 | 0.0% |
| aim50_eps8_seed0 | 0.407322 | 0.417028 | 0.431819 | 0 | 0.9893 | 0.0% |
| ctgan_qt_seed0 | 0.109369 | 0.161788 | 0.202502 | 0 | 0.9856 | 0.0% |
| ctgan_seed0 | 0.147428 | 0.168204 | 0.196809 | 0 | 0.9839 | 0.0% |
| ctgan_seed1 | 0.129056 | 0.158923 | 0.189008 | 0 | 0.9835 | 0.0% |
| ctgan_seed2 | 0.141015 | 0.167019 | 0.195335 | 0 | 0.9837 | 0.0% |
| ddpm_g_seed0 | 0.119324 | 0.165524 | 0.206129 | 0 | 0.9772 | 0.0% |
| ddpm_seed0 | 0.111378 | 0.156273 | 0.210933 | 0 | 0.975 | 0.0% |
| ddpm_seed1 | 0.086661 | 0.15649 | 0.207848 | 0 | 0.9743 | 0.0% |
| ddpm_seed2 | 0.108321 | 0.154574 | 0.206776 | 0 | 0.9752 | 0.0% |
| dpctgan_eps10_seed0 | 0.205648 | 0.214072 | 0.224143 | 0 | 0.994 | 0.0% |
| dpctgan_eps15_seed0 | 0.225382 | 0.236973 | 0.249251 | 0 | 0.9972 | 0.0% |
| dpctgan_eps15_seed1 | 0.197797 | 0.20666 | 0.215375 | 0 | 0.9784 | 0.0% |
| dpctgan_eps15_seed2 | 0.190904 | 0.202557 | 0.21349 | 0 | 0.9951 | 0.0% |
| dpctgan_eps1_seed0 | 0.27532 | 0.28972 | 0.301889 | 0 | 0.9924 | 0.0% |
| dpctgan_eps20_seed0 | 0.225028 | 0.237021 | 0.24682 | 0 | 0.99 | 0.0% |
| dpctgan_eps5_seed0 | 0.221232 | 0.232603 | 0.242875 | 0 | 0.982 | 0.0% |
| dpctgan_eps8_seed0 | 0.237484 | 0.257045 | 0.268152 | 0 | 0.9814 | 0.0% |
| gaussian_copula_seed0 | 0.09822 | 0.128447 | 0.173361 | 0 | 0.977 | 0.0% |
| gaussian_copula_seed1 | 0.09215 | 0.127892 | 0.172758 | 0 | 0.9761 | 0.0% |
| gaussian_copula_seed2 | 0.092445 | 0.129439 | 0.174425 | 0 | 0.9769 | 0.0% |
| mst_eps10_seed0 | 0.054177 | 0.067838 | 0.094382 | 0 | 0.9534 | 0.0% |
| mst_eps15_seed0 | 0.055828 | 0.070155 | 0.099913 | 0 | 0.9532 | 0.0% |
| mst_eps15_seed1 | 0.055251 | 0.064443 | 0.099313 | 0 | 0.942 | 0.0% |
| mst_eps15_seed2 | 0.054521 | 0.070677 | 0.098285 | 0 | 0.9529 | 0.0% |
| mst_eps1_seed0 | 0.05681 | 0.077688 | 0.154393 | 0 | 0.9651 | 0.0% |
| mst_eps20_seed0 | 0.052896 | 0.06866 | 0.104814 | 0 | 0.9615 | 0.2% |
| mst_eps5_seed0 | 0.05463 | 0.06839 | 0.101063 | 0 | 0.9565 | 0.0% |
| mst_eps8_seed0 | 0.049707 | 0.065489 | 0.100472 | 0 | 0.9548 | 0.3% |
| patectgan_eps15_seed0 | 0.052545 | 0.079657 | 0.115746 | 0 | 0.9582 | 0.1% |
| patectgan_eps1_seed0 | 0.168392 | 0.19619 | 0.224133 | 0 | 0.9859 | 0.0% |
| patectgan_eps5_seed0 | 0.075018 | 0.101973 | 0.129277 | 0 | 0.9711 | 0.0% |
| tvae_cap256_seed0 | 0.028839 | 0.050197 | 0.082718 | 0 | 0.928 | 7.9% |
| tvae_ep1000_seed0 | 0.028017 | 0.051489 | 0.084611 | 0 | 0.93 | 6.9% |
| tvae_ind_seed0 | 0.019173 | 0.049515 | 0.080325 | 0 | 0.9289 | 9.2% |
| tvae_qt_seed0 | 0.029844 | 0.052731 | 0.084205 | 0 | 0.928 | 6.3% |
| tvae_qt_seed1 | 0.016791 | 0.049942 | 0.082361 | 0 | 0.9298 | 8.6% |
| tvae_qt_seed2 | 0.025761 | 0.050549 | 0.082761 | 0 | 0.9315 | 8.2% |
| tvae_seed0 | 0.02778 | 0.048767 | 0.082238 | 0 | 0.9355 | 8.9% |
| tvae_seed1 | 0.029485 | 0.050035 | 0.084938 | 0 | 0.9316 | 7.2% |
| tvae_seed2 | 0.031771 | 0.050822 | 0.083266 | 0 | 0.9307 | 8.1% |

Reading the table: `closer than holdout p5` is the share of synthetic records nearer to some training record than the closest 5% of unseen-real-patient distances -- ~5% is the no-memorization expectation; well above that suggests the model echoes the individuals it trained on. `exact matches` must be 0 for any release. NNDR near 1 means records sit between real records (population structure), near 0 means they lock onto one real record.

## Limitations
- DCR/NNDR against the holdout baseline bound record-copying with a genuine
  unseen-data reference. A full adversarial membership-inference evaluation
  (shadow models, per-record attack scores) remains future work; for DP
  synthesizers the epsilon guarantee bounds membership inference by
  construction.
- Width-limited (AIM) runs generate a column subset; their absent columns are
  padded as missing on the synthetic side before encoding. Their DCR values
  are therefore NOT directly comparable to full-width runs -- compare
  width-limited runs only against each other and against the shared baseline.
