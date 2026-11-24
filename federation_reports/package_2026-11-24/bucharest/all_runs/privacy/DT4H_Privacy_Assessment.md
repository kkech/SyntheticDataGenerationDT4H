# Privacy Assessment: distance to closest training record

Distances are Gower-style mixed-type distances in [0,1] over 38 numeric and 63 categorical columns, computed in sentinel space (a synthetic record is only close to a real one if it matches its values AND its missingness pattern). The baseline is the HOLDOUT distribution: real patients the generators never saw, measured against the training records -- exactly what an innocent 'new' record's distance profile looks like.

**Holdout-to-train baseline**: DCR p5 = `0.025555`, median = `0.046082`, NNDR median = `0.9239`.

| run | DCR min | DCR p5 | DCR median | exact matches | NNDR median | closer than holdout p5 |
|---|---|---|---|---|---|---|
| ctgan_qt_seed0 | 0.028039 | 0.057732 | 0.082836 | 0 | 0.9655 | 0.0% |
| ctgan_seed0 | 0.030747 | 0.058028 | 0.083531 | 0 | 0.9643 | 0.0% |
| ctgan_seed1 | 0.037298 | 0.067058 | 0.09493 | 0 | 0.9672 | 0.0% |
| ctgan_seed2 | 0.024253 | 0.057275 | 0.083602 | 0 | 0.9654 | 0.0% |
| ddpm_g_seed0 | 0.012949 | 0.031094 | 0.049438 | 0 | 0.9332 | 1.2% |
| ddpm_seed0 | 0.011039 | 0.027178 | 0.045523 | 0 | 0.9283 | 3.2% |
| ddpm_seed1 | 0.0123 | 0.027672 | 0.045777 | 0 | 0.9259 | 3.3% |
| ddpm_seed2 | 0.012324 | 0.027768 | 0.045059 | 0 | 0.9283 | 3.1% |
| dpctgan_eps10_seed0 | 0.099088 | 0.116402 | 0.128218 | 0 | 0.9892 | 0.0% |
| dpctgan_eps15_seed0 | 0.122605 | 0.13716 | 0.148365 | 0 | 0.9859 | 0.0% |
| dpctgan_eps15_seed1 | 0.106242 | 0.120897 | 0.131974 | 0 | 0.9873 | 0.0% |
| dpctgan_eps15_seed2 | 0.124886 | 0.136917 | 0.151194 | 0 | 0.9915 | 0.0% |
| dpctgan_eps1_seed0 | 0.098691 | 0.114701 | 0.130716 | 0 | 0.9792 | 0.0% |
| dpctgan_eps20_seed0 | 0.112699 | 0.12804 | 0.139701 | 0 | 0.9868 | 0.0% |
| dpctgan_eps5_seed0 | 0.08604 | 0.099358 | 0.11239 | 0 | 0.9668 | 0.0% |
| dpctgan_eps8_seed0 | 0.083795 | 0.098743 | 0.11261 | 0 | 0.9751 | 0.0% |
| gaussian_copula_seed0 | 0.02749 | 0.057866 | 0.086224 | 0 | 0.9628 | 0.0% |
| gaussian_copula_seed1 | 0.024836 | 0.057869 | 0.086636 | 0 | 0.9612 | 0.0% |
| gaussian_copula_seed2 | 0.022778 | 0.058354 | 0.08648 | 0 | 0.9623 | 0.0% |
| mst_eps15_seed0 | 0.063862 | 0.072517 | 0.087961 | 0 | 0.9626 | 0.0% |
| patectgan_eps15_seed0 | 0.0173 | 0.033698 | 0.05113 | 0 | 0.9374 | 0.5% |
| patectgan_eps1_seed0 | 0.110779 | 0.155381 | 0.200483 | 0 | 0.9858 | 0.0% |
| patectgan_eps5_seed0 | 0.023686 | 0.039169 | 0.057309 | 0 | 0.9455 | 0.1% |
| tvae_cap256_seed0 | 0.007488 | 0.023359 | 0.038711 | 0 | 0.9169 | 8.0% |
| tvae_ep1000_seed0 | 0.00952 | 0.024274 | 0.041429 | 0 | 0.9196 | 6.3% |
| tvae_ind_seed0 | 0.009819 | 0.022294 | 0.037412 | 0 | 0.9143 | 10.0% |
| tvae_qt_seed0 | 0.010031 | 0.024088 | 0.039637 | 0 | 0.9186 | 6.9% |
| tvae_qt_seed1 | 0.011321 | 0.024831 | 0.041336 | 0 | 0.9221 | 5.7% |
| tvae_qt_seed2 | 0.010945 | 0.023235 | 0.039005 | 0 | 0.9193 | 8.1% |
| tvae_seed0 | 0.008194 | 0.023801 | 0.040173 | 0 | 0.9184 | 7.5% |
| tvae_seed1 | 0.011456 | 0.023466 | 0.039331 | 0 | 0.917 | 8.2% |
| tvae_seed2 | 0.008366 | 0.024315 | 0.040821 | 0 | 0.9211 | 6.5% |

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
