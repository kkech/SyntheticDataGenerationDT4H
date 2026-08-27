# Privacy Assessment: distance to closest training record

Distances are Gower-style mixed-type distances in [0,1] over 61 numeric and 150 categorical columns, computed in sentinel space (a synthetic record is only close to a real one if it matches its values AND its missingness pattern). The baseline is the HOLDOUT distribution: real patients the generators never saw, measured against the training records -- exactly what an innocent 'new' record's distance profile looks like.

**Holdout-to-train baseline**: DCR p5 = `0.062785`, median = `0.102966`, NNDR median = `0.9398`.

| run | DCR min | DCR p5 | DCR median | exact matches | NNDR median | closer than holdout p5 |
|---|---|---|---|---|---|---|
| aim40_eps1_seed0 | 0.376438 | 0.386752 | 0.404519 | 0 | 0.9933 | 0.0% |
| aim50_eps1_seed0 | 0.354517 | 0.366649 | 0.384297 | 0 | 0.9934 | 0.0% |
| ctgan_qt_seed0 | 0.068466 | 0.106037 | 0.184491 | 0 | 0.9795 | 0.0% |
| ctgan_seed0 | 0.091785 | 0.132437 | 0.179687 | 0 | 0.9801 | 0.0% |
| ctgan_seed1 | 0.094216 | 0.140644 | 0.18135 | 0 | 0.9807 | 0.0% |
| ctgan_seed2 | 0.069094 | 0.103882 | 0.166656 | 0 | 0.9772 | 0.0% |
| ddpm_g_seed0 | 0.195071 | 0.313725 | 0.401206 | 0 | 0.987 | 0.0% |
| ddpm_seed0 | 0.20816 | 0.326814 | 0.414036 | 0 | 0.987 | 0.0% |
| ddpm_seed1 | 0.258163 | 0.327822 | 0.406726 | 0 | 0.9855 | 0.0% |
| ddpm_seed2 | 0.202616 | 0.327535 | 0.410581 | 0 | 0.9859 | 0.0% |
| dpctgan_eps10_seed0 | 0.183537 | 0.195024 | 0.206085 | 0 | 0.9913 | 0.0% |
| dpctgan_eps15_seed0 | 0.200225 | 0.211402 | 0.221402 | 0 | 0.9832 | 0.0% |
| dpctgan_eps15_seed1 | 0.234269 | 0.248353 | 0.261603 | 0 | 0.9916 | 0.0% |
| dpctgan_eps15_seed2 | 0.176825 | 0.189107 | 0.199433 | 0 | 0.984 | 0.0% |
| dpctgan_eps1_seed0 | 0.180818 | 0.209548 | 0.222229 | 0 | 0.9752 | 0.0% |
| dpctgan_eps20_seed0 | 0.233465 | 0.243828 | 0.253363 | 0 | 0.9939 | 0.0% |
| dpctgan_eps5_seed0 | 0.202849 | 0.212131 | 0.220268 | 0 | 0.9873 | 0.0% |
| dpctgan_eps8_seed0 | 0.222628 | 0.236763 | 0.247279 | 0 | 0.9878 | 0.0% |
| gaussian_copula_seed0 | 0.084111 | 0.129515 | 0.173709 | 0 | 0.9776 | 0.0% |
| gaussian_copula_seed1 | 0.065711 | 0.129621 | 0.173972 | 0 | 0.978 | 0.0% |
| gaussian_copula_seed2 | 0.089612 | 0.127807 | 0.173459 | 0 | 0.9781 | 0.0% |
| mst_eps0p5_seed0 | 0.040759 | 0.063976 | 0.158893 | 0 | 0.9689 | 4.7% |
| mst_eps10_seed0 🚨 | 0.037313 | 0.051453 | 0.086731 | 0 | 0.9574 | 17.8% |
| mst_eps15_seed0 🚨 | 0.034419 | 0.044012 | 0.085083 | 0 | 0.9529 | 24.0% |
| mst_eps15_seed1 🚨 | 0.043334 | 0.051117 | 0.085691 | 0 | 0.9483 | 17.9% |
| mst_eps15_seed2 🚨 | 0.038685 | 0.045899 | 0.08532 | 0 | 0.9502 | 26.2% |
| mst_eps1_seed0 🚨 | 0.042455 | 0.05295 | 0.100318 | 0 | 0.9636 | 24.7% |
| mst_eps20_seed0 🚨 | 0.037173 | 0.04861 | 0.088044 | 0 | 0.9563 | 19.1% |
| mst_eps5_seed0 🚨 | 0.043184 | 0.04899 | 0.089495 | 0 | 0.9497 | 20.8% |
| mst_eps8_seed0 🚨 | 0.037941 | 0.046861 | 0.086909 | 0 | 0.9509 | 28.8% |
| patectgan_eps15_seed0 | 0.0425 | 0.073347 | 0.113533 | 0 | 0.961 | 1.3% |
| patectgan_eps1_seed0 | 0.160478 | 0.191571 | 0.221605 | 0 | 0.986 | 0.0% |
| patectgan_eps5_seed0 | 0.051585 | 0.081631 | 0.117398 | 0 | 0.9625 | 0.3% |
| tvae_cap256_seed0 🚨 | 0.028081 | 0.053457 | 0.090436 | 0 | 0.941 | 11.8% |
| tvae_ep1000_seed0 🚨 | 0.020357 | 0.056132 | 0.089641 | 0 | 0.9395 | 11.1% |
| tvae_ind_seed0 🚨 | 0.0198 | 0.052534 | 0.087355 | 0 | 0.938 | 13.2% |
| tvae_qt_seed0 🚨 | 0.025916 | 0.054376 | 0.08855 | 0 | 0.9386 | 12.6% |
| tvae_qt_seed1 🚨 | 0.01734 | 0.055847 | 0.089755 | 0 | 0.9422 | 10.1% |
| tvae_qt_seed2 🚨 | 0.031496 | 0.055858 | 0.090803 | 0 | 0.9423 | 10.2% |
| tvae_seed0 🚨 | 0.021459 | 0.053666 | 0.089061 | 0 | 0.9391 | 12.2% |
| tvae_seed1 🚨 | 0.017366 | 0.053134 | 0.088403 | 0 | 0.9412 | 12.8% |
| tvae_seed2 🚨 | 0.01983 | 0.054375 | 0.088856 | 0 | 0.9399 | 10.8% |

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
