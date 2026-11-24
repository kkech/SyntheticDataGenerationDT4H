# Privacy Assessment: distance to closest training record

Distances are Gower-style mixed-type distances in [0,1] over 65 numeric and 86 categorical columns, computed in sentinel space (a synthetic record is only close to a real one if it matches its values AND its missingness pattern). The baseline is the HOLDOUT distribution: real patients the generators never saw, measured against the training records -- exactly what an innocent 'new' record's distance profile looks like.

**Holdout-to-train baseline**: DCR p5 = `0.028125`, median = `0.053831`, NNDR median = `0.9209`.

| run | DCR min | DCR p5 | DCR median | exact matches | NNDR median | closer than holdout p5 |
|---|---|---|---|---|---|---|
| aim40_eps1_seed0 | 0.15587 | 0.173268 | 0.19752 | 0 | 0.9814 | 0.0% |
| aim40_eps5_seed0 | 0.155803 | 0.171096 | 0.196392 | 0 | 0.9812 | 0.0% |
| aim50_eps1_seed0 | 0.129904 | 0.151664 | 0.171508 | 0 | 0.9803 | 0.0% |
| ctgan_qt_seed0 | 0.067294 | 0.0936 | 0.144845 | 0 | 0.9779 | 0.0% |
| ctgan_seed0 | 0.061279 | 0.091229 | 0.137158 | 0 | 0.9791 | 0.0% |
| ctgan_seed1 | 0.066219 | 0.089949 | 0.137371 | 0 | 0.9782 | 0.0% |
| ctgan_seed2 | 0.061854 | 0.089733 | 0.133288 | 0 | 0.98 | 0.0% |
| ddpm_g_seed0 | 0.03104 | 0.050432 | 0.08191 | 0 | 0.9608 | 0.0% |
| ddpm_seed0 | 0.017688 | 0.046317 | 0.082614 | 0 | 0.9516 | 0.2% |
| ddpm_seed1 | 0.015203 | 0.047298 | 0.083737 | 0 | 0.9532 | 0.3% |
| ddpm_seed2 | 0.018634 | 0.047615 | 0.083786 | 0 | 0.9488 | 0.3% |
| dpctgan_eps10_seed0 | 0.184542 | 0.195317 | 0.204331 | 0 | 0.9798 | 0.0% |
| dpctgan_eps15_seed0 | 0.140874 | 0.152177 | 0.162454 | 0 | 0.9756 | 0.0% |
| dpctgan_eps15_seed1 | 0.142118 | 0.156091 | 0.168424 | 0 | 0.9877 | 0.0% |
| dpctgan_eps15_seed2 | 0.19757 | 0.209102 | 0.218916 | 0 | 0.9895 | 0.0% |
| dpctgan_eps1_seed0 | 0.161603 | 0.173147 | 0.185546 | 0 | 0.9401 | 0.0% |
| dpctgan_eps20_seed0 | 0.200167 | 0.212667 | 0.225059 | 0 | 0.9919 | 0.0% |
| dpctgan_eps5_seed0 | 0.162027 | 0.173332 | 0.18519 | 0 | 0.9931 | 0.0% |
| dpctgan_eps8_seed0 | 0.191324 | 0.204886 | 0.216296 | 0 | 0.9867 | 0.0% |
| gaussian_copula_seed0 | 0.063506 | 0.089234 | 0.13068 | 0 | 0.9765 | 0.0% |
| gaussian_copula_seed1 | 0.062296 | 0.091287 | 0.130545 | 0 | 0.9766 | 0.0% |
| gaussian_copula_seed2 | 0.047137 | 0.090272 | 0.130785 | 0 | 0.9764 | 0.0% |
| mst_eps0p5_seed0 | 0.05662 | 0.068966 | 0.108722 | 0 | 0.975 | 0.0% |
| mst_eps10_seed0 | 0.05806 | 0.062818 | 0.090426 | 0 | 0.9555 | 0.0% |
| mst_eps15_seed0 | 0.054636 | 0.06445 | 0.090159 | 0 | 0.9569 | 0.0% |
| mst_eps15_seed1 | 0.058593 | 0.06346 | 0.088671 | 0 | 0.9587 | 0.0% |
| mst_eps15_seed2 | 0.050042 | 0.062812 | 0.088964 | 0 | 0.9559 | 0.0% |
| mst_eps1_seed0 | 0.054242 | 0.063849 | 0.09081 | 0 | 0.9628 | 0.0% |
| mst_eps20_seed0 | 0.055018 | 0.063544 | 0.090241 | 0 | 0.9593 | 0.0% |
| mst_eps5_seed0 | 0.050875 | 0.065336 | 0.089401 | 0 | 0.9636 | 0.0% |
| mst_eps8_seed0 | 0.050875 | 0.062795 | 0.089168 | 0 | 0.9655 | 0.0% |
| patectgan_eps15_seed0 | 0.055002 | 0.081445 | 0.110954 | 0 | 0.9571 | 0.0% |
| patectgan_eps1_seed0 | 0.153059 | 0.190941 | 0.226973 | 0 | 0.9883 | 0.0% |
| patectgan_eps5_seed0 | 0.049074 | 0.073838 | 0.098291 | 0 | 0.9637 | 0.0% |
| tvae_cap256_seed0 | 0.011215 | 0.031737 | 0.057075 | 0 | 0.9267 | 2.7% |
| tvae_ep1000_seed0 | 0.013294 | 0.03292 | 0.057545 | 0 | 0.9285 | 1.6% |
| tvae_ind_seed0 | 0.009524 | 0.030212 | 0.053925 | 0 | 0.9259 | 3.0% |
| tvae_qt_seed0 | 0.009375 | 0.032243 | 0.056304 | 0 | 0.9279 | 2.0% |
| tvae_qt_seed1 | 0.011482 | 0.033619 | 0.059545 | 0 | 0.9263 | 1.7% |
| tvae_qt_seed2 | 0.013146 | 0.032284 | 0.056944 | 0 | 0.9269 | 2.4% |
| tvae_seed0 | 0.013251 | 0.029927 | 0.054581 | 0 | 0.9248 | 3.6% |
| tvae_seed1 | 0.013223 | 0.03201 | 0.057119 | 0 | 0.9233 | 2.6% |
| tvae_seed2 | 0.011925 | 0.030381 | 0.054208 | 0 | 0.9223 | 3.2% |

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
