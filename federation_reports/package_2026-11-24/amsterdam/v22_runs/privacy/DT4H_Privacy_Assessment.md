# Privacy Assessment: distance to closest training record

Distances are Gower-style mixed-type distances in [0,1] over 64 numeric and 153 categorical columns, computed in sentinel space (a synthetic record is only close to a real one if it matches its values AND its missingness pattern). The baseline is the HOLDOUT distribution: real patients the generators never saw, measured against the training records -- exactly what an innocent 'new' record's distance profile looks like.

**Holdout-to-train baseline**: DCR p5 = `0.070828`, median = `0.113321`, NNDR median = `0.9478`.

| run | DCR min | DCR p5 | DCR median | exact matches | NNDR median | closer than holdout p5 |
|---|---|---|---|---|---|---|
| aim40_eps10_seed0 | 0.371988 | 0.382724 | 0.398205 | 0 | 0.9918 | 0.0% |
| aim40_eps15_seed0 | 0.372214 | 0.382417 | 0.397857 | 0 | 0.9914 | 0.0% |
| aim40_eps1_seed0 | 0.373695 | 0.385673 | 0.400823 | 0 | 0.9922 | 0.0% |
| aim40_eps20_seed0 | 0.368422 | 0.379658 | 0.397019 | 0 | 0.9912 | 0.0% |
| aim40_eps8_seed0 | 0.370528 | 0.382829 | 0.397367 | 0 | 0.9921 | 0.0% |
| aim50_eps1_seed0 | 0.347282 | 0.362428 | 0.379258 | 0 | 0.9921 | 0.0% |
| aim50_eps5_seed0 | 0.345653 | 0.356607 | 0.375368 | 0 | 0.9918 | 0.0% |
| ctgan_qt_seed0 | 0.074498 | 0.107018 | 0.182255 | 0 | 0.9789 | 0.0% |
| ctgan_seed0 | 0.092295 | 0.12097 | 0.177884 | 0 | 0.9799 | 0.0% |
| ctgan_seed1 | 0.077638 | 0.125786 | 0.168656 | 0 | 0.9797 | 0.0% |
| ctgan_seed2 | 0.082284 | 0.131311 | 0.19418 | 0 | 0.9821 | 0.0% |
| ddpm_g_seed0 | 0.109255 | 0.131992 | 0.158554 | 0 | 0.9735 | 0.0% |
| ddpm_seed0 | 0.082169 | 0.133459 | 0.183068 | 0 | 0.9722 | 0.0% |
| ddpm_seed1 | 0.084148 | 0.134911 | 0.180742 | 0 | 0.9731 | 0.0% |
| ddpm_seed2 | 0.089882 | 0.136327 | 0.186039 | 0 | 0.9729 | 0.0% |
| dpctgan_eps10_seed0 | 0.220308 | 0.240814 | 0.25107 | 0 | 0.9851 | 0.0% |
| dpctgan_eps15_seed0 | 0.218273 | 0.232922 | 0.242562 | 0 | 0.9864 | 0.0% |
| dpctgan_eps15_seed1 | 0.208294 | 0.217953 | 0.228142 | 0 | 0.9873 | 0.0% |
| dpctgan_eps15_seed2 | 0.236171 | 0.247782 | 0.258929 | 0 | 0.9417 | 0.0% |
| dpctgan_eps1_seed0 | 0.213099 | 0.231148 | 0.24458 | 0 | 0.9856 | 0.0% |
| dpctgan_eps20_seed0 | 0.199191 | 0.211464 | 0.221923 | 0 | 0.9833 | 0.0% |
| dpctgan_eps5_seed0 | 0.235278 | 0.246717 | 0.255613 | 0 | 0.9924 | 0.0% |
| dpctgan_eps8_seed0 | 0.214532 | 0.227714 | 0.239049 | 0 | 0.9893 | 0.0% |
| gaussian_copula_seed0 | 0.072982 | 0.125918 | 0.175474 | 0 | 0.9763 | 0.0% |
| gaussian_copula_seed1 | 0.079182 | 0.12563 | 0.17536 | 0 | 0.9762 | 0.0% |
| gaussian_copula_seed2 | 0.077352 | 0.125134 | 0.174837 | 0 | 0.9771 | 0.0% |
| mst_eps0p5_seed0 🚨 | 0.038057 | 0.055283 | 0.130336 | 0 | 0.9602 | 17.2% |
| mst_eps10_seed0 🚨 | 0.048256 | 0.057836 | 0.107197 | 0 | 0.9505 | 22.9% |
| mst_eps15_seed0 🚨 | 0.048108 | 0.058478 | 0.103616 | 0 | 0.9498 | 22.1% |
| mst_eps15_seed1 🚨 | 0.049729 | 0.058549 | 0.101026 | 0 | 0.9499 | 23.9% |
| mst_eps15_seed2 🚨 | 0.046007 | 0.057332 | 0.106503 | 0 | 0.9483 | 23.5% |
| mst_eps1_seed0 🚨 | 0.045477 | 0.060997 | 0.119891 | 0 | 0.953 | 20.1% |
| mst_eps20_seed0 🚨 | 0.048719 | 0.057648 | 0.0968 | 0 | 0.9508 | 25.4% |
| mst_eps5_seed0 🚨 | 0.045965 | 0.05761 | 0.095667 | 0 | 0.9483 | 22.5% |
| mst_eps8_seed0 🚨 | 0.038891 | 0.057018 | 0.110765 | 0 | 0.9497 | 27.5% |
| patectgan_eps15_seed0 | 0.056598 | 0.082586 | 0.120536 | 0 | 0.9623 | 1.1% |
| patectgan_eps1_seed0 | 0.173851 | 0.208091 | 0.237982 | 0 | 0.9857 | 0.0% |
| patectgan_eps5_seed0 | 0.074246 | 0.11416 | 0.146584 | 0 | 0.9698 | 0.0% |
| tvae_cap256_seed0 🚨 | 0.027993 | 0.059946 | 0.096813 | 0 | 0.9424 | 13.1% |
| tvae_ep1000_seed0 🚨 | 0.040067 | 0.061426 | 0.099403 | 0 | 0.9446 | 11.3% |
| tvae_ind_seed0 🚨 | 0.024492 | 0.057639 | 0.096138 | 0 | 0.9419 | 15.7% |
| tvae_qt_seed0 🚨 | 0.034135 | 0.062681 | 0.100801 | 0 | 0.9412 | 10.8% |
| tvae_qt_seed1 | 0.029819 | 0.063096 | 0.102395 | 0 | 0.9405 | 9.9% |
| tvae_qt_seed2 🚨 | 0.03121 | 0.05925 | 0.100154 | 0 | 0.943 | 12.8% |
| tvae_seed0 🚨 | 0.025559 | 0.061037 | 0.099244 | 0 | 0.9415 | 12.2% |
| tvae_seed1 🚨 | 0.037494 | 0.060586 | 0.097774 | 0 | 0.9442 | 12.2% |
| tvae_seed2 🚨 | 0.031986 | 0.060694 | 0.097106 | 0 | 0.942 | 12.8% |

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
