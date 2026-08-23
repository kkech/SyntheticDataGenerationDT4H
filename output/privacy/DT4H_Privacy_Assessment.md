# Privacy Assessment: distance to closest training record

Distances are Gower-style mixed-type distances in [0,1] over 61 numeric and 150 categorical columns, computed in sentinel space (a synthetic record is only close to a real one if it matches its values AND its missingness pattern). The baseline is the HOLDOUT distribution: real patients the generators never saw, measured against the training records -- exactly what an innocent 'new' record's distance profile looks like.

**Holdout-to-train baseline**: DCR p5 = `0.062785`, median = `0.102966`, NNDR median = `0.9398`.

| run | DCR min | DCR p5 | DCR median | exact matches | NNDR median | closer than holdout p5 |
|---|---|---|---|---|---|---|
| aim50_eps1_seed0 | 0.353027 | 0.369786 | 0.386943 | 0 | 0.9936 | 0.0% |
| aim50_eps5_seed0 | 0.343727 | 0.361055 | 0.379328 | 0 | 0.9927 | 0.0% |
| ctgan_seed0 | 0.091785 | 0.132437 | 0.179687 | 0 | 0.9801 | 0.0% |
| ctgan_seed1 | 0.094216 | 0.140644 | 0.18135 | 0 | 0.9807 | 0.0% |
| ctgan_seed2 | 0.069094 | 0.103882 | 0.166656 | 0 | 0.9772 | 0.0% |
| dpctgan_eps10_seed0 | 0.216043 | 0.226885 | 0.235274 | 0 | 0.992 | 0.0% |
| dpctgan_eps15_seed0 | 0.191367 | 0.201716 | 0.211122 | 0 | 0.987 | 0.0% |
| dpctgan_eps15_seed1 | 0.191506 | 0.204414 | 0.214251 | 0 | 0.9871 | 0.0% |
| dpctgan_eps15_seed2 | 0.214699 | 0.226565 | 0.235164 | 0 | 0.9846 | 0.0% |
| dpctgan_eps1_seed0 | 0.196087 | 0.213498 | 0.226291 | 0 | 0.9877 | 0.0% |
| dpctgan_eps20_seed0 | 0.214006 | 0.224433 | 0.232842 | 0 | 0.9897 | 0.0% |
| dpctgan_eps5_seed0 | 0.21927 | 0.229584 | 0.239385 | 0 | 0.9728 | 0.0% |
| dpctgan_eps8_seed0 | 0.211142 | 0.222462 | 0.232486 | 0 | 0.9689 | 0.0% |
| gaussian_copula_seed0 | 0.084111 | 0.129515 | 0.173709 | 0 | 0.9776 | 0.0% |
| gaussian_copula_seed1 | 0.065711 | 0.129621 | 0.173972 | 0 | 0.978 | 0.0% |
| gaussian_copula_seed2 | 0.089612 | 0.127807 | 0.173459 | 0 | 0.9781 | 0.0% |
| mst_eps10_seed0 🚨 | 0.041391 | 0.049608 | 0.082917 | 0 | 0.957 | 21.5% |
| mst_eps15_seed0 🚨 | 0.032434 | 0.047899 | 0.086262 | 0 | 0.9492 | 22.0% |
| mst_eps15_seed1 🚨 | 0.031894 | 0.044792 | 0.084358 | 0 | 0.9454 | 25.2% |
| mst_eps15_seed2 🚨 | 0.037906 | 0.047552 | 0.086286 | 0 | 0.9607 | 21.2% |
| mst_eps1_seed0 🚨 | 0.044871 | 0.055546 | 0.097692 | 0 | 0.9588 | 10.8% |
| mst_eps20_seed0 🚨 | 0.035466 | 0.048297 | 0.085216 | 0 | 0.9471 | 20.0% |
| mst_eps5_seed0 🚨 | 0.039638 | 0.053611 | 0.093727 | 0 | 0.9628 | 19.2% |
| mst_eps8_seed0 🚨 | 0.03597 | 0.049183 | 0.085861 | 0 | 0.9473 | 21.8% |
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
