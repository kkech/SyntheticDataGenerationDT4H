# Privacy Assessment: distance to closest training record

Distances are Gower-style mixed-type distances in [0,1] over 61 numeric and 150 categorical columns, computed in sentinel space (a synthetic record is only close to a real one if it matches its values AND its missingness pattern). The baseline is the HOLDOUT distribution: real patients the generators never saw, measured against the training records -- exactly what an innocent 'new' record's distance profile looks like.

**Holdout-to-train baseline**: DCR p5 = `0.062785`, median = `0.102966`, NNDR median = `0.9398`.

| run | DCR min | DCR p5 | DCR median | exact matches | NNDR median | closer than holdout p5 |
|---|---|---|---|---|---|---|
| aim50_eps1_seed0 | 0.431167 | 0.442377 | 0.454952 | 0 | 0.9963 | 0.0% |
| aim50_eps5_seed0 | 0.424296 | 0.436431 | 0.450604 | 0 | 0.996 | 0.0% |
| ctgan_seed0 | 0.432214 | 0.467959 | 0.503157 | 0 | 0.9967 | 0.0% |
| ctgan_seed1 | 0.448901 | 0.474971 | 0.506133 | 0 | 0.9969 | 0.0% |
| ctgan_seed2 | 0.44086 | 0.455806 | 0.495856 | 0 | 0.9967 | 0.0% |
| dpctgan_eps10_seed0 | 0.541685 | 0.550177 | 0.556149 | 0 | 0.9984 | 0.0% |
| dpctgan_eps15_seed0 | 0.535994 | 0.545774 | 0.552005 | 0 | 0.9987 | 0.0% |
| dpctgan_eps15_seed1 | 0.515571 | 0.523417 | 0.53161 | 0 | 0.9968 | 0.0% |
| dpctgan_eps15_seed2 | 0.535329 | 0.543181 | 0.549243 | 0 | 0.9922 | 0.0% |
| dpctgan_eps1_seed0 | 0.532229 | 0.543909 | 0.553631 | 0 | 0.9979 | 0.0% |
| dpctgan_eps20_seed0 | 0.538456 | 0.546731 | 0.549123 | 0 | 0.9986 | 0.0% |
| dpctgan_eps5_seed0 | 0.554854 | 0.561864 | 0.568413 | 0 | 0.9982 | 0.0% |
| dpctgan_eps8_seed0 | 0.553713 | 0.564963 | 0.573134 | 0 | 0.9984 | 0.0% |
| gaussian_copula_seed0 | 0.438925 | 0.470511 | 0.503704 | 0 | 0.9964 | 0.0% |
| gaussian_copula_seed1 | 0.441434 | 0.470516 | 0.503766 | 0 | 0.9962 | 0.0% |
| gaussian_copula_seed2 | 0.440563 | 0.468803 | 0.503173 | 0 | 0.9962 | 0.0% |
| mst_eps10_seed0 | 0.419071 | 0.423397 | 0.438271 | 0 | 0.9941 | 0.0% |
| mst_eps15_seed0 | 0.417941 | 0.424552 | 0.438451 | 0 | 0.9935 | 0.0% |
| mst_eps15_seed1 | 0.416012 | 0.423189 | 0.437611 | 0 | 0.9938 | 0.0% |
| mst_eps15_seed2 | 0.417569 | 0.425298 | 0.438144 | 0 | 0.9937 | 0.0% |
| mst_eps1_seed0 | 0.420427 | 0.427246 | 0.450365 | 0 | 0.9957 | 0.0% |
| mst_eps20_seed0 | 0.4199 | 0.426456 | 0.438132 | 0 | 0.9951 | 0.0% |
| mst_eps5_seed0 | 0.421921 | 0.424822 | 0.440067 | 0 | 0.9937 | 0.0% |
| mst_eps8_seed0 | 0.417494 | 0.424396 | 0.438155 | 0 | 0.9936 | 0.0% |
| tvae_seed0 | 0.414884 | 0.424317 | 0.439392 | 0 | 0.9934 | 0.0% |
| tvae_seed1 | 0.414547 | 0.424304 | 0.438954 | 0 | 0.993 | 0.0% |
| tvae_seed2 | 0.413931 | 0.424099 | 0.438867 | 0 | 0.9931 | 0.0% |

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
