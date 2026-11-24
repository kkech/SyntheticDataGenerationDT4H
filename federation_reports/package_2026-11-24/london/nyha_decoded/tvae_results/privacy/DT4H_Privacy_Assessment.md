# Privacy Assessment: distance to closest training record

Distances are Gower-style mixed-type distances in [0,1] over 65 numeric and 155 categorical columns, computed in sentinel space (a synthetic record is only close to a real one if it matches its values AND its missingness pattern). The baseline is the HOLDOUT distribution: real patients the generators never saw, measured against the training records -- exactly what an innocent 'new' record's distance profile looks like.

**Holdout-to-train baseline**: DCR p5 = `0.048053`, median = `0.090306`, NNDR median = `0.9441`.

| run | DCR min | DCR p5 | DCR median | exact matches | NNDR median | closer than holdout p5 |
|---|---|---|---|---|---|---|
| tvae_seed0 🚨 | 0.024079 | 0.034234 | 0.074748 | 0 | 0.9289 | 20.2% |

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
