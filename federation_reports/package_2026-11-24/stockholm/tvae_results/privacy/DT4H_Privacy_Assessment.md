# Privacy Assessment: distance to closest training record

Distances are Gower-style mixed-type distances in [0,1] over 62 numeric and 120 categorical columns, computed in sentinel space (a synthetic record is only close to a real one if it matches its values AND its missingness pattern). The baseline is the HOLDOUT distribution: real patients the generators never saw, measured against the training records -- exactly what an innocent 'new' record's distance profile looks like.

**Holdout-to-train baseline**: DCR p5 = `0.047177`, median = `0.104689`, NNDR median = `0.9299`.

| run | DCR min | DCR p5 | DCR median | exact matches | NNDR median | closer than holdout p5 |
|---|---|---|---|---|---|---|
| tvae_seed0 | 0.032293 | 0.04914 | 0.086454 | 0 | 0.9357 | 4.3% |

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
