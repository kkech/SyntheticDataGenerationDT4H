# Privacy Assessment: distance to closest record

Distances are Gower-style mixed-type distances in [0,1] over 61 numeric and 150 categorical columns, computed in sentinel space (a synthetic record is only close to a real one if it matches its values AND its missingness pattern).

**Real-to-real baseline** (leave-one-out): DCR p5 = `0.05968`, median = `0.099393`, NNDR median = `0.9427`.

| synthesizer | DCR min | DCR p5 | DCR median | exact matches | NNDR median | closer than real p5 |
|---|---|---|---|---|---|---|
| ctgan | 0.428468 | 0.451765 | 0.484035 | 0 | 0.997 | 0.0% |
| dpctgan | 0.532562 | 0.544072 | 0.552738 | 0 | 0.9969 | 0.0% |
| gaussian_copula | 0.450915 | 0.475548 | 0.50598 | 0 | 0.9965 | 0.0% |
| mst | 0.414992 | 0.422327 | 0.435109 | 0 | 0.9942 | 0.0% |
| tvae | 0.412759 | 0.423046 | 0.437167 | 0 | 0.9937 | 0.0% |

Reading the table: `closer than real p5` is the share of synthetic records nearer to some real record than the closest 5% of real-to-real neighbor distances -- ~5% is the no-memorization expectation; well above that suggests the model echoes individuals. `exact matches` must be 0 for any release. NNDR near 1 means records sit between real records (population structure), near 0 means they lock onto one real record.

## Limitations
- DCR/NNDR bound record-copying, not membership inference. A proper membership-
  inference evaluation requires a holdout excluded from training; this pipeline
  currently trains on all rows. For DP synthesizers the epsilon guarantee covers
  membership inference by construction; for non-DP synthesizers this is an open
  item for the limitations section.
