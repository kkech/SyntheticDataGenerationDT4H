# Federation evaluation reports

Aggregate evaluation reports from every DT4H site that has run the pipeline,
kept here so the multi-site numbers in `paper/main.tex` can be traced to their
source. **No patient records, no synthetic tables, no trained models**: reports,
metrics, figures, gate reports and logs only.

## Contents

| path | what it is |
|---|---|
| `package_2026-11-24/` | The partner's package exactly as received on 24 Nov 2026 (`CardioSynth_structured_evaluation_2026-11-11-3.zip`, MD5 `0bda40e650c98d02d24e96b7d81a0b31`). Its own `README.md` describes each folder and how each site's reports were produced. |
| `package_2026-11-24/structured_metrics.csv` | The partner's headline table: one row per run, 215 rows. |
| `gate_verdicts.csv` | Derived here from every `DT4H_Release_Gate_*.json` in the package: one row per gated file, with the verdict under both policies, the failed checks, the coherence rate and limits, and the distance share. 218 rows. |

## Which folders are current

| site | current folder | runs | note |
|---|---|---|---|
| Amsterdam | `amsterdam/v22_runs/` | 47 | CDM v2.2 extract, 4,445 patients; gated 24 Nov 2026 |
| Amsterdam (superseded) | `amsterdam/all_runs/` | 42 | earlier extract, 4,694 patients; the campaign the manuscript's Sections 4.1 to 4.9 still report |
| Bucharest | `bucharest/all_runs/` | 32 | |
| Rome | `rome/all_runs/` | 43 | |
| Brno | `brno/all_runs/` | 45 | |
| Utrecht | `utrecht/{mst,tvae}_results/` | 2 | |
| London | `london/{mst,tvae}_results/` | 2 | gate verdicts of record are in `london/nyha_decoded/` (NYHA columns decoded to missing) |
| Stockholm | `stockholm/{mst,tvae}_results/` | 2 | |

Counting current folders only: 173 gated files, 26 pass open release, 75 pass
controlled access.

## Not in this package

1. Amsterdam: the distance-filtered TVAE seed-0 candidate, its row map and its
   gate report (to be produced on the Amsterdam VM with
   `postprocess_candidate.py --no-snap`, then `release_gate.py --file`).
2. Every site's `public_domains.json`. Four different declarations were used
   (hash prefixes `756745e7` Amsterdam v2.2, `7b486344` Brno and Rome,
   `283379d2` Bucharest, `111791df` Utrecht, London and Stockholm).
3. Utrecht: preprocessing summary, numeric-missing encoding map, column
   profiles and figures (left out of the site's export).
4. Amsterdam v2.2: preprocessing summary and encoding map (the v2.2
   preprocessing summary is in this repository under `output/preprocess/`).
5. Survival results at Rome (events exist in the extract but do not reach the
   survival step) and Bucharest (almost no events in the extract).
6. Attack results for any post-processed candidate: the attacks step covers
   `DT4H_Synthetic_*.csv` only.
7. Barcelona: not started.
