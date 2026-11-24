# CardioSynth: structured synthetic data evaluation results

Evaluation reports from the structured synthetic-data pipeline (`SyntheticDataGenerationTabularDT4H`),
one folder per centre, as produced at each site. Prepared 11 November 2026; Brno, Rome and Amsterdam added, Brno, Bucharest, Rome, Utrecht, Stockholm and London gated, Anonymeter added at Brno and Bucharest, 23 November 2026; Amsterdam v2.2 gated, 24 November 2026.

**No patient records are included.** The package holds the pipeline's reports, metrics, figures and
documentation only. It leaves out the real data, the synthetic tables, the trained models and the
holdout row positions. The column profiles (`profile_data/`) are aggregates: counts below the site's
display threshold and identifier columns are suppressed, and numeric columns show min/max, mean/std
and quantiles.

## Centres

| folder | centre | patients | train / holdout | runs | figures |
|---|---|---|---|---|---|
| `brno/all_runs/` | St. Anne's University Hospital, Brno (CZ) | 2,065 | 1,549 / 516 | 45 (full model comparison, incl. MST ε=15) | yes |
| `rome/all_runs/` | Fondazione Policlinico A. Gemelli IRCCS, Rome (IT) | 5,031 | 3,773 / 1,258 | 43 evaluated (completed 23 Nov 2026, see Notes) | yes |
| `amsterdam/v22_runs/` | Amsterdam UMC (NL) | 4,445 | 3,334 / 1,111 | 47 (**v2.2 run**, the files the notes were generated from) | yes |
| `amsterdam/all_runs/` | Amsterdam UMC (NL), superseded | 4,694 (evaluated cohort) | 3,520 / 1,174 | 42 evaluated (pre-v2.2 run, see Notes) | yes |
| `bucharest/all_runs/` | Bucharest Emergency Clinical Hospital (RO) | 6,274 | 4,706 / 1,568 | 32 (full model comparison, incl. MST ε=15) | yes |
| `utrecht/{mst,tvae}_results/` | UMC Utrecht (NL) | 2,482 | 1,862 / 620 | MST ε=15, TVAE | no (not in the site's export) |
| `stockholm/{mst,tvae}_results/` | Region Stockholm (SE) | 802 | 602 / 200 | MST ε=15, TVAE | yes |
| `london/{mst,tvae}_results/` | UCL Hospital, London (UK) | 1,005 | 754 / 251 | MST ε=15, TVAE | yes |

Each generator is trained on 75% of the patients and produces as many synthetic records. The 25%
holdout is never shown to a generator and serves as the real-data reference for fidelity and
privacy.

## Headline metrics: `structured_metrics.csv`

One row per run (215 in all; Amsterdam has both its v2.2 and its superseded pre-v2.2 runs, told apart by `report_folder`). The columns are:

- **Fidelity (train vs synthetic):** `ks_mean` (numeric marginals), `tvd_mean` (categorical
  marginals), `wasserstein_std_mean`, `c2st_auc` (classifier two-sample test; 0.5 means
  indistinguishable).
- **Clinical coherence:** `rows_violating` is the share of synthetic rows breaking at least one
  clinical-logic rule. `rows_violating_real_holdout` is the same rate on real unseen patients, for
  reference.
- **Privacy distance:** `closer_than_holdout_p5` is the share of synthetic records closer to a training
  patient than the closest 5% of holdout patients are (about 5% expected). `exact_matches` counts
  synthetic records identical to a training row.
- **Privacy attacks:** `mia_auc` and `mia_learned_auc` are membership-inference attack AUCs (0.5 is
  chance). `empirical_epsilon` is a lower bound on ε. `singling_out` and `linkability` are Anonymeter
  risks (empty where Anonymeter was not run). 
- **Freshness:** `stale_cells` counts cells the current decoder would change (see Notes).

## What each report folder contains

| subfolder | content |
|---|---|
| `evaluate/` | fidelity: KS, TVD, Wasserstein/std, associations (incl. fabricated pairs), subgroups, C2ST |
| `coherence/` | clinical-logic rule audit and the rule set |
| `utility/` | TSTR (train on synthetic, test on real) AUC gaps |
| `privacy/` | DCR / NNDR distances against the holdout baseline |
| `attacks/` | membership inference, attribute inference, Anonymeter |
| `survival/` | Kaplan–Meier / hazard fidelity for the available endpoints |
| `generate/`, `preprocess/`, `profile_*` | run plan, preprocessing log, column profiles of real and preprocessed data |
| `release_docs/` | datasheet, codebook, capability labels, environment freeze |
| `figures/` | the pipeline's standard figures (PNG and PDF) |

Each `*.md` is the readable version of the `*.json` next to it.

## Notes

- **Release gate.** Gate reports (`<site>/all_runs/generate/DT4H_Release_Gate_*.{md,json}`, verdicts
  under both policies) and labels carrying the verdict exist for:
  - **Brno**: run 23 Nov 2026, 45 files; 3 pass `release`, 20 pass `controlled`. `mst_eps15_seed0` and
    `tvae_seed0` both FAIL release on coherence (1.22% and 1.05% of rule checks, against a limit of
    1.02%) and PASS controlled.
  - **Bucharest**: run 23 Nov 2026, 32 files; 10 pass `release`, 19 pass `controlled`.
    `mst_eps15_seed0` FAILs release on coherence (1.75% against 1.00%) and PASSes controlled.
    `tvae_seed0` PASSes both.
  - **Amsterdam (v2.2)**: run 24 Nov 2026 on site, 47 files; 2 pass `release`, 8 pass `controlled`.
    `mst_eps15_seed0` FAILs both policies on coherence (4.19% against limits of 1.04% and 3.04%) and
    distance (22.1%). `tvae_seed0` FAILs both on distance (12.2%, limit 10%); its coherence (2.12%)
    fails release only. The earlier gate of the pre-v2.2 files (27 Aug, `amsterdam/all_runs/`) is
    superseded.

  - **Stockholm**: run 23 Nov 2026. `mst_eps15_seed0` FAILs both policies on coherence (3.84% against
    limits of 1.01% and 3.01%). `tvae_seed0` FAILs release on coherence (1.030% against 1.013%) and
    PASSes controlled.
  - **London**: run 23 Nov 2026 on the files with the two NYHA columns decoded (`london/nyha_decoded/`).
    `mst_eps15_seed0` FAILs release on coherence (2.88% against 1.03%) and PASSes controlled.
    `tvae_seed0` FAILs both on distance (20.2% of records closer than the holdout p5, limit 10%); its
    coherence (0.63%) passes. The gate on the files as first produced (`london/*_results/`) also failed
    freshness, which the decoding fixes.

  - **Rome**: run 23 Nov 2026, 43 files; 11 pass `release`, 24 pass `controlled`. `mst_eps15_seed0` and
    `tvae_seed0` both FAIL release on coherence (1.04% and 1.12%, against a limit of 1.00%) and PASS
    controlled (distance 0% and 3.6%).

  - **Utrecht**: run 23 Nov 2026 on site. Both files FAIL release on coherence (MST 2.52%, TVAE 1.43%,
    against a limit of 1.02%) and PASS controlled (distance 0% and 6.1%).

  Stockholm and London were gated with the pipeline's `deploy` code (`a297d8d`), with no change to
  the synthetic files beyond the gate reports and labels. The gate logs are
  `<site>/release_gate_*2026112[34].log`.
- **Bucharest:** the reports come from the 24 Sep 2026 fold-in rerun, which added `mst_eps15_seed0` to
  the 31-run campaign (`fold_in_mst.log`). On the Bucharest VM these files are uncommitted on
  `logs/buch`; `origin/logs/buch` still holds the 31-run version without MST. Survival was skipped
  there (no endpoint column).
- **Utrecht:** the site's export holds the evaluation reports but no preprocessing summary, column
  profiles or figures.
- **Survival endpoints:** all-cause death and HF rehospitalisation at Utrecht, HF rehospitalisation
  only at Stockholm, all-cause death only at London.
- **Anonymeter at Brno and Bucharest was added on 23 Nov 2026.** The first attacks reports there lacked it
  (the package was not installed), so we installed Anonymeter 1.1.0 and reran only the attacks step,
  then refreshed the labels. All 77 runs now have singling-out and linkability risks (CardioSynth files:
  Brno MST 0.000 / 0.000, TVAE 0.007 / 0.000; Bucharest MST 0.005 / 0.000, TVAE 0.000 / 0.000). The
  membership and attribute attacks are unchanged. All 215 runs in this package have Anonymeter results.
- **Stale files.** Both London files (MST ε=15 and TVAE) were flagged stale, with 1,508 undecoded cells
  each: the two NYHA columns, which hold class 0 ("not assessed") in all 754 records because NYHA was
  recorded for no London patient. Decoding them to missing with the pipeline's own decoder leaves 0
  stale cells and changes nothing else; `london/nyha_decoded/` holds the privacy and gate reports of the
  decoded files. The evaluation in `london/*_results/` scored the files as first produced. Bucharest's
  `tvae_ind_seed0` (not a CardioSynth file) has 4 stale cells.
- **London TVAE:** 20.2% of records sit closer to a training patient than the holdout's 5th percentile.
  The membership attacks are at chance (AUC 0.51–0.52).
- **Brno:** the reports are the 45-run campaign in `kon/output` on the ICRC server (attacks rerun with Anonymeter on 23 Nov 2026).
  Survival covers all-cause death only.
- **Rome was completed on 23 Nov 2026.** Its 17 Sep run had stopped after coherence: survival failed
  on a missing all-cause death column, and utility, privacy, attacks, figures and release docs never ran.
  We added the pipeline's survival fix (`a297d8d`, which skips effect replication when that column is
  absent) and ran only the missing steps. Generation, evaluation and coherence are unchanged. Anonymeter
  ran here, so Rome has singling-out and linkability results, as do Brno and Bucharest since their 23 Nov rerun.
- **Amsterdam v2.2** (`amsterdam/v22_runs/`): the run in `generationV2/dt4h-v22-probe` on the CDM v2.2
  extract (4,445 patients, 3,334 / 1,111), whose 3,334-record synthetic files the notes were generated
  from. All steps ran on 18 Sep 2026 (code `claude/bounded-improvements` @ `f77bc78`, not `deploy`), with
  Anonymeter for all 47 runs. The release gate and the label refresh ran on 24 Nov 2026 on a copy of that
  folder on the same workspace, with the same code. Survival, utility and the rest are as computed on
  18 Sep.
- **Amsterdam pre-v2.2** (`amsterdam/all_runs/`, superseded): from
  github.com/kkech/SyntheticDataGenerationDT4H (branch `claude/bounded-improvements`, 14 Sep 2026),
  computed on the earlier cohort (3,520 / 1,174), except `preprocess/` and `profile_*`, which were rerun
  on the v2.2 extract. Kept for the record.
- **Barcelona** has not started.
