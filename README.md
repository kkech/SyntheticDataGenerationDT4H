# SyntheticDataGenerationDT4H

A reproducible pipeline for generating and evaluating synthetic versions
of the DataTools4Heart (DT4H) UC1 heart-failure cohort, with and without
differential privacy, intended to support a public dataset release.

## Pipeline

```
python main.py
```

Thirteen steps, run in order, each skipped once completed (tracked in
`pipeline_status.json`) unless forced. Steps 6-13 are ANALYSIS steps —
they read the generated files and never regenerate anything, so
`python main.py --analysis` reruns all of them cheaply over existing
outputs. The status file reflects the
true lifecycle at all times — every step a run will execute is marked
⏳ pending up front (replacing any stale entry from a previous run),
then 🔄 running, then ✅ completed or ❌ failed — and a step's previous
output files are deleted right before it reruns, so nothing stale ever
sits next to fresh results:

| # | step | what it does | output |
|---|------|--------------|--------|
| 1 | `load_data` | concatenates the transfer's Spark `part-*.parquet` files into the full dataset | `output/load_data/` (local only) |
| 2 | `profile_data` | privacy-safe per-column statistics of the raw data (rare values suppressed), metadata copy, seeded row sample | `output/profile_data/` |
| 3 | `preprocess` | metadata-driven feature engineering; **no imputation anywhere** (see Principles); hard-fails if any null/NaN survives; splits the result 75/25 into **train/holdout** (seeded, manifest committed) | `output/preprocess/` |
| 4 | `profile_preprocessed_data` | same profiler over the preprocessed frame | `output/profile_preprocessed_data/` |
| 5 | `generate` | executes the run plan (seeds × ε × models) on the **train split only**, writes one synthetic CSV per run, saves fitted generators, checks for verbatim training records | `output/generate/` |
| 6 | `evaluate` | marginal fidelity (KS, Wasserstein, TVD, missingness), association structure incl. **fabricated-association counts**, **C2ST full-joint distinguishability**, and **subgroup fidelity** (gender/age strata) — every number read against its own train-vs-holdout noise floor; mean ± sd across seeds and the ε-sweep view | `output/evaluate/` |
| 7 | `coherence` | **row-level clinical coherence audit**: implications mined from the train split, learned category-range consistency (CKD stage vs eGFR, …), survival logic — synthetic violation rates vs the real holdout's own rate; rule set committed | `output/coherence/` |
| 8 | `survival` | **Kaplan-Meier fidelity** for death and HF rehospitalization (nulls = censoring at 5y), log-rank vs holdout, **TOST equivalence tests** (Greenwood SEs, ±5pp margin at 1y/3y/5y — a positive claim, not non-significance), and **effect-estimate replication** (Cox via lifelines, or native logistic fallback) | `output/survival/` |
| 9 | `utility` | Train-Synthetic-Test-Real on the **holdout**, with two model classes (HistGB + logistic), **Brier-score calibration**, **subgroup TSTR** per sex/age stratum, and an **augmentation** arm (real+synthetic vs real alone); family-diversified targets incl. mortality | `output/utility/` |
| 10 | `privacy` | distance-to-closest-training-record per run against the **holdout-to-train baseline**, with committed DCR histograms | `output/privacy/` |
| 11 | `attacks` | **adversarial attacks**: membership inference (distance attack + **learned cv-classifier attack**, AUC + bootstrap CI), **who-is-at-risk profile** (MIA per patient-atypicality quartile), an **empirical DP audit** (attack-derived ε lower bound vs the claimed budget), attribute inference (membership advantage), and anonymeter singling-out/linkability — all calibrated by the holdout | `output/attacks/` |
| 12 | `figures` | publication-quality figures (ε-curves, TSTR gaps, KS profiles vs floor, DCR histograms, KM overlays, coherence, C2ST, MIA) as PNG+PDF, regenerated from the committed step results | `output/figures/` |
| 13 | `release_docs` | **codebook** (per-column semantics incl. what a null means), **Datasheet for the Dataset**, **per-file capability labels** (one JSON per released file aggregating fidelity/coherence/distance/attack/gate evidence), and the exact `pip freeze` of the producing environment | `output/release_docs/` |

### Long runs (survives SSH disconnect)

A full run takes hours; run it as a detached job so closing the terminal
does not kill it:

```
./run_job.sh start --force      # preflight, then detach the full run
./run_job.sh status             # running? + per-step status + latest log lines
./run_job.sh follow             # stream logs.txt live (Ctrl-C detaches, job keeps running)
./run_job.sh stop               # stop without losing completed steps
```

`start` refuses to launch if preflight fails or a job is already running,
archives the previous `logs.txt` into `logs/`, and reports a startup
crash immediately instead of detaching silently. Progress is always
visible in three places: `logs.txt` (timestamped, written live),
`pipeline_status.json` (per-step completion), and each step's files
appearing under `output/`.

### CLI

```
python main.py --preflight          # verify libs, GPU, inputs, disk BEFORE a long run
python main.py --data-dir /path/to/extract [--metadata /path/to/metadata.json]
                                   # point the pipeline at a new site's part-*.parquet
                                   #   extract (Amsterdam and beyond); --metadata only
                                   #   when the JSON is not inside the data dir.
                                   #   Combine with --extended/--force/--preflight.
python main.py --status             # step completion status
python main.py --force              # rerun everything
python main.py --force-step generate --force-step evaluate --force-step privacy
python main.py --only evaluate --force-step evaluate
python main.py --analysis          # rerun ALL analysis steps (6-13) on existing outputs
python backup_results.py           # SNAPSHOT results before any --force rerun: the synthetic
                                   #   CSVs, fitted models and split parquets are gitignored
                                   #   and exist nowhere else (--list / --restore NAME --yes)
python main.py --extended --force  # full campaign PLUS the roadmap runs (quantile-transform
                                   #   variants, TVAE capacity sweep, indicator ablation,
                                   #   AIM 40-column sweep, MST eps=0.5 anchor, the native
                                   #   diffusion baseline `ddpm` x3 seeds, PATE-CTGAN x3 eps)
python regenerate.py --model output/generate/models/<name>.pkl --rows N --out file.csv
python conditional_demo.py --model output/generate/models/tvae_seed0.pkl \
    --rows 500 --condition patient_demographics_gender=female --out sample.csv
python run_one.py --list           # re-run ONE plan entry in place (e.g. after a timeout),
python run_one.py --run-id aim50_eps5_seed0 --timeout 21600   #   merging it into the summary
                                   #   -- unlike `--only generate`, nothing else is deleted
python release_gate.py --file output/generate/DT4H_Synthetic_<run>.csv   # go/no-go before distributing
python release_gate.py --file <...> --policy controlled --note "consortium 2026-08-27"
python respell_released_files.py    # one-time: canonicalize pre-fix CSV spellings on disk
python postprocess_candidate.py --file output/generate/DT4H_Synthetic_<run>.csv \
    [--model output/generate/models/<run>.pkl]   # granularity snap + distance-tail filter
python coherent_sample.py --model output/generate/models/<run>.pkl --rows N \
    --out output/generate/DT4H_Candidate_<run>_coherent.csv   # rule-rejection sampling
```

Post-processing tools write `DT4H_Candidate_*` files by design: the
analysis steps ingest only `DT4H_Synthetic_*`, so a filtered or
rule-cleaned candidate is never silently double-counted as a run.
Re-gate every candidate before distributing it.

All console output (stdout, stderr, warnings) is teed to `logs.txt` with
per-line timestamps.

## The run plan

The generate step executes a plan built in `pipeline/config.py`
(`resolved_run_plan()`), cheapest/most-reliable first so a late failure
or timeout costs only the tail of a run — 31 runs by default:

| model | library | DP | runs | notes |
|------|---------|----|------|-------|
| `gaussian_copula` | SDV | no | 3 seeds | statistical baseline, seconds |
| `tvae` | SDV | no | 3 seeds | the strongest non-DP model here |
| `ctgan` | SDV | no | 3 seeds | the long-standing GAN baseline |
| `dpctgan` | smartnoise-synth | ε | ε ∈ {1,5,8,10,15,20} + 2 extra seeds at ε=15 | DP-GAN comparison point |
| `aim` | smartnoise-synth | ε | ε sweep on the **top-50 outcome-relevant columns** | Private-PGM cannot handle full width (timed out at 6 h on 211 columns); column selection is data-driven and committed (`DT4H_AIM_Column_Selection.json`), with its own 2 h/run timeout |
| `mst` | smartnoise-synth | ε | ε sweep + 2 extra seeds at ε=15 | marginal-based; excellent categorical fidelity; ~2.8 h/run, runs last |

`--extended` appends 21 roadmap runs (52 total, ~52 h): quantile-transform
variants (`tvae_qt` ×3, `ctgan_qt`), a TVAE capacity/epochs sweep
(`tvae_cap256`, `tvae_ep1000`), the sentinel-vs-indicator encoding
ablation (`tvae_ind`), AIM's full ε sweep on a 40-column subset
(`aim40` ×6), an MST ε=0.5 low-budget anchor (own 4 h timeout), the
**native diffusion baseline** `ddpm` ×3 seeds (in-repo Gaussian DDPM,
no new dependencies), **logic-guided diffusion** `ddpm_g` (the mined
coherence rules as a differentiable sampling-time prior, with the
unguided seeds as its exact control), and `patectgan` at ε ∈ {1,5,15}.
Variant runs record qualified model names so grouping never averages
them into the base families.

**DP numeric bounds are declared, not measured.** Every DP run is
bounded by `public_domains.json`: an a-priori, human-**reviewed** public
domain `[lo, hi]` per numeric column. `make_public_domains.py` writes the
template (observed ranges rounded outward to one significant figure,
purely to save typing); a human edits every range for clinical
plausibility and sets `"reviewed": true`, and DP fitting **refuses to
start** until they have. This matters more than it sounds: bounds taken
from the training data — what this pipeline did previously — make the
released mechanism depend on private records through an unnoised
channel, so the ε claim does not hold however small ε is. For
sentinel-encoded columns the lower bound is
`pub_lo − max(0.25·(pub_hi − pub_lo), 1)`, a *pure function of the public
domain* mirroring the data-side sentinel formula; containment of the
training values is asserted at fit time, so a too-narrow declared range
fails in seconds instead of after hours. Bound discovery therefore
spends **zero ε** legitimately — the whole budget goes to synthesis at
every ε, which is what makes the ε=1 runs possible — and every run
records its **(ε, δ)** together with the SHA-256 of the domain file that
bounded it. Two residual leaks are disclosed rather than hidden:
snsynth's `LabelTransformer` still learns **categorical vocabularies**
from the training data at zero ε (standard practice, but a real
disclosure), and **AIM's column selection** is computed on the train
split without noise, so width-limited runs are DP *given* that column
set.

Every run records full provenance: its own seed, library versions, git
commit, hardware, and the SHA-256 of the exact training file. Expect
the full plan to take **roughly 30–40 hours**; partial results are
written after every run, so an interrupted plan keeps everything
already finished.

## Principles

- **Missingness is information, never noise.** Numeric nulls are not
  imputed: time-to-event nulls mean *the event never happened*, lab
  nulls mean *not measured* (and clinicians measure selectively). Both
  are encoded as per-column sentinels below the observed range, modelled
  jointly by the synthesizer, and decoded back to null in the output —
  so synthetic patients have realistic missingness patterns. Categorical
  nulls become an explicit `Missing` category.
- **Nothing is fabricated.** No placeholder values, no bootstrap fills.
  The evaluation proves preprocessing is distribution-preserving
  (original vs preprocessed: KS = 0, TVD = 0 on every common column).
- **Every claim is calibrated.** Generators only ever see the train
  split; fidelity is read against the train-vs-holdout sampling-noise
  floor, utility is tested on the holdout, privacy against the
  holdout's own distance distribution, and headline numbers carry
  mean ± sd across seeds instead of a single draw.
- **The pipeline is the only path that produces files.** Reports verify
  files as they are on disk and flag stale ones loudly; nothing is ever
  silently repaired.
- **Reports cannot look better than the data.** Leakage is checked in
  sentinel space before decoding; every claim in a summary is computed,
  not assumed.

## Data & privacy boundaries

Committed to git: statistics, summaries, metadata, small seeded samples,
evaluation and privacy reports (aggregates only). **Never committed**
(`.gitignore`): the full patient-level parquets, synthetic CSVs pending
review, and fitted generator pickles (a trained generator memorizes
aspects of the real data and is treated as sensitively as the data
itself).

The privacy step's DCR/NNDR analysis bounds record-copying against a
genuine unseen-data baseline (the holdout split, which no generator ever
saw). The attacks step then attacks rather than asserts: distance and
learned membership inference, attribute inference, anonymeter
singling-out/linkability, a per-patient-atypicality risk profile, and
an empirical ε lower bound that a DP run's claimed budget must exceed —
the formal guarantee is audited, not merely stated. The release gate
(`release_gate.py`) is the per-file go/no-go over all of that evidence.

Four of the gate's six checks are absolute facts (schema,
representation, freshness, verbatim leakage). The other two are
thresholded by a named **policy**, because "how much clinical
incoherence is acceptable" and "how close to a training record is too
close" are governance decisions, not measurements:

| policy | coherence limit | distance limit | for |
|---|---|---|---|
| `release` (default) | 10× the holdout's own violation rate | 2× the natural 5% share | open or brokered release |
| `controlled` | 20× that rate | 2× — **unchanged** | controlled-access sharing under a DUA |

The two limits are deliberately independent: `controlled` accepts lower
clinical coherence **without touching the memorization margin**, which
is the privacy-protective check. Every report states the verdict under
*every* policy and writes a JSON sidecar naming the one used, so a
relaxed pass can never be read as an open-release pass — the capability
labels carry `release_gate_policy` and `cleared_for_open_release` for
the same reason. `--coherence-multiplier` / `--distance-multiplier`
override the thresholds ad hoc and are recorded as a custom policy;
`--note` records who authorized it.

## Setup

Python 3.10+, a CUDA GPU recommended for the GAN/VAE models.

```
python -m venv .synthenv && source .synthenv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install lifelines anonymeter    # optional evaluators
pip install numpy==2.2.6            # AFTER anonymeter -- restores numpy 2
python main.py --preflight          # must show: ✅ import mbi (MST/AIM backend)
```

**One numpy for everything:** anonymeter declares `numpy<1.27` and its
install downgrades numpy — which silently breaks MST/AIM generation
(their backend needs numpy ≥ 2 via jax). The pin is conservative
packaging only: anonymeter runs correctly under numpy 2.2.6 (verified
end to end; ignore pip's resolver warning). Reinstall numpy 2.2.6 after
anonymeter, and always preflight before a long run — the `mbi` check
exists precisely to catch this before hour one instead of hour ten. See
the caution block in `requirements.txt`.
