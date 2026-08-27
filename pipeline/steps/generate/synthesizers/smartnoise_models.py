"""
Differentially private synthesizers from OpenDP smartnoise-synth.

Two families, and the choice between them matters:

  * Marginal-based (AIM, MST, PATE-CTGAN's cousins): measure noisy
    low-dimensional marginals under DP and fit a graphical model to them.
    Utility-oriented benchmarks consistently report these OUTPERFORMING
    DP-GANs on tabular data -- at moderate epsilon, AIM has been reported
    close to real-data utility. AIM is workload-aware and generally beats
    MST, especially at higher epsilon. These are CPU-only: there is no
    GPU training to speed up.

  * DP-GAN (DPCTGAN): the deep-learning route. Included for comparison
    with the project's original approach, but the literature suggests it
    is the weaker choice for tabular data.

SCALING CAVEAT: AIM and MST both build on Private-PGM, which is documented
to struggle as the column count grows, in both fitting and sampling. This
dataset has ~329 columns, which is squarely in the risky range. Use
`max_columns` to trial a subset before committing to a full run, and fall
back to MST (cheaper, pairwise-tree-based) if AIM will not fit in memory.

WHERE THE NUMERIC BOUNDS COME FROM (this is the privacy-critical part)
----------------------------------------------------------------------
snsynth needs a [lower, upper] domain per continuous column. Preprocessing
that domain out of the training data -- what this module used to do, with
`preprocessor_eps=0.0` and lo/hi taken from `df.min()/df.max()` -- is a
formal DP violation: the released mechanism then depends on private
records through an unnoised channel, so no epsilon claim holds. It is
also the classic silent one: nothing crashes, the summary still prints an
epsilon, and the guarantee is simply not there.

The bounds are now A PRIORI PUBLIC and human-reviewed. `public_domains.json`
(template written by make_public_domains.py, reviewed and signed off by a
human setting `reviewed: true`) declares [lo, hi] per numeric column. DP
fitting REFUSES to start if that file is missing or unreviewed -- the guard
exists so a formally-invalid campaign cannot be launched by accident.

Sentinel-encoded columns need one extra step. Preprocessing replaced every
numeric null with a per-column sentinel placed BELOW the observed range:

    sentinel = min_observed - max(0.25 * (max_observed - min_observed), 1.0)

so the training frame contains values below the public `lo`. The bound
passed to snsynth for such a column is therefore

    [pub_lo - max(0.25 * (pub_hi - pub_lo), 1.0),  pub_hi]

which is a PURE FUNCTION OF THE PUBLIC DOMAIN -- it mirrors the data-side
formula with public numbers substituted for private ones. Because a
reviewed public domain must contain the observed range (pub_lo <=
min_observed, pub_hi >= max_observed), this public sentinel is always <=
the data sentinel, so every training value lands inside the public bound.
That containment is asserted at fit time against the actual frame, so a
too-narrow reviewed range fails in seconds instead of after hours of
fitting. The check reads private data, but only to abort: it releases
nothing and produces no output when it fires. Columns without a sentinel
entry simply get [pub_lo, pub_hi].

REMAINING CAVEATS, to be disclosed rather than papered over:

  * CATEGORICAL VOCABULARIES ARE STILL PRIVATE. snsynth's LabelTransformer
    learns each categorical column's category list from the training data,
    at zero epsilon. This is standard practice in the DP-synthesis
    literature and in snsynth's own defaults, but it is a real leak: the
    presence of a rare category is disclosed by the fact that the model
    can emit it. Declaring public vocabularies would close it; until then
    it belongs in the release documentation.
  * THE TRAINING VALUES ARE DATA-DERIVED, and that is fine. The sentinel
    encoding is a transform of private values, not a released statistic;
    DP requires the BOUNDS (the mechanism's parameters) to be public, not
    the inputs.
  * AIM's column selection (see column_selection.py) is computed on the
    train split without noise, so width-limited AIM runs carry a
    data-dependent choice of which columns exist at all.
"""

import hashlib
import json
import math
import os

import pandas as pd

from pipeline.steps.generate.synthesizers.base import Synthesizer

#: Same gap fraction the preprocessing sentinel used
#: (preprocess.transforms.SENTINEL_GAP_FRACTION). Duplicated as a literal
#: rather than imported so the DP-side formula cannot silently change
#: meaning if preprocessing is retuned: if these two ever diverge, the
#: containment assert below fires loudly.
SENTINEL_GAP_FRACTION = 0.25
SENTINEL_GAP_FLOOR = 1.0

_REVIEW_INSTRUCTIONS = (
    "DP runs require a reviewed public domain file; run make_public_domains.py "
    "and review it (edit every range for clinical plausibility, then set "
    "\"reviewed\": true). Data-derived bounds are a formal DP violation, so this "
    "is a hard stop, not a warning."
)


def sentinel_public_bound(pub_lo: float, pub_hi: float) -> float:
    """The public lower bound for a sentinel-encoded column.

    A pure function of the declared public domain, mirroring the data-side
    sentinel formula. Since the reviewed domain contains the observed
    range, this is always at or below the actual sentinel in the data.
    """
    return pub_lo - max((pub_hi - pub_lo) * SENTINEL_GAP_FRACTION, SENTINEL_GAP_FLOOR)


def file_sha256(path: str) -> str | None:
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_public_domains(path: str) -> tuple[dict, str]:
    """
    Load the reviewed public numeric-domain declaration.

    Raises rather than falling back: every fallback here is a silently
    invalid privacy claim. Returns ({column: {"lo", "hi"}}, sha256) so
    the run summary can record exactly which revision of the declaration
    a released file was produced under.
    """
    if not path or not os.path.exists(path):
        raise FileNotFoundError(
            f"No public domain file at {path!r}. {_REVIEW_INSTRUCTIONS}")
    with open(path) as f:
        doc = json.load(f)
    if doc.get("reviewed") is not True:
        raise ValueError(
            f"{path} is present but not marked reviewed "
            f"(reviewed={doc.get('reviewed')!r}). {_REVIEW_INSTRUCTIONS}")
    domains = doc.get("domains") or {}
    if not domains:
        raise ValueError(f"{path} is marked reviewed but declares no domains.")
    return domains, file_sha256(path)


class _SmartNoiseBase(Synthesizer):
    """Shared fit/sample for smartnoise-synth's Synthesizer.create API."""

    algorithm: str
    is_dp = True
    #: Which TableTransformer style the algorithm consumes: 'cube'
    #: (binned, for the marginal methods) or 'gan' (min-max scaled).
    transform_style = "cube"

    def fit(self, df, categorical_columns, continuous_columns) -> None:
        from snsynth import Synthesizer as SNSynthesizer

        epsilon = self.params.get("epsilon")
        if epsilon is None:
            raise ValueError(
                f"Synthesizer '{self.name}' is differentially private and requires an "
                f"explicit epsilon. There is no default: a DP run whose budget came "
                f"from a fallback cannot be reported honestly.")
        epsilon = float(epsilon)

        # (epsilon, delta) is the guarantee, so delta is chosen and
        # recorded rather than left to a library default. 1/(n*sqrt(n))
        # is the value snsynth's DP-GANs compute internally at fit time,
        # so using it for the marginal methods too keeps one convention
        # across the DP family. Comfortably below 1/n, the usual bar.
        n_rows = int(len(df))
        delta = self.params.get("delta")
        delta = float(delta) if delta is not None else 1.0 / (n_rows * math.sqrt(n_rows))
        self._delta = delta
        self._epsilon = epsilon

        create_kwargs = {"epsilon": epsilon, "delta": delta}
        # Only the GAN-based synthesizers take epochs/batch_size; passing
        # them to AIM/MST raises.
        if self.params.get("pass_training_params", False):
            create_kwargs["epochs"] = self.params.get("epochs", 300)
            create_kwargs["batch_size"] = self.params.get("batch_size", 50)

        domains, domains_sha = load_public_domains(self.params.get("public_domains_path"))
        encoding = self._load_encoding(self.params.get("numeric_encoding_path"))
        self._public_domains_sha256 = domains_sha
        constraints = self._bound_constraints(df, continuous_columns, domains, encoding)
        n_sentinel = sum(1 for c in continuous_columns if c in encoding)
        print(f"  Bounds for {len(constraints)} continuous column(s) taken from the "
              f"REVIEWED public domain declaration (sha256 {str(domains_sha)[:12]}...); "
              f"{n_sentinel} of them widened to the public sentinel bound. No epsilon "
              f"is spent on bound discovery: the bounds are not a function of this data.")
        print(f"  Guarantee recorded as (ε={epsilon:g}, δ={delta:.3g}).")

        create_kwargs.update(
            self._algorithm_kwargs(len(categorical_columns) + len(continuous_columns)))

        # Not every snsynth algorithm's constructor accepts `delta` (the
        # DP-GANs derive it from n at fit time). Pass it where it is
        # accepted, and where it is not, keep the computed value for the
        # record -- it is the same formula the library applies -- rather
        # than dropping delta from the provenance.
        try:
            self._model = SNSynthesizer.create(self.algorithm, **create_kwargs)
            self._delta_passed_to_library = True
        except TypeError as e:
            if "delta" not in str(e):
                raise
            create_kwargs.pop("delta")
            self._model = SNSynthesizer.create(self.algorithm, **create_kwargs)
            self._delta_passed_to_library = False
            print(f"  Note: {self.algorithm} does not accept a delta kwarg; it computes "
                  f"delta=1/(n*sqrt(n)) internally. Recorded value: {delta:.3g}.")

        self._model.fit(
            df,
            transformer=constraints,
            categorical_columns=categorical_columns,
            continuous_columns=continuous_columns,
            preprocessor_eps=0.0,  # nothing left to discover: bounds are public
            nullable=False,  # preprocessing guarantees no nulls/NaN remain
        )

    def _algorithm_kwargs(self, n_columns: int) -> dict:
        """Extra constructor kwargs for the underlying algorithm."""
        return {}

    @staticmethod
    def _load_encoding(path: str | None) -> dict:
        """The committed sentinel map, used only to know WHICH columns
        were sentinel-encoded. Its numbers are not used as bounds."""
        if not path or not os.path.exists(path):
            return {}
        with open(path) as f:
            return json.load(f)

    def _bound_constraints(self, df, continuous_columns, domains, encoding) -> dict:
        """
        Per-column snsynth transformers bounded by the PUBLIC domain.

        See the module docstring for why the bounds may not come from
        `df`. `df` is touched here only for the containment guard, which
        aborts the run and releases nothing.
        """
        from snsynth.transform import BinTransformer, MinMaxTransformer

        out = {}
        missing, violations, degenerate = [], [], []
        for c in continuous_columns:
            spec = domains.get(c)
            if spec is None:
                missing.append(c)
                continue
            pub_lo, pub_hi = float(spec["lo"]), float(spec["hi"])
            if pub_hi <= pub_lo:
                degenerate.append(f"{c} [{pub_lo}, {pub_hi}]")
                continue
            lower = sentinel_public_bound(pub_lo, pub_hi) if c in encoding else pub_lo

            col_min, col_max = float(df[c].min()), float(df[c].max())
            if col_min < lower or col_max > pub_hi:
                violations.append(
                    f"{c}: training values span [{col_min:g}, {col_max:g}] but the "
                    f"public bound is [{lower:g}, {pub_hi:g}]")
                continue

            out[c] = (BinTransformer(lower=lower, upper=pub_hi)
                      if self.transform_style == "cube"
                      else MinMaxTransformer(lower=lower, upper=pub_hi))

        if missing:
            raise ValueError(
                f"{len(missing)} continuous column(s) have no entry in the reviewed "
                f"public domain file: {missing[:10]}"
                + (" ..." if len(missing) > 10 else "")
                + ". Re-run make_public_domains.py (it enumerates every modelled "
                  "numeric column) and review the added ranges.")
        if degenerate:
            raise ValueError(
                f"Public domain(s) with hi <= lo: {degenerate[:10]}. Fix the reviewed "
                f"file: a domain must be a real interval.")
        if violations:
            # Fail BEFORE fitting. A too-narrow reviewed range is the one
            # mistake that would otherwise surface hours later as clipped
            # values, and clipping silently reintroduces a data-dependent
            # bound.
            raise ValueError(
                "Training values fall outside the reviewed public domain -- the "
                "declared domain must contain the data it bounds (widen it in "
                f"{self.params.get('public_domains_path')}):\n  "
                + "\n  ".join(violations[:20])
                + (f"\n  ... and {len(violations) - 20} more" if len(violations) > 20 else ""))
        return out

    def sample(self, n_rows: int) -> pd.DataFrame:
        return self._model.sample(n_rows)

    def describe(self) -> dict:
        """Full (epsilon, delta) provenance for the run summary, including
        which revision of the public domain declaration bounded the run."""
        d = super().describe()
        sha = getattr(self, "_public_domains_sha256", None)
        d.update({
            "epsilon": getattr(self, "_epsilon", self.params.get("epsilon")),
            "delta": getattr(self, "_delta", None),
            "delta_passed_to_library": getattr(self, "_delta_passed_to_library", None),
            "bounds_source": (f"public_domains.json rev {sha}" if sha
                              else "public_domains.json (not yet loaded)"),
            "public_domains_sha256": sha,
            "privacy_caveats": [
                "categorical vocabularies are learned from the training data by "
                "snsynth's LabelTransformer at zero epsilon (standard practice, "
                "disclosed leak)",
                "numeric bounds are public and human-reviewed; sentinel bounds are a "
                "pure function of the public domain",
            ],
        })
        return d


class AIMSynthesizer(_SmartNoiseBase):
    """Marginal-based, workload-aware. Current recommendation for DP
    tabular synthesis -- but watch the Private-PGM scaling caveat above."""

    name = "aim"
    algorithm = "aim"
    uses_gpu = False

    def _algorithm_kwargs(self, n_columns: int) -> dict:
        """Cap AIM's measurement rounds. The library default is 16 x
        columns (800 at 50 columns), and runtime scales with rounds --
        measured: the default timed out even at 15 columns at high
        epsilon, because unlike low-epsilon runs the budget never runs
        out early. Capping rounds keeps the epsilon guarantee fully
        intact (the budget is split across fewer, less-noisy
        measurements); it is an accuracy/runtime hyperparameter, and the
        value used is recorded in the run summary. Override with
        synthesizer_params['aim']['rounds']."""
        rounds = int(self.params.get("rounds") or max(3 * n_columns, 30))
        print(f"  AIM rounds capped at {rounds} (library default would be {16 * n_columns}).")
        return {"rounds": rounds}


class MSTSynthesizer(_SmartNoiseBase):
    """Marginal-based over a maximum spanning tree of pairwise
    correlations. Cheaper than AIM; the fallback if AIM will not scale."""

    name = "mst"
    algorithm = "mst"
    uses_gpu = False


class PATECTGANSynthesizer(_SmartNoiseBase):
    """DP-GAN variant using the PATE framework."""

    name = "patectgan"
    algorithm = "patectgan"
    uses_gpu = True
    transform_style = "gan"

    def __init__(self, **params):
        params.setdefault("pass_training_params", True)
        super().__init__(**params)


class DPCTGANSynthesizer(_SmartNoiseBase):
    """
    The project's original DP approach. Kept for comparison, but note the
    benchmark evidence favours AIM/MST for tabular data -- treat this as
    the baseline being measured against, not the default.
    """

    name = "dpctgan"
    algorithm = "dpctgan"
    uses_gpu = True
    transform_style = "gan"

    def __init__(self, **params):
        params.setdefault("pass_training_params", True)
        super().__init__(**params)
