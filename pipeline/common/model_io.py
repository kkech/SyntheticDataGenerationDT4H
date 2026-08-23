"""
Loading saved generators across a numpy version change.

The generation campaign pickled the fitted models under numpy 2.x;
installing anonymeter later downgraded the environment to numpy 1.26
(its numba dependency requires it), and numpy changed how BitGenerator
random states are pickled between those versions: 2.x stores the
BitGenerator CLASS where 1.26's reconstructor expects its NAME, so a
plain pickle.load dies with "MT19937 is not a known BitGenerator".

load_generator() applies a narrow, reversible shim that accepts either
form, restoring cross-version loads without touching the environment.
The models themselves are unaffected -- only the RNG-state header
format changed.
"""

import contextlib
import importlib
import pickle
import sys


def _install_numpy_module_aliases() -> None:
    """numpy 2.x renamed numpy.core -> numpy._core; pickles created
    under 2.x reference the new names, which do not exist on 1.26.
    Alias them (idempotent, additive -- existing modules untouched)."""
    import numpy as np

    if int(np.__version__.split(".")[0]) >= 2:
        return  # numpy 2.x ships its own numpy.core compatibility shim
    import numpy.core as _core

    sys.modules.setdefault("numpy._core", _core)
    for sub in ("numeric", "multiarray", "umath", "_multiarray_umath",
                "fromnumeric", "numerictypes", "_dtype", "overrides"):
        try:
            sys.modules.setdefault(f"numpy._core.{sub}",
                                   importlib.import_module(f"numpy.core.{sub}"))
        except ImportError:
            pass


@contextlib.contextmanager
def _bit_generator_compat():
    try:
        import numpy.random._pickle as np_pickle
    except ImportError:
        yield
        return
    orig = np_pickle.__bit_generator_ctor

    def compat_ctor(bit_generator="MT19937"):
        if isinstance(bit_generator, type):
            bit_generator = bit_generator.__name__
        return orig(bit_generator)

    np_pickle.__bit_generator_ctor = compat_ctor
    try:
        yield
    finally:
        np_pickle.__bit_generator_ctor = orig


def save_environment_sidecar(model_path: str) -> None:
    """Record the environment a model was pickled under, next to the
    pickle, so a later load failure names its cause instead of guessing."""
    import json

    import numpy as np

    with open(model_path + ".env.json", "w") as f:
        json.dump({"numpy": np.__version__,
                   "python": sys.version.split()[0]}, f)


def load_generator(path: str):
    """Load a saved synthesizer pickle. Minor numpy drifts are absorbed
    by the shims above; a MAJOR-version mismatch (the RNG state payload
    format changed between numpy 1.x and 2.x) cannot be shimmed safely
    and raises a clear, actionable error instead of a cryptic one."""
    import json
    import os

    import numpy as np

    _install_numpy_module_aliases()
    try:
        with _bit_generator_compat():
            with open(path, "rb") as f:
                return pickle.load(f)
    except Exception as e:
        saved_numpy = None
        sidecar = path + ".env.json"
        if os.path.exists(sidecar):
            with open(sidecar) as f:
                saved_numpy = json.load(f).get("numpy")
        hint = (f"saved under numpy {saved_numpy}, " if saved_numpy
                else "likely saved under a different numpy major version, ")
        raise RuntimeError(
            f"Cannot load {path}: {type(e).__name__}: {e}. This model was "
            f"{hint}and the current environment runs numpy {np.__version__} -- "
            f"the pickled RNG state format differs between numpy 1.x and 2.x. "
            f"Fix: match the generation-time numpy (see the generation "
            f"summary's library_versions), e.g. `pip install numpy==2.2.6`. "
            f"Note: anonymeter/numba forces numpy<2 -- it is only needed by "
            f"the attacks step, which skips it gracefully when absent."
        ) from e
