"""
UC1 data pipeline entrypoint.

Runs load_data -> profile_data -> preprocess -> profile_preprocessed_data
-> generate -> evaluate -> utility -> privacy in order, skipping any step already marked completed
(tracked in pipeline_status.json) unless explicitly forced.

Which synthesizers the generate step runs is config-driven
(config.synthesizers), so comparing models is a config change.

Usage:
    python main.py                          # run everything not yet done
    python main.py --force                  # rerun every step
    python main.py --force-step preprocess   # rerun just one step (repeatable)
    python main.py --only preprocess          # run just this step (repeatable;
                                               #   still respects its own completed
                                               #   status unless also forced)
    python main.py --status                 # print current step-completion status

All console output (including warnings and tracebacks, which shell
redirection alone would miss) is teed to logs.txt by default; override
with --log <path>.
"""

import argparse
import time
from datetime import datetime

from pipeline.config import PipelineConfig
from pipeline.logging_setup import start_logging, stop_logging
from pipeline.state import PipelineState
from pipeline.steps.load_data import LoadDataStep
from pipeline.steps.profile_data import ProfileDataStep
from pipeline.steps.preprocess import PreprocessStep
from pipeline.steps.profile_preprocessed_data import ProfilePreprocessedDataStep
from pipeline.steps.generate import GenerateStep
from pipeline.steps.evaluate import EvaluateStep
from pipeline.steps.privacy import PrivacyStep
from pipeline.steps.utility import UtilityStep

STEPS = [
    LoadDataStep(),
    ProfileDataStep(),
    PreprocessStep(),
    ProfilePreprocessedDataStep(),
    GenerateStep(),
    EvaluateStep(),
    UtilityStep(),
    PrivacyStep(),
]


def run_pipeline(
    config: PipelineConfig | None = None,
    force: bool = False,
    force_steps: list[str] | None = None,
    only: list[str] | None = None,
) -> None:
    config = config or PipelineConfig()
    state = PipelineState(config.status_path)
    force_steps = set(force_steps or [])

    steps = [s for s in STEPS if only is None or s.name in only]
    if not steps:
        raise ValueError(f"No matching step(s) for --only {only}. Known steps: {[s.name for s in STEPS]}")

    import os
    import shutil

    to_run = []
    for step in steps:
        should_force = force or step.name in force_steps
        if state.is_completed(step.name) and not should_force:
            print(f"⏭️  Skipping '{step.name}' (already completed). "
                  f"Use --force or --force-step {step.name} to rerun.")
            continue
        to_run.append(step)

    # Mark every step this run WILL execute as pending up front, so a
    # status check mid-run never shows a stale 'completed' from a
    # previous run for a step that is queued to be redone.
    for step in to_run:
        state.mark_pending(step.name)

    for step in to_run:
        # A rerun replaces the step's outputs wholesale: delete the old
        # ones first so nothing stale can survive next to fresh files.
        step_out = config.step_dir(step.name)
        if os.path.isdir(step_out):
            if step.name == "load_data" and not os.path.isdir(config.transfer_folder):
                raise FileNotFoundError(
                    f"Refusing to delete {step_out} before rerunning load_data: the transfer "
                    f"folder {config.transfer_folder} is not available to rebuild it from."
                )
            shutil.rmtree(step_out)
            print(f"🧹 Deleted previous outputs of '{step.name}' ({step_out}) for a clean rerun.")

        state.mark_running(step.name)
        started = time.time()
        print(f"\n{'=' * 70}\n▶️  RUNNING '{step.name}' (started {datetime.now().strftime('%H:%M:%S')})\n{'=' * 70}")
        try:
            step.run(config)
        except BaseException as e:
            # BaseException so a Ctrl-C or SIGTERM mid-step is recorded
            # too, instead of leaving the status file claiming 'running'
            # while the step actually died half-way.
            state.mark_failed(step.name, f"{type(e).__name__}: {e}")
            print(f"❌ Step '{step.name}' "
                  f"{'interrupted' if isinstance(e, (KeyboardInterrupt, SystemExit)) else 'failed'} "
                  f"after {time.time() - started:.0f}s: {e}")
            raise
        state.mark_completed(step.name)
        print(f"✅ '{step.name}' completed in {time.time() - started:.0f}s.")


def preflight(config: PipelineConfig | None = None) -> bool:
    """Everything a long run needs, checked in seconds. Returns True if
    the run can proceed."""
    import importlib
    import os
    import shutil

    config = config or PipelineConfig()
    ok = True

    def check(name, passed, detail=""):
        nonlocal ok
        mark = "✅" if passed else "❌"
        print(f"  {mark} {name}" + (f" -- {detail}" if detail else ""))
        ok = ok and passed

    print("Preflight checks:")
    for mod in ("polars", "pandas", "numpy", "scipy", "sdv", "snsynth", "torch", "cloudpickle"):
        try:
            m = importlib.import_module(mod)
            check(f"import {mod}", True, getattr(m, "__version__", ""))
        except Exception as e:
            check(f"import {mod}", False, f"{type(e).__name__}: {e}")

    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            check("CUDA GPU", True, f"{torch.cuda.get_device_name(0)}, {free/1e9:.1f} GB free of {total/1e9:.1f} GB")
        else:
            print("  ⚠️  no CUDA GPU -- ctgan/tvae/dpctgan will train on CPU (much slower); "
                  "gaussian_copula/mst/aim unaffected")
    except Exception:
        pass

    have_transfer = os.path.isdir(config.transfer_folder)
    have_loaded = os.path.exists(config.local_full_dataset_path)
    check("input data", have_transfer or have_loaded,
          config.transfer_folder if have_transfer else config.local_full_dataset_path)
    check("metadata.json", os.path.exists(config.metadata_path) or have_transfer, config.metadata_path)

    free_gb = shutil.disk_usage(os.path.dirname(config.output_dir) or ".").free / 1e9
    check("disk space >= 5 GB", free_gb >= 5, f"{free_gb:.1f} GB free")

    from pipeline.steps.generate.synthesizers import REGISTRY

    unknown = [n for n in config.synthesizers if n not in REGISTRY]
    check("synthesizers registered", not unknown, ", ".join(config.synthesizers))
    print(f"\n  Plan: {' -> '.join(config.synthesizers)}")
    print(f"  Per-model timeout: fit {config.synthesizer_timeout_seconds/3600:.1f}h, "
          f"sample {config.synthesizer_timeout_seconds/3600:.1f}h | seed {config.seed} | "
          f"DP epsilon {config.epsilon}")
    print(f"  Ordering is cheapest/most-reliable first, so a late timeout costs only the tail of the run.")
    print("\n" + ("Preflight PASSED -- ready for a long run." if ok else "Preflight FAILED -- fix the items above first."))
    return ok


def print_status(config: PipelineConfig | None = None) -> None:
    config = config or PipelineConfig()
    state = PipelineState(config.status_path)
    summary = state.summary()
    print(f"Pipeline status ({config.status_path}):")
    for step in STEPS:
        info = summary.get(step.name)
        if info is None:
            print(f"  {step.name}: never run")
        elif info.get("completed"):
            print(f"  {step.name}: ✅ completed at {info.get('completed_at')}")
        elif info.get("running"):
            print(f"  {step.name}: 🔄 running (started {info.get('started_at')})")
        elif info.get("pending"):
            print(f"  {step.name}: ⏳ pending (queued in the current run)")
        else:
            print(f"  {step.name}: ❌ failed at {info.get('failed_at')} -- {info.get('error')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="UC1 data preparation pipeline.")
    parser.add_argument("--force", action="store_true", help="Rerun every step, even if already completed.")
    parser.add_argument("--force-step", action="append", default=[],
                         help="Rerun this step even if already completed (repeatable).")
    parser.add_argument("--only", action="append", default=None,
                         help="Run only these step(s) (repeatable).")
    parser.add_argument("--status", action="store_true", help="Print step-completion status and exit.")
    parser.add_argument("--preflight", action="store_true",
                         help="Verify libraries, GPU, inputs, disk and config, then exit. "
                              "Run this before a long run.")
    parser.add_argument("--log", default="logs.txt",
                         help="File to tee all console output (stdout, stderr and warnings) to. "
                              "Default: logs.txt. Pass --log '' to disable.")
    args = parser.parse_args()

    if args.status:
        print_status()
        return

    if args.preflight:
        raise SystemExit(0 if preflight() else 1)

    # ./run_job.sh stop (and plain `kill`) send SIGTERM, which by default
    # ends the process without unwinding Python -- leaving the status
    # file claiming a step is still 'running'. Raise instead, so the
    # normal failure path records the interrupted step and the log gets
    # its closing line.
    import signal

    def _on_sigterm(signum, frame):
        raise SystemExit("terminated (SIGTERM)")

    signal.signal(signal.SIGTERM, _on_sigterm)

    # Tee everything to a log file so a failing run can be shared whole,
    # rather than only the stdout half that shell redirection captures.
    handle = start_logging(args.log) if args.log else None
    try:
        run_pipeline(force=args.force, force_steps=args.force_step, only=args.only)
    finally:
        if handle:
            stop_logging(handle)
            print(f"\nFull log written to {args.log}")


if __name__ == "__main__":
    main()
