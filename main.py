"""
UC1 data pipeline entrypoint.

Runs load_data -> profile_data -> preprocess in order, skipping any step
already marked completed (tracked in pipeline_status.json) unless
explicitly forced.

Usage:
    python main.py                          # run everything not yet done
    python main.py --force                  # rerun every step
    python main.py --force-step preprocess   # rerun just one step (repeatable)
    python main.py --only preprocess          # run just this step (repeatable;
                                               #   still respects its own completed
                                               #   status unless also forced)
    python main.py --status                 # print current step-completion status
"""

import argparse

from pipeline.config import PipelineConfig
from pipeline.state import PipelineState
from pipeline.steps.load_data import LoadDataStep
from pipeline.steps.profile_data import ProfileDataStep
from pipeline.steps.preprocess import PreprocessStep

STEPS = [LoadDataStep(), ProfileDataStep(), PreprocessStep()]


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

    for step in steps:
        should_force = force or step.name in force_steps
        if state.is_completed(step.name) and not should_force:
            print(f"⏭️  Skipping '{step.name}' (already completed). "
                  f"Use --force or --force-step {step.name} to rerun.")
            continue

        print(f"\n{'=' * 70}\n▶️  RUNNING '{step.name}'\n{'=' * 70}")
        try:
            step.run(config)
        except Exception as e:
            state.mark_failed(step.name, str(e))
            print(f"❌ Step '{step.name}' failed: {e}")
            raise
        state.mark_completed(step.name)
        print(f"✅ '{step.name}' completed.")


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
    args = parser.parse_args()

    if args.status:
        print_status()
        return

    run_pipeline(force=args.force, force_steps=args.force_step, only=args.only)


if __name__ == "__main__":
    main()
