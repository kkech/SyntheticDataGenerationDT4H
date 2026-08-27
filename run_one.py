"""
Re-run ONE generation run, in place, without touching the other outputs.

    python run_one.py --list
    python run_one.py --run-id aim50_eps5_seed0 --timeout 21600

Why this exists: `main.py --only generate` deletes output/generate/ before
rerunning, which is correct for a campaign (nothing stale survives next to
fresh files) and fatal for a retry -- it would destroy every CSV and fitted
generator the campaign already produced. This tool executes a single run
from the SAME plan, with the same code path (GenerateStep._run_one), and
merges the result into the existing generation summary: the run's row is
replaced if it is already there, appended otherwise.

The intended use is a run that failed for an infrastructural reason rather
than a quality one -- the classic case being an AIM run that hit its time
limit -- where the only thing that needs to change is the limit.

Guards, because this writes into a directory holding finished results:
  * refuses if the training split's SHA-256 no longer matches the one the
    summary was built from (a different split means every other row in
    that summary describes different data);
  * refuses to overwrite an existing output CSV unless --replace;
  * reuses the campaign's committed column selection for width-limited
    runs, so the retry trains on exactly the columns the campaign chose.

The new file is NOT evaluated by this tool. After it lands:
    python release_gate.py --file output/generate/DT4H_Synthetic_<run>.csv
    python main.py --analysis          # to fold it into the analysis steps
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import polars as pl  # noqa: E402

from pipeline.config import PipelineConfig  # noqa: E402
from pipeline.steps.generate.reproducibility import provenance  # noqa: E402
from pipeline.steps.generate.step import GenerateStep  # noqa: E402

SUMMARY_NAME = "DT4H_Generation_Summary.json"


def _load_summary(out_dir: str) -> dict:
    path = os.path.join(out_dir, SUMMARY_NAME)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. This tool retries one run of an existing campaign; "
            f"run the generate step at least once first (python main.py --only generate)."
        )
    with open(path) as f:
        return json.load(f)


def _column_subset(step, train, config, out_dir, k):
    """The top-k column selection for a width-limited run: the campaign's
    committed JSON if it is there (so the retry sees the same columns),
    recomputed otherwise."""
    path = os.path.join(
        out_dir,
        "DT4H_AIM_Column_Selection.json" if k == config.aim_max_columns
        else f"DT4H_Column_Selection_top{k}.json",
    )
    if os.path.exists(path):
        with open(path) as f:
            selected = json.load(f)["selected_columns"]
        print(f"Reusing the committed top-{k} column selection "
              f"({len(selected)} columns) -> {path}")
        return selected
    print(f"No committed selection at {path} -- recomputing it "
          f"(deterministic given this training split).")
    return step._select_top_columns(train, config, out_dir, k=k)


def _print_plan(config: PipelineConfig, summary: dict) -> None:
    status = {r.get("run_id"): r.get("status") for r in summary.get("runs", [])}
    print(f"{'run_id':<34} {'model':<12} {'ε':>6}  status")
    print("-" * 70)
    for spec in config.resolved_run_plan():
        eps = f"{spec['epsilon']:g}" if spec.get("epsilon") is not None else "-"
        print(f"{spec['run_id']:<34} {spec.get('record_as', spec['synthesizer']):<12} "
              f"{eps:>6}  {status.get(spec['run_id'], 'not in summary')}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-run a single generation run in place (see module docstring).")
    parser.add_argument("--run-id", help="Run to execute, e.g. aim50_eps5_seed0.")
    parser.add_argument("--list", action="store_true",
                        help="List the plan's run ids with their status in the summary.")
    parser.add_argument("--timeout", type=int,
                        help="Override the run's time limit, in seconds (applies to fit "
                             "and to sampling separately).")
    parser.add_argument("--rows", type=int,
                        help="Synthetic rows to generate (default: the campaign's count).")
    parser.add_argument("--replace", action="store_true",
                        help="Allow overwriting an existing output CSV for this run.")
    args = parser.parse_args()

    # The extended plan is a superset of the default one, so every run id
    # the campaign could have produced resolves here.
    config = PipelineConfig(extended_plan=True)
    out_dir = config.step_dir("generate")
    summary = _load_summary(out_dir)

    if args.list:
        _print_plan(config, summary)
        return 0
    if not args.run_id:
        parser.error("--run-id is required (or --list to see the options).")

    spec = next((dict(s) for s in config.resolved_run_plan()
                 if s["run_id"] == args.run_id), None)
    if spec is None:
        print(f"❌ '{args.run_id}' is not in the run plan. Use --list to see the run ids.")
        return 2

    if not os.path.exists(config.train_output_path):
        raise FileNotFoundError(
            f"{config.train_output_path} not found -- the training split this run must "
            f"be fitted on is missing. Run the preprocess step first.")

    # The summary describes one training split; a retry fitted on a
    # different one must not be merged into it.
    prov = provenance(config.train_output_path, config.seed)
    recorded_sha = (summary.get("provenance") or {}).get("training_data", {}).get("sha256")
    current_sha = prov["training_data"]["sha256"]
    if recorded_sha and current_sha and recorded_sha != current_sha:
        print("❌ The training split has changed since this summary was written "
              f"(summary {recorded_sha[:12]}..., current {current_sha[:12]}...).\n"
              "   Every other run in that summary was fitted on the older split, so a new "
              "run cannot be merged into it. Re-run the full generate step instead.")
        return 2

    csv_path = os.path.join(out_dir, f"DT4H_Synthetic_{args.run_id}.csv")
    if os.path.exists(csv_path) and not args.replace:
        print(f"❌ {csv_path} already exists. Pass --replace to overwrite it "
              f"(the previous file is not backed up -- use backup_results.py first if "
              f"you want to keep it).")
        return 2

    if args.timeout:
        spec["timeout_seconds"] = args.timeout

    df_pl = pl.read_parquet(config.train_output_path)
    real = df_pl.to_pandas()
    print(f"Loaded TRAINING split: {real.shape[0]} rows x {real.shape[1]} columns")

    step = GenerateStep()
    step._report_environment(prov)
    train, constants = step._split_constant_columns(real, config)
    recorded_constants = summary.get("constant_columns_held_out") or {}
    if recorded_constants and set(recorded_constants) != set(constants):
        print(f"⚠️  Constant columns held out now ({len(constants)}) differ from the "
              f"campaign's ({len(recorded_constants)}) -- the output schema of this run "
              f"may not match the other files.")

    width = spec.get("columns")
    width_k = config.aim_max_columns if width == "top" else width
    column_subsets = ({width_k: _column_subset(step, train, config, out_dir, width_k)}
                      if width_k else {})

    n_rows = args.rows or summary.get("n_synthetic_rows") or config.n_synthetic_rows \
        or real.shape[0]
    timeout = spec.get("timeout_seconds") or config.synthesizer_timeout_seconds
    print(f"\nRetrying a single run: {args.run_id} "
          f"({n_rows} rows, time limit {timeout}s per phase)")

    record = step._run_one(spec, train, real, constants, config, out_dir, n_rows,
                           column_subsets)

    # Parallel lanes (run_v3.sh dp-cpu / dp-gpu) run several run_one
    # processes at once; the summary loaded before the hours-long fit is
    # stale by now, and writing it back would silently drop any row a
    # concurrent retry merged in the meantime. Re-read and write under an
    # exclusive lock so merges serialize.
    import fcntl

    lock_path = os.path.join(out_dir, SUMMARY_NAME + ".lock")
    with open(lock_path, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        summary = _load_summary(out_dir)
        runs = summary.setdefault("runs", [])
        previous = None
        for i, r in enumerate(runs):
            if r.get("run_id") == args.run_id:
                previous = r.get("status")
                runs[i] = record
                break
        else:
            runs.append(record)
        summary["run_plan_size"] = len(runs)
        summary.setdefault("single_run_updates", []).append({
            "run_id": args.run_id,
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": prov["git"]["commit"],
            "timeout_seconds": spec.get("timeout_seconds"),
            "rows": n_rows,
            "status": record["status"],
            "previous_status": previous,
        })
        step._write_summary(summary, out_dir)
        fcntl.flock(lock, fcntl.LOCK_UN)
    print(f"Merged into {os.path.join(out_dir, SUMMARY_NAME)} "
          + (f"(replaced the previous '{previous}' row)." if previous
             else "(appended: this run was not in the summary)."))

    if record["status"] != "ok":
        print(f"\n❌ {args.run_id} did not produce a file: "
              f"{record.get('error_type')}: {record.get('error')}")
        return 1

    print("\nNext:")
    print(f"  python release_gate.py --file {csv_path}")
    print("  python main.py --analysis      # fold the new file into the analysis steps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
