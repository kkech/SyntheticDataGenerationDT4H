#!/usr/bin/env bash
# The v3 (corrected-methods) campaign, packaged as runnable stages so the
# whole rerun is a few commands instead of a page of shell. Follows the
# same conventions as run_job.sh: venv auto-activation, setsid+nohup
# detachment, per-stage PID files and logs under logs/.
#
#   ./run_v3.sh check          # everything that must be true BEFORE any run
#   ./run_v3.sh fix-artifacts  # scrub raw scores from committed selection JSONs
#   ./run_v3.sh pilot          # ~5 min DP wiring pilot (dpctgan eps=1) + auto-verify
#   ./run_v3.sh analysis       # analysis steps over EXISTING outputs + re-gate all files
#   ./run_v3.sh aim            # AIM eps=5 flagship retry (6h cap)
#   ./run_v3.sh dp-cpu         # MST re-fit lane (CPU-bound; ~21h thinned)
#   ./run_v3.sh dp-gpu         # DP-GAN re-fit lane (GPU-bound; ~6h thinned)
#   ./run_v3.sh ddpm           # fixed diffusion baseline re-runs (~20 min)
#   ./run_v3.sh finalize       # analysis + re-gate again over the re-fit outputs
#   ./run_v3.sh all            # the whole sequence, detached, lanes in parallel
#   ./run_v3.sh status|follow <stage>|stop <stage>
#
# Long stages (analysis, aim, dp-cpu, dp-gpu, ddpm, all) detach themselves
# (survive SSH disconnect) and log to logs/v3-<stage>.log. FULL=1 widens the
# DP sweeps back to every epsilon point; DRY=1 prints a lane's run list
# without executing it.
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p logs
BACKUP_MARKER=".v3_backup_done"
VALIDATED_MARKER=".v3_validated"

activate_venv() {
  if [ -z "${VIRTUAL_ENV:-}" ]; then
    for v in .testVenv .synthenv .venv venv; do
      if [ -f "$v/bin/activate" ]; then
        # shellcheck disable=SC1090
        source "$v/bin/activate"
        echo "Activated virtualenv: $v"
        return 0
      fi
    done
    echo "⚠️  No virtualenv found (.testVenv/.synthenv/.venv/venv) -- using $(command -v python3 || true)"
  fi
  return 0
}

py() { if command -v python >/dev/null 2>&1; then python "$@"; else python3 "$@"; fi; }

pid_file() { echo ".v3_${1}.pid"; }
log_file() { echo "logs/v3-${1}.log"; }

stage_running() {
  local pf; pf=$(pid_file "$1")
  [ -f "$pf" ] || return 1
  local pid; pid=$(cat "$pf")
  kill -0 "$pid" 2>/dev/null || return 1
  echo "$pid"
}

# Re-invoke this stage detached unless we already are the detached child.
maybe_detach() {
  local stage="$1"
  [ "${V3_DETACHED:-}" = "1" ] && return 0
  if pid=$(stage_running "$stage"); then
    echo "Stage '$stage' is already running (PID $pid). ./run_v3.sh follow $stage"
    exit 1
  fi
  local log; log=$(log_file "$stage")
  V3_DETACHED=1 setsid nohup bash "$0" "$stage" > "$log" 2>&1 < /dev/null &
  echo $! > "$(pid_file "$stage")"
  sleep 2
  if pid=$(stage_running "$stage"); then
    echo "✅ Stage '$stage' started detached (PID $pid). Safe to disconnect."
    echo "   ./run_v3.sh follow $stage   # live log"
    echo "   ./run_v3.sh status          # all stages"
  else
    echo "❌ Stage '$stage' exited immediately -- startup failure:"
    tail -n 30 "$log"
    rm -f "$(pid_file "$stage")"
    exit 1
  fi
  exit 0
}

domains_reviewed() {
  py - <<'EOF'
import json, sys
try:
    d = json.load(open("public_domains.json"))
except FileNotFoundError:
    print("MISSING"); sys.exit(0)
print("REVIEWED" if d.get("reviewed") is True else "UNREVIEWED")
EOF
}

require_reviewed_domains() {
  case "$(domains_reviewed | tail -1)" in
    REVIEWED) echo "✅ public_domains.json is reviewed." ;;
    MISSING)
      echo "❌ public_domains.json not found. Run: python make_public_domains.py, review it, set reviewed:true."
      exit 1 ;;
    *)
      echo "❌ public_domains.json has reviewed:false. Review every range for clinical"
      echo "   plausibility (edit lo/hi), then set \"reviewed\": true. DP fits refuse to start until then."
      exit 1 ;;
  esac
}

ensure_backup() {
  if [ -f "$BACKUP_MARKER" ]; then
    echo "Backup already taken this campaign ($(cat "$BACKUP_MARKER")). Delete $BACKUP_MARKER to force a new one."
    return 0
  fi
  echo "Snapshotting what the campaign can overwrite (slim backup: generate outputs + DP pickles)..."
  py backup_results.py --slim
  date -u +%Y-%m-%dT%H:%M:%SZ > "$BACKUP_MARKER"
}

run_ids() { # run_ids <lane>
  case "$1" in
    dp-cpu)
      if [ "${FULL:-0}" = "1" ]; then
        echo mst_eps0p5_seed0 mst_eps1_seed0 mst_eps5_seed0 mst_eps8_seed0 mst_eps10_seed0 \
             mst_eps15_seed0 mst_eps15_seed1 mst_eps15_seed2 mst_eps20_seed0
      else
        echo mst_eps0p5_seed0 mst_eps1_seed0 mst_eps5_seed0 \
             mst_eps15_seed0 mst_eps15_seed1 mst_eps15_seed2 mst_eps20_seed0
      fi ;;
    dp-gpu)
      if [ "${FULL:-0}" = "1" ]; then
        echo dpctgan_eps1_seed0 dpctgan_eps5_seed0 dpctgan_eps8_seed0 dpctgan_eps10_seed0 \
             dpctgan_eps15_seed0 dpctgan_eps15_seed1 dpctgan_eps15_seed2 dpctgan_eps20_seed0 \
             patectgan_eps1_seed0 patectgan_eps5_seed0 patectgan_eps15_seed0
      else
        echo dpctgan_eps1_seed0 dpctgan_eps15_seed0 dpctgan_eps15_seed1 dpctgan_eps15_seed2 \
             patectgan_eps1_seed0 patectgan_eps5_seed0 patectgan_eps15_seed0
      fi ;;
    ddpm) echo ddpm_seed0 ddpm_seed1 ddpm_seed2 ddpm_g_seed0 ;;
  esac
}

run_lane() { # run_lane <lane> [extra run_one args...]
  local lane="$1"; shift || true
  local ids; ids=$(run_ids "$lane")
  echo "Lane '$lane' run list: $ids"
  if [ "${DRY:-0}" = "1" ]; then echo "(DRY=1 -- not executing)"; return 0; fi
  local failed=""
  for id in $ids; do
    echo ""
    echo "=== [$lane] $id ($(date -u +%H:%M:%SZ)) ==="
    if ! py run_one.py --run-id "$id" --replace "$@"; then
      echo "❌ [$lane] $id FAILED -- continuing with the rest of the lane."
      failed="$failed $id"
    fi
  done
  if [ -n "$failed" ]; then
    echo ""
    echo "⚠️  Lane '$lane' finished with failures:$failed"
    return 1
  fi
  echo ""
  echo "✅ Lane '$lane' finished cleanly."
}

gate_all() {
  local pass=0 fail=0
  for f in output/generate/DT4H_Synthetic_*.csv; do
    [ -e "$f" ] || { echo "No synthetic CSVs to gate."; return 0; }
    echo ""
    echo "=== GATE $(basename "$f") ==="
    if py release_gate.py --file "$f"; then pass=$((pass+1)); else fail=$((fail+1)); fi
  done
  echo ""
  echo "Gate summary: $pass PASS, $fail FAIL (FAIL is a verdict, not an error -- see the per-file reports)."
}

verify_pilot() {
  py - <<'EOF'
import json, sys
s = json.load(open("output/generate/DT4H_Generation_Summary.json"))
run = next((r for r in s.get("runs", []) if r.get("run_id") == "dpctgan_eps1_seed0"), None)
ok = True
def chk(name, cond, detail=""):
    global ok
    print(f"  {'✅' if cond else '❌'} {name}" + (f" -- {detail}" if detail else ""))
    ok = ok and cond
if run is None:
    print("  ❌ pilot run not found in the generation summary"); sys.exit(1)
chk("status ok", run.get("status") == "ok", str(run.get("status")))
chk("delta recorded", run.get("delta") is not None, str(run.get("delta")))
chk("delta passed to library (or recorded fallback)", "delta_passed_to_library" in run,
    str(run.get("delta_passed_to_library")))
chk("bounds from reviewed public domains", "public_domains.json" in str(run.get("bounds_source")),
    str(run.get("bounds_source")))
chk("no verbatim training rows", (run.get("leakage") or {}).get("exact_duplicates_of_training_rows", 1) == 0)
print("\nPILOT " + ("PASSED -- the DP wiring is live; the long lanes are safe to start."
                    if ok else "FAILED -- fix the items above before any long DP run."))
sys.exit(0 if ok else 1)
EOF
}

require_validated() {
  if [ ! -f "$VALIDATED_MARKER" ]; then
    echo "❌ The cheap validation has not passed yet. Run the simple things first:"
    echo "     ./run_v3.sh            # check + fix-artifacts + pilot (~10 min)"
    echo "   Long stages refuse to start until it succeeds ($VALIDATED_MARKER)."
    exit 1
  fi
  echo "✅ Validation previously passed ($(cat "$VALIDATED_MARKER"))."
}

stage_validate() {
  activate_venv
  echo "################################################################"
  echo "# v3 VALIDATE: every cheap check, no long-running task started #"
  echo "################################################################"
  stage_check
  echo ""
  echo "=== scrubbing committed selection artifacts ==="
  stage_fix_artifacts
  echo ""
  stage_pilot
  date -u +%Y-%m-%dT%H:%M:%SZ > "$VALIDATED_MARKER"
  echo ""
  echo "✅ VALIDATION PASSED. Long stages are now unlocked:"
  echo "     ./run_v3.sh analysis    # hours, no GPU -- do this next and read the results"
  echo "     ./run_v3.sh dp-cpu  and  ./run_v3.sh dp-gpu   # the re-fit lanes, in parallel"
  echo "     ./run_v3.sh all         # or the whole remaining sequence in one detached job"
}

stage_check() {
  activate_venv
  echo "=== v3 campaign pre-run check ==="
  require_reviewed_domains
  if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
    echo "⚠️  Uncommitted changes present. Commit before the campaign so the recorded"
    echo "   git commit in every run's provenance actually reproduces the code that ran."
  else
    echo "✅ Git tree clean."
  fi
  for j in output/generate/DT4H_AIM_Column_Selection.json output/generate/DT4H_Column_Selection_top40.json; do
    if [ -f "$j" ] && py -c "import json,sys; sys.exit(0 if 'scores' in json.load(open('$j')) else 1)"; then
      echo "⚠️  $j still contains the raw 'scores' block -- run: ./run_v3.sh fix-artifacts"
    fi
  done
  py main.py --preflight --min-free-gb 2
}

stage_fix_artifacts() {
  activate_venv
  py - <<'EOF'
import json, os
for p in ("output/generate/DT4H_AIM_Column_Selection.json",
          "output/generate/DT4H_Column_Selection_top40.json"):
    if not os.path.exists(p):
        print(f"  (absent) {p}"); continue
    d = json.load(open(p))
    if "scores" in d:
        del d["scores"]
        d.setdefault("disclosure",
            "raw association scores removed: they were exact statistics of the "
            "real training split; the ranked order remains for auditability")
        json.dump(d, open(p, "w"), indent=2)
        print(f"  ✅ scrubbed raw scores from {p}")
    else:
        print(f"  ✅ already clean: {p}")
EOF
}

stage_pilot() {
  activate_venv
  require_reviewed_domains
  ensure_backup
  echo "=== DP wiring pilot: dpctgan_eps1_seed0 (~5 min) ==="
  py run_one.py --run-id dpctgan_eps1_seed0 --replace
  echo ""
  echo "=== verifying the pilot's recorded DP provenance ==="
  verify_pilot
}

stage_analysis() {
  require_validated
  maybe_detach analysis
  activate_venv
  echo "=== analysis rerun over existing outputs ==="
  py main.py --analysis
  echo ""
  echo "=== re-gating every synthetic file ==="
  gate_all
}

stage_aim() {
  require_validated
  maybe_detach aim
  activate_venv
  require_reviewed_domains
  ensure_backup
  echo "=== AIM eps=5 flagship retry (cap 6h) ==="
  py run_one.py --run-id aim50_eps5_seed0 --timeout 21600 --replace
}

stage_dp_cpu() { require_validated; maybe_detach dp-cpu; activate_venv; require_reviewed_domains; ensure_backup; run_lane dp-cpu; }
stage_dp_gpu() { require_validated; maybe_detach dp-gpu; activate_venv; require_reviewed_domains; ensure_backup; run_lane dp-gpu; }
stage_ddpm()   { require_validated; maybe_detach ddpm; activate_venv; ensure_backup; run_lane ddpm; }

stage_finalize() {
  require_validated
  maybe_detach finalize
  activate_venv
  echo "=== final analysis + gate over the re-fit outputs ==="
  py main.py --analysis
  gate_all
}

stage_all() {
  maybe_detach all
  activate_venv
  require_reviewed_domains
  if [ -n "$(git status --porcelain 2>/dev/null)" ] && [ "${ALLOW_DIRTY:-0}" != "1" ]; then
    echo "❌ Uncommitted changes present. Commit first (or ALLOW_DIRTY=1 to override --"
    echo "   the run provenance will then record a dirty tree)."
    exit 1
  fi
  ensure_backup
  echo "########## v3 campaign: full sequence ##########"
  stage_fix_artifacts
  echo ""
  echo "########## pilot ##########"
  py run_one.py --run-id dpctgan_eps1_seed0 --replace
  verify_pilot
  echo ""
  echo "########## analysis over existing outputs ##########"
  py main.py --analysis
  gate_all
  echo ""
  echo "########## DP re-fit lanes (parallel) ##########"
  cpu_log=$(log_file dp-cpu); gpu_log=$(log_file dp-gpu)
  ( { py run_one.py --run-id aim50_eps5_seed0 --timeout 21600 --replace || true; }
    run_lane dp-cpu || true ) > "$cpu_log" 2>&1 &
  cpu_pid=$!
  ( run_lane dp-gpu || true
    run_lane ddpm  || true ) > "$gpu_log" 2>&1 &
  gpu_pid=$!
  echo "CPU lane (aim + mst) PID $cpu_pid -> $cpu_log"
  echo "GPU lane (gans + ddpm) PID $gpu_pid -> $gpu_log"
  wait "$cpu_pid" "$gpu_pid"
  echo ""
  echo "########## finalize ##########"
  py main.py --analysis
  gate_all
  echo ""
  echo "########## v3 campaign COMPLETE ##########"
}

case "${1:-validate}" in
  validate)      stage_validate ;;
  check)         stage_check ;;
  fix-artifacts) stage_fix_artifacts ;;
  pilot)         stage_pilot ;;
  analysis)      stage_analysis ;;
  aim)           stage_aim ;;
  dp-cpu)        stage_dp_cpu ;;
  dp-gpu)        stage_dp_gpu ;;
  ddpm)          stage_ddpm ;;
  finalize)      stage_finalize ;;
  all)           stage_all ;;

  status)
    for s in analysis aim dp-cpu dp-gpu ddpm finalize all; do
      if pid=$(stage_running "$s"); then
        echo "🟢 $s RUNNING (PID $pid, elapsed $(ps -o etime= -p "$pid" | tr -d ' '))"
      elif [ -f "$(log_file "$s")" ]; then
        echo "⚪ $s not running (log exists: $(log_file "$s"))"
      fi
    done
    activate_venv >/dev/null 2>&1
    py main.py --status 2>/dev/null || true
    ;;

  follow)
    s="${2:-all}"
    [ -f "$(log_file "$s")" ] || { echo "No log for stage '$s' yet."; exit 1; }
    exec tail -n 30 -f "$(log_file "$s")"
    ;;

  stop)
    s="${2:-}"
    [ -n "$s" ] || { echo "Usage: $0 stop <stage>"; exit 1; }
    if pid=$(stage_running "$s"); then
      echo "Stopping stage '$s' (PID $pid)..."
      kill -- "-$pid" 2>/dev/null || kill "$pid"
      rm -f "$(pid_file "$s")"
      echo "Stopped. Completed runs/steps are kept; rerun the stage to continue."
    else
      echo "Stage '$s' is not running."
      rm -f "$(pid_file "$s")"
    fi
    ;;

  *)
    echo "v3 (corrected-methods) campaign runner."
    echo ""
    echo "  ./run_v3.sh                RUN THIS FIRST: check + fix-artifacts + pilot (~10 min,"
    echo "                             nothing long); long stages stay locked until it passes"
    echo ""
    echo "Stages, in order:"
    echo ""
    echo "  ./run_v3.sh check          pre-run gate: reviewed domains, clean tree, preflight"
    echo "  ./run_v3.sh fix-artifacts  scrub raw scores from the committed selection JSONs"
    echo "  ./run_v3.sh pilot          ~5 min DP wiring pilot + automatic verification"
    echo "  ./run_v3.sh analysis       analysis steps over EXISTING outputs + re-gate (detached)"
    echo "  ./run_v3.sh aim            AIM eps=5 flagship retry, 6h cap (detached)"
    echo "  ./run_v3.sh dp-cpu         MST re-fit lane, run in parallel with... (detached)"
    echo "  ./run_v3.sh dp-gpu         ...the DP-GAN re-fit lane (detached)"
    echo "  ./run_v3.sh ddpm           fixed diffusion baseline, ~20 min (detached)"
    echo "  ./run_v3.sh finalize       analysis + re-gate over the re-fit outputs (detached)"
    echo ""
    echo "  ./run_v3.sh all            everything above in sequence, lanes in parallel (detached)"
    echo "  ./run_v3.sh status | follow <stage> | stop <stage>"
    echo ""
    echo "  FULL=1  widen the DP sweeps to every epsilon point (adds ~9h to dp-cpu)"
    echo "  DRY=1   print a lane's run list without executing"
    exit 1
    ;;
esac
