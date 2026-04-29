#!/usr/bin/env bash
# Probe Amarel/Slurm limits before launching the full hparam grid.
#
# From ~/aim3_runner:
#   bash experiments/amarel/probe_amarel_slurm_limits.sh \
#     --partition gpu-redhat --account general --gres gpu:1 \
#     --cpus-per-task 4 --mem 16G --array-size 200 \
#     --times "24:00:00 48:00:00 72:00:00" \
#     --array-concurrency "16 32 48 64 96 128 160 200"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ART_DIR="$ROOT/artifacts/amarel_probe"

PARTITION="${PARTITION:-${SLURM_PARTITION:-}}"
ACCOUNT="${ACCOUNT:-${SLURM_ACCOUNT:-}}"
GRES="${GRES:-gpu:1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEM="${MEM:-16G}"
ARRAY_SIZE="${ARRAY_SIZE:-200}"
TIME_CANDIDATES="${TIME_CANDIDATES:-24:00:00 48:00:00 72:00:00}"
NODE_CANDIDATES="${NODE_CANDIDATES:-1 2 4 8}"
ARRAY_CONCURRENCY_CANDIDATES="${ARRAY_CONCURRENCY_CANDIDATES:-16 32 48 64 96 128 160 200}"
SUBMIT_SMOKE=0
SMOKE_CONCURRENCY="${SMOKE_CONCURRENCY:-4}"
SMOKE_SLEEP_SECONDS="${SMOKE_SLEEP_SECONDS:-60}"
OUT=""

usage() {
  sed -n '1,12p' "$0"
  cat <<'EOF'

Options:
  --partition NAME              Slurm partition to test.
  --account NAME                Slurm account/allocation to test.
  --gres VALUE                  GRES request, e.g. gpu:1. Use "none" to omit.
  --cpus-per-task N             CPUs per task for test jobs.
  --mem VALUE                   Memory request, e.g. 16G.
  --array-size N                Array size used for test-only array jobs.
  --times "T1 T2 ..."           Walltime candidates.
  --nodes "N1 N2 ..."           Node-count candidates for test-only jobs.
  --array-concurrency "C ..."   Array throttle candidates for --array=0-N%C.
  --submit-smoke                Submit a short sleep array to check real scheduling.
  --smoke-concurrency N         Array throttle for smoke submission.
  --smoke-sleep SECONDS         Sleep duration for smoke tasks.
  --out PATH                    Output log path.
  -h, --help                    Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --partition) PARTITION="$2"; shift 2 ;;
    --account) ACCOUNT="$2"; shift 2 ;;
    --gres) GRES="$2"; shift 2 ;;
    --cpus-per-task) CPUS_PER_TASK="$2"; shift 2 ;;
    --mem) MEM="$2"; shift 2 ;;
    --array-size) ARRAY_SIZE="$2"; shift 2 ;;
    --times) TIME_CANDIDATES="$2"; shift 2 ;;
    --nodes) NODE_CANDIDATES="$2"; shift 2 ;;
    --array-concurrency) ARRAY_CONCURRENCY_CANDIDATES="$2"; shift 2 ;;
    --submit-smoke) SUBMIT_SMOKE=1; shift ;;
    --smoke-concurrency) SMOKE_CONCURRENCY="$2"; shift 2 ;;
    --smoke-sleep) SMOKE_SLEEP_SECONDS="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

mkdir -p "$ART_DIR"
if [[ -z "$OUT" ]]; then
  OUT="$ART_DIR/probe_$(date +%Y%m%d_%H%M%S).log"
fi

log() {
  printf '%s\n' "$*" | tee -a "$OUT"
}

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

run_info_cmd() {
  local label="$1"
  shift
  log ""
  log "### $label"
  log "$ $*"
  set +e
  "$@" 2>&1 | tee -a "$OUT"
  local rc=${PIPESTATUS[0]}
  set -e
  log "INFO_RESULT label=\"$label\" rc=$rc"
}

SBATCH_BASE=()
if [[ -n "$PARTITION" ]]; then
  SBATCH_BASE+=(--partition "$PARTITION")
fi
if [[ -n "$ACCOUNT" ]]; then
  SBATCH_BASE+=(--account "$ACCOUNT")
fi
if [[ -n "$GRES" && "$GRES" != "none" ]]; then
  SBATCH_BASE+=(--gres "$GRES")
fi
SBATCH_BASE+=(--cpus-per-task "$CPUS_PER_TASK" --mem "$MEM")

run_sbatch_test() {
  local kind="$1"
  local detail="$2"
  shift 2
  local cmd=(sbatch --test-only "${SBATCH_BASE[@]}" "$@" --wrap "sleep 60")
  log ""
  log "### TEST kind=$kind detail=$detail"
  log "$ ${cmd[*]}"
  set +e
  local output
  output="$("${cmd[@]}" 2>&1)"
  local rc=$?
  set -e
  printf '%s\n' "$output" | tee -a "$OUT"
  if [[ $rc -eq 0 ]]; then
    log "RESULT kind=$kind detail=$detail status=OK rc=$rc"
  else
    log "RESULT kind=$kind detail=$detail status=FAIL rc=$rc"
  fi
}

submit_smoke_array() {
  local last_index=$((SMOKE_CONCURRENCY - 1))
  local output_path="$ART_DIR/smoke_%A_%a.out"
  local cmd=(
    sbatch
    "${SBATCH_BASE[@]}"
    --job-name aim3-probe-smoke
    --array "0-${last_index}%${SMOKE_CONCURRENCY}"
    --time 00:05:00
    --output "$output_path"
    --wrap "echo host=\$(hostname) task=\${SLURM_ARRAY_TASK_ID}; sleep ${SMOKE_SLEEP_SECONDS}"
  )
  log ""
  log "### SUBMIT smoke array"
  log "$ ${cmd[*]}"
  set +e
  local output
  output="$("${cmd[@]}" 2>&1)"
  local rc=$?
  set -e
  printf '%s\n' "$output" | tee -a "$OUT"
  log "SMOKE_RESULT rc=$rc"
}

log "AIM3 Amarel Slurm probe"
log "timestamp=$(date -Is)"
log "repo_root=$ROOT"
log "output=$OUT"
log "user=${USER:-unknown}"
log "host=$(hostname)"
log "partition=${PARTITION:-<unset>}"
log "account=${ACCOUNT:-<unset>}"
log "gres=$GRES"
log "cpus_per_task=$CPUS_PER_TASK"
log "mem=$MEM"
log "array_size=$ARRAY_SIZE"
log "time_candidates=$TIME_CANDIDATES"
log "node_candidates=$NODE_CANDIDATES"
log "array_concurrency_candidates=$ARRAY_CONCURRENCY_CANDIDATES"
log "submit_smoke=$SUBMIT_SMOKE"

if ! have_cmd sbatch; then
  log "ERROR: sbatch not found. Run this script on an Amarel login node."
  exit 1
fi

run_info_cmd "Slurm version" sbatch --version
if have_cmd sinfo; then
  run_info_cmd "sinfo GPU-oriented summary" sinfo -o "%P %a %l %D %G %c %m %f"
fi
if have_cmd scontrol; then
  run_info_cmd "scontrol show partition" scontrol show partition
  run_info_cmd "scontrol show config" scontrol show config
fi
if have_cmd sacctmgr; then
  run_info_cmd "sacctmgr user associations" sacctmgr show assoc user="${USER:-}" format=Cluster,Account,Partition,QOS,MaxJobs,MaxSubmit,MaxWall,GrpTRES%40
  run_info_cmd "sacctmgr QOS summary" sacctmgr show qos format=Name,MaxWall,MaxJobsPU,MaxSubmitJobsPU,MaxTRESPU%40,GrpTRES%40
fi

log ""
log "## sbatch --test-only: single-job walltime/node matrix"
for nodes in $NODE_CANDIDATES; do
  for walltime in $TIME_CANDIDATES; do
    run_sbatch_test "single" "nodes=${nodes},time=${walltime}" \
      --nodes "$nodes" \
      --ntasks-per-node 1 \
      --time "$walltime"
  done
done

log ""
log "## sbatch --test-only: job-array concurrency matrix"
array_last=$((ARRAY_SIZE - 1))
for walltime in $TIME_CANDIDATES; do
  for concurrency in $ARRAY_CONCURRENCY_CANDIDATES; do
    run_sbatch_test "array" "size=${ARRAY_SIZE},concurrency=${concurrency},time=${walltime}" \
      --array "0-${array_last}%${concurrency}" \
      --time "$walltime"
  done
done

if [[ "$SUBMIT_SMOKE" -eq 1 ]]; then
  submit_smoke_array
else
  log ""
  log "Smoke submission skipped. Re-run with --submit-smoke to submit a short sleep array."
fi

log ""
log "Probe complete. Paste this log back for recommended hparam job-array settings:"
log "$OUT"
