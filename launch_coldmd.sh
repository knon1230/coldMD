#!/usr/bin/env bash
# ColdMD 2.1.2: validate, preflight, launch, fork, and strict-resume wrapper.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-${SCRIPT_DIR}}"
CONFIG_FILE="${CONFIG_FILE:-}"
RUN_BASE="${RUN_BASE:-${SCRIPT_DIR}/../coldmd-runs}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/../coldmd-logs}"
COLDMD_ENV="${COLDMD_ENV:-coldmd-v212}"
CPU_THREADS="${CPU_THREADS:-${SLURM_CPUS_PER_TASK:-24}}"
DRYRUN_NVT_STEPS="${DRYRUN_NVT_STEPS:-100}"
STARTUP_CHECK_SECONDS="${STARTUP_CHECK_SECONDS:-1}"
EXPECTED_COLDMD_VERSION="2.1.2"

COLDMD_BIN=""
PYTHON_BIN=""
PREFLIGHT_RESULT_DIR=""

usage() {
  cat <<'EOF'
Usage:
  ./launch_coldmd.sh check CONFIG.yaml
  ./launch_coldmd.sh new CONFIG.yaml [run-label]
  ./launch_coldmd.sh fork CONFIG.yaml PARENT_RUN STATE.extxyz [run-label]
  ./launch_coldmd.sh resume RUN_DIRECTORY

Commands:
  check   Validate YAML/calculator and run a fixed-cell NVT smoke test.
  new     Pass the same preflight, then start a fresh run with nohup.
  fork    Validate a parent stage EXTXYZ (including momenta), then start an
          independent child run. STATE may be an absolute path or a filename
          below PARENT_RUN/stage_structures.
  resume  Continue the same run from its latest exact checkpoint.

Environment overrides:
  CONDA_SH             Path to conda.sh when conda is not discoverable.
  COLDMD_ENV           Conda environment (default: coldmd-v212).
  PROJECT_DIR          Installed project directory (default: this directory).
  CONFIG_FILE          Fallback config for check/new.
  RUN_BASE             Parent directory for new and fork child runs (default: sibling coldmd-runs).
  LOG_DIR              Parent directory for preflight/terminal logs (default: sibling coldmd-logs).
  CPU_THREADS          Thread count (default: SLURM_CPUS_PER_TASK or 24).
  DRYRUN_NVT_STEPS     Fresh-run smoke steps (default: 100).
  STARTUP_CHECK_SECONDS  Background-process startup check delay (default: 1).
EOF
}

fail() {
  echo "Error: $*" >&2
  exit 1
}

require_positive_integer() {
  local value="$1"
  local label="$2"
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || fail "${label} must be a positive integer"
}

require_nonnegative_decimal() {
  local value="$1"
  local label="$2"
  [[ "$value" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || \
    fail "${label} must be a non-negative decimal number"
}

activate_environment() {
  if [[ -n "${CONDA_SH:-}" ]]; then
    [[ -f "$CONDA_SH" ]] || fail "CONDA_SH does not exist: $CONDA_SH"
    # shellcheck source=/dev/null
    source "$CONDA_SH"
  elif command -v conda >/dev/null 2>&1; then
    local conda_base
    conda_base="$(conda info --base)"
    # shellcheck source=/dev/null
    source "${conda_base}/etc/profile.d/conda.sh"
  else
    fail "Conda was not found. Set CONDA_SH to .../etc/profile.d/conda.sh"
  fi
  conda activate "$COLDMD_ENV"
}

setup_runtime() {
  [[ -d "$PROJECT_DIR" ]] || fail "Project directory not found: $PROJECT_DIR"
  require_positive_integer "$CPU_THREADS" "CPU_THREADS"
  require_nonnegative_decimal "$STARTUP_CHECK_SECONDS" "STARTUP_CHECK_SECONDS"
  mkdir -p "$RUN_BASE" "$LOG_DIR"
  PROJECT_DIR="$(realpath "$PROJECT_DIR")"
  RUN_BASE="$(realpath "$RUN_BASE")"
  LOG_DIR="$(realpath "$LOG_DIR")"
  activate_environment

  export OMP_NUM_THREADS="$CPU_THREADS"
  export MKL_NUM_THREADS="$CPU_THREADS"
  export OPENBLAS_NUM_THREADS="$CPU_THREADS"
  cd "$PROJECT_DIR"
  COLDMD_BIN="$(command -v coldmd || true)"
  PYTHON_BIN="$(command -v python || true)"
  [[ -n "$COLDMD_BIN" ]] || fail "coldmd was not found in '${COLDMD_ENV}'"
  [[ -n "$PYTHON_BIN" ]] || fail "python was not found in '${COLDMD_ENV}'"
  local version_output
  version_output="$("$COLDMD_BIN" --version)"
  [[ "$version_output" == "coldmd ${EXPECTED_COLDMD_VERSION}" ]] || \
    fail "Expected coldmd ${EXPECTED_COLDMD_VERSION}, found: ${version_output}. Install this project in '${COLDMD_ENV}'."
}

resolve_config() {
  local candidate="$1"
  [[ -n "$candidate" ]] || fail "A CONFIG.yaml argument is required, or set CONFIG_FILE"
  [[ -f "$candidate" ]] || fail "Config file not found: $candidate"
  realpath "$candidate"
}

safe_stem() {
  local stem
  stem="$(basename "$1")"
  stem="${stem%.*}"
  stem="${stem//[^A-Za-z0-9._-]/_}"
  [[ -n "$stem" ]] || stem="config"
  printf '%s\n' "$stem"
}

append_stderr_if_present() {
  local label="$1"
  local path="$2"
  local log="$3"
  if [[ -s "$path" ]]; then
    {
      echo "--- ${label} stderr ---"
      sed -n '1,240p' "$path"
      echo "--- end ${label} stderr ---"
    } >> "$log"
  fi
}

inspect_command_json() {
  local stdout_path="$1"
  local json_path="$2"
  local kind="$3"
  "$PYTHON_BIN" - "$stdout_path" "$json_path" "$kind" <<'PY'
import json
import sys
from pathlib import Path

stdout_path, json_path, kind = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
text = stdout_path.read_text(encoding="utf-8")
try:
    payload = json.loads(text)
except json.JSONDecodeError as strict_error:
    decoder = json.JSONDecoder()
    candidates = []
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith("{"):
            start = offset + len(line) - len(stripped)
            try:
                candidate, _ = decoder.raw_decode(text, start)
            except json.JSONDecodeError:
                pass
            else:
                if isinstance(candidate, dict) and "status" in candidate:
                    candidates.append(candidate)
        offset += len(line)
    if len(candidates) != 1:
        raise SystemExit(
            f"Could not extract one JSON result from {stdout_path}: "
            f"found {len(candidates)} objects after {strict_error}"
        )
    payload = candidates[0]

if not isinstance(payload, dict):
    raise SystemExit("Command JSON root must be an object")
if payload.get("status") != "valid":
    raise SystemExit(f"{kind} status is not 'valid': {payload.get('status')!r}")
if kind == "validate":
    if payload.get("calculator_checked") is not True:
        raise SystemExit("Validation did not check the calculator")
    total_steps = payload.get("total_steps")
    if isinstance(total_steps, bool) or not isinstance(total_steps, int) or total_steps <= 0:
        raise SystemExit(f"Invalid production total_steps: {total_steps!r}")
    print(f"Validation passed: {total_steps} production steps")
    if payload.get("strict_resume_available") is False:
        print("WARNING: strict resume is unavailable; use a local content-hashed MACE model.")
elif kind == "fork":
    if payload.get("fork_ready") is not True:
        raise SystemExit("Fork preflight did not report fork_ready=true")
    origin = payload.get("child_origin")
    if not isinstance(origin, dict) or origin.get("global_step") != 0 or origin.get("time_fs") != 0.0 or origin.get("volume_ratio_from_engine_start") != 1.0:
        raise SystemExit(f"Fork child origin contract is invalid: {origin!r}")
    provenance = payload.get("fork_provenance")
    digest = provenance.get("source_state_sha256") if isinstance(provenance, dict) else None
    if not isinstance(digest, str) or len(digest) != 64:
        raise SystemExit("Fork source hash is absent or malformed")
    print(f"Fork source passed: {provenance.get('source_state')}")
    print("Child origin: step=0, time_fs=0, V/V0=1; source momentum preserved")

temporary = json_path.with_suffix(json_path.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(json_path)
PY
}

inspect_dryrun_json() {
  local validation_json="$1"
  local dryrun_json="$2"
  local expected_steps="$3"
  "$PYTHON_BIN" - "$validation_json" "$dryrun_json" "$expected_steps" <<'PY'
import json
import sys
from pathlib import Path

validation = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
payload = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
expected = int(sys.argv[3])
if not isinstance(payload, dict) or payload.get("status") != "passed":
    raise SystemExit(f"Dry-run status is not 'passed': {payload.get('status')!r}")
if payload.get("dryrun_nvt_steps") != expected:
    raise SystemExit("Dry-run step count differs from the request")
if payload.get("dryrun_state") != "input_cell_fixed_nvt":
    raise SystemExit("Dry run did not use the fixed input cell")
if payload.get("production_total_steps") != validation.get("total_steps"):
    raise SystemExit("Dry run and validation disagree on production steps")
print(f"Dry run passed: {expected} fixed-cell NVT steps")
PY
}

run_preflight() {
  local config_path="$1"
  local label="$2"
  local stamp="$3"
  require_positive_integer "$DRYRUN_NVT_STEPS" "DRYRUN_NVT_STEPS"

  local directory="${LOG_DIR}/preflight-${label}-${stamp}"
  local validation_stdout="${directory}/validate.stdout.log"
  local validation_stderr="${directory}/validate.stderr.log"
  local validation_json="${directory}/validate.json"
  local dryrun_dir="${directory}/dryrun"
  local dryrun_stdout="${directory}/dryrun.stdout.log"
  local dryrun_stderr="${directory}/dryrun.stderr.log"
  local log="${directory}/preflight.log"
  [[ ! -e "$directory" ]] || fail "Preflight path exists: $directory"
  mkdir -p "$dryrun_dir"
  {
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "config=${config_path}"
    echo "coldmd=$("$COLDMD_BIN" --version)"
  } > "$log"

  echo "Validating configuration and calculator..."
  if ! "$COLDMD_BIN" validate "$config_path" --json > "$validation_stdout" 2> "$validation_stderr"; then
    append_stderr_if_present "validation" "$validation_stderr" "$log"
    echo "Preflight failed during validation. Inspect: $directory" >&2
    return 1
  fi
  append_stderr_if_present "validation" "$validation_stderr" "$log"
  local summary
  if ! summary="$(inspect_command_json "$validation_stdout" "$validation_json" validate)"; then
    echo "Preflight failed while inspecting validation JSON. Inspect: $directory" >&2
    return 1
  fi
  printf '%s\n' "$summary" | tee -a "$log"

  echo "Running ${DRYRUN_NVT_STEPS}-step fixed-cell NVT dry run..."
  if ! "$COLDMD_BIN" dryrun "$config_path" --nvt-steps "$DRYRUN_NVT_STEPS" --output-dir "$dryrun_dir" --json > "$dryrun_stdout" 2> "$dryrun_stderr"; then
    append_stderr_if_present "dryrun" "$dryrun_stderr" "$log"
    echo "Preflight failed during dry run. Inspect: $directory" >&2
    return 1
  fi
  append_stderr_if_present "dryrun" "$dryrun_stderr" "$log"
  local dryrun_json="${dryrun_dir}/dryrun.json"
  [[ -f "$dryrun_json" ]] || {
    echo "Preflight failed: missing $dryrun_json" >&2
    return 1
  }
  if ! summary="$(inspect_dryrun_json "$validation_json" "$dryrun_json" "$DRYRUN_NVT_STEPS")"; then
    echo "Preflight failed while inspecting dry-run JSON. Inspect: $directory" >&2
    return 1
  fi
  printf '%s\n' "$summary" | tee -a "$log"
  echo "Preflight completed successfully."
  echo "Preflight artifacts: $directory"
  PREFLIGHT_RESULT_DIR="$directory"
}

run_fork_preflight() {
  local config_path="$1"
  local parent="$2"
  local state="$3"
  local label="$4"
  local stamp="$5"
  local directory="${LOG_DIR}/preflight-fork-${label}-${stamp}"
  local stdout_path="${directory}/fork-check.stdout.log"
  local stderr_path="${directory}/fork-check.stderr.log"
  local json_path="${directory}/fork-check.json"
  local log="${directory}/preflight.log"
  [[ ! -e "$directory" ]] || fail "Fork preflight path exists: $directory"
  mkdir -p "$directory"
  {
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "config=${config_path}"
    echo "parent_run=${parent}"
    echo "source_state=${state}"
    echo "coldmd=$("$COLDMD_BIN" --version)"
  } > "$log"

  echo "Validating fork source, momenta, child YAML, and calculator..."
  if ! "$COLDMD_BIN" fork "$config_path" --from-run "$parent" --state "$state" --check-only --json > "$stdout_path" 2> "$stderr_path"; then
    append_stderr_if_present "fork check" "$stderr_path" "$log"
    echo "Fork preflight failed. Inspect: $directory" >&2
    return 1
  fi
  append_stderr_if_present "fork check" "$stderr_path" "$log"
  local summary
  if ! summary="$(inspect_command_json "$stdout_path" "$json_path" fork)"; then
    echo "Fork preflight JSON inspection failed. Inspect: $directory" >&2
    return 1
  fi
  printf '%s\n' "$summary" | tee -a "$log"
  echo "Fork preflight completed successfully."
  echo "Preflight artifacts: $directory"
  PREFLIGHT_RESULT_DIR="$directory"
}

launch_background() {
  local mode="$1"
  local run_dir="$2"
  local stamp="$3"
  shift 3
  local command_args=("$@")
  local stem="${mode}-$(basename "$run_dir")-${stamp}"
  local log_file="${LOG_DIR}/${stem}.out"
  local pid_file="${LOG_DIR}/${stem}.pid"
  {
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "mode=${mode}"
    echo "run_directory=${run_dir}"
    [[ -z "$PREFLIGHT_RESULT_DIR" ]] || echo "preflight_directory=${PREFLIGHT_RESULT_DIR}"
    printf 'command='
    printf '%q ' "$COLDMD_BIN" "${command_args[@]}"
    echo
  } > "$log_file"
  nohup "$COLDMD_BIN" "${command_args[@]}" >> "$log_file" 2>&1 < /dev/null &
  local pid="$!"
  echo "$pid" > "$pid_file"
  sleep "$STARTUP_CHECK_SECONDS"
  if ! kill -0 "$pid" 2>/dev/null; then
    if wait "$pid"; then
      echo "ColdMD completed before the background startup check."
      echo "Run directory: $run_dir"
      echo "Terminal log: $log_file"
      echo "PID file: $pid_file"
      return 0
    else
      local status="$?"
      echo "coldmd exited during startup with status ${status}; inspect: $log_file" >&2
      return "$status"
    fi
  fi
  echo "ColdMD started in background."
  echo "Mode: $mode"
  echo "PID: $pid"
  echo "Run directory: $run_dir"
  echo "Terminal log: $log_file"
  echo "PID file: $pid_file"
}

[[ $# -ge 1 ]] || { usage; exit 1; }
MODE="$1"
shift

case "$MODE" in
  check)
    [[ $# -le 1 ]] || { usage; exit 1; }
    CONFIG_PATH="$(resolve_config "${1:-$CONFIG_FILE}")"
    LABEL="$(safe_stem "$CONFIG_PATH")"
    STAMP="$(date +%Y%m%d-%H%M%S)-$$"
    setup_runtime
    run_preflight "$CONFIG_PATH" "$LABEL" "$STAMP"
    ;;
  new)
    [[ $# -le 2 ]] || { usage; exit 1; }
    CONFIG_PATH="$(resolve_config "${1:-$CONFIG_FILE}")"
    LABEL="${2:-$(safe_stem "$CONFIG_PATH")}"
    [[ "$LABEL" =~ ^[A-Za-z0-9._-]+$ ]] || fail "run-label contains unsupported characters"
    setup_runtime
    STAMP="$(date +%Y%m%d-%H%M%S)-$$"
    RUN_DIR="${RUN_BASE}/${LABEL}-${STAMP}"
    [[ ! -e "$RUN_DIR" ]] || fail "Refusing to reuse output path: $RUN_DIR"
    run_preflight "$CONFIG_PATH" "$LABEL" "$STAMP"
    launch_background new "$RUN_DIR" "$STAMP" run "$CONFIG_PATH" --output-dir "$RUN_DIR"
    ;;
  fork)
    [[ $# -ge 3 && $# -le 4 ]] || { usage; exit 1; }
    CONFIG_PATH="$(resolve_config "$1")"
    [[ -d "$2" ]] || fail "Parent run directory not found: $2"
    PARENT_RUN="$(realpath "$2")"
    if [[ -f "$3" ]]; then
      STATE_PATH="$(realpath "$3")"
    elif [[ -f "${PARENT_RUN}/stage_structures/$3" ]]; then
      STATE_PATH="$(realpath "${PARENT_RUN}/stage_structures/$3")"
    else
      fail "Stage snapshot not found: $3"
    fi
    LABEL="${4:-$(safe_stem "$CONFIG_PATH")-fork}"
    [[ "$LABEL" =~ ^[A-Za-z0-9._-]+$ ]] || fail "run-label contains unsupported characters"
    setup_runtime
    STAMP="$(date +%Y%m%d-%H%M%S)-$$"
    RUN_DIR="${RUN_BASE}/${LABEL}-${STAMP}"
    [[ ! -e "$RUN_DIR" ]] || fail "Refusing to reuse output path: $RUN_DIR"
    run_fork_preflight "$CONFIG_PATH" "$PARENT_RUN" "$STATE_PATH" "$LABEL" "$STAMP"
    launch_background fork "$RUN_DIR" "$STAMP" fork "$CONFIG_PATH" \
      --from-run "$PARENT_RUN" --state "$STATE_PATH" --output-dir "$RUN_DIR"
    ;;
  resume)
    [[ $# -eq 1 ]] || { usage; exit 1; }
    [[ -d "$1" ]] || fail "Run directory not found: $1"
    RUN_DIR="$(realpath "$1")"
    [[ -f "$RUN_DIR/resolved-config.yaml" ]] || fail "Missing resolved-config.yaml in: $RUN_DIR"
    [[ -f "$RUN_DIR/checkpoints/checkpoint.json" ]] || fail "Missing checkpoint.json in: $RUN_DIR"
    setup_runtime
    if ! "$PYTHON_BIN" - "$RUN_DIR/resolved-config.yaml" <<'PY'
import sys
from pathlib import Path
import yaml

payload = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
calculator = payload.get("calculator") if isinstance(payload, dict) else None
if not isinstance(calculator, dict) or calculator.get("kind") != "mace":
    raise SystemExit("Strict resume requires calculator.kind='mace'")
model_path = calculator.get("model_path")
if not isinstance(model_path, str) or not model_path.strip():
    raise SystemExit("Strict resume requires a non-empty local model_path")
PY
    then
      fail "Strict resume requires a local, content-hashed MACE model"
    fi
    STAMP="$(date +%Y%m%d-%H%M%S)-$$"
    launch_background resume "$RUN_DIR" "$STAMP" resume "$RUN_DIR"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage
    exit 1
    ;;
esac
