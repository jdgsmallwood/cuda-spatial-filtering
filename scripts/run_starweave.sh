#!/usr/bin/env bash
# Human-friendly four-FPGA Starweave launcher.
#
# Adapted from /data/BLACKMESA_0/lambda/scripts/correlate_lambda.sh, but kept
# in this repository deliberately: the commissioning scripts are an external
# operational dependency and must not be edited for Starweave experiments.

set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
BUILD_APPS="${REPO_ROOT}/build/apps"

CHANNELS=40
MIN_CHANNEL=176
DURATION=60
INTERFACES="enp94s0np0,enp134s0np0,enp175s0np0,enp216s0np0"
CAPTURE_CPUS="1,11,13,15"
WORKER_CPUS="3,4,5"
CUDA_ROOT="/data/BLACKMESA_1/sma075/miniconda3/targets/x86_64-linux"
OUTPUT_DIR="${REPO_ROOT}/build/starweave_runs"
OUTPUT_NAME=""
BINARY=""
DELAY_FILE="${BUILD_APPS}/alveo_delays.json"
GAINS_FILE="${BUILD_APPS}/weights.json"
EIGENVALUE_FILE="${BUILD_APPS}/nr-signal-eigenvalues.json"
CONFIG_FILE="${BUILD_APPS}/config.json"
STREAM_MAP="${REPO_ROOT}/stream_antenna_map.json"
DIAGNOSTIC=false
APPLY_GAINS=false
CONFIGURE_FPGA=false
DRY_RUN=false
EXTRA_ARGS=()

usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS] [-- STARWEAVE_OPTIONS...]

Run the four-FPGA Starweave correlator with safe 40-channel defaults.

Defaults:
  channels              40 (binary starweave_4_40)
  channel range         176-215
  duration              60 seconds
  backend               ibverbs, direct into the packet ring
  interfaces            ${INTERFACES}
  capture CPUs          ${CAPTURE_CPUS}
  worker CPUs           ${WORKER_CPUS}
  output directory      ${OUTPUT_DIR}

Options:
  -n, --channels N              Compiled channel count (8,16,24,32,40,48)
  -f, --min-channel N           Lowest frequency-channel number
  -t, --duration SECONDS        Observation length (0 means run until stopped)
  -o, --output-dir PATH         HDF5, log, streams CSV, and app.log directory
  -p, --name NAME               Output basename, without .hdf5/.log
      --binary PATH             Override the selected Starweave executable
      --diagnostic              Use starweave_4_N.diagnostic
      --capture-cpus LIST       Capture-thread CPU list
      --worker-cpus LIST        Packet-worker CPU list
      --interfaces LIST         Comma-separated NIC list
      --cuda-home PATH          CUDA toolkit root used by runtime compilation
      --delay-file PATH         FPGA delay JSON
      --gains-file PATH         Gain/weight JSON
      --config-file PATH        Starweave configuration JSON
      --stream-map PATH         Hardware-stream to antenna mapping JSON
      --apply-gains             Apply gains to visibilities
      --configure-fpga          Explicitly tune FPGAs 0-3 before capture
      --dry-run                 Validate and print the command without running
  -h, --help                    Show this help

Examples:
  $(basename "$0")
  $(basename "$0") --duration 600 --name science_40ch
  $(basename "$0") --diagnostic --duration 120
  $(basename "$0") --configure-fpga --duration 600

Arguments after -- are passed directly to Starweave.
EOF
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

require_value() {
  [[ $# -ge 2 ]] || die "option $1 requires a value"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--channels)
      require_value "$@"; CHANNELS=$2; shift 2 ;;
    -f|--min-channel)
      require_value "$@"; MIN_CHANNEL=$2; shift 2 ;;
    -t|--duration)
      require_value "$@"; DURATION=$2; shift 2 ;;
    -o|--output-dir)
      require_value "$@"; OUTPUT_DIR=$2; shift 2 ;;
    -p|--name)
      require_value "$@"; OUTPUT_NAME=${2%.hdf5}; shift 2 ;;
    --binary)
      require_value "$@"; BINARY=$2; shift 2 ;;
    --diagnostic)
      DIAGNOSTIC=true; shift ;;
    --capture-cpus)
      require_value "$@"; CAPTURE_CPUS=$2; shift 2 ;;
    --worker-cpus)
      require_value "$@"; WORKER_CPUS=$2; shift 2 ;;
    --interfaces)
      require_value "$@"; INTERFACES=$2; shift 2 ;;
    --cuda-home)
      require_value "$@"; CUDA_ROOT=$2; shift 2 ;;
    --delay-file)
      require_value "$@"; DELAY_FILE=$2; shift 2 ;;
    --gains-file)
      require_value "$@"; GAINS_FILE=$2; shift 2 ;;
    --config-file)
      require_value "$@"; CONFIG_FILE=$2; shift 2 ;;
    --stream-map)
      require_value "$@"; STREAM_MAP=$2; shift 2 ;;
    --apply-gains)
      APPLY_GAINS=true; shift ;;
    --configure-fpga)
      CONFIGURE_FPGA=true; shift ;;
    --dry-run)
      DRY_RUN=true; shift ;;
    -h|--help)
      usage; exit 0 ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break ;;
    *)
      die "unknown option '$1' (use --help; use -- before raw Starweave options)" ;;
  esac
done

[[ "$CHANNELS" =~ ^(8|16|24|32|40|48)$ ]] ||
  die "--channels must be one of 8,16,24,32,40,48"
[[ "$MIN_CHANNEL" =~ ^[0-9]+$ ]] || die "--min-channel must be a non-negative integer"
[[ "$DURATION" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "--duration must be non-negative"

if [[ -z "$BINARY" ]]; then
  BINARY="${BUILD_APPS}/starweave_4_${CHANNELS}"
  if [[ "$DIAGNOSTIC" == true ]]; then
    BINARY+=".diagnostic"
  fi
elif [[ "$DIAGNOSTIC" == true ]]; then
  die "use either --binary or --diagnostic, not both"
fi

[[ -x "$BINARY" ]] || die "Starweave binary is missing or not executable: $BINARY"
[[ -d "$CUDA_ROOT" ]] || die "CUDA toolkit directory does not exist: $CUDA_ROOT"
for input in "$DELAY_FILE" "$GAINS_FILE" "$EIGENVALUE_FILE" "$CONFIG_FILE" "$STREAM_MAP"; do
  [[ -r "$input" ]] || die "required input is not readable: $input"
done

IFS=',' read -r -a NIC_ARRAY <<< "$INTERFACES"
[[ ${#NIC_ARRAY[@]} -eq 4 ]] || die "the four-FPGA launcher requires exactly four interfaces"
for nic in "${NIC_ARRAY[@]}"; do
  [[ -d "/sys/class/net/$nic" ]] || die "network interface does not exist: $nic"
done

GETCAP=/usr/sbin/getcap
if [[ -x "$GETCAP" ]]; then
  CAPABILITIES=$($GETCAP "$BINARY" 2>/dev/null || true)
  if [[ "$CAPABILITIES" != *cap_net_raw* ]]; then
    if [[ "$DRY_RUN" == true ]]; then
      printf 'WARNING: %s does not have cap_net_raw. Before a real run:\n' "$BINARY" >&2
      printf '  sudo setcap cap_net_raw+ep %q\n' "$BINARY" >&2
    else
      die "$BINARY needs cap_net_raw; run: sudo setcap cap_net_raw+ep '$BINARY'"
    fi
  fi
else
  printf 'WARNING: getcap is unavailable; unable to verify cap_net_raw on %s\n' "$BINARY" >&2
fi

mkdir -p -- "$OUTPUT_DIR"
OUTPUT_DIR=$(cd -- "$OUTPUT_DIR" && pwd)
MAX_CHANNEL=$((MIN_CHANNEL + CHANNELS - 1))
if [[ -z "$OUTPUT_NAME" ]]; then
  SAFE_DURATION=${DURATION//./p}
  OUTPUT_NAME="$(date +'%Y-%m-%dT%H%M%S')_starweave_4_${CHANNELS}_ch${MIN_CHANNEL}-${MAX_CHANNEL}_${SAFE_DURATION}s"
fi
[[ "$OUTPUT_NAME" != */* ]] || die "--name must be a basename, not a path"

HDF5_FILE="${OUTPUT_DIR}/${OUTPUT_NAME}.hdf5"
LOG_FILE="${OUTPUT_DIR}/${OUTPUT_NAME}.log"
[[ ! -e "$HDF5_FILE" ]] || die "refusing to overwrite existing output: $HDF5_FILE"
[[ ! -e "$LOG_FILE" ]] || die "refusing to overwrite existing log: $LOG_FILE"

# The measured 40-channel writer produced about 20 GiB in 600 seconds. Add a
# 25% margin and 2 GiB working reserve. For indefinite runs, only print free
# space because there is no meaningful required-size estimate.
FREE_KIB=$(df -Pk "$OUTPUT_DIR" | awk 'NR==2 {print $4}')
if [[ "$DURATION" != 0 ]]; then
  REQUIRED_KIB=$(awk -v seconds="$DURATION" 'BEGIN {printf "%.0f", seconds * 34953 * 1.25 + 2097152}')
  if (( FREE_KIB < REQUIRED_KIB )); then
    die "insufficient output space: need approximately $((REQUIRED_KIB / 1048576)) GiB including reserve, have $((FREE_KIB / 1048576)) GiB"
  fi
fi

if [[ "$CHANNELS" == 48 ]]; then
  printf 'WARNING: 48 channels is not production-safe on this GPU; sustained GPU throughput is below the input rate.\n' >&2
fi

if [[ "$CONFIGURE_FPGA" == true ]]; then
  CONFIGURE_SCRIPT="/data/BLACKMESA_0/lambda/scripts/configure-fpga.sh"
  [[ -x "$CONFIGURE_SCRIPT" ]] || die "FPGA configuration script is unavailable: $CONFIGURE_SCRIPT"
  CENTRE_CHANNEL=$((MIN_CHANNEL + CHANNELS / 2))
  printf 'Explicitly tuning FPGAs 0-3 to channels %d-%d...\n' "$MIN_CHANNEL" "$MAX_CHANNEL"
  if [[ "$DRY_RUN" == true ]]; then
    printf '  (cd %q && %q -a 0,1,2,3 -n %q -f %q -m tune)\n' \
      "$(dirname "$CONFIGURE_SCRIPT")" "$CONFIGURE_SCRIPT" "$CHANNELS" "$CENTRE_CHANNEL"
  else
    (
      cd -- "$(dirname "$CONFIGURE_SCRIPT")"
      "$CONFIGURE_SCRIPT" -a 0,1,2,3 -n "$CHANNELS" -f "$CENTRE_CHANNEL" -m tune
    )
  fi
fi

COMMAND=(
  "$BINARY"
  --capture-backend ibverbs
  --network-interface "$INTERFACES"
  --min_freq_channel "$MIN_CHANNEL"
  --obs-length "$DURATION"
  --config-file "$CONFIG_FILE"
  --delay-file "$DELAY_FILE"
  --gains "$GAINS_FILE"
  --eigenvalue-num-filename "$EIGENVALUE_FILE"
  --stream-antenna-map "$STREAM_MAP"
  --vis_output_file "$HDF5_FILE"
)
if [[ "$APPLY_GAINS" == true ]]; then
  COMMAND+=(--apply-gains-to-vis)
fi
COMMAND+=("${EXTRA_ARGS[@]}")

printf '%s\n' '============================================================'
printf 'Starweave four-FPGA capture\n'
printf 'Binary:          %s\n' "$BINARY"
printf 'Channels:        %d-%d (%d)\n' "$MIN_CHANNEL" "$MAX_CHANNEL" "$CHANNELS"
printf 'Duration:        %s seconds\n' "$DURATION"
printf 'Interfaces:      %s\n' "$INTERFACES"
printf 'Capture CPUs:    %s\n' "$CAPTURE_CPUS"
printf 'Worker CPUs:     %s\n' "$WORKER_CPUS"
printf 'CUDA_HOME:       %s\n' "$CUDA_ROOT"
printf 'HDF5 output:     %s\n' "$HDF5_FILE"
printf 'Console log:     %s\n' "$LOG_FILE"
printf 'Free disk:       %d GiB\n' "$((FREE_KIB / 1048576))"
printf 'Apply gains:     %s\n' "$APPLY_GAINS"
printf 'Command:'
printf ' %q' env "CUDA_HOME=$CUDA_ROOT" "SPATIAL_CAPTURE_CPUS=$CAPTURE_CPUS" \
  "SPATIAL_WORKER_CPUS=$WORKER_CPUS" "${COMMAND[@]}"
printf '\n%s\n' '============================================================'

if [[ "$DRY_RUN" == true ]]; then
  exit 0
fi

export CUDA_HOME="$CUDA_ROOT"
export SPATIAL_CAPTURE_CPUS="$CAPTURE_CPUS"
export SPATIAL_WORKER_CPUS="$WORKER_CPUS"

# Put app.log beside the observation artifacts. pipefail preserves Starweave's
# exit status through tee, while the foreground process group keeps Ctrl-C
# behaviour intuitive for an operator.
cd -- "$OUTPUT_DIR"
"${COMMAND[@]}" 2>&1 | tee "$LOG_FILE"
