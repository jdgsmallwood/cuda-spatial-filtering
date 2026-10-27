#!/usr/bin/env bash
# Human-friendly Starspin launcher for one to four FPGAs.
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
BUILD_APPS="${REPO_ROOT}/build/apps"

CHANNELS=40
FPGAS=4
MIN_CHANNEL=176
DURATION=60
INTERFACES=""
CAPTURE_CPUS=""
WORKER_CPUS=""
PIPELINE_CPUS=""
RING_NUMA_OVERRIDE=""
CUDA_ROOT="${CUDA_HOME:-/usr/local/cuda}"
OUTPUT_DIR="${REPO_ROOT}/build/starspin_runs"
OUTPUT_NAME=""
BINARY=""
CAPTURE_BACKEND="ibverbs"
DELAY_FILE=""
DELAY_FILE_EXPLICIT=false
AUTO_SINGLE_FPGA_DELAY=false
GAINS_FILE="${BUILD_APPS}/weights.json"
EIGENVALUE_FILE="${BUILD_APPS}/nr-signal-eigenvalues.json"
CONFIG_FILE=""
STREAM_MAP=""
USE_STREAM_MAP=true
APPLY_GAINS=false
CONFIGURE_FPGA=false
DRY_RUN=false
EXTRA_ARGS=()
CONFIGURE_SCRIPT=""
# This is a trusted, operator-owned shell profile, kept outside the source tree.
if [[ -n "${STARSPIN_SITE_PROFILE:-}" ]]; then
  source "$STARSPIN_SITE_PROFILE"
fi

usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS] [-- STARSPIN_OPTIONS...]

Run Starspin with an external STARSPIN_SITE_PROFILE and boot-matched delays.

Defaults:
  channels              40 (binary starspin_4_40)
  FPGA sources          4 (the first four listed interfaces)
  channel range         176-215
  duration              60 seconds
  backend               ${CAPTURE_BACKEND}, direct into the packet ring
  interfaces            ${INTERFACES}
  capture CPUs          ${CAPTURE_CPUS}
  worker CPUs           ${WORKER_CPUS}
  pipeline CPUs         ${PIPELINE_CPUS}
  output directory      ${OUTPUT_DIR}

Options:
  -n, --channels N              Even coarse channel count (8..330); binary must be built
      --fpgas N                 Number of FPGA sources (1..4)
  -f, --min-channel N           Lowest frequency-channel number
  -t, --duration SECONDS        Observation length (0 means run until stopped)
  -o, --output-dir PATH         HDF5, log, streams CSV, and app.log directory
  -p, --name NAME               Output basename, without .hdf5/.log
      --binary PATH             Override the selected Starspin executable
      --capture-backend NAME    kernel or ibverbs (default: ibverbs)
      --capture-cpus LIST       Capture-thread CPU list
      --worker-cpus LIST        Packet-worker CPU list
      --pipeline-cpus LIST      Coordinator, GPU feeder, and writer CPU set
      --ring-numa-nodes LIST    Ring memory NUMA node per selected NIC
      --interfaces LIST         Comma-separated NIC list
      --cuda-home PATH          CUDA toolkit root used by runtime compilation
      --delay-file PATH         FPGA delay JSON (automatic zero for one FPGA)
      --gains-file PATH         Gain/weight JSON
      --eigenvalue-file PATH    Eigenvalue-count JSON
      --config-file PATH        Starspin configuration JSON
      --stream-map PATH         Hardware-stream to antenna mapping JSON
      --no-stream-map           Use Starspin's built-in antenna mapping
      --apply-gains             Apply gains to visibilities
      --configure-fpga          Explicitly tune selected FPGAs before capture
      --dry-run                 Validate and print the command without running
  -h, --help                    Show this help

Examples:
  $(basename "$0")
  $(basename "$0") --duration 600 --name science_40ch
  $(basename "$0") --configure-fpga --duration 600

Arguments after -- are passed directly to Starspin.
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
    --fpgas)
      require_value "$@"; FPGAS=$2; shift 2 ;;
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
    --capture-backend)
      require_value "$@"; CAPTURE_BACKEND=$2; shift 2 ;;
    --capture-cpus)
      require_value "$@"; CAPTURE_CPUS=$2; shift 2 ;;
    --worker-cpus)
      require_value "$@"; WORKER_CPUS=$2; shift 2 ;;
    --pipeline-cpus)
      require_value "$@"; PIPELINE_CPUS=$2; shift 2 ;;
    --ring-numa-nodes)
      require_value "$@"; RING_NUMA_OVERRIDE=$2; shift 2 ;;
    --interfaces)
      require_value "$@"; INTERFACES=$2; shift 2 ;;
    --cuda-home)
      require_value "$@"; CUDA_ROOT=$2; shift 2 ;;
    --delay-file)
      require_value "$@"; DELAY_FILE=$2; DELAY_FILE_EXPLICIT=true; shift 2 ;;
    --gains-file)
      require_value "$@"; GAINS_FILE=$2; shift 2 ;;
    --eigenvalue-file)
      require_value "$@"; EIGENVALUE_FILE=$2; shift 2 ;;
    --config-file)
      require_value "$@"; CONFIG_FILE=$2; shift 2 ;;
    --stream-map)
      require_value "$@"; STREAM_MAP=$2; USE_STREAM_MAP=true; shift 2 ;;
    --no-stream-map)
      USE_STREAM_MAP=false; shift ;;
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
      die "unknown option '$1' (use --help; use -- before raw Starspin options)" ;;
  esac
done

for setting in INTERFACES CAPTURE_CPUS WORKER_CPUS PIPELINE_CPUS CONFIG_FILE; do
  [[ -n "${!setting}" ]] || die "set $setting in STARSPIN_SITE_PROFILE or its command-line override"
done

[[ "$CHANNELS" =~ ^[0-9]+$ ]] || die "--channels must be an even integer from 8 to 330"
(( CHANNELS >= 8 && CHANNELS <= 330 && CHANNELS % 2 == 0 )) ||
  die "--channels must be an even integer from 8 to 330"
[[ "$FPGAS" =~ ^[1-4]$ ]] || die "--fpgas must be 1, 2, 3, or 4"
if [[ "$FPGAS" == 1 && "$DELAY_FILE_EXPLICIT" == false ]]; then
  AUTO_SINGLE_FPGA_DELAY=true
fi
[[ "$CAPTURE_BACKEND" == kernel || "$CAPTURE_BACKEND" == ibverbs ]] ||
  die "--capture-backend must be kernel or ibverbs"
[[ "$MIN_CHANNEL" =~ ^[0-9]+$ ]] || die "--min-channel must be a non-negative integer"
(( MIN_CHANNEL + CHANNELS <= 512 )) || die "channel window must end at or below 511"
[[ "$DURATION" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "--duration must be non-negative"
command -v taskset >/dev/null || die "taskset is required for pipeline CPU placement"
taskset -c "$PIPELINE_CPUS" true >/dev/null || die "invalid or unavailable --pipeline-cpus: $PIPELINE_CPUS"
if [[ -z "$BINARY" ]]; then
  BINARY="${BUILD_APPS}/starspin_${FPGAS}_${CHANNELS}"
fi

[[ -x "$BINARY" ]] || die "Starspin binary is missing or not executable: $BINARY (configure with -DSTARSPIN_CHANNEL_COUNTS='${CHANNELS}' and build target starspin_${FPGAS}_${CHANNELS})"
BINARY=$(realpath -- "$BINARY")
[[ -d "$CUDA_ROOT" ]] || die "CUDA toolkit directory does not exist: $CUDA_ROOT"
for input in "$GAINS_FILE" "$EIGENVALUE_FILE" "$CONFIG_FILE"; do
  [[ -r "$input" ]] || die "required input is not readable: $input"
done
if [[ "$AUTO_SINGLE_FPGA_DELAY" == false ]]; then
  [[ -r "$DELAY_FILE" ]] || die "required input is not readable: $DELAY_FILE"
  DELAY_FILE=$(realpath -- "$DELAY_FILE")
fi
if [[ "$USE_STREAM_MAP" == true ]]; then
  [[ -r "$STREAM_MAP" ]] || die "required input is not readable: $STREAM_MAP"
fi
# Capture runs from its output directory; preserve caller-relative inputs.
GAINS_FILE=$(realpath -- "$GAINS_FILE")
EIGENVALUE_FILE=$(realpath -- "$EIGENVALUE_FILE")
CONFIG_FILE=$(realpath -- "$CONFIG_FILE")
if [[ "$USE_STREAM_MAP" == true ]]; then
  STREAM_MAP=$(realpath -- "$STREAM_MAP")
fi

IFS=',' read -r -a NIC_ARRAY <<< "$INTERFACES"
(( ${#NIC_ARRAY[@]} >= FPGAS )) || die "need at least $FPGAS interfaces"
NIC_ARRAY=("${NIC_ARRAY[@]:0:FPGAS}")
INTERFACES=$(IFS=,; printf '%s' "${NIC_ARRAY[*]}")
IFS=',' read -r -a CAPTURE_CPU_ARRAY <<< "$CAPTURE_CPUS"
(( ${#CAPTURE_CPU_ARRAY[@]} >= FPGAS )) || die "need at least $FPGAS capture CPUs"
CAPTURE_CPUS=$(IFS=,; printf '%s' "${CAPTURE_CPU_ARRAY[*]:0:FPGAS}")
for nic in "${NIC_ARRAY[@]}"; do
  [[ -d "/sys/class/net/$nic" ]] || die "network interface does not exist: $nic"
done
RING_NUMA_NODES=""
for nic in "${NIC_ARRAY[@]}"; do
  node=$(cat "/sys/class/net/$nic/device/numa_node")
  [[ "$node" =~ ^[0-9]+$ ]] || die "cannot determine NUMA node for $nic"
  RING_NUMA_NODES+="${RING_NUMA_NODES:+,}${node}"
done
if [[ -n "$RING_NUMA_OVERRIDE" ]]; then
  IFS=',' read -r -a RING_NODE_ARRAY <<< "$RING_NUMA_OVERRIDE"
  (( ${#RING_NODE_ARRAY[@]} == FPGAS )) ||
    die "--ring-numa-nodes needs one node per selected NIC"
  for node in "${RING_NODE_ARRAY[@]}"; do
    [[ "$node" =~ ^[0-9]+$ && -d "/sys/devices/system/node/node$node" ]] ||
      die "invalid ring NUMA node: $node"
  done
  RING_NUMA_NODES=$(IFS=,; printf '%s' "${RING_NODE_ARRAY[*]}")
fi

GETCAP=/usr/sbin/getcap
if [[ -x "$GETCAP" ]]; then
  CAPABILITIES=$($GETCAP "$BINARY" 2>/dev/null || true)
  if [[ "$CAPABILITIES" != *cap_net_raw* ]]; then
    if [[ "$DRY_RUN" == true ]]; then
      printf 'WARNING: %s does not have cap_net_raw. Before a real run:\n' "$BINARY" >&2
      printf '  sudo setcap cap_net_raw=ep %q\n' "$BINARY" >&2
    else
      die "$BINARY needs cap_net_raw; run: sudo setcap cap_net_raw=ep '$BINARY'"
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
  OUTPUT_NAME="$(date +'%Y-%m-%dT%H%M%S')_starspin_${FPGAS}_${CHANNELS}_ch${MIN_CHANNEL}-${MAX_CHANNEL}_${SAFE_DURATION}s"
fi
[[ "$OUTPUT_NAME" != */* ]] || die "--name must be a basename, not a path"

HDF5_FILE="${OUTPUT_DIR}/${OUTPUT_NAME}.hdf5"
LOG_FILE="${OUTPUT_DIR}/${OUTPUT_NAME}.log"
[[ ! -e "$HDF5_FILE" ]] || die "refusing to overwrite existing output: $HDF5_FILE"
[[ ! -e "$LOG_FILE" ]] || die "refusing to overwrite existing log: $LOG_FILE"
if [[ "$AUTO_SINGLE_FPGA_DELAY" == true ]]; then
  # The app parses a JSON delay file even though one source needs no relative delay.
  DELAY_FILE="${OUTPUT_DIR}/${OUTPUT_NAME}.zero_delays.json"
  if [[ "$DRY_RUN" == false ]]; then
    printf '{}\n' > "$DELAY_FILE"
  fi
fi

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
  printf 'WARNING: 48 channels completed a 120-second live run with sparse packet loss; monitor Missing during longer captures.\n' >&2
elif [[ "$CHANNELS" == 54 ]]; then
  printf 'WARNING: 54 channels is experimental; monitor packet loss and GPU buffer returns.\n' >&2
fi

if [[ "$CONFIGURE_FPGA" == true ]]; then
  [[ -n "$CONFIGURE_SCRIPT" ]] || die "set CONFIGURE_SCRIPT in the external site profile"
  [[ -x "$CONFIGURE_SCRIPT" ]] || die "FPGA configuration script is unavailable: $CONFIGURE_SCRIPT"
  CENTRE_CHANNEL=$((MIN_CHANNEL + CHANNELS / 2))
  FPGA_ID_LIST=$(seq -s, 0 $((FPGAS - 1)))
  printf 'Explicitly tuning FPGAs %s to channels %d-%d...\n' "$FPGA_ID_LIST" "$MIN_CHANNEL" "$MAX_CHANNEL"
  if [[ "$DRY_RUN" == true ]]; then
    printf '  (cd %q && %q -a %q -n %q -f %q -m tune)\n' \
      "$(dirname "$CONFIGURE_SCRIPT")" "$CONFIGURE_SCRIPT" "$FPGA_ID_LIST" "$CHANNELS" "$CENTRE_CHANNEL"
  else
    (
      cd -- "$(dirname "$CONFIGURE_SCRIPT")"
      "$CONFIGURE_SCRIPT" -a "$FPGA_ID_LIST" -n "$CHANNELS" -f "$CENTRE_CHANNEL" -m tune
    )
  fi
fi

COMMAND=(
  "$BINARY"
  --capture-backend "$CAPTURE_BACKEND"
  --network-interface "$INTERFACES"
  --min_freq_channel "$MIN_CHANNEL"
  --obs-length "$DURATION"
  --config-file "$CONFIG_FILE"
  --delay-file "$DELAY_FILE"
  --gains "$GAINS_FILE"
  --eigenvalue-num-filename "$EIGENVALUE_FILE"
  --vis_output_file "$HDF5_FILE"
)
if [[ "$USE_STREAM_MAP" == true ]]; then
  COMMAND+=(--stream-antenna-map "$STREAM_MAP")
fi
if [[ "$APPLY_GAINS" == true ]]; then
  COMMAND+=(--apply-gains-to-vis)
fi
COMMAND+=("${EXTRA_ARGS[@]}")
COMMAND=(taskset -c "$PIPELINE_CPUS" "${COMMAND[@]}")

printf '%s\n' '============================================================'
printf 'Starspin %s-FPGA capture\n' "$FPGAS"
printf 'Binary:          %s\n' "$BINARY"
printf 'Channels:        %d-%d (%d)\n' "$MIN_CHANNEL" "$MAX_CHANNEL" "$CHANNELS"
printf 'Delay file:      %s\n' "$DELAY_FILE"
printf 'Duration:        %s seconds\n' "$DURATION"
printf 'Interfaces:      %s\n' "$INTERFACES"
printf 'Capture CPUs:    %s\n' "$CAPTURE_CPUS"
printf 'Worker CPUs:     %s\n' "$WORKER_CPUS"
printf 'Pipeline CPUs:   %s\n' "$PIPELINE_CPUS"
printf 'Ring NUMA nodes: %s\n' "$RING_NUMA_NODES"
printf 'CUDA_HOME:       %s\n' "$CUDA_ROOT"
printf 'HDF5 output:     %s\n' "$HDF5_FILE"
printf 'Console log:     %s\n' "$LOG_FILE"
printf 'Free disk:       %d GiB\n' "$((FREE_KIB / 1048576))"
printf 'Apply gains:     %s\n' "$APPLY_GAINS"
printf 'Command:'
printf ' %q' env "CUDA_HOME=$CUDA_ROOT" "SPATIAL_CAPTURE_CPUS=$CAPTURE_CPUS" \
  "SPATIAL_WORKER_CPUS=$WORKER_CPUS" "SPATIAL_RING_NUMA_NODES=$RING_NUMA_NODES" \
  "SPATIAL_NUMA_NODE=-1" \
  "${COMMAND[@]}"
printf '\n%s\n' '============================================================'

if [[ "$DRY_RUN" == true ]]; then
  exit 0
fi

export CUDA_HOME="$CUDA_ROOT"
export SPATIAL_CAPTURE_CPUS="$CAPTURE_CPUS"
export SPATIAL_WORKER_CPUS="$WORKER_CPUS"
export SPATIAL_RING_NUMA_NODES="$RING_NUMA_NODES"
# Keep the inherited CPU set for the coordinator and GPU feeder; the capture
# and packet workers explicitly pin themselves to their requested CPUs.
export SPATIAL_NUMA_NODE=-1

# Put app.log beside the observation artifacts. pipefail preserves Starspin's
# exit status through tee, while the foreground process group keeps Ctrl-C
# behaviour intuitive for an operator.
cd -- "$OUTPUT_DIR"
"${COMMAND[@]}" 2>&1 | tee "$LOG_FILE"
