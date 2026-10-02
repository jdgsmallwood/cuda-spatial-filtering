#!/usr/bin/env bash
# Fold a generated four-FPGA PCAP through the real pulsar/DADA/DSPSR path.
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
FIXTURE=${1:?usage: run_pulsar_replay.sh FIXTURE_DIR voltage|intensity [RUN_DIR]}
MODE=${2:?usage: run_pulsar_replay.sh FIXTURE_DIR voltage|intensity [RUN_DIR]}
CONSUMER=${REPLAY_CONSUMER:-dspsr}
[[ $MODE == voltage || $MODE == intensity ]] || { echo "mode must be voltage or intensity" >&2; exit 2; }
[[ $CONSUMER == dspsr || $CONSUMER == dbdisk || $CONSUMER == null ]] || { echo "invalid REPLAY_CONSUMER" >&2; exit 2; }
FIXTURE=$(cd -- "$FIXTURE" && pwd)
RUN_DIR=${3:-"${FIXTURE}/${MODE}_run"}
[[ ! -e $RUN_DIR ]] || { echo "run directory already exists: $RUN_DIR" >&2; exit 2; }
mkdir -p -- "$RUN_DIR"
RUN_DIR=$(cd -- "$RUN_DIR" && pwd)
mkfifo "$RUN_DIR/dada.keepalive"
exec {keepalive_fd}<>"$RUN_DIR/dada.keepalive"

PSRDADA_BIN=/data/BLACKMESA_1/sma075/psrdada-local/bin
TEMPO2_ROOT=/data/BLACKMESA_1/sma075/psrdada-local/tempo2
[[ -f $FIXTURE/pulsar.pcap && -f $FIXTURE/truth.json ]] || { echo "fixture incomplete" >&2; exit 2; }
readarray -t META < <(python3 - "$FIXTURE/truth.json" <<'PY'
import json, sys
x=json.load(open(sys.argv[1]))
print(x['steering_utc'])
print(x['min_channel'])
print(x['coarse_channels'])
PY
)
STEERING_UTC=${META[0]}
MIN_CHANNEL=${META[1]}
COARSE_CHANNELS=${META[2]}
[[ $COARSE_CHANNELS == 1 || $COARSE_CHANNELS == 40 ]] || {
  echo "replay supports one or 40 coarse channels" >&2; exit 2;
}
if [[ $COARSE_CHANNELS == 1 ]]; then
  BINARY="${REPO_ROOT}/build/apps/pulsar_fold_replay_${MODE}_4_1"
else
  if [[ $MODE == intensity ]]; then
    BINARY="${REPO_ROOT}/build/apps/pulsar_fold_intensity_4_40"
  else
    BINARY="${REPO_ROOT}/build/apps/pulsar_fold_4_40"
  fi
fi
[[ -x $BINARY ]] || { echo "missing binary $BINARY" >&2; exit 2; }

python3 "${SCRIPT_DIR}/validate_fine_beam_gains.py" "$FIXTURE/measured_fine_gains.json" \
  --stream-map "$REPO_ROOT/stream_antenna_map.json" --min-channel "$MIN_CHANNEL" \
  --coarse-channels "$COARSE_CHANNELS" \
  --processing-channels "$((COARSE_CHANNELS * 28))" --excluded-antennas 14,36
python3 "${SCRIPT_DIR}/generate_fine_dada_header.py" \
  --output "$RUN_DIR/pulsar.hdr" --source J0534+2200 \
  --min-coarse-channel "$MIN_CHANNEL" --coarse-channels "$COARSE_CHANNELS" \
  $( [[ $MODE == intensity ]] && echo --detected-intensity )
printf 'DADA INFO:\nkey dada\n' > "$RUN_DIR/dada.info"
if [[ $MODE == intensity ]]; then
  BLOCK_BYTES=$((512 * (COARSE_CHANNELS * 27) * 4))
  BUFFERS=800
  # One channel's 800-block ring is ~46 MB; keep the DSPSR work unit below it.
  if [[ $COARSE_CHANNELS == 40 ]]; then
    DSPSR_ARGS=(-K -d 1 -U 1100)
  else
    DSPSR_ARGS=(-K -d 1 -U 16)
  fi
  DSPSR_DEVICE_ARGS=(-cuda 0)
else
  BLOCK_BYTES=$((512 * (COARSE_CHANNELS * 27) * 16))
  BUFFERS=300
  # One detected total-intensity profile from complex voltage (PP + QQ).
  # DSPSR allocates a distinct detector output for -d 1, so this mode does
  # not depend on patches/dspsr-cuda-detection-out-of-place.patch.
  DSPSR_ARGS=(-d 1 -U 16)
  DSPSR_DEVICE_ARGS=(-cuda 0)
fi

if ipcs -m | awk '$1 == "0x0000dada" {found=1} END {exit !found}'; then
  echo "DADA key 0xdada exists; refusing to take ownership" >&2
  exit 2
fi
RING_CREATED=false
DADA_PID=""
DSPSR_PID=""
APP_PID=""
cleanup() {
  status=$?
  trap - EXIT INT TERM
  for pid in "$APP_PID" "$DSPSR_PID"; do
    if [[ -n $pid ]] && kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
  if [[ $RING_CREATED == true ]]; then
    "$PSRDADA_BIN/dada_db" -d -k dada >/dev/null 2>&1 || true
  fi
  if [[ -n $DADA_PID ]] && kill -0 "$DADA_PID" 2>/dev/null; then
    kill -TERM "$DADA_PID" 2>/dev/null || true
    wait "$DADA_PID" 2>/dev/null || true
  fi
  rm -f -- "$RUN_DIR/downstream.ready"
  exit "$status"
}
trap cleanup EXIT INT TERM

setsid "$PSRDADA_BIN/dada_db" -k dada -n "$BUFFERS" -b "$BLOCK_BYTES" -g 0 -w \
  <&$keepalive_fd >"$RUN_DIR/dada_db.log" 2>&1 &
DADA_PID=$!
for _ in $(seq 1 100); do
  if ipcs -m | awk '$1 == "0x0000dada" {found=1} END {exit !found}'; then
    RING_CREATED=true
    break
  fi
  sleep 0.1
done
[[ $RING_CREATED == true ]] || { echo "DADA ring did not start" >&2; exit 1; }
sleep 1
kill -0 "$DADA_PID" 2>/dev/null || { echo "DADA ring creator exited" >&2; exit 1; }

if [[ $CONSUMER == dspsr ]]; then
  env PATH="${PSRDADA_BIN}:${TEMPO2_ROOT}/bin:${PATH}" TEMPO2="$TEMPO2_ROOT" \
    "$PSRDADA_BIN/dspsr" "$RUN_DIR/dada.info" -A -L 10 -y \
    "${DSPSR_ARGS[@]}" -b 128 "${DSPSR_DEVICE_ARGS[@]}" -E "$FIXTURE/synthetic.par" \
    -O "$RUN_DIR/pulsar" >"$RUN_DIR/dspsr.log" 2>&1 &
elif [[ $CONSUMER == dbdisk ]]; then
  "$PSRDADA_BIN/dada_dbdisk" -k dada -D "$RUN_DIR" -s \
    >"$RUN_DIR/dbdisk.log" 2>&1 &
else
  "$PSRDADA_BIN/dada_dbnull" -k dada -z -q \
    >"$RUN_DIR/dbnull.log" 2>&1 &
fi
DSPSR_PID=$!

(
  cd -- "$RUN_DIR"
  PCAP_ARGS=()
  if [[ ${REPLAY_LOOP_PCAP:-0} == 1 ]]; then PCAP_ARGS+=(--loop); fi
  env CUDA_HOME=/data/BLACKMESA_1/sma075/miniconda3/targets/x86_64-linux \
      SPATIAL_CAPTURE_READY_FILE="$RUN_DIR/downstream.ready" \
      "$BINARY" --pcap_file "$FIXTURE/pulsar.pcap" \
      "${PCAP_ARGS[@]}" \
      --capture-backend kernel --network-interface 0,1,2,3 \
      --min_freq_channel "$MIN_CHANNEL" \
      --obs-length "${REPLAY_OBS_LENGTH:-180}" \
      --config-file "$REPO_ROOT/config.json" \
      --delay-file "$FIXTURE/zero_fpga_delays.json" \
      --gains "$FIXTURE/measured_fine_gains.json" \
      --eigenvalue-num-filename "$REPO_ROOT/build/apps/nr-signal-eigenvalues.json" \
      --stream-antenna-map "$REPO_ROOT/stream_antenna_map.json" \
      --targets-filename "$REPO_ROOT/config/crab_targets.json" \
      --steering-utc "$STEERING_UTC" --beam-excluded-antennas 14,36 \
      --dada-header "$RUN_DIR/pulsar.hdr" --apply-gains-to-vis \
      >"$RUN_DIR/pulsar_fold.log" 2>&1
) &
APP_PID=$!

ready=false
for _ in $(seq 1 1200); do
  if [[ $CONSUMER != dspsr ]] || \
     grep -q 'prepared in ' "$RUN_DIR/dspsr.log" 2>/dev/null || \
     { [[ $MODE == voltage ]] && grep -q 'blocksize=' "$RUN_DIR/dspsr.log" 2>/dev/null; }; then
    touch -- "$RUN_DIR/downstream.ready"
    ready=true
    break
  fi
  if ! kill -0 "$APP_PID" 2>/dev/null || ! kill -0 "$DSPSR_PID" 2>/dev/null; then
    break
  fi
  sleep 0.1
done
[[ $ready == true ]] || { echo "consumer did not prepare; see $RUN_DIR" >&2; exit 1; }
echo "$CONSUMER prepared; replay running ($MODE)"
wait "$APP_PID"
APP_PID=""
if [[ $CONSUMER == null ]]; then
  # dada_dbnull keeps waiting for another observation after EOD.
  kill -TERM "$DSPSR_PID" 2>/dev/null || true
  wait "$DSPSR_PID" 2>/dev/null || true
  DSPSR_PID=""
  echo "Completed $MODE replay with $CONSUMER consumer in $RUN_DIR"
  exit 0
fi
for _ in $(seq 1 600); do
  if ! kill -0 "$DSPSR_PID" 2>/dev/null; then break; fi
  sleep 0.1
done
if kill -0 "$DSPSR_PID" 2>/dev/null; then
  echo "DSPSR did not stop after EOD" >&2
  exit 1
fi
wait "$DSPSR_PID"
DSPSR_PID=""
if [[ $CONSUMER == dspsr ]]; then
  shopt -s nullglob
  archives=("$RUN_DIR"/pulsar*.ar)
  (( ${#archives[@]} > 0 )) || { echo "DSPSR produced no archive" >&2; exit 1; }
  printf 'Completed %s replay: %s\n' "$MODE" "${archives[*]}"
else
  echo "Completed $MODE replay with $CONSUMER consumer in $RUN_DIR"
fi
