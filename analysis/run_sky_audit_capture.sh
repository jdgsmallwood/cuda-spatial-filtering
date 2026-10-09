#!/usr/bin/env bash
# Same-tuning 40-channel sky control for the 2026-10-04 delay audit.
# The FPGA state and six-pair packet delay closure must be checked beforehand.
set -Eeuo pipefail

if [[ $# != 2 ]]; then
  printf 'Usage: %s NAME DURATION_SECONDS\n' "$0" >&2
  exit 2
fi

name=$1
duration=$2
[[ "$name" =~ ^[A-Za-z0-9_-]+$ ]] || { printf 'Invalid capture name\n' >&2; exit 2; }
[[ "$duration" =~ ^[0-9]+$ ]] || { printf 'Invalid duration\n' >&2; exit 2; }

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
out="$repo/build/starspin_runs/night_delay_audit_20261004"
vis="$out/$name.hdf5"
log="$out/$name.log"
[[ ! -e "$vis" && ! -e "$log" ]] || { printf 'Refusing to overwrite capture\n' >&2; exit 2; }
binary="$repo/build/apps/starspin_4_40"
delay="$out/delays_40ch_400_after_fpga2_reboot.json"
[[ -x "$binary" && -r "$delay" ]] || { printf 'Missing binary or delay file\n' >&2; exit 2; }
[[ "$(/usr/sbin/getcap "$binary")" == *cap_net_raw* ]] || {
  printf 'The rebuilt Starspin binary needs cap_net_raw=ep for libibverbs\n' >&2
  exit 2
}

export CUDA_HOME=/data/BLACKMESA_1/sma075/miniconda3/targets/x86_64-linux
export SPATIAL_CAPTURE_CPUS=1,11,13,15
export SPATIAL_WORKER_CPUS=3,4,5,6
export SPATIAL_RING_NUMA_NODES=0,1,1,1
export SPATIAL_NUMA_NODE=-1

printf 'Capture %s started %s UTC\n' "$name" "$(date -u +%FT%TZ)" | tee "$log"
printf 'Binary: %s\n' "$binary" | tee -a "$log"
printf 'Delay file: %s\n' "$delay" | tee -a "$log"
taskset -c 7-9 "$binary" \
  --capture-backend ibverbs \
  --network-interface enp94s0np0,enp134s0np0,enp175s0np0,enp216s0np0 \
  --min_freq_channel 400 --obs-length "$duration" \
  --config-file "$repo/config.json" --delay-file "$delay" \
  --gains "$repo/build/apps/weights.json" \
  --eigenvalue-num-filename "$repo/build/apps/nr-signal-eigenvalues.json" \
  --stream-antenna-map "$repo/stream_antenna_map.json" \
  --vis_output_file "$vis" --accumulation-length 56 \
  --deripple-config /data/BLACKMESA_1/sma075/rfi-monitor/calibration/20261003_1755utc/starspin_empirical_fine_deripple.json \
  2>&1 | tee -a "$log"
printf 'Capture %s finished %s UTC\n' "$name" "$(date -u +%FT%TZ)" | tee -a "$log"
