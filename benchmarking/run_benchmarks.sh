#!/usr/bin/env bash
# Run the full LAMBDA benchmark suite and generate figures.
# Usage: ./benchmarking/run_benchmarks.sh [--duration SECS] [--output-dir DIR]
#
# ── PERSISTENCE ACROSS CONTAINER VERSIONS ────────────────────────────────────
#
# What persists automatically (lives in the git-tracked workspace):
#   • Source changes: cmake/find_cutensor.cmake, tests/CMakeLists.txt, this script
#
# What must be (re)done in each fresh container:
#   • Build: run ./build.sh from the repo root — it documents and scripts the
#     full cmake configure + make with the required flags (see build.sh).
#
# Environment variables required at runtime (set in apptainer.def %environment;
# this script exports them itself as a safety net, but they should already be
# set by the container):
#
#   CUTENSOR_ROOT=/opt/conda
#     cmake's find_cutensor.cmake looks for ${CUTENSOR_ROOT}/lib/libcutensor.so.
#     The apptainer.def used to set this to /usr/lib/x86_64-linux-gnu/libcutensor
#     (wrong — version-numbered subdirs, no flat layout).  Fixed in apptainer.def.
#
#   CUDA_HOME=/opt/conda/targets/x86_64-linux
#     Required so that NVRTC can find cooperative_groups/*.h at runtime.
#     Any binary that constructs a tcc::Correlator (e.g. bench_gpu, observe)
#     will fail with NVRTC_ERROR_COMPILATION without this.
#
#   LD_LIBRARY_PATH must include /opt/conda/lib (before system paths)
#     conda's libibverbs / libnl-route-3 depend on conda's libnl-3, which
#     exports versioned @libnl_3 symbols that the system libnl-3 lacks.
#
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$REPO_ROOT/build"

DURATION=30
OUTPUT_DIR="$SCRIPT_DIR/$(date +%Y%m%d_%H%M%S)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --duration) DURATION="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

mkdir -p "$OUTPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Duration per config: ${DURATION}s"

export CUDA_HOME=/opt/conda/targets/x86_64-linux
export LD_LIBRARY_PATH="/opt/conda/lib:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

APPS="$BUILD_DIR/apps"

check_binary() {
  if [[ ! -x "$1" ]]; then
    echo "ERROR: binary not found or not executable: $1" >&2
    echo "       Run cmake --build build from the repo root first." >&2
    exit 1
  fi
}

check_binary "$APPS/bench_gpu"
check_binary "$APPS/bench_processor"
check_binary "$APPS/bench_writers"

SCRATCH="$OUTPUT_DIR/scratch_writers"
mkdir -p "$SCRATCH"

run_bench() {
  local name="$1"; shift
  local outfile="$OUTPUT_DIR/${name}.txt"
  echo "=== $name ===" | tee "$outfile"
  if "$@" 2>&1 | tee -a "$outfile"; then
    echo "[$name] done — output: $outfile"
  else
    echo "[$name] FAILED (exit $?) — see $outfile" >&2
  fi
  echo
}

echo "Starting benchmark suite..."
echo "Git commit: $(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo

run_bench "bench_processor" \
  "$APPS/bench_processor" --duration "$DURATION"
run_bench "bench_gpu" \
  "$APPS/bench_gpu" --duration "$DURATION" --lambda-only


run_bench "bench_writers" \
  "$APPS/bench_writers" --duration "$DURATION" -o "$SCRATCH"

echo "All benchmarks complete."
echo

PLOT_SCRIPT="$SCRIPT_DIR/plot_benchmarks.py"
if [[ -x "$PLOT_SCRIPT" ]] || [[ -f "$PLOT_SCRIPT" ]]; then
  echo "Generating figures..."
  PYTHON="${PYTHON:-python3}"
  if "$PYTHON" "$PLOT_SCRIPT" --results-dir "$OUTPUT_DIR"; then
    echo "Figures saved to $OUTPUT_DIR"
  else
    echo "Plotting failed — results are still in $OUTPUT_DIR" >&2
  fi
else
  echo "plot_benchmarks.py not found at $PLOT_SCRIPT — skipping figures"
fi

echo "Done: $OUTPUT_DIR"
