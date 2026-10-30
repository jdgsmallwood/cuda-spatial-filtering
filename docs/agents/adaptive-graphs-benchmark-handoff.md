# Coarse/fine CUDA Graphs benchmark handoff

## Goal

Measure eager versus CUDA Graph execution for coarse and fine decomposition,
with rank-one protection off/on, using the same build, inputs and buffer counts.
The earlier graph result applied only to weights-only; do not reuse it as a
coarse/fine graph measurement. This change is opt-in for adaptive modes.

Keep deployment identities, raw artifacts, calibration files and machine paths
local. Nothing in this handoff authorizes live observing or FPGA changes. For
later live tests, follow AGENTS.md and the original production handoff's boot,
tuning, delay-calibration and closure requirements.

## Using the GitHub branch

The changes are also on `codex/adaptive-beam-production-benchmarks`. When using
that branch, the project changes are already present: skip `project.patch`.
Initialize submodules, then apply the matching dependency patch from
`patches/tcc/`. For the pinned local dependency:

```bash
git submodule update --init --recursive
PKG="$PWD/patches/tcc"
git -C extern/tcc apply --check "$PKG/cuda-include-local-pinned.patch"
git -C extern/tcc apply "$PKG/cuda-include-local-pinned.patch"
```

For the production TCC revision listed below, use
`cuda-include-production.patch` instead. Preserve any existing dependency changes;
if the patch is already applied, verify it with `git apply --reverse --check`.
The submodule reference stays at its published upstream revision; the dependency
fix is distributed as a patch, rather than an unavailable submodule commit.
Continue at **Build and correctness gates** after reading the environment controls.

## Transport and source base

The accompanying `adaptive-graphs-benchmark-handoff.zip` contains:

- `project.patch`: all project changes relative to
  `75d08d9749010461faf3f4055a604ae80fde48d9`, including the earlier weights-only
  graph/profiler patches and this coarse/fine extension.
- `tcc-production.patch`: the existing CUDA-header fix for the measurement TCC
  revision `13e143301be1059caef7a351c8edb3ff800772c5`.
- `tcc-local-pinned.patch`: equivalent fix for the local pinned TCC revision
  `36e5edc3bc49772670edb5451f7b397342d26602`. Choose the correct one, not both.
- This handoff, validation notes, and checksums.

Use a clean checkout/worktree at the project base. Preserve the production
agent's existing uncommitted changes. `project.patch` already includes the
previous weights-only patch: do not stack it blindly on a dirty graph checkout.
Use the same production dependency revision and build options as the previous
measurements so this is a same-stack comparison. Record every revision and patch.

```bash
# PKG: absolute path to the unpacked handoff; run in the clean base checkout.
git rev-parse HEAD
# Must be 75d08d9749010461faf3f4055a604ae80fde48d9 before patching.
(cd "$PKG" && sha256sum -c MANIFEST.sha256)
git apply --check "$PKG/project.patch"
git apply "$PKG/project.patch"
# If using the production TCC revision above, and its fix is not already applied:
git -C extern/tcc apply --check "$PKG/tcc-production.patch"
git -C extern/tcc apply "$PKG/tcc-production.patch"
git diff --check
```

If the dependency patch is already present, verify with `git apply --reverse
--check` inside that submodule and retain it. Do not replace an unknown or dirty
dependency just to make a patch apply. The existing production reproduce-build
script can supply the verified site toolchain; inspect it and save the exact
script/configuration used. The local validation used the pinned local TCC plus
backport, so production capture compatibility still needs testing on its stack.

## What changed

- One compute graph per GPU buffer, prepared after masks and protection have
  been configured and after each buffer's libraries have warmed up.
- Correlation, covariance preparation, cuSOLVER, spatial filtering and beamforming
  are captured together when the CUDA/library stack supports it.
- H2D input transfers, steering/signature updates, visibility D2H transfers and
  all output callbacks stay outside capture. Visibility callbacks were moved
  after compute so their per-block host addresses are never cached in a graph.
- Replacing a mask or first enabling protection invalidates graphs before old
  device/workspace pointers are freed. `prepare_cuda_graphs()` recreates them
  before ingest starts. Signature content updates keep their device addresses.
- The harness reports `captured_graphs=N`; application HDF5 output records
  `captured_cuda_graphs`. The replay runner retains and checks that attribute.

Environment controls use variable **presence**, consistent with the existing
weights-only control. Unset a variable to disable it; `=0` still enables it.

| Control | Meaning |
|---|---|
| `SPATIAL_DISABLE_CUDA_GRAPH=1` | Force eager execution in every mode |
| `SPATIAL_ENABLE_ADAPTIVE_CUDA_GRAPH=1` | Opt into coarse/fine graph preparation |
| `SPATIAL_REQUIRE_CUDA_GRAPH=1` | Fail startup unless every GPU buffer has a graph |
| `SPATIAL_BENCH_PROFILE_RANGE=1` | Bracket measured harness blocks for Nsight |

Weights-only remains graph-enabled by default. Adaptive modes remain eager by
default. Strict mode is mandatory for graph performance claims. A capture or
solver error, allocation failure, timeout or missing graph is a failed/unsupported
case, not an eager result to label as graph performance. Some CUDA/library errors
terminate the process directly; keep the complete log and exit status.

## Build and correctness gates

Use the established production build directory or reproduce its configuration
in a fresh directory. Confirm the fixed harness shape: four sources, 40 receivers,
40 coarse channels, 32 PFB bins/28 retained, two polarizations, paired beams, and
rank two for adaptive modes. The GPU harness has 40 correlator receiver slots;
the full-app target has 64. Do not compare their rates as identical workloads.

```bash
BUILD=build_production_bench
cmake --build "$BUILD" --target FineChannelizedAdaptiveTests adaptive_beam \
  bench_adaptive_beam_4_40 bench_weights_pipeline_40 -j "$BUILD_JOBS"
# Start without inherited experiment controls.
unset SPATIAL_DISABLE_CUDA_GRAPH SPATIAL_ENABLE_ADAPTIVE_CUDA_GRAPH
unset SPATIAL_REQUIRE_CUDA_GRAPH SPATIAL_BENCH_PROFILE_RANGE
"$BUILD/tests/FineChannelizedAdaptiveTests"
SPATIAL_ENABLE_ADAPTIVE_CUDA_GRAPH=1 SPATIAL_REQUIRE_CUDA_GRAPH=1 \
  ADAPTIVE_BEAM_FULL_BINARY="$PWD/$BUILD/apps/bench_adaptive_beam_4_40" \
  python3 -m pytest -q -ra tests/test_adaptive_beam_full_app.py
SPATIAL_ENABLE_ADAPTIVE_CUDA_GRAPH=1 SPATIAL_REQUIRE_CUDA_GRAPH=1 \
  ADAPTIVE_BEAM_BINARY="$PWD/$BUILD/apps/adaptive_beam" \
  python3 -m pytest -q -ra tests/test_adaptive_beam_app.py
python3 -m pytest -q tests/test_adaptive_graph_benchmark_cli.py
```

The small app pytest fixture requires the regular app's default one-source,
eight-coarse-channel dimensions; use a separate default-shape build for it if
the production `adaptive_beam` was configured differently. Its covariance tests
exercise visibility callbacks outside capture. Put pytest temporary files on a
local filesystem with sufficient capacity (`--basetemp`); earlier production
checks exhausted `/tmp`. Do not change calibration or skip output checks to make
performance tests pass.

The new numerical test compares eager/graph beam voltages for coarse/fine with
protection off/on, two changing input blocks, a replaced input mask, a changed
protected signature, nonzero delay and stream/polarization permutations. It
asserts real graph capture and zero solver failures. Together with the existing
physical-invariant tests this gates the performance experiment.

## GPU benchmark matrix

Use fresh result directories and one GPU workload at a time. Record GPU clocks,
load, available memory, NUMA/PCIe state, huge-page registration, source and build
fingerprints. The runner clears inherited graph/profiling controls, randomizes
case order, interleaves each eager/graph pair, and saves logs plus `runs.json/csv`.
It rejects mismatched graph counts and inconsistent byte/time accounting. Failed
cases stay in the result table, and any failure gives a nonzero overall exit.

```bash
# Plan first; no GPU work or output files are created by --dry-run.
python3 scripts/benchmark/adaptive_graphs.py \
  "$BUILD/apps/bench_weights_pipeline_40" "$RESULTS/gpu" \
  --modes weights-only coarse fine --buffers 1 2 3 \
  --protection off rank-one --blocks 192 --repeats 3 --dry-run

# Execute the same command with --dry-run removed.
```

For a short compatibility gate use `--buffers 1 --blocks 32 --repeats 1` and
`--modes coarse fine` first, with a different result directory. Do not publish
those short smoke runs as production capacity measurements. If three buffers do
not fit, record allocation failure; do not silently reduce the shape or buffers.

Report median and min/max for three repeats per exact configuration. Include the
full per-buffer table, not only the winning buffer count. Calculate graph/eager
speedup only within the same mode, rank, protection and buffer count. Startup,
JIT and graph instantiation occur before timing. Input rates include H2D and
compute but exclude packet parsing and output writes.

## Profile and full-app follow-up

Profile one successful case per mode/protection combination after the untraced
matrix. Example for protected coarse with three buffers:

```bash
SPATIAL_ENABLE_ADAPTIVE_CUDA_GRAPH=1 SPATIAL_REQUIRE_CUDA_GRAPH=1 \
SPATIAL_BENCH_PROFILE_RANGE=1 nsys profile --trace=cuda \
  --cuda-graph-trace=node --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  -o "$RESULTS/coarse-graph-3" \
  "$BUILD/apps/bench_weights_pipeline_40" 192 3 coarse protected
```

Use an installed Nsight version supporting these flags. Save CUDA kernels,
memcpy timings and wall time separately; summed stream work is not wall latency.
Compare traced and untraced throughput because graph-node tracing previously
perturbed weights-only results.

For matched short file replay with paired HDF5 writing:

```bash
SPATIAL_DISABLE_CUDA_GRAPH=1 python3 scripts/benchmark/adaptive_beam.py \
  "$BUILD/apps/bench_adaptive_beam_4_40" "$RESULTS/replay-eager" \
  --fpgas 4 --channels 40 --modes coarse fine --ranks 2 \
  --skip-visibilities --gpu-buffers 3 --repeats 3 --seconds 5
SPATIAL_ENABLE_ADAPTIVE_CUDA_GRAPH=1 SPATIAL_REQUIRE_CUDA_GRAPH=1 \
  python3 scripts/benchmark/adaptive_beam.py \
  "$BUILD/apps/bench_adaptive_beam_4_40" "$RESULTS/replay-graph" \
  --fpgas 4 --channels 40 --modes coarse fine --ranks 2 \
  --skip-visibilities --gpu-buffers 3 --repeats 3 --seconds 5
```

Unset any inherited `SPATIAL_DISABLE_CUDA_GRAPH` before the graph command. Repeat
with covariance recording (omit `--skip-visibilities`) only after the initial
matrix passes. Full replay includes host scheduling/writing and cannot substitute
for GPU timing. It is not live-network or durable-storage capacity. If writing a
wrapper, name duration variables `REPLAY_DURATION`, never Bash's special `SECONDS`.

## Return artifacts and completion

Return generic, anonymized summary tables, per-run CSV/JSON, complete logs,
source/dependency patches and fingerprints, correctness results, and representative
Nsight traces. State the exact number of captured graphs for each graph run.
Keep private deployment details in local provenance only. Successful completion
requires correctness gates, all requested supported cases measured with matched
controls, and explicit failure/unsupported records for everything else. Make no
coarse/fine graph speedup claim until the new measurements support it.
