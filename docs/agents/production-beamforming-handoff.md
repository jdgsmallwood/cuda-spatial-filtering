# Production beamforming benchmark handoff

## Objective and completion

Benchmark this branch on the production hardware before further optimization.
Compare Starspin correlation-only, weights-only fine-channel beamforming, coarse
spatial decomposition and fine spatial decomposition. Establish sustained input
throughput, loss-free operating bandwidth, GPU stage costs and RFI suppression
while retaining the Sun. Report throughput and scientific validation separately.

Finish with a local results table, reproducible commands, configuration/build
fingerprints, observing provenance, before/after spectra and a ranked bottleneck
assessment. State explicitly which comparisons are unmatched or unverified.
Keep machine identities, network addresses, credentials, calibration files, raw
captures and production logs in local artifacts. Public summaries must use
anonymous labels and contain no production-identifying information. Publishing
production artifacts is outside this handoff.

Read [the design decision](../adr/0001-coarse-and-fine-protected-beamforming.md)
and the repository's `AGENTS.md` before starting. Use existing site operating
procedures for FPGA control and calibration; the examples below are application
commands, not authority to invent site-specific control commands.

## What is implemented

- `adaptive_beam`: `--decomposition weights-only|coarse|fine`, fixed `--null-rank`,
  `--sun-protection off|rank-one`, `--calibration gains|unity`, `--gpu-buffers`,
  and optional `--record-visibilities`.
- All modes produce paired original/filtered fine-channel complex voltages.
  Weights-only duplicates the unfiltered beam intentionally to match output
  volume; it bypasses correlation and eigensolving. Adaptive rank zero still
  correlates and decomposes, so it is not the weights-only baseline.
- Coarse mode correlates pre-channelizer samples and reuses a coarse filter
  across retained fine bins. Fine mode correlates and decomposes fine channels.
- Rank-one protection restricts the nulled eigenspace to the protected signature's
  orthogonal complement. Actual solar retention depends on correct calibration,
  steering, coordinates and source-model adequacy. Coarse-centre protection is
  not a guarantee across the whole passband.
- Coarse preparation now shares the fused Starspin raw gather, generating both
  channelizer input and correlation input. Four intermediate buffers and the
  separate packing/clearing passes are removed. ccglib optimized complex GEMM
  performs beam multiplication. Receiver padding is initialized once per buffer.
- Input allocation requests huge pages and CUDA registration, with pinned-memory
  fallback. Build flags alone do not prove that huge pages were obtained.
- Fixed RA/Dec tracking and solar signatures use the pointing code. Verify the
  coordinate convention and time updates against an independent reference before
  interpreting sky data. Automatic null-rank selection and online calibration
  are future work; record covariances for offline rule development.
- Starspin correlation-only no longer requires gains/eigenvalue-count files.
  The monitor now samples up to eight connected receiver pairs by default;
  monitoring settings must be held constant in comparisons.

## Local reference, not a production expectation

Local reference hardware: RTX 4060 8 GiB, Ryzen 7 7700. Configuration: four FPGA
sources, 40 receivers, two polarizations, 40 coarse channels, 32 PFB bins with
28 retained, one pointing target and paired outputs. Input rates are decimal
GB/s of successfully processed int8 complex payload, excluding headers/scales.

| Measurement | Local result |
|---|---:|
| Weights-only GPU path, one / three buffers | about 5.09 / 7.84 GB/s |
| Protected coarse GPU path, one buffer, before / after fusion | 2.84 / 4.20 GB/s |
| Protected coarse GPU path, three buffers | 6.25 GB/s |
| Protected fine GPU path, one buffer | 1.90 GB/s |
| Full replay + voltage writer, weights-only / protected coarse | about 3.33 / 3.38 GB/s |
| Pinned H2D transfer bandwidth | about 13.4 GB/s |

GPU harness rates include H2D, exclude parsing and output transfers/writing.
The matched coarse GPU comparison improved 48%. Other measurements came from
separate runs. Synthetic matrices affect eigensolver cost; rates are not sky
performance predictions. Local CPU sampling points to replay feeding and
synchronization as the end-to-end weights-only/coarse limit, not a proven live
NIC bottleneck. Enabling AVX-512 did not raise that local replay ceiling.

Protected coarse now spends about 3.16 ms/block in fused preparation, 2.62 ms
in channelization, 2.25 ms in correlation, 1.97 ms in ccglib GEMM and 3.48 ms
in its transpose kernels. Total kernels plus H2D: about 24.88 ms/block.
Fine eigensolve/filter work alone was about 29.46 ms/block, compared with
1.32 ms for coarse. Fine solves 2240 matrices per block versus 80 for coarse.
Seven numerical and six full-array application tests passed after optimization.
The raw local traces are not committed; reproduce these measurements as needed.

## 1. Prepare and verify the build

Fetch branch `codex/adaptive-beam-production-benchmarks` into a clean checkout
or worktree without discarding local production changes. Record the commit,
submodule revisions, GPU/CPU, CUDA/driver/library versions, NUMA placement,
PCIe negotiated generation/width and build flags in local artifacts.

Use the site's supported dependency environment and a fresh build directory.
Set numeric `CMAKE_CUDA_ARCHITECTURES` for the actual GPU. Native host compilation
must happen on the deployment CPU. Example configuration for the 4-source,
40-channel comparison (adapt dimensions to the verified capture configuration):

```bash
cmake -S . -B build_production_bench \
  -DBUILD_TESTING=ON -DCMAKE_CUDA_ARCHITECTURES="$GPU_ARCH" \
  -DSPATIAL_NATIVE_ARCH=ON \
  -DNR_OBSERVING_FPGA_SOURCES=4 -DNR_OBSERVING_CHANNELS=40 \
  -DNR_OBSERVING_PADDED_RECEIVERS=64 \
  -DNR_OBSERVING_FINE_CHANNELS=32 -DNR_OBSERVING_FINE_CHANNEL_EDGE_TRIM=2 \
  -DSTARSPIN_CHANNEL_COUNTS=40
cmake --build build_production_bench --target adaptive_beam starspin_4_40 \
  bench_adaptive_beam_4_40 bench_weights_pipeline_40 bench_starspin_h2d \
  FineChannelizedAdaptiveTests PointingTests -j "$BUILD_JOBS"
build_production_bench/tests/FineChannelizedAdaptiveTests
build_production_bench/tests/PointingTests
ADAPTIVE_BEAM_FULL_BINARY="$PWD/build_production_bench/apps/bench_adaptive_beam_4_40" \
  python3 -m pytest -q tests/test_adaptive_beam_full_app.py
```

Set `GPU_ARCH` and `BUILD_JOBS` from the actual machine before executing. The
fixed local GPU harness always represents four sources/40 receivers/40 coarse
channels; changing the regular app's CMake dimensions does not change that
harness. Inspect printed dimensions for every executable. Starspin's four-source
40+ channel targets optimize for 36 connected receivers, unlike the 40-slot beam
app. Report this distinction; add a matched compute configuration if an exact
algorithm-cost comparison is needed. Similarly, existing Starspin GPU harnesses
are 48/54-channel targets and are not a matched 40-channel comparison.

Confirm casacore can load DE200/DE405 and current Earth-orientation data. The
Dockerfile adds ephemeris lookup links; old running images may need the equivalent
local setup. A successful unity-gain test does not validate observing calibration.

Completion: build and relevant tests pass, dimensions and dependencies are known.

## 2. Establish memory and isolated GPU performance

Run one GPU workload at a time, with monitoring/profiling overhead controlled.

```bash
build_production_bench/apps/bench_starspin_h2d 40 128 combined 1 default 3
build_production_bench/apps/bench_starspin_h2d 40 128 combined 1 register 3
build_production_bench/apps/bench_starspin_h2d 40 128 combined 1 register_no_thp 3
build_production_bench/apps/bench_weights_pipeline_40 192 1 weights-only
build_production_bench/apps/bench_weights_pipeline_40 192 1 coarse protected
build_production_bench/apps/bench_weights_pipeline_40 192 1 fine protected
```

Repeat the GPU matrix with protection off (omit `protected`) and buffer counts
1, 2, 3 where memory permits. Record allocation failures rather than silently
changing dimensions. Verify registered host pointers, 2 MiB alignment and actual
huge-page backing from the harness/page reports; distinguish fallback allocation.
Check NUMA locality and memlock limits if registration or bandwidth differs from
expectation. Verify AVX-512 support and actual generated stores before attributing
copy performance to it. Keep normal logging consistent across comparisons.

Profile a representative steady-state run with CUDA-only Nsight tracing, excluding
startup/JIT/warmup. Compare traced and untraced throughput. Report H2D, preparation,
channelizer, correlation, eigensolve/filter, GEMM/transposes, output transfer and
GPU idle gaps separately. Summed GPU durations are not wall time when overlapped.
CPU sampling is a follow-up if gaps remain; broad OS-runtime tracing substantially
perturbed the local benchmark and was not used for performance conclusions.

Completion: stable repeated rates and a stage breakdown with measurement boundaries.

For a targeted synthetic full-app replay (paired output writing included), use
case selection instead of editing or copying the runner:

```bash
python3 scripts/benchmark/adaptive_beam.py \
  build_production_bench/apps/bench_adaptive_beam_4_40 local-results \
  --fpgas 4 --channels 40 --modes coarse --ranks 2 --skip-visibilities --dry-run
```

Remove `--dry-run` to execute. The default remains the full 19-case matrix;
`--weights-only` remains available. Dry runs need no GPU, capture or scientific
Python dependencies. Before handoff, run monitoring tests in required mode as
specified in `tests/TESTING.md` so missing headers cannot silently skip checks.

## 3. Verify the observing session before live comparisons

At the start of every live session, verify FPGA boot state and the delay solution.
If the boot state or matching delay calibration cannot be verified, reboot the
selected FPGAs, run full `init-tune` at eight channels, solve fresh packet delays,
and confirm delay graph closure. Only then retune to the requested wider range.
Read back the channel range on every board before capture. Use the delay file
produced for that boot state. Consecutive captures in the same session reuse the
verified boot/delay solution; changing the binary or channel range is not a reason
to reboot. Record boot, tuning, delay-file hash and closure evidence with captures.

Validate antenna mapping, polarizations, exclusions, gains frequency axis, array
location, observation time, and Sun steering-vector coordinates. Use existing
calibrated `gains.json` for scientific comparisons. Explicit `--calibration unity`
is suitable for throughput/diagnostics but must not be reported as validated solar
protection. Test fixed RA/Dec tracking over multiple steering refresh intervals;
a frozen `--steering-utc` is for reproducible replay, not tracking a live sky.

Completion: observing provenance and calibration/pointing checks are recorded.
No CPU/GPU/receive-ring bottleneck claim may rely on an unverified observing run.

## 4. Run the live benchmark matrix

Reuse the verified site's ingest configuration, capture backend, interfaces and
packet format. Use `adaptive_beam --help` for exact shared options. Core arguments:

```text
adaptive_beam <verified live ingest/config/delay/target arguments>
  --decomposition coarse --null-rank 2 --sun-protection rank-one
  --calibration gains --gains <local-gains.json> --gpu-buffers 1
  --beam-output-file <local-output.h5> --obs-length 60
```

Start with a short smoke capture, then at least three 60-second measured runs per
case; extend if noise or thermal/storage behavior makes the results unstable.
Interleave modes to reduce sky/time bias, reuse the verified session, and read back
board ranges before each capture. Sweep offered bandwidth to locate a sustained
loss-free limit; a run merely keeping up with a low offered rate is not capacity.

| Case | Null rank | Protection | Output |
|---|---|---|---|
| Starspin correlation-only | N/A | N/A | Standard visibilities |
| Weights-only | 0 | Off | Paired complex voltages |
| Coarse | 0, 1, 2, 4 | Off and rank-one | Paired complex voltages |
| Fine | 0, 1, 2, 4 | Off and rank-one | Paired complex voltages |

First run weights-only and rank-two coarse/fine with one buffer, then sweep buffer
counts and ranks. Keep channel range, integration/update cadence, beam targets,
gains, exclusions and storage destination matched. Starspin writes different
products: report its end-to-end rate as a system comparison, with output bytes/s,
not as a pure algorithm comparison. Hold monitoring settings constant.

For coarse and fine rank two, repeat with `--record-visibilities` and a distinct
`--vis_output_file` to quantify recording overhead. Use those covariances for
offline eigenspectra/null-rank research. Do not add synchronous online eigenvector
telemetry to the throughput runs. Prefer existing replay fixtures or a bounded
verified recording; packet capture through tshark can drop packets and is not
assumed to provide a lossless reference.

Per run record successfully processed input bytes/elapsed seconds, offered rate,
received/missing/discarded packets, capture drops, arrival and output-block
accounting, solver failures, GPU memory/utilization, CPU load, output bytes/s and
profile overhead. Treat page-cache writes separately from durable-storage speed.
Report minimum/median/maximum across repeats. Stop escalating bandwidth when loss
or invalid outputs appear; preserve the evidence and diagnose the failed stage.

## 5. Validate suppression and deliver findings

Use an FM-band selection where permitted by the verified tuning procedure. Plot
original/filtered power spectra from paired complex voltages, with matched
normalization, frequency axes, integration durations and exclusion masks. Compare
protection on/off and coarse/fine without confusing gain changes with suppression.
Measure retention on the Sun separately from RFI suppression, across retained fine
bins. Assess fixed RA/Dec pointing stability over time. Where the data cannot
separate solar power from interference, label solar retention inconclusive.

Deliver locally:

1. Run manifest with commands, commit/build/config fingerprints and observing
   verification evidence, plus raw outputs or bounded diagnostic excerpts.
2. Throughput/loss table, buffer-count and bandwidth sweeps, and GPU stage chart.
3. Before/after spectra and quantitative RFI suppression/solar-retention results,
   with uncertainties and any calibration/pointing limitations.
4. A ranked recommendation: ingest/scheduling, PCIe/NUMA, output writing, or GPU
   stages. Fine eigensolving remains the main GPU candidate; a slower update cadence
   is a behavior change and needs a separate scientific evaluation. Keep adaptive
   rank selection and online calibration as explicit follow-up design work.
