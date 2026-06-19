# Benchmark sweep

`run_benchmarks.py` runs the per-component throughput microbenchmarks built in
`apps/` and prints a summary table of packets/sec (or blocks/sec) and GB/sec for
each pipeline stage, so the slowest stage -- the bottleneck -- is obvious at a
glance:

| Binary | What it measures |
|---|---|
| `bench_capture` | Raw kernel-socket packet capture (`recvmmsg` ingestion) |
| `bench_processor` | Ring-buffer reassembly (`ProcessorState::process_packets + pipeline_feeder`), fed by synthetic packets so it measures only the CPU path, not GPU time |
| `bench_gpu` | GPU correlate+beamform pipeline (`LambdaCorrBeamOnlyGPUPipeline`), **no writer overhead** (output_=nullptr so no D2H; measures raw GPU compute throughput) |
| `bench_writers` | HDF5 visibilities/eigendata writer throughput |

`bench_processor`, `bench_gpu`, and `bench_writers` each test **eight configs**
in a single run without rebuilding: `1ch/1fpga`, `8ch/1fpga`, `16ch/1fpga`,
`24ch/1fpga`, `32ch/1fpga`, `8ch/4fpga`, `16ch/4fpga`, `32ch/4fpga`. This
lets you see how each stage scales with channel count and FPGA count
from 8 to 32 channels in steps of 8.

### GPU pipeline scaling behaviour (RTX 4060, 8 GB VRAM)

Real-time baseline: 15500 packets/sec/channel/FPGA × 2660 bytes/packet.
`real-time %` = (actual runs/sec) ÷ (15500 / NR_PACKETS_FOR_CORRELATION) × 100.
Channel and FPGA counts cancel out — only NR_PACKETS_FOR_CORRELATION sets the
required runs/sec threshold.

### Channel sweep (NR_PACKETS_FOR_CORRELATION=256, required 60.5 runs/sec)

| Config | runs/sec | input_GB/sec | real-time % |
|---|---|---|---|
| 8ch/1fpga | 804 | 4.25 | 1329% |
| 16ch/1fpga | 399 | 4.21 | 659% |
| 24ch/1fpga | 262 | 4.16 | 433% |
| 32ch/1fpga | 196 | 4.14 | 324% |
| 8ch/4fpga | 321 | 6.78 | 531% |
| 16ch/4fpga | 160 | 6.78 | 265% |
| 24ch/4fpga | 105 | 6.67 | 174% |
| 32ch/4fpga | 77 | 6.53 | 127% |

### Corr-packet sweep (8ch, 1fpga and 4fpga, corr=64→1024)

Integration time per run = NR_PACKETS_FOR_CORRELATION / 15500 packets/sec/channel.

| corr pkts | integ. time | required runs/sec | 8ch/1fpga runs/sec | real-time % | 8ch/4fpga runs/sec | real-time % |
|---|---|---|---|---|---|---|
| 64 | 4.1 ms | 242.2 | 3460 | 1428% | 1418 | 586% |
| 128 | 8.3 ms | 121.1 | 1625 | 1342% | 648 | 535% |
| 256 | 16.5 ms | 60.5 | 805 | 1330% | 320 | 529% |
| 512 | 33.0 ms | 30.3 | 402 | 1327% | 158 | 521% |
| 1024 | 66.1 ms | 15.1 | 201 | 1331% | 79 | 523% |

| corr pkts | 8ch/1fpga GB/sec | 8ch/4fpga GB/sec |
|---|---|---|
| 64 | 4.68 | 7.67 |
| 128 | 4.33 | 6.90 |
| 256 | 4.25 | 6.76 |
| 512 | 4.23 | 6.66 |
| 1024 | 4.22 | 6.60 |

Key findings:
- **1fpga (10 rx → 32 padded)**: ~4.1–4.7 GB/sec across all configs. Runs/sec
  halves when channels or corr_packets double — GPU is compute-bound on TCC.
  Real-time % is ~1330% for all corr_packet values: margin is set by compute,
  not integration length.
- **4fpga (40 rx → 64 padded)**: ~6.6–7.7 GB/sec. Larger correlation matrices
  give better GPU SM utilization. Real-time % ~520–590%, roughly constant across
  corr_packet values for the same reason.
- **Smaller NR_PACKETS_FOR_CORRELATION gives higher GB/sec** (shorter CUDA
  graph → less per-graph overhead amortised over more launches). 4fpga benefits
  more: +13% GB/sec at corr=64 vs 256 (7.67 vs 6.76), vs +10% for 1fpga.
  Choose the smallest corr value that gives acceptable integration time for your
  science use-case.
- **3 pipeline buffers** is the sweet spot for 4fpga configs on 8 GB VRAM: 38%
  faster than 2 buffers (6.24 vs ~4.5 GB/sec for 8ch/4fpga) while avoiding OOM.
- `bench_gpu --with-output` adds D2H cost (~0.7 GB/sec for 8ch/1fpga) and
  measures the combined H2D+compute+D2H path as seen by writers.
- **32ch/4fpga at 127%** is the tightest margin — only 27% headroom above
  real-time on this GPU. Writer overhead and CPU-side processing will eat into
  that, making it the bottleneck config to watch for live deployments.

Optionally (`--with-observe`) it also runs the full `observe` pipeline against a
looping multi-FPGA PCAP replay, as an end-to-end comparison point.

## Build

```bash
mkdir build_apps && cd build_apps
CUTENSOR_ROOT=/opt/conda cmake -DBUILD_TESTING=OFF \
  -DCMAKE_EXE_LINKER_FLAGS="-Wl,--sysroot=/ -L/usr/lib/x86_64-linux-gnu -Wl,-rpath-link,/usr/lib/x86_64-linux-gnu -Wl,--allow-shlib-undefined" \
  ..
cmake --build . -- -j$(nproc)
```

The bench binaries hard-code the three benchmark configs; they do not read
`-DNR_OBSERVING_*` CMake variables (those only affect `apps/observe`,
`gpu_benchmark`, etc.).

## Running

```bash
export CUDA_HOME=/opt/conda/targets/x86_64-linux   # needed by gpu_benchmark/observe (NVRTC)

python3 scripts/benchmark/run_benchmarks.py \
    --build-dir /workspace/build_apps --duration 10 \
    --interfaces enp134s0np0,enp175s0np0,enp216s0np0,enp220s0np0
```

Use `--skip <stage> [...]` (choices: `capture`, `processor`, `gpu`, `writers`,
`observe`) to exclude stages, and `--json-out <path>` to dump the full results
dict (with git commit + timestamp) for later comparison:

```bash
python3 scripts/benchmark/run_benchmarks.py \
    --build-dir /workspace/build_apps --duration 10 \
    --json-out /tmp/bench_main.json
```

### Reading the summary

```
==============================================================================
SUMMARY
==============================================================================
stage                config         packets|blocks/sec         GB/sec
------------------------------------------------------------------------------
capture              —                        15,644.50         0.0217
processor            1ch/1fpga             2,375,450.00         3.2830
processor            8ch/1fpga             1,850,000.00         2.5558
processor            8ch/4fpga               950,000.00         1.3127
vis_writer           1ch/1fpga                    49.50         0.0000
vis_writer           8ch/1fpga                    42.10         0.0003
vis_writer           8ch/4fpga                    18.70         0.0020
eigen_writer         1ch/1fpga                   200.30         0.0003
eigen_writer         8ch/1fpga                   185.10         0.0024
eigen_writer         8ch/4fpga                    92.40         0.0192
------------------------------------------------------------------------------
Bottleneck (lowest GB/sec): capture[—] @ 0.0217 GB/sec
```

The bottleneck line picks the stage+config with the lowest GB/sec.  Note the
stages measure different things (raw wire bytes for capture/processor, HDF5
output bytes for the writers) -- use the table to spot which stage is
*relatively* far below the others, not to compare absolute numbers across stages
as if they were the same unit of work.

## PR-to-PR comparison

Collect a JSON result from both the base branch and the PR branch, then diff:

```bash
# On main / base branch:
python3 scripts/benchmark/run_benchmarks.py \
    --build-dir /workspace/build_apps --duration 10 \
    --json-out /tmp/bench_baseline.json

# After switching to PR branch and rebuilding:
python3 scripts/benchmark/run_benchmarks.py \
    --build-dir /workspace/build_apps --duration 10 \
    --json-out /tmp/bench_pr.json

# Compare:
python3 scripts/benchmark/compare_benchmarks.py \
    /tmp/bench_baseline.json /tmp/bench_pr.json
```

Example output:

```
==============================================================================
BENCHMARK COMPARISON
==============================================================================
  baseline : bench_baseline.json  commit=a1b2c3d  branch=main  ts=2026-06-19T10:00:00Z
  current  : bench_pr.json        commit=e5f6g7h  branch=feat/faster-ring  ts=2026-06-19T10:30:00Z

  metric                                         baseline      current  change
------------------------------------------------------------------------------
  [processor][1ch/1fpga] packets/sec         2375450.00  -> 2450000.00  (+3.1%)
  [processor][1ch/1fpga] GB/sec                   3.28  ->       3.39  (+3.1%)
  [processor][8ch/1fpga] packets/sec         1850000.00  -> 1820000.00  (-1.6%)
  [processor][8ch/1fpga] GB/sec                   2.56  ->       2.51  (-1.6%)
  [processor][8ch/4fpga] GB/sec                   1.31  ->       1.00  (-23.7%)  !! REGRESSION
==============================================================================

!! 1 REGRESSION(S) detected (>5% slowdown):
   [processor][8ch/4fpga] GB/sec      1.31 ->  1.00  (-23.7%)  !! REGRESSION
```

`compare_benchmarks.py` exits with code 1 if any regression is detected (useful
for CI gating).

## Loopback self-test for `bench_capture` (no real NICs needed)

The default interface (`lo`) automatically replays `example_packets.pcapng`
(or a synthetic pcap) via `udp_sender`, so no extra flags are needed:

```bash
python3 scripts/benchmark/run_benchmarks.py \
    --build-dir /workspace/build_apps --duration 10 \
    --skip processor gpu writers
```

To generate a synthetic pcap manually:

```bash
python3 scripts/create_pcap.py \
    --number_packets 500 --number_channels 8 --number_receivers 10 \
    --output /tmp/loopback_test.pcap

python3 scripts/benchmark/run_benchmarks.py \
    --build-dir /workspace/build_apps --duration 10 \
    --interfaces lo --capture-loopback-pcap /tmp/loopback_test.pcap \
    --skip processor gpu writers
```

`udp_sender` replays at ~5 µs/packet -- this measures receive-path throughput
at ~15 Kpkt/s, limited by replay rate, not the socket ingestion ceiling.

## End-to-end `observe` comparison (`--with-observe`)

`--with-observe` additionally runs the full `observe` binary against a looping
4-FPGA PCAP replay (auto-generated at `--observe-pcap`) and computes
`packets/sec`/`GB/sec` from the `Stats: Received=..., Processed=...` lines it
prints every 5 seconds.  Use `--observe-duration` (must be >5s) to control how
long it runs.

## Known issues

- **`bench_writers` eigen writer can run longer than `--writer-duration`**:
  if the HDF5 drain thread can't keep up, the ring fills and you'll see
  `EigenWriter ring buffer is full. Waiting...` log lines.  The reported
  `blocks/sec`/`GB/sec` are still computed from actual elapsed time.

- **`bench_capture` on loopback measures replay rate, not NIC ceiling**: the
  ~15 Kpkt/s on loopback reflects `udp_sender`'s 5 µs/packet pacing, not the
  kernel socket's maximum ingestion rate.  Use real NICs to benchmark ingestion
  throughput.
