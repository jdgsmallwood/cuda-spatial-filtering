# Starspin capture runbook

Practical build/run commands for the `starspin_{NR_FPGAS}_{NR_CHANNELS}` targets
(correlation-only, visibilities-only counterpart to `observe` — see `apps/starspin.cu` /
`LambdaStarspinPipeline`), captured from live 4-FPGA/24-channel tuning on `blackmesa`
(2026-09-20). See `docs/architecture.md` for the pipeline's data-flow diagrams; this file is
the operational "how do I actually build and run it" companion, plus the specific environment
variables that turned out to matter for real packet-loss-free capture on this host.

For a plain-English description of the architecture and the changes that took the system from
24 to 40 channels, read `docs/starspin-24-to-40-explained.md`.

## Easiest operator command (four FPGA, 40 channels)

The repository-local wrapper supplies every required path and the proven defaults:

```bash
cd /data/BLACKMESA_1/sma075/cuda-spatial-filtering
scripts/run_starspin.sh --duration 600
```

It defaults to `starspin_4_40`, ibverbs direct-to-ring receive, interfaces
`enp94s0np0,enp134s0np0,enp175s0np0,enp216s0np0`, channels 176--215, capture CPUs
`1,11,13,15`, workers `3,4,5`, the current delay/gain/config/mapping files, and timestamped
outputs under `build/starspin_runs/`. It checks input files, NIC existence, free disk space,
and `cap_net_raw` before starting, prints the complete resolved command, and tees console output
to the run's log.

Useful variants:

```bash
# Safe command preview; does not touch the backend.
scripts/run_starspin.sh --dry-run --duration 600

# Prepared loss-attribution binary (after granting cap_net_raw to that copy).
scripts/run_starspin.sh --diagnostic --duration 120

# Explicit output name and directory.
scripts/run_starspin.sh --duration 600 --name science_40ch \
  --output-dir /path/with/space

# Retune FPGAs 0-3 first. This is deliberately opt-in.
scripts/run_starspin.sh --configure-fpga --duration 600
```

Run `scripts/run_starspin.sh --help` for all path, affinity, channel, and binary overrides.
The launcher was adapted from `/data/BLACKMESA_0/lambda/scripts/correlate_lambda.sh`; nothing in
`/data/BLACKMESA_0/lambda/scripts` was changed.

The launcher was used for a 1,200-second diagnostics-enabled 40-channel observation. It completed
at 2.314779 Mpps with 2,777,734,646 received, 2,777,733,056 processed, 168,395 Missing
(**0.006061947%**), zero discarded/future/stuck packets, and a clean shutdown. Missing by FPGA
index was `[41928,41960,42000,42507]`; the near-equal split despite radically different capture
thread context-switch totals rules out per-NIC thread scheduling as the primary loss source. See
`docs/ingest-optimization-handoff.md` for the full interpretation and artifact paths.

**2026-09-21 32-channel result:** current code initializes the shared four-FPGA timeline once
from the first valid packet and the configured delay file. It does not re-seed at runtime and it
does not wait for all four FPGAs before starting. A few packets discarded during roughly the first
second are acceptable; sustained or growing discards are not. For the current restarted-FPGA
state use the delay file in `build/apps/alveo_delays.json` and `--min_freq_channel 180`. A live
60-second ibverbs zero-copy run completed cleanly with 111,037,786 received, 111,037,568
processed, zero `NICDrops`, zero `StuckUnprocessed`, and 3,390 pipeline runs. The 4,640 startup
discards stayed flat for the whole run; `Missing` stayed flat for the final eight reports
(roughly 40 seconds), proving steady-state convergence. A separate 10-second NIC-counter sample
measured 1.8582 Mpps actually offered, consistent with the application's 1.8506 Mpps rather than
the nominal 1.984 Mpps assumption. See
`docs/ibverbs-zerocopy-handoff.md` for the complete evidence and artifact paths.

A follow-up 600-second run confirmed the result at length: 1,111,114,882 received,
1,111,113,728 processed, 32,461 missing (**0.002921395%**, including startup), zero `NICDrops`,
zero stuck work, and 33,908 pipeline runs. The final missing increment occurred around minute
eight and the counter stayed flat for the last two minutes. Its artifacts use the prefix
`build/starspin_smoke_out/ibverbs_32ch_oneshot_600s`.

**Update (2026-09-21): `--capture-backend ibverbs` automatically uses direct-to-packet-ring
receive and is the recommended backend, not the plain kernel backend described as "known-good"
below.**
After a round of real bug fixes (see `docs/ibverbs-zerocopy-handoff.md` for the full story), it
sustains ~1.39M pps at 24ch with essentially zero loss (0.0095%) and no drops/timeouts — well
above anything the kernel backend reached this session. The kernel-backend section below is kept
as a still-valid, still-working fallback (and because it needs no `cap_net_raw`/root at all), but
for new work, read `docs/ibverbs-zerocopy-handoff.md` first — it also documents an **open issue**
at 32 channels when one FPGA carries an unusually large relative delay (e.g. after an FPGA
restart), and the exact `cap_net_raw` re-grant workflow every ibverbs rebuild needs.

## Build

This host's toolchain lives under `/data/BLACKMESA_1/sma075/miniconda3` (not `/opt/conda` —
this is *not* the Docker container `CLAUDE.md`'s main instructions describe). From the existing
`build/` directory (do not delete/recreate it — see below):

```bash
cd /data/BLACKMESA_1/sma075/cuda-spatial-filtering/build
export CUDA_HOME=/data/BLACKMESA_1/sma075/miniconda3/targets/x86_64-linux

# Reconfigure (only needed when changing a cache variable, e.g. worker thread count):
cmake -DNR_OBSERVING_PACKET_WORKER_THREADS=3 -DSPATIAL_NATIVE_ARCH=OFF .

# Build just the target you need (much faster than a full build):
cmake --build . --target starspin_4_24 -- -j$(nproc)
```

For the current 32-channel investigation, substitute `starspin_4_32`. Every rebuild replaces
the executable and clears its file capability; restore it before an ibverbs run:

```bash
sudo setcap cap_net_raw+ep \
  /data/BLACKMESA_1/sma075/cuda-spatial-filtering/build/apps/starspin_4_32
getcap /data/BLACKMESA_1/sma075/cuda-spatial-filtering/build/apps/starspin_4_32
```

The target grid now also includes `starspin_4_40`. Its current test range starts at channel 176
and ends at 215. Apply the same capability command with `starspin_4_40` as the filename and use
`--min_freq_channel 176`.

The first 60-second 40-channel ibverbs zero-copy run succeeded at 2.313595 Mpps (about 6.154
GB/s): 138,815,720 received, 138,814,912 processed, zero `NICDrops`, zero stuck work, and
73,522 missing (**0.052935705%**, including startup). Its artifacts use the prefix
`build/starspin_smoke_out/ibverbs_40ch_oneshot_60s`.

A second 60-second validation of the exact checkpoint binary completed with 138,851,828
received, 138,850,560 processed, 50,504 missing (**0.036359361%**, including startup), zero
future-queued/stuck work, and 3,389 pipeline runs. `Missing` stayed flat for the final 15 seconds
and shutdown was clean. Its artifacts use the prefix
`build/starspin_smoke_out/ibverbs_40ch_recheck_60s`.

The target grid also includes `starspin_4_48`. Before the pre-filter GPU transforms were fused,
the GPU pipeline sustained only 55--56 of the required 60.55 blocks/s; all eight host assembly
buffers eventually became GPU-owned and receive collapsed. See
`docs/ingest-optimization-handoff.md` for the historical evidence.

**2026-09-29 live retest after the fused pre-filter path:** A 60-second 48-channel run now
completed all 3,390 pipeline blocks with no host-buffer backlog, but `Missing` reached 2,206,468
of 166,625,280 packet slots (1.32%). A 20-second repeat reached 729,599 missing. All four
FPGA indices lost nearly equal numbers. The receive ring's `ReserveReadWaits` rose by about
214,000 per five seconds at 48 channels, versus essentially zero after startup at 40 channels;
`ReserveProcessedWaits` and `EmptyBufferWaits` stayed zero. Moving the three packet workers from
NUMA-node-0 CPUs 3,4,5 to NUMA-node-1 CPUs 16,17,18 raised 20-second missing to 13,385,504.
The current limit is in host packet consumption / receive-slot return, not sustained GPU service.
The existing `NICDrops=0` field does not reliably count libibverbs hardware drops. The successful
60-second 40-channel run had only 1,832 startup missing packets, with no further growth. After
the experiments, all four FPGAs were restored to channels 176--215.

**2026-09-29 CPU-side follow-up:** Four packet helpers pinned to node-0 CPUs 3,4,5,6 closed
the sustained loss gap at 48 channels. In three separate 20-second runs with 64-, 128-, and
256-slot dispatches, `Missing` stayed flat after the first report. The 256-slot version had the
fewest receive-ring wait events in the short comparison. A 120-second 256-slot run received
333,259,324 packets with 93,572 missing (0.02807%, including startup and a few small later
steps), zero discarded/future/stuck packets, 6,782 queued pipeline runs, and clean shutdown.
The HDF5 visibility samples checked were finite and nonzero, with monotonically increasing
sequence numbers. The normal `starspin_4_48` target now builds with four packet helpers, a
256-slot dispatch limit, and 36 active antennas; the launcher chooses the four local worker
CPUs by default for 48 channels. This is a capacity-stable two-minute result, not yet a longer
production observation.

A further 60-second run through the launcher's default 48-channel CPU affinity (using the
already-capable comparison executable with the same compile settings) received 166,527,662
packets with 152,911 missing (0.09174%). Missing advanced in a few steps but was flat over the
final five-second report. `ReserveReadWaits` remained below 150 per FPGA and
`EmptyBufferWaits` stayed zero. The normal target must receive `cap_net_raw` again after its
final rebuild before it can be run with ibverbs. The FPGAs were retuned to the proven
40-channel range (176--215) after this comparison.

**2026-09-30 54-channel throughput work:** With the same boot and calibrated delay
file, the four FPGAs offer about 3.125 million packets/s at channels 169--222.
The retained `starspin_4_54_capture_only` target runs the complete CPU capture
and packet assembly path, then immediately releases each finished input buffer.
The original configuration received about 2.50 million packets/s. Native CPU
code and smaller per-NIC rings raised this to about 2.81 million packets/s.
Batching per-channel completion watermark publication then received 62,501,586
packets in 20 seconds, matching the full offered rate; `Missing` stayed at its
46,624 startup value after the first report. The normal four-FPGA 54-channel
and higher targets now use that per-NIC worker configuration by default.

GPU input allocation also changed for normal Starspin targets. The samples and
scales share a 2 MiB-aligned slab on the GPU's NUMA node, advised for transparent
huge pages and registered with CUDA. The allocator falls back to the prior CUDA
pinned allocation if the hugepage path is unavailable. A 54-channel synthetic
pipeline benchmark measured 78.55 blocks/s with automatic NUMA placement,
versus 50.75 blocks/s with the old pinned allocation; the live required rate is
about 56.5 blocks/s. The same synthetic benchmark at 60 channels measured
70.19 blocks/s, also above the required rate. Six output-equivalence tests passed.
With the visibility D2H copy included every 56 blocks, 54 channels sustained
78.72 blocks/s and 11.41 GB/s H2D input, plus about 0.045 GB/s average D2H;
60 channels sustained 70.50 blocks/s and 11.35 GB/s H2D plus about 0.045 GB/s
D2H. A visibility-sized standalone D2H comparison measured 7.9 GB/s into
ordinary pinned memory and 13.2 GB/s into GPU-local hugepage-backed registered
memory. Enabling the latter in `SingleHostMemoryOutput` raised full-pipeline
54-channel input-equivalent throughput from 11.405 to 11.422 GB/s, and
60-channel throughput from 11.350 to 11.369 GB/s. D2H is only one visibility
copy per 56 input blocks, so the overall gain is small.

The full live 54-channel run received 93,717,476 packets in 30 seconds,
completed 1,696 GPU blocks, had zero empty-buffer waits, and held `Missing` at
84,680 for its last 15 seconds. The full live 60-channel run received
104,145,738 packets in 30 seconds, completed 1,696 GPU blocks, had zero
empty-buffer waits, and held `Missing` at 65,074 after startup. Both runs
shut down cleanly and wrote HDF5 visibilities with finite, nonzero values and
increasing sequence numbers. Artifacts are under `build/starspin_runs/` with
prefixes `full_54_thp_batch_20260930` and `full_60_thp_batch_20260930`.

The subsequent full GPU benchmark sweep with hugepage input and output reached
66.19 blocks/s (11.37 GB/s H2D input) at 64 channels, 59.11 blocks/s
(11.42 GB/s) at 72 channels, and 53.07 blocks/s (11.39 GB/s) at 80 channels.
The 80-channel result is below the approximately 56.5-block/s live cadence.

The 64-channel live run (`full_64_thp_batch_20260930`) completed 30 seconds,
received 110,993,417 packets and handed off 1,695 GPU blocks with no
empty-buffer waits. Missing rose late to 152,954, so it is not yet a proven
long-duration lossless setting. Its HDF5 visibilities were finite and nonzero
with increasing sequence numbers. The 72-channel run initially failed before
capture because its HDF5 visibility chunk exceeded the HDF5 4 GiB limit.
`HDF5VisibilitiesWriter` now caps chunks near 256 MiB; all six writer tests
passed. The subsequent 30-second live run (`full_72_thp_batch_hdf5fix_20260930`)
handed off 1,696 blocks with zero empty-buffer waits and wrote valid HDF5 data,
but had 534,343 missing packets. Its last four full integrations had none.

A five-minute 72-channel run (`full_72_long_20260930`) exposed sustained loss:
1,185,805,012 received and 64,277,722 missing, averaging 5.14% missing per
full HDF5 integration. All 16,955 buffer handoffs progressed and there were
zero empty-buffer waits; per-NIC ring claim/read gaps often reached the 6,144
slot limit. This run used workers on CPUs `3,4,5,6`, although the NIC rings
were allocated on NUMA nodes `0,1,1,1`.

An A/B run with workers `3,16,17,18` on the NICs' NUMA nodes
(`full_72_long_numa_20260930`) using those NUMA-local
workers was worse: 504,381,997 received, 162,362,645 missing, and 24.36% mean
missing per full integration. Its rings also repeatedly filled. Worker locality
alone therefore does not resolve the 72-channel throughput deficit. A fresh
synthetic full GPU pipeline run still sustained 59.08 blocks/s (11.41 GB/s
input), only about 4.6% above the live block cadence. The launcher's
channel-independent default is now workers `3,4,5,6` on node 0: the
120-second capture-only run on those CPUs received 499,999,899 packets and
held `Missing` at 71,743 after startup, while the same host-only path with
NUMA-local workers received 383,913,953 packets and had 116,145,068 missing.
Both runs did normal ibverbs receive and full correlation-input packet assembly,
then released each completed buffer without GPU work. Repost-only and ring-only
ibverbs benchmarks each received about 4.17 million packets/s at 72 channels.
With a separate H2D-only benchmark pinned to CPU 7, the host-only capture
accumulated about 7.6 million missing packets in 60 seconds. This is evidence
of resource contention even without a launch-thread affinity collision.

The full 72-channel pipeline with packet workers on CPUs `3,4,5,6` and the
coordinator, GPU feeder, and writer confined to CPUs `7-9` completed 120
seconds (`full_72_feeder_isolated_20260930`) with 498,514,304 received and
1,543,628 missing (0.31% of expected); its HDF5 output was finite, nonzero,
and sequence ordered. The launcher now applies this CPU layout by default at
every channel count. A five-minute repeat with the launcher default
(`full_72_pipeline_affinity_5min_20260930`) received 1,227,343,762 packets,
reported 22,730,000 missing, and averaged 1.82% missing across 302 full HDF5
integrations. Of those, 163 had zero missing, but the worst lost 38.24%.
The missing count rose mostly in short bursts that filled the packet rings;
buffer handoffs continued at the expected cadence and `EmptyBufferWaits` stayed
zero. This is an improvement over the 5.14% five-minute average without
pipeline CPU isolation, but 72 channels is not yet a reliable observing mode.
The H2D-only contention test and the host-only A/B together point to packet
copy throughput under concurrent GPU traffic as the main limit. The exact
cause of the residual bursts remains unmeasured.
After the trials, all four FPGAs were restored to channels 176--215 (40
channels, centre 196); a second `configure-fpga.sh -m tune` pass reported
matching settings for boards 0--3. The calibrated delay file is still from
this boot and does not need repeating until the next observing session.

Cache variables relevant to starspin specifically (`apps/CMakeLists.txt`):
- `NR_OBSERVING_PACKET_WORKER_THREADS` (default 3) — CPU worker threads for packet reassembly.
- `NR_OBSERVING_PACKET_BUFFERS` (default 8) — host packet-assembly buffer count.
- `SPATIAL_NATIVE_ARCH` (default OFF) — enables `-march=native` globally. The
  four-FPGA 54-channel and higher Starspin targets enable native host code
  locally because the direct-NIC live capture benchmark improved from about
  2.50 to 2.77 million packets/s with this option, before batching took it
  to the full offered rate. This target-specific result supersedes earlier
  measurements on the older coordinator/ring architecture.

**Do not delete/recreate `build/`** — a from-scratch reconfigure on this host needs the
container-specific CMake flags from `CLAUDE.md`'s main build section (`CUTENSOR_ROOT`, the
linker `--sysroot` workaround for the glibc version mismatch, `-DCMAKE_CUDA_ARCHITECTURES=89`,
`-DIBVERBS_LIBRARY:FILEPATH=` if ibverbs-dev isn't installed, etc.) which are already baked into
the existing cache; reconfiguring in-place with just `-D<VAR>=<value> .` preserves all of that.

### Temporary loss-attribution build (40 channels)

`SPATIAL_DIAGNOSTICS` is a compile-time CMake option and defaults to `OFF`. A diagnostic
40-channel executable prepared on 2026-09-21 is preserved as
`build/apps/starspin_4_40.diagnostic` (SHA-256
`c300553c8fdceb579e0718d4371c54c2e99e90d5a8d956ebcfe385c03e6cad50`). It reports cumulative
missing packets by FPGA index plus per-QP completion/context-switch counters. With the normal
NIC order, indexes 0,1,2,3 mean `enp94s0np0`, `enp134s0np0`, `enp175s0np0`, and `enp216s0np0`.

Grant capability before using the copied diagnostic executable:

```bash
sudo setcap cap_net_raw+ep \
  /data/BLACKMESA_1/sma075/cuda-spatial-filtering/build/apps/starspin_4_40.diagnostic
```

At the current pause point the CMake cache has diagnostics enabled. After diagnosis, configure
`-DSPATIAL_DIAGNOSTICS=OFF`, rebuild `starspin_4_40`, and restore `cap_net_raw` on that rebuilt
production executable. See `docs/ingest-optimization-handoff.md` for the exact resume sequence.

## Run: known-good baseline (0 socket-level drops, 24ch)

The critical, non-obvious piece: **set explicit CPU affinity for capture and worker threads.**
Without it, the OS scheduler is free to place capture/worker/main threads on cores that are
*already* saturated by unrelated NIC softirq/NAPI processing (measured on this host: cores
**2, 9, 12** sit at 85-96% `%soft` *even with no app running at all* — background FPGA traffic
processed regardless of whether anything is listening). Getting this wrong causes 40%+ packet
loss even though the same binary reaches 0 drops with the right pinning.

```bash
cd /data/BLACKMESA_1/sma075/cuda-spatial-filtering/build/apps
env CUDA_HOME=/data/BLACKMESA_1/sma075/miniconda3/targets/x86_64-linux \
  SPATIAL_CAPTURE_CPUS="1,11,13,15" SPATIAL_WORKER_CPUS="3,4,5" \
  ./starspin_4_24 \
  --capture-backend kernel \
  --network-interface enp94s0np0,enp134s0np0,enp175s0np0,enp216s0np0 \
  --min_freq_channel 184 \
  --obs-length 600 \
  --delay-file alveo_delays.json \
  --gains weights.json \
  --eigenvalue-num-filename nr-signal-eigenvalues.json \
  --stream-antenna-map /data/BLACKMESA_1/sma075/cuda-spatial-filtering/stream_antenna_map.json \
  --vis_output_file /path/to/output.hdf5 \
  > /path/to/run.log 2>&1
```

Notes on the flags:
- `SPATIAL_CAPTURE_CPUS` / `SPATIAL_WORKER_CPUS`: comma lists, one entry per capture thread (in
  NIC order) / worker thread. **The safe core set on this host is anything except 2, 9, 12** —
  re-measure with `mpstat -P ALL 1 5` (idle, no app running) if the FPGA cabling, NIC firmware,
  or RSS config ever changes, since the hot cores are wherever each live NIC's single RSS queue
  happens to hash to.
- `--stream-antenna-map`: **without this flag the pipeline silently falls back to the hardcoded
  legacy `AntennaMapRegistry`**, not the corrected/swapped antenna wiring in
  `stream_antenna_map.json`. Check the `mapping_source` column of the `.streams.csv` audit file
  every run writes (`legacy_registry` vs `stream_antenna_map`) to confirm which was used. Easy to
  forget — always pass this flag for real data-taking runs.
- `--min_freq_channel`: for the 24-channel target, must leave room for 24 channels above it
  (e.g. 184 → channels 184-207).
- Write `--vis_output_file`/log output somewhere with real disk headroom, **not `/tmp`** — it's a
  small (4.9GB), shared filesystem on this host, and a single 10-minute 24ch run's HDF5 output is
  several GB. Use a path under the project's own disk (e.g. `build/starspin_smoke_out/`, 4.7TB
  free) instead.

Expected result with this config at 24ch: 0 `NICDrops`, 0 `StuckUnprocessed` (i.e. our own
software fully keeps up), but a residual ~7.5-7.7% `Missing` that is a genuine NIC **hardware**
RX-ring-overflow condition (`rx_missed_errors` in
`/sys/class/net/<nic>/statistics/rx_missed_errors`, overwhelmingly on `enp94s0np0`) — happens
before packets reach the kernel's networking stack at all, so no application/socket-level change
can reach it. Current cumulative counters show both `enp94s0np0` and `enp134s0np0` are heavily
affected; take before/after snapshots around each run rather than interpreting absolute values.
Fixing that needs `ethtool -G <nic> rx <bigger>` (bigger hardware ring) or RSS/flow steering
across more queues, both root-only; see the ingest-optimization handoff for the exact sequence.

The Starspin ibverbs backend now always registers the ProcessorState packet pool and posts raw
receive work requests directly into ring slots, removing the staging-buffer memcpy. There is no
`SPATIAL_IBVERBS_ZERO_COPY` switch anymore. It requires a working RDMA device and `cap_net_raw`.
The host has four active mlx5 verbs devices and `/dev/infiniband/uverbs0`–`uverbs3`.

## Monitoring a live run

Before each live observing session, verify FPGA boot state and delays. If the
state is uncertain, reboot all selected FPGAs, `init-tune` at eight channels,
solve fresh delays with all six FPGA pairs, and require closure before retuning
to the science band. The 2026-09-29 54-channel trial using a reused delay file
stalled; a subsequent reboot and eight-channel solve changed FPGA 3's relative
offset from about -160.6 million to -17.3 million samples. The new solve closed
within one sample. Reusing a delay file without verifying the boot state can
invalidate a throughput or data-quality comparison.

`Stats:` lines print periodically to stdout:

```
Stats: Received=<N>, Processed=<N>, Missing=<N>, Discarded=<N>, FutureQueued=<N>, StuckUnprocessed=<N>, NICDrops=<N>
```

- `NICDrops` growing → our own socket-buffer-level drain isn't keeping up (fixable via
  affinity/worker count — this is what the CPU pinning above already solves).
- `StuckUnprocessed` growing *without bound* → backlog building, GPU/CPU processing falling
  behind (also an affinity/worker-count problem).
- `StuckUnprocessed` flat (not zero, but *not growing*) → a one-time transient (e.g. startup
  burst), not an ongoing rate problem.
- `Missing` growing steadily with `NICDrops=0` and `StuckUnprocessed=0` → real upstream NIC
  hardware loss (see above), not something in this codebase.

For `ibverbs`, `NICDrops=0` does not prove absence of bursts or lost receive
work requests. Check `ReserveReadWaits`, `ReserveProcessedWaits`, claim/read
gaps, future-queue sizes, and receive-slot replacement timeouts as well.

### 2026-09-29 48/54-channel comparison

After all four FPGAs were rebooted and fully initialized at channels 192–199,
`build/starspin_runs/alveo_delays_20260929_rebooted.json` solved all six pairs
with at most one sample of closure error. Readback confirmed all four boards
at each test range. With that delay file, the 20-second 48-channel control at
172–219 received 55,511,447 packets; `FutureQueued=0`, `StuckUnprocessed=0`,
and `Missing` stopped at 54,356 after startup. Its HDF5 output has 36 antennas,
666 baselines, and complete steady windows. The 30-second 54-channel run at
169–222 received only 6,177,096 packets before stalling, with growing future
queues of roughly 500–576 slots per NIC and receive-slot replacement timeouts.
Later HDF5 windows are almost entirely missing. Do not advance to 60 channels
until 54 is stable.

The normal ring has 6,144 slots interleaved across four NICs: 1,536 slots per
NIC, with 1,024 posted ibverbs receives. An isolated
`starspin_4_54_per_nic` target uses four contiguous 4,096-slot rings,
first-touched on the corresponding NIC NUMA nodes (0,1,1,1), and one processing
worker per NIC. A cap-free kernel-socket smoke run verified ring construction
and worker processing, but kernel socket drops were high, so the ibverbs A/B
test remains necessary. The new binary needs `cap_net_raw=ep` after linking.

### 2026-09-29 fresh-session live validation

Rebooted all four boards again (all four done pins HIGH), ran full `init-tune`
at 192–199 (all JESD/ADC groups OK), and solved all six FPGA delay pairs from
fresh packets. `build/starspin_runs/alveo_delays_20260929_session2.json` has
maximum closure residual **0 samples**. Each board read back the requested
169–222 range before the 54-channel test.

The 30-second ibverbs run of `starspin_4_54_per_nic` is in
`build/starspin_runs/live_54ch_per_nic_20260929_session2.*`. It received
4,666,892 packets, processed 4,654,432, and reported 81,939,104 missing.
Its four future queues reached about 3,100 entries each, `ReserveProcessedWaits`
grew to `[14161,14907,15693,15634]`, and later HDF5 windows were over 99%
missing. `NICDrops` remained zero. The larger NUMA-local rings and four
per-NIC workers did not prevent the same backpressure failure seen with the
normal 54-channel ring; do not treat this run as a throughput success or try
60 channels yet. More ring capacity only changes how long the pipeline can
absorb a service-rate deficit. The observed buffer handoffs increased by
roughly 260 per five-second report, against about 282 needed at the measured
FPGA packet cadence: a roughly 8% sustained service-rate deficit. Releases
tracked handoffs with all eight host buffers eventually occupied. Measure the
GPU pipeline's per-stage time before changing the receive path again.

The shared `/data/BLACKMESA_0/lambda/starspin/scripts/run_starspin.sh` command
was also tested on live data after copying this session's delay solution. At
40 channels, 288–327, its 15-second run
`/data/BLACKMESA_0/lambda/starspin/build/starspin_runs/live_40_shared_20260929_session2.*`
received 34,686,140 packets and processed 34,686,115. Missing stayed fixed
at 43,528 after startup; HDF5 visibility windows from the third onward had
zero missing packets. All four boards were then restored to 176–215, with
readback matching on each.

### Direct per-NIC workers (pending live validation)

`starspin_4_54_per_nic_direct` is a separate binary that retains the four
NUMA-local receive rings but removes the per-batch dispatch and barrier. Each
worker continuously consumes the committed prefix of its own NIC ring and
publishes its read cursor. The remaining processor thread handles only future
packet draining and buffer completion/switching; a shared/exclusive lock
prevents it from completing a host window while any worker is copying into
that window. The launcher assigns workers to CPUs `3,16,17,18` on the NICs'
NUMA nodes `0,1,1,1`. Synthetic tests verify exact samples/scales under four
independent producers, zero-missing completion across three windows, and a
fully committed ring whose start and end indices coincide. The
binary requires its own `cap_net_raw=ep` grant before an ibverbs trial. The
first live result is below.

### 2026-09-30 direct-worker live trial

The new session started with an FPGA reboot. FPGA 3's first attempt reported a
configuration-memory boot error despite the wrapper's zero exit status; a
single-board retry reached DONE HIGH. All four boards then passed full
eight-channel `init-tune` at 192–199, including 20/20 ADC group checks. The
fresh six-pair solution is
`build/starspin_runs/alveo_delays_20260929_direct54_boot.json`, with maximum
closure residual one sample. Do not reboot again between captures in this
same session.

`starspin_4_54_per_nic_direct` was run for 30 seconds at 169–222 after all
four boards read back the requested range. Artifacts are
`build/starspin_runs/live_54ch_direct_20260930_boot.*`. It received 6,143,486
packets and processed 6,131,054; `Missing` rose to 79,468,389, the four
future queues reached about 3,100 entries each, and later HDF5 windows were
99.6–99.9% missing. Host buffer handoffs rose by roughly 260 per five seconds
against roughly 282 required by the offered packet cadence. Removing the
per-batch coordinator did not make 54 channels stable. The growing
`EmptyBufferWaits` and near-full future queues indicate backpressure at or
after host-buffer completion; this run alone does not identify a specific GPU
kernel. Do not try 60 until 54 is stable.

As a same-boot control, the known 48-channel binary ran for 20 seconds at
172–219 using the same delay solution:
`build/starspin_runs/live_48ch_control_20260930_directboot.*`. It received
55,556,621 packets and processed 55,556,525. `FutureQueued=0`,
`StuckUnprocessed=0`, `NICDrops=0`, and `Missing=9010` including startup.
The last five HDF5 windows had zero missing packets. FPGAs 0–3 were then
restored to 176–215 and all four read back matching settings.

Per-NIC hardware ring-drop ground truth (bypasses this app entirely):
```bash
for nic in enp94s0np0 enp134s0np0 enp175s0np0 enp216s0np0; do
  echo "$nic: $(cat /sys/class/net/$nic/statistics/rx_missed_errors)"
done
```
This is a **cumulative-since-boot** counter — always diff two snapshots around a run, never read
it as a single absolute number.

## Debugging startup stalls

`FineChannelizer` now shares one stateless `gpu-filter` object across all coarse channels, so it
JIT-compiles the identical PFB kernel once rather than once per channel. On the 48-channel target
this reduced the observed filter-construction phase from roughly six minutes to roughly ten
seconds. A fresh JIT can still vary with host load; wait for `Setup completed. Ready to receive!`
before measuring the capture interval.

## Experimental: `libvma` kernel-bypass offload

Tried as a lever to reduce the residual hardware ring loss. Loads and engages real hardware
offload (confirmed via absence of the "Offloaded resources are restricted to root..." warning),
but has not beaten the plain kernel-backend baseline in any tested configuration — see the
architecture-review docs for the full chronicle (patchelf `DT_NEEDED`/RPATH recipe to get it
loading without `sudo` per run, the `CAP_NET_ADMIN`-gated `SO_BUSY_POLL` dead end, the unexplained
periodic-burst behavior). Not recommended for production capture on this host as of 2026-09-20.
