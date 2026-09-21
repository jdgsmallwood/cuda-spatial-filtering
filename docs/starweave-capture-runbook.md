# Starweave capture runbook

Practical build/run commands for the `starweave_{NR_FPGAS}_{NR_CHANNELS}` targets
(correlation-only, visibilities-only counterpart to `observe` — see `apps/starweave.cu` /
`LambdaStarweavePipeline`), captured from live 4-FPGA/24-channel tuning on `blackmesa`
(2026-09-20). See `docs/architecture.md` for the pipeline's data-flow diagrams; this file is
the operational "how do I actually build and run it" companion, plus the specific environment
variables that turned out to matter for real packet-loss-free capture on this host.

For a plain-English description of the architecture and the changes that took the system from
24 to 40 channels, read `docs/starweave-24-to-40-explained.md`.

## Easiest operator command (four FPGA, 40 channels)

The repository-local wrapper supplies every required path and the proven defaults:

```bash
cd /data/BLACKMESA_1/sma075/cuda-spatial-filtering
scripts/run_starweave.sh --duration 600
```

It defaults to `starweave_4_40`, ibverbs direct-to-ring receive, interfaces
`enp94s0np0,enp134s0np0,enp175s0np0,enp216s0np0`, channels 176--215, capture CPUs
`1,11,13,15`, workers `3,4,5`, the current delay/gain/config/mapping files, and timestamped
outputs under `build/starweave_runs/`. It checks input files, NIC existence, free disk space,
and `cap_net_raw` before starting, prints the complete resolved command, and tees console output
to the run's log.

Useful variants:

```bash
# Safe command preview; does not touch the backend.
scripts/run_starweave.sh --dry-run --duration 600

# Prepared loss-attribution binary (after granting cap_net_raw to that copy).
scripts/run_starweave.sh --diagnostic --duration 120

# Explicit output name and directory.
scripts/run_starweave.sh --duration 600 --name science_40ch \
  --output-dir /path/with/space

# Retune FPGAs 0-3 first. This is deliberately opt-in.
scripts/run_starweave.sh --configure-fpga --duration 600
```

Run `scripts/run_starweave.sh --help` for all path, affinity, channel, and binary overrides.
The launcher was adapted from `/data/BLACKMESA_0/lambda/scripts/correlate_lambda.sh`; nothing in
`/data/BLACKMESA_0/lambda/scripts` was changed.

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
`build/starweave_smoke_out/ibverbs_32ch_oneshot_600s`.

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
cmake --build . --target starweave_4_24 -- -j$(nproc)
```

For the current 32-channel investigation, substitute `starweave_4_32`. Every rebuild replaces
the executable and clears its file capability; restore it before an ibverbs run:

```bash
sudo setcap cap_net_raw+ep \
  /data/BLACKMESA_1/sma075/cuda-spatial-filtering/build/apps/starweave_4_32
getcap /data/BLACKMESA_1/sma075/cuda-spatial-filtering/build/apps/starweave_4_32
```

The target grid now also includes `starweave_4_40`. Its current test range starts at channel 176
and ends at 215. Apply the same capability command with `starweave_4_40` as the filename and use
`--min_freq_channel 176`.

The first 60-second 40-channel ibverbs zero-copy run succeeded at 2.313595 Mpps (about 6.154
GB/s): 138,815,720 received, 138,814,912 processed, zero `NICDrops`, zero stuck work, and
73,522 missing (**0.052935705%**, including startup). Its artifacts use the prefix
`build/starweave_smoke_out/ibverbs_40ch_oneshot_60s`.

A second 60-second validation of the exact checkpoint binary completed with 138,851,828
received, 138,850,560 processed, 50,504 missing (**0.036359361%**, including startup), zero
future-queued/stuck work, and 3,389 pipeline runs. `Missing` stayed flat for the final 15 seconds
and shutdown was clean. Its artifacts use the prefix
`build/starweave_smoke_out/ibverbs_40ch_recheck_60s`.

The target grid also includes `starweave_4_48`, but it is not currently production-safe. Live
measurement showed the GPU pipeline sustaining only 55--56 of the required 60.55 blocks/s; all
eight host assembly buffers eventually become GPU-owned and receive collapses. Use the proven
40-channel target until the pre-filter GPU transforms are fused. See
`docs/ingest-optimization-handoff.md` for the evidence.

Cache variables relevant to starweave specifically (`apps/CMakeLists.txt`):
- `NR_OBSERVING_PACKET_WORKER_THREADS` (default 3) — CPU worker threads for packet reassembly.
- `NR_OBSERVING_PACKET_BUFFERS` (default 8) — host packet-assembly buffer count.
- `SPATIAL_NATIVE_ARCH` (default OFF) — enables `-march=native`, unlocking `copy_nt`'s AVX-512
  path. **Do not enable on this box** — measured as a net *regression* (this CPU, a Cascade Lake
  Xeon Silver 4210, downclocks the whole socket under sustained AVX-512, which cost more overall
  throughput than the narrower copy speedup gained — see `docs/architecture.md`'s NUMA/CPU
  section or ask for the perf-tuning memory notes).

**Do not delete/recreate `build/`** — a from-scratch reconfigure on this host needs the
container-specific CMake flags from `CLAUDE.md`'s main build section (`CUTENSOR_ROOT`, the
linker `--sysroot` workaround for the glibc version mismatch, `-DCMAKE_CUDA_ARCHITECTURES=89`,
`-DIBVERBS_LIBRARY:FILEPATH=` if ibverbs-dev isn't installed, etc.) which are already baked into
the existing cache; reconfiguring in-place with just `-D<VAR>=<value> .` preserves all of that.

### Temporary loss-attribution build (40 channels)

`SPATIAL_DIAGNOSTICS` is a compile-time CMake option and defaults to `OFF`. A diagnostic
40-channel executable prepared on 2026-09-21 is preserved as
`build/apps/starweave_4_40.diagnostic` (SHA-256
`c300553c8fdceb579e0718d4371c54c2e99e90d5a8d956ebcfe385c03e6cad50`). It reports cumulative
missing packets by FPGA index plus per-QP completion/context-switch counters. With the normal
NIC order, indexes 0,1,2,3 mean `enp94s0np0`, `enp134s0np0`, `enp175s0np0`, and `enp216s0np0`.

Grant capability before using the copied diagnostic executable:

```bash
sudo setcap cap_net_raw+ep \
  /data/BLACKMESA_1/sma075/cuda-spatial-filtering/build/apps/starweave_4_40.diagnostic
```

At the current pause point the CMake cache has diagnostics enabled. After diagnosis, configure
`-DSPATIAL_DIAGNOSTICS=OFF`, rebuild `starweave_4_40`, and restore `cap_net_raw` on that rebuilt
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
  ./starweave_4_24 \
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
  several GB. Use a path under the project's own disk (e.g. `build/starweave_smoke_out/`, 4.7TB
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

The Starweave ibverbs backend now always registers the ProcessorState packet pool and posts raw
receive work requests directly into ring slots, removing the staging-buffer memcpy. There is no
`SPATIAL_IBVERBS_ZERO_COPY` switch anymore. It requires a working RDMA device and `cap_net_raw`.
The host has four active mlx5 verbs devices and `/dev/infiniband/uverbs0`–`uverbs3`.

## Monitoring a live run

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
