# Handoff: four-FPGA kernel ingest optimization

Updated 2026-09-20, approximately 04:19 UTC. This is the continuation entry point.

## 2026-09-21 continuation status

Work has moved to the ibverbs zero-copy path because it produced the substantial throughput
improvement sought here: the 4-FPGA/24-channel target sustained about 1.39 Mpps with essentially
zero loss. The active 32-channel problem and exact live-run procedure are now maintained in
`docs/ibverbs-zerocopy-handoff.md` and `docs/starweave-capture-runbook.md`; use those as the
primary continuation documents.

The latest change removed all runtime buffer-timeline re-seeding. The first valid packet now
initializes the four FPGA boundaries exactly once through the configured delay file. Startup
packets older than that boundary may be discarded, by explicit operator choice, but the timeline
cannot be rewritten after publication. Signed delay arithmetic is covered with the live-scale
FPGA-2 offset. `ProcessorTests` now passes **18/18**, including the new huge-delay one-shot
bootstrap test and both legacy/mailbox handoff cases. The subsequent live 60-second
`starweave_4_32` ibverbs zero-copy run at `--min_freq_channel 180` completed without freezing:
111,037,786 received, 111,037,568 processed, zero `NICDrops`, zero stuck work, and 3,390 pipeline
runs. Its 4,640 startup discards stayed flat for every report. See the ibverbs handoff for the
rate interpretation and saved artifacts. `Missing` remained flat for the final roughly 40 seconds,
which is the operator's acceptance criterion. An independent 10-second NIC sample measured only
1.8582 Mpps offered, explaining why the application observed 1.8506 rather than the nominal
1.984 Mpps target without indicating an ingest-capacity shortfall.

The later ten-minute confirmation received 1,111,114,882 packets with 32,461 missing:
**0.002921395% missing**, including startup. It had zero `NICDrops`, zero stuck work, 33,908
pipeline runs, and a clean shutdown. Missing stayed flat for the final two minutes.

A 40-channel target was subsequently added and tested over channels 176–215. Its 60-second
ibverbs zero-copy smoke run sustained 2.313595 Mpps (about 6.154 GB/s) with zero `NICDrops` and
zero stuck work. It reported 73,522 missing out of 138,815,720 received (**0.052935705%**,
including startup). See the ibverbs handoff for the detailed counter timeline and artifacts.

## User objective and constraints

Improve packet capture and CPU processing before the GPU pipeline, ideally by multiples,
and prove gains. Target: **32 × 15,500 × 4 = 1,984,000 packets/s**, approximately
**5.277 GB/s at 2,660 bytes/packet**. Focus on four FPGAs.

- Ignore ibverbs unless necessary for a substantial throughput improvement.
- **No adaptive receive batch size.** Keep receive batch capacity fixed.
- User authorized implementation, benchmarking, and rebuilding in repository `build/`.
- **Do not delete and recreate `build/`.** It has been preserved throughout.
- Do not spawn agents unless user explicitly authorizes; current instructions disallow proactive delegation.
- Latest user request: write a handoff before their usage runs out so another model can continue.

## State at handoff

**The target is not yet proven through capture + reassembly + H2D. No new multiple-fold speedup is established.**

Stage 1 implemented an optional coordinator/worker mailbox handoff, retaining the existing
barrier and default legacy path for A/B comparison. CPU-only comparisons show no clear win.
Adding real asynchronous H2D exposed a much more important correctness problem in BOTH modes:
the worker path passed a zero horizon to packet placement, disabling future-packet deferral.
This was already present in HEAD as well as the user's modified worker path.

Fixed the horizon snapshot in `process_work_range()`. A first short H2D check then reported
zero missing packets, but throughput was only **1.157 Mpps**, below target, and there were
**122 future-queued / 122 stuck** packets. That run overlapped the end of test compilation;
do not treat it as a clean performance measurement. Investigate the residual stuck count and
repeat without competing compilation before drawing conclusions.

**Build succeeded; all 17 ProcessorTests passed**, including three new cases described below.
No production default was changed to mailboxes. No queued-job/per-buffer asynchronous worker
architecture has been implemented yet. No physical NIC capture was run in this session.

## Continuation update (2026-09-20)

The existing `build/` tree was reconfigured with the conda CMake and rebuilt without deleting
or recreating it. `ProcessorTests` completed **17/17 passed**. The new four-FPGA threaded test
covers both legacy and mailbox handoffs with partial fixed batches, delayed commits, ring wrap,
and exact samples/scales checks. The worker-range horizon regression also passes.

The horizon fix is now present: `process_work_range()` snapshots `global_max_end_seq` before
packet placement. Without this, its local all-zero horizon disabled future deferral, which is
unsafe when asynchronous H2D recycling lets producers run ahead.

Two clean five-second H2D files were run concurrently, so they are **not a valid A/B**:

| run | processed Mpps | missing | future/stuck | note |
|---|---:|---:|---:|---|
| `h2d-legacy-postfix.txt` | 0.739 | 0 | 132 / 132 | competed with the other process |
| `h2d-mailboxes-postfix.txt` | 1.429 | 0 | 0 / 0 | competed with the other process |

They establish no performance conclusion. The low rates show that this synchronous H2D/null
benchmark is dominated by buffer recycling or GPU scheduling under contention. Repeat
sequentially, with no compilation or competing benchmark, before interpreting mailbox results.
The nonzero legacy `stuck` count occurred with zero missing; determine whether it is expected
shutdown tail cleanup or a retained-slot bug.

Sequential follow-up results are also inconsistent and expose an unresolved benchmark issue:

| run | processed Mpps | missing | future/stuck | dispatch wait / total |
|---|---:|---:|---:|---:|
| `h2d-legacy-sequential.txt` | 1.857 | 0 | 0 / 0 | 3.258 s / 4.965 s |
| `h2d-mailboxes-sequential.txt` | 0.147 | 0 | 37 / 37 | 15 ms / 159 ms |

These were sequential, but the mailbox case collapsed to 0.147 Mpps and stopped producing
after only 22 buffers. That is a functional liveness problem in the benchmark or mailbox
shutdown/recycling path, not evidence of a useful optimization. Do not enable mailboxes in
production. Next debugging should instrument producer reservation waits, buffer release
callbacks, and worker exit/request generations; inspect CUDA errors and stream callbacks.
The legacy H2D run reached 1.857 Mpps, below the 1.984 Mpps target, with no missing/stuck
packets, so the H2D benchmark currently has only ~0.94× target capacity and needs more work.

An isolated sequential CPU-only mailbox run, `cpu-mailboxes-sequential.txt`, reached
**4.229 Mpps**, with zero missing/discarded/future/stuck packets. Therefore the mailbox
implementation is not intrinsically deadlocking in the CPU-only path; its collapse appears
only when combined with the current synchronous H2D/null-pipeline recycling. Treat this as an
interaction/lifecycle bug, not a reason to optimize mailbox atomics further. The next useful
step is to instrument `NullPipeline::execute_pipeline`, callback completion, and
`advance_to_next_buffer()` around buffer reuse, or remove the H2D benchmark's synchronous
mode in favor of the production asynchronous feeder before evaluating worker handoff.

## Starweave 4×24 live mailbox run (2026-09-20)

Added a runtime opt-in in `apps/starweave.cu`: set `SPATIAL_WORKER_MAILBOXES=1` to enable
`ProcessorState::use_worker_mailboxes`; unset it (or set `0`/`false`) to retain the default
handoff. Built the existing target successfully:

```bash
/data/BLACKMESA_1/sma075/miniconda3/bin/cmake --build build \
  --target starweave_4_24 -j 4
```

Ran the runbook command with the required affinity and **`--min_freq_channel 184`**:

```bash
SPATIAL_WORKER_MAILBOXES=1 \
SPATIAL_CAPTURE_CPUS="1,11,13,15" SPATIAL_WORKER_CPUS="3,4,5" \
./starweave_4_24 --capture-backend kernel \
  --network-interface enp94s0np0,enp134s0np0,enp175s0np0,enp216s0np0 \
  --min_freq_channel 184 --obs-length 60 \
  --delay-file alveo_delays.json --gains weights.json \
  --eigenvalue-num-filename nr-signal-eigenvalues.json \
  --stream-antenna-map /data/BLACKMESA_1/sma075/cuda-spatial-filtering/stream_antenna_map.json \
  --vis_output_file starweave_mailbox_runs/starweave_mailbox_4_24_60s.hdf5
```

Output is in `build/apps/starweave_mailbox_runs/`:
`starweave_mailbox_4_24_60s.log` and the 1.7 GB HDF5 result. Startup JIT took about four
minutes; the run then completed normally after the 60-second observation window.

Final application counters:

| Counter | Value |
|---|---:|
| Received | 77,819,863 |
| Processed | 77,819,524 |
| Missing | 5,535,455 (≈7.11% of received) |
| Discarded | 0 |
| Future queued | 0 |
| Stuck unprocessed | 0 |
| NICDrops (socket ancillary) | 0 |

The mailbox did **not** stop packet loss. It did keep the CPU-side application healthy:
processed trailed received by only 339 packets, with no stuck backlog and no socket-level
NICDrops. The missing count matches the runbook's known hardware RX-ring overflow condition,
which occurs before packets reach the kernel socket and cannot be fixed by worker handoff.
The cumulative hardware counters observed after this run were:
`enp94s0np0=131462301`, `enp134s0np0=131406968`, `enp175s0np0=40145`,
`enp216s0np0=16743`; no pre-run snapshot was taken, so these absolute values cannot be
converted to this run's hardware delta. Take before/after snapshots for the next run.

This is a mailbox run only; there is no same-window default-handoff live control run yet.
The result already answers the loss question: the dominant loss is upstream hardware RX
overflow, while the mailbox path neither introduced application backlog nor removed it.

## What `Missing` means and how to fix it

### Could this be thread scheduling?

There are two different schedulers/threads to distinguish:

* The four application capture threads are explicitly pinned by
  `SPATIAL_CAPTURE_CPUS` in NIC order. In the tested command this is
  `enp94s0np0→CPU 1`, `enp134s0np0→CPU 11`, `enp175s0np0→CPU 13`, and
  `enp216s0np0→CPU 15`. CPU 1 is on NUMA node 0 (the enp94 NIC's node), and
  CPU 11 is on NUMA node 1 (the enp134 NIC's node), so these two user threads
  are not accidentally sharing one CPU.
* That affinity does **not** select the CPU which services the NIC's RX queue.
  Each NIC exposes 20 RX queues. The current kernel IRQ affinity assigns those
  queues one-per-CPU across CPUs 0–19 (with queue numbering rotated between
  NUMA nodes). RPS is disabled (`queues/rx-*/rps_cpus` is zero), so hardware RSS
  plus the IRQ/NAPI CPU determines where a packet is drained before `recvmmsg`.

Therefore it is possible for a single FPGA UDP flow to hash to one RX queue
whose IRQ/NAPI CPU is one of the already-hot CPUs (historically 2, 9, or 12),
while the corresponding application capture thread is pinned to a different,
apparently idle CPU. That produces hardware `rx_missed_errors` without
`NICDrops`, because the loss happens before the socket. It also explains why
the symptom can follow enp94/enp134 rather than a particular application
worker. This remains a hypothesis until per-queue IRQ/RSS activity is measured:
enp175/enp216 also have queues on those CPUs.

The capture-thread mapping is relevant to software loss. The current mapping is
local for enp94 (node 0) and enp134/enp175/enp216 (node 1), while the three
production workers are pinned to CPUs 3–5 (node 0, chosen for the GPU). This
creates remote-NUMA reads for three NICs, but the live mailbox run had
`NICDrops=0`, `StuckUnprocessed=0`, and almost equal Received/Processed, so it
does not explain the observed `Missing` counter. It remains a throughput lever
only after hardware RX loss is removed.

Required proof/fix sequence:

1. Around a short fixed-rate run, snapshot each NIC's `rx_missed_errors`, then
   collect per-IRQ deltas from `/proc/interrupts` (or driver per-queue counters
   from `ethtool -S`). Also record `ethtool -x <nic>` (RSS indirection),
   `ethtool -l <nic>` (queue limits), and `ethtool -S <nic>`. This identifies
   whether one queue/CPU carries the enp94/enp134 loss.
2. If one queue/CPU is hot, move only that NIC's RX IRQs to isolated CPUs on
   the NIC's NUMA node, or change the RSS indirection table. Moving the
   application capture thread alone does not move NAPI work.
3. If one UDP 5-tuple hashes to one queue, RSS cannot distribute it across
   queues. Add source-port/flow entropy at the FPGA (preferred), or use NIC
   ntuple/flow-steering rules if the stream format cannot change.
4. Increase the hardware RX ring (`ethtool -G`, root-only) after IRQ/RSS is
   sane. This absorbs short bursts but cannot compensate for a permanently
   overloaded queue.

No application worker-handoff change can repair packets dropped at this stage;
the mailbox experiment only changes work after `recvmmsg` has already received
the datagram.

### Would libibverbs make this disappear?

Not automatically. The CPU-memory ibverbs path replaces the UDP socket,
`recvmmsg`, socket receive queue, and kernel UDP/NAPI delivery with a raw-packet
QP, pre-posted receive work requests, and a busy-polled completion queue. That
can remove the application-side scheduling/NAPI/socket-queue bottleneck and
make CPU capture much more deterministic. It still uses the same physical NIC
and still requires the NIC to have available receive descriptors/buffers; a
busy-polled QP can overrun if its receive queue or hardware resources are
exhausted. Ring/flow/IRQ measurements therefore remain necessary.

The current ibverbs implementation has one QP per NIC, 1,024 pre-posted MTU
receive frames, and then performs a CPU `memcpy` into the shared packet ring.
It is not compiled or validated on this host (libibverbs headers/devices are
absent here). The GPUDirect variant is even more environment-dependent
(GPUDirect RDMA and `nvidia-peermem`) and is also unvalidated. Consequently,
ibverbs is a candidate experiment for eliminating kernel scheduling/socket
loss, not evidence that the enp94/enp134 hardware loss will vanish or that the
4-FPGA target will be met.

The safe order is: first measure per-NIC hardware deltas and queue/IRQ load;
then fix RSS/IRQ/ring configuration. Only if the kernel path remains the
limiter should we build and A/B test ibverbs on the same fixed-rate stream,
checking `rx_missed_errors`, QP completion errors, application Missing, and
throughput separately.

### Zero-copy ibverbs implementation update

The CPU-memory ibverbs backend now has an opt-in direct-ring mode. Set
`SPATIAL_IBVERBS_ZERO_COPY=1` with `--capture-backend ibverbs` in `starweave`.
After the `ProcessorState` ring is allocated, the backend registers the
contiguous packet pool with ibverbs, reserves ring slots per capture thread,
and posts receive WRs directly to those slot addresses. Completions commit the
slot metadata and reserve a replacement slot; the former staging-buffer
`memcpy` is not executed. The existing staging path remains the default.

`starweave_4_24` builds successfully with `HAVE_IBVERBS` enabled. The earlier
claim that runtime validation was blocked was caused by the restricted tool
environment, not the host: outside that sandbox the four `/dev/infiniband/uverbs*`
nodes exist, `ibv_devinfo` sees `mlx5_0`–`mlx5_3`, and RDMA links map to all
four NICs (`enp94s0np0`, `enp134s0np0`, `enp175s0np0`, `enp216s0np0`). A real
A/B run is therefore feasible; it still needs to measure QP completion errors,
hardware counters, Missing, and throughput.

There is supporting host-level evidence: the current `/proc/softirqs` snapshot
has very large cumulative `NET_RX` counts on CPU 2 (about 217 million), CPU 9
(about 407 million), and CPU 12 (about 241 million), while most CPUs are below
a few million. These are cumulative counters, not a per-run attribution, but
they confirm that the three CPUs called out in the runbook really are carrying
disproportionate network-softirq work. A per-run IRQ/queue snapshot is still
needed to prove which NIC's flow is landing there.

The application does not attribute `Missing` to a specific NIC. Each completed correlation
buffer has an arrival bit for every `(frequency channel, FPGA source, packet index)`. A zero
arrival bit increments `packets_missing` and its scales are zeroed before the GPU handoff.
Therefore the 5,535,455 value is a count of expected stream packets absent from the assembled
windows across all four FPGA sources, not a socket-drop counter. `Missing / Received` is only
an approximate loss fraction because the denominator omits absent packets.

The config maps FPGA 0→`enp94s0np0`, FPGA 1→`enp134s0np0`, FPGA 2→`enp175s0np0`, and FPGA
3→`enp216s0np0`. Current cumulative `rx_missed_errors` values are `131485041`, `131429540`,
`40145`, and `16743` respectively. These counters are cumulative since boot; only a
before/after pair around one run proves that run's per-NIC loss. The older runbook wording
saying loss was overwhelmingly on `enp94s0np0` is too narrow: current totals show
`enp134s0np0` is also heavily affected, while the node-1 NICs are nearly flat.

The packet path is:

```text
FPGA -> NIC DMA receive descriptors/ring -> NIC/NAPI -> Linux UDP socket -> recvmmsg -> ring
```

`rx_missed_errors` means the NIC could not place a packet because its hardware receive
descriptors/ring were exhausted. The packet is lost before Linux creates a socket datagram;
the application cannot recover it. `SO_RXQ_OVFL`/`NICDrops` counts a later socket-queue loss,
which is why it remained zero. Worker mailboxes only affect processing after `recvmmsg`.

Required remediation, in order:

1. Take per-NIC snapshots immediately before and after an identical 60-second run:
   `/sys/class/net/<nic>/statistics/rx_missed_errors`, `rx_packets`, and driver per-queue
   counters from `ethtool -S`. Install/use `ethtool` if it is absent in the environment.
2. As root, inspect maximum/current hardware ring sizes with `ethtool -g <nic>`, then raise
   the receive ring to the device maximum, for example `ethtool -G <nic> rx <MAX>`. Apply to
   `enp94s0np0` and `enp134s0np0` first; verify that both hardware deltas and application
   `Missing` fall. This consumes NIC DMA memory and may require persistent network config.
3. Inspect active channels and RSS with `ethtool -l/-x`. Each interface exposes 20 RX queues,
   but a single UDP 5-tuple can still hash all packets from one FPGA onto one queue. If ring
   enlargement is insufficient, create flow entropy (distinct FPGA source UDP ports per
   channel/stream) or install NIC ntuple/flow-steering rules so traffic uses multiple queues;
   duplicating application sockets alone will not reliably split one hardware RSS flow.
4. Re-run the same mailbox/default command with before/after counters. Success means hardware
   missed-error deltas approach zero and `Missing` stops growing, while `NICDrops` and
   `StuckUnprocessed` remain zero.

If the NIC maximum ring and flow steering still cannot sustain the offered rate, the remaining
options are NIC/FPGA traffic shaping or a kernel-bypass receive path (AF_XDP, DPDK, or the
existing ibverbs backend). Changing coordinator/worker handoff cannot repair packets dropped
before the kernel.

## Files changed by this work

1. `include/spatial/spatial.hpp` — this file ALREADY had user modifications; preserve them.
   - Public `use_worker_mailboxes = false`, set before threads start.
   - Optional `profile_worker_dispatch`, and coordinator-owned batch/slot/wait/total counters.
   - Cache-line-separated per-worker requested/completed generation counters and exit flag.
   - Mailbox dispatch and worker loop, preserving the existing task splitting and full barrier.
   - Legacy shared-counter path retained unchanged in behavior for comparison.
   - **Correctness fix:** `process_work_range()` now calls
     `get_global_max_packet_array(global_max)` before processing its range.
     Previously the local array remained all zeros and future deferral was disabled.
   - Worker `packets_processed` atomic increments intentionally remain the same in both modes;
     this isolates the signaling experiment.
2. `apps/bench_processor.cu`
   - `--four-fpga`: ONLY 32ch / 4 FPGA, four synthetic producers, three helpers + coordinator.
     Each producer owns one FPGA's streams because flattened stream index modulo four equals FPGA ID.
   - `--mailboxes`, `--profile-dispatch` require `--four-fpga`.
   - The old default eight-config suite still uses three producers and six helpers.
   - `--with-h2d` now copies **scales as well as samples**, with the release callback following
     both transfers on the same stream.
   - Optional thread affinity via `SPATIAL_BENCH_PROCESSOR_CPUS` (first entry) and
     `SPATIAL_BENCH_PRODUCER_CPUS` (one per producer). Worker affinity already existed as
     `SPATIAL_WORKER_CPUS`. Affinity additions are built but have NOT yet been used in an A/B sweep.
3. `tests/test_processor.cu`
   - Parameterized four-FPGA threaded test runs legacy and mailbox modes: small fixed ring,
     independent producers, varying partial work batches, delayed commits, multiple ring wraps,
     and exact comparison of every sample and scale in an incomplete window after joining.
     This tests data publication, not H2D or buffer recycling.
   - Four-FPGA worker-range regression checks that packets beyond the buffer horizon are queued,
     retain their ring slots, and do not advance completion watermarks. It targets the real
     worker range function, whereas older horizon tests exercised only the single-thread drain.
4. `scripts/profiling/run_handoff_comparison.sh`
   - Runs alternating legacy/mailbox pairs, CPU-only then H2D.
   - Usage: `bash scripts/profiling/run_handoff_comparison.sh 10 5`.
   - Saves raw outputs, binary hash, working-tree diff, CPU/GPU info and relevant environment
     into a fresh `build/handoff-ab.XXXXXX` subdirectory; does not replace build/.
   - Does not configure runtime library paths or choose CPU affinity for you.
   - Existing results predate the horizon fix; rerun after correctness investigation.
5. `docs/architecture.md`: short description of the optional mailbox path.
6. `docs/kernel-ingest-performance-plan.md`: initial architecture investigation/plan.
   Its statement about unavailable GPU/network was from sandbox visibility and is superseded
   by this handoff. Its "no production code changed" describes the initial plan stage, not now.
7. This handoff.

## Existing user changes: DO NOT discard

At start the worktree already had modifications to `CMakeLists.txt`, `apps/CMakeLists.txt`,
`apps/starweave.cu`, `config.json`, `extern/tcc`, `include/spatial/pipeline/common.hpp`,
`include/spatial/pipeline/lambda_starweave_pipeline.hpp`, and `include/spatial/spatial.hpp`.
There are many untracked observation outputs, scripts, and calibration files. Leave them alone.
The preexisting spatial.hpp change makes the coordinator process a slice itself via the
extracted `process_work_range`; do not accidentally revert it when reviewing our changes.
HEAD was `266c53697fe2f0c5ca7e73a87cd56289819510f2`. No commits made.

## Build and runtime: exact working setup

Repository: `/data/BLACKMESA_1/sma075/cuda-spatial-filtering`.

**GPU operations must run outside the Codex sandbox (`require_escalated`).** Inside it,
`nvidia-smi` fails and `/proc/net/dev` exposes only loopback. Outside it, the host has an idle
RTX 4000 Ada, 20 GB, driver 580.178.04, plus four physical FPGA NICs. This was approved by
automatic review. Do not repeat the earlier mistaken conclusion that hardware is unavailable.

Host CPU: two Xeon Silver 4210 sockets, CPUs 0–9 node 0 and 10–19 node 1, no SMT. GPU node
historically 0; verify PCI/NUMA placement if changing placement. Other user processes and
NIC softirq activity exist. Avoid compiling during performance comparisons.

Use the original **conda CMake**, not `/usr/bin/cmake` (3.25):

```bash
/data/BLACKMESA_1/sma075/miniconda3/bin/cmake -S . -B build \
  -DBUILD_TESTING=ON -DFETCHCONTENT_UPDATES_DISCONNECTED=ON
/data/BLACKMESA_1/sma075/miniconda3/bin/cmake --build build \
  --target bench_processor ProcessorTests -j 4
```

These commands succeeded without recreating build/. Cache now has `BUILD_TESTING=ON` and
`FETCHCONTENT_UPDATES_DISCONNECTED=ON` (previously tests OFF). A first configure using system
CMake failed with FindHDF5 trying C before it was enabled; rerunning original conda CMake
succeeded. No source CMake changes were made by this task.

Existing RPATH mixes system and conda libraries; working runtime environment after reconfigure:

```bash
export LD_PRELOAD=/data/BLACKMESA_1/sma075/miniconda3/lib/libstdc++.so.6:/data/BLACKMESA_1/sma075/miniconda3/lib/libcrypto.so.3:/data/BLACKMESA_1/sma075/miniconda3/lib/libssh2.so.1
export LD_LIBRARY_PATH=/data/BLACKMESA_1/sma075/observe-libs:/data/BLACKMESA_1/sma075/psrdada-local/lib:/data/BLACKMESA_1/sma075/cuda-spatial-filtering/build/_deps/mathdx-build:/data/BLACKMESA_1/sma075/miniconda3/lib:/data/BLACKMESA_1/sma075/miniconda3/targets/x86_64-linux/lib

# Run from this directory so the benchmark's app.log doesn't modify the user's root app.log.
cd /data/BLACKMESA_1/sma075/cuda-spatial-filtering/build/ingest-handoff-results
timeout 60 ../tests/ProcessorTests > processor-tests.txt 2>&1
timeout 30 ../apps/bench_processor --four-fpga --duration 3 --with-h2d
timeout 30 ../apps/bench_processor --four-fpga --duration 3 --with-h2d --mailboxes
```

`libmathdx-helper.so` exists in build/_deps/mathdx-build (rg without hidden/ignored files missed
it initially). `observe-libs` supplies libgsl.so.25. The preload overrides conflicting system
libstdc++, libcrypto and libssh2 chosen by existing RPATH. Prefer eventual build-runtime cleanup,
but don't confuse these environmental errors with processor failures.

For a later controlled A/B, optional explicit placement (UNTESTED starting point):

```bash
export SPATIAL_BENCH_NODE=0
export SPATIAL_BENCH_PROCESSOR_CPUS=0
export SPATIAL_BENCH_PRODUCER_CPUS=1,2,3,4
export SPATIAL_WORKER_CPUS=5,6,7
# Choose/revise against actual IRQ/NAPI activity. Do not assume this is optimal.
cd /data/BLACKMESA_1/sma075/cuda-spatial-filtering
bash scripts/profiling/run_handoff_comparison.sh 10 5
```

Workers inherit the coordinator mask before applying their own affinity: if pinning the
coordinator to one CPU, also set the worker CPU list, otherwise they share its one CPU.
`--profile-dispatch` adds clocks; use separate diagnostic runs, not throughput A/B results.

## Evidence already collected

- `build/ingest-handoff-results/processor-tests.txt`: **17/17 passed** (2026-09-20 04:18 UTC).
  Includes both threaded handoff variants and the new horizon regression.
- `build/handoff-ab.4NgwlY/`: pre-horizon-fix alternating 10-second comparisons.
  CPU-only legacy roughly 2.14–4.69 Mpps (one large outlier); mailbox roughly 4.28–4.85 Mpps.
  No decisive mailbox gain. Some configuration work overlapped early runs; don't overinterpret.
  CPU-only missing=0; one legacy run had 16,250 discarded packets.
- Same directory H2D runs: roughly 3.137 Mpps processed, but **20–23 million missing and
  21–23 million stuck packets per 10 s run** in both modes. These are correctness failures,
  NOT throughput successes. The sweep was stopped after this was clear. Some last log may
  be partial; only analyze complete result lines.
- Initial profiled 3-second runs before horizon fix:
  legacy 4.079 Mpps, wait=326,915,115 ns, total dispatch=2,967,817,863 ns;
  mailbox 4.029 Mpps, wait=335,922,270 ns, total=2,957,251,626 ns.
  Residual barrier wait about 11% of dispatch time, not evidence of a 2× opportunity from
  signaling alone. These outputs were tool results, not separate saved files.
- First post-horizon-fix H2D short run (tool output only):
  `packets_fed=3500544 packets_processed=3474664 packets_missing=0 packets_discarded=0
  future_queued=122 stuck=122 buffers_completed=105 elapsed=3.003
  packets/sec=1157138.72 GB/sec=3.082618`.
  Compile overlap; needs clean repeat and investigation of nonzero stuck.
- Older saved benchmark `benchmarking/20260708_115002/bench_processor.txt` reported
  5.784 Mpps at 32ch4fpga. It bypasses capture, uses a different thread configuration, and
  does not prove H2D was enabled. Don't present it as current integrated performance.

## Immediate next work, in order

1. Rerun short corrected H2D cases without compilation, with explicit recorded affinity.
   If missing/stuck remain, prioritize correctness before further handoff tuning.
2. Inspect asynchronous buffer lifecycle. `release_buffer()` currently changes window sequence
   fields from a CUDA callback while workers scan `buffers[]`; ownership/synchronization needs
   scrutiny. Placement loops scan all buffers, including `is_ready=false` entries. Determine
   whether late overlap packets can write into buffers undergoing H2D. Do not blindly remove
   barriers or change atomic ordering. The existing unit tests do not cover this lifecycle.
3. Add a controlled asynchronous-recycling test with meaningful data patterns and completion
   verification. The benchmark currently uses constant payloads and counters, so zero missing
   alone cannot prove correct placement or DMA lifetime. Account for startup/shutdown residue
   separately from steady-state missing packets.
4. Investigate corrected H2D sustainable capacity: measure raw pinned H2D bandwidth/latency,
   ready-buffer reuse stalls and coordinator time before assuming CPU copies are the limiter.
   At target, 256-packet windows must complete at 60.547/s (~16.516 ms each).
5. Re-run clean mailbox vs legacy comparisons. Keep default legacy unless a reproducible,
   correctness-preserving advantage appears. If no win, remove/deprioritize the experimental
   path while retaining the useful tests, benchmark controls and horizon fix.
6. Only then implement fixed-size queued jobs across ready producers and per-buffer completion
   tracking. Current design dispatches one producer's range, joins all workers, then moves to
   the next producer; queued jobs could overlap independent work. Requires immutable window
   generations and tracking active writers before buffer publication/recycling.
7. Kernel capture work remains: fixed capacity=256, publish actual filled slots, separate
   reservation from ready publication. Fix SO_RXQ_OVFL ancillary storage: currently attached
   only to reserved-1 but inspected at returned-1, so partial batches miss the counter.
   Do not add adaptive batch sizing. Validate real capture without interfering with active
   observation sockets; check listeners/traffic before binding.

## Design observations worth preserving

- Kernel capture already receives directly into ring slots; normal path has no separate
  staging-to-ring memcpy. `SPATIAL_CAPTURE_STAGE_LOCAL` adds a copy when PRESENT, even `=0`.
- Remaining CPU deposit copies 2,560 sample bytes + 40 scale bytes; sample copy fences every
  packet. Batch fences are a later experiment requiring publication ordering changes.
- Current mailbox experimental path keeps all worker packet-count atomics, so any change
  measured is from request/completion signaling rather than counter aggregation.
- The GPU feeder already uses a padded SPSC queue. Worker handoff and asynchronous buffer
  lifecycle are more urgent than replacing that queue.
- Ring has 6,144 slots, only ~3.1 ms at target; deferred ownership and empty slots reduce slack.
- No full data-integrity proof at the GPU handoff, ten-minute target run, or live capture
  result exists yet. Report this honestly.

No build/test process is intentionally left running at handoff. The long comparison was
terminated; the build and ProcessorTests completed successfully. Check host process state
before starting a new timing run in case a final timed child briefly survived termination.

## 2026-09-21: 48-channel Starweave investigation (current work)

The working 40-channel state is checkpointed at commit
`4413a83 feat: add high-throughput ibverbs ingest through 40 channels`. Preserve that commit as
the rollback point; the 48-channel experiments below are deliberately uncommitted.

The 48-channel target receives channels 172--219 from four FPGAs using
`SPATIAL_IBVERBS_ZERO_COPY=1`, `--capture-backend ibverbs`, and the established capture/worker
affinity. Every test has reached full offered receive rate initially with `NICDrops=0`, then
stopped when all eight host packet-assembly buffers ceased returning. Failed variants include:

- original handoff: stopped at 11,383,508 received / 233 pipeline runs;
- retire a complete CQ batch before posting replacements: 16,862,793 / 343 runs;
- 64-slot processor dispatch for >=48 channels: 15,176,250 / 326 runs;
- committed-prefix dispatch: 18,117,842 / 401 runs;
- all three combined: 18,566,501 / 401 runs;
- CUDA graphs disabled: 20,379,542 / 445 runs;
- one GPU pipeline buffer: 687,969 / 14 runs;
- one GPU buffer plus `CUDA_LAUNCH_BLOCKING=1`: 690,116 / 14 runs, with no CUDA error reported.

Committed-prefix dispatch removed the earlier worker diagnostic about reserved-but-uncommitted
ring slots, but did not prevent the freeze. Disabling CUDA graphs also did not prevent it. A
single GPU stream fails sooner, not later, and synchronous launching reports no invalid-access or
launch error. These results make capture throughput, worker scheduling, CUDA graphs, and a simple
cross-stream dependency cycle unlikely root causes. The live failure follows asynchronous host
packet-buffer recycling.

`LambdaPipelineIngest::ingest_and_scale()` currently records one reusable `ingest_copy_done`
event per GPU pipeline buffer, makes a separate host stream wait for it, then launches the
`release_buffer()` host callback. Multiple release operations can be outstanding while the same
event is recorded again. The next A/B is `SPATIAL_RELEASE_ON_COMPUTE_STREAM=1`, which places the
release callback directly after the H2D copies on the compute stream and removes event reuse and
the cross-stream wait. `SPATIAL_CUDA_SYNC_PIPELINE=1` is also available to synchronize each full
pipeline execution diagnostically. Both toggles are off by default.

If direct-stream release fixes the freeze, replace the diagnostic path with the production
design: one completion event per host packet buffer (eight events, since a buffer cannot be
reused until its own event completes) plus a small reclaimer that queries completed events and
calls `release_buffer()`. That avoids both event reuse ambiguity and a host callback bubble in
the compute stream.

Unrelated but proven useful: `FineChannelizer` previously constructed one identical
`ppf::Filter` per coarse channel, causing 48 identical NVRTC compilations at 48 channels.
Ring/history mode is disabled and the object has no per-launch mutable signal state, so it now
constructs one shared filter object and still launches it over each independent coarse-channel
slice. Startup fell from roughly 6 minutes (48 compilations) to roughly 10 seconds (one). This
does not yet reduce the 48 runtime kernel launches; batching coarse channels through the filter's
receiver dimension is a separate potential runtime optimization requiring output-layout changes.
ProcessorTests pass 18/18 and FineChannelizerTests pass 6/6 after this change.

The rebuilt `build/apps/starweave_4_48` needs `cap_net_raw` restored before the next live run:

```bash
sudo setcap cap_net_raw+ep \
  /data/BLACKMESA_1/sma075/cuda-spatial-filtering/build/apps/starweave_4_48
```

### 2026-09-21 continuation: callback freeze fixed; sustained GPU rate is now the limiter

The direct-compute-stream callback A/B did not fix the freeze: it stopped at 13,812,960
received / 293 pipeline runs. A stage-synchronized run completed every GPU stage for runs 1--14
and then received no run 15, proving those kernels were not themselves hung. Giving every host
assembly buffer its own event while retaining CUDA host callbacks still stopped at exactly 401
runs. All scheduled visibility dumps and writer callbacks completed, ruling out the HDF writer.

The current implementation no longer uses `cudaLaunchHostFunc` for normal Starweave host-buffer
recycling. It records one H2D completion event per host assembly buffer and a dedicated CPU
reclaimer thread polls `cudaEventQuery`, clears ownership, and calls `release_buffer()`. The old
direct-compute-stream callback remains only behind the diagnostic
`SPATIAL_RELEASE_ON_COMPUTE_STREAM` switch. `ProcessorStateBase::input_buffer_count()` was added
so the pipeline can allocate the correct number of events. ProcessorTests pass 18/18 and
FineChannelizerTests pass 6/6.

This fixed the exact callback deadlock: a 20-second 48-channel smoke run crossed the old 401-run
ceiling and shut down at 456 runs. It did **not** make 48 channels lossless. The important counters
were:

```text
Received=18535180, Processed=18533016, Missing=3915430,
FutureQueued=169061, StuckUnprocessed=8226, Pipeline Runs Queued=456
```

Do not use `NICDrops=0` as evidence here; the ibverbs backend does not provide a trustworthy drop
counter. Packet accounting and the arrival bitmap are the acceptance criteria.

`SPATIAL_RECLAIMER_DIAG=1` now prints one-second submitted/reclaimed/pending counts. A live run
gave:

```text
submitted=57  reclaimed=56  pending=1
submitted=114 reclaimed=112 pending=2
submitted=227 reclaimed=224 pending=3
submitted=340 reclaimed=335 pending=5
submitted=396 reclaimed=390 pending=6
submitted=453 reclaimed=446 pending=7
submitted=456 reclaimed=456 pending=0   # input had already collapsed
```

The required block cadence is `15500 / 256 = 60.546875` buffers/s. The GPU pipeline sustains only
about 55--56 buffers/s, roughly 7.5% short. Pending host buffers therefore grow until all eight
are GPU-owned. The CPU assembly horizon then stops advancing, future packets retain packet-ring
slots, QPs run out of replacement receives, and `Missing` grows catastrophically. The GPU later
reclaims every submitted buffer, so it is slow rather than deadlocked.

A 48-channel-only expansion of the packet ring from 6,144 to 24,576 slots was tested and removed.
It still froze (16,205,823 received / 413 runs, Missing=4,145,228,
FutureQueued=639,433). This proves that more ring elasticity only delays the same downstream
backpressure and is not the fix. The working <=40 configuration remains on the original 6,144
slots.

Synchronous per-stage timings (`SPATIAL_CUDA_SYNC_STAGES=1`) over 14 representative buffers were
very stable:

```text
ingest (H2D + scale/convert)       ~17.1--17.5 ms
alignment                          ~5.8 ms
channelize + correlate + trim      ~9.6--9.7 ms
accumulate                         ~0.36 ms
```

These timings serialize the streams and therefore must not be added to predict normal
three-stream throughput, but they identify the large stages. Nsight Systems captured the live
run but failed during report conversion with `No GPU associated to the given UUID`, so there is
no usable `.nsys-rep` yet.

The most promising next optimization is the pre-filter data path, not another receive-ring or
worker-handoff tweak. Today each ~127 MB raw sample block is copied H2D and then makes multiple
full-buffer GPU-memory passes:

1. `scale_and_convert_to_half` (compact complex int8 + scales -> half),
2. `packetToPreAlign` permutation,
3. `apply_delays`,
4. `reorder_streams`,
5. `reorder_to_filter_input` (half -> float and gpu-filter layout).

For fine-channelized pipelines, fuse these transforms so compact H2D input is read once and the
final float `FineChannelizer::FilterInputType` is written directly, applying scale/gain, integer
delay, canonical receiver/polarization mapping, and end padding in one kernel. This removes
several 100+ MB intermediate reads/writes and should comfortably exceed the ~8% gain needed for
48 channels. Preserve the existing path for `NR_FINE_CHANNELS == 1` and initially gate the fused
path for A/B validation. Separately, batching the 48 identical gpu-filter launches by flattening
coarse channel into its receiver dimension could reduce launch overhead, but output layout must
change from `[coarse][fine][block][receiver][pol][time]` to the filter's batched
`[fine][block][coarse*receiver][pol][time]`; do this only after the fused-ingest A/B or if timing
shows more margin is required.

### 2026-09-21 production focus returned to 40 channels

The FPGA backend was returned to channels 176--215. The exact checkpoint-era
`starweave_4_40` binary was re-run for 60 seconds before rebuilding anything:

```text
Received=138851828, Processed=138850560, Missing=50504,
Discarded=13397, FutureQueued=0, StuckUnprocessed=0
Pipeline Runs Queued=3389
```

This is 2.314197 Mpps and **0.036359361%** missing using
`Missing/(Received+Missing)`, including startup. Missing was flat for the final three reports
(15 seconds), discarded packets were startup-only, and shutdown completed normally. Artifacts:

- `build/starweave_smoke_out/ibverbs_40ch_recheck_60s.log`
- `build/starweave_smoke_out/ibverbs_40ch_recheck_60s.hdf5`
- `build/starweave_smoke_out/ibverbs_40ch_recheck_60s.streams.csv`

The exact executable was preserved before rebuilding as
`build/apps/starweave_4_40.checkpoint-4413a83` (same SHA-256 as the validated binary; its
`cap_net_raw` xattr could not be copied without privilege, so restore capability before using it).

The current source was then simplified around the choices intended for normal 40-channel use:

- ibverbs direct-to-packet-ring receive is automatic whenever
  `--capture-backend ibverbs` is selected; `SPATIAL_IBVERBS_ZERO_COPY` was removed from
  `apps/starweave.cu`;
- the one shared gpu-filter object is the normal implementation, avoiding 40 duplicate NVRTC
  compilations while retaining one launch per coarse-channel slice;
- per-host-buffer CUDA events plus the event-polled reclaimer are the only normal Starweave
  release path;
- experimental Starweave runtime switches for GPU-buffer count, mailbox handoff, callback
  release, CUDA stage/full-pipeline synchronization, reclaimer telemetry, and graph disabling
  were removed from production execution;
- noisy ibverbs `ITER`/periodic CQ telemetry is now compiled only with
  `SPATIAL_DIAGNOSTICS`; temporary processor stall heartbeats were removed;
- CPU/NUMA affinity variables remain because they are deployment controls and the safe cores can
  change with host/NIC IRQ placement.

After cleanup, `starweave_4_40`, ProcessorTests, and FineChannelizerTests built successfully;
ProcessorTests passed 18/18 and FineChannelizerTests passed 6/6.

The rebuilt optimized binary then passed a 60-second live run:

```text
Received=138829444, Processed=138828992, Missing=73238,
Discarded=13953, FutureQueued=0, StuckUnprocessed=0
Pipeline Runs Queued=3389
```

This is **0.05273%** missing. Missing advanced in isolated steps but did not grow
continuously; shutdown was clean.

A requested 10-minute run also completed cleanly:

```text
Received=1388325534, Processed=1388325056, Missing=613642,
Discarded=3072, FutureQueued=0, StuckUnprocessed=0
Pipeline Runs Queued=33908
```

That is 2.314 Mpps and **0.044180624%** missing using
`Missing/(Received+Missing)`. The GPU/assembly path remained caught up for the entire run:
`FutureQueued` and `StuckUnprocessed` were always zero and processed stayed within one partial
256-packet block of received. Artifacts are:

- `build/starweave_smoke_out/ibverbs_40ch_optimized_600s.log`
- `build/starweave_smoke_out/ibverbs_40ch_optimized_600s.hdf5` (20 GiB)
- `build/starweave_smoke_out/ibverbs_40ch_optimized_600s.streams.csv`

This is capacity-stable but not yet lossless. `Missing` rose in short steps separated by many
flat reports. During one live correlated interval it rose by 11,474 while all four mlx5
`rx_out_of_buffer` counters remained bit-for-bit unchanged, so receive-WQE starvation was not
the cause of that burst. Do not use a post-shutdown `rx_out_of_buffer` sample: the FPGAs keep
transmitting after the QPs close and those counters then rise rapidly.

The four capture threads were correctly pinned to `1,11,13,15`; topology is 20 distinct
physical cores (not SMT siblings). CPU 1/NIC 94 are on NUMA node 0 and CPUs 11/13/15 plus NICs
134/175/216 are on NUMA node 1. A mid-run sample nevertheless showed substantially more
involuntary context switches on the first two capture threads (~687/~496) than the latter two
(~92/~102). This is a lead, not yet proof of causation.

Next loss-elimination step: add low-overhead, compile-time-only `SPATIAL_DIAGNOSTICS` attribution
of missing bitmap entries by FPGA source (and ideally frequency channel), plus periodic capture
thread/CQ counters. Run a short diagnostic binary and correlate each aggregate Missing step with
the source and scheduler counters. If the first two capture threads dominate, test real-time
scheduling or cleaner dedicated cores; this will require `CAP_SYS_NICE` or launching through
`chrt` with privilege. Do not change affinities based on the disproved SMT hypothesis.

### Paused state while waiting for the FPGA backend

The compile-time diagnostic described above is now implemented:

- `ProcessorState` counts absent completed-buffer bitmap entries by FPGA index, entirely off the
  per-packet hot path;
- the normal five-second stats report is followed by
  `DIAG MissingByFpgaIndex=[...]`;
- the ibverbs two-second diagnostic reports cumulative CQ completions and per-capture-thread
  voluntary/involuntary context switches using `getrusage(RUSAGE_THREAD)`;
- all of it is compiled only when the CMake option `SPATIAL_DIAGNOSTICS=ON`; the option defaults
  to `OFF`.

For the standard interface order used here, FPGA diagnostic indexes map as follows:

```text
index 0 -> enp94s0np0
index 1 -> enp134s0np0
index 2 -> enp175s0np0
index 3 -> enp216s0np0
```

The ready diagnostic executable is:

```text
build/apps/starweave_4_40.diagnostic
SHA-256 c300553c8fdceb579e0718d4371c54c2e99e90d5a8d956ebcfe385c03e6cad50
```

It currently has no file capability (empty `getcap` output). Before the next live test run:

```bash
sudo setcap cap_net_raw+ep \
  /data/BLACKMESA_1/sma075/cuda-spatial-filtering/build/apps/starweave_4_40.diagnostic
```

The backend was not available immediately after this build, so no live diagnostic result exists
yet. When it is free, run the diagnostic for roughly 120 seconds at channels 176--215 using the
same four NICs and affinity (`SPATIAL_CAPTURE_CPUS=1,11,13,15`,
`SPATIAL_WORKER_CPUS=3,4,5`). Save the log/HDF5 as
`build/starweave_smoke_out/ibverbs_40ch_loss_diag_120s.*`. Correlate each change in aggregate
`Missing` with the corresponding delta in `MissingByFpgaIndex` and with each QP's
`involuntary_cs`; also take live (not post-shutdown) `rx_out_of_buffer` snapshots if useful.

Important build state: the in-place CMake cache is temporarily
`SPATIAL_DIAGNOSTICS:BOOL=ON`, and both `build/apps/starweave_4_40` and the `.diagnostic` copy are
the diagnostic executable with the SHA above. After the diagnostic experiment, restore the
normal production build and its capability:

```bash
cd /data/BLACKMESA_1/sma075/cuda-spatial-filtering
cmake -S . -B build -DSPATIAL_DIAGNOSTICS=OFF -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build --target starweave_4_40 -j4
sudo setcap cap_net_raw+ep build/apps/starweave_4_40
```

Do not overwrite `starweave_4_40.checkpoint-4413a83`; it is the preserved pre-experiment
working binary. Reconfiguration also exposed and fixed two latent build problems: the top-level
project now enables C because it requests the HDF5 C component, and hard-coded per-target
`CUDA_ARCHITECTURES=native` overrides were removed so the cache-controlled `89` value applies
reliably on this Ada GPU. `/build` was reconfigured in place and was never deleted/recreated.

### Human-facing launcher and architecture explanation

`scripts/run_starweave.sh` is now the recommended operator entry point. It defaults to the proven
four-FPGA 40-channel configuration and sets the ibverbs backend, all NICs, channel range,
capture/worker affinities, CUDA root, JSON inputs, antenna map, output paths, and duration. It
validates `cap_net_raw`, NICs, inputs, overwrite risk, and available disk space before running.
FPGA tuning is deliberately opt-in with `--configure-fpga`; a normal capture cannot silently
retune a backend someone else is using. `--diagnostic` selects the preserved diagnostic binary,
and `--dry-run` is safe while the backend is unavailable.

The launcher was adapted from the human interface of
`/data/BLACKMESA_0/lambda/scripts/correlate_lambda.sh`. No file in that external directory was
changed. Its usage is documented in `docs/starweave-capture-runbook.md`.

`docs/starweave-24-to-40-explained.md` is the new plain-English technical narrative. It covers
the complete pre-GPU path, direct-to-ring ibverbs receive, raw-header scatter/gather, the
reserve/commit/processed ownership states, four strided producer lanes, fixed 256-slot
coordinator dispatch, the coordinator doing a share of work, the required worker barrier,
why mailboxes were rejected as the production default, one-shot delay-aware startup, the SPSC
GPU feeder, CUDA-event host-buffer reclamation, measured proof through 40 channels, and why
48 channels is a different downstream GPU service-rate problem.

### Copying the work to another computer

The Blackmesa checkout cannot push to GitHub. A source-only snapshot is therefore created at:

```text
build/starweave-40-source-snapshot-20260921.tar.gz
```

It contains the current versions of every source file changed on the server branch relative to
`origin/main`, plus the uncommitted 40-channel cleanup, diagnostic code, launcher, runbook, plans,
and handoff documents. It deliberately excludes `config.json`, `extern/tcc`, build products,
HDF5 captures, logs, and unrelated top-level operational files.

From another computer, copy it and extract it over a clean checkout of this repository:

```bash
scp sma075@blackmesa:/data/BLACKMESA_1/sma075/cuda-spatial-filtering/build/starweave-40-source-snapshot-20260921.tar.gz .
tar -xzf starweave-40-source-snapshot-20260921.tar.gz -C /path/to/cuda-spatial-filtering
cd /path/to/cuda-spatial-filtering
git status --short
git add -A
git commit -m "feat: optimize four-FPGA Starweave through 40 channels"
```

Review `git diff --cached` before committing if the local checkout contains its own work. The
archive is a file overlay, not a Git repository and not a destructive reset; it does not remove
local files. `build/starweave-transfer-20260921-MANIFEST.txt` inside the archive lists its scope.
