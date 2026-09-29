# Handoff: four-FPGA kernel ingest optimization

Updated 2026-09-20, approximately 04:19 UTC. This is the continuation entry point.

## 2026-09-23 pulsar intensity/GPU continuation

The first live detected-intensity/GPU-DSPSR attempt established two concrete
requirements:

- DSPSR needs `-U 1100` for this 1081-channel intensity input.  Its default
  256 MB limit fails before capture with `a minimum of "-U 1008"`; at 1100 MB
  it selected an 11,268-sample/1,104.18 MB block and reported `prepared in
  12.8788 seconds`.
- `apps/pulsar-fold.cu` had a capture initialization-order bug not present in
  Starspin.  It called `arm_ibverbs_zero_copy_captures()` before setting
  `state.nr_capture_threads`.  Initial direct-to-ring WR reservations therefore
  used stride one for all four QPs and overlapped.  Payload written by one NIC
  could consequently be committed with another NIC's source-IP metadata and
  classified as the wrong FPGA.  The evidence was an FPGA-3-like sample count
  near 36,593,431,xxx being compared with the FPGA-2 buffer boundary
  36,602,105,920; capture then froze at 7,872 received / 3,072 processed with
  no pipeline blocks.  This was not an Alveo-delay error and does not require
  an 8-channel delay recalibration.

The producer count is now published before ibverbs arming, matching the proven
ordering in `apps/starspin.cu`.  The corrected
`pulsar_fold_intensity_4_40` has been rebuilt; rebuilding removes
`cap_net_raw`, so it must be restored before the next live test.  The launcher
also now supplies the required intensity-mode DSPSR `-U 1100` setting.  The
next proof run should verify that counters advance well beyond the old ~8k
startup freeze and that the pipeline count grows continuously for 30 seconds.

The properly matched 24-channel control subsequently succeeded after all four
FPGAs were explicitly retuned to channels 176--199 (`-n 24 -f 188`).  Do not
treat a narrow receiver over a wider FPGA transmission as an equivalent test.
The 30-second run `build/pulsar_runs/crab_intensity_gpu_24ch_matched_30s`
completed cleanly with 41,645,141 received, 41,643,968 processed, 26,277
missing (0.063097397%), zero discarded/future/stuck/NIC drops, and 1,694 GPU
pipeline runs.  DSPSR progressed to 29.9 seconds and wrote `crab.ar` with four
subintegrations, 649 channels, one polarization, 256 phase bins, and 27.162 s
usable duration.  Direct `pdv -R -t` inspection confirmed non-zero folded
profile data; the earlier `psrstat all:*` zero result was a misleading query,
not an empty archive.  DSPSR zero-weighted 19 corrupted channel/subintegration
profiles.  A separate `BEAM_DEBUG=1` run showed fully populated, finite complex
beam output after startup (typical extrema about -5 to +5), proving the GPU
beamformer itself is producing real data.

In contrast, matched 40-channel intensity runs repeatedly stopped at 59--60
pipeline blocks while DSPSR remained at 0.8 seconds, independent of whether
the GPU DADA ring held 300 or 800 blocks and also with `-asynch-fold`.  This
rules out simple DADA capacity and points to single-GPU coexistence/resource
behaviour between 32 and 40 channels.  The next controlled step is a matched
32-channel run over channels 176--207.

## 2026-09-21 continuation status

Work has moved to the ibverbs zero-copy path because it produced the substantial throughput
improvement sought here: the 4-FPGA/24-channel target sustained about 1.39 Mpps with essentially
zero loss. The active 32-channel problem and exact live-run procedure are now maintained in
`docs/ibverbs-zerocopy-handoff.md` and `docs/starspin-capture-runbook.md`; use those as the
primary continuation documents.

The latest change removed all runtime buffer-timeline re-seeding. The first valid packet now
initializes the four FPGA boundaries exactly once through the configured delay file. Startup
packets older than that boundary may be discarded, by explicit operator choice, but the timeline
cannot be rewritten after publication. Signed delay arithmetic is covered with the live-scale
FPGA-2 offset. `ProcessorTests` now passes **18/18**, including the new huge-delay one-shot
bootstrap test and both legacy/mailbox handoff cases. The subsequent live 60-second
`starspin_4_32` ibverbs zero-copy run at `--min_freq_channel 180` completed without freezing:
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

## Starspin 4×24 live mailbox run (2026-09-20)

Added a runtime opt-in in `apps/starspin.cu`: set `SPATIAL_WORKER_MAILBOXES=1` to enable
`ProcessorState::use_worker_mailboxes`; unset it (or set `0`/`false`) to retain the default
handoff. Built the existing target successfully:

```bash
/data/BLACKMESA_1/sma075/miniconda3/bin/cmake --build build \
  --target starspin_4_24 -j 4
```

Ran the runbook command with the required affinity and **`--min_freq_channel 184`**:

```bash
SPATIAL_WORKER_MAILBOXES=1 \
SPATIAL_CAPTURE_CPUS="1,11,13,15" SPATIAL_WORKER_CPUS="3,4,5" \
./starspin_4_24 --capture-backend kernel \
  --network-interface enp94s0np0,enp134s0np0,enp175s0np0,enp216s0np0 \
  --min_freq_channel 184 --obs-length 60 \
  --delay-file alveo_delays.json --gains weights.json \
  --eigenvalue-num-filename nr-signal-eigenvalues.json \
  --stream-antenna-map /data/BLACKMESA_1/sma075/cuda-spatial-filtering/stream_antenna_map.json \
  --vis_output_file starspin_mailbox_runs/starspin_mailbox_4_24_60s.hdf5
```

Output is in `build/apps/starspin_mailbox_runs/`:
`starspin_mailbox_4_24_60s.log` and the 1.7 GB HDF5 result. Startup JIT took about four
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
`SPATIAL_IBVERBS_ZERO_COPY=1` with `--capture-backend ibverbs` in `starspin`.
After the `ProcessorState` ring is allocated, the backend registers the
contiguous packet pool with ibverbs, reserves ring slots per capture thread,
and posts receive WRs directly to those slot addresses. Completions commit the
slot metadata and reserve a replacement slot; the former staging-buffer
`memcpy` is not executed. The existing staging path remains the default.

`starspin_4_24` builds successfully with `HAVE_IBVERBS` enabled. The earlier
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
`apps/starspin.cu`, `config.json`, `extern/tcc`, `include/spatial/pipeline/common.hpp`,
`include/spatial/pipeline/lambda_starspin_pipeline.hpp`, and `include/spatial/spatial.hpp`.
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

## 2026-09-21: 48-channel Starspin investigation (current work)

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

The rebuilt `build/apps/starspin_4_48` needs `cap_net_raw` restored before the next live run:

```bash
sudo setcap cap_net_raw+ep \
  /data/BLACKMESA_1/sma075/cuda-spatial-filtering/build/apps/starspin_4_48
```

### 2026-09-21 continuation: callback freeze fixed; sustained GPU rate is now the limiter

The direct-compute-stream callback A/B did not fix the freeze: it stopped at 13,812,960
received / 293 pipeline runs. A stage-synchronized run completed every GPU stage for runs 1--14
and then received no run 15, proving those kernels were not themselves hung. Giving every host
assembly buffer its own event while retaining CUDA host callbacks still stopped at exactly 401
runs. All scheduled visibility dumps and writer callbacks completed, ruling out the HDF writer.

The current implementation no longer uses `cudaLaunchHostFunc` for normal Starspin host-buffer
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
`starspin_4_40` binary was re-run for 60 seconds before rebuilding anything:

```text
Received=138851828, Processed=138850560, Missing=50504,
Discarded=13397, FutureQueued=0, StuckUnprocessed=0
Pipeline Runs Queued=3389
```

This is 2.314197 Mpps and **0.036359361%** missing using
`Missing/(Received+Missing)`, including startup. Missing was flat for the final three reports
(15 seconds), discarded packets were startup-only, and shutdown completed normally. Artifacts:

- `build/starspin_smoke_out/ibverbs_40ch_recheck_60s.log`
- `build/starspin_smoke_out/ibverbs_40ch_recheck_60s.hdf5`
- `build/starspin_smoke_out/ibverbs_40ch_recheck_60s.streams.csv`

The exact executable was preserved before rebuilding as
`build/apps/starspin_4_40.checkpoint-4413a83` (same SHA-256 as the validated binary; its
`cap_net_raw` xattr could not be copied without privilege, so restore capability before using it).

The current source was then simplified around the choices intended for normal 40-channel use:

- ibverbs direct-to-packet-ring receive is automatic whenever
  `--capture-backend ibverbs` is selected; `SPATIAL_IBVERBS_ZERO_COPY` was removed from
  `apps/starspin.cu`;
- the one shared gpu-filter object is the normal implementation, avoiding 40 duplicate NVRTC
  compilations while retaining one launch per coarse-channel slice;
- per-host-buffer CUDA events plus the event-polled reclaimer are the only normal Starspin
  release path;
- experimental Starspin runtime switches for GPU-buffer count, mailbox handoff, callback
  release, CUDA stage/full-pipeline synchronization, reclaimer telemetry, and graph disabling
  were removed from production execution;
- noisy ibverbs `ITER`/periodic CQ telemetry is now compiled only with
  `SPATIAL_DIAGNOSTICS`; temporary processor stall heartbeats were removed;
- CPU/NUMA affinity variables remain because they are deployment controls and the safe cores can
  change with host/NIC IRQ placement.

After cleanup, `starspin_4_40`, ProcessorTests, and FineChannelizerTests built successfully;
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

- `build/starspin_smoke_out/ibverbs_40ch_optimized_600s.log`
- `build/starspin_smoke_out/ibverbs_40ch_optimized_600s.hdf5` (20 GiB)
- `build/starspin_smoke_out/ibverbs_40ch_optimized_600s.streams.csv`

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
build/apps/starspin_4_40.diagnostic
SHA-256 c300553c8fdceb579e0718d4371c54c2e99e90d5a8d956ebcfe385c03e6cad50
```

It currently has no file capability (empty `getcap` output). Before the next live test run:

```bash
sudo setcap cap_net_raw+ep \
  /data/BLACKMESA_1/sma075/cuda-spatial-filtering/build/apps/starspin_4_40.diagnostic
```

The backend was not available immediately after this build, so no live diagnostic result exists
yet. When it is free, run the diagnostic for roughly 120 seconds at channels 176--215 using the
same four NICs and affinity (`SPATIAL_CAPTURE_CPUS=1,11,13,15`,
`SPATIAL_WORKER_CPUS=3,4,5`). Save the log/HDF5 as
`build/starspin_smoke_out/ibverbs_40ch_loss_diag_120s.*`. Correlate each change in aggregate
`Missing` with the corresponding delta in `MissingByFpgaIndex` and with each QP's
`involuntary_cs`; also take live (not post-shutdown) `rx_out_of_buffer` snapshots if useful.

Important build state: the in-place CMake cache is temporarily
`SPATIAL_DIAGNOSTICS:BOOL=ON`, and both `build/apps/starspin_4_40` and the `.diagnostic` copy are
the diagnostic executable with the SHA above. After the diagnostic experiment, restore the
normal production build and its capability:

```bash
cd /data/BLACKMESA_1/sma075/cuda-spatial-filtering
cmake -S . -B build -DSPATIAL_DIAGNOSTICS=OFF -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build --target starspin_4_40 -j4
sudo setcap cap_net_raw+ep build/apps/starspin_4_40
```

Do not overwrite `starspin_4_40.checkpoint-4413a83`; it is the preserved pre-experiment
working binary. Reconfiguration also exposed and fixed two latent build problems: the top-level
project now enables C because it requests the HDF5 C component, and hard-coded per-target
`CUDA_ARCHITECTURES=native` overrides were removed so the cache-controlled `89` value applies
reliably on this Ada GPU. `/build` was reconfigured in place and was never deleted/recreated.

### Human-facing launcher and architecture explanation

`scripts/run_starspin.sh` is now the recommended operator entry point. It defaults to the proven
four-FPGA 40-channel configuration and sets the ibverbs backend, all NICs, channel range,
capture/worker affinities, CUDA root, JSON inputs, antenna map, output paths, and duration. It
validates `cap_net_raw`, NICs, inputs, overwrite risk, and available disk space before running.
FPGA tuning is deliberately opt-in with `--configure-fpga`; a normal capture cannot silently
retune a backend someone else is using. `--diagnostic` selects the preserved diagnostic binary,
and `--dry-run` is safe while the backend is unavailable.

The launcher was adapted from the human interface of
`/data/BLACKMESA_0/lambda/scripts/correlate_lambda.sh`. No file in that external directory was
changed. Its usage is documented in `docs/starspin-capture-runbook.md`.

`docs/starspin-24-to-40-explained.md` is the new plain-English technical narrative. It covers
the complete pre-GPU path, direct-to-ring ibverbs receive, raw-header scatter/gather, the
reserve/commit/processed ownership states, four strided producer lanes, fixed 256-slot
coordinator dispatch, the coordinator doing a share of work, the required worker barrier,
why mailboxes were rejected as the production default, one-shot delay-aware startup, the SPSC
GPU feeder, CUDA-event host-buffer reclamation, measured proof through 40 channels, and why
48 channels is a different downstream GPU service-rate problem.

### Copying the work to another computer

The Blackmesa checkout cannot push to GitHub. A source-only snapshot is therefore created at:

```text
build/starspin-40-source-snapshot-20260921.tar.gz
```

It contains the current versions of every source file changed on the server branch relative to
`origin/main`, plus the uncommitted 40-channel cleanup, diagnostic code, launcher, runbook, plans,
and handoff documents. It deliberately excludes `config.json`, `extern/tcc`, build products,
HDF5 captures, logs, and unrelated top-level operational files.

From another computer, copy it and extract it over a clean checkout of this repository:

```bash
scp sma075@blackmesa:/data/BLACKMESA_1/sma075/cuda-spatial-filtering/build/starspin-40-source-snapshot-20260921.tar.gz .
tar -xzf starspin-40-source-snapshot-20260921.tar.gz -C /path/to/cuda-spatial-filtering
cd /path/to/cuda-spatial-filtering
git status --short
git add -A
git commit -m "feat: optimize four-FPGA Starspin through 40 channels"
```

Review `git diff --cached` before committing if the local checkout contains its own work. The
archive is a file overlay, not a Git repository and not a destructive reset; it does not remove
local files. `build/starspin-transfer-20260921-MANIFEST.txt` inside the archive lists its scope.

### 20-minute 40-channel diagnostic observation

After the backend again became available at channels 176--215, the diagnostics-enabled
`build/apps/starspin_4_40` completed a requested 1,200-second capture using the new launcher.
Final counters were:

```text
Received=2777734646, Processed=2777733056, Missing=168395,
Discarded=0, FutureQueued=0, StuckUnprocessed=0, NICDrops=0
Pipeline Runs Queued=67819
DIAG MissingByFpgaIndex=[41928,41960,42000,42507]
```

This sustained **2.314779 Mpps**. The missing fraction was **0.006061947%**, calculated as
`Missing/(Received+Missing)`. The Received/Processed shutdown tail was 1,590 packets. All 240
five-second reports had zero discarded, future-queued, and stuck-unprocessed packets; shutdown
was normal. Artifacts are:

- `build/starspin_smoke_out/ibverbs_40ch_loss_diag_1200s.log` (762 KiB)
- `build/starspin_smoke_out/ibverbs_40ch_loss_diag_1200s.hdf5` (36 GiB)
- `build/starspin_smoke_out/ibverbs_40ch_loss_diag_1200s.streams.csv`

The diagnostic materially changes the loss diagnosis. Most Missing changes were common-mode:
each event added nearly the same count to all four FPGA indexes. Final per-source totals differ by
only 579 packets from lowest to highest (about 1.4% of an individual source total). In contrast,
the final per-capture-thread involuntary context-switch totals differed dramatically:

```text
enp94s0np0  / FPGA index 0: 1856 involuntary switches, Missing=41928
enp134s0np0 / FPGA index 1: 1492 involuntary switches, Missing=41960
enp175s0np0 / FPGA index 2:  323 involuntary switches, Missing=42000
enp216s0np0 / FPGA index 3:  318 involuntary switches, Missing=42507
```

Therefore capture-thread scheduling is not a plausible primary cause of these loss bursts: the
first two threads were descheduled roughly five times as often but did not lose more packets.
Likewise, the application never accumulated backlog, so neither CPU assembly nor GPU service
capacity caused the Missing events.

There were 25 post-startup Missing changes across 239 subsequent reports. They were short steps
separated by flat intervals, including an approximately 240-second clean stretch. The strongest
remaining hypothesis is a common channel/time gap upstream of the four independent receive paths,
or a shared completion-window/accounting condition. The next diagnostic should count missing
bitmap entries by coarse frequency channel as well as FPGA. If each burst concentrates in the
same channel(s) across all four FPGAs, inspect backend emission/channel scheduling and the
per-channel completion threshold. Do not pursue real-time capture scheduling first; this run is
direct evidence against it.

### Active eight-hour 40-channel capture (2026-09-21/22)

An eight-hour capture using the same proven diagnostic 40-channel configuration started at
**2026-09-21 21:57:28 AEST** and is expected to finish at approximately
**2026-09-22 05:57:28 AEST**. It is running under a persistent execution session with the
following process chain at launch: wrapper PID 1050115, `timeout` PID 1050116, launcher PID
1050117, and Starspin PID 1050129. The effective Starspin command uses ibverbs on
`enp94s0np0,enp134s0np0,enp175s0np0,enp216s0np0`, `--min_freq_channel 176`, and
`--obs-length 28800`; capture CPU affinity is `1,11,13,15` and worker affinity is `3,4,5` via
the launcher defaults.

Artifacts are being written to:

- `build/starspin_smoke_out/ibverbs_40ch_8h_20260921_run.hdf5`
- `build/starspin_smoke_out/ibverbs_40ch_8h_20260921_run.log`
- `build/starspin_smoke_out/ibverbs_40ch_8h_20260921_run.launcher.log`
- `build/starspin_smoke_out/ibverbs_40ch_8h_20260921_run.streams.csv`

At the early health check, the latest complete report was:

```text
Received=266153133, Processed=266152704, Missing=68347,
Discarded=0, FutureQueued=0, StuckUnprocessed=0, NICDrops=0
DIAG MissingByFpgaIndex=[17040,17056,17096,17155]
```

This is **0.025672985% Missing** using `Missing/(Received+Missing)`. Missing was flat over the
latest reports, all four source totals were nearly equal, and there was no application backlog or
error. Treat this only as an early snapshot: recalculate the percentage from final counters after
normal shutdown. Monitor with:

```bash
tail -f build/starspin_smoke_out/ibverbs_40ch_8h_20260921_run.log
```

At the 34-minute-43-second health check the process was still live and healthy. Counters were
`Received=4756851617`, `Processed=4756850944`, `Missing=281931`, with zero `Discarded`,
`FutureQueued`, and `StuckUnprocessed`. This is **0.005926489% Missing**; per-FPGA totals were
`[70448,70448,70491,70544]`. The HDF5 output had reached 55 GiB and the filesystem still had
4.4 TiB available.

After this eight-hour capture had started, the operator confirmed that physical antennas 8 and
12 require X/Y polarization reversal. `stream_antenna_map.json` now swaps only those two antenna
pairs, and `scripts/run_starspin.sh` already uses that file by default. This change therefore
applies to future captures only: the active eight-hour run and the preceding 20-minute run loaded
the old unswapped map at process startup. Do not replace this map wholesale with
`/data/BLACKMESA_0/lambda/cuda-spatial-filtering/bin/stream_antenna_map.json`: that untracked file
also changes the complete FPGA-3 antenna-to-stream ordering and swaps antennas 8 and 22, not 8
and 12.

At the 2026-09-22 00:57 AEST checkpoint (about three hours elapsed), the active eight-hour run
remained healthy. The latest counters were `Received=25033206418`, `Processed=25033205952`,
`Missing=2405532`, with zero `Discarded`, `FutureQueued`, and `StuckUnprocessed`. Cumulative loss
was **0.009608441%**. Over the preceding approximately 15 minutes, Received increased by about
2.083 billion and Missing by 150,891. The most recent per-FPGA split was still essentially even
at `[601338,601352,601392,601450]`. The HDF5 file was 299 GiB and 4.1 TiB remained free.

At 02:00 AEST (about four hours elapsed), counters were `Received=33655153456`,
`Processed=33655152832`, and `Missing=3404948`, for **0.010116143%** cumulative Missing. The
latest approximately 15-minute window received 2,083,319,902 packets and added 78,180 Missing,
or **0.003752523%** for that window. `Discarded`, `FutureQueued`, and `StuckUnprocessed` remained
zero; the per-FPGA split `[851200,851208,851248,851292]` remained even. Output was 403 GiB with
4.0 TiB free.

The eight-hour run completed normally at about 05:57 AEST with process exit status 0. Final
counters were:

```text
Received=66662187344, Processed=66662186944, Missing=6805136,
Discarded=0, FutureQueued=0, StuckUnprocessed=0, NICDrops=0
Pipeline Runs Queued=1627660
DIAG MissingByFpgaIndex=[1701200,1701240,1701280,1701416]
```

Final Missing was **0.010207348%** using `Missing/(Received+Missing)`, and average receive rate
was **2.314659 Mpps**. The Received/Processed shutdown tail was only 400 packets. The log shows
the duration limit being reached, workers joining, feeder/receiver/processor shutdown, visibility
dump, writer stop, and `Shutdown complete.` There were no matched CUDA, CQ, timeout, failure, or
pipeline-backlog errors. The final HDF5 is 855,215,433,616 bytes (about 797 GiB) and was reopened
successfully with h5py; its `visibilities` dataset has shape
`(29066, 1120, 820, 2, 2, 2)`. The capture file is:

```text
/data/BLACKMESA_1/sma075/cuda-spatial-filtering/build/starspin_smoke_out/ibverbs_40ch_8h_20260921_run.hdf5
```

### Crab pulsar feasibility review (2026-09-23)

The commissioning material reviewed for this assessment is the checkout at
`/data/BLACKMESA_1/sma075/lambda-commissioning`, not the older BLACKMESA_0 copy.
Its current 8-hour processing checkpoint provides materially better evidence for usable coherent
calibration than the historical pulsar instructions alone:

- seven consecutive 20-minute calibration candidates were accepted against independent trailing
  five-minute holdouts;
- the earlier 40-channel 20-minute capture had adjacent-100-second closure scatter of 2.76 degrees
  XX and 3.12 degrees YY, and calibrated long-baseline coherence of 0.683/0.707;
- the production calibration deliberately excludes A14 and A36, uses A34 as the XX reference
  fallback because A1 XX is poor, and applies the capture-local A8/A12 receptor correction;
- candidate A5/A29 and A11/A13 physical-identity swaps remain unconfirmed. For a first Crab beam,
  excluding those four antennas as well as A14/A36 is safer than baking an unverified geometry
  hypothesis into geometric steering. The sensitivity reduction is acceptable for a first folded
  detection attempt.

This supports a **folded Crab detection attempt**, but it is not yet a turnkey 4-FPGA/40-channel
pulsar system. The remaining engineering gates are:

1. Add a dedicated 4-FPGA, 40-coarse-channel `pulsar_fold` build target. The current standalone
   target inherits the cache defaults (presently 1 FPGA and 8 coarse channels), unlike the observe
   and Starspin target grids.
2. Wire the pre-correlation fine channelizer into `apps/pulsar-fold.cu`. Although the shared
   pipeline supports it, this app's `LambdaConfig` does not pass `NR_OBSERVING_FINE_CHANNELS` or
   edge trim. Forty coarse channels with 32-way channelization and two bins trimmed from each edge
   should produce 1120 retained channels. Fine resolution is important for the Crab's dispersion.
3. Fix fine-channel steering frequencies before enabling that configuration.
   `compute_steering_weights()` currently loops over widened `T::NR_CHANNELS` but evaluates
   `channel_to_frequency_hz(min_freq_channel + chan)`, incorrectly treating every fine bin as a
   new 781.25-kHz coarse channel.
4. Provide and validate an explicit conversion from the accepted CASA K/B/G solution (or a fresh
   short same-configuration calibrator/Crab visibility solve) into the beamformer's per-frequency,
   per-hand, per-antenna complex weights. `transfer_night_calibration.py` transfers exact-frequency
   legacy HDF5 gains between captures; it does not export the current CASA tables to beam weights.
5. Correct and validate PSRDADA/DSPSR metadata for 1120 retained channels and the actual output
   polarization layout. A prior trial reached DSPSR but failed with `ipol=1 >= npol=1`.
6. For a detection-only fold, wall-clock start time can be checked and tolerated. Meaningful
   absolute pulse times/TOAs require deriving `UTC_START` and offsets from the FPGA sample timeline,
   not overwriting them with process wall clock.

Astropy 8.0.1 with the capture's pinned `finals2000A.all`, the documented Narrabri position, and
built-in lunar ephemeris gives the following geometry for 2026-09-23 AEST. The Crab is above 20
degrees from approximately 02:19 to 08:44 and culminates at approximately 05:31 at 37.65 degrees.
At culmination the Moon is about 131.5 degrees from the Crab and 21 degrees below the horizon; the
Sun is about 95.3 degrees away and 5.1 degrees below the horizon. The best low-Sun interval is
roughly 02:20--05:00 AEST. There is no lunar-occultation concern for this attempt.

The `lambda-commissioning/docs/OBSERVING_PULSARS.md` procedure is useful as a component inventory,
but its examples are stale for this run: they reference BLACKMESA_0, Vela, 8 channels and the old
Sun-to-`weights.json` flow. Do not execute it verbatim for the 4-FPGA/40-channel Crab attempt.

### Crab pulsar implementation checkpoint (2026-09-23)

Work completed in the CUDA repository after the feasibility review:

- Added the dedicated `pulsar_fold_4_40` target. It is compiled for four FPGA sources, 40 coarse
  channels and 64 padded receivers, inherits the configured 32-way fine PFB / two-bin edge trim,
  and embeds the runtime-library paths needed when `cap_net_raw` puts the executable into secure
  loader mode.
- `apps/pulsar-fold.cu` now passes `NR_OBSERVING_FINE_CHANNELS` and
  `NR_OBSERVING_FINE_CHANNEL_EDGE_TRIM` into `LambdaConfig`; the previous executable silently used
  one fine channel even when the shared build configuration requested 32.
- Removed the unused FFT/eigen HDF5 writers from the non-RFI pulsar app. The app now retains only
  the non-null output sentinel required to enable its PSRDADA write path.
- Added `--dada-header` and `--beam-excluded-antennas`. Exclusions operate on physical antenna IDs
  after the canonical stream reorder and make every channel/polarization beam weight exactly zero.
  The planned launcher must pass at least `--beam-excluded-antennas 14`; the accepted CASA
  calibration also excluded A36, so either produce a new valid A36 solution or explicitly exclude
  both `14,36`.
- Fixed geometric steering on the widened fine-channel axis. For 32 PFB bins with two trimmed at
  each edge, retained-bin offsets are `(k-13.5)/27` coarse-channel widths. The old code incorrectly
  advanced by one full 781.25-kHz coarse channel for every fine bin.
- Added strict per-fine-channel beam calibration. `LambdaConfig::FineAntennaGains` is indexed by
  all 1120 retained processing channels; `compute_steering_weights` multiplies each fine bin's
  geometric phasor by that exact bin's inverse complex antenna gain. It no longer maps all 28 bins
  back to one coarse gain. The gains JSON must state
  `"channel_axis": "fine_processing_index"` and must contain every connected antenna and both
  hands for keys `176..1295` (key = minimum coarse channel + flattened processing-channel index).
  Missing, zero, or non-finite solutions are fatal rather than silently replaced with unity.
- The PSRDADA writer now rejects a ring whose data-block size differs from its compile-time beam
  block, preventing the former silent partial copy into the old 2-MiB/8-channel ring.
- Tests no longer hardcode CUDA architecture `native`; they honor `CMAKE_CUDA_ARCHITECTURES`.
  This permits a reproducible SM89 build even when driver discovery is unavailable in a sandbox.
  The PipelineTests discovery timeout was raised to 60 seconds for CUDA initialization.

Build/test evidence, run outside the sandbox on the RTX 4000 Ada (compute capability 8.9):

```text
cmake -S . -B build -DBUILD_TESTING=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build --target PointingTests PipelineTests pulsar_fold_4_40 -j 6
PointingTests: 8/8 passed
Targeted CUDA steering tests: 3/3 passed
  ComputeSteeringWeightsMatchesGeometricPhaseFormula
  BeamCalibrationIsAppliedIndependentlyPerFineChannel
  ComputeSteeringWeightsZeroesNullAntennas
pulsar_fold_4_40: built successfully
```

Important newly identified frequency-axis issue: the beamformer currently retains 28 bins per
coarse channel, or 1120 array entries, but adjacent coarse channels share their boundary-bin
frequency. There are therefore only `40*27+1 = 1081` unique, uniformly spaced sky frequencies.
The accepted CASA `bandpass.B` table correctly has 1081 channels. A converter must assign the same
CASA solution to both internal copies of each of the 39 shared boundaries. Before claiming the
DSPSR path is frequency-correct without `OS_FACTOR`, decide and implement one of:

1. compact the DADA beam output to the 1081 unique-frequency bins (preferred; exact uniform axis),
   discarding one duplicate at every internal coarse boundary; or
2. prove that DSPSR can describe the non-uniform 1120-entry axis with repeated boundary centres.

Do not put `OS_FACTOR` into the fine-channelized DADA header: the operator explicitly said it will
be removed after edge clipping. The launcher/header/ring work was not yet complete at this
checkpoint. The raw 1120-channel beam block is 9,175,040 bytes
(`1 beam * 512 times * 1120 channels * 2 pol * 2 complex * 4-byte float`). A compact 1081-channel
block would be 8,855,552 bytes. The intended sample interval is 34.56 microseconds and DSPSR
subintegrations are 10 seconds.

Timing-model warning: the `PSRCAT_FILE` currently exported in the shell points to
`/data/BLACKMESA_0/lambda/psrdada-local/psrcat/psrcat.db`. Its J0534+2200 spin model has
`PEPOCH 48442.5` (1991). It is fine for source identity/position/DM but unsafe for a 2026 Crab fold
because the Crab has glitched repeatedly since then. DSPSR source-name lookup will resolve this
same stale entry. Use a current monthly Crab timing ephemeris for folding unless the catalogue is
updated and its epoch verified first.

### Crab implementation continuation and first smoke (2026-09-23)

The frequency-axis, calibration-export and operational work described as open above is now
implemented:

- The GPU keeps all 1120 retained bins through fine calibration and coherent beamforming, then
  `compact_shared_fine_boundaries_kernel` removes the second copy of each of the 39 shared coarse
  boundaries before PSRDADA. The DADA stream therefore has 1081 unique ascending channels and an
  exact block size of 8,855,552 bytes. `FineChannelizedPulsarFoldTest.`
  `DadaCompactionDropsOnlyRepeatedCoarseBoundary` passes on the RTX 4000 Ada, as does the physical
  pipeline invariant test. The generated header contains no `OS_FACTOR`.
- `scripts/generate_fine_dada_header.py` generates a 4096-byte header with NCHAN=1081,
  FREQ=152.734375 MHz, BW=31.278935185185 MHz, TSAMP=34.56 us and
  RESOLUTION=8,855,552. Its 137.109375--168.359375 MHz channel centres exactly match the accepted
  CASA bandpass table.
- `scripts/export_casa_fine_beam_gains.py` exports the accepted CASA
  `delay.K * bandpass.B * phase.G * gain.G` product into all 1120 internal processing bins.
  The K-Jones convention was not guessed: comparison with the accepted MS's raw DATA and CASA
  CORRECTED_DATA reproduced the complex baseline correction to about 5e-9 rad using
  `exp(+2*pi*i*K_seconds*(frequency-delay_solution_centroid))`. Flagged B samples are filled by
  amplitude/unwrapped-phase interpolation, matching CASA's non-`nearestflag` behaviour.
- The selected window-006 solution has no A14 or A36 solution and no A1-XX B/phase/G solution.
  The production export is `build/apps/crab_fine_gains_window006.json`: whole antennas A14/A36
  are excluded and metadata zero-weights only receptor A1-XX, retaining A1-YY. The strict
  validator reports 1120 channels, 34 included antennas and 67 calibrated receptors. Shared
  internal boundary copies have byte-for-byte identical JSON values.
- `scripts/prepare_pulsar_stream_map.py` verifies the mandatory A8/A12 X/Y reversals. Normal runs
  make no identity changes. `--identity-swaps` in the launcher creates a run-local map with the
  experimental A5<->A29 and A11<->A13 swaps; it never modifies the authoritative repository map.
- `config/crab_jodrell_2026-07-24.par` contains the latest row currently returned by NASA HEASARC
  CRABTIME (F0=29.5357554730, F1=-3.6584566e-10, DM=56.71090, PEPOCH=61245). This is much safer
  than the 1991 installed psrcat model, but is still extrapolated to September and must not be
  presented as a phase-connected absolute-timing solution. The operator only needs a fold.
- `scripts/run_crab_pulsar.sh` is the human-facing launcher. It validates capability, calibration,
  four NICs and all inputs; creates only a new run directory; creates a 300-block GPU DADA ring;
  starts DSPSR with one multi-subintegration archive, 10-second subintegrations and 256 phase
  bins; runs the ibverbs beamformer; waits for EOD/final archive; and destroys only the ring it
  created. Defaults are four FPGAs, coarse channels 176--215, A14/A36 excluded, normal antenna
  identities and the A8/A12 receptor map.

At 2026-09-23 02:56 AEST, the Crab was 25.8 degrees elevation and rising to 37.7 degrees; Moon
separation was about 85 degrees. All four FPGA control registers reported channels 176--215.

The first 60-second smoke directory is
`build/pulsar_runs/crab_smoke_20260923`. It is a **failed machinery test, not an observation**:
capture reached exactly `Received=6144` (the packet-ring size), while `Processed=0`, then the
monitor timed out. No `.ar` was produced. The DADA/header/calibration/steering setup had already
passed, including the exact ring block-size check. Root cause was an app-integration omission:
`pulsar-fold.cu` constructed libibverbs captures but did not call
`arm_ibverbs_zero_copy_captures(capture, state)`. Thus the captures used the legacy shared
sequential writer while `ProcessorState` consumed four per-thread strided claim streams. This is
why the ring filled without one processable dispatch; it was not GPU overload, source geometry,
or packet loss.

The missing arming call has now been added immediately after capture construction, matching the
proven Starspin path and removing the ibverbs staging memcpy as intended. The corrected
`pulsar_fold_4_40` rebuilt successfully. Relinking cleared `cap_net_raw`; an attempted
non-interactive restore failed because sudo needs a password. Before the next smoke, run:

```bash
sudo setcap cap_net_raw+ep \
  /data/BLACKMESA_1/sma075/cuda-spatial-filtering/build/apps/pulsar_fold_4_40
```

Then verify and run a fresh name (never reuse the failed directory):

```bash
/usr/sbin/getcap build/apps/pulsar_fold_4_40
./scripts/run_crab_pulsar.sh --duration 60 --name crab_smoke_zc_retry
```

Do not start the 15-minute fold until this retry both advances Processed/Pipeline Runs and emits a
DSPSR `.ar`. If the normal-map fold works, inspect it first; the identity-swap variant is an A/B
experiment, not the default.

### DSPSR startup fix and final pre-ibverbs diagnostic (2026-09-23)

Further machinery testing found that DSPSR needs about 18 seconds to prepare this 1081-channel
coherent-dedispersion/folding configuration after it receives the DADA header. Previously the
pipeline committed that header only with its first live data block. The former 64-block ring held
only 1.13246208 seconds at 34.56 microseconds/sample, so a full-rate producer could fill it and stall
while DSPSR was still preparing.

`LambdaPulsarFoldPipeline` now commits the observation header synchronously in its constructor,
before CUDA/PFB warm-up. In the outside-sandbox kernel diagnostic
`build/pulsar_runs/crab_machinery_early_header_15s_retry`, the log proves the intended ordering:

```text
[PulsarFoldPipeline] PSRDADA header published before GPU warmup
dspsr: prepared in 17.5843 seconds
Initializing packet capture...
```

DSPSR connected, recognized `INSTRUMENT dspsr`, loaded the current ephemeris and consumed beam
blocks. The kernel packet backend then lost most of the offered traffic
(`Received=1,360,892`, `Missing=1,158,727`, `NICDrops=533,667`) and queued only 61 pipeline runs,
providing roughly one second of usable beam data. DSPSR therefore discarded the fragments shorter
than the requested 10-second subintegration and wrote no archive. This is expected from the
diagnostic kernel backend and is not evidence against the DADA/DSPSR startup fix.

The final production test must use ibverbs outside the sandbox. The last rebuild removed the file
capability again (`/usr/sbin/getcap` currently prints nothing), so the operator must first run the
`setcap` command above. Then run `crab_smoke_zc_retry`; no further rebuild should occur between
`setcap` and that smoke.

Two subsequent smoke launch attempts (`crab_smoke_zc_retry` and `crab_smoke_zc_retry2`) opened no
NICs because the GPU DADA ring vanished before DSPSR attached. This exposed a launcher bug:
GPU-backed `dada_db` mandates persistence mode (`-w`), but a background process can inherit an
EOF stdin and immediately destroy the ring. The launcher now gives `dada_db -w` a private FIFO
whose read/write descriptor remains open for the run, and explicitly checks that both the IPC key
and creator process remain alive before starting DSPSR. This is a script-only change and does not
relink or clear `cap_net_raw`.

### 300-block voltage test and explicit DSPSR-ready handshake (2026-09-23)

The first working ibverbs-to-archive smoke, `crab_smoke_zc_retry3`, was not a sustained
observation. It received 2,417,016 packets with zero reported NIC drops, completed 58 pipeline
blocks, then all four capture threads blocked reserving replacement packet-ring slots. Its archive
contained 1.00348416 seconds, exactly consistent with those blocks. The FPGA traffic had not
stopped; this was downstream backpressure.

The operator requested a 300-block experiment. A 300-block voltage DADA ring is about 2.66 GB and
holds 5.308416 seconds. `crab_voltage_300buf_15s` nevertheless stopped at 58 pipeline blocks, far
short of the DADA depth, proving that merely enlarging the DADA ring does not fix the startup
stall. Its DSPSR process reported `prepared in 37.1171 seconds`, whereas the application finished
warm-up and opened capture about 19 seconds after publishing the header. DSPSR was still building
its coherent-dedispersion plan on the same GPU when capture started.

Two fixes are now compiled but await a post-`setcap` live test:

- the launcher monitors the DSPSR log and creates a per-run ready sentinel only after `prepared
  in`; `pulsar-fold` waits up to 120 seconds for that sentinel before it constructs capture;
- pulsar-fold now uses the same per-host-buffer CUDA-event reclaimer as the proven Starspin path,
  removing its remaining CUDA host-callback release mechanism.

The launcher default is now 300 voltage blocks. A separate
`pulsar_fold_intensity_4_40` target is also built without replacing the voltage binary. It applies
the same fine-channel calibration and geometric steering to complex X/Y, then emits
`I=|X|^2+|Y|^2`. This reduces a DADA block from 8,855,552 to 2,213,888 bytes. The new CUDA test
`DetectedIntensitySumsBothPolarizationsAndCompactsBoundaries` and the existing voltage compaction
test both pass. Intensity mode is a fallback if steady-state voltage DSPSR is still too slow after
the readiness fix; at these frequencies/channel widths it incurs roughly 4 ms of residual
intra-channel dispersion smearing.

Both pulsar binaries were relinked, so both currently need `cap_net_raw` restored before live
testing.

### Multi-FPGA packet-scale indexing regression (2026-09-23)

The calibration audit found a definite indexing defect in
`scale_and_convert_to_half_kernel`: samples and gains include the `fpga_idx` receiver offset, but
the packet-scale lookup does not. FPGA 0 therefore uses its own scales, while every other FPGA
uses FPGA 0's scale for the corresponding receiver-within-packet and polarization. The omission
was introduced by the flattened multi-FPGA kernel rewrite in commit `795deb07` (2026-05-05).

A regression test was added as
`StreamReorderTest.PacketScalesUseFpgaReceiverOffset`. It drives the existing two-FPGA pipeline
fixture with constant complex samples and distinct scale sentinels:

- polarization 0 scales: 1, 3, 11, 13; correct coherent sum 28;
- polarization 1 scales: 2, 4, 12, 14; correct coherent sum 32.

Before changing the production kernel, the focused test was built and run outside the sandbox:

```bash
cmake --build build --target StreamReorderTests -j 8
build/tests/StreamReorderTests \
  --gtest_filter=StreamReorderTest.PacketScalesUseFpgaReceiverOffset \
  --gtest_color=no
```

It failed with exit status 1 exactly as predicted: every time sample was 8 instead of 28 for
polarization 0 and 12 instead of 32 for polarization 1. Those wrong sums are FPGA 0's pair of
scales used twice. This is direct end-to-end proof of the bug.

The kernel was then fixed by adding
`fpga_idx * NR_RECEIVERS_PER_PACKET * NR_POLARIZATIONS` to `scale_ptr`. The identical focused test
then passed, and the complete `StreamReorderTests` binary passed all 6 tests, including cross-FPGA
permutation, polarization reversal, per-antenna gains and fine delays. Any production pulsar or
Starspin executable must be rebuilt before it contains this header-only CUDA fix; rebuilding also
removes `cap_net_raw` from the output executable and therefore requires the operator to restore it.

The earlier suite missed the defect because all end-to-end pipeline, fine-channel, pulsar and
stream-reorder tests supplied unity packet scales. GPUDirect scatter tests verify that distinct
scale bytes land in the correct buffer locations, but do not exercise their consumption by this
kernel.

### Fresh 240 MHz calibration capture after the scale fix (2026-09-23)

The corrected `starspin_4_24`, `pulsar_fold_4_24`, and
`pulsar_fold_intensity_4_24` targets were rebuilt, and the operator restored
`cap_net_raw=ep` on all three.  The backend was then explicitly retuned to 24
coarse channels 296--319 (centre selector 308), and a fresh raw-visibility
capture was taken with:

```bash
scripts/run_starspin.sh --channels 24 --min-channel 296 --duration 300 \
  --configure-fpga \
  --output-dir build/calibration_runs/crab_240mhz_20260923 \
  --name crab_240mhz_cal_5min
```

The capture shut down normally and wrote
`build/calibration_runs/crab_240mhz_20260923/crab_240mhz_cal_5min.hdf5`
(6.6 GB). Final counters were `Received=416625531`, `Processed=416625088`,
`Missing=59928`, `Discarded=0`, `FutureQueued=0`, `StuckUnprocessed=0`, and
`NICDrops=0`. Thus the reported missing fraction including startup is
`59928 / (416625531 + 59928) = 0.01439%`; the missing counter was nearly flat
through steady state. Per-FPGA missing counts were `[14872,14920,15056,15080]`.

Calibration processing is in progress. The accepted path is raw MS plus the
apparent Haslam/AEP model, followed by CASA K/B/G and then
`scripts/export_casa_fine_beam_gains.py`; do not reuse the old 176-band JSON.
The Crab launcher now accepts `--min-channel`, and gain validation checks the
recorded processing/unique channel counts and exact frequency endpoints. A
regression command proved the old 176--215 / 1120-bin gain file is rejected for
the requested 296--319 / 672-bin observation before any live process starts.

#### Calibration outcome: no high-band weights approved

The 10-minute Crab observation was deliberately **not started** because every
fresh calibration candidate failed scientific QA. Preserve the products below
for diagnosis, but do not pass any JSON from this directory to a live run
without a new acceptance result.

The commissioning legacy solver first failed before writing output because its
dense solve contained NaNs. This is caused by unsupported receptors propagating
through its antenna-power/solve path, not by a truncated HDF5 file. To let the
independent raw-MS branch proceed without claiming a legacy calibration,
`scripts/create_raw_ms_export_companion.py` now creates a clearly labelled
unity-only compatibility companion. It supplies only the legacy diagnostic
column/masks required by the current exporter; it is never a source of beam
weights.

The nside-64 raw export completed successfully:

- 11 time bins x 630 baselines x 649 unique fine channels x two hands;
- MS phase centre RA 103.2460 deg, Dec -30.2850 deg;
- files `crab_240mhz_cal_5min.ms`, `_legacy.ms`, and `.uvfits` under the
  calibration-run directory;
- 495291/8995140 MS samples (5.506%) are intrinsically invalid/unsupported.

CASA 6.7 is usable directly from the commissioning virtualenv even though the
documented standalone binary is gone and `/usr/local/bin/casa` points to an
unmounted `/nfs/casa`. Use `CASASITECONFIG=scripts/casa_lambda_config.py` plus
the existing private measures directory and invoke `calibrate_night_ms.py`
with that virtualenv Python.

Calibration trials and their rejection evidence:

- all-sky K/B/phase/G at `minsnr=3`: final `gain.G` had no unflagged solutions;
- all-sky at `minsnr=1`: only 39/72 final receptor gains survived;
- a clean-flag, phase-final trial excluded empirically dead A1/A11/A20 plus
  required A14/A36. It produced 59/72 final gains, but bandpass coverage was
  only 27878/46728 (59.66%). Full-data model coherence remained only
  0.00357 -> 0.01298 (XX) and 0.01092 -> 0.03736 (YY), far below accepted
  commissioning values around 0.4--0.7;
- a Tau A component-model solve with a 7 m baseline cut was worse: 11/72 final
  gains and 6030/46728 bandpass samples survived; corrected XX coherence was
  only 0.0289 and YY had no usable corrected samples;
- the Sun was at 11.69 deg elevation during the calibration, while Crab was at
  34.42 deg. A bounded commissioning solar solve takes about 14 minutes per
  refinement pass per hand. The default 30-pass run was stopped before output.
  A one-pass unbounded diagnostic produced pathological reference gains with
  phases only 0/180 deg and amplitude outliers up to ~1.6e4, and was stopped
  before writing a calibration group.

The file `crab_240mhz_fine_gains_clean31.json` is a schema/provenance test
artifact only. It excludes whole antennas A1,A11,A14,A20,A36 and receptors
A13-YY,A24-YY,A25-XX, and it passes structural validation, but its parent
calibration fails the coherence QA above. **Do not observe with it.**

CASA `applycal(..., applymode="calflag")` mutates MS flags. Never run successive
calibration trials on the same MS and assume they are independent.
`scripts/reset_ms_flags_from_raw_data.py` reconstructs the exporter's original
raw validity mask from non-finite DATA, and
`scripts/summarize_casa_calibration.py` reports K/B/phase/G coverage. Make a
new MS copy and reset it before each trial. The Tau-A helper
`scripts/set_ms_point_source_model.py` writes an explicit off-axis component
model, but that trial did not validate.

### 2026-09-23 solar retry paused; software signal-chain audit

The operator requested a fresh 40-coarse-channel solar calibration for the
Crab, with the inter-FPGA delay solve at the clear 8-channel band centred on
selector 196. The corrected `build/apps/starspin_4_40` was built and had
`cap_net_raw=ep` before the audit. All four Alveos were rebooted successfully.
`configure-fpga.sh -a 0,1,2,3 -n 8 -f 196 -m init-tune -g 20` then selected
coarse 192--199, but on **every** FPGA the JESD clock PLL reported locked
while all five quads showed rapidly changing error/character counters and
`RXBUFSTATUS=1111 1111 1111 1111 1111` (ADC sync error). The script still
exited 0 despite those error logs. A second init pass was interrupted after
the operator suggested that ADC power may be down. The script checks JESD
before issuing its `adc` command, so the first-pass result alone cannot
separate a sequencing problem from ADC power/clock distribution failure.
**No delay solve, 40-channel retune, solar capture, or accepted calibration
was performed.** Current FPGA state is partially initialized and must be
rechecked after power is confirmed. The old delay file must not be assumed
valid across these reboots.

The software audit drove synthetic *wire-format* packets through the real
`ProcessorState`, Starspin CUDA pipeline, and host visibility output, with
independent CPU expectations. Its findings are:

| Stage | Evidence / result |
|---|---|
| Parse and ibverbs layout | Actual framed size is 2664 B; ibverbs zero-copy slot is 2622 B (CustomHeader + 2600-B payload). A new regression proved a 2660-B truncated frame was previously accepted with a non-null payload and 2596-B reported payload size. The parser now rejects incomplete frames; both valid layouts pass. |
| FPGA scales, channel, hand, baseline | New four-FPGA, two-coarse-channel CPU oracle checks every complex visibility from distinct per-FPGA/receiver/hand samples and scales. All real and imaginary values pass. Stored baseline `(i,j)` is `E_i * conj(E_j)`; the commissioning MS exporter explicitly conjugates at the export boundary. |
| Fine PFB and correlation | New four-FPGA, two-coarse-channel test sends independent DC and +2-bin tones. They peak in retained local bins 3 and 5 after FFT shift/edge trim; cross-FPGA phases and scale ratios match an independent oracle. |
| Antenna map | The active `stream_antenna_map.json` contains the A8/A12 receptor reversals. Existing antenna-ordering tests cover permutation, polarisation swaps, disconnected slots, and HDF5 baseline IDs. Physical hand labels are Y then X on the raw correlator axis; the current commissioning MS exporter uses `RAW_POLS={YY:(0,0), XX:(1,1)}`. The older `read_hdf5_data_capture()` helper instead labels raw 0 as XX, so its legacy `calibrate-sun` path is not a reliable acceptance test without correction. |
| Accumulator and HDF5 frequency metadata | Starspin used `atomicAdd` into a `cudaMalloc` visibility accumulator without explicitly zeroing it. It is now zeroed at construction. The app previously wrote `max_channel=min_channel+post-PFB bin count-1`; it now writes the true coarse-channel endpoint (`min_channel+NR_FPGA_CHANNELS-1`). Existing commissioning readers contain a compatibility branch for older incorrect metadata. |
| HDF5 sequence metadata | In the earlier post-scale-fix 24-channel capture, `vis_seq_nums` starts at 18446744072883320576 and every end value is zero, although the 303 visibility blocks and last packet total are nonempty. Starspin stored a 64-bit FPGA sample count in a signed 32-bit `int` and never propagated the real end count. A new test using counter 8589935592 failed before the fix (observed start 1000, end 0). Starspin now stores full 64-bit start/end counts and skips zero-unit shutdown dumps. The old capture's sequence metadata cannot be repaired without an independent absolute counter anchor; its visibilities themselves remain as recorded. |

Focused test source: `tests/test_fine_channelized_starspin.cu` and
`tests/test_packet_formats.cpp`. The parser regression was observed failing
before its fix; the 64-bit sequence regression also failed before its fix.
After the fixes, 15 packet/stream-map CPU tests, all four focused Starspin
tests (including both new four-FPGA golden tests), and all 28 focused ibverbs-placement,
reorder, antenna-ordering, PFB, and HDF5 writer tests pass. `git diff --check`
also passes. This is strong evidence for the
software arithmetic and indexing with ideal synthetic packets; it does not
prove live ADC samples, shared-clock coherence, absolute antenna wiring, or
a scientifically valid solar gain solution. A 40-channel on-sky capture and
held-out calibration coherence check remain the acceptance gate after power
is restored.

#### Cross-FPGA coherence in the last high-band capture

I sampled the post-scale-fix 24-channel capture
`build/calibration_runs/crab_240mhz_20260923/crab_240mhz_cal_5min.hdf5`
at 5 times across its five minutes and at fine bins 13, 349, and 657
(coarse channels 296, 308, 319). For each baseline I calculated
`abs(Vij)/sqrt(real(Vii)*real(Vjj))`, excluding A1/A11/A14/A20/A36 and
nonpositive autos, and grouped by the FPGA IDs recorded in the capture's
embedded antenna mapping. Distance matching matters: at 5--10 m and raw
hand 0, median coherence was 0.1017 within FPGA 1 (195 measurements),
0.0052 for FPGA 0--1 (165), 0.0054 for 1--2 (840), and 0.0052 for 1--3
(750). The corresponding other cross-board pairs were 0.0955 for 0--2,
0.0322 for 0--3, and 0.0454 for 2--3. Raw hand 1 gives the same pattern:
within FPGA 1 is 0.0169; all cross-pairs involving FPGA 1 are about
0.0050--0.0053. The pattern is stable in the sampled times/frequencies.

This establishes a **capture-specific inter-FPGA coherence failure involving
FPGA 1**, not just an antenna-distance effect or a dead FPGA 1 input.
It does *not* identify the physical cause: wrong/stale inter-FPGA delay,
clock distribution or drift, live input timing, and other hardware-dependent
effects remain possible. The embedded audit manifest gives the exact delay
values used in this capture (`0:8241432, 1:0, 2:-8402956, 3:-17079637`),
so do not rely on the mutable current `alveo_delays.json` to interpret it.
The synthetic four-board tests make a simple software scale/channel/baseline
indexing fault less likely, but they cannot prove live timing. Once power is
restored, solve fresh delays at 8 channels centered on selector 196 and
*before* attempting a solar gain solve, inspect distance-matched within- and
cross-FPGA coherence in a short 40-channel capture. Require FPGA 1 to show
real cross-board signal rather than a ~0.005 noise floor.

`starspin_4_40` was rebuilt again at 16:04 AEST, including the 64-bit
sequence-metadata fix. All four focused Starspin GPU tests passed after that
build; the 28-test placement/reorder/antenna/PFB/HDF5 suite was rerun and
passed 28/28. `git diff --check` passed. The relink cleared its file capability
(`/usr/sbin/getcap` returned empty), so before the next live ibverbs run the
operator must run:

```bash
sudo setcap cap_net_raw+ep /data/BLACKMESA_1/sma075/cuda-spatial-filtering/build/apps/starspin_4_40
```

Do not run GPU tests concurrently with a live capture.

### 2026-09-23 17:22 AEST hardware retry and grouped-delay proposal

At the operator's request, `reboot_all_alveos.sh -a 0,1,2,3` completed:
all four Vivado `boot_hw_device` operations reported `Done pin status: HIGH`.
`configure-fpga.sh -a 0,1,2,3 -n 8 -f 196 -m init-tune -g 20` then
connected to each board and set 192--199 with filterbank packets enabled.
All four report JESD PLL lock `0xF` and 100G GTM lock, but all five JESD quads
on every FPGA still have rapidly changing error/K-character counters and
`RXBUFSTATUS=0xFFFFF`. A **read-only post-configuration check** showed this
persists after the script's `adc` command. On FPGA 0 all 20 ADC maximum
statistics read exactly `32768`; running `init` a second time after `adc`
still reported errors on every quad and `RXBUFSTATUS=0xFFFFF`, then `fbon`
restored packet output. Thus the script-ordering concern does not explain
the fault. FPGA control and Ethernet are live, but the ADC/JESD input is
not healthy. `configure-fpga.sh` exits 0 despite these reported errors.
Do **not** perform a delay solve, solar calibration capture, or accept new
weights until the common ADC power/clock/sync issue has been resolved and
post-init flags/counters are clean. The exact upstream cause is not yet proven.

The requested commissioning experiment is a *hierarchical* delay model:
`tau_antenna = tau_FPGA_group + tau_receptor_residual`, with one reference
FPGA fixed to zero and zero-mean residuals within each group. This is a useful
diagnostic/regularizer, but it adds no independent degrees of freedom over
CASA's existing antenna-based `gaincal(gaintype="K")` if left unconstrained.
Compare ordinary K/B/G against a group-pooled clock-delay prior followed by
residual per-antenna K/B/G on the **same fresh, scale-corrected capture**;
judge on held-out cross-FPGA coherence and gain coverage, not solver success
alone. Do not use the old pre-FPGA-scale-fix data for the comparison. A
group term cannot recover cross-board visibility already decorrelated before
the approximately 1-second HDF5 integration, so short-integration or raw
packet timing remains necessary if the new capture repeats the ~0.005
FPGA-1 cross-board coherence floor. No group-delay calibration code or
comparison result is in place yet.

#### 18:22 AEST additional all-FPGA reboot

The operator requested one more reboot. `reboot_all_alveos.sh -a 0,1,2,3`
again reported `Done pin status: HIGH` on each of the four boards. Full
`init-tune` to 8 channels centered at selector 196 (192--199, ADC gain 20)
again reported JESD PLL lock, but every board and all five quads produced
changing JESD error counters and `RXBUFSTATUS=1111 1111 1111 1111 1111`.
Importantly, a **separate read-only check after the `adc` command completed**
showed `jesd_clock_status=0xF`, `rxbufstat=0xFFFFF`, and two different
`adc_box1_jesd_rx_errors` readings on each board. All four report filterbank
packets enabled and 192--199 tuned. Thus this third all-board reboot did
not fix the common ADC/JESD fault. Do not solve delays or capture calibration
from this state. The delay file is stale after this reboot.

### Coarse-PFB de-rippling (shared fine-channel implementation; firmware read-only)

The upstream firmware coefficients are in the **separate, read-only**
`/data/BLACKMESA_0/lambda/lambda_firmware/ip_1024/filtertaps_1.mem` through
`filtertaps_18.mem`. The firmware's
`matlab1024/write_filterbank1024_roms.m` specifies 512 36-bit ROM words per
file, each packing two signed 18-bit FIR coefficients, low 18 bits first.
`vhdl/filterbank_1024.vhd` confirms 18 taps, 1024 FFT points and 32/27
oversampling. No firmware file may be modified for this work. Verify that the
running bitstream's tap ROMs match this source (the live firmware reports
build hash `0xC088A0E1`) before adopting a correction table.

A preliminary read-only decode of the ROMs gives 18,432 coefficients. Its
prototype response at the 28 retained fine-bin centres (offsets -0.5 to
+0.5 coarse bins in steps of 1/27) is -0.1844 to -0.0125 dB relative to
the coarse-bin centre. This is a **first-stage tone-centre estimate**, not
yet the complete two-stage PFB response. The maximum inverse amplitude over
the retained band is only 1.0215; guard bins must remain discarded. This
small correction flattens shape but cannot improve intrinsic SNR or recover
lost/aliased signal. Measure the actual two-stage fine-bin response before
calling a GPU change worthwhile.

Implementation and validation (2026-09-23):

1. Make an offline generator that reads the firmware ROMs *without writing to
   that repository*, decodes their signed coefficients, simulates the coarse
   PFB followed by the same 32-bin GPU fine PFB, and measures each retained
   bin's voltage response using tones and flat-spectrum noise. Compare the
   predicted shape with measured valid hardware data once ADC/JESD sync is
   restored. Record the 18 ROM-file hashes and active firmware build hash.
2. Runtime config lives outside source control and contains only 28 real,
   positive *voltage* gains. Schema v1 fields are `kind` =
   `coarse_pfb_amplitude_deripple`, `fine_channels` = 32, `edge_trim` = 2,
   `normalization` = `coarse_bin_centre`, nonempty `response_id`, and
   `voltage_gains` (28 entries, each in (0, 1.1]). No production coefficient
   table or ROM bytes live in this repo. `--deripple-config PATH` is accepted by
   apps using `parse_common_args`; the shared fine channelizer validates and
   uploads the table once per pipeline at construction.
   The `response_id` must identify a validated, active firmware response;
   this cannot yet be asserted from the observed bitstream hash alone.
3. The correction is fused into all three shared fine-channel gathers, after
   GPU fine-channelization but **before** correlation/beamforming/detection:
   complex correlation input and beamforming input get voltage D; antenna
   spectra get power D². Visibilities therefore scale by D² for a common
   per-fine-bin D on all antennas; beam voltage scales by D. Starspin and
   other fine-channel binaries now apply it when the flag is given. There is
   no extra GPU pass or copy. Pulsar's previous DADA-output correction was
   removed so it cannot apply D twice. The no-config branches retain the
   original direct assignments. Coarse-only (`NR_FINE_CHANNELS=1`) builds
   have no fine-bin gather and should not be launched with this option.
4. A corrected Starspin capture is now required for matching calibration.
   Its HDF5 audit embeds the exact runtime config file. Export per-fine CASA
   gains with `scripts/export_casa_fine_beam_gains.py --deripple-config PATH
   --calibration-visibilities CAPTURE.h5`; the exporter compares the requested
   config against the embedded audit and **does not rescale** the solved gains.
   This exporter audit check needs Python `h5py` in the environment used for
   CASA gain export; the repository's CMake/conda Python currently lacks it,
   while `/usr/bin/python3` has it on this host.
   The audit check verifies the HDF5 file the operator supplies; it cannot
   independently prove that the CASA solution directory came from that file.
   It stamps `deripple_response_id`, the exact voltage-gain table, and
   `deripple_export_state=measured_from_derippled_visibilities` for pulsar-fold
   to validate. Old gains from uncorrected visibilities must not be reused
   with the corrected beam. The pulsar DADA header gets `DERIPPLE`/`DRIPRID`.
5. Focused C++/CUDA tests cover schema failures, provenance mismatch, and all
   three shared gathers with/without D. Python tests cover matching/mismatched
   capture audit. These synthetic tests establish implementation correctness,
   **not** measured flattening or sensitivity gain. Before observing, verify
   actual firmware ROM identity, simulate/measure the full two-stage response,
   create the runtime table, and compare matched short captures. Do not enable
   the provisional first-stage-only table for science observations.

Build verification of the shared-gather change passed for representative
fine-channel binaries: `starspin_4_40`, `observe_4_40`, both 40-channel pulsar
variants, `beamformed_bandpass`, `adaptive_beamformed_bandpass`,
`fft_antenna_spectra`, and `get_projection_matrix`. The pre-existing
fine-channel observe/bandpass steering-gain type mismatch was fixed in those
apps so they compile with per-fine calibration gains. All variants share the
same gather code, but other channel-count/FPGA-count binaries were **not** all
rebuilt. Rebuilding Starspin removed its `cap_net_raw` file capability; before
a live ibverbs run, ask the user to apply `sudo setcap cap_net_raw+ep
/data/BLACKMESA_1/sma075/cuda-spatial-filtering/build/apps/starspin_4_40`.

The exact CUDA gather tests pass for complex correlation input (D), complex
beam input (D), antenna power (D²), and no-config identity, across two coarse
channels. Python tests validate the file schema and capture-audit match (the
HDF5 loader test passes under `/usr/bin/python3`, which has `h5py`). A synthetic
single-run Starspin capture with a table and a single-run pulsar beam with a
table both pass. Earlier two-run-in-one-process comparison tests were removed
after intermittent segfaults in the synthetic pipeline harness. The harness
also had a lifetime-order bug: its processor state could be destroyed while a
pipeline callback/reclaimer still referenced it. The affected tests now
explicitly destroy the pipeline first. The previously flaky existing
uncorrected Starspin CPU-oracle test passed five consecutive repetitions;
the full focused de-ripple/fine-channel/audit suite passed 16/16 tests with
`ctest --test-dir build -R 'Deripple|FineChannelizedStarspinTest|FineChannelizedPulsarFoldTest|HDF5RunAuditTest' --output-on-failure -j1`
on 2026-09-23, outside the sandbox. `git diff --check` also passed.
Successful compilation and synthetic checks are not evidence of a valid sky
response. The plot at
`/tmp/lambda_coarse_pfb_deripple_preview.png` is deliberately outside the
repository and depicts only the ROM-derived first-stage estimate, not a
science-ready correction table.
