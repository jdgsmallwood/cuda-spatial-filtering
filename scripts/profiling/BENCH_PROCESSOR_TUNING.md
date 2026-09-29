# `bench_processor` throughput tuning — findings log

> **TL;DR after round 3:** 3.2 GB/s (as found) → **~14.7 GB/s mean / 16.8 peak** with
> `packets_missing = 0` on every config, via a NUMA pin + a production watermark-race fix +
> 3 strided producers + 6 workers.  Two real production bugs found and fixed along the way
> (completion-watermark race, shutdown barrier race) plus one production hazard flagged
> (ring size must be divisible by nr_capture_threads).  Details in "Round 3" below.
>
> **Round 4:** rolled GPU-NUMA-node pinning out to the six real (non-bench) apps via
> `parse_common_args`, and worked out the recommended per-NIC capture-thread placement for
> this box's real 4-NIC (3 currently live) topology.  Details in "Round 4" below.

Task: `apps/bench_processor` was getting **3–7 GB/s** on this box but **28 GB/s** on a
home machine. Goal: recover as much throughput as possible here. Constraints: tests must
still pass; the reassembly path must still deposit into CUDA **pinned** memory.

All measurements below are the mean `GB/sec` across the 8 built-in configs (each counts the
full 2664-byte wire packet), `--duration 3`, unless noted. Numbers are noisy (±0.3 GB/s).

## The machine (why it differs from "home")

```
2× Intel Xeon Silver 4210 @ 2.20 GHz   (dual socket, 20 cores total, NO hyperthreading)
2 NUMA nodes:  node0 = cpu 0–9,  node1 = cpu 10–19
L3: 27.5 MiB total but SPLIT into 2 instances → ~13.75 MiB per socket
AVX-512 present (avx512f/bw/cd/dq/vl)
perf(1) unavailable: /proc/sys/kernel/perf_event_paranoid = 3
```

The two facts that explain the gap vs. the "home machine":

1. **Split L3.** The existing code sized the ring at `8192 slots × ~2752 B ≈ 22 MB`,
   with a comment saying it "fits within the machine's 32 MB L3." That is true for a
   single unified 32 MB L3 (the home desktop). On this box each socket only sees
   **~13.75 MB** of L3, so a 22 MB ring **spills to DRAM on every worker read**.
2. **Dual socket / NUMA.** Nothing pinned the producer / processor / feeder / worker
   threads, so the scheduler scattered them across both sockets. Every ring read and
   every pinned-buffer write could then cross the inter-socket link.

## Key structural observation

`packets/sec` is ~constant (~1.2M/s at baseline) across every config, and every config
emits the **same 2664-byte packet** (payload size is fixed by `NR_RECEIVERS_PER_PACKET`,
`NR_TIME_STEPS_PER_PACKET`, `NR_POLARIZATIONS` — none of which vary across the 8 configs;
the multi-FPGA configs just split the same 10 receivers across 4 sources). So the bench is
**per-packet-overhead / latency bound, not payload-bandwidth bound.**

## Diagnostic experiments (each reverted after measuring)

Isolating where the per-packet time goes (single socket, ring=2048, WC=4, ~8.5 baseline):

| Experiment                                             | avg GB/s | Interpretation |
|--------------------------------------------------------|---------:|----------------|
| Full path (optimized baseline)                         | 8.5      | — |
| Skip the `copy_nt` deposit into pinned memory          | 9.3      | the pinned-mem deposit is only ~10% |
| Skip the **producer's** 2664 B memcpy into the ring    | 10.8     | single-threaded producer is ~20% |
| `atomic_max_u64(latest_packet_received)` → relaxed store| 8.65     | that CAS is **not** contended (monotonic feed → succeeds first try) |

Conclusion: the cost is **diffuse per-packet work** (producer memcpy, parse, ring
bookkeeping, fork/join sync), not one hot line. The deposit copy itself is cheap, so we are
nowhere near NT-store bandwidth — this is a latency/overhead regime.

## What moved the needle (all changes are in `apps/bench_processor.cu` only — no shared code touched, so tests are unaffected)

Cumulative, each building on the previous:

| Change                                                          | avg GB/s |
|-----------------------------------------------------------------|---------:|
| **Baseline** (as found)                                         | ~3.2 |
| Cheaper producer: build one wire template once, per-packet just `memcpy` it + patch the 14-byte header (was rebuilding 1280 samples via per-element lambda calls) | ~3.5 (noisy) |
| + Pin everything to one NUMA node (`numactl` proved it: ~5.2)   | ~5.2 |
| + Shrink ring `8192 → 2048` so ~5.6 MB fits in one socket's L3  | ~8.0 |
| + `WORKER_COUNT 6 → 4`                                          | ~8.7 |

Final: **~8.0–8.9 GB/s**, i.e. **~2.5–2.7× the 3.2 GB/s baseline**, with no external
`numactl` needed (the NUMA pinning is baked into the binary).

### Verified before/after (same optimized binary)

| Config                                         | mean GB/s |
|------------------------------------------------|----------:|
| Original binary (as found)                     | ~3.2 |
| Optimized, pinning **on** (default)            | ~8.0–8.6 |
| Optimized, pinning **off** (`SPATIAL_BENCH_NODE=-1`) | ~3.7 |

The pinning-off row shows NUMA confinement is the dominant lever: the ring-size / worker /
template changes barely help until the run is confined to one socket (then the small ring
stays hot in that socket's L3). Every run has `packets_missing = 0` and only a handful to a
few thousand `packets_discarded` out of ~8–10M processed, i.e. correctness is unchanged.

### Test status

The bench changes are confined to `apps/bench_processor.cu`. There is also **one shared-code
change** in `include/spatial/spatial.hpp` (the `packets_processed` sharding, below): it adds a
defaulted `processed_accum` param to `process_packet_data` so concurrent workers accumulate
locally and flush once per slice; the single-threaded callers (`process_all_available_packets`,
the future-queue drain) pass the default `nullptr` and increment the global atomic exactly as
before — so the exact counts the tests assert are preserved. Verified by building
`ProcessorTests` against the modified header (`build_test`, `BUILD_TESTING=ON`): **12/12
PASSED**, including the `packets_processed` count assertions and
`MissingPacketCountIsZeroForCompleteBuffer`. The worker (accumulator) path is additionally
validated by the bench itself: `packets_processed == packets_fed` with `packets_missing = 0`.

### Details

- **NUMA pinning baked in.** `pin_to_numa_node()` reads
  `/sys/devices/system/node/node<N>/cpulist`, builds a `cpu_set_t`, and
  `sched_setaffinity(0, …)` on the whole process *before any allocation*. Child threads
  inherit the mask; default first-touch then places the ring + pinned buffers node-local
  (the constructor runs on this thread). Honours `SPATIAL_BENCH_NODE` (default 0; `-1`
  disables). No libnuma dependency.
- **Ring size sweep** (single socket): 8192→4096 gave ~7, 4096→2048 gave ~8 (best),
  1024 dropped to ~6.5 (producer starts stalling), 512 nearly deadlocks. 2048 = the knee.
- **Worker count sweep** (single socket, ring=2048): 2→6.1, 3→8.1, **4→8.5**, 5→7.2,
  8→7.1, 9 oversubscribes 10 cores and thrashes. More workers *hurt* → the limiter is
  shared-state / memory contention and fork-join tail latency, not compute. 4 is the knee
  (producer + processor/dispatcher + feeder + 4 workers = 7 threads on 10 cores).
- **Template producer.** The payload is identical for every packet (constant sample/scale
  fns), so rebuilding it per packet was pure harness overhead starving the consumer.

## Pinned-memory requirement — still satisfied

The deposit target `LambdaFinalPacketData::{samples,scales,arrivals}` is allocated with
`cudaHostAlloc(...)` (`include/spatial/packet_formats.hpp:132`). `copy_nt` streams the
payload into `samples` (pinned). None of the bench changes touch this — the reassembly path
still deposits into pinned host memory exactly as before.

## Round 2 — trying to break the ~10 GB/s ceiling

The "~10–11 GB/s" ceiling comes from one experiment: commenting out the producer's per-packet
memcpy (feeding the ring "for free") raised throughput to ~10.8. So the single producer is
~20% of the cost, and the two things pinning the ceiling are (1) the single-threaded producer
and (2) whatever caps useful workers at 4. We tried to fix both.

### Contention: sharding `packets_processed` (kept — safe, correct)

Every processed packet did a `fetch_add` on one global atomic from all workers — a classic
ping-pong. Changed the workers to accumulate locally and flush once per dispatched slice
(single-threaded callers unchanged; see Test status). **On this box it was within noise**
(~8.0–8.5 either way): with the current single producer the workers rarely run all at once, so
that cacheline isn't actually hot. Kept anyway — it's correct, it removes a real latent
scaling bottleneck, and it costs nothing. (Diagnostic aside: turning the per-packet
`atomic_max` on `latest_packet_received` into a plain relaxed store also did nothing — 8.65 —
so that CAS isn't contended either. The per-packet cost is genuinely diffuse, not one hot
atomic.)

### Multi-producer: reached 10–12 GB/s but is NOT correct — reverted

Fed from 3 producer threads via the existing strided path (`nr_capture_threads>0`,
`reserve_write_batch_strided` + `commit_write_batch`, partitioning whole `(channel,fpga)`
streams round-robin). Results:

- **No cross-producer sync:** several configs hit **10–12 GB/s** (ch=32f1 12.2, ch=8f4 11.8,
  ch=24 10.3) — but with **millions of `packets_missing`**. Producers drift apart without
  bound, so a fast stream fills far-future buffers while a lagging stream's packets never
  arrive and the reassembler completes those buffers with the laggard counted missing.
- **Drift-≤1 throttle** and **hard per-round barrier:** both fixed the drift for the
  evenly-divisible configs (ch=24 stayed correct at ~11 GB/s) but *stalled* most others to
  near-zero, and a big ring (16384, needed so one round fits) both reintroduced
  `packets_missing` and destroyed the L3 locality win.

**Root cause (the reason it can't be made correct in the bench alone):**
`check_buffer_completion()` marks a channel done once
`latest_packet_received[channel][fpga] >= end_seq + NR_BETWEEN/2`, but
`copy_data_to_input_buffer_if_able()` bumps `latest_packet_received` (`spatial.hpp:444`)
**before** the future-queue deferral (`:450`) **and before** the `copy_nt`. So a packet that
is only *seen* — deferred or not yet copied — can satisfy the completion test. A single
sequential producer never exposes this (the processor only checks completion after
`dispatch_and_wait`, and in-order feed keeps deferrals rare); concurrent/reordered feed does.
Making the parallel feed correct therefore needs a **production reassembler change** (move the
watermark update after the copy, or exclude deferred packets), whose effect on real
observations we can't validate here — so it's left as a flagged finding, not a change.

### Net

Kept the correct single-producer bench (~8.0–8.9 GB/s, `packets_missing = 0`) plus the safe
`packets_processed` sharding. The honest ceiling on this dual-socket, split-L3, 2.2 GHz box for
this path is ~10 GB/s single-socket; the parallel-producer route to it is gated on a
production reassembler fix (the watermark-before-copy ordering above) that's out of scope for
a benchmark.

### If someone wants to pursue it (in priority order)

*(Round 3 below did exactly this — kept for the historical record.)*

1. Fix the `latest_packet_received` watermark ordering in
   `copy_data_to_input_buffer_if_able` (update it only for packets actually copied into a
   buffer, i.e. after the deferral check and copy). This is the real blocker; it may also be a
   latent robustness issue for genuinely out-of-order live capture. Needs domain review +
   the full reassembler test suite.
2. With that fixed, re-enable the strided multi-producer feed (3 producers) + a hard per-round
   barrier, and a ring large enough that one round fits without spilling L3 (tension with the
   locality win — may need per-config ring sizing).
3. Only then revisit worker count — cutting contention may raise the useful-worker knee above 4.

## Round 3 — production fixes + bounded-skew multi-producer → ~14.5 GB/s

User-approved direction: fix both problems properly.  Domain fact from the user that made it
safe: **real FPGA streams stay roughly sequence-aligned over time** — bounded skew is the
production reality, and the reassembler's slack (8 buffer windows + half-window completion
margin) is designed to absorb it.

### Production fix 1: the completion-watermark race (`spatial.hpp`)

Moved the `atomic_max_u64(latest_packet_received…)` bump **below** the future-queue deferral
branch in `copy_data_to_input_buffer_if_able()`.  Deferred (beyond-horizon) packets no longer
advance the completion watermark at parse time — they advance it when the drain re-injects
them through the copy path.  This kills the race where a deferred packet's window completes
the instant it rotates into existence, before the drain lands the data (silently zero-filled
as "missing").  **This path is live in production today** for multi-queue capture
(`nr_capture_threads > 0`), so this was a real capture-loss bug, not just a bench artifact.

Loss-recovery semantics preserved via a **stall safety net** (user chose this over
strict-stall): the drain (`drain_future_packets`, extracted from `process_packets` so tests
can call it) arms a timer when a stream's queue head is stuck beyond the horizon; if still
stuck after `future_stall_force_threshold` (default 100 ms ≈ 6 production buffer periods —
the hysteresis that distinguishes a genuine stream jump from benign skew), it force-bumps the
watermarks for every queued packet of that FPGA so buffers complete zero-filled and the
horizon chases the jumped stream.  Same recovery outcome as the old parse-time bump, gated so
it can never race benign skew.

Tests: existing `MissingPacketHandlingTest` (which encodes the jump-recovery semantics)
updated to trigger the safety net explicitly (zero threshold + two drain calls); new
regression test `FuturePacketDoesNotPrematurelyCompleteBuffer` **fails against the old code,
passes with the fix** (verified both ways).  `PacketOrder` gained a `freq_channel` field so
the safety net knows which watermark to bump.

### Production fix 2: the shutdown barrier race (`spatial.hpp`)

Hunted an intermittent residue of tens-to-hundreds of missing packets that survived fix 1.
Instrumented hole positions at completion → holes were scattered mid-window packets whose
channel watermark was already a round ahead, yet the packets never appeared as discarded /
stuck / future-queued / consumer-skipped (all counters zero) — they were mid-copy.  Root
cause: `dispatch_and_wait()`'s completion barrier **broke out early on `!running`** while
workers were still copying, and the final `handle_buffer_completion()` then raced the
in-flight copies — completing buffers whose arrival flags weren't written yet.  Another real
production bug (shutdown during live capture could zero-fill the last buffers' real data).

Fix: workers' committed-spin now bails on `!running` (so slices always terminate even if a
producer bailed pre-commit at shutdown), a `workers_alive` counter distinguishes "workers
still finishing" from "workers gone", and the barrier waits until slices actually finish.
After this: zero missing across every configuration, every run.

### Production hazard flagged: strided ring divisibility

With N capture threads, thread t owns ring slots t, t+N, t+2N…  Slot ownership is only
**disjoint** when N divides `RING_BUFFER_SIZE`; otherwise (e.g. N=3, ring 2048/8192) the
residues interleave and two threads whose claim rates drift more than a ring apart claim the
SAME slots and serialise on each other's `processed` flags — observed as multi-second stalls
under uneven per-thread packet rates (bench: 8 streams over 3 producers → 3,3,2 shares →
0.03–0.5 GB/s).  `process_packets()` now ERROR-logs at startup if
`RING_BUFFER_SIZE % nr_capture_threads != 0`.  Live capture with RSS-uneven queues + a
non-divisible ring would hit the same coupling.

### Bench: bounded-skew multi-producer + tuning

3 strided producers (same reserve/fill/commit protocol as `KernelSocketPacketCapture`),
whole `(channel,fpga)` streams partitioned `stream % 3`, and a **bounded-skew throttle**
(per-producer completed-round atomics; nobody starts round R until the slowest has completed
R−2) replacing the failed hard barrier — matching the "roughly sequence-aligned" reality
without serialising producers.

Ring constraints discovered while sweeping:
1. Divisible by PRODUCER_COUNT (above).
2. `RING / PRODUCER_COUNT ≤` the smallest config's horizon (~2056 packets for 1-channel):
   ring 8190/3 = 2730 lets a producer outrun the horizon → floods the future queue → trips
   the 100 ms safety net → periodic force-completes with holes (observed: missing=4494).

Sweep (mean of 8 configs, 2 s each; missing must be 0 to count):
| PC | RING | WC | mean GB/s | missing |
|---:|-----:|---:|----------:|--------:|
| 3 | 2046 | 6 | 13.4 | 0 |
| 2 | 2048 | 7 | 14.0 | 0 |
| 3 | 4092 | 6 | 14.3 | 0 |
| **3** | **6144** | **6** | **14.5** | **0** |
| 3 | 8190 | 6 | 14.1 | 4494 (horizon) |
| 4 | 4096 | 5 | 13.2 | 0 |
| 4 | 4096 | 6 | 0.7 | — (12 busy threads / 10 cores: collapse) |

Winner: **PC=3, RING=6144, WC=6** → 3 producers + 6 workers + processor = 10 busy threads on
the 10-core socket (feeder sleeps in synchronous mode).

### Final verified numbers (4 s runs, repeated)

| Config | GB/s |
|---|---:|
| ch=1 (single stream — inherently serial) | ~7.0 |
| ch=8…32, 1 and 4 FPGA | 14.4–16.8 |
| **Mean across 8 configs** | **~14.7** |
| `packets_missing` / `future_queued` / `stuck` | **0, every config, every run** |

History: 3.2 (found) → 8.5–9.2 (round 1: NUMA pin + ring + template producer) → **14.7**
(round 3).  ~4.6× total.  The previous "~10–11 GB/s ceiling" was the single-producer design
ceiling, as predicted; the new wall is consumer per-packet overhead across 6 workers with all
10 cores busy.

Tests after all shared-code changes: `ProcessorTests` **13/13** (12 existing + new
regression), `PipelineTests` **17/17**, `PipelineHarnessSelfTest` **2/2**.

### Deployment notes (from `nvidia-smi topo -m` on this box)

- **GPU0 (RTX 4000 Ada) is on NUMA node 0** (CPU 0–9); NIC0 on node 0; **NIC1–3 on node 1**.
  Live capture on NIC1–3 with GPU-side pinned buffers on node 0 has a built-in cross-socket
  hop — choose capture-thread affinity (`SPATIAL_WORKER_CPUS`) and buffer placement
  accordingly, or capture on NIC0.
- Dual-instance (one pipeline per socket) was considered and **dropped**: with a single
  node0 GPU, a node1 instance would DMA across UPI into a shared GPU — the ~2× aggregate a
  bench would show is unusable here.  Elsewhere it would need: one GPU per socket, sharding
  by frequency channel (runtime `min_freq_channel` already supports it), and channel-group
  steering *before* the socket (FPGA dest-port config or XDP — `freq_channel` is invisible
  to NIC RSS/flow rules).

## Round 4 — rolling the NUMA pin out to the real (non-bench) apps

`bench_processor`'s `pin_to_numa_node()` is a synthetic-benchmark trick: with no real
hardware to respect, pinning the *whole* process (capture + reassembly) to one socket was
pure upside.  The real apps (`observe`, `beamformed_bandpass`, `adaptive_beamformed_bandpass`,
`fft_antenna_spectra`, `get_projection_matrix`, `pulsar_fold` — everything that calls
`parse_common_args`) capture from **physically fixed NICs**, so the same blanket pin would be
wrong: it would force a live capture thread for a node1 NIC onto node0 cores, turning a
node-local `recvmmsg` copy into a cross-socket one that doesn't exist today.

### Real topology on this box (`nvidia-smi topo -m` + sysfs, confirmed 2026-07-07)

| Device | NUMA node | Notes |
|---|---|---|
| GPU0 (the only GPU) | **0** | cpu 0–9 |
| NIC0 `enp94s0np0` (mlx5_0, → FPGA0) | **0** | up, 100 Gb |
| NIC1 `enp134s0np0` (mlx5_1, → FPGA1) | 1 | **down** — not currently connected |
| NIC2 `enp175s0np0` (mlx5_2, → FPGA2) | 1 | up, 100 Gb |
| NIC3 `enp216s0np0` (mlx5_3, → FPGA3) | 1 | up, 100 Gb |

So "capture from all 4 NICs" is currently 3 live streams (FPGA0, FPGA2, FPGA3) — FPGA1's NIC
has no link.  Per-NIC RX-queue IRQs are already spread across their own node's cores (checked
`/proc/irq/*/smp_affinity_list`): NIC0's queues → cpu 0,1,2,3,4…; NIC2/NIC3's queues → cpu
10,11,12,13,14… (both node1 NICs currently share the same core range for their first several
queues — a secondary, lower-confidence tuning knob if capture-thread-vs-NAPI cache locality
ever matters more than the coarser NUMA-node effect below).

### Recommended layout

Because the buffer pipeline is a **single shared ring + shared pinned `d_samples` per
`ProcessorState`** (every FPGA contributes to every correlation buffer — see
`copy_data_to_input_buffer_if_able`), it can't be sharded per NUMA node without a real
architectural change (two rings + a merge step). Given that constraint, the highest-value
placement is:

1. **Ring + pinned buffers + reassembly threads (process_packets dispatcher, workers,
   pipeline_feeder) → node 0**, matching the GPU. This is the one that matters most:
   `cudaMemcpyAsync` DMA crossing sockets is markedly worse than a CPU crossing UPI for
   ordinary memory access (DMA engines hide the latency far less well), and it would happen
   on **every** completed buffer — a sustained cost, not a one-off. `copy_nt` (ring →
   pinned buffer) also gets to be a same-socket copy this way.
2. **Each capture thread → the NUMA node of the NIC it's reading**, via the existing
   `SPATIAL_CAPTURE_CPUS` mechanism (`KernelSocketPacketCapture::get_packets`, one CPU id per
   thread in `-i` order) — so NIC0's thread stays on node 0, NIC2/NIC3's threads stay on
   node 1. Their `recvmmsg` copy (kernel skb → our ring slot) is then a node1-local *read*;
   only the *destination write* into the node0-resident ring crosses, which is unavoidable
   for those two NICs no matter which node the ring lives on (the alternative — ring on
   node1 — would instead make the GPU DMA cross, which is worse, as per point 1).

Net: **one unavoidable cross-socket hop per byte from NIC2/NIC3** (there's no layout that
avoids it while keeping a single shared ring/buffer set), placed at the cheapest point in the
pipeline (a CPU-mediated copy) rather than the most expensive one (GPU DMA).

Concrete recommendation for this box, capturing all 3 live NICs in canonical FPGA order
(`-i enp94s0np0,enp175s0np0,enp216s0np0` for FPGA0,2,3 — or include `enp134s0np0` once FPGA1
is connected):

```bash
export SPATIAL_CAPTURE_CPUS="2,12,14"   # one entry per -i interface, in that order
```

(`2` = a node0 core away from NIC0's own IRQ cores 0–4; `12,14` = node1 cores away from
NIC2/NIC3's shared IRQ cores 10–14 as far as the node allows. This is a starting point, not
a validated-under-live-traffic number — worth confirming with real packet rates.)

### What changed (`include/spatial/common.hpp`)

Added `gpu_numa_node()` (reads the active CUDA device's PCI bus id via
`cudaDeviceGetPCIBusId`, looks up its sysfs `numa_node`), `nic_numa_node(ifname)` (same sysfs
lookup for `/sys/class/net/<ifname>/device/numa_node`), and
`pin_current_thread_to_numa_node(node)` (parses the node's sysfs cpulist,
`sched_setaffinity`s the calling thread to it — the same logic as bench_processor's helper,
generalized; -1 or an unparseable/absent cpulist is a safe no-op).

Wired into `parse_common_args()`, after the NIC→FPGA-id mapping is built: pins the calling
(main) thread to the GPU's NUMA node — Linux `clone()` semantics mean every `std::thread`
spawned afterwards by main **inherits that mask unless it explicitly overrides its own
affinity**, so `ProcessorState`'s constructor (called from main right after) first-touches the
ring/pinned buffers node-local, and `process_packets`/`pipeline_feeder`/the worker threads
(which never touch their own affinity) stay confined there — while capture threads, which
already call `pthread_setaffinity_np` on themselves via `SPATIAL_CAPTURE_CPUS`, are
unaffected regardless of what they inherited. Also logs one `WARN_LOG` per live-capture NIC
whose NUMA node doesn't match the GPU's, if `SPATIAL_CAPTURE_CPUS` isn't set — a
run-time nudge with the exact node number, not a guess.

Bench-only files (`bench_processor.cu`, `bench_capture.cu`, `bench_writers.cu`,
`bench_gpu.cu`, `gpu_benchmark.cu`) were left untouched — they don't call
`parse_common_args`, and (aside from `bench_processor`, already NUMA-pinned in round 1) have
no real NIC to reason about the way this section does; worth a separate pass if useful.

### Verification

- New helper logic validated with a standalone harness compiled outside the CMake tree
  (`nvcc -x cu`, linked only against `cudart`) that calls all three functions directly:
  `gpu_numa_node() == 0`, `nic_numa_node("enp94s0np0") == 0`,
  `nic_numa_node("enp134s0np0"/"enp175s0np0"/"enp216s0np0") == 1`,
  `nic_numa_node("no_such_nic0") == -1`, `pin_current_thread_to_numa_node(0)` leaves
  `sched_getaffinity` reporting exactly `{0..9}`, and `pin_current_thread_to_numa_node(-1)`
  is a safe no-op — every result matches the real sysfs topology above.
- All six apps that call `parse_common_args` (`observe`, `beamformed_bandpass`,
  `adaptive_beamformed_bandpass`, `fft_antenna_spectra`, `get_projection_matrix`,
  `pulsar_fold`) rebuilt cleanly against the change, no new warnings or errors.
- No test includes `common.hpp` (`grep -rl` came up empty), so the existing suite
  (`ProcessorTests`/`PipelineTests`/etc.) is untouched by construction — not re-run for this
  change.
- **Not** run end-to-end against live capture: that needs root/raw-socket permissions and
  would bind real sockets on NICs that may carry live FPGA traffic on this box — deliberately
  out of scope for an unattended validation pass.

## Reproduce

```bash
cd build && export CUDA_HOME=/data/BLACKMESA_1/sma075/miniconda3   # or your CUDA root
cmake --build . --target bench_processor -- -j$(nproc)
./apps/bench_processor --duration 3          # pinning is automatic
SPATIAL_BENCH_NODE=-1 ./apps/bench_processor --duration 3   # disable pinning (old behaviour)
```
