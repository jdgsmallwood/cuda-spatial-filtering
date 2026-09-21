# Kernel capture and pre-GPU throughput plan

Reviewed 2026-09-20 against working tree at `266c536`, including existing local edits.
Scope: kernel UDP capture, packet ownership, CPU reassembly, and H2D handoff.
Ibverbs is excluded unless measured kernel limits justify reconsidering it.
Receive batch capacity stays fixed, as requested; no adaptive receive-batch policy.
This is an investigation and implementation plan; no production code was changed.

## Target and current evidence

- Minimum: 32 channels × 4 FPGAs × 15,500 = **1,984,000 packets/s**.
- At the requested 2,660 bytes: **5.27744 GB/s / 42.21952 Gb/s**, excluding any additional framing overhead.
- Assuming four equally loaded NICs: **496,000 packets/s per NIC**.
- Use packets/s as the comparison metric: the synthetic format is 2,622 UDP payload bytes (22 header + 40 scales + 2,560 samples), or 2,664 Ethernet-frame bytes excluding FCS. Confirm actual packet lengths before comparing GB/s.
- At 256 packets per correlation window, the required completion rate is **60.546875 windows/s**, one every **16.516 ms**. A window has 32,768 unique packets; boundary overlap is additional storage/work, not additional unique input throughput.
- The 6,144-slot ring represents only **3.10 ms** at target rate, before accounting for empty reservations, deferred packets, or uneven producer ownership.

Saved raw evidence: `benchmarking/20260708_115002/bench_processor.txt`, 32ch/4fpga, 30 seconds:
**5,783,963 packets/s (2.915× target)**, 15.408478 GB/s, missing=0, discarded=16,458,
fed=173,540,862 and processed=173,523,070. This is historical synthetic reassembly capacity,
not a new result and not proof of lossless capture. It has shutdown residue/discards and
no evidence in the log that H2D was enabled. Do not equate processed with uniquely delivered packets.

`scripts/profiling/BENCH_PROCESSOR_TUNING.md` reports historical improvements from ~3.2 to
~14.7 GB/s, including NUMA placement, producer changes and production fixes. Those changes
are largely already present; that historical 4.6× is not an available new speedup.
Its copy-ablation experiment increased ~8.5 to ~9.3 GB/s: only ~1.09× when removing the
reassembly deposit. This argues against predicting multiples from deleting that copy alone.
It is a different configuration, so remeasure before extrapolating.

Fresh execution is blocked here: `build/apps/bench_processor --duration 0.1`, invoked from
`/tmp`, exits 127 because `libmathdx-helper.so` is unavailable to the loader. `nvidia-smi`
cannot communicate with the NVIDIA driver. `/proc/net/dev` exposes only loopback even though
sysfs lists physical NICs. CPU topology is visible: two 10-core Xeon Silver 4210 sockets,
separate NUMA nodes and L3 caches. No fresh end-to-end speedup is claimed.

## Actual copy and ownership path

```text
NIC -> kernel receive buffers
    -> recvmmsg directly into application packet ring       [kernel-to-user copy]
    -> parse + placement + samples/scales in pinned buffers [CPU reassembly copy]
    -> asynchronous H2D                                    [DMA]
    -> GPU pipeline
```

`KernelSocketPacketCapture::get_packets` already eliminates a separate staging-to-ring
copy by pointing iovecs at ring slots. `SPATIAL_CAPTURE_STAGE_LOCAL` deliberately adds
staging for a NUMA experiment; it is normally off. Even the value `0` enables it because
the code tests variable presence. Unset it for the direct-receive baseline.
The staging path currently copies slot capacity rather than returned packet length.

Reassembly uses non-temporal sample stores, a 40-byte scale copy, and arrival bookkeeping.
Deleting this copy would avoid roughly **10.32 GB/s of logical read+write traffic** at target
rate, before overlap/cache effects. Logical traffic is not necessarily measured DRAM traffic.
Receiving directly into final ordered buffers is not a straightforward socket change:
the destination depends on headers that have not arrived yet. Guessing sequence/channel
order is unsafe with loss and reordering. A pinned receive pool plus GPU gather could remove
CPU reassembly copying, but changes GPU ingest and extends receive-buffer lifetimes.

## Ranked changes

### 1. Publish useful completed batches, and stop dispatching empty slots

Highest-priority kernel-path experiment. In `spatial.hpp`, receive capacity is fixed at 256
(the adaptive-batching comment is stale). Keep fixed capacity. Every short receive abandons
the remaining slots; the processor still traverses their published range. At 16 returned
packets this is **16× slot traversal**, at 64 it is 4×, compared with useful packets.
These are exact work-amplification factors, not measured throughput gains.

Reservation also publishes `per_thread_claim` before `recvmmsg` fills the slots. The
dispatcher handles producers sequentially and waits for every worker; workers spin on
`committed`. A stalled reservation can consequently delay already-ready work from other NICs.

First measure returned batch histograms, reserved/filled/abandoned counts and committed-spin
time. Then separate private reservation from publicly ready work. Publish descriptors for
filled batches only, recycle unused slots privately, and dispatch only ready batches.
Reservation must return available capacity without waiting to assemble a full batch; publish
partial progress safely. Preserve ownership of slots retained by the future queue.
Use a fixed 256-entry receive array and publish the actual returned count; constant offered
load does not guarantee each nonblocking call returns 256 packets. Drain nonblocking receives
while busy and poll when idle; measure CPU use as well as syscall reduction. If measurements
show nearly all calls already return full batches, empty-slot elimination has little upside;
prioritize ready-work publication, affinity and worker synchronization instead.

### 2. NUMA placement and persistent per-NIC processing lanes

First sweep affinity and worker counts with the existing design. Production caches inspected
specify three helper workers; the synthetic benchmark hardcodes six helpers and three producers.
Those results do not transfer automatically to four live NICs plus IRQ/NAPI and GPU feeder work.
Measure actual NIC/GPU topology and memory placement; historical topology is not current proof.
Keep IRQ/NAPI capacity available and avoid oversubscribing spin-waiting threads.

If dispatcher/barrier time remains substantial, replace the strided shared ring and repeated
fork/join with contiguous per-NIC receive rings and persistent consumers. Receive and parse on
NIC-local memory; write assembled payload to GPU-local pinned buffers. Publish compact completion
metadata across lanes. Assign exclusive stream ownership where possible to reduce shared atomics.
This is the largest architectural opportunity for a multiple-fold gain **if synchronization
or NUMA is the measured limiter**. It is not a promised 2× or 4× result.

Completion must wait for all relevant lane copies and boundary duplicates before recycling
buffers. A watermark alone is insufficient while writers remain active. Account explicitly
for missing/stalled streams, future-queue retention, duplicates and shutdown. Current worker
barriers provide some of this ordering and cannot simply be deleted.

More receive sockets alone do not guarantee scaling of one FPGA flow: RSS uses flow hashes.
Inspect per-queue traffic before adding threads or changing steering. See the
[Linux networking scaling documentation](https://docs.kernel.org/networking/scaling.html).

### 3. Amortize the remaining per-packet work

`copy_nt` fences after every sample copy. Benchmark one fence per completed worker batch,
with publication/reuse delayed until after that fence. Do not merely remove `_mm_sfence`:
the current release of a packet slot relies on its writes being visible.
This also needs a safe path for future-queue drains and single-threaded processing.

Consider a direct calculation of the candidate buffer plus boundary neighbours instead of
scanning all input windows, and per-stream local counters/watermarks flushed per batch.
Preserve delayed FPGA arithmetic and out-of-order handling exactly. Rank these only after
profiling; memory-copy bandwidth alone is not currently demonstrated to be the limiting factor.

### 4. Remove CPU reassembly copying only if evidence warrants it

Prototype a pinned packet pool, batched H2D and GPU gather only if the previous phases leave
the CPU deposit demonstrably dominant. Include descriptor transfer, GPU gather cost, packet-pool
retention and interference with the real GPU workload in the measurement. This is a GPU-ingest
interface change and a later option, not the first kernel-path optimization.

## Proof plan and acceptance gates

1. **Trustworthy accounting.** Instrument real receive batch sizes, ring occupancy/wait,
   worker useful/spin time, ready-buffer queue depth, and H2D completion latency. Fix drop
   instrumentation before declaring zero loss: ancillary storage is attached to message
   `reserved-1`, but inspected at `ret_val-1`. Short batches therefore usually inspect a
   message without control storage. Provide control storage for every possible returned
   message and track cumulative counter deltas/wrap. Check truncation flags as well.
   Correlate socket loss with NIC counters and sender sequence numbers.
   [recvmmsg semantics](https://www.man7.org/linux/man-pages/man2/recvmmsg.2.html)
   and [receive ancillary data](https://man7.org/linux/man-pages/man2/recv.2.html).
2. **Three measured boundaries.** Run capture-only, actual capture + production reassembler
   + null sink, and capture + reassembler + real samples/scales H2D with completion-gated
   recycling. Finally enable the real GPU workload to expose buffer-release backpressure.
   `bench_capture` currently has a recycling dummy ring and no consumer, so cannot establish
   integrated throughput. `bench_processor` bypasses sockets, uses bounded-skew template
   producers, and its optional H2D copies samples only. Extend coverage before treating either
   result as the requested proof.
3. **Representative input.** Use an external generator or actual FPGAs, four independently
   identified streams of 32 channels, realistic interleaving/skew and exact packet lengths.
   Existing synthetic sender/processor traffic needs checking against live ordering and delay
   semantics. Loopback is useful for mechanics, not physical NIC or NUMA capacity proof.
4. **Controlled A/B.** Record commit plus working-tree diff, build flags, CPU affinity,
   topology, receive-buffer settings and load. Change one feature at a time, warm up, run at
   least five 30-second repetitions in alternating order, and report median/range plus CPU cost.
   Find sustainable rates at 1.984, 2.48, 2.976 and 3.968 Mpps (1×, 1.25×, 1.5×, 2× target).
   Validate that the generator actually sustains each offered rate.
5. **Correctness and sustained acceptance.** Verify deterministic sample/scales contents at
   the GPU handoff, not just packet counts; test reordering, duplicates, skew, loss, idle NICs,
   ring wrap and shutdown. Count unique expected packets separately from overlap copies,
   rejected packets and intentional loss injection. Require zero unexplained loss/corruption
   over a ten-minute target-rate run and bounded queues. Prefer at least 25% sustained headroom.
   Claim a multiple-fold speedup only from equivalent integrated A/B runs passing these checks.

Recommendation: instrument and repair the batch handoff first, then tune placement and test
persistent lanes if needed. Historical reassembly capacity already exceeds the target by nearly
3×; the unproven part is carrying real kernel traffic through that reassembler and the GPU handoff.
