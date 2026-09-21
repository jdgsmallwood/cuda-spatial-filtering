# How Starweave went from 24 to 40 channels

This is the plain-language explanation of the four-FPGA packet path and the changes that made a
stable 40-channel Starweave capture possible. It focuses on the path before the GPU, including
ibverbs receive ownership, ring reservations/commits, and the coordinator/worker handoff.

The short version is: 40 channels did not come from one magic optimization. It came from making
packet ownership explicit, removing an entire CPU copy, preventing four NIC threads from
blocking one another, giving the coordinator useful work instead of making it wait, pinning the
hot threads sensibly, and making buffer release asynchronous and reliable.

## The target

Each of four FPGAs sends one packet per selected coarse channel at approximately 15,500 packets
per second per channel. Therefore:

| Channels | Nominal packet rate | Payload bandwidth at 2,660 B/packet |
|---:|---:|---:|
| 24 | 1.488 Mpps | 3.96 GB/s |
| 32 | 1.984 Mpps | 5.28 GB/s |
| 40 | 2.480 Mpps | 6.60 GB/s |

The live backend during the successful 40-channel tests offered about 2.314 Mpps, or 6.154 GB/s.
Starweave processed that rate continuously. The difference from 2.480 Mpps was in offered/live
traffic, not a growing application backlog.

## The packet's journey now

```text
four FPGA/NIC streams
        |
        v
one RAW_PACKET ibverbs QP and capture thread per NIC
        |
        | NIC DMA, directly into that thread's reserved ring slots
        v
6,144-slot shared packet ring, split into four disjoint strided lanes
        |
        | coordinator dispatches fixed batches
        v
coordinator + three persistent workers parse and place packets
        |
        v
eight pinned host assembly buffers + arrival bitmaps
        |
        | completed-buffer SPSC queue
        v
GPU pipeline feeder -> asynchronous H2D -> GPU processing
        |
        | one CUDA event per host buffer
        v
event-polling reclaimer returns the host buffer to assembly
```

There are two different kinds of buffer in that diagram:

- The packet ring contains individual network packets waiting to be parsed.
- The eight assembly buffers contain packets reorganized by coarse channel, time, FPGA/receiver,
  and polarization for the GPU.

Keeping those ownership domains separate is important. A packet-ring slot can be released after
its data has been placed into an assembly buffer. An assembly buffer cannot be reused until its
asynchronous H2D read has finished.

## 1. Why ibverbs helped

The kernel socket path already used `recvmmsg`, but packets still travelled through the kernel's
normal UDP receive machinery. Under the live load, packet loss could occur in the NIC/kernel path
before the application had a chance to process anything. Changing the coordinator/worker handoff
cannot recover a packet that never reached userspace.

The ibverbs backend creates one RAW_PACKET queue pair (QP) per NIC and steers the desired UDP flow
to it. Each QP has 1,024 posted receive work requests (WRs), so the NIC normally has many places
available for its next DMA. A capture thread busy-polls each completion queue (CQ).

The original ibverbs implementation received into a registered staging allocation and then copied
each packet into the application's ring. The direct-to-ring path registers the ring itself and
posts its slots as the receive destinations. This removes the staging-to-ring `memcpy` for every
packet—about 6.15 GB/s of copied bytes at the measured 40-channel rate, plus the corresponding
read traffic.

This is CPU-memory zero-copy, not GPUDirect. The NIC writes directly into the CPU packet ring; the
workers still reorganize packet payloads into pinned assembly buffers, and CUDA still performs an
H2D transfer from those assembly buffers.

### Raw headers had to land separately

A RAW_PACKET QP receives Ethernet, IP, and UDP headers as well as the application payload. The
existing packet parser expects the ring slot to begin with the custom FPGA header, just like a UDP
socket payload. Posting the whole raw frame at `slot->data[0]` shifted every parsed field and
corrupted interpretation.

Each posted WR therefore has two scatter/gather entries:

1. Ethernet/IP/UDP bytes go to a small per-WR header area.
2. The UDP payload goes directly to the packet-ring slot.

The header area must be per WR, not one shared discard buffer, because several DMA completions can
arrive together. The source IP is read from that WR's header area to recover the FPGA identity.

### The CQ was enlarged

There are 1,024 posted receives per QP, but the CQ is four times that depth. This protects startup
against completion bursts. A CQ overrun can put mlx5 into an error state and look like a silent,
fixed-count freeze rather than a clean packet-loss report.

## 2. Reservation, commit, processing, and release

The ring uses a two-phase producer protocol because reserving memory for an upcoming NIC DMA does
not mean a complete packet exists there yet.

Each slot has two important state flags:

- `processed == true`: the slot is free and a capture producer may reserve it.
- `committed == true`: DMA and metadata publication are complete and a worker may read it.

The normal lifecycle is:

```text
FREE
  processed=true
        |
        | reserve
        v
NIC-OWNED / NOT YET READABLE
  committed=false
        |
        | NIC DMA completes; capture thread writes length/source metadata
        | and performs committed.store(true, release)
        v
COMMITTED / WORKER-READABLE
  processed=false, committed=true
        |
        | worker performs committed.load(acquire), parses and places packet
        v
FREE AGAIN
  processed=true
```

The release/acquire pair is the memory-ordering contract: if a worker observes
`committed=true`, it must also see the NIC-written payload and the capture thread's length/source
metadata.

If a reserved receive fails or is unused during shutdown, `abandon_write_batch()` publishes it as
an empty committed slot. That matters because a consumer waiting forever on `committed=false`
would stop the entire lane.

### Why four strided lanes are used

Four capture threads cannot safely increment one shared ring cursor at this rate without lock
contention and head-of-line blocking. Instead, capture thread `t` owns:

```text
t, t + 4, t + 8, t + 12, ...
```

For example, thread 0 owns slots 0,4,8,... and thread 1 owns 1,5,9,... . The 6,144-slot ring is
divisible by four, so these lanes never overlap. Each producer publishes its own claim watermark,
and the coordinator publishes a separate per-lane read watermark after processing. A fast NIC
therefore does not need a global producer mutex and an idle/slow NIC cannot prevent other lanes
from being consumed.

The initial cursor must start at `thread_id`, not zero. Starting all four at zero made every QP
reserve the same slots and caused an early deterministic stall.

### Claimed is not the same as committed

Ibverbs reserves many slots in advance because those addresses must be posted to the NIC before
packets arrive. Consequently, the producer's claim watermark can be far ahead of actual CQ
completions. The workers always check `committed` before reading a slot.

For the proven 40-channel path, the coordinator processes at most 256 claimed slots from one lane
per turn. Reducing the old multi-thousand-slot handoff to 256 was important: otherwise one lane
could make all workers wait on a large tail of not-yet-completed WRs while already-committed work
from the other three NICs received no attention.

A later 48-channel experiment added committed-prefix dispatch and reduced its fixed handoff to 64.
Those changes helped diagnose 48 channels but did not make it sustainable; they are not the reason
40 channels works. The <=40 path deliberately retains its proven fixed size of 256. There is no
adaptive receive batching.

### Retire a CQ batch before waiting for replacements

The current capture loop handles a polled CQ batch in two phases:

1. Commit or abandon every completion already removed from the CQ.
2. Reserve replacement ring slots and repost those WRs.

This avoids a circular wait where the capture thread blocks while reserving a replacement for an
early completion even though a later completion in the same already-polled batch is the packet the
consumer needs to advance and free space. This is a sound ownership safeguard added during the
48-channel investigation; the working 40-channel checkpoint predates it, so it should not be
presented as the measured source of the 24-to-40 speedup.

## 3. The coordinator-to-worker handoff

The coordinator walks each producer lane in round-robin order. For each fixed batch it divides the
ring slots across four processing lanes:

- the coordinator handles one slice itself;
- three persistent worker threads handle the other slices.

Previously, the coordinator mostly signalled workers and spun while they did all the copying.
Making the coordinator process a slice adds useful memory-copy bandwidth without adding another
thread or another completion mechanism.

Each worker parses its assigned slots, maps FPGA ID/channel/sample counter to an assembly-buffer
position, copies samples/scales, and sets the arrival bit. Packet counts are accumulated locally
and added atomically once per slice rather than once per packet. The code also prefetches packet
headers, metadata, and several payload cache lines ahead of use.

The coordinator must wait until every dispatched worker has finished before checking whether an
assembly buffer is complete. This is a correctness barrier, not optional overhead. Without it, one
worker's high-water mark could make a buffer appear complete while another worker was still
copying into it; real data would then be counted as Missing and zero-filled.

### Why the mailbox experiment is not the production answer

An alternative per-worker generation mailbox was built and tested to reduce contention on the
shared `num_workers_with_tasks` counter. CPU-only synthetic results could be fast, but H2D tests
showed a lifecycle/liveness problem and live 24-channel testing did not remove upstream packet
loss. There was no reproducible production advantage.

The mailbox runtime switch has therefore been removed from normal Starweave execution. The
proven default remains the simpler task flag plus shared completion count. This is a useful result:
worker notification was not the limiting factor that separated 24 from 40 channels.

## 4. Reassembly and startup alignment

Workers do not send individual packets to the GPU. They place them into one of eight large pinned
host buffers. Placement uses:

- coarse frequency channel;
- FPGA/source index;
- packet sample counter;
- configured relative FPGA delays;
- receiver and polarization layout.

An arrival bitmap records which expected packet positions were filled. When later packets prove a
window complete, absent bitmap entries are counted as `Missing` and their scales are zeroed so the
GPU sees deterministic zeros rather than stale data.

Packets slightly beyond the current assembly horizon go to a future queue and are retried after a
buffer advances. A ring slot retained by that queue cannot be reused until the packet is placed or
discarded, which is why reservation checks the individual slot's `processed` flag as well as the
lane distance.

Startup now follows one simple rule: the first valid packet initializes all four FPGA timelines
once. The delay file translates its sample counter into the corresponding initial counter for the
other three FPGAs. Runtime “re-seeding” was removed because repeatedly shifting all buffer
boundaries destroyed partially assembled data and caused non-convergent behaviour. A few early
packets may be discarded; after startup, the boundaries remain stable.

Signed sequence arithmetic was also required because one live relative FPGA delay was about
-80.8 billion samples. Unsigned subtraction wrapped that into an enormous positive counter.

## 5. Handing completed buffers to the GPU

Assembly and GPU submission are decoupled with a bounded single-producer/single-consumer queue.
The coordinator enqueues a completed assembly-buffer index and immediately continues packet work.
A dedicated pipeline feeder dequeues it and launches the GPU pipeline. This prevents GPU launch
and writer latency from sitting directly in the packet coordinator's critical path.

Starweave has eight assembly buffers. For each real submission, the pipeline records a CUDA event
after H2D has finished reading that particular host buffer. A CPU reclaimer thread polls those
events and only then calls `release_buffer()`.

This per-buffer event scheme replaced CUDA host callbacks during the 48-channel investigation.
Callbacks produced a repeatable freeze; event polling removed that deadlock. It is now the normal,
simple release mechanism for 40 channels too. It does not make 48 channels fast enough, but it
makes ownership reliable.

The fine-channelizer now also uses one shared filter object rather than compiling the same filter
once per coarse channel. It still launches once per coarse-channel slice, but startup takes about
ten seconds instead of minutes. This is mainly an operator/startup improvement, not the central
steady-state throughput gain.

## 6. Thread placement mattered

Capture threads busy-poll, so letting the scheduler place them on already-hot IRQ/NAPI cores can
cause severe stalls even when average CPU utilization looks acceptable. The proven mapping is:

```text
capture threads: 1,11,13,15
packet workers:  3,4,5
```

NIC 94 and CPU 1 are on NUMA node 0. NICs 134/175/216 and CPUs 11/13/15 are on node 1. The GPU and
most processing memory are node-local according to the application's GPU NUMA selection. These
settings remain operator-visible because safe cores are a property of the host's current IRQ and
workload placement, not a universally correct compile-time constant.

The CPUs are 20 distinct physical cores; CPU 1 and CPU 11 are not SMT siblings.

## 7. What was measured

| Configuration | Result |
|---|---|
| 4 FPGA × 24 channels, ibverbs | About 1.39 Mpps; application stayed healthy |
| 4 FPGA × 32 channels, 60 s | 111,037,786 received; Missing became flat; no future/stuck backlog |
| 4 FPGA × 32 channels, 600 s | 1,111,114,882 received; 0.002921395% Missing; no future/stuck backlog |
| 4 FPGA × 40 channels, checkpoint 60 s | 138,851,828 received; 0.036359361% Missing; no future/stuck backlog |
| 4 FPGA × 40 channels, optimized 600 s | 1,388,325,534 received; 0.044180624% Missing; no future/stuck backlog |

For the ten-minute 40-channel run, `Processed` remained within one partial 256-packet block of
`Received`, `FutureQueued` and `StuckUnprocessed` stayed zero, 33,908 GPU pipeline runs were queued,
and shutdown was clean. That proves the application had sustained capacity for the live input.

`NICDrops=0` is not trusted for ibverbs because that backend does not expose a reliable equivalent
of the kernel socket drop counter. Acceptance uses packet sequence/arrival accounting (`Missing`),
bounded queues, and clean continued pipeline progress.

## 8. What did not produce the gain

- Adaptive receive batching was neither wanted nor used. Handoff sizes are compile-time fixed.
- Worker mailboxes did not show a reliable end-to-end improvement and are no longer a runtime
  production option.
- Expanding the ring fourfold delayed the 48-channel failure but did not fix it, so 40 channels
  keeps the 6,144-slot ring.
- `-march=native`/AVX-512 improved an isolated copy benchmark but reduced whole-application
  throughput on this Xeon because sustained AVX-512 downclocked the socket. It remains off.
- More host assembly buffers cannot cure a GPU whose sustained service rate is below the input
  rate; they only postpone backpressure.

## 9. Why 48 channels still fails

At 48 channels the input requires 60.55 completed 256-packet assembly blocks per second. Direct
measurement showed the GPU pipeline completing only about 55–56 blocks per second. Pending host
buffers therefore rise until all eight are GPU-owned. Assembly then cannot advance, the future
queue and packet ring fill, receive WRs cannot be replaced, and Missing grows rapidly.

The GPU is slow rather than deadlocked: after input collapses it eventually reclaims every queued
buffer. The likely fix is to fuse several full-buffer GPU transforms before the fine-channelizer,
reducing device-memory traffic. Ring, CQ, worker-mailbox, and buffer-count changes cannot solve a
persistent downstream service-rate deficit.

## 10. Remaining 40-channel loss investigation

Forty channels is stable but not yet completely lossless. In the ten-minute run, Missing rose in
short steps separated by long flat periods. During one live interval Missing increased by 11,474
while every NIC `rx_out_of_buffer` counter remained unchanged, ruling out receive-WQE starvation
for that burst.

A compile-time diagnostic build now attributes Missing to FPGA index and records per-QP CQ and
context-switch totals. The next 120-second diagnostic run will determine whether the bursts come
from one/two FPGA paths and whether they align with capture-thread descheduling. Diagnostics are
off in production builds.

## Operator entry point

Use the repository-local launcher:

```bash
scripts/run_starweave.sh --duration 600
```

It supplies the four NICs, ibverbs backend, channels 176–215, affinity, CUDA path, delay/gain/config
files, antenna map, output naming, disk-space check, and capability check. Run
`scripts/run_starweave.sh --help` for overrides and diagnostic mode. It is based on the operator
shape of `/data/BLACKMESA_0/lambda/scripts/correlate_lambda.sh`; nothing in that external scripts
directory was changed.

For exact commands, artifacts, and the current paused diagnostic state, see:

- `docs/starweave-capture-runbook.md`
- `docs/ingest-optimization-handoff.md`
- `docs/ibverbs-zerocopy-handoff.md`
