# observe.cu performance analysis — target 2–5 Mpps

Signal chain analyzed: `KernelSocketPacketCapture::get_packets` (recvmmsg) → packet ring
(`ProcessorState`, 900k slots) → 3 worker threads (`process_packet_data` →
`copy_data_to_input_buffer_if_able`) → pinned `d_samples` input buffers (24) →
`LambdaGPUPipeline::execute_pipeline` (H2D, scale/convert, cuTensor permutations, TCC
correlator, cuSOLVER eigendecomposition, ccglib beamform, cuFFT) → `BufferedOutput`
writer threads (HDF5/Redis).

## Back-of-envelope budget

With the deployed shape (8 channels, **4 FPGA sources**, 10 receivers/packet, 64
timesteps, 256 packets/correlation):

- Packet payload = 22 B custom header + 40 B scales + 2560 B samples ≈ **2.6 KB**.
- 2 Mpps ≈ **42 Gb/s**; 5 Mpps ≈ **105 Gb/s** aggregate on the wire — but spread
  across 4 FPGA sources, each with its own NIC/receiver thread, that is
  **0.5–1.25 Mpps and ~10–26 Gb/s per socket**. Per-socket rate, not the
  aggregate, is what bounds the kernel capture path.
- One GPU buffer = 8 ch × 256 pkts × 4 FPGAs = **8192 packets** → at 2–5 Mpps the
  pipeline must complete a buffer every **4.1 ms / 1.6 ms** (×`NR_OBSERVING_BUFFERS`=3
  if the per-buffer streams genuinely overlap). `output_timings.csv` (already produced
  by observe) gives the measured per-run GPU time — see `analyze_output_timings.py`
  with `--fpga-sources 4`.
- CPU memory traffic per packet today: kernel→staging (recvmmsg), staging→ring slot
  (memcpy #1), ring→pinned d_samples (memcpy #2), then DMA. At 5 Mpps that's roughly
  3 × 13 GB/s payload × (read+write) ≈ **~75 GB/s of CPU memory bandwidth just on
  copies** — likely more than one memory controller channel can spare on a busy node.

## Findings, ordered by expected impact

### 1. Kernel UDP sockets are marginal at the top of the range (capture)

`recvmmsg` on a single socket tops out around 1–1.5 Mpps/core in practice (softirq side
of the kernel UDP stack). With the load split across 4 FPGA sockets, 2 Mpps aggregate
(0.5 Mpps/socket) should be comfortable with tuning, but **5 Mpps aggregate
(1.25 Mpps/socket) sits right at the per-core ceiling** — expect it to work only with
clean core isolation (receiver thread and its NIC IRQ on dedicated cores, same NUMA
node) and to have no headroom. Measure first (`01_ingest_sweep.sh` +
`04_cpu_hotspots.sh` shows `%soft` per core), then in increasing order of effort:

- **SO_REUSEPORT fan-out**: N sockets on the same port, one receiver thread each, with
  NIC RSS spreading flows across queues. Cheap change to `KernelSocketPacketCapture`
  (the two-phase reserve/commit ring protocol already supports concurrent producers).
  Caveat: RSS hashes on the 5-tuple — with a single FPGA source/port pair all packets
  hash to one queue, so the FPGAs may need to vary source port, or use
  `SO_ATTACH_REUSEPORT_CBPF` to fan out round-robin.
- **`busy_poll_us`** is already plumbed through (`-…--busy-poll`); it defaults to 0.
  Worth ~10–20% latency/throughput on dedicated cores.
- **Kernel bypass**: `include/spatial/libibverbs.hpp` already has an ibverbs capture
  path — wiring observe to it (or AF_XDP) is the only realistic route to a reliable
  5 Mpps × 2.6 KB ingest. This also enables receiving headers and payload to separate
  buffers (header/data split), which would remove CPU copy #1 below.

### 2. Eliminate the staging→ring memcpy (capture) — ✅ IMPLEMENTED

`get_packets` now polls for readability, reserves a batch of ring slots (adaptively
sized 8–256 from the recent drain rate), points the `recvmmsg` iovecs directly at
`slot->data`, commits the filled slots, and abandons the remainder as empty via the
new `abandon_write_batch()`. This removes one full copy of the data stream
(~26 GB/s of memory traffic at 5 Mpps). The poll-before-reserve ordering means no
ring slot is ever held while waiting for packets, so consumers never spin on
uncommitted slots.

### 3. Ring buffer geometry: 3.7 GB working set (ring) — ✅ IMPLEMENTED

`PacketEntry` was 4096 B data + metadata ≈ 4.2 KB, × `PACKET_RING_BUFFER_SIZE`=900,000 ≈
**3.7 GB**: every producer write and consumer read missed LLC and stressed the TLB.

- `PacketEntry::DATA_CAPACITY` is now derived from the config (max frame + 42 B
  headers, cache-line rounded — ~2.7 KB for the default shape) instead of the fixed
  4 KB `BUFFER_SIZE`; producers size their receives via the new
  `slot_data_capacity()` and the PCAP/ibverbs paths clamp to it.
- `PACKET_RING_BUFFER_SIZE` in observe.cu dropped 900,000 → 2^19 = 524,288 (~100 ms
  at 5 Mpps aggregate; power of two so the compile-time modulo becomes a mask).
  Total pool ≈ 1.4 GB, down from 3.7 GB.
- The pool is 2 MB-aligned with `MADV_HUGEPAGE` so it's THP-backed — verify with
  `grep AnonHugePages /proc/<pid>/smaps_rollup` and the `dTLB-load-misses` counter
  in `04_cpu_hotspots.sh`.

### 4. Worker copy into pinned memory: streaming stores + flat FPGA lookup (workers) — ✅ IMPLEMENTED

`copy_data_to_input_buffer_if_able` runs per packet on the worker threads:

- The `unordered_map` hash lookup per packet is replaced by a flat 256-entry
  `fpga_index_lut` (FPGA ids are the IP third octet); ids ≥ 256 fall back to the map.
- The sample `memcpy` into pinned `d_samples` is now `copy_nt()` — AVX2/SSE2
  streaming stores (with the required `sfence`) that skip the read-for-ownership
  and don't pollute LLC, roughly halving this copy's memory traffic. The 40 B
  scales stay plain memcpy (stride-unaligned).
- Workers no longer hard-pin to cores 0..N-1 (where NIC IRQs usually land).
  Pinning is opt-in: `SPATIAL_WORKER_CPUS="4,5,6"` (one entry per worker), and the
  worker count is a `ProcessorState` template parameter (default 3). Place
  receivers, workers, and NIC IRQs on the NUMA node closest to the NIC/GPU
  (check `cat /sys/class/net/<if>/device/numa_node`).

### 5. GPU: `dump_visibilities()` does two `cudaDeviceSynchronize()` inline (pipeline)

`execute_pipeline` calls `dump_visibilities()` every `NR_CORRELATED_BLOCKS_TO_ACCUMULATE`
(56) buffers, and that function brackets its work with **two full-device syncs**. That
stalls all three pipeline streams and the feeder thread once every ~56 buffers (~every
90–230 ms at target rates with 4 FPGAs) — a periodic hiccup the ring then has to
absorb. Replace with:
record an event on `buffers[0].stream` after the memcpy/memset, and have only the
visibilities path wait on it (or double-buffer `d_visibilities_accumulator` and flip).

### 6. GPU: elementwise kernels capped at 8–16 blocks (pipeline) — ✅ IMPLEMENTED

The launchers in `src/spatial.cu` did `num_blocks = std::min(8, n/1024+1)` (16 for
`convert_float_to_half`) — 8–16 blocks total for multi-megabyte arrays, ~10% of the
SMs for purely bandwidth-bound kernels. They now use `elementwise_grid_size(n)`:
enough blocks to cover n, capped at 32 blocks/SM (queried once). `update_weights`'s
grid also now accounts for the ×2 real/imag elements its kernel iterates.
Before/after: `03_ncu_kernels.sh` DRAM throughput on `convert_*`/`accumulate_*`.

### 7. GPU: eigendecomposition + its 3-permutation feeder chain is only used for the eigen HDF5 output (pipeline)

In `LambdaGPUPipeline`, `update_weights_kernel` **ignores the eigenvalues — it's a pure
copy of the weights**. So per buffer, the chain
`visCorrToDecomp` → `unpack_triangular_baseline_batch` → `cusolverDnXsyevBatched` (32
batched hermitian eigensolves) exists solely to feed `HDF5EigenWriter`. cuSOLVER's
batched Xsyev also has poor small-matrix throughput and may host-sync internally.
Options: gate it behind a flag, run it every Nth buffer (eigen monitoring rarely needs
~kHz cadence), or move it to a separate stream off a snapshot copy so it never blocks
the correlate/beamform critical path. Measure its share with `02_nsys_gpu.sh`.

### 8. GPU: ~25–40 launches per buffer → CUDA Graphs (pipeline) — ✅ IMPLEMENTED (partial, safe subset)

`LambdaGPUPipeline::execute_pipeline`'s two static sections are now captured into
CUDA graphs per buffer at construction (`enqueue_pre_eigen`: permutations → delays →
correlator → visibility reshaping, ~12 launches; `enqueue_post_eigen`: accumulate →
beamform GEMM → output permutations, ~9 launches) and replayed as one launch each.
Deliberately left eager: the per-run-varying ingest/output copies and host funcs,
**cuSOLVER** (may do host-side work per call that capture wouldn't replay), and
**cuFFT + downsample** (capture-compatibility caution). Capture failure at startup
falls back to eager execution with a warning; `SPATIAL_DISABLE_CUDA_GRAPH=1` forces
the eager path. Further fusion (e.g. `scale_and_convert_to_half` emitting the
prealign layout directly; collapsing the visibilities permute→trim→permute→permute
chain into one kernel) is still open — fuse only after nsys shows which passes
matter.

### 9. Correctness flag: input buffer released before H2D copy completes (pipeline)

`LambdaPipelineIngest::ingest_and_scale` enqueues the samples/scales H2D
`cudaMemcpyAsync` on `b.stream`, but enqueues `release_buffer_host_func` on
`b.host_stream` — a different stream with **no event dependency**. The host func can
run before the DMA finishes, returning the pinned buffer to `ProcessorState`, which
resets its sequence range and lets workers overwrite it while the copy is in flight.
With 24 input buffers a collision is rare (which may be why 24 works and fewer
doesn't), but it's a latent data-corruption race and it forces the large
`num_packet_buffers`. Fix: `cudaEventRecord(ev, b.stream)` after the two memcpys, then
`cudaStreamWaitEvent(b.host_stream, ev)` before the host func. After this, 24 buffers
can probably shrink to ~4–6, releasing pinned memory and cache.

### 10. Smaller items

- `state.packets_received += reserved` is a plain `uint64_t` incremented from multiple
  receiver threads concurrently — a data race (undercounts; UB strictly). Make it
  atomic with relaxed ordering.
- `apps/udp_sender.cpp` cannot generate the target load: hard-coded 127.0.0.1:12345,
  `usleep(5)` per packet (~≤60 kpps real-world, far below even 0.25 Mpps). Use
  `scripts/profiling/fast_udp_sender.c` (sendmmsg, rate-controlled, multi-thread) or
  `tcpreplay --pps` instead.
- The per-batch worker handoff in `process_packets` (cv broadcast + wait for all
  workers per ≤6000-packet batch) is a barrier: throughput is set by the slowest
  worker each round and the main thread sleeps in between. Fine at 2 Mpps
  (~1 batch/ms); if it shows up in `04_cpu_hotspots.sh` as cv/futex time, move to a
  work-stealing or per-worker-range design without a global barrier.
- `Writer::handle_buffer_full` spin-waits in the *pipeline* thread when a writer ring
  fills — at high rates a slow HDF5 flush back-pressures into `execute_pipeline`.
  Worth a drop counter + drop-oldest policy for the monitoring streams (eigen, FFT
  Redis): stale monitoring data is worthless, dropped science data is not.

## Status / remaining work

Implemented (this branch): **#2** (receive directly into ring slots), **#3** (slot
sizing, 2^19 ring, huge pages), **#4** (NT stores, flat FPGA lookup, opt-in worker
affinity via `SPATIAL_WORKER_CPUS`), **#6** (grid sizing), **#8** (CUDA graphs for
the two static sections, eager fallback + `SPATIAL_DISABLE_CUDA_GRAPH`). Also fixed
along the way: the `packets_received` data race (#10), and `reserve_write_batch` no
longer spins past shutdown when the ring is full.

Still open, in suggested order:

1. Re-run the baseline numbers to quantify the above: `01_ingest_sweep.sh`,
   `analyze_output_timings.py`, `02_nsys_gpu.sh`, `04_cpu_hotspots.sh`.
2. #9 (buffer-release race: event-link `host_stream` to the ingest copies) and
   #5 (replace `dump_visibilities`' device-wide syncs) — small diffs, removes stalls
   and probably lets `num_packet_buffers` shrink from 24.
3. #7 (gate/decimate the eigendecomposition chain — it only feeds the eigen HDF5
   output in this pipeline).
4. #1 (SO_REUSEPORT fan-out, then ibverbs/AF_XDP) — needed for headroom at the
   5 Mpps end of the range.
5. Kernel fusion guided by nsys (see #8 notes).
