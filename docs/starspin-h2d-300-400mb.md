# H2D tuning for 300–400 MB transfers

Completed 2026-10-01 on BLACKMESA, NVIDIA RTX 4000 Ada. Sizes and rates below use decimal MB and GB/s.

## Result

Best confirmed configuration: GPU-local NUMA memory, 2 MiB-aligned and rounded allocations, transparent huge pages first-touched before `cudaHostRegister`, and one whole-buffer `cudaMemcpyAsync` per transfer. No tested configuration reached 13 GB/s.

| Transfer | Median H2D | Effective time per transfer |
|---|---:|---:|
| 300 MB | 12.337 GB/s | 24.32 ms |
| 350 MB | 12.338 GB/s | 28.37 ms |
| 400 MB | 12.337 GB/s | 32.42 ms |

Ordinary `cudaHostAlloc`, `cudaHostAllocWriteCombined`, and `cudaHostRegister` with huge pages disabled all measured approximately **8.01 GB/s** at each size. Registered huge pages improved throughput by about **54%**.

## Other options tested at 350 MB

| Configuration | H2D GB/s |
|---|---:|
| 1 stream / 1 buffer, exact allocation | 12.235 |
| 1 stream / 3 buffers | 12.234 |
| 2 streams / 2 buffers | 12.234 |
| 3 streams / 3 buffers | 12.235 |
| 8 streams / 8 buffers | 12.235 |
| 1 MiB chunks | 11.800 |
| 4 MiB chunks | 12.090 |
| 16 MiB chunks | 12.198 |
| 64 MiB chunks | 12.226 |
| Allocation rounded to 2 MiB, same tuning round | 12.270 |
| GPU kernel copying mapped host memory, 32–512 blocks | 12.05–12.13 |
| Concurrent copy engine + mapped-host kernel, 25–90% DMA share | 11.26–12.20 |

Whole-page rounding was a small improvement (~0.3%) in the directly comparable tuning round. Later rounded runs ranged from 12.29 to 12.34 GB/s; do not attribute all between-round variation to rounding. More streams/buffers did not increase isolated H2D bandwidth. Keep pipeline concurrency for compute/transfer overlap; this test does not measure that benefit.

## Method and limits

- 86 focused runs: 36 allocation comparisons, 26 stream/chunk/kernel trials, 15 hybrid trials, and 9 final confirmations. Randomized case order within repetitions.
- CPU 7 and memory node 0, local to the GPU; telemetry/controller on CPU 18. Each timed pass transfers at least 16 GB after an equally long untimed warmup. Allocation, first touch, registration, and validation are outside timing.
- Timing includes copy submission and final device synchronization. Reported milliseconds are bytes divided by sustained throughput, not measured per-transfer latency percentiles.
- CUDA host-pointer type/flags, Linux smaps huge-page backing, and numa_maps node placement checked. Final rounded allocations were 100% backed by huge pages in the overlapping VMAs.
- Every run passed transfer spot checks. The final 24 hybrid/confirmation runs also verified every destination byte outside the timed interval.
- Every PCIe sample matched to a timed interval reported Gen 3 x16. No clock, power, PCIe configuration, or FPGA settings changed.
- These are H2D-only results. CPU input construction, the correlator, D2H output, and live ingest contention are not included.
- Experimental kernel/hybrid paths remain benchmark options. Production allocation and copy code were not changed by this retest; production already defaults to registered huge-page input buffers.
- Extended PCIe capabilities could not be read without administrator access (`lspci`: capabilities access denied; noninteractive sudo requires a password). PCIe request/payload-size limits therefore remain unverified.

## Reproduce the preferred transfer

```bash
cmake --build build --target bench_starspin_h2d -j 4
numactl --all --physcpubind=7 --membind=0 \
  build/apps/bench_starspin_h2d bytes:350000000 100 combined 3 register 3 h2d 0 0 rounded
```

Use `bytes:300000000` or `bytes:400000000` for the endpoints. The last arguments are chunk bytes (`0` = whole), mapped-copy kernel blocks (`0` = copy engine), allocation rounding, and optional DMA prefix percent for hybrid copies. Streams and buffers support 1–8.

Allocation controls: replace `register` with `default`, `wc`, or `register_no_thp`; omit `rounded` for nonregistered modes. `wc` is a separate CUDA allocation mode, not a flag to pass to `cudaHostRegister`.

## Artifacts

Raw logs, per-run commands, JSON summaries, PCIe telemetry, and the sweep driver are in `build/benchmarks/h2d_memory_retest_20260930/`. Focused subdirectories:

- `focused_300_400MB/`
- `focused_tuning_350MB/`
- `focused_hybrid_350MB/`
- `focused_confirm_300_400MB/`

The top-level older sweep includes other sizes measured before the task was narrowed. The conclusions above use the focused 300–400 MB runs.
