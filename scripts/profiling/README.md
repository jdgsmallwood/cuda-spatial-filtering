# observe.cu profiling toolkit

Companion to [ANALYSIS.md](ANALYSIS.md) (signal-chain analysis + ranked
recommendations for reaching 2–5 Mpps). Everything here must run on the Linux
observing/GPU machine, not a laptop. Remember
`export CUDA_HOME=/opt/conda/targets/x86_64-linux` (or equivalent) before
running anything that constructs the TCC correlator.

| Tool | What it measures | Which ANALYSIS.md findings it tests |
|---|---|---|
| `fast_udp_sender.c` | Rate-controlled LAMBDA-format load generator (sendmmsg, multi-thread). `apps/udp_sender` caps out below 0.1 Mpps and can't exercise the target range. | prerequisite for everything |
| `01_ingest_sweep.sh` | Max sustainable ingest rate; splits losses into kernel-socket drops (`RcvbufErrors`) vs app-side drops (observe's `Missing`/`Discarded` stats). | #1 socket ceiling, #2–#4 CPU path |
| `02_nsys_gpu.sh` | GPU timeline + kernel/memcpy summaries: per-buffer time breakdown, stream overlap, the periodic `dump_visibilities` full-device stall. | #5 device syncs, #7 eigen cost, #8 launch count |
| `03_ncu_kernels.sh` | Per-kernel speed-of-light for the hand-written elementwise kernels and permutations. | #6 8-block grid caps |
| `04_cpu_hotspots.sh` | perf stat/record + mpstat on a running observe: softirq share, memcpy cycles, LLC/dTLB miss rates, futex/cv overhead. | #1 softirq, #2 copies, #3 ring footprint, #10 barrier |
| `analyze_output_timings.py` | Converts observe's own `output_timings.csv` (per-buffer GPU event timings) into Mpps capacity vs target with margin verdicts. | overall GPU budget |

## Typical session

```bash
# 0. build the sender (once)
gcc -O2 -o fast_udp_sender fast_udp_sender.c -lpthread

# 1. GPU headroom from an existing run's CSV (cheapest first number)
#    --target-pps is the aggregate rate across all FPGA sources
./analyze_output_timings.py /path/to/output_timings.csv \
    --channels 8 --packets-for-correlation 256 --fpga-sources 4 \
    --num-buffers 3 --target-pps 2e6 5e6

# 2. ingest ceiling (observe running in live mode in another terminal)
#    one sender process == one FPGA source/socket, so sweep PER-SOCKET rates:
#    2-5 Mpps aggregate over 4 FPGAs = 0.5-1.25 Mpps per socket
./01_ingest_sweep.sh 127.0.0.1:36001 250000 500000 750000 1000000 1250000 1500000

# 3. while observe is under load from step 2:
./04_cpu_hotspots.sh           # CPU side
./02_nsys_gpu.sh ./build/apps/observe ...   # GPU side (separate run, PCAP loop is fine)
./03_ncu_kernels.sh ./build/apps/observe ... # per-kernel detail (slows execution a lot)
```

## Measuring an optimization

Re-run the same commands before/after a change and compare:

- ingest changes (#1–#4): the `loss%` column of `01_ingest_sweep.sh` at each rate,
  plus `RcvbufErrors`; and cycles/packet from `04_cpu_hotspots.sh`
  (`cycles / packets_processed` over the same window).
- GPU changes (#5–#8): mean/p99 ms in `analyze_output_timings.py` (observe already
  emits the CSV every run — no instrumentation needed), and the kernel summary
  from `02_nsys_gpu.sh`.

## Multi-FPGA note

observe is built with `OVERWRITE_FPGA_ID_WITH_IP_THIRD_OCTET=true`, so the FPGA id
comes from the sender's source-IP third octet. To emulate N FPGAs locally, add
aliases (e.g. `ip addr add 10.0.1.1/24 dev lo` … `10.0.N.1`) and run one
`fast_udp_sender --src-ip 10.0.<id>.1` per FPGA.
