#!/usr/bin/env python3
"""GPU headroom analysis from observe's output_timings.csv.

observe records a cudaEvent pair around each execute_pipeline run (the last 100
runs, ring-buffered) and dumps them to output_timings.csv on shutdown. This
script turns those per-buffer GPU times into a packets-per-second capacity and
compares it against target ingest rates.

A buffer covers NR_CHANNELS x NR_PACKETS_FOR_CORRELATION x NR_FPGA_SOURCES
packets (e.g. 8 x 256 x 4 = 8192 with 4 FPGAs), so required buffer rate =
target_pps / packets_per_buffer, where target_pps is the AGGREGATE rate across
all FPGA sources.

Capacity is reported two ways:
  serial     1000 / mean_ms          (no stream overlap)
  overlapped num_buffers x serial    (perfect overlap across the per-buffer
                                      streams -- upper bound; check the nsys
                                      timeline for the real overlap factor)

Usage:
  ./analyze_output_timings.py output_timings.csv \
      --channels 8 --packets-for-correlation 256 --fpga-sources 4 \
      --num-buffers 3 --target-pps 2e6 5e6
"""

import argparse
import csv
import statistics
import sys


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv_file", help="output_timings.csv written by observe")
    p.add_argument("--channels", type=int, default=8)
    p.add_argument("--packets-for-correlation", type=int, default=256)
    p.add_argument("--fpga-sources", type=int, default=4)
    p.add_argument("--num-buffers", type=int, default=3,
                   help="NR_OBSERVING_BUFFERS (pipeline streams)")
    p.add_argument("--target-pps", type=float, nargs="+", default=[2e6, 5e6],
                   help="aggregate packet rate across all FPGA sources")
    args = p.parse_args()

    times_ms = []
    with open(args.csv_file) as f:
        for row in csv.DictReader(f):
            t = float(row["time"])
            if t > 0:
                times_ms.append(t)

    if len(times_ms) < 5:
        print(f"only {len(times_ms)} nonzero samples in {args.csv_file}; "
              "run observe longer (it keeps the last 100 runs)", file=sys.stderr)
        return 1

    times_ms.sort()
    n = len(times_ms)
    pct = lambda q: times_ms[min(n - 1, int(q * n))]
    mean = statistics.fmean(times_ms)

    pkts_per_buffer = (args.channels * args.packets_for_correlation
                       * args.fpga_sources)

    print(f"samples: {n}   packets/buffer: {pkts_per_buffer}")
    print(f"GPU ms/buffer  mean={mean:.3f}  p50={pct(0.50):.3f}  "
          f"p90={pct(0.90):.3f}  p99={pct(0.99):.3f}  max={times_ms[-1]:.3f}")

    serial_bps = 1000.0 / mean
    overlap_bps = serial_bps * args.num_buffers
    print(f"buffer capacity: serial={serial_bps:.1f}/s  "
          f"overlapped(x{args.num_buffers} upper bound)={overlap_bps:.1f}/s")
    print(f"packet capacity: serial={serial_bps * pkts_per_buffer / 1e6:.2f} Mpps  "
          f"overlapped={overlap_bps * pkts_per_buffer / 1e6:.2f} Mpps")
    print()

    for target in args.target_pps:
        need = target / pkts_per_buffer
        print(f"target {target / 1e6:.1f} Mpps -> {need:.1f} buffers/s "
              f"({1000.0 / need:.3f} ms budget/buffer)")
        for label, cap in (("serial", serial_bps), ("overlapped", overlap_bps)):
            margin = cap / need
            verdict = "OK" if margin >= 1.2 else ("MARGINAL" if margin >= 1.0
                                                  else "TOO SLOW")
            print(f"  {label:>10}: {margin * 100:.0f}% of required -> {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
