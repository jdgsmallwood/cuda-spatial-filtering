#!/usr/bin/env python3
"""Summarize the in-memory Starspin ingest trace exported after shutdown."""
import argparse
import bisect
import csv
from collections import defaultdict
from pathlib import Path

NAMES = {1: "completion gate", 2: "worker pause", 3: "slow worker batch",
         4: "receive ring wait", 5: "receive progress gap",
         6: "slow receive batch", 7: "pipeline submission"}


def summarize(path):
    with path.open() as source:
        rows = [{k: int(v) for k, v in row.items()} for row in
                csv.DictReader(line for line in source if not line.startswith("#"))]
    if not rows:
        print(f"{path}: no recorded events")
        return
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["event"]].append(row)
    origin = min(row["start_ns"] for row in rows)
    print(path)
    print("Durations are wall time; overlapping thread intervals must not be added as elapsed time.")
    print("Slow worker/receive events are thresholded; their percentiles exclude short events.")
    for event, items in sorted(grouped.items()):
        us = sorted((row["end_ns"] - row["start_ns"]) / 1000 for row in items)
        quant = lambda fraction: us[int(fraction * (len(us) - 1))]
        print(f"{NAMES[event]:24s} count={len(us):8d} p50_us={quant(.5):9.1f} "
              f"p99_us={quant(.99):9.1f} max_us={us[-1]:9.1f}")

    gates = sorted(grouped[1], key=lambda row: row["start_ns"])
    starts = [row["start_ns"] for row in gates]
    print("Longest completion gates: relative_start_s, cpu, wall_ms, lock_wait_ms, hold_ms, thread_cpu_ms, missing_delta")
    for row in sorted(gates, key=lambda r: r["end_ns"] - r["start_ns"], reverse=True)[:12]:
        print(f"{(row['start_ns']-origin)/1e9:.6f}, {row['cpu']}, "
              f"{(row['end_ns']-row['start_ns'])/1e6:.3f}, "
              f"{(row['arg0']-row['start_ns'])/1e6:.3f}, "
              f"{(row['end_ns']-row['arg0'])/1e6:.3f}, "
              f"{row['arg2']/1e6:.3f}, {row['arg1']}")

    print("Longest receive ring waits: relative_start_s, receiver, duration_ms, overlap_with_completion_gates_ms")
    for row in sorted(grouped[4], key=lambda r: r["end_ns"]-r["start_ns"], reverse=True)[:12]:
        overlap = 0
        index = max(0, bisect.bisect_right(starts, row["start_ns"]) - 1)
        while index < len(gates) and gates[index]["start_ns"] < row["end_ns"]:
            gate = gates[index]
            overlap += max(0, min(gate["end_ns"], row["end_ns"]) -
                           max(gate["start_ns"], row["start_ns"]))
            index += 1
        print(f"{(row['start_ns']-origin)/1e9:.6f}, {row['lane']-5}, "
              f"{(row['end_ns']-row['start_ns'])/1e6:.3f}, {overlap/1e6:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path)
    for trace in parser.parse_args().traces:
        summarize(trace)
