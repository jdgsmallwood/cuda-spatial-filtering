#!/usr/bin/env python3
"""Compare two benchmark JSON outputs to detect regressions between PRs.

Usage:
    python3 scripts/benchmark/compare_benchmarks.py baseline.json current.json

Both JSON files should be produced by run_benchmarks.py --json-out.

Exit code: 0 if no regressions, 1 if any metric regressed beyond the threshold.
"""

import json
import sys
from pathlib import Path

# A regression is flagged when a throughput metric drops by more than this fraction.
REGRESSION_THRESHOLD = 0.05  # 5%

# Metrics where a *higher* value is better (packets/sec, GB/sec, blocks/sec).
# Everything in the results dict is "higher is better" by convention.
RATE_KEYS = ("packets/sec", "GB/sec", "blocks/sec", "runs/sec",
             "input_GB/sec", "output_GB/sec")


def pct(delta: float) -> str:
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta * 100:.1f}%"


def compare_scalar(label: str, key: str, a, b) -> tuple[bool, str]:
    """Return (is_regression, formatted_line)."""
    if a is None and b is None:
        return False, f"  {label:<45} {'n/a':>12} -> {'n/a':>12}"
    if a is None:
        return False, f"  {label:<45} {'(new)':>12} -> {b:>12.4f}"
    if b is None:
        return False, f"  {label:<45} {a:>12.4f} -> {'(missing)':>12}"

    delta = (b - a) / a if a != 0 else 0.0
    regression = delta < -REGRESSION_THRESHOLD
    flag = "  !! REGRESSION" if regression else ""
    line = f"  {label:<45} {a:>12.4f} -> {b:>12.4f}  ({pct(delta)}){flag}"
    return regression, line


def config_key(row: dict) -> tuple:
    return (int(row.get("nr_channels", 0)), int(row.get("nr_fpga_sources", 0)))


def config_label(row: dict) -> str:
    return f"{int(row.get('nr_channels', '?'))}ch/{int(row.get('nr_fpga_sources', '?'))}fpga"


def compare_stage(stage_name: str, base_val, curr_val, report: list, regressions: list):
    """Compare one stage's results.  Values may be a dict (single config) or list of dicts."""

    def compare_row(label_prefix: str, b: dict, c: dict):
        for key in RATE_KEYS:
            if key in b or key in c:
                label = f"{label_prefix} {key}"
                reg, line = compare_scalar(label, key, b.get(key), c.get(key))
                report.append(line)
                if reg:
                    regressions.append(line.strip())

    if isinstance(base_val, list) or isinstance(curr_val, list):
        base_by_cfg = {config_key(r): r for r in (base_val or [])}
        curr_by_cfg = {config_key(r): r for r in (curr_val or [])}
        all_keys = sorted(set(base_by_cfg) | set(curr_by_cfg))
        for cfg in all_keys:
            b = base_by_cfg.get(cfg, {})
            c = curr_by_cfg.get(cfg, {})
            lbl_row = b or c
            label_prefix = f"[{stage_name}][{config_label(lbl_row)}]"
            compare_row(label_prefix, b, c)
    elif isinstance(base_val, dict) or isinstance(curr_val, dict):
        compare_row(f"[{stage_name}]", base_val or {}, curr_val or {})


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} baseline.json current.json", file=sys.stderr)
        sys.exit(2)

    baseline_path = Path(sys.argv[1])
    current_path = Path(sys.argv[2])

    baseline = json.loads(baseline_path.read_text())
    current = json.loads(current_path.read_text())

    # --- Header ---
    print("=" * 78)
    print("BENCHMARK COMPARISON")
    print("=" * 78)
    base_meta = baseline.get("meta", {})
    curr_meta = current.get("meta", {})
    print(f"  baseline : {baseline_path.name}  "
          f"commit={base_meta.get('git_commit', '?')}  "
          f"branch={base_meta.get('git_branch', '?')}  "
          f"ts={base_meta.get('timestamp', '?')}")
    print(f"  current  : {current_path.name}  "
          f"commit={curr_meta.get('git_commit', '?')}  "
          f"branch={curr_meta.get('git_branch', '?')}  "
          f"ts={curr_meta.get('timestamp', '?')}")
    print()

    print(f"  {'metric':<45} {'baseline':>12}    {'current':>12}  change")
    print("-" * 78)

    report: list[str] = []
    regressions: list[str] = []

    stages = sorted(
        set(k for k in list(baseline) + list(current) if k != "meta")
    )
    for stage in stages:
        base_val = baseline.get(stage)
        curr_val = current.get(stage)
        if base_val is None and curr_val is None:
            continue
        compare_stage(stage, base_val, curr_val, report, regressions)

    for line in report:
        print(line)

    print("=" * 78)
    if regressions:
        print(f"\n!! {len(regressions)} REGRESSION(S) detected (>{REGRESSION_THRESHOLD*100:.0f}% slowdown):")
        for r in regressions:
            print(f"   {r}")
        sys.exit(1)
    else:
        print("\nNo regressions detected.")
        sys.exit(0)


if __name__ == "__main__":
    main()
