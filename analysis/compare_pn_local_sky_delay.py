#!/usr/bin/env python3
"""Compare independent PN local slopes with previously fitted sky fine-bin delays."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pn_multitone", type=Path)
    ap.add_argument("sky_report", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--mapping", type=Path, default=Path("stream_antenna_map.json"))
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.pn_multitone) as z:
        fit = z["fit_delay_ns"]
        channels = z["channels"]
    if not np.array_equal(channels, np.arange(400, 408)):
        raise ValueError("Expected PN local fit on channels 400-407")
    local = np.full((2, 80), np.nan)
    for run in range(2):
        for raw in range(80):
            if raw == 2:
                local[run, raw] = 0
            else:
                local[run, raw] = np.nanmedian(fit[run, :6, :, raw])
    mapping = json.loads(args.mapping.read_text())["fpgas"]
    raw_for_antenna = {
        int(stream["antenna_id"]): int(board) * 20 + raw
        for board, fpga in mapping.items()
        for raw, stream in enumerate(fpga["streams"])
        if stream["antenna_id"] > 0 and stream["polarization"] == 0
    }
    sky = json.loads(args.sky_report.read_text())
    sky_delay = dict(zip(sky["antenna_ids"], sky["per_antenna_delay_ns"]))
    reference = raw_for_antenna[1]
    rows = []
    for antenna, raw in sorted(raw_for_antenna.items()):
        if antenna in (1, 8, 12, 14) or antenna not in sky_delay:
            continue
        pn_delay = local[1, raw] - local[1, reference]
        if not np.isfinite(pn_delay):
            continue
        rows.append({
            "antenna_id": antenna, "fpga": raw // 20,
            "pn_local_delay_relative_to_antenna_1_ns": float(pn_delay),
            "sky_fine_delay_relative_to_antenna_1_ns": float(sky_delay[antenna]),
            "pn_toggle_repeat_abs_ns": float(abs(local[1, raw] - local[0, raw])),
        })
    x = np.array([row["pn_local_delay_relative_to_antenna_1_ns"] for row in rows])
    y = np.array([row["sky_fine_delay_relative_to_antenna_1_ns"] for row in rows])
    repeat = np.array([row["pn_toggle_repeat_abs_ns"] for row in rows])
    bias = float(np.median(y - x))
    result = {
        "pn_multitone": str(args.pn_multitone),
        "sky_report": str(args.sky_report),
        "pn_local_fit_channels": [400, 401, 402, 403, 404, 405],
        "n_antenna_hands": len(rows),
        "excluded_antenna_ids": [1, 8, 12, 14],
        "pearson_correlation": float(np.corrcoef(x, y)[0, 1]),
        "median_abs_difference_ns": float(np.median(abs(y - x))),
        "median_sky_minus_pn_ns": bias,
        "median_abs_difference_after_common_bias_ns": float(np.median(abs(y - x - bias))),
        "pn_toggle_repeat_median_ns": float(np.median(repeat)),
        "pn_toggle_repeat_p90_ns": float(np.quantile(repeat, .9)),
        "by_fpga": {
            str(board): {
                "n": sum(row["fpga"] == board for row in rows),
                "median_pn_local_delay_ns": float(np.median([row["pn_local_delay_relative_to_antenna_1_ns"] for row in rows if row["fpga"] == board])),
                "median_sky_fine_delay_ns": float(np.median([row["sky_fine_delay_relative_to_antenna_1_ns"] for row in rows if row["fpga"] == board])),
            } for board in range(4)
        },
        "antenna_results": rows,
        "limitation": "The sky fit used a nighttime point-source model with RFI and modest held-out visibility correlation. Agreement is independent evidence for a delay-like component, not a validated absolute hardware correction."
    }
    (args.output / "pn_local_vs_sky_fine_delay.json").write_text(json.dumps(result, indent=2) + "\n")
    fig, ax = plt.subplots(figsize=(8, 7), layout="constrained")
    for board in range(4):
        subset = np.array([row["fpga"] == board for row in rows])
        ax.scatter(x[subset], y[subset], label=f"FPGA {board}", s=32)
    bounds = [-200, 850]
    ax.plot(bounds, bounds, color="0.5", ls="--", label="Equality")
    ax.plot(bounds, np.array(bounds) + bias, color="tab:purple", ls=":",
            label=f"Equality + median {bias:.0f} ns")
    ax.set(xlabel="PN local slope relative to antenna 1 (ns)",
           ylabel="Sky fine-bin delay relative to antenna 1 (ns)",
           title=f"Independent delay estimates: r={result['pearson_correlation']:.3f}",
           xlim=bounds, ylim=bounds)
    ax.grid(alpha=.25)
    ax.legend()
    fig.savefig(args.output / "pn_local_vs_sky_fine_delay.png", dpi=170)
    print(json.dumps({k: v for k, v in result.items() if k not in ("antenna_results", "by_fpga")}, indent=2))


if __name__ == "__main__":
    main()
