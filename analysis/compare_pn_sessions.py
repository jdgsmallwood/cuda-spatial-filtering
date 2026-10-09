#!/usr/bin/env python3
"""Measure how an ADC-PN phase candidate changes across FPGA boot sessions.

The coarse phase difference is fitted with a circular frequency slope per raw
stream. A coherent slope is still a changed calibration, even when the phase
at one coarse centre happens to agree. The fit is modulo 1/781250 Hz.
"""

import argparse
import json
from pathlib import Path

import numpy as np


COARSE_HZ = 781_250.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("earlier", type=Path)
    parser.add_argument("later", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    with np.load(args.earlier) as earlier, np.load(args.later) as later:
        channels = earlier["channels"].astype(int)
        if not np.array_equal(channels, later["channels"]):
            raise ValueError("PN candidates have different coarse-channel axes")
        phase_difference = np.exp(1j * (later["centre_phase_rad"] -
                                       earlier["centre_phase_rad"]))
        accepted = earlier["stable"] & later["stable"]
        local_change = later["local_delay_ns"] - earlier["local_delay_ns"]
    trial_ns = np.arange(-640.0, 641.0, 1.0)
    x_hz = (channels - channels[0]) * COARSE_HZ
    slopes = np.full(phase_difference.shape[1], np.nan)
    coherence = np.full_like(slopes, np.nan)
    uncorrected_coherence = np.full_like(slopes, np.nan)
    for stream in range(phase_difference.shape[1]):
        valid = accepted[:, stream]
        if valid.sum() < 12:
            continue
        observed = phase_difference[valid, stream]
        uncorrected_coherence[stream] = abs(observed.mean())
        trial = np.exp(-2j * np.pi * x_hz[valid, None] *
                       trial_ns[None, :] * 1e-9)
        scores = abs(np.mean(observed[:, None] * trial, axis=0))
        index = int(np.argmax(scores))
        slopes[stream] = trial_ns[index]
        coherence[stream] = scores[index]
    report = {
        "earlier": str(args.earlier), "later": str(args.later),
        "reference": "Each candidate references FPGA0 raw stream 2 independently in each coarse channel",
        "fit": "Circular phase difference versus coarse-centre frequency; delay aliases every 1280 ns",
        "phase_difference_linear_slope_ns_by_stream": slopes.tolist(),
        "slope_fit_coherence_by_stream": coherence.tolist(),
        "uncorrected_coherence_by_stream": uncorrected_coherence.tolist(),
        "within_coarse_local_delay_change_ns_by_stream": local_change.tolist(),
        "by_fpga": {},
        "interpretation": "A fresh boot changes the apparent PN phase slopes. The coarse-centre and within-coarse changes disagree, so a previous boot's PN table must not be silently reused; neither term alone identifies a physical clock delay.",
    }
    for fpga in range(4):
        sl = slice(fpga * 20, (fpga + 1) * 20)
        report["by_fpga"][str(fpga)] = {
            "median_coarse_slope_change_ns_modulo_1280": float(np.nanmedian(slopes[sl])),
            "median_coarse_slope_fit_coherence": float(np.nanmedian(coherence[sl])),
            "median_uncorrected_coherence": float(np.nanmedian(uncorrected_coherence[sl])),
            "median_within_coarse_local_delay_change_ns": float(np.nanmedian(local_change[sl])),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["by_fpga"], indent=2))


if __name__ == "__main__":
    main()
