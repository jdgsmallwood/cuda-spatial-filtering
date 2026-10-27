#!/usr/bin/env python3
"""Build an offline per-coarse-plus-local-delay PN phase correction candidate.

The correction is diagnostic: it is not applied to live packets, GPU buffers,
or science HDF5 files. For a stream, phase(c, f) = phi_c + 2*pi*(f-f_c)*tau.
The four comb teeth used to measure phi_c average 36.621 kHz below f_c, so
their mean phase is shifted to the centre using the separately fitted tau.
"""

import argparse
import json
from pathlib import Path

import numpy as np

FINE_HZ = 781250.0 / 27.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pn_band", type=Path)
    ap.add_argument("pn_local_fit", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.pn_band) as z:
        channels = z["channels"].astype(int)
        if "phase_rad" in z:
            mean_phase = z["phase_rad"]
            repeat = z["repeat_error_rad"]
        else:
            # A single eight-channel fit already contains both independent
            # PN-on toggles, so it can serve as the band phase measurement.
            runs = z["coarse_mean_phase_rad"]
            if runs.shape != (2, len(channels), 80):
                raise ValueError("PN toggle phases have unexpected shape")
            mean_phase = np.angle(np.mean(np.exp(1j * runs), axis=0))
            repeat = abs(np.angle(np.exp(1j * (runs[1] - runs[0]))))
    with np.load(args.pn_local_fit) as z:
        fit = z["fit_delay_ns"]
        offsets = z["line_offsets_hz"]
        local_channels = z["channels"].astype(int)
    if not len(channels) or not np.array_equal(channels, np.arange(channels[0], channels[0] + len(channels))):
        raise ValueError("PN band channels must be contiguous")
    if not np.array_equal(local_channels, channels[:len(local_channels)]) or len(local_channels) < 6:
        raise ValueError("Local PN fit must cover at least the first six band channels")
    if mean_phase.shape != (len(channels), 80) or repeat.shape != mean_phase.shape:
        raise ValueError("PN band phase and repeat arrays have unexpected shape")
    if fit.shape != (2, len(local_channels), 4, 80):
        raise ValueError("PN local delay fit has unexpected shape")
    tau_ns = np.full((2, 80), np.nan)
    for run in range(2):
        for raw in range(80):
            tau_ns[run, raw] = 0 if raw == 2 else np.nanmedian(fit[run, :6, :, raw])
    tau_s = tau_ns[1] * 1e-9
    local_repeat_ns = abs(tau_ns[1] - tau_ns[0])
    mean_offset = float(np.mean(offsets))
    centre_phase = np.angle(np.exp(1j * (mean_phase -
                                        2 * np.pi * mean_offset * tau_s[None, :])))
    raw_fine_bin = np.arange(3, 30)
    fine_offset = (raw_fine_bin - 16) * FINE_HZ
    total_phase = centre_phase[:, None, :] + \
        2 * np.pi * fine_offset[None, :, None] * tau_s[None, None, :]
    correction = np.exp(-1j * total_phase).astype(np.complex64)
    stable = (repeat <= .5) & (local_repeat_ns[None, :] <= 100)
    np.savez_compressed(args.output / "pn_gain_candidate.npz",
                        channels=channels, raw_fine_bin=raw_fine_bin,
                        fine_frequency_hz=channels[:, None] * 781250.0 +
                        fine_offset[None, :],
                        mean_selected_tooth_offset_hz=mean_offset,
                        centre_phase_rad=centre_phase,
                        local_delay_ns=tau_ns[1],
                        local_delay_repeat_abs_ns=local_repeat_ns,
                        coarse_phase_repeat_abs_rad=repeat,
                        stable=stable, phase_rad=total_phase,
                        correction=correction)
    report = {
        "channels": [int(channels[0]), int(channels[-1])],
        "raw_fine_bins": [int(raw_fine_bin[0]), int(raw_fine_bin[-1])],
        "reference": "Raw reference stream selected by the input calibration; arbitrary common phase cancels on baselines",
        "mean_selected_tooth_offset_hz": mean_offset,
        "formula": "phi(c,f) = wrap(phi_mean_tooth(c) - 2*pi*mean_tooth_offset*tau_local + 2*pi*(f-f_c)*tau_local); voltage correction = exp(-i*phi); baseline correction = C_a*conj(C_b)",
        "local_delay_repeat_median_ns_all_raw_streams": float(np.median(local_repeat_ns)),
        "local_delay_repeat_p90_ns_all_raw_streams": float(np.quantile(local_repeat_ns, .9)),
        "stable_channel_stream_pairs": int(stable.sum()),
        "total_channel_stream_pairs": int(stable.size),
        "status": "Offline candidate only; requires held-out sky visibility validation before use in production or science data",
    }
    (args.output / "pn_gain_candidate_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
