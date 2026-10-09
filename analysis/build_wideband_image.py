#!/usr/bin/env python3
"""Make cached full-hemisphere YY images from every unmasked fine channel.

The input capture already has the verified FPGA sample delays applied. This
script averages only nearby integrations to limit sky-motion smearing and
applies candidate gains per frequency before UV gridding. Output is diagnostic.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from calibration_viewer_utils import read_masked_spectrum, wideband_dirty_image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("report", type=Path)
    p.add_argument("fine_candidates", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--time-index", type=int, default=150,
                   help="Index in the comparison's sampled integration list")
    p.add_argument("--sample-count", type=int, default=15,
                   help="Nearby sampled integrations, about two seconds apart")
    args = p.parse_args()
    report = json.loads(args.report.read_text())
    run = args.report.parent.parent
    config = json.loads((run / "starspin_config.json").read_text())
    with np.load(run / "flags_corrected_v2/transient_mask.npz") as z:
        good = z["good"].astype(bool)
        frequency = z["frequency_hz"].astype(float)
    with np.load(args.fine_candidates) as z:
        ids = z["antenna_ids"].astype(int)
        direct_gains = z["gains"][:]
        delay = z["delay_s"][:]
    with np.load(args.report.parent / "candidates.npz") as z:
        held = z["held_weights"][:]
        centre = np.full((40, len(ids)), np.nan + 0j, np.complex128)
        for coarse in range(40):
            key = f"gains_A_multisource_amplitude_YY_{coarse}"
            if key in z:
                centre[coarse] = z[key]
    used = np.asarray(report["used_integrations"], dtype=int)
    if not (0 <= args.time_index < len(used)) or args.sample_count < 1:
        p.error("time-index or sample-count is outside the comparison")
    lo = max(0, args.time_index - args.sample_count // 2)
    hi = min(len(used), lo + args.sample_count)
    selected = used[lo:hi]
    pairs, spectrum, counts = read_masked_spectrum(
        report["capture"], selected, good, "YY")
    positions = {int(a): np.array([v["east"], v["north"], v["up"]])
                 for board in config["antenna_positions"].values()
                 for a, v in board.items()}
    index = {int(a): i for i, a in enumerate(ids)}
    held_pair = np.array([
        (int(a) in index and int(b) in index and
         held[index[int(a)], index[int(b)]] > 0)
        if int(a) in index and int(b) in index else False
        for a, b in pairs], dtype=bool)
    coarse = np.arange(len(frequency)) // 27
    centre_freq = np.asarray(report["frequencies_hz"])[coarse]
    smooth_gains = centre[coarse] * np.exp(
        2j * np.pi * (frequency - centre_freq)[:, None] * delay[None, :])
    all_images = []
    channel_counts = []
    sample_counts = []
    for name, allowed in (("all", None), ("held", held_pair)):
        mode_images = []
        mode_bins = []
        mode_samples = []
        for mode, gain in (("raw", None), ("fine", direct_gains),
                           ("delay", smooth_gains)):
            grid, image, nbins, nsamples = wideband_dirty_image(
                pairs, spectrum, counts, frequency, positions,
                gains=gain, antenna_ids=ids, allowed_pairs=allowed)
            mode_images.append(image.astype(np.float32))
            mode_bins.append(nbins)
            mode_samples.append(nsamples)
            print(name, mode, "fine bins", nbins,
                  "baseline-frequency samples", nsamples, flush=True)
        all_images.append(mode_images)
        channel_counts.append(mode_bins)
        sample_counts.append(mode_samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, axis_lm=grid,
                        images=np.asarray(all_images),
                        fine_bins=np.asarray(channel_counts),
                        uv_samples=np.asarray(sample_counts),
                        integration_indices=selected,
                        time_mjd=np.load(args.report.parent / "candidates.npz")
                        ["times_mjd"][lo:hi],
                        frequency_hz=frequency,
                        contributing_integrations=counts,
                        modes=np.array(["raw", "fine", "delay"]),
                        baseline_sets=np.array(["all", "held"]))
    metadata = dict(status="diagnostic dirty image; no accepted science gains",
                    capture=report["capture"], input_report=str(args.report),
                    input_fine_candidates=str(args.fine_candidates),
                    sampled_integrations=selected.tolist(),
                    frequency_hz=[float(frequency[0]), float(frequency[-1])],
                    baseline_sets=["all", "held"], modes=["raw", "fine", "delay"],
                    fine_bins=np.asarray(channel_counts).tolist(),
                    baseline_frequency_samples=np.asarray(sample_counts).tolist(),
                    notes=["Every RFI-unmasked selected time/frequency cell contributes.",
                           "Fine gains are unavailable for 13 channels; those channels are omitted only from that candidate image.",
                           "Each frequency is normalized by robust cross-visibility amplitude before gridding; intensity is arbitrary.",
                           "Bilinear UV gridding uses the array's nearly coplanar east/north coordinates; no CLEAN or beam correction."])
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
