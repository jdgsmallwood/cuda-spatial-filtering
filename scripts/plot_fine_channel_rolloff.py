#!/usr/bin/env python3
"""Measure retained 32-way fine-bin roll-off and compare duplicate seam bins.

The live pipeline keeps raw gpu-filter bins 2..29. Bin 29 of coarse c and bin
2 of coarse c+1 nominally sample the same seam frequency in a 32/27 system.
"""
import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('output_prefix', type=Path)
    args = parser.parse_args()
    with h5py.File(args.capture) as capture:
        pairs = capture['baseline_antenna_ids'][:]
        autos = np.flatnonzero((pairs[:, 0] == pairs[:, 1]) &
                              (pairs[:, 0] > 0) & (pairs[:, 0] <= 36) &
                              (~np.isin(pairs[:, 0], [9, 14])))
        times = np.linspace(5, len(capture['visibilities']) - 6, 5).astype(int)
        results = {}
        seam_profiles = {}
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True,
                                 constrained_layout=True)
        for hand, pol in ((1, 'XX'), (0, 'YY')):
            power = np.stack([
                capture['visibilities'][int(t), :, autos, hand, hand, 0].T
                for t in times
            ])  # time, antenna, processing frequency
            power = power.reshape(len(times), len(autos), 40, 28)
            local = power / np.maximum(np.median(power, axis=-1, keepdims=True), 1e-30)
            db = 10 * np.log10(np.maximum(local, 1e-30))
            profile = np.median(db, axis=(0, 1, 2))
            # Compare the two copies of each of the 39 internal seam centres.
            seam_db = 10 * np.log10(np.maximum(power[:, :, 1:, 0], 1e-30) /
                                    np.maximum(power[:, :, :-1, -1], 1e-30))
            seam_profiles[pol] = np.median(seam_db, axis=(0, 1))
            seam_db = seam_db[np.isfinite(seam_db) & (abs(seam_db) < 10)]
            results[pol] = {
                'raw_gpu_bin_2_relative_power_db': float(profile[0]),
                'raw_gpu_bin_3_relative_power_db': float(profile[1]),
                'raw_gpu_bin_29_relative_power_db': float(profile[-1]),
                'next_raw_2_minus_previous_raw_29_db_median': float(np.median(seam_db)),
                'seam_ratio_db_p10_p90': np.percentile(seam_db, [10, 90]).tolist(),
                'fraction_of_seams_raw_29_stronger': float(np.mean(seam_db < 0)),
            }
            axis = axes[0 if pol == 'XX' else 1]
            axis.axvspan(2.5, 29.5, color='tab:green', alpha=.08,
                         label='proposed raw bins 3–29')
            axis.plot(np.arange(2, 30), profile, 'o-', ms=3)
            axis.axhline(0, color='black', lw=.7)
            axis.set(title=pol, xlabel='Original gpu-filter fine-bin index',
                     ylabel='Median power relative to coarse median (dB)',
                     xticks=[2, 3, 8, 16, 24, 29], ylim=(-1.2, .35))
            axis.grid(alpha=.3)
            axis.legend(fontsize=8)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle('32-way fine channelizer: measured retained bins 2–29')
    fig.savefig(args.output_prefix.with_suffix('.png'), dpi=170)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(11, 4), constrained_layout=True)
    for pol, color in (('XX', 'tab:blue'), ('YY', 'tab:orange')):
        axis.plot(np.arange(1, 40), seam_profiles[pol], 'o-', ms=3,
                  color=color, label=pol)
    axis.axhline(0, color='black', lw=.7)
    axis.set(xlabel='Boundary after coarse channel (0-based within capture)',
             ylabel='Next raw bin 2 / previous raw bin 29 (dB)',
             title='Two measurements of each seam frequency; one copy is consistently attenuated')
    axis.grid(alpha=.3)
    axis.legend()
    fig.savefig(args.output_prefix.with_name(args.output_prefix.name + '_seams').with_suffix('.png'),
                dpi=170)
    plt.close(fig)
    result = {
        'capture': str(args.capture),
        'retained_raw_bins': [2, 29],
        'recommended_unique_raw_bins': [3, 29],
        'unmeasured_discarded_raw_bins': [0, 1, 30, 31],
        'fft_fine_spacing_hz': 781250.0 / 27.0,
        'standard_fft_bin_offsets_from_coarse_center': {'raw_2': -14, 'raw_3': -13,
                                                         'raw_16': 0, 'raw_29': 13},
        'current_metadata_offset_from_fft_centers_hz': 0.5 * 781250.0 / 27.0,
        'sampled_visibility_rows': times.tolist(),
        'polarizations': results,
    }
    args.output_prefix.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
