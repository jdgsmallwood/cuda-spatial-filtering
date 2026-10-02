#!/usr/bin/env python3
"""Plot a dedispersed, Stokes-I PSRFITS fold and report diagnostic pulse scores.

Convert a PSRCHIVE archive using ``pam -D -a PSRFITS -e fits archive.ar``.
Scores maximize over phase and several widths: they are NOT trial-corrected
detection significances. Inspect time/frequency consistency and the DM curve.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def robust_sigma(x, axis=-1):
    median = np.median(x, axis=axis, keepdims=True)
    return 1.4826 * np.median(np.abs(x - median), axis=axis)


def boxcar_scores(profile, width):
    """Circular window starting at each phase, with the DC component removed."""
    x = np.asarray(profile, dtype=float)
    centered = x - x.mean()
    sums = sum(np.roll(centered, -i) for i in range(width))
    sigma = max(float(robust_sigma(centered)), np.finfo(float).tiny)
    return sums / (sigma * np.sqrt(width * (1 - width / len(x))))


def best_pulse(profile):
    candidates = []
    for width in (1, 2, 4, 8, 16, 32, 64):
        if width >= len(profile) // 2:
            continue
        scores = boxcar_scores(profile, width)
        phase = int(np.argmax(scores))
        candidates.append(dict(score=float(scores[phase]), width_bins=width,
                               start_bin=phase))
    return max(candidates, key=lambda x: x['score'])


def analyze(filename, prefix, dm_max=100):
    from astropy.io import fits
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with fits.open(filename) as h:
        tab = h['SUBINT'].data
        hdr = h['SUBINT'].header
        nchan, nbin, npol = hdr['NCHAN'], hdr['NBIN'], hdr['NPOL']
        if npol != 1:
            raise ValueError('Expected a single Stokes-I polarization')
        if str(h[0].header.get('OBS_MODE', '')).strip() != 'PSR':
            raise ValueError('Expected a folded PSRFITS archive')
        nsub = len(tab)
        raw = np.asarray(tab['DATA'], dtype=float).reshape(nsub, npol, nchan, nbin)[:, 0]
        scales = np.asarray(tab['DAT_SCL'], dtype=float).reshape(nsub, nchan)
        offsets = np.asarray(tab['DAT_OFFS'], dtype=float).reshape(nsub, nchan)
        data = raw * scales[..., None] + offsets[..., None]
        weights = np.asarray(tab['DAT_WTS'], dtype=float).copy()
        freqs = np.asarray(tab['DAT_FREQ'], dtype=float).copy()
        durations = np.asarray(tab['TSUBINT'], dtype=float).copy()
        times = np.asarray(tab['OFFS_SUB'], dtype=float).copy()
        if 'HISTORY' not in h or not int(h['HISTORY'].data[-1]['DEDISP']):
            raise ValueError('Input must be dedispersed first (pam -D)')
        periods = np.asarray(tab['PERIOD'], dtype=float)
        valid_periods = periods[np.isfinite(periods) & (periods > 0)]
        # PSRCHIVE can leave SUBINT.PERIOD zero when a timing predictor is
        # present. HISTORY.TBIN records the folded phase-bin duration.
        period = (float(np.median(valid_periods)) if len(valid_periods) else
                  float(h['HISTORY'].data[-1]['TBIN']) * nbin)
        if not np.isfinite(period) or period <= 0:
            raise ValueError('No valid folding period in archive metadata')
        dm = float(hdr['DM'])
        source = h[0].header.get('SRC_NAME', '?')

    finite = np.isfinite(data).all(axis=-1)
    data[~finite] = 0
    noise = robust_sigma(data)
    usable = finite & (weights > 0) & (noise > 0)
    weights = np.where(usable, weights, 0)
    if not usable.any():
        raise ValueError('No finite, nonconstant, unflagged profiles')
    white = (data - np.median(data, axis=-1)[..., None]) / np.where(noise > 0, noise, 1)[..., None]
    # Whiten each folded profile, then weight by sqrt(integration time).
    # With stationary noise this recovers the usual duration weighting.
    weight = weights * np.sqrt(durations)[:, None]
    weighted = white * weight[..., None]
    profile = weighted.sum(axis=(0, 1))
    best = best_pulse(profile)
    retained = np.flatnonzero((weights > 0).any(axis=0))
    halves = {
        'time_first': weighted[:max(1, nsub // 2)].sum(axis=(0, 1)),
        'time_second': weighted[max(1, nsub // 2):].sum(axis=(0, 1)),
        'frequency_low': weighted[:, retained[:len(retained) // 2]].sum(axis=(0, 1)),
        'frequency_high': weighted[:, retained[len(retained) // 2:]].sum(axis=(0, 1)),
    }
    split_scores = {name: float(boxcar_scores(p, best['width_bins'])[best['start_bin']])
                    if np.any(p) else None for name, p in halves.items()}

    # Input must already be dedispersed at the archive DM. Shift from that DM
    # to each trial using the cold-plasma delay relative to the highest band.
    spectrum = np.fft.rfft(weighted.sum(axis=0), axis=-1)
    frequency = np.median(freqs, axis=0)
    harmonics = np.arange(spectrum.shape[-1])
    delay_per_dm = 4.148808e3 * (frequency ** -2 - frequency.max() ** -2)
    near_dm = dm + np.linspace(-2, 2, 161)
    near_dm = near_dm[(near_dm >= 0) & (near_dm <= dm_max)]
    trials = np.unique(np.r_[np.linspace(0, dm_max, 201), near_dm, dm])
    dm_scores = []
    for trial in trials:
        advance = delay_per_dm * (trial - dm) / period
        shifted = np.fft.irfft((spectrum * np.exp(2j * np.pi * advance[:, None] * harmonics)).sum(axis=0), n=nbin)
        dm_scores.append(best_pulse(shifted)['score'])
    report = dict(source=source, input=str(filename), subintegrations=nsub,
                  duration_seconds=float(durations.sum()), period_seconds=period,
                  archive_dm=dm, retained_channels=int(len(retained)),
                  retained_channels_per_subint=(weights > 0).sum(axis=1).tolist(),
                  finite_profiles=int(finite.sum()), total_profiles=int(finite.size),
                  best_pulse=best, fixed_phase_split_scores=split_scores,
                  best_trial_dm=float(trials[np.argmax(dm_scores)]),
                  best_trial_dm_score=float(max(dm_scores)),
                  warning='Diagnostic scores include phase/width/DM trials; not a detection decision.')
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix('.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    np.savez_compressed(prefix.with_suffix('.npz'), profile=profile, time_phase=weighted.sum(axis=1),
                        frequency_phase=weighted.sum(axis=0), frequencies_mhz=frequency,
                        trials=trials, dm_scores=dm_scores, times=times)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    phase = np.arange(nbin) / nbin
    norm = max(float(robust_sigma(profile)), np.finfo(float).tiny)
    axes[0, 0].plot(phase, (profile - np.median(profile)) / norm)
    axes[0, 0].set(xlabel='Pulse phase', ylabel='Profile / robust noise',
                   title=f"Best boxcar score {best['score']:.1f}, width {best['width_bins']} bins")
    time_image = weighted.sum(axis=1)
    time_image /= np.maximum(robust_sigma(time_image)[:, None], 1e-20)
    time_edges = np.r_[times - durations / 2, times[-1] + durations[-1] / 2]
    time_edges -= time_edges[0]
    axes[0, 1].pcolormesh(np.linspace(0, 1, nbin + 1), time_edges, time_image,
                          vmin=-3, vmax=6, shading='flat')
    axes[0, 1].set(xlabel='Pulse phase', ylabel='Time into observation (s)')
    freq_image = weighted.sum(axis=0)
    freq_image /= np.maximum(robust_sigma(freq_image)[:, None], 1e-20)
    freq_image[(weights > 0).sum(axis=0) == 0] = np.nan
    axes[1, 0].imshow(freq_image, origin='lower', aspect='auto',
                      extent=[0, 1, frequency[0], frequency[-1]], vmin=-3, vmax=6, interpolation='nearest')
    axes[1, 0].set(xlabel='Pulse phase', ylabel='Frequency (MHz)')
    axes[1, 1].plot(trials, dm_scores)
    axes[1, 1].axvline(dm, color='tab:red', linestyle='--', label=f'Archive DM {dm:.3f}')
    axes[1, 1].set(xlabel='Trial DM (pc cm$^{-3}$)', ylabel='Best phase/width score')
    axes[1, 1].legend()
    fig.suptitle(f'{source}: {durations.sum():.1f} s, {len(retained)} retained channels\n'
                 'Diagnostic plots; scores are not corrected for search trials')
    fig.savefig(prefix.with_suffix('.png'), dpi=160)
    plt.close(fig)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('fits', type=Path)
    p.add_argument('--output-prefix', required=True, type=Path)
    p.add_argument('--dm-max', type=float, default=100)
    a = p.parse_args()
    print(json.dumps(analyze(a.fits, a.output_prefix, a.dm_max), indent=2))
