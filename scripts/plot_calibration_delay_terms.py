#!/usr/bin/env python3
"""Visualize FPGA-associated and per-receptor gain-phase slopes.

The split uses a gauge: FPGA 1 is the reference and both polarizations contribute
to one group-associated term. This fitted slope is not a physical residual FPGA
clock delay after integer-sample alignment.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def circular_fit(freq_hz, phasor, coarse, trial_delay_ns):
    """Search phase slope while allowing one nuisance phase per coarse bin."""
    centered = freq_hz - np.median(freq_hz)
    corrected = np.exp(-2j * np.pi * 1e-9 * trial_delay_ns[:, None] * centered) * phasor
    objective = np.zeros(len(trial_delay_ns))
    for band in np.unique(coarse):
        objective += abs(corrected[:, coarse == band].sum(axis=1))
    objective /= len(phasor)
    peak = int(np.argmax(objective))
    best = corrected[peak]
    residual = np.empty(len(best))
    for band in np.unique(coarse):
        use = coarse == band
        residual[use] = np.angle(best[use] * np.exp(-1j * np.angle(best[use].sum())))
    return float(trial_delay_ns[peak]), float(objective[peak]), objective, residual, best


def wrap_delay(delay_ns, period_ns):
    return (delay_ns + period_ns / 2) % period_ns - period_ns / 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fit', type=Path, help='native calibration validation.npz')
    parser.add_argument('stream_map', type=Path)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--reference-antenna', type=int, default=26,
                        help='stable receptor on reference FPGA 1')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.fit) as fit:
        freq = fit['frequencies_hz'] / 1e6
        f0 = float(np.median(freq))
        gains = fit['gains']
        good = fit['good']
        order = fit['antenna_order']
    mapping = json.loads(args.stream_map.read_text())['fpgas']
    fpga_by_ant = {}
    for board, value in mapping.items():
        for stream in value['streams']:
            antenna = int(stream['antenna_id'])
            if antenna > 0:
                fpga_by_ant[antenna] = int(board)
    fpga = np.array([fpga_by_ant[int(antenna)] for antenna in order])
    reference = int(np.flatnonzero(order == args.reference_antenna)[0])
    if fpga[reference] != 1:
        parser.error('reference antenna must be on FPGA 1')
    spacing_hz = float(np.median(np.diff(np.unique(freq * 1e6))))
    alias_period_ns = 1e9 / spacing_hz
    trial_delay_ns = np.arange(-20000., 20000.1, 20.)
    coarse = np.arange(len(freq)) // 28
    delay_ns = np.full((2, len(order)), np.nan)
    phase_f0 = np.full_like(delay_ns, np.nan)
    scatter_deg = np.full_like(delay_ns, np.nan)
    coherence = np.full_like(delay_ns, np.nan)
    phase_traces = {}
    for pi, pol in enumerate(('XX', 'YY')):
        use = good[:, pi]
        if use.sum() < 10:
            continue
        for ai, antenna in enumerate(order):
            if np.median(abs(gains[use, pi, ai])) < 1e-5:
                continue
            relative = gains[use, pi, ai] * gains[use, pi, reference].conj()
            relative /= np.maximum(abs(relative), 1e-30)
            fitted, strength, objective, residual, corrected = circular_fit(
                freq[use] * 1e6, relative, coarse[use], trial_delay_ns)
            delay_ns[pi, ai] = fitted
            coherence[pi, ai] = strength
            phase_f0[pi, ai] = np.angle(corrected.sum())
            scatter_deg[pi, ai] = np.rad2deg(np.median(abs(residual)))
            phase_traces[(pi, int(antenna))] = (freq[use], residual, objective)
    group_delay = np.full((2, 4), np.nan)
    group_phase = np.full((2, 4), np.nan)
    group_weight = np.zeros((2, 4))
    for pi in range(2):
        for board in range(4):
            members = (fpga == board) & np.isfinite(delay_ns[pi]) & (coherence[pi] > .55)
            if members.any():
                weight = coherence[pi, members] ** 2
                angle = 2 * np.pi * delay_ns[pi, members] / alias_period_ns
                group_delay[pi, board] = np.angle(np.sum(weight * np.exp(1j * angle))) * alias_period_ns / (2 * np.pi)
                group_phase[pi, board] = np.angle(np.sum(weight * np.exp(1j * phase_f0[pi, members])))
                group_weight[pi, board] = weight.sum()
    by_pol_delay = wrap_delay(group_delay - group_delay[:, 1, None], alias_period_ns)
    by_pol_phase = np.angle(np.exp(1j * (group_phase - group_phase[:, 1, None])))
    # The physical board term is shared by both hands. Their independent gain
    # phase gauges are first set to zero on FPGA 1, then the group summaries
    # are pooled; remaining hand-specific structure belongs to receptor terms.
    pooled_delay = np.angle(np.nansum(group_weight * np.exp(
        2j * np.pi * group_delay / alias_period_ns), axis=0)) * alias_period_ns / (2 * np.pi)
    common_delay = wrap_delay(pooled_delay - pooled_delay[1], alias_period_ns)
    common_phase = np.angle(np.nansum(group_weight * np.exp(1j * by_pol_phase), axis=0))
    common_phase[1] = 0.0
    cable_delay = wrap_delay(delay_ns - common_delay[None, fpga], alias_period_ns)
    receptor_phase = np.angle(np.exp(1j * (phase_f0 - group_phase[:, 1, None] - common_phase[None, fpga])))
    sample_period_ns = 1080.0
    nearest_samples = np.rint(common_delay / sample_period_ns).astype(int)
    report = {
        'reference_fpga': 1, 'reference_antenna': args.reference_antenna,
        'gauge': 'FPGA 1 reference; both hands pooled for one FPGA term; hand-dependent residual assigned to receptors',
        'phase_caveat': 'parallel-hand gain phases have independent gauges; shared FPGA phase is imposed, not independently identifiable without cross-hands or external timing',
        'delay_caveat': 'group-associated gain slope is not a physical post-alignment FPGA clock delay; nearest-sample decomposition is arithmetic only',
        'sample_period_ns': sample_period_ns,
        'nearest_sample_multiple_of_gain_slope': nearest_samples.tolist(),
        'gain_slope_remainder_ns_after_nearest_sample': (common_delay-nearest_samples*sample_period_ns).tolist(),
        'delay_alias_period_ns': alias_period_ns,
        'fit_method': 'circular delay search; one nuisance phase per coarse frequency bin',
        'frequency_mhz': [float(freq.min()), float(freq.max())],
        'fpga_common_delay_ns': common_delay.tolist(),
        'fpga_common_phase_rad': common_phase.tolist(),
        'fpga_group_diagnostic_delay_ns_by_pol': {
            pol: by_pol_delay[pi].tolist() for pi, pol in enumerate(('XX', 'YY'))},
        'fpga_group_diagnostic_phase_rad_by_pol': {
            pol: by_pol_phase[pi].tolist() for pi, pol in enumerate(('XX', 'YY'))},
        'antenna_terms': [
            {'antenna': int(antenna), 'fpga': int(fpga[ai]), 'polarization': pol,
             'relative_delay_ns': float(delay_ns[pi, ai]),
             'within_fpga_delay_ns': float(cable_delay[pi, ai]),
             'within_fpga_phase_rad': float(receptor_phase[pi, ai]),
             'within_coarse_phase_coherence': float(coherence[pi, ai]),
             'phase_residual_median_deg': float(scatter_deg[pi, ai])}
            for ai, antenna in enumerate(order) for pi, pol in enumerate(('XX', 'YY'))
            if np.isfinite(delay_ns[pi, ai])],
    }
    (args.output_dir / 'delay_terms.json').write_text(json.dumps(report, indent=2) + '\n')

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    colors = ('tab:blue', 'tab:orange')
    axes[0, 0].plot(range(4), common_delay, 'ko-', label='Shared XX+YY term')
    axes[0, 1].plot(range(4), np.rad2deg(common_phase), 'ko-', label='Shared XX+YY gauge')
    for pi, pol in enumerate(('XX', 'YY')):
        axes[0, 0].plot(range(4), by_pol_delay[pi], 'o:', color=colors[pi], alpha=.55, label=pol+' diagnostic')
        axes[0, 1].plot(range(4), np.rad2deg(by_pol_phase[pi]), 'o:', color=colors[pi], alpha=.55, label=pol+' diagnostic')
        axes[1, 0].scatter(order + (-.12 if pi == 0 else .12), cable_delay[pi],
                           s=20, color=colors[pi], label=pol)
        axes[1, 1].scatter(order + (-.12 if pi == 0 else .12), scatter_deg[pi],
                           s=20, color=colors[pi], label=pol)
    axes[0, 0].set(title='FPGA-associated gain slope, shared fit across hands', ylabel='Equivalent phase slope (ns), not residual clock')
    axes[0, 1].set(title=f'Group-associated gain phase at {f0:.1f} MHz (gauge)', ylabel='Phase relative to FPGA 1 (deg)')
    axes[1, 0].set(title='Per-receptor slope after FPGA median', xlabel='Physical antenna', ylabel='Residual delay (ns)')
    axes[1, 1].set(title='Phase left after slope and coarse phase fit', xlabel='Physical antenna', ylabel='Median absolute residual (deg)')
    for ax in axes.flat:
        ax.grid(alpha=.3)
        ax.legend()
    for ax in axes[0]:
        ax.set_xticks(range(4)); ax.set_xlabel('FPGA')
    fig.suptitle('Solar gain-phase decomposition (FPGA 1 reference); physical clock residual not isolated')
    fig.savefig(args.output_dir / 'delay_terms.png', dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True, constrained_layout=True)
    representatives = {board: (int(order[max(
        (i for i in range(len(order)) if fpga[i] == board and np.isfinite(coherence[1, i])),
        key=lambda i: coherence[1, i])])) for board in range(4)}
    for board, antenna in representatives.items():
        if antenna is None:
            continue
        for pi, pol in enumerate(('XX', 'YY')):
            trace = phase_traces.get((pi, antenna))
            if trace is None:
                continue
            f, residual, objective = trace
            axes[pi, board].plot(f, np.rad2deg(residual), '.', ms=2, label='phase residual')
            axes[pi, board].set(title=f'FPGA {board}, A{antenna} {pol}', ylabel='Residual phase (deg)')
            axes[pi, board].grid(alpha=.3)
            axes[pi, board].set_xlabel('Frequency (MHz)')
            axes[pi, board].legend(fontsize=8)
    fig.suptitle(f'Phase remaining after fitted slope and per-coarse offsets, relative to A{args.reference_antenna}')
    fig.savefig(args.output_dir / 'phase_slopes.png', dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True, sharey=True,
                             constrained_layout=True)
    for board, antenna in representatives.items():
        for pi, pol in enumerate(('XX', 'YY')):
            trace = phase_traces.get((pi, antenna))
            if trace is None:
                continue
            axes[pi, board].plot(trial_delay_ns / 1000, trace[2])
            axes[pi, board].axvline(delay_ns[pi, np.where(order == antenna)[0][0]] / 1000,
                                    color='tab:red', ls='--')
            axes[pi, board].set(title=f'FPGA {board}, A{antenna} {pol}',
                                xlabel='Trial delay (microseconds)', ylabel='Within-coarse coherence')
            axes[pi, board].grid(alpha=.3)
    fig.suptitle(f'Circular delay likelihoods; aliases repeat every {alias_period_ns / 1000:.2f} microseconds')
    fig.savefig(args.output_dir / 'delay_likelihoods.png', dpi=170)
    plt.close(fig)
    print('Wrote', args.output_dir / 'delay_terms.json',
          args.output_dir / 'delay_terms.png', args.output_dir / 'phase_slopes.png',
          args.output_dir / 'delay_likelihoods.png')
    print('Shared FPGA relative delay ns:', report['fpga_common_delay_ns'])


if __name__ == '__main__':
    main()
