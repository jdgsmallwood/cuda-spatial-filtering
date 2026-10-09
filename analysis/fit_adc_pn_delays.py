#!/usr/bin/env python3
"""Test whether the repeated ADC-box PN phases support a delay correction.

The fit is deliberately held out: fit three comb teeth within each coarse
channel and predict the fourth, then compare the two independent PN-on runs.
No gain table or FPGA setting is changed by this program.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lambda_commissioning.utils import _LAMBDA_PACKET_DTYPE, get_udp_payload_bytes

SAMPLE_SECONDS = 1.08e-6
COARSE_HZ = 781250.0
CHANNELS = np.arange(400, 408)
LINES_HZ = -329589.84375 + 195312.5 * np.arange(4)


def wrap(angle):
    return np.angle(np.exp(1j * angle))


def read_board(path, packet_count):
    _, payloads = get_udp_payload_bytes(str(path), packets=packet_count)
    records = np.frombuffer(b"".join(payloads), dtype=_LAMBDA_PACKET_DTYPE)
    return {int(ch): np.sort(records[records["freq_chan"] == ch], order="seq_no")
            for ch in CHANNELS}


def project(records, delays):
    """Return line projections [coarse, board, line, raw input/pol]."""
    result = np.zeros((len(CHANNELS), 4, len(LINES_HZ), 20), np.complex128)
    spans = []
    for ci, channel in enumerate(CHANNELS):
        counter = [board[channel]["seq_no"].astype(np.int64) - delays[bi]
                   for bi, board in enumerate(records)]
        lo = max(n.min() for n in counter)
        hi = min(n.max() for n in counter)
        spans.append((hi - lo) * SAMPLE_SECONDS)
        if hi - lo < 30000:
            raise ValueError(f"Insufficient overlap on channel {channel}")
        for bi, board in enumerate(records):
            r = board[channel]
            keep = (counter[bi] >= lo) & (counter[bi] < hi)
            samples = r["samples"][keep].astype(np.float32)
            volts = (samples[..., 0] + 1j * samples[..., 1])
            volts *= r["scales"][keep, None, :, :]
            volts = volts.reshape(-1, 20)
            indices = (counter[bi][keep, None] + np.arange(64) - lo).ravel()
            weight = np.exp(-2j * np.pi * LINES_HZ[:, None] *
                            SAMPLE_SECONDS * indices[None, :])
            result[ci, bi] = weight @ volts
    return result, spans


def fit_three_predict_four(phase, amplitude):
    """Circular delay grid, per coarse; returns held-out errors and delays."""
    delays_ns = np.arange(-1000., 1000.01, 1.)
    omega = 2 * np.pi * LINES_HZ
    errors = np.full((len(CHANNELS), 4, 80), np.nan)
    constant_errors = np.full((len(CHANNELS), 4, 80), np.nan)
    fits = np.full((len(CHANNELS), 4, 80), np.nan)
    for ci in range(len(CHANNELS)):
        for inp in range(80):
            if inp == 2:
                continue
            p = phase[ci, :, inp]
            a = amplitude[ci, :, inp]
            if not np.all(np.isfinite(p)) or np.min(a) <= 0:
                continue
            for held in range(4):
                train = np.arange(4) != held
                # Circular intercept is fit analytically for every trial delay.
                residual = np.exp(1j * (p[train, None] -
                                        omega[train, None] * delays_ns[None, :] * 1e-9))
                coherence = abs(residual.mean(axis=0))
                di = int(np.argmax(coherence))
                tau = delays_ns[di]
                intercept = np.angle(residual[:, di].mean())
                errors[ci, held, inp] = abs(wrap(p[held] - intercept -
                                                 omega[held] * tau * 1e-9))
                constant = np.angle(np.mean(np.exp(1j * p[train])))
                constant_errors[ci, held, inp] = abs(wrap(p[held] - constant))
                fits[ci, held, inp] = tau
    return errors, fits, constant_errors


def fit_coarse_delay(phase):
    """Fit the first four coarse centres, predict the remaining centres."""
    mean = np.angle(np.mean(np.exp(1j * phase), axis=1))
    freqs = (CHANNELS - CHANNELS[0]) * COARSE_HZ
    trial_ns = np.arange(-640., 640.01, 1.)
    errors = np.full((len(CHANNELS) - 4, 80), np.nan)
    constant_errors = np.full((len(CHANNELS) - 4, 80), np.nan)
    delays = np.full(80, np.nan)
    intercepts = np.full(80, np.nan)
    for inp in range(80):
        if inp == 2:
            continue
        residual = np.exp(1j * (mean[:4, inp, None] -
                                2 * np.pi * freqs[:4, None] * trial_ns[None, :] * 1e-9))
        di = int(np.argmax(abs(residual.mean(axis=0))))
        tau = trial_ns[di]
        intercept = np.angle(residual[:, di].mean())
        errors[:, inp] = abs(wrap(mean[4:, inp] - intercept -
                                 2 * np.pi * freqs[4:] * tau * 1e-9))
        constant = np.angle(np.mean(np.exp(1j * mean[:4, inp])))
        constant_errors[:, inp] = abs(wrap(mean[4:, inp] - constant))
        delays[inp] = tau
        intercepts[inp] = intercept
    return mean, errors, delays, intercepts, constant_errors


def main():
    global CHANNELS
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pcap_root", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--delays", type=Path, required=True)
    ap.add_argument("--first-label", default="all_on")
    ap.add_argument("--second-label", default="all_on_repeat")
    ap.add_argument("--first-packets", type=int, default=20000)
    ap.add_argument("--second-packets", type=int, default=20000)
    ap.add_argument("--min-channel", type=int, default=400)
    ap.add_argument("--channel-count", type=int, default=8,
                    help="Number of consecutive captured coarse channels to analyze")
    args = ap.parse_args()
    if args.channel_count < 8:
        ap.error("--channel-count must be at least 8")
    CHANNELS = np.arange(args.min_channel, args.min_channel + args.channel_count)
    args.output.mkdir(parents=True, exist_ok=True)
    delay_dict = json.loads(args.delays.read_text())
    delays = [int(delay_dict[str(i)]) for i in range(4)]
    projections = []
    spans = []
    for label, packet_count in ((args.first_label, args.first_packets),
                                (args.second_label, args.second_packets)):
        records = [read_board(args.pcap_root / label / f"fpga{i}.pcapng", packet_count)
                   for i in range(4)]
        value, span = project(records, delays)
        projections.append(value.reshape(len(CHANNELS), 4, 4, 20)
                           .transpose(0, 2, 1, 3)
                           .reshape(len(CHANNELS), 4, 80))
        spans.append(span)
    projections = np.asarray(projections)
    # Same line in same coarse channel on FPGA0/input1/X is the phase origin.
    ref = projections[:, :, :, 2]
    relative = projections / ref[..., None]
    phases = np.angle(relative)
    amplitude = abs(projections)
    repeat_error = abs(wrap(phases[1] - phases[0]))
    cv = []
    fit = []
    local_constant = []
    for ri in range(2):
        e, tau, const = fit_three_predict_four(phases[ri], amplitude[ri])
        cv.append(e)
        fit.append(tau)
        local_constant.append(const)
    cv = np.asarray(cv)
    fit = np.asarray(fit)
    local_constant = np.asarray(local_constant)
    coarse = [fit_coarse_delay(phases[ri]) for ri in range(2)]
    coarse_mean = np.asarray([x[0] for x in coarse])
    coarse_cv = np.asarray([x[1] for x in coarse])
    coarse_tau = np.asarray([x[2] for x in coarse])
    constant_cv = np.asarray([x[4] for x in coarse])
    freq_offset = (CHANNELS[4:] - CHANNELS[0])[:, None] * COARSE_HZ
    cross_run_error = abs(wrap(coarse[1][0][4:] - coarse[0][3][None, :] -
                               2 * np.pi * freq_offset * coarse_tau[0][None, :] * 1e-9))
    # An independent-run test of the delay estimator, rather than the phase.
    delay_repeat = abs(np.median(fit[1], axis=1) - np.median(fit[0], axis=1))
    np.savez_compressed(args.output / "pn_multitone.npz", channels=CHANNELS,
                        line_offsets_hz=LINES_HZ, relative_phase_rad=phases,
                        line_amplitude=amplitude, repeat_error_rad=repeat_error,
                        cv_error_rad=cv, fit_delay_ns=fit,
                        local_constant_cv_error_rad=local_constant,
                        coarse_mean_phase_rad=coarse_mean,
                        coarse_cv_error_rad=coarse_cv,
                        coarse_fit_delay_ns=coarse_tau,
                        constant_cv_error_rad=constant_cv,
                        cross_run_held_out_error_rad=cross_run_error)
    good = np.arange(80) != 2
    local_median = np.full(80, np.nan)
    local_median[good] = np.nanmedian(fit[0, :, :, good], axis=(1, 2))
    coarse_first = coarse_tau[0]
    result = {
        "channels": CHANNELS.tolist(), "line_offsets_hz": LINES_HZ.tolist(),
        "fit_first_four_coarse_channels_only": True,
        "common_counter_span_s": spans,
        "single_tone_repeatability_rad": {
            "median": float(np.median(repeat_error[..., good])),
            "p90": float(np.quantile(repeat_error[..., good], .9)),
            "by_channel_median": np.median(repeat_error[..., good], axis=(1, 2)).tolist(),
        },
        "three_tone_delay_predicts_held_out_fourth_rad": {
            "median_by_run": [float(np.nanmedian(x)) for x in cv],
            "p90_by_run": [float(np.nanquantile(x, .9)) for x in cv],
            "constant_phase_median_by_run": [float(np.nanmedian(x)) for x in local_constant],
            "by_channel_median": np.nanmedian(cv, axis=(2, 3)).tolist(),
            "median_local_delay_ns_by_fpga_first_run": [float(np.nanmedian(fit[0, :, :, b*20:(b+1)*20])) for b in range(4)],
        },
        "fitted_delay_repeatability_ns": {
            "median": float(np.nanmedian(delay_repeat)),
            "p90": float(np.nanquantile(delay_repeat, .9)),
            "max": float(np.nanmax(delay_repeat)),
        },
        "coarse_channel_delay": {
            "fit_channels": CHANNELS[:4].tolist(),
            "held_out_channels": CHANNELS[4:].tolist(),
            "delay_repeatability_ns_median": float(np.nanmedian(abs(coarse_tau[1] - coarse_tau[0]))),
            "delay_repeatability_ns_p90": float(np.nanquantile(abs(coarse_tau[1] - coarse_tau[0]), .9)),
            "delay_repeatability_modulo_1280ns_median": float(np.nanmedian(abs((coarse_tau[1] - coarse_tau[0] + 640) % 1280 - 640))),
            "delay_repeatability_modulo_1280ns_p90": float(np.nanquantile(abs((coarse_tau[1] - coarse_tau[0] + 640) % 1280 - 640), .9)),
            "held_out_error_rad_median_by_run": np.nanmedian(coarse_cv, axis=(1, 2)).tolist(),
            "cross_run_held_out_error_rad_median": float(np.nanmedian(cross_run_error)),
            "held_out_constant_phase_error_rad_median_by_run": np.nanmedian(constant_cv, axis=(1, 2)).tolist(),
            "held_out_error_rad_median_by_channel": np.nanmedian(coarse_cv, axis=(0, 2)).tolist(),
            "held_out_const_error_rad_median_by_channel": np.nanmedian(constant_cv, axis=(0, 2)).tolist(),
            "median_abs_delay_ns_by_fpga_first_run": [float(np.nanmedian(abs(coarse_tau[0, b*20:(b+1)*20]))) for b in range(4)],
            "median_signed_delay_ns_by_fpga_first_run": [float(np.nanmedian(coarse_tau[0, b*20:(b+1)*20])) for b in range(4)],
            "held_out_error_rad_median_by_fpga_first_run": [float(np.nanmedian(coarse_cv[0, :, b*20:(b+1)*20])) for b in range(4)],
            "held_out_const_error_rad_median_by_fpga_first_run": [float(np.nanmedian(constant_cv[0, :, b*20:(b+1)*20])) for b in range(4)],
        },
        "delay_fit_range_ns": [-1000, 1000],
        "caveat": "Each coarse is fit independently. A good local delay does not establish a single smooth full-band delay or diagnose the antenna path.",
    }
    (args.output / "multitone_report.json").write_text(json.dumps(result, indent=2) + "\n")
    fig, ax = plt.subplots(3, 1, figsize=(12, 11), layout="constrained")
    for ch in range(len(CHANNELS)):
        ax[0].scatter(np.full(4 * 79, CHANNELS[ch]) +
                      np.repeat(np.arange(4) - 1.5, 79) * .12,
                      repeat_error[ch, :, good].ravel(), s=2, alpha=.25)
    ax[0].set(title="Independent PN-on toggles: same-tone phase repeatability",
              xlabel="Coarse channel", ylabel="Absolute phase change (rad)")
    ax[0].grid(alpha=.25)
    for ri, label in enumerate(("First PN-on run", "Second PN-on run")):
        ax[1].plot(CHANNELS, np.nanmedian(cv[ri], axis=(1, 2)), ".-", label=label)
    ax[1].set(title="Fit a delay to 3 comb teeth; predict the 4th",
              xlabel="Coarse channel", ylabel="Median held-out error (rad)")
    ax[1].grid(alpha=.25)
    ax[1].legend()
    for inp, name in ((4, "FPGA0 receiver 2/raw pol 0"),
                      (20, "FPGA1 receiver 0/raw pol 0"),
                      (40, "FPGA2 receiver 0/raw pol 0"),
                      (60, "FPGA3 receiver 0/raw pol 0")):
        ax[2].plot(CHANNELS, np.unwrap(coarse_mean[0, :, inp]), ".-", label=name)
    ax[2].axvline(CHANNELS[3] + .5, color="0.5", ls="--", label="fit / held-out")
    ax[2].set(title="Example coarse-centre phase tracks relative to FPGA0 receiver 1/raw pol 0",
              xlabel="Coarse channel", ylabel="Unwrapped relative phase (rad)")
    ax[2].grid(alpha=.25)
    ax[2].legend(ncol=2)
    fig.savefig(args.output / "pn_multitone_delay_check.png", dpi=170)
    fig2, ax2 = plt.subplots(figsize=(7, 6), layout="constrained")
    for board in range(4):
        sl = slice(board * 20, (board + 1) * 20)
        ax2.scatter(coarse_first[sl], local_median[sl], s=22,
                    label=f"FPGA {board}")
    ax2.plot([-650, 650], [-650, 650], color="0.5", ls="--",
             label="One common physical delay")
    ax2.set(xlabel="Slope across coarse centres (ns, modulo 1280 ns)",
            ylabel="Slope within coarse channel (ns)",
            title="The two PN delay estimators disagree systematically",
            xlim=(-650, 650), ylim=(-650, 650))
    ax2.grid(alpha=.25)
    ax2.legend()
    fig2.savefig(args.output / "pn_coarse_vs_local_delay.png", dpi=170)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
