#!/usr/bin/env python3
"""Offline two-stage test: PN phases, then sky phase, on held-out Sun data.

The Sun is separated from a constant terrestrial component on each baseline
and coarse channel by fitting V(t) = A * SunFringe(t) + C. The Sun coefficient A
is fit independently in the first and second halves of an observation. A
constant sky phase per baseline is trained on channels 400-403 in the first
half; prediction uses channels 404-407 in the second half. A PN correction is
compared with the same sky fit without PN. Nothing is written to the HDF5.
"""

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, get_sun
from astropy.time import Time
from astropy.utils import iers

from lambda_commissioning.capture_timing import timing_pcap_fpga, visibility_clock
from lambda_commissioning.utils import _LAMBDA_PACKET_DTYPE, get_udp_payload_bytes

C = 299792458.0
SAMPLE_SECONDS = 1.08e-6
CHANNELS = np.arange(400, 408)


def sun_coefficient(vis, fringe, valid):
    """Complex A and empirical significance for V = A*fringe + constant."""
    v = vis[valid]
    m = fringe[valid]
    if len(v) < 10:
        return np.nan + 0j, np.nan, np.nan
    centred = m - np.mean(m)
    denom = np.vdot(centred, centred).real
    if denom < 0.02 * len(v):
        return np.nan + 0j, np.nan, np.nan
    a = np.vdot(centred, v - np.mean(v)) / denom
    residual = v - a * m - np.mean(v - a * m)
    sigma = np.sqrt(np.mean(abs(residual)**2)) / np.sqrt(denom)
    significance = abs(a) / max(sigma, 1e-30)
    return a, significance, denom / len(v)


def pn_antenna_phases(path, mapping):
    with np.load(path) as z:
        phase = z["coarse_mean_phase_rad"]
    phases = np.angle(np.mean(np.exp(1j * phase), axis=0))
    result = {}
    for board in range(4):
        for raw, stream in enumerate(mapping[str(board)]["streams"]):
            if stream["antenna_id"] < 0 or stream["polarization"] != 0:
                continue
            result[int(stream["antenna_id"])] = phases[:, board * 20 + raw]
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("capture", type=Path)
    ap.add_argument("timing_pcap", type=Path)
    ap.add_argument("pn_table", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--config", type=Path, default=Path("config.json"))
    ap.add_argument("--mapping", type=Path, default=Path("stream_antenna_map.json"))
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    iers.conf.auto_download = False
    iers.conf.auto_max_age = None
    config = json.loads(args.config.read_text())
    mapping = json.loads(args.mapping.read_text())["fpgas"]
    pn = pn_antenna_phases(args.pn_table, mapping)
    pos = {int(k): np.array([v[a] for a in ("east", "north", "up")])
           for fpga in config["antenna_positions"].values() for k, v in fpga.items()}
    board_of = {int(k): int(board) for board, fpga in config["antenna_positions"].items()
                for k in fpga}
    site = EarthLocation.from_geodetic(config["array_location"]["longitude_deg"] * u.deg,
                                       config["array_location"]["latitude_deg"] * u.deg,
                                       config["array_location"]["height_m"] * u.m)
    pc_times, payloads = get_udp_payload_bytes(str(args.timing_pcap), packets=12000)
    packets = np.frombuffer(b"".join(payloads), dtype=_LAMBDA_PACKET_DTYPE)
    epoch = float(np.median(np.asarray(pc_times) - packets["seq_no"] * SAMPLE_SECONDS))
    with h5py.File(args.capture) as h5:
        manifest = json.loads(h5["audit/run_manifest_json"][()])
        clock = visibility_clock(manifest, epoch, timing_pcap_fpga(args.timing_pcap))
        seq = h5["vis_seq_nums"][:]
        missing = h5["vis_missing_nums"][:]
        times = clock["epoch_unix"] + (seq[:, 0] + seq[:, 1]) * .5 * SAMPLE_SECONDS
        all_pairs = h5["baseline_antenna_ids"][:].astype(int)
        chosen = np.array([i for i, (a, b) in enumerate(all_pairs)
                           if a != b and a not in (8, 12, 14) and b not in (8, 12, 14)
                           and a in pn and b in pn and a in pos and b in pos
                           and np.linalg.norm(pos[b] - pos[a]) >= 7])
        pairs = all_pairs[chosen]
        baseline = np.array([pos[b] - pos[a] for a, b in pairs])
        retained = int(h5.attrs["fine_channels_per_coarse"])
        centre = 16 - int(h5.attrs["first_raw_fine_bin"])
        bins = np.arange(8) * retained + centre
        vis = np.empty((len(times), 8, len(chosen)), np.complex64)
        ds = h5["visibilities"]
        for start in range(0, len(times), ds.chunks[0]):
            stop = min(start + ds.chunks[0], len(times))
            block = ds[start:stop][:, bins][:, :, chosen, 0, 0, :]
            vis[start:stop] = block[..., 0] + 1j * block[..., 1]
    at = Time(times, format="unix")
    sun = get_sun(at).transform_to(AltAz(obstime=at, location=site))
    direction = np.stack((np.cos(sun.alt.rad) * np.sin(sun.az.rad),
                          np.cos(sun.alt.rad) * np.cos(sun.az.rad),
                          np.sin(sun.alt.rad)), axis=-1)
    path = direction @ baseline.T
    freq = CHANNELS * 781250.0
    # The pilot data favoured this correlator/sky phase convention.
    fringe = np.exp(-2j * np.pi * freq[None, :, None] / C * path[:, None, :])
    complete = missing[:, 0] == 0
    split = len(times) // 2
    train = complete & (np.arange(len(times)) < split)
    held = complete & (np.arange(len(times)) >= split)
    coefficient = np.full((2, 8, len(chosen)), np.nan + 0j)
    significance = np.full((2, 8, len(chosen)), np.nan)
    modulation = np.full((2, 8, len(chosen)), np.nan)
    for group, mask in enumerate((train, held)):
        for ci in range(8):
            for bi in range(len(chosen)):
                a, snr, motion = sun_coefficient(vis[:, ci, bi],
                                                 fringe[:, ci, bi], mask)
                coefficient[group, ci, bi] = a
                significance[group, ci, bi] = snr
                modulation[group, ci, bi] = motion
    pn_baseline = np.array([pn[a] - pn[b] for a, b in pairs]).T
    # Require independent detection in both halves on all scored channels.
    good = np.all(significance[:, 4:] >= 3, axis=(0, 1)) & \
        np.all(significance[0, :4] >= 3, axis=0)
    # Static-terrestrial subtraction can yield large spurious fitted A on
    # poorly modulated baselines, so also require useful Sun fringe motion.
    good &= np.all(modulation[:, 4:] >= .05, axis=(0, 1))
    scores = {}
    per_baseline = {}
    for label, phase_model in (("sky_only", np.zeros_like(pn_baseline)),
                               ("pn_then_sky", pn_baseline),
                               ("pn_conjugate_control", -pn_baseline)):
        train_phase = np.angle(coefficient[0, :4]) - phase_model[:4]
        sky_phase = np.angle(np.mean(np.exp(1j * train_phase), axis=0))
        prediction = sky_phase[None, :] + phase_model[4:]
        error = np.angle(coefficient[1, 4:] * np.exp(-1j * prediction))
        cosine = np.cos(error)
        accepted = cosine[:, good]
        scores[label] = {
            "n_baselines": int(good.sum()),
            "median_abs_phase_error_rad": float(np.median(abs(error[:, good]))) if good.any() else None,
            "median_phase_cosine": float(np.median(accepted)) if good.any() else None,
            "mean_phase_cosine": float(np.mean(accepted)) if good.any() else None,
            "median_phase_cosine_by_channel": np.median(accepted, axis=1).tolist() if good.any() else None,
            "same_fpga_median_cosine": float(np.median(cosine[:, good & (np.array([board_of[a] == board_of[b] for a,b in pairs]))])) if np.any(good & (np.array([board_of[a] == board_of[b] for a,b in pairs]))) else None,
            "cross_fpga_median_cosine": float(np.median(cosine[:, good & (np.array([board_of[a] != board_of[b] for a,b in pairs]))])) if np.any(good & (np.array([board_of[a] != board_of[b] for a,b in pairs]))) else None,
        }
        per_baseline[label] = error
    report = {
        "capture": str(args.capture), "timing_pcap": str(args.timing_pcap),
        "pn_table": str(args.pn_table), "clock": clock,
        "sun_altitude_deg": [float(sun.alt.deg[0]), float(sun.alt.deg[-1])],
        "total_integrations": len(times),
        "complete_integrations": int(complete.sum()),
        "fit": "V(t)=A*SunFringe(t)+C for each baseline and coarse channel; separate first/second half regressions; constant sky phase fitted only on channels 400-403 in first half; scored on channels 404-407 in second half",
        "scores": scores,
        "limitation": "Per-baseline sky phase absorbs spatial calibration but tests only PN frequency transfer. Eight PN channels; not a full 40-channel gain solution. The Sun can be resolved or contaminated on some baselines.",
    }
    (args.output / "pn_sun_two_stage_report.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(args.output / "pn_sun_two_stage.npz", pairs=pairs,
                        channel=CHANNELS, sun_coefficient=coefficient,
                        significance=significance, modulation=modulation,
                        pn_baseline_phase=pn_baseline, accepted_baseline=good,
                        **{f"error_{k}": v for k, v in per_baseline.items()})
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    for label, errors in per_baseline.items():
        if good.any():
            ax.plot(CHANNELS[4:], np.median(np.cos(errors[:, good]), axis=1),
                    ".-", label=label)
    ax.set(xlabel="Held-out coarse channel", ylabel="Median predicted phase cosine",
           title=f"PN then Sun: {int(good.sum())} solar-modulated baselines")
    ax.set_ylim(-1, 1)
    ax.grid(alpha=.25)
    if good.any():
        ax.legend()
    fig.savefig(args.output / "pn_sun_two_stage.png", dpi=170)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
