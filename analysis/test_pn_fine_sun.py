#!/usr/bin/env python3
"""Offline test of PN local delay across fine bins using moving Sun visibilities.

For each selected fine bin, fit and subtract a stationary component on each
baseline: V(t) = A * SunFringe(t) + C. The centre-bin sky phase is the second
calibration stage. Test whether an independent PN local delay predicts the
off-centre Sun coefficients better than centre phase alone. No HDF5 is edited.
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

from test_pn_then_sun import sun_coefficient

C = 299792458.0
SAMPLE_SECONDS = 1.08e-6
CHANNELS = np.arange(400, 408)
LOCAL_BINS = np.array([2, 6, 10, 13, 16, 20, 24])  # among retained raw bins 3..29
CENTRE_LOCAL = 3


def pn_local_delays(path, mapping):
    with np.load(path) as z:
        fit = z["fit_delay_ns"]
        if not np.array_equal(z["channels"], CHANNELS):
            raise ValueError("PN fit must cover channels 400-407")
    # Use the longer, independent third toggle (run index 1), and the six
    # channels with the cleanest selected comb teeth.
    tau = np.full(80, np.nan)
    for raw in range(80):
        tau[raw] = 0 if raw == 2 else np.nanmedian(fit[1, :6, :, raw])
    return {int(stream["antenna_id"]): tau[int(board) * 20 + raw] * 1e-9
            for board, fpga in mapping.items()
            for raw, stream in enumerate(fpga["streams"])
            if stream["antenna_id"] > 0 and stream["polarization"] == 0}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("capture", type=Path)
    ap.add_argument("timing_pcap", type=Path)
    ap.add_argument("pn_multitone", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--config", type=Path, default=Path("config.json"))
    ap.add_argument("--mapping", type=Path, default=Path("stream_antenna_map.json"))
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    iers.conf.auto_download = False
    iers.conf.auto_max_age = None
    config = json.loads(args.config.read_text())
    mapping = json.loads(args.mapping.read_text())["fpgas"]
    tau = pn_local_delays(args.pn_multitone, mapping)
    pos = {int(k): np.array([v[a] for a in ("east", "north", "up")])
           for fpga in config["antenna_positions"].values() for k, v in fpga.items()}
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
                           and a in tau and b in tau and a in pos and b in pos
                           and np.linalg.norm(pos[b] - pos[a]) >= 7])
        pairs = all_pairs[chosen]
        baseline = np.array([pos[b] - pos[a] for a, b in pairs])
        retained = int(h5.attrs["fine_channels_per_coarse"])
        raw0 = int(h5.attrs["first_raw_fine_bin"])
        if retained != 27 or raw0 != 3:
            raise ValueError("Expected 27 retained raw fine bins 3..29")
        flat_bins = (np.arange(8)[:, None] * retained + LOCAL_BINS[None, :]).ravel()
        vis = np.empty((len(times), 8, len(LOCAL_BINS), len(chosen)), np.complex64)
        ds = h5["visibilities"]
        for start in range(0, len(times), ds.chunks[0]):
            stop = min(start + ds.chunks[0], len(times))
            block = ds[start:stop][:, flat_bins][:, :, chosen, 0, 0, :]
            vis[start:stop] = (block[..., 0] + 1j * block[..., 1]).reshape(
                stop - start, 8, len(LOCAL_BINS), len(chosen))
    at = Time(times, format="unix")
    sun = get_sun(at).transform_to(AltAz(obstime=at, location=site))
    direction = np.stack((np.cos(sun.alt.rad) * np.sin(sun.az.rad),
                          np.cos(sun.alt.rad) * np.cos(sun.az.rad),
                          np.sin(sun.alt.rad)), axis=-1)
    path = direction @ baseline.T
    fine_offset = ((LOCAL_BINS + raw0) - 16) * 781250.0 / 27.0
    freq = CHANNELS[:, None] * 781250.0 + fine_offset[None, :]
    fringe = np.exp(-2j * np.pi * freq[None, :, :, None] / C *
                    path[:, None, None, :])
    complete = missing[:, 0] == 0
    split = len(times) // 2
    masks = (complete & (np.arange(len(times)) < split),
             complete & (np.arange(len(times)) >= split))
    coefficient = np.full((2, 8, len(LOCAL_BINS), len(chosen)), np.nan + 0j)
    significance = np.full((2, 8, len(LOCAL_BINS), len(chosen)), np.nan)
    modulation = np.full((2, 8, len(LOCAL_BINS), len(chosen)), np.nan)
    for half, mask in enumerate(masks):
        for ci in range(8):
            for fi in range(len(LOCAL_BINS)):
                for bi in range(len(chosen)):
                    a, snr, motion = sun_coefficient(vis[:, ci, fi, bi],
                                                     fringe[:, ci, fi, bi], mask)
                    coefficient[half, ci, fi, bi] = a
                    significance[half, ci, fi, bi] = snr
                    modulation[half, ci, fi, bi] = motion
    baseline_tau = np.array([tau[a] - tau[b] for a, b in pairs])
    phase_step = 2 * np.pi * (fine_offset - fine_offset[CENTRE_LOCAL])[None, :, None] * \
        baseline_tau[None, None, :]
    valid = (significance[0, :, CENTRE_LOCAL, :] >= 3)[:, None, :] & \
            (significance[1] >= 3) & \
            (modulation[0, :, CENTRE_LOCAL, :] >= .05)[:, None, :] & \
            (modulation[1] >= .05)
    valid[:, CENTRE_LOCAL] = False
    scores = {}
    errors = {}
    for stage2, centre_phase in (
        ("train_centre", np.angle(coefficient[0, :, CENTRE_LOCAL, :])),
        ("held_centre", np.angle(coefficient[1, :, CENTRE_LOCAL, :])),
    ):
        for candidate, sign in (("sky_centre_only", 0),
                                ("pn_local_then_sky", +1),
                                ("opposite_pn_control", -1)):
            key = f"{stage2}_{candidate}"
            prediction = centre_phase[:, None, :] + sign * phase_step
            error = np.angle(coefficient[1] * np.exp(-1j * prediction))
            cosine = np.cos(error)
            errors[key] = error
            accepted = cosine[valid]
            scores[key] = {
                "n_predictions": int(accepted.size),
                "n_baselines": int(np.any(valid, axis=(0, 1)).sum()),
                "median_abs_phase_error_rad": float(np.median(abs(error[valid]))) if accepted.size else None,
                "median_phase_cosine": float(np.median(accepted)) if accepted.size else None,
                "mean_phase_cosine": float(np.mean(accepted)) if accepted.size else None,
                "median_cosine_by_fine_offset_hz": [
                    float(np.median(cosine[:, fi][valid[:, fi]]))
                    if valid[:, fi].any() else None
                    for fi in range(len(LOCAL_BINS))],
            }
    report = {
        "capture": str(args.capture), "timing_pcap": str(args.timing_pcap),
        "pn_multitone": str(args.pn_multitone), "clock": clock,
        "sun_altitude_deg": [float(sun.alt.deg[0]), float(sun.alt.deg[-1])],
        "selected_coarse_channels": CHANNELS.tolist(),
        "selected_raw_fine_bins": (LOCAL_BINS + raw0).tolist(),
        "fine_offsets_from_coarse_centre_hz": fine_offset.tolist(),
        "total_integrations": len(times),
        "complete_integrations": int(complete.sum()),
        "regression": "Complex Sun moving fringe plus stationary component, independently in each half",
        "scores": scores,
        "limitation": "Stage-two centre sky phase is fitted per baseline/coarse, not antenna-based. The train-centre result also tests temporal stability; held-centre isolates frequency transfer. A strong result must precede applying correction to visibilities."
    }
    (args.output / "pn_fine_sun_report.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(args.output / "pn_fine_sun.npz", pairs=pairs,
                        frequency_hz=freq, valid=valid, coefficient=coefficient,
                        significance=significance, modulation=modulation,
                        baseline_pn_local_delay_s=baseline_tau,
                        **{f"error_{k}": value for k, value in errors.items()})
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), layout="constrained")
    for axis, stage2 in zip(axes, ("train_centre", "held_centre")):
        for candidate in ("sky_centre_only", "pn_local_then_sky", "opposite_pn_control"):
            value = scores[f"{stage2}_{candidate}"]["median_cosine_by_fine_offset_hz"]
            axis.plot(fine_offset / 1e3,
                      [np.nan if x is None else x for x in value], ".-", label=candidate)
        axis.set(title=f"{stage2.replace('_', ' ')}: PN fine-bin prediction",
                 xlabel="Offset from coarse centre (kHz)",
                 ylabel="Median held-out Sun phase cosine", ylim=(-1, 1))
        axis.grid(alpha=.25)
        axis.legend()
    fig.savefig(args.output / "pn_fine_sun.png", dpi=170)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
