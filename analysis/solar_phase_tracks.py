#!/usr/bin/env python3
"""Compare uncalibrated baseline phase motion with the predicted Sun fringe.

Fit only one constant phase per baseline/frequency on the first third. Score
held-out times against the Sun in both visibility conventions and a stationary
control. No antenna gains or hardware delays are changed.
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


def score(data, model, valid, train):
    phasor = np.divide(data, np.maximum(abs(data), 1e-30)) * model.conj()
    reference = np.where(valid & train[:, None, None], phasor, 0).sum(axis=0)
    reference = np.divide(reference, np.maximum(abs(reference), 1e-30))
    held = valid & ~train[:, None, None]
    count = held.sum(axis=0)
    cosine = np.real(phasor * reference.conj())
    result = np.divide(np.where(held, cosine, 0).sum(axis=0),
                       np.maximum(count, 1))
    result[count < 10] = np.nan
    return result, reference


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("capture", type=Path)
    ap.add_argument("timing_pcap", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--config", type=Path, default=Path("config.json"))
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    iers.conf.auto_download = False
    iers.conf.auto_max_age = None
    conf = json.loads(args.config.read_text())
    loc = conf["array_location"]
    site = EarthLocation.from_geodetic(loc["longitude_deg"] * u.deg,
                                       loc["latitude_deg"] * u.deg,
                                       loc["height_m"] * u.m)
    positions = {int(k): np.array([v[a] for a in ("east", "north", "up")])
                 for fpga in conf["antenna_positions"].values()
                 for k, v in fpga.items()}
    boards = {int(k): int(fpga) for fpga, d in conf["antenna_positions"].items()
              for k in d}
    packet_times, payloads = get_udp_payload_bytes(str(args.timing_pcap), packets=12000)
    packets = np.frombuffer(b"".join(payloads), dtype=_LAMBDA_PACKET_DTYPE)
    packet_epoch = float(np.median(np.asarray(packet_times) -
                                   packets["seq_no"] * 1.08e-6))
    with h5py.File(args.capture) as h5:
        manifest = json.loads(h5["audit/run_manifest_json"][()])
        clock = visibility_clock(manifest, packet_epoch,
                                 timing_pcap_fpga(args.timing_pcap))
        seq = h5["vis_seq_nums"][:]
        missing = h5["vis_missing_nums"][:]
        times = clock["epoch_unix"] + (seq[:, 0] + seq[:, 1]) * .5 * 1.08e-6
        pairs = h5["baseline_antenna_ids"][:].astype(int)
        chosen = np.array([i for i, (a, b) in enumerate(pairs)
                           if a != b and a != 14 and b != 14
                           and a in positions and b in positions])
        pairs = pairs[chosen]
        baseline = np.array([positions[b] - positions[a] for a, b in pairs])
        length = np.linalg.norm(baseline, axis=1)
        group = np.array(["same" if boards[a] == boards[b] else "cross"
                          for a, b in pairs])
        ncoarse = int(h5.attrs["max_channel"] - h5.attrs["min_channel"] + 1)
        retained = int(h5.attrs["fine_channels_per_coarse"])
        raw0 = int(h5.attrs["first_raw_fine_bin"])
        centre = 16 - raw0
        bins = np.arange(ncoarse) * retained + centre
        frequency = (int(h5.attrs["min_channel"]) + np.arange(ncoarse)) * 781250.
        vis = np.empty((len(seq), ncoarse, len(chosen), 2), np.complex64)
        ds = h5["visibilities"]
        chunk = ds.chunks[0]
        for start in range(0, len(seq), chunk):
            stop = min(start + chunk, len(seq))
            block = ds[start:stop][:, bins][:, :, chosen]
            for pol, hand in enumerate((1, 0)):
                vis[start:stop, :, :, pol] = (block[:, :, :, hand, hand, 0] +
                                               1j * block[:, :, :, hand, hand, 1])
    at = Time(times, format="unix")
    sun = get_sun(at).transform_to(AltAz(obstime=at, location=site))
    az, alt = sun.az.rad, sun.alt.rad
    direction = np.stack((np.cos(alt) * np.sin(az),
                          np.cos(alt) * np.cos(az), np.sin(alt)), axis=-1)
    path = direction @ baseline.T
    model = np.exp(2j * np.pi * frequency[None, :, None] / C *
                   path[:, None, :]).astype(np.complex64)
    complete = missing[:, 0] == 0
    complete[:3] = False
    train = np.arange(len(seq)) < len(seq) // 3
    summaries = {}
    band_summaries = {}
    fig, axes = plt.subplots(2, 3, figsize=(17, 8), layout="constrained")
    for pol, label in enumerate(("XX", "YY")):
        data = vis[:, :, :, pol]
        valid = complete[:, None, None] & np.isfinite(data) & (abs(data) > 0)
        metrics = {}
        references = {}
        for name, prediction in (("Sun +", model), ("Sun -", model.conj()),
                                 ("stationary", np.ones_like(model))):
            metrics[name], references[name] = score(data, prediction, valid, train)
        summaries[label] = {}
        band_summaries[label] = {}
        for category, mask in (("all", length >= 7),
                               ("same FPGA", (length >= 7) & (group == "same")),
                               ("cross FPGA", (length >= 7) & (group == "cross")),
                               ("antenna 1", (length >= 7) & np.any(pairs == 1, axis=1))):
            summaries[label][category] = {name: float(np.nanmedian(value[:, mask]))
                                          for name, value in metrics.items()}
        for band_name, band_mask in (
            ("coarse 400-415 (312.5-324.2 MHz)", np.arange(ncoarse) < 16),
            ("coarse 416-439 (325.0-343.0 MHz)", np.arange(ncoarse) >= 16),
        ):
            band_summaries[label][band_name] = {
                category: {name: float(np.nanmedian(value[band_mask][:, mask]))
                           for name, value in metrics.items()}
                for category, mask in (
                    ("all", length >= 7),
                    ("same FPGA", (length >= 7) & (group == "same")),
                    ("cross FPGA", (length >= 7) & (group == "cross")),
                    ("antenna 1", (length >= 7) & np.any(pairs == 1, axis=1)),
                )
            }
        ax = axes[pol, 0]
        for name, values in metrics.items():
            ax.plot(frequency / 1e6, np.nanmedian(values[:, length >= 7], axis=1),
                    ".-", label=name)
        ax.set_title(label + " held-out phase cosine (7m+ baselines)")
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylim(-1, 1)
        ax.grid(alpha=.3)
        ax.legend()
        ax.axvline(324.6, color="0.5", ls="--", lw=1)
        # Show the SAME cross-FPGA baseline on a quieter lower channel and a
        # high channel dominated by the stationary component.
        candidate = np.flatnonzero(np.all(pairs == [1, 21], axis=1))
        bi = int(candidate[0]) if len(candidate) else int(
            np.flatnonzero((length > 15) & (group == "cross"))[0])
        clock_min = (times - times[0]) / 60
        for col, fidx in enumerate((4, 32), start=1):
            actual = data[:, fidx, bi]
            positive = model[:, fidx, bi]
            axis = axes[pol, col]
            axis.plot(clock_min[complete],
                      np.unwrap(np.angle(actual[complete])), ".", ms=2,
                      label="measured")
            axis.plot(clock_min,
                      np.unwrap(np.angle(positive *
                                         references["Sun +"][fidx, bi])),
                      label="Sun +")
            axis.plot(clock_min,
                      np.unwrap(np.angle(positive.conj() *
                                         references["Sun -"][fidx, bi])),
                      label="Sun -")
            axis.set_title(f"{label} baseline {pairs[bi, 0]}–{pairs[bi, 1]}, "
                           f"{frequency[fidx] / 1e6:.1f} MHz")
            axis.set_xlabel("Minutes after start")
            axis.set_ylabel("Unwrapped phase (rad)")
            axis.grid(alpha=.3)
            axis.legend()
    fig.savefig(args.output / "solar_phase_tracks.png", dpi=170)
    result = {"capture": str(args.capture), "timing_pcap": str(args.timing_pcap),
              "clock": clock, "sun_altitude_deg": [float(alt[0] * 180 / np.pi),
                                                   float(alt[-1] * 180 / np.pi)],
              "duration_s": float(times[-1] - times[0]),
              "total_integrations": len(seq),
              "complete_integrations": int(complete.sum()),
              "total_missing_packets": float(missing[:, 0].sum()),
              "packet_loss_fraction": float(missing[:, 0].sum() /
                                            max(missing[:, 1].sum(), 1)),
              "summary": summaries,
              "frequency_band_summary": band_summaries,
              "limitation": "A short low-elevation pilot and an extended/variable Sun can dilute phase scores."}
    (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
