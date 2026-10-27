#!/usr/bin/env python3
"""Export CASA K/B/G tables as strict per-fine-channel beam gains.

The JSON contains the *measured voltage* antenna Jones gain. The MS export
conjugates the native covariance, so CASA solutions must be conjugated back
before use on native voltages. pulsar_fold's strict
fine-gain loader conjugate-inverts it before multiplying it into the coherent
beam weights.  CASA K delay values are seconds after conversion from the
FPARAM nanosecond convention, with phase
``+2*pi*delay*(frequency-delay_solution_centroid)``.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from casacore.tables import table
from coarse_pfb_deripple import load_deripple, validate_calibration_capture


def casa_to_voltage_gains(casa_gains: np.ndarray) -> np.ndarray:
    """Undo the covariance conjugation performed by the MS exporter."""
    return np.conj(casa_gains)


def read_table(path: Path, parameter: str) -> dict[str, np.ndarray]:
    with table(str(path), ack=False) as tab:
        return {
            parameter: tab.getcol(parameter),
            "FLAG": tab.getcol("FLAG"),
            "TIME": tab.getcol("TIME"),
            "ANTENNA1": tab.getcol("ANTENNA1"),
        }


def spectral_frequencies(path: Path) -> np.ndarray:
    with table(str(path / "SPECTRAL_WINDOW"), ack=False) as tab:
        return np.asarray(tab.getcol("CHAN_FREQ")[0], dtype=float)


def interpolate_complex_frequency(values: np.ndarray, good: np.ndarray) -> np.ndarray:
    """CASA-like fill of flagged B samples using amplitude/unwrapped phase."""
    x = np.arange(values.size, dtype=float)
    if np.count_nonzero(good) < 2:
        raise ValueError("fewer than two unflagged bandpass channels")
    amplitude = np.interp(x, x[good], np.abs(values[good]))
    phase = np.interp(x, x[good], np.unwrap(np.angle(values[good])))
    return amplitude * np.exp(1j * phase)


def interpolate_complex_time(
    values: np.ndarray, flags: np.ndarray, times: np.ndarray, epoch: float
) -> complex:
    good = ~flags
    if not np.any(good):
        raise ValueError("no unflagged time solution")
    values = values[good]
    times = times[good]
    order = np.argsort(times)
    values, times = values[order], times[order]
    if values.size == 1 or epoch <= times[0]:
        return complex(values[0])
    if epoch >= times[-1]:
        return complex(values[-1])
    upper = int(np.searchsorted(times, epoch))
    lower = upper - 1
    fraction = (epoch - times[lower]) / (times[upper] - times[lower])
    amplitude = (1.0 - fraction) * abs(values[lower]) + fraction * abs(values[upper])
    pair_phase = np.unwrap(np.angle([values[lower], values[upper]]))
    phase = (1.0 - fraction) * pair_phase[0] + fraction * pair_phase[1]
    return complex(amplitude * np.exp(1j * phase))


def scalar_solution(data: dict[str, np.ndarray], parameter: str, antenna: int, pol: int):
    rows = np.flatnonzero(data["ANTENNA1"] == antenna)
    good = rows[~data["FLAG"][rows, 0, pol]]
    if good.size == 0:
        raise ValueError("no unflagged scalar solution")
    return data[parameter][good[0], 0, pol]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration_directory", type=Path,
                        help="directory containing delay.K, bandpass.B, phase.G, gain.G")
    parser.add_argument("output", type=Path)
    parser.add_argument("--min-channel", type=int, default=176)
    parser.add_argument("--coarse-channels", type=int, default=40)
    parser.add_argument("--retained-fine-channels", type=int, default=28)
    parser.add_argument("--exclude-antennas", default="")
    parser.add_argument(
        "--exclude-receptors",
        default="",
        help="comma-separated antenna/polarization receptors, e.g. 1:XX",
    )
    parser.add_argument("--phase-epoch", type=float,
                        help="CASA seconds epoch; default is midpoint of phase.G")
    parser.add_argument("--deripple-config", type=Path,
                        help="runtime PFB table used for the calibration visibility capture; "
                             "records matching provenance (does not alter CASA gains)")
    parser.add_argument("--calibration-visibilities", type=Path,
                        help="Starspin HDF5 used for this CASA solution; required with "
                             "--deripple-config to verify its embedded run audit")
    args = parser.parse_args()

    deripple = None
    if args.deripple_config:
        # The fine channelizer is 32 bins with 2 trimmed at each edge.
        deripple = load_deripple(args.deripple_config, 32, 2)
        if args.retained_fine_channels != 28:
            parser.error("de-ripple export requires 28 retained fine bins")
        if not args.calibration_visibilities:
            parser.error("--deripple-config requires --calibration-visibilities")
        try:
            validate_calibration_capture(args.calibration_visibilities, deripple)
        except (ImportError, OSError, KeyError, ValueError) as error:
            parser.error(f"calibration visibility audit does not match: {error}")

    cal = args.calibration_directory.resolve()
    paths = {name: cal / name for name in ("delay.K", "bandpass.B", "phase.G", "gain.G")}
    for path in paths.values():
        if not path.is_dir():
            parser.error(f"missing CASA table: {path}")

    delay = read_table(paths["delay.K"], "FPARAM")
    bandpass = read_table(paths["bandpass.B"], "CPARAM")
    phase = read_table(paths["phase.G"], "CPARAM")
    gain = read_table(paths["gain.G"], "CPARAM")
    frequencies = spectral_frequencies(paths["bandpass.B"])
    delay_frequency = float(spectral_frequencies(paths["delay.K"])[0])

    expected_unique = args.coarse_channels * (args.retained_fine_channels - 1)
    if frequencies.size != expected_unique:
        parser.error(
            f"bandpass has {frequencies.size} channels, expected {expected_unique}"
        )
    if not np.all(np.diff(frequencies) > 0):
        parser.error("bandpass frequency axis is not strictly ascending")
    fine_step_hz = 781_250.0 / 27.0
    expected_first_hz = args.min_channel * 781_250.0 - 13 * fine_step_hz
    expected_grid_hz = expected_first_hz + np.arange(expected_unique) * fine_step_hz
    if not np.allclose(frequencies, expected_grid_hz, rtol=0, atol=1.0):
        parser.error('bandpass frequency grid does not match original GPU bins 3..29')

    excluded = {int(value) for value in args.exclude_antennas.split(",") if value}
    excluded_receptors = set()
    for value in filter(None, args.exclude_receptors.split(",")):
        antenna, separator, polarization = value.partition(":")
        polarization = polarization.upper()
        if not separator or polarization not in {"XX", "YY"}:
            parser.error(f"invalid excluded receptor {value!r}; expected ANTENNA:XX or ANTENNA:YY")
        excluded_receptors.add((int(antenna), polarization))
    phase_epoch = args.phase_epoch
    if phase_epoch is None:
        phase_epoch = float((np.min(phase["TIME"]) + np.max(phase["TIME"])) / 2.0)

    antenna_indices = sorted({int(value) for value in bandpass["ANTENNA1"]})
    antenna_gains: dict[tuple[int, int], np.ndarray] = {}
    failures = []
    for antenna_index in antenna_indices:
        antenna_id = antenna_index + 1
        if antenna_id in excluded:
            continue
        band_rows = np.flatnonzero(bandpass["ANTENNA1"] == antenna_index)
        if band_rows.size != 1:
            failures.append(f"A{antenna_id}: expected one bandpass row, found {band_rows.size}")
            continue
        row = int(band_rows[0])
        for pol in range(2):
            pol_name = "XX" if pol == 0 else "YY"
            if (antenna_id, pol_name) in excluded_receptors:
                continue
            try:
                b = interpolate_complex_frequency(
                    bandpass["CPARAM"][row, :, pol],
                    ~bandpass["FLAG"][row, :, pol],
                )
                k_seconds = float(scalar_solution(delay, "FPARAM", antenna_index, pol)) * 1e-9
                p_rows = np.flatnonzero(phase["ANTENNA1"] == antenna_index)
                p = interpolate_complex_time(
                    phase["CPARAM"][p_rows, 0, pol],
                    phase["FLAG"][p_rows, 0, pol],
                    phase["TIME"][p_rows],
                    phase_epoch,
                )
                g = complex(scalar_solution(gain, "CPARAM", antenna_index, pol))
                k = np.exp(2j * np.pi * k_seconds * (frequencies - delay_frequency))
                combined = casa_to_voltage_gains(b * p * g * k)
                if not np.all(np.isfinite(combined)) or np.any(np.abs(combined) < 1e-12):
                    raise ValueError("combined solution is zero or non-finite")
                antenna_gains[(antenna_id, pol)] = combined
            except ValueError as error:
                failures.append(f"A{antenna_id} {pol_name}: {error}")

    if failures:
        raise SystemExit("calibration export failed:\n" + "\n".join(failures))

    # The internal GPU axis retains raw bins 2..29. The CASA/DADA science axis
    # keeps raw bins 3..29, or internal k=1..27, for every coarse channel.
    # Internal k=0 is never emitted to DADA; fill its mandatory gain slot from
    # the adjacent k=1 solely to maintain a complete processing weight table.
    weights: dict[str, dict] = {}
    processing_channels = args.coarse_channels * args.retained_fine_channels
    for internal in range(processing_channels):
        coarse, fine = divmod(internal, args.retained_fine_channels)
        unique = coarse * (args.retained_fine_channels - 1) + max(fine - 1, 0)
        channel = {"XX": {}, "YY": {}}
        for (antenna_id, pol), values in antenna_gains.items():
            value = values[unique]
            channel["XX" if pol == 0 else "YY"][str(antenna_id)] = {
                "real": float(value.real),
                "imag": float(value.imag),
            }
        weights[str(args.min_channel + internal)] = channel

    document = {
        "channel_axis": "fine_processing_index",
        "excluded_receptors": [
            {"antenna_id": antenna, "polarization": polarization}
            for antenna, polarization in sorted(excluded_receptors)
        ],
        "weights": weights,
        "provenance": {
            "calibration_directory": str(cal),
            "casa_product": "conj(delay.K * bandpass.B * phase.G * gain.G)",
            "voltage_gain_convention": "native voltage response; conjugate of CASA Jones",
            "required_runtime_polarization_order": ["XX", "YY"],
            "delay_phase_convention": "+2*pi*K_seconds*(frequency-delay_solution_centroid)",
            "phase_epoch_casa_seconds": phase_epoch,
            "unique_frequency_channels": int(frequencies.size),
            "processing_channels_with_boundary_duplicates": processing_channels,
            "frequency_first_hz": float(frequencies[0]),
            "frequency_last_hz": float(frequencies[-1]),
            "excluded_antennas": sorted(excluded),
            "excluded_receptors": [
                f"A{antenna}:{polarization}"
                for antenna, polarization in sorted(excluded_receptors)
            ],
        },
    }
    if deripple is not None:
        document["provenance"]["deripple_response_id"] = deripple["response_id"]
        document["provenance"]["deripple_voltage_gains"] = deripple["voltage_gains"]
        document["provenance"]["deripple_export_state"] = (
            "measured_from_derippled_visibilities"
        )
        document["provenance"]["calibration_visibilities"] = str(
            args.calibration_visibilities.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n")
    print(f"wrote {args.output}")
    print(
        f"{processing_channels} processing bins from {frequencies.size} unique frequencies; "
        f"{len(antenna_gains)} calibrated receptors; excluded antennas={sorted(excluded)}; "
        f"excluded receptors={sorted(excluded_receptors)}"
    )
    print(f"phase epoch (CASA seconds): {phase_epoch:.6f}")


if __name__ == "__main__":
    main()
