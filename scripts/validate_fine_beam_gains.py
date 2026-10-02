#!/usr/bin/env python3
"""Validate the strict flattened-fine-channel beam-gain JSON schema."""

import argparse
import json
import math
from pathlib import Path


def comma_ints(value: str) -> set[int]:
    return {int(item) for item in value.split(",") if item}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gains", type=Path)
    parser.add_argument("--stream-map", type=Path, required=True)
    parser.add_argument("--min-channel", type=int, default=176)
    parser.add_argument("--coarse-channels", type=int)
    parser.add_argument("--processing-channels", type=int, default=1120)
    parser.add_argument("--excluded-antennas", default="14")
    parser.add_argument("--channel-mask", type=Path,
                        help="Required frequency mask for a calibration with rejected channels")
    args = parser.parse_args()

    gains = json.loads(args.gains.read_text())
    if gains.get("channel_axis") != "fine_processing_index":
        raise SystemExit('channel_axis must equal "fine_processing_index"')
    weights = gains.get("weights")
    if not isinstance(weights, dict):
        raise SystemExit("weights must be a JSON object")

    provenance = gains.get("provenance")
    if not isinstance(provenance, dict):
        raise SystemExit("gains must contain frequency provenance")
    if provenance.get("channel_mask_required") and not args.channel_mask:
        raise SystemExit("this calibration requires --channel-mask; rejected channels must not be summed")
    if args.channel_mask:
        mask = json.loads(args.channel_mask.read_text())
        frequencies, good = mask.get("frequency_hz", []), mask.get("good", [])
        if (len(frequencies) != provenance.get("unique_frequency_channels")
                or len(good) != len(frequencies) or not any(good)
                or not all(isinstance(value, bool) for value in good)):
            raise SystemExit("invalid or incompatible calibration channel mask")
        first, last = provenance["frequency_first_hz"], provenance["frequency_last_hz"]
        for i, frequency in enumerate(frequencies):
            expected = first + i * (last-first)/(len(frequencies)-1)
            if not math.isfinite(frequency) or abs(frequency-expected) > 1:
                raise SystemExit("channel mask frequency grid does not match the gain file")
    expected_processing = args.processing_channels
    recorded_processing = provenance.get("processing_channels_with_boundary_duplicates")
    if recorded_processing != expected_processing:
        raise SystemExit(
            "gain provenance processing-channel mismatch: "
            f"recorded {recorded_processing}, requested {expected_processing}"
        )
    if args.coarse_channels is not None:
        expected_unique = args.coarse_channels * 27
        recorded_unique = provenance.get("unique_frequency_channels")
        if recorded_unique != expected_unique:
            raise SystemExit(
                "gain provenance unique-frequency mismatch: "
                f"recorded {recorded_unique}, expected {expected_unique}"
            )
        # The DADA/science axis keeps original GPU bins 3..29: offsets -13..13
        # fine-bin spacings about every 0.78125-MHz coarse centre.
        channel_width_hz = 781_250.0
        fine_step_hz = channel_width_hz / 27
        expected_first = args.min_channel * channel_width_hz - 13 * fine_step_hz
        expected_last = (
            (args.min_channel + args.coarse_channels - 1) * channel_width_hz
            + 13 * fine_step_hz
        )
        for label, expected in (
            ("frequency_first_hz", expected_first),
            ("frequency_last_hz", expected_last),
        ):
            try:
                recorded = float(provenance[label])
            except (KeyError, TypeError, ValueError):
                raise SystemExit(f"gain provenance is missing valid {label}") from None
            if not math.isclose(recorded, expected, rel_tol=0.0, abs_tol=1.0):
                raise SystemExit(
                    f"gain provenance {label} mismatch: recorded {recorded}, "
                    f"expected {expected} for min channel {args.min_channel}"
                )

    stream_map = json.loads(args.stream_map.read_text())
    antennas = {
        int(entry["antenna_id"])
        for fpga in stream_map["fpgas"].values()
        for entry in fpga["streams"]
        if int(entry["antenna_id"]) >= 0
    }
    excluded = comma_ints(args.excluded_antennas)
    required_antennas = sorted(antennas - excluded)
    excluded_receptors = {
        (int(entry["antenna_id"]), str(entry["polarization"]))
        for entry in gains.get("excluded_receptors", [])
    }

    errors = []
    for offset in range(args.processing_channels):
        key = str(args.min_channel + offset)
        channel = weights.get(key)
        if not isinstance(channel, dict):
            errors.append(f"missing channel {key}")
            continue
        for pol in ("XX", "YY"):
            pol_data = channel.get(pol)
            if not isinstance(pol_data, dict):
                errors.append(f"missing channel {key} pol {pol}")
                continue
            for antenna in required_antennas:
                if (antenna, pol) in excluded_receptors:
                    continue
                value = pol_data.get(str(antenna))
                if not isinstance(value, dict):
                    errors.append(f"missing channel {key} pol {pol} antenna {antenna}")
                    continue
                try:
                    real = float(value["real"])
                    imag = float(value["imag"])
                except (KeyError, TypeError, ValueError):
                    errors.append(f"invalid channel {key} pol {pol} antenna {antenna}")
                    continue
                if not math.isfinite(real) or not math.isfinite(imag) or real * real + imag * imag < 1e-12:
                    errors.append(f"zero/non-finite channel {key} pol {pol} antenna {antenna}")

    if errors:
        preview = "\n".join(errors[:20])
        raise SystemExit(f"fine gain validation failed ({len(errors)} errors):\n{preview}")
    print(
        f"valid: {args.processing_channels} processing channels, 2 pols, "
        f"{len(required_antennas)} included antennas; excluded={sorted(excluded)}; "
        f"excluded_receptors={sorted(excluded_receptors)}"
    )


if __name__ == "__main__":
    main()
