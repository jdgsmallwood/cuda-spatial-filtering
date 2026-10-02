#!/usr/bin/env python3
"""Validate A8/A12 receptor swaps and optionally apply antenna-ID swaps."""

import argparse
import json
from pathlib import Path


def parse_swaps(text: str) -> list[tuple[int, int]]:
    result = []
    for item in filter(None, text.split(",")):
        left, separator, right = item.partition(":")
        if not separator:
            raise ValueError(f"invalid swap {item!r}; expected A:B")
        result.append((int(left), int(right)))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--antenna-id-swaps",
        default="",
        help="Comma-separated identity swaps, e.g. 5:29,11:13",
    )
    parser.add_argument(
        "--physical-xx-first", action="store_true",
        help="Convert the native (YY,XX) correlator hands to (XX,YY), matching physical CASA gain labels",
    )
    args = parser.parse_args()

    document = json.loads(args.input.read_text())
    entries = [
        entry
        for fpga in document["fpgas"].values()
        for entry in fpga["streams"]
    ]

    # The physical A8/A12 wiring reversal is authoritative for every run.
    # For each antenna, even/odd hardware stream polarity must be inverted.
    for antenna in (8, 12):
        matches = [entry for entry in entries if int(entry["antenna_id"]) == antenna]
        if len(matches) != 2:
            raise SystemExit(f"expected exactly two stream entries for antenna {antenna}")
        observed = {(int(e["stream"]) % 2, int(e["polarization"])) for e in matches}
        if observed != {(0, 1), (1, 0)}:
            raise SystemExit(
                f"antenna {antenna} does not have the required X/Y reversal: {observed}"
            )

    swaps = parse_swaps(args.antenna_id_swaps)
    for left, right in swaps:
        left_count = sum(int(entry["antenna_id"]) == left for entry in entries)
        right_count = sum(int(entry["antenna_id"]) == right for entry in entries)
        if left_count != 2 or right_count != 2:
            raise SystemExit(
                f"identity swap {left}:{right} requires exactly two entries per antenna; "
                f"found {left_count} and {right_count}"
            )
        for entry in entries:
            antenna = int(entry["antenna_id"])
            if antenna == left:
                entry["antenna_id"] = right
            elif antenna == right:
                entry["antenna_id"] = left

    if args.physical_xx_first:
        for entry in entries:
            entry["polarization"] = 1 - int(entry["polarization"])
        document["description"] = (
            "Observation stream map: polarization 0 is physical XX and 1 is "
            "physical YY. Antenna identities and receptor wiring corrections "
            "are recorded in pulsar_observation."
        )

    document["pulsar_observation"] = {
        "verified_receptor_swaps": [8, 12],
        "experimental_antenna_identity_swaps": [list(pair) for pair in swaps],
        "source_map": str(args.input.resolve()),
        "physical_polarization_order": ["XX", "YY"] if args.physical_xx_first else ["YY", "XX"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n")
    print(f"wrote {args.output}")
    print("verified receptor swaps: A8, A12")
    print(f"experimental antenna identity swaps: {swaps or 'none'}")
    print("physical polarization order:", document["pulsar_observation"]["physical_polarization_order"])


if __name__ == "__main__":
    main()
