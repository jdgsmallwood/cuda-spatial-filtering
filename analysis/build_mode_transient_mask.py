#!/usr/bin/env python3
"""Build a conservative full-time transient mask from a mode-tracker result.

A time is rejected if either hand has a strong cross-antenna mode in any
selected frequency. The same time mask is applied to both hands and all
selected frequencies, so a broadband impulse cannot train a sky gain.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("modes", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--coherence-limit", type=float, default=0.02)
    args = parser.parse_args()
    if not 0 < args.coherence_limit < 1:
        parser.error("coherence limit must be in (0, 1)")
    if args.output.exists():
        parser.error(f"refusing to replace {args.output}")
    with np.load(args.modes) as data:
        coherence = data["coherence"]
        integrations = data["integrations"]
        channels = data["channels"]
        frequencies = data["frequency_hz"]
    if coherence.shape != (2, len(channels), len(integrations)):
        raise ValueError("mode axes differ from channel/time keys")
    time_good = np.max(coherence, axis=(0, 1)) <= args.coherence_limit
    good = np.broadcast_to(time_good[:, None],
                           (len(integrations), len(channels))).copy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, integrations=integrations,
                        channels=channels, frequency_hz=frequencies, good=good)
    report = {
        "source": str(args.modes),
        "coherence_limit": args.coherence_limit,
        "definition": "reject a whole integration if either hand at any selected frequency exceeds the cross-antenna coherence limit",
        "sampled_integrations": len(integrations),
        "rejected_integrations": integrations[~time_good].astype(int).tolist(),
        "retained_fraction": float(time_good.mean()),
        "note": "The threshold is an explicit diagnostic choice on this capture; the mask is not an operational RFI detector."
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
