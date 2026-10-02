#!/usr/bin/env python3
"""Replace an MS MODEL_DATA column with an off-axis point-source component."""

import argparse
from pathlib import Path

from casatasks import ft
from casatools import componentlist


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurement_set", type=Path)
    parser.add_argument("component_list", type=Path)
    parser.add_argument("--direction", default="J2000 05h34m31.94s +22d00m52.2s")
    parser.add_argument("--flux-jy", type=float, default=1300.0)
    parser.add_argument("--reference-frequency", default="240.234375MHz")
    parser.add_argument("--spectral-index", type=float, default=-0.27)
    args = parser.parse_args()
    if args.component_list.exists():
        parser.error(f"refusing to overwrite {args.component_list}")

    components = componentlist()
    try:
        components.addcomponent(
            dir=args.direction,
            flux=args.flux_jy,
            fluxunit="Jy",
            freq=args.reference_frequency,
            shape="point",
            spectrumtype="spectral index",
            index=args.spectral_index,
        )
        components.rename(str(args.component_list))
    finally:
        components.close()
    ft(
        vis=str(args.measurement_set),
        complist=str(args.component_list),
        usescratch=True,
        incremental=False,
    )
    print(f"wrote Tau A point-source MODEL_DATA in {args.measurement_set}")


if __name__ == "__main__":
    main()
