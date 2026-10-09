#!/usr/bin/env python3
"""Test a CASA phase-only G correction on already Sun-calibrated visibilities."""

import argparse
from pathlib import Path

import numpy as np
from casatasks import applycal, gaincal
from casatools import table


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("ms", type=Path)
    p.add_argument("caltable", type=Path)
    p.add_argument("--timerange", required=True)
    p.add_argument("--uvrange", default=">7m")
    p.add_argument("--refant", default="A2")
    args = p.parse_args()
    if args.caltable.exists():
        raise FileExistsError(args.caltable)
    gaincal(vis=str(args.ms), caltable=str(args.caltable),
            gaintype="G", calmode="p", solint="inf", refant=args.refant,
            timerange=args.timerange, uvrange=args.uvrange, minsnr=3)
    t = table()
    t.open(str(args.caltable))
    try:
        gains, flags = t.getcol("CPARAM"), t.getcol("FLAG")
    finally:
        t.close()
    if not np.any(~flags) or not np.isfinite(gains[~flags]).all():
        raise RuntimeError("CASA returned no finite unflagged gains")
    applycal(vis=str(args.ms), gaintable=[str(args.caltable)],
             interp=["linear"], calwt=False, applymode="calflag")
    print(f"wrote and applied {args.caltable}; flags={float(flags.mean()):.4f}")


if __name__ == "__main__":
    main()
