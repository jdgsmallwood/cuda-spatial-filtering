#!/usr/bin/env python3
"""Run a controlled Briggs-robustness/minimum-baseline WSClean sweep."""

import argparse
import os
from pathlib import Path
import subprocess


WSCLEAN = Path(
    "/data/BLACKMESA_1/sma075/lambda-commissioning/.local-tools/root/usr/bin/wsclean"
)
WSCLEAN_LIB = Path(
    "/data/BLACKMESA_1/sma075/lambda-commissioning/.local-tools/root/usr/lib/x86_64-linux-gnu"
)


def robust_tag(value: float) -> str:
    if value == 0:
        return "r0"
    return ("rm" if value < 0 else "rp") + str(abs(value)).replace(".", "p")


def run(directory: Path, pol: str, robust: float, cut_m: float) -> None:
    tag = robust_tag(robust)
    stem = f"wsclean_{pol}_joint4_{tag}_min{cut_m:g}m_clean"
    prefix = directory / stem
    expected = directory / f"{stem}-MFS-residual.fits"
    if expected.exists():
        print(f"skip existing {stem}", flush=True)
        return
    command = [str(WSCLEAN), "-j", "16", "-name", str(prefix),
               "-data-column", "DATA", "-size", "512", "512",
               "-scale", "0.2238deg", "-pol", pol,
               "-weight", "briggs", str(robust),
               "-minuvw-m", str(cut_m),
               "-horizon-mask", "5deg", "-no-reorder",
               "-no-update-model-required"]
    if pol == "yy":
        command += ["-channel-range", "0", "435"]
    command += ["-channels-out", "4", "-join-channels",
                "-fit-spectral-pol", "2", "-niter", "10000",
                "-mgain", "0.8", "-nmiter", "10", "-auto-mask", "5",
                "-auto-threshold", "2", "-multiscale",
                "-multiscale-scales", "0,4,8,16,32",
                str(directory / "sun_10s_legacy.ms")]
    environment = os.environ.copy()
    environment["OPENBLAS_NUM_THREADS"] = "1"
    environment["LD_LIBRARY_PATH"] = str(WSCLEAN_LIB)
    print(f"run {stem}: {' '.join(command)}", flush=True)
    with (directory / f"{stem}.log").open("w") as log:
        subprocess.run(command, env=environment, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    if not expected.exists():
        raise RuntimeError(f"WSClean finished without {expected}")
    print(f"done {stem}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--polarization", choices=("xx", "yy", "both"),
                        default="both")
    args = parser.parse_args()
    pols = ("xx", "yy") if args.polarization == "both" else (args.polarization,)
    for pol in pols:
        for robust, cut_m in ((-.5, 5), (.5, 5), (0, 3), (0, 7)):
            run(args.directory, pol, robust, cut_m)
