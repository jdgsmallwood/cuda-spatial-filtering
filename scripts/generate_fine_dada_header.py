#!/usr/bin/env python3
"""Generate the exact PSRDADA header for compacted Lambda fine channels."""

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

HEADER_BYTES = 4096
COARSE_WIDTH_MHZ = 0.78125
FINE_OVERSAMPLE_DENOMINATOR = 27
INPUT_TSAMP_US = 1.08


def pointing_fields(targets_file):
    targets=json.loads(targets_file.read_text())['targets']
    if len(targets)!=1 or targets[0].get('mode')!='radec':
        raise ValueError('DADA pointing metadata requires one RA/Dec target')
    ra=float(targets[0]['ra_deg']);dec=float(targets[0]['dec_deg'])
    if not (math.isfinite(ra) and math.isfinite(dec) and 0<=ra<360 and -90<=dec<=90):
        raise ValueError('invalid pointing coordinates')
    result=[]
    for name,value in [('RA',ra/15),('DEC',abs(dec))]:
        total=round(value*3600,6)
        units,remainder=divmod(total,3600);minutes,seconds=divmod(remainder,60)
        sign=('+' if dec>=0 else '-') if name=='DEC' else ''
        result.append((name,f'{sign}{int(units):02d}:{int(minutes):02d}:{seconds:09.6f}'))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", default="J0534+2200")
    parser.add_argument("--telescope", required=True, help="Observatory identifier from the external site profile")
    parser.add_argument("--min-coarse-channel", type=int, default=176)
    parser.add_argument("--coarse-channels", type=int, default=40)
    parser.add_argument("--fine-channels", type=int, default=32)
    parser.add_argument("--edge-trim", type=int, default=2)
    parser.add_argument("--utc-start")
    parser.add_argument('--targets-file',type=Path,help='Single RA/Dec beam target for archive pointing metadata')
    parser.add_argument(
        "--detected-intensity",
        action="store_true",
        help="describe one 32-bit Stokes-I sample per time/channel",
    )
    args = parser.parse_args()

    effective_fine = args.fine_channels - 2 * args.edge_trim
    if args.fine_channels != 32 or effective_fine != 28:
        parser.error("the compact DADA path currently requires 32 fine bins and edge trim 2")

    # The internal axis has raw GPU bins 2..29. DADA drops raw bin 2 from
    # every coarse channel, retaining 3..29: exactly 27 centres per coarse.
    nchan = args.coarse_channels * (effective_fine - 1)
    channel_spacing_mhz = COARSE_WIDTH_MHZ / FINE_OVERSAMPLE_DENOMINATOR
    first_center_mhz = (
        args.min_coarse_channel * COARSE_WIDTH_MHZ - 13 * channel_spacing_mhz
    )
    last_center_mhz = first_center_mhz + (nchan - 1) * channel_spacing_mhz
    center_frequency_mhz = (first_center_mhz + last_center_mhz) / 2
    # DSPSR interprets BW/NCHAN as channel-centre spacing.
    bandwidth_mhz = nchan * channel_spacing_mhz
    tsamp_us = INPUT_TSAMP_US * args.fine_channels
    npol = 1 if args.detected_intensity else 2
    ndim = 1 if args.detected_intensity else 2
    block_bytes = 1 * 512 * nchan * npol * ndim * 4
    utc_start = args.utc_start or datetime.now(timezone.utc).strftime(
        "%Y-%m-%d-%H:%M:%S"
    )

    fields = [
        ("HDR_VERSION", "1.0"),
        ("HDR_SIZE", str(HEADER_BYTES)),
        ("SOURCE", args.source),
        ("MODE", "PSR"),
        ("TELESCOPE", args.telescope),
        # DSPSR's generic 32-bit FloatUnpacker deliberately matches only the
        # machine name "dspsr". LAMBDA is provenance, not an unpacker type.
        ("INSTRUMENT", "dspsr"),
        ("FREQ", f"{center_frequency_mhz:.12f}"),
        ("BW", f"{bandwidth_mhz:.12f}"),
        ("NCHAN", str(nchan)),
        ("NPOL", str(npol)),
        ("NDIM", str(ndim)),
        ("NBIT", "32"),
        ("TSAMP", f"{tsamp_us:.8f}"),
        ("UTC_START", utc_start),
        ("OBS_OFFSET", "0"),
        ("RESOLUTION", str(block_bytes)),
    ]
    if args.detected_intensity:
        fields.insert(13, ("STATE", "Intensity"))
    if args.targets_file:
        fields.extend(pointing_fields(args.targets_file))
    # Deliberately no OS_FACTOR: DADA contains a unique uniform frequency axis.
    payload = "".join(f"{key:<21}{value}\n" for key, value in fields).encode("ascii")
    if len(payload) > HEADER_BYTES:
        raise RuntimeError("header content exceeds the 4096-byte DADA header block")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload + b"\0" * (HEADER_BYTES - len(payload)))

    print(f"wrote {args.output}")
    print(f"NCHAN={nchan} FREQ={center_frequency_mhz:.12f} MHz")
    print(f"BW={bandwidth_mhz:.12f} MHz TSAMP={tsamp_us:.8f} us")
    print(f"DADA_BLOCK_BYTES={block_bytes}")


if __name__ == "__main__":
    main()
