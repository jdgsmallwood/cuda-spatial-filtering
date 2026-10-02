#!/usr/bin/env python3
"""Write masked PSRCHIVE copies, matching rejected channels by frequency."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mask',type=Path)
    p.add_argument('archives',type=Path,nargs='+')
    p.add_argument('--paz',default='paz')
    args=p.parse_args()
    mask=json.loads(args.mask.read_text())
    if len(mask['frequency_hz'])!=len(mask['good']) or not any(mask['good']):
        p.error('invalid channel mask')
    rejected=' '.join(f'{freq/1e6:.9f}' for freq,good in zip(mask['frequency_hz'],mask['good']) if not good)
    for archive in args.archives:
        dest=archive.with_suffix('.masked.ar')
        if dest.exists(): p.error(f'refusing to overwrite {dest}')
        cmd=[args.paz,'-e','masked.ar']
        if rejected: cmd+=['-f',rejected]
        subprocess.run(cmd+[str(archive)],check=True)
        print(dest)


if __name__=='__main__':main()
