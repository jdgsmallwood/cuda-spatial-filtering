#!/usr/bin/env python3
"""Interleave strict eager/graph GPU-harness trials; retain failures as failures.

No live ingest or output writing. Use a fresh output directory for each session.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('binary', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--modes', nargs='+', choices=('weights-only', 'coarse', 'fine'), default=['coarse', 'fine'])
    p.add_argument('--buffers', nargs='+', type=int, choices=(1, 2, 3), default=[1, 2, 3])
    p.add_argument('--protection', nargs='+', choices=('off', 'rank-one'), default=['off', 'rank-one'])
    p.add_argument('--blocks', type=int, default=192)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--timeout', type=float, default=300)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if not 1 <= args.blocks <= 1000 or args.repeats < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        p.error('blocks must be 1..1000; repeats and finite timeout must be positive')
    cases = [(m,b,False if m=='weights-only' else protection=='rank-one')
             for m in dict.fromkeys(args.modes) for b in dict.fromkeys(args.buffers)
             for protection in (['off'] if m=='weights-only' else dict.fromkeys(args.protection))]
    plan=[]
    for repeat in range(args.repeats):
        pairs=list(cases)
        rng=random.Random(41+repeat)
        rng.shuffle(pairs)
        for m,b,protected in pairs:
            executions=['eager','graph']
            rng.shuffle(executions)
            plan.extend(dict(mode=m,gpu_buffers=b,protected=protected,execution=e,repeat=repeat) for e in executions)
    if args.dry_run:
        print(json.dumps(plan,indent=2));return 0
    binary=args.binary.resolve()
    if not binary.is_file():p.error('benchmark executable does not exist')
    if args.output.exists() and any(args.output.iterdir()):p.error('output directory must be empty')
    args.output.mkdir(parents=True,exist_ok=True)
    results=[]
    for index,case in enumerate(plan):
        command=[str(binary),str(args.blocks),str(case['gpu_buffers']),case['mode']]
        if case['protected']:command.append('protected')
        env=os.environ.copy()
        for key in ('SPATIAL_DISABLE_CUDA_GRAPH','SPATIAL_ENABLE_ADAPTIVE_CUDA_GRAPH','SPATIAL_REQUIRE_CUDA_GRAPH','SPATIAL_BENCH_PROFILE_RANGE'):
            env.pop(key,None)
        if case['execution']=='graph':
            env.update(SPATIAL_ENABLE_ADAPTIVE_CUDA_GRAPH='1',SPATIAL_REQUIRE_CUDA_GRAPH='1')
        else:env['SPATIAL_DISABLE_CUDA_GRAPH']='1'
        name=f"{index:03d}-{case['mode']}-{case['execution']}-b{case['gpu_buffers']}-p{int(case['protected'])}.log"
        row=dict(**case,status='failed',input_GB_per_s=None,seconds=None,captured_graphs=None,
                 blocks=args.blocks,log=name,error='',command=command)
        try:
            with (args.output/name).open('w') as log:
                proc=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=args.timeout)
            if proc.returncode:raise RuntimeError(f'executable exit {proc.returncode}; see log')
            lines=[s for s in (args.output/name).read_text(errors='replace').splitlines() if s.startswith('mode=')]
            if len(lines)!=1:raise RuntimeError('expected exactly one benchmark result')
            data=dict(item.split('=',1) for item in lines[0].split())
            count=int(data['captured_graphs'])
            row['captured_graphs']=count
            expected=case['gpu_buffers'] if case['execution']=='graph' else 0
            if count!=expected:raise RuntimeError(f'expected {expected} captured graphs, observed {count}')
            if data['mode']!=case['mode'] or int(data['gpu_buffers'])!=case['gpu_buffers'] or bool(int(data['protected']))!=case['protected'] or int(data['blocks'])!=args.blocks:
                raise RuntimeError('result configuration does not match requested case')
            seconds=float(data['seconds']);rate=float(data['input_GB_per_s'])
            if not (math.isfinite(seconds) and seconds>0 and math.isfinite(rate) and rate>0):
                raise RuntimeError('invalid duration/throughput')
            expected_rate=args.blocks*int(data['input_bytes_per_block'])/seconds/1e9
            if not math.isclose(rate,expected_rate,rel_tol=2e-5):raise RuntimeError('inconsistent throughput accounting')
            row.update(status='ok',seconds=seconds,input_GB_per_s=rate)
        except (OSError,subprocess.TimeoutExpired,RuntimeError,KeyError,ValueError) as error:
            row['error']=str(error)
        results.append(row)
        (args.output/'runs.json').write_text(json.dumps(results,indent=2)+'\n')
        with (args.output/'runs.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader();writer.writerows(results)
        print(name,row['status'],row['input_GB_per_s'],row['error'],flush=True)
    return int(any(r['status']!='ok' for r in results))


if __name__=='__main__':
    sys.exit(main())
