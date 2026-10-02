#!/usr/bin/env python3
"""Plot local and full-search lag evidence from analyze_delay_samples.py."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('report',type=Path);p.add_argument('output',type=Path)
a=p.parse_args();r=json.loads(a.report.read_text());z=np.load(a.report.with_suffix('.npz'))
lag=z['lags'];scores=z['scores'].sum(axis=2)
for wide in (False,True):
 fig,axs=plt.subplots(3,2,figsize=(11,9),sharex=True)
 for e,ax in enumerate(axs.flat):
  for group,label in enumerate(('All data','First half','Second half')):
   values=scores[group,e];ax.plot(lag,values/max(values),label=label,lw=1.2)
  edge=r['partitions'][0]['edges'][e]
  ax.set_title(f"FPGA {edge['first']}–{edge['second']}: lag {edge['lag']:+d}; votes {edge['votes_at_ensemble_peak']}/{edge['total_votes']}")
  ax.set_xlim((-max(abs(lag)),max(abs(lag))) if wide else (-8,8))
  ax.set_ylim((.001,1.2) if wide else (0,1.08))
  if wide:ax.set_yscale('log')
  ax.grid(alpha=.25);ax.set_ylabel('Score / peak score');ax.set_xlabel('Residual lag (samples)')
 axs[0,0].legend(fontsize=8)
 fig.suptitle(f"Delay evidence: {len(z['channels'])} retained coarse channels\nTime halves are independent; seed delays select overlap but do not force lag zero")
 fig.tight_layout(rect=(0,0,1,.94))
 dest=a.output.with_name(a.output.stem+('_wide' if wide else '')+a.output.suffix)
 dest.parent.mkdir(parents=True,exist_ok=True);fig.savefig(dest,dpi=160);plt.close(fig)
 print(dest)
