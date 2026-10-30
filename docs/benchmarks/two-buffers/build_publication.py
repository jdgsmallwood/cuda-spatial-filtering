"""Fixed two-buffer comparison; excludes measurements at other buffer counts."""
from pathlib import Path
import csv,hashlib,statistics,math,os,json
os.environ.setdefault('MPLCONFIGDIR','/tmp/adaptive-graphs-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
OUT=Path(__file__).resolve().parent
provenance=json.loads((OUT/'input_provenance.json').read_text())
assert hashlib.sha256((OUT/'trials.csv').read_bytes()).hexdigest()==provenance['trials_sha256']
rows=list(csv.DictReader((OUT/'trials.csv').open()))
assert all(int(r['gpu_buffers'])==2 for r in rows)
assert len(rows)==30
for r in rows:
    assert r['status']=='ok' and int(r['blocks'])==192
    assert int(r['captured_graphs'])==(2 if r['execution']=='graph' else 0)
    assert math.isclose(float(r['input_GB_per_s']),192*104857600/float(r['seconds'])/1e9,rel_tol=2e-5)
    for k in ('command','log','error'):r.pop(k,None)
def csvout(name,data):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator="\n");w.writeheader();w.writerows(data)
csvout('trials.csv',rows)
configs=[('weights-only','False','Weights only'),('coarse','False','Coarse, unprotected'),('coarse','True','Coarse, protected'),('fine','False','Fine, unprotected'),('fine','True','Fine, protected')]
summary=[]
for m,p,label in configs:
    s=dict(configuration=label,gpu_buffers=2)
    for e in ('eager','graph'):
        rr=[r for r in rows if r['mode']==m and r['protected']==p and r['execution']==e]
        assert len(rr)==3 and len({r['repeat'] for r in rr})==3
        v=[float(r['input_GB_per_s']) for r in rr]
        s.update({e+'_median':statistics.median(v),e+'_min':min(v),e+'_max':max(v)})
    s['change_percent']=100*(s['graph_median']/s['eager_median']-1);summary.append(s)
csvout('throughput.csv',summary)
headers=['Configuration','Eager GB/s [min, max]','Graphs GB/s [min, max]','Change']
fmt=lambda s,e:f"{s[e+'_median']:.3f} [{s[e+'_min']:.3f}, {s[e+'_max']:.3f}]"
data=[[s['configuration'],fmt(s,'eager'),fmt(s,'graph'),f"{s['change_percent']:+.2f}%"] for s in summary]
caption='Two GPU buffers for every method. Median [minimum, maximum] of three 192-block trials; rates include H2D transfers and GPU compute, excluding packet parsing and output writing. Adaptive nulling rank is two; protected denotes rank-one spatial protection. Ranges are observed extrema, not confidence intervals.'
md=['| '+' | '.join(headers)+' |','|---|---:|---:|---:|']+['| '+' | '.join(r)+' |' for r in data]
(OUT/'throughput.md').write_text('\n'.join(md)+'\n\n'+caption+'\n')
tex=[r'% Requires booktabs',r'\begin{table*}',r'\centering',r'\small',r'\caption{'+caption+'}',r'\label{tab:beam-two-buffers}',r'\begin{tabular}{lrrr}',r'\toprule',' & '.join(headers)+r' \\',r'\midrule']
tex+=[' & '.join(r).replace('%',r'\%')+r' \\' for r in data]
tex +=[r'\bottomrule',r'\end{tabular}',r'\end{table*}']
(OUT/'throughput.tex').write_text('\n'.join(tex)+'\n')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','savefig.dpi':240})
fig,ax=plt.subplots(figsize=(7.4,3.5),layout='constrained');y=np.arange(5)
for e,offset,color in [('eager',-.16,'#666666'),('graph',.16,'#0072B2')]:
    vals=np.array([s[e+'_median'] for s in summary]);lo=np.array([s[e+'_min'] for s in summary]);hi=np.array([s[e+'_max'] for s in summary])
    bars=ax.barh(y+offset,vals,height=.29,color=color,xerr=[vals-lo,hi-vals],capsize=2,label='CUDA Graphs' if e=='graph' else 'Eager')
    for bar,v in zip(bars,vals):ax.text(v+.14,bar.get_y()+bar.get_height()/2,f'{v:.3f}',va='center',fontsize=8)
ax.set(yticks=y,yticklabels=[s['configuration'] for s in summary],xlim=(0,13.5),xlabel='Input payload throughput (GB/s)',title='Two GPU buffers for every method');ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True);ax.legend(loc='lower right',frameon=False)
for ext in ('pdf','svg','png'):fig.savefig(OUT/f'throughput.{ext}')
plt.close(fig)
print((OUT/'throughput.md').read_text())

svg=OUT/'throughput.svg'
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
files=sorted(f for f in OUT.iterdir() if f.is_file() and f.name!='MANIFEST.sha256')
(OUT/'MANIFEST.sha256').write_text(''.join(f'{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.name}\n' for f in files))
