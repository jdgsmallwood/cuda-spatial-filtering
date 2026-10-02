#!/usr/bin/env python3
"""Validate existing FPGA integer delays using bounded live voltage samples.

The seed selects overlap, not the fitted answer. Search residual lags on all six
FPGA pairs. Coherently average cross-spectra over time, then combine normalized
lag power across same-hand baselines/channels. No observing file is modified.
Requires numpy and scipy; no commissioning package or GPU Python stack.
"""
import argparse
import ctypes
import json
from pathlib import Path

import numpy as np
from scipy.fft import fft, ifft
from fit_fpga_delay_graph import fit_delays as fit_graph

PACKET = np.dtype([('sample','<u8'),('fpga','<u4'),('channel','<u2'),
                   ('padding','V8'),('scales','<u2',(20,)),('iq','i1',(64,20,2))])
PAIRS = [(i,j) for i in range(4) for j in range(i+1,4)]


class GpuCorrelator:
    """Optional existing-stack CUDA/cuFFT backend; normalized like the CPU path."""
    def __init__(self,library):
        self.library=ctypes.CDLL(str(Path(library).resolve()))
        self.call=self.library.spatial_delay_correlations
        self.call.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*5+[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t]
        self.call.restype=ctypes.c_int

    def __call__(self,x,y,max_lag,groups):
        if len(groups)!=3:raise ValueError('GPU backend requires three groups')
        x=np.ascontiguousarray(x,dtype=np.complex64);y=np.ascontiguousarray(y,dtype=np.complex64)
        w,n,nx=x.shape;ny=y.shape[2]
        masks=np.zeros((3,w),np.uint8)
        for g,indices in enumerate(groups):masks[g,indices]=1
        result=np.empty((3,2*max_lag+1,nx,ny),np.complex64)
        error=ctypes.create_string_buffer(1024)
        code=self.call(x.ctypes.data,y.ctypes.data,masks.ctypes.data,w,n,nx,ny,max_lag,
                       result.ctypes.data,error,len(error))
        if code:raise RuntimeError(error.value.decode())
        for g,indices in enumerate(groups):
            ex=np.sum(abs(x[indices])**2,axis=(0,1));ey=np.sum(abs(y[indices])**2,axis=(0,1))
            result[g]/=np.maximum(np.sqrt(ex[:,None]*ey[None,:])[None],1e-30)
        result*=(n/(n-abs(np.arange(-max_lag,max_lag+1))))[None,:,None,None]
        return result


def assemble(packets, channel, seed, start, samples):
    """Respect absolute counters, sub-packet offsets, duplicates and gaps."""
    records=packets[packets['channel']==channel]
    if not len(records):
        raise ValueError(f'no records for channel {channel}')
    _,first=np.unique(records['sample'],return_index=True)
    duplicates=len(records)-len(first)
    records=records[np.sort(first)]
    offsets=records['sample'].astype(np.int64)-int(seed)-int(start)
    index=(offsets[:,None]+np.arange(64)).ravel()
    valid=(index>=0)&(index<samples)
    raw=records['iq'].astype(np.float32)
    values=(raw[...,0]+1j*raw[...,1])*records['scales'][:,None,:]
    out=np.zeros((samples,20),np.complex64)
    present=np.zeros(samples,bool)
    out[index[valid]]=values.reshape(-1,20)[valid]
    present[index[valid]]=True
    return out,present,duplicates


def lag_correlations(x,y,max_lag,groups):
    """x/y: windows x samples x receptors. Positive lag means x lags y."""
    n=x.shape[1];length=2*n
    X=fft(x,n=length,axis=1,workers=1)
    Y=fft(y,n=length,axis=1,workers=1)
    lags=np.arange(-max_lag,max_lag+1)
    result=[]
    for group in groups:
        cross=np.zeros((length,x.shape[2],y.shape[2]),np.complex64)
        for w in group:
            cross+=X[w,:, :,None]*Y[w,:,None,:].conj()
        energy_x=np.sum(abs(x[group])**2,axis=(0,1))
        energy_y=np.sum(abs(y[group])**2,axis=(0,1))
        norm=np.sqrt(energy_x[:,None]*energy_y[None,:])
        corr=ifft(cross,axis=0,workers=1)[lags%length]
        corr/=np.maximum(norm[None],1e-30)
        # Linear correlation has fewer contributing samples at nonzero lag.
        corr*= (n/(n-abs(lags)))[:,None,None]
        result.append(corr)
    return np.asarray(result)


def summarize_score(lags,score):
    best=int(np.argmax(score));away=abs(lags-lags[best])>4
    runner=int(np.argmax(np.where(away,score,-np.inf)))
    other=int(np.argmax(np.where(np.arange(len(lags))!=best,score,-np.inf)))
    background=score[away];median=np.median(background)
    mad=1.4826*np.median(abs(background-median))
    return {'lag':int(lags[best]),'score':float(score[best]),
        'runner_lag_excluding_4_samples':int(lags[runner]),
        'peak_to_runner_ratio':float(score[best]/max(score[runner],1e-20)),
        'best_other_integer_lag':int(lags[other]),
        'peak_to_other_integer_ratio':float(score[best]/max(score[other],1e-20)),
        'peak_above_background_mad_units':float((score[best]-median)/max(mad,1e-20)),
        'at_search_boundary':best in (0,len(lags)-1)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('capture',type=Path)
    p.add_argument('--stream-map',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True,help='New diagnostic JSON; not an observing delay file')
    p.add_argument('--block-samples',type=int,default=16384)
    p.add_argument('--max-lag',type=int,default=1024)
    p.add_argument('--min-coverage',type=float,default=.995)
    p.add_argument('--exclude-antennas',default='14')
    p.add_argument('--channels',help='Comma-separated subset of retained channels; default all')
    p.add_argument('--gpu-library',type=Path,help='Optional libdelay_correlation.so; uses CUDA device 0')
    args=p.parse_args()
    correlate=GpuCorrelator(args.gpu_library) if args.gpu_library else lag_correlations
    if args.output.exists() or args.output.with_suffix('.npz').exists():p.error('output exists')
    if args.block_samples<256 or not 8<=args.max_lag<=args.block_samples//4:p.error('invalid block/lag size')
    if not 0<args.min_coverage<=1:p.error('invalid coverage threshold')
    manifest=json.loads((args.capture/'manifest.json').read_text())
    if manifest['format']!='lambda-delay-packets-v1' or manifest['packet_bytes']!=PACKET.itemsize:
        p.error('unsupported capture format')
    if not manifest['complete']:p.error('capture did not reach its requested end on all boards')
    if any(v['capacity_rejections'] or v['malformed'] for v in manifest['fpgas'].values()):
        p.error('capture contains malformed packets or capacity overflow')
    samples=manifest['samples'];block=args.block_samples;nwin=samples//block
    if samples%block or nwin<8:p.error('require at least eight complete time windows')
    mapping=json.loads(args.stream_map.read_text())['fpgas']
    excluded={int(v) for v in args.exclude_antennas.split(',') if v}
    streams={}
    for b in range(4):
        streams[b]=[[int(s['stream']) for s in mapping[str(b)]['streams']
            if s['antenna_id']>0 and s['antenna_id'] not in excluded and s['polarization']==pol]
            for pol in (0,1)]
        if any(not v for v in streams[b]):p.error(f'FPGA {b} has no live same-hand streams')
    packets=[np.memmap(args.capture/manifest['fpgas'][str(b)]['file'],dtype=PACKET,mode='r') for b in range(4)]
    seeds=[manifest['fpgas'][str(b)]['seed_delay'] for b in range(4)]
    lags=np.arange(-args.max_lag,args.max_lag+1)
    channels=manifest['channels'] if not args.channels else [int(v) for v in args.channels.split(',')]
    if not channels or len(set(channels))!=len(channels) or not set(channels)<=set(manifest['channels']):
        p.error('requested channels must be a distinct subset of captured channels')
    scores=np.zeros((3,6,len(channels),len(lags)))
    votes=np.zeros((3,6,len(channels),len(lags)),int)
    strong_votes=np.zeros_like(votes);counts=np.zeros((3,6,len(channels)),int)
    quality=[]
    for ci,channel in enumerate(channels):
        data=[];present=[]
        for b in range(4):
            x,m,duplicates=assemble(packets[b],channel,seeds[b],manifest['start_sample'],samples)
            data.append(x.reshape(nwin,block,20));present.append(m.reshape(nwin,block))
            quality.append({'fpga':b,'channel':channel,'coverage':float(m.mean()),'duplicates':duplicates})
        # Common window selection for every edge avoids comparing different data.
        coverage=np.array([m.mean(axis=1) for m in present])
        usable=np.flatnonzero(np.min(coverage,axis=0)>=args.min_coverage)
        groups=[usable,usable[usable<nwin//2],usable[usable>=nwin//2]]
        if any(len(g)<4 for g in groups):
            raise ValueError(f'channel {channel}: too few complete independent windows: {[len(g) for g in groups]}')
        for b in range(4):
            # Remove DC without turning missing samples into nonzero voltages.
            denominator=np.maximum(present[b].sum(axis=1)[:,None],1).astype(np.float32)
            mean=data[b].sum(axis=1)/denominator
            data[b]=(data[b]-mean[:,None,:])*present[b][:,:,None]
        for edge,(i,j) in enumerate(PAIRS):
            for pol in (0,1):
                x=data[i][:,:,streams[i][pol]];y=data[j][:,:,streams[j][pol]]
                # Extremely weak/dead receptors must not contribute noise votes.
                ex=np.mean(abs(x)**2,axis=(0,1));ey=np.mean(abs(y)**2,axis=(0,1))
                x=x[:,:,ex>max(np.median(ex)*.001,1e-20)]
                y=y[:,:,ey>max(np.median(ey)*.001,1e-20)]
                if not x.shape[2] or not y.shape[2]:continue
                corr=correlate(x,y,args.max_lag,groups)
                power=abs(corr.reshape(3,len(lags),-1))**2
                floor=np.maximum(np.median(power,axis=1),1e-30)
                ratio=power/floor[:,None,:]
                # Bound each baseline's influence so one narrowband interferer
                # cannot win simply by having arbitrarily large correlated power.
                scores[:,edge,ci]+=np.sum(np.clip(ratio-1,0,100),axis=2)
                for g in range(3):
                    peak=np.argmax(power[g],axis=0)
                    votes[g,edge,ci]+=np.bincount(peak,minlength=len(lags))
                    strong=np.max(ratio[g],axis=0)>20
                    strong_votes[g,edge,ci]+=np.bincount(peak[strong],minlength=len(lags))
                    counts[g,edge,ci]+=power.shape[2]
            print('channel',channel,'pair',i,j,'peak',int(lags[np.argmax(scores[0,edge,ci])]),flush=True)
        print('completed channel',channel,'windows',[len(g) for g in groups],flush=True)
    combined=scores.sum(axis=2)
    reports=[];solutions=[]
    for g,name in enumerate(('all','first_half','second_half')):
        edges=[]
        for e,(i,j) in enumerate(PAIRS):
            summary=summarize_score(lags,combined[g,e]);k=int(np.argmax(combined[g,e]))
            vote=votes[g,e].sum(axis=0);strong=strong_votes[g,e].sum(axis=0)
            summary.update(first=i,second=j,delay=int(seeds[j]-seeds[i]-summary['lag']),
                votes_at_ensemble_peak=int(vote[k]),total_votes=int(vote.sum()),
                strong_votes_at_ensemble_peak=int(strong[k]),total_strong_votes=int(strong.sum()))
            edges.append(summary)
        try:
            offsets,residual=fit_graph(edges,reference=1,max_residual=0)
            solutions.append(offsets);closure={'offsets':offsets,'max_residual_samples':float(max(abs(residual)))}
        except ValueError as exc:
            solutions.append(None);closure={'error':str(exc)}
        reports.append({'partition':name,'edges':edges,'closure':closure})
    # Leave one frequency channel out; correlated baselines are not independent
    # trials, so no nominal binomial confidence/probability is assigned.
    leave_one_out=[]
    for ci,channel in enumerate(channels):
        leave_one_out.append({'omitted_channel':channel,'lags':[
            int(lags[np.argmax(combined[0,e]-scores[0,e,ci])]) for e in range(6)]})
    best=[e['lag'] for e in reports[0]['edges']]
    stable=solutions[0] is not None and all(s==solutions[0] for s in solutions)
    stable&=all(row['lags']==best for row in leave_one_out)
    separated=all(not e['at_search_boundary'] and e['peak_to_runner_ratio']>1.5
                  and e['peak_to_other_integer_ratio']>1.1 for r in reports for e in r['edges'])
    report={'capture':str(args.capture),'seed_delays':dict(enumerate(seeds)),
        'backend':'cuFFT' if args.gpu_library else 'scipy.fft',
        'search_radius_samples':args.max_lag,'block_samples':block,'time_windows':nwin,
        'sample_seconds':manifest['sample_seconds'],'quality':quality,'partitions':reports,
        'leave_one_channel_out':leave_one_out,'stable_across_partitions':bool(stable),
        'separated_peaks':bool(separated),'validated':bool(stable and separated),
        'interpretation':'Residual search around seed; not a blind post-reboot delay solve. Votes are agreement, not SNR. MAD scores are diagnostics, not Gaussian significance. No observing file changed.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(args.output.with_suffix('.npz'),lags=lags,scores=scores,votes=votes,
        strong_votes=strong_votes,channels=channels,pairs=PAIRS)
    print(json.dumps({'validated':report['validated'],'partitions':[r['closure'] for r in reports]},indent=2))


if __name__=='__main__':main()
