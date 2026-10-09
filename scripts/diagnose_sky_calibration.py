#!/usr/bin/env python3
"""Compare sky hypotheses on native covariance data with held-out baselines/time.

Requires the lambda_commissioning Python environment. Outputs are diagnostics,
not automatically accepted beam weights. No measurement set is required.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from astropy.coordinates import AltAz
from astropy.time import Time
from astropy.utils import iers

from lambda_commissioning.constants import NARRIBRI, CHANWIDTH, c
from lambda_commissioning.modelling import calc_sky_model, calc_lmn, _aep_interp, sum_point_sources
from lambda_commissioning.utils import make_telescope_model, read_hdf5_time_freq
from fine_frequency_axis import processing_frequencies_hz, unique_processing_frequencies_hz
from lambda_commissioning.vistools import BRIGHT_SOURCE_CATALOG, BRIGHT_SOURCE_ALPHA

FLUX150 = {'Cyg A':10500,'Cas A':13000,'Tau A':1500,'Vir A':1200,
           'Cen A':1300,'Her A':800,'Hyd A':1000,'For A':750,'Pic A':550}


def fit_gains(data, model, mask, phase_only=False, iterations=500):
    """Damped alternating least squares, with a spectral phase initial guess."""
    h = np.where(mask, data * model.conj(), 0)
    _, vectors = np.linalg.eigh(h)
    phase = np.exp(1j*np.angle(vectors[:, -1]))
    scale = np.sqrt(np.sum(abs(data[mask])*abs(model[mask])) /
                    max(np.sum(abs(model[mask])**2), 1e-30))
    g = phase * (1 if phase_only else scale)
    mm = np.where(mask, abs(model)**2, 0)
    for _ in range(iterations):
        new = (h @ g) / np.maximum(mm @ abs(g)**2, 1e-30)
        if phase_only:
            new = np.exp(1j*np.angle(new))
        new = .5*(new+g)
        if phase_only:
            new = np.exp(1j*np.angle(new))
        new *= np.exp(-1j*np.angle(new[1]))
        change = np.linalg.norm(new-g)/max(np.linalg.norm(g),1e-30)
        g = new
        if change < 1e-8:
            break
    return g


def score(data, prediction, mask):
    d, m = data[mask], prediction[mask]
    denom = np.sqrt(np.vdot(d,d).real*np.vdot(m,m).real)
    return float(np.vdot(m,d).real/max(denom,1e-30))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('capture',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--coarse-stride',type=int,default=4)
    p.add_argument('--samples',type=int,default=20)
    p.add_argument('--nside',type=int,default=64)
    p.add_argument('--antenna-id-swaps',default='',help='Capture label pairs, e.g. 5:29,11:13')
    args=p.parse_args()
    iers.conf.auto_download=False; iers.conf.auto_max_age=None
    times,freqs=read_hdf5_time_freq(str(args.capture))
    with h5py.File(args.capture) as capture:
        first=int(capture.attrs['min_channel'])
        ncoarse=int(capture.attrs['max_channel'])-first+1
    if len(freqs)==ncoarse*28:
        freqs=processing_frequencies_hz(first,ncoarse)
        fine_per_coarse, center = 28, 14
    elif len(freqs)==ncoarse*27:
        freqs=unique_processing_frequencies_hz(first,ncoarse)
        fine_per_coarse, center = 27, 13
    else:
        fine_per_coarse, center = 1, 0
    channels=np.arange(0,ncoarse,args.coarse_stride)*fine_per_coarse+center
    channels=np.unique(np.r_[channels,(ncoarse-1)*fine_per_coarse+center])
    n=args.samples
    windows=[np.arange(3,3+n),np.arange(len(times)-n-3,len(times)-3)]
    stamps=Time([np.mean(times[w].jd) for w in windows],format='jd')
    raw=np.empty((2,len(channels),2,36,36),complex)
    with h5py.File(args.capture) as f:
        pairs=f['baseline_antenna_ids'][:]
        valid=(pairs>0).all(axis=1)&(pairs<=36).all(axis=1)
        a,b=pairs[valid].T-1
        for wi,win in enumerate(windows):
            accum=np.zeros((len(channels),valid.sum(),2),complex)
            for ti in win:
                block=f['visibilities'][int(ti),channels][:,valid]
                for pi,hand in enumerate((1,0)):
                    accum[:,:,pi]+=block[:,:,hand,hand,0]+1j*block[:,:,hand,hand,1]
            for pi in range(2):
                raw[wi,:,pi,a,b]=(accum[:,:,pi]/n).T
                raw[wi,:,pi,b,a]=(accum[:,:,pi].conj()/n).T
    print('read native covariances',stamps.isot.tolist(),flush=True)
    array=make_telescope_model(np.arange(1,37))
    order=np.arange(36)
    for pair in filter(None,args.antenna_id_swaps.split(',')):
        left,right=map(int,pair.split(':'))
        order[left-1],order[right-1]=order[right-1],order[left-1]
    lengths=np.hypot(array.uu_m,array.vv_m)[order][:,order]
    upper=np.triu(np.ones((36,36),bool),1)
    fit=upper&(lengths>7)
    ii,jj=np.where(fit)
    withheld=np.zeros_like(fit);withheld[ii[::5],jj[::5]]=True
    train=(fit&~withheld);train|=train.T
    norms=np.sqrt(raw.diagonal(axis1=-2,axis2=-1).real)
    data=raw/(norms[..., :,None]*norms[...,None,:])
    models={'point_static':np.ones((2,len(channels),36,36),complex)}
    for wi,stamp in enumerate(stamps):
        aa={name:coord.transform_to(AltAz(obstime=stamp,location=NARRIBRI))
            for name,coord in BRIGHT_SOURCE_CATALOG.items()}
        for ci,freq in enumerate(freqs[channels]):
            for pi,pol in enumerate(('XX','YY')):
                sky=calc_sky_model(stamp,freq,array,pol,nside=args.nside)
                candidates={'diffuse_'+pol:sky}
                points=np.zeros((36,36),complex)
                for name,flux150 in FLUX150.items():
                    src=aa[name]
                    if src.alt.deg<=0: continue
                    l,m,z=calc_lmn(src.alt.deg,src.az.deg)
                    beam=float(_aep_interp(round(freq/1e6/CHANWIDTH),pol)([[m,l]])[0])
                    flux=flux150*(freq/150e6)**BRIGHT_SOURCE_ALPHA[name]*beam
                    pt=sum_point_sources(array,c/freq,np.array([l]),np.array([m]),np.array([z]),np.array([flux]))
                    points+=pt
                    candidates['point_'+name]=pt/max(flux,1e-30)
                candidates['points_'+pol]=points
                candidates['diffuse_points_'+pol]=sky+.1*points
                for name,mat in candidates.items():
                    if name not in models: models[name]=np.zeros((2,len(channels),36,36),complex)
                    models[name][wi,ci]=mat/max(float(np.diag(mat).real.mean()),1e-30)
            print('model',wi,ci,round(freq/1e6,3),flush=True)
    models={name:mat[:,:,order][:,:,:,order] for name,mat in models.items()}
    # Conjugation and receptor interchange are explicit hypotheses.
    for name in list(models):
        if name.startswith('diffuse_'):
            models[name+'_conj']=models[name].conj()
    records=[];solutions={}
    for pi,pol in enumerate(('XX','YY')):
        for name,model in models.items():
            gs=[];fit_scores=[];base_scores=[];time_scores=[]
            for ci in range(len(channels)):
                g=fit_gains(data[0,ci,pi],model[0,ci],train)
                prediction=model[:,ci]*g[None,:,None]*g.conj()[None,None,:]
                gs.append(g)
                fit_scores.append(score(data[0,ci,pi],prediction[0],train&upper))
                base_scores.append(score(data[0,ci,pi],prediction[0],withheld))
                time_scores.append(score(data[1,ci,pi],prediction[1],fit))
            key=pol+'_'+name;solutions[key]=np.array(gs)
            record={'pol':pol,'model':name,'fit':fit_scores,'held_baselines':base_scores,'held_time':time_scores}
            records.append(record)
            print(key,'median fit/base/time',*[round(float(np.median(v)),3) for v in (fit_scores,base_scores,time_scores)],flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(str(args.output)+'.npz',raw=raw,data=data,channels=channels,
        frequencies_hz=freqs[channels],times_jd=stamps.jd,train=train,withheld=withheld,
        **{'model_'+k:v for k,v in models.items()},**{'gains_'+k:v for k,v in solutions.items()})
    Path(str(args.output)+'.json').write_text(json.dumps({'capture':str(args.capture),
        'antenna_id_swaps':args.antenna_id_swaps,
        'utc':stamps.isot.tolist(),'frequencies_hz':freqs[channels].tolist(),'results':records},indent=2)+'\n')


if __name__=='__main__':
    main()
