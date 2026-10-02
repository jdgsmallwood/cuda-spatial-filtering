#!/usr/bin/env python3
"""Test four-port ADC assignments using independent triangle phases."""
import argparse
import itertools
import json
from pathlib import Path

import h5py
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('capture',type=Path)
    p.add_argument('model_cache',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--pairs',default='5:29,11:13')
    p.add_argument('--bands',default='31,32,33,34,35,36,38')
    p.add_argument('--samples',type=int,default=40)
    p.add_argument('--reference-swaps',default='',help='Confirmed whole-antenna swaps applied before testing')
    p.add_argument('--exclude-antennas',default='5,29,11,13,14')
    args=p.parse_args()
    pairs=[tuple(int(x)-1 for x in item.split(':')) for item in args.pairs.split(',')]
    bands=np.array([int(x) for x in args.bands.split(',')])
    z=np.load(args.model_cache);channels=z['channels'][bands]
    model=np.stack([z['model_diffuse_'+pol][:,bands] for pol in ('XX','YY')],axis=2)
    raw=np.zeros((2,len(channels),72,72),complex)
    with h5py.File(args.capture) as f:
        nt=f['visibilities'].shape[0]
        windows=[np.arange(3,3+args.samples),np.arange(nt-args.samples-3,nt-3)]
        baselines=f['baseline_antenna_ids'][:]
        good=(baselines>0).all(axis=1)&(baselines<=36).all(axis=1)
        a,b=baselines[good].T-1
        for wi,window in enumerate(windows):
            v=np.zeros((len(channels),len(a),2,2),complex)
            for ti in window:
                block=f['visibilities'][int(ti),channels][:,good]
                v+=block[...,0]+1j*block[...,1]
            v/=len(window)
            for pp in range(2):
                for qq in range(2):
                    i,j=2*a+pp,2*b+qq
                    raw[wi,:,i,j]=v[:,:,pp,qq].T
                    raw[wi,:,j,i]=v[:,:,pp,qq].conj().T
    power=np.sqrt(raw.diagonal(axis1=-2,axis2=-1).real)
    raw/=power[..., :,None]*power[...,None,:]
    tris=np.array(list(itertools.combinations(range(36),3)))
    i,j,k=tris.T
    # Canonical physical (XX,YY) starts from native raw hands (1,0).
    base=np.array([[2*a+1,2*a] for a in range(36)])
    for pair in filter(None,args.reference_swaps.split(',')):
        a,b=[int(x)-1 for x in pair.split(':')]
        base[[a,b]]=base[[b,a]]
    output=[]
    for a,b in pairs:
        other=[int(v)-1 for v in args.exclude_antennas.split(',') if v and int(v)-1 not in (a,b)]
        selected=((tris==a)|(tris==b)).any(axis=1)&(~np.isin(tris,other).any(axis=1))
        ii,jj,kk=i[selected],j[selected],k[selected]
        candidates=[]
        for permutation in itertools.permutations(range(4)):
            mapping=base.copy();source=base[[a,b]].ravel()
            mapping[[a,b]]=source[list(permutation)].reshape(2,2)
            results={}
            for pol in range(2):
                idx=mapping[:,pol]
                d=raw[:,:,idx][:,:,:,idx]
                m=model[:,:,pol]
                dc=d[:,:,ii,jj]*d[:,:,jj,kk]*d[:,:,kk,ii]
                mc=m[:,:,ii,jj]*m[:,:,jj,kk]*m[:,:,kk,ii]
                cosine=np.cos(np.angle(dc*mc.conj()))
                weight=np.sqrt(abs(dc)*abs(mc))
                # Same sky-null cut for every assignment.
                ok=(abs(mc)>.01*np.median(abs(mc),axis=-1,keepdims=True))
                weight*=ok
                train_bands=np.arange(len(bands))%2==0
                test_bands=~train_bands
                for split,wi,bi in [('train',0,train_bands),('held_frequency',0,test_bands),('held_time',1,np.ones(len(bands),bool))]:
                    w=weight[wi,bi];v=cosine[wi,bi]
                    results[('XX','YY')[pol]+'_'+split]=float(np.sum(w*v)/np.maximum(np.sum(w),1e-30))
            candidates.append({'target_to_source':list(permutation),**results,
                               'train_score':.5*(results['XX_train']+results['YY_train'])})
        candidates.sort(key=lambda x:x['train_score'],reverse=True)
        record={'antennas':[a+1,b+1],'slot_order':[f'A{a+1}:XX',f'A{a+1}:YY',f'A{b+1}:XX',f'A{b+1}:YY'],'candidates':candidates}
        output.append(record)
        print(json.dumps({'pair':record['antennas'],'top':candidates[:5]},indent=2),flush=True)
    args.output.write_text(json.dumps({'capture':str(args.capture),'model_cache':str(args.model_cache),'bands':bands.tolist(),'results':output},indent=2)+'\n')
    np.savez_compressed(str(args.output)+'.npz',raw=raw,model=model,bands=bands)


if __name__=='__main__':main()
