#!/usr/bin/env python3
"""Make matched all-sky dirty images of calibrated native data and sky model.

Uses the last time window, never used to fit the plotted instrumental phases.
Requires the same commissioning environment as calibrate_native_sky.py.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation
from astropy.time import Time
from astropy.utils import iers

from lambda_commissioning.utils import make_telescope_model
from lambda_commissioning.vistools import BRIGHT_SOURCE_CATALOG
from fine_frequency_axis import science_channel_mask


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('validation',type=Path)
    p.add_argument('smooth',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--config-file',type=Path,required=True)
    p.add_argument('--channels',type=int,default=16)
    p.add_argument('--pixels',type=int,default=201)
    args=p.parse_args()
    iers.conf.auto_download=False;iers.conf.auto_max_age=None
    source=np.load(args.validation);smooth=np.load(args.smooth)
    freq=source['frequencies_hz'];unique=science_channel_mask(freq)
    accepted=np.flatnonzero(smooth['good'].all(axis=1)&unique)
    if not len(accepted):raise ValueError('no jointly accepted channels to image')
    channels=accepted[np.unique(np.linspace(0,len(accepted)-1,min(args.channels,len(accepted))).astype(int))]
    data=source['data'][2,channels];model=source['model'][2,channels]
    gains=smooth['gains'][channels];phase=smooth['phases'][channels]
    corrected=data*phase.conj()[:,:,:,None]*phase[:,:,None,:]
    predicted=model*abs(gains)[:,:,:,None]*abs(gains)[:,:,None,:]
    selection=source['selected'];ii,jj=np.where(selection)
    order=source['antenna_order']-1;array=make_telescope_model(np.arange(1,37))
    uu=array.uu_m[order][:,order][ii,jj]
    vv=array.vv_m[order][:,order][ii,jj]
    ww=array.ww_m[order][:,order][ii,jj]
    axis=np.linspace(-1,1,args.pixels);ll,mm=np.meshgrid(axis,axis)
    sky=ll**2+mm**2<=1;l=ll[sky];m=mm[sky];n=np.sqrt(np.maximum(0,1-l*l-m*m))
    images=np.zeros((2,3,len(l)))
    # Equal weights in normalized covariance units; no autocorrelations.
    for ci,channel in enumerate(channels):
        values=np.stack([corrected[ci,:,ii,jj].T,predicted[ci,:,ii,jj].T],axis=1)
        values=np.concatenate((values,values[:,0:1]-values[:,1:2]),axis=1)
        for start in range(0,len(l),1024):
            stop=start+1024
            path=uu[:,None]*l[None,start:stop]+vv[:,None]*m[None,start:stop]+ww[:,None]*(n[None,start:stop]-1)
            fringe=np.exp(2j*np.pi*freq[channel]/299792458*path)
            images[:,:,start:stop]+=np.real(values.reshape(6,-1)@fringe).reshape(2,3,-1)
    images/=len(channels)*len(ii)
    stamp=Time(source['times_jd'][2],format='jd')
    site=json.loads(args.config_file.read_text())['array_location']
    location=EarthLocation.from_geodetic(site['longitude_deg']*u.deg,site['latitude_deg']*u.deg,site['height_m']*u.m)
    frame=AltAz(obstime=stamp,location=location)
    labels=[]
    for name in ('For A','Pic A','Cyg A','Cas A','Tau A','Cen A'):
        aa=BRIGHT_SOURCE_CATALOG[name].transform_to(frame)
        if aa.alt.deg>0:
            labels.append((name,float(np.cos(aa.alt.rad)*np.sin(aa.az.rad)),float(np.cos(aa.alt.rad)*np.cos(aa.az.rad))))
    fig,axs=plt.subplots(2,3,figsize=(15,10),sharex=True,sharey=True)
    vmax=np.max(abs(images[:,:2]));rms=[]
    for pi in range(2):
        rms.append(float(np.sqrt(np.mean(images[pi,2]**2))))
        for column in range(3):
            plane=np.full(ll.shape,np.nan);plane[sky]=images[pi,column]
            ax=axs[pi,column];im=ax.imshow(plane,origin='lower',extent=(-1,1,-1,1),cmap='RdBu_r',vmin=-vmax,vmax=vmax)
            theta=np.linspace(0,2*np.pi,400)
            for elevation in (0,30,60):
                radius=np.cos(np.deg2rad(elevation));ax.plot(radius*np.sin(theta),radius*np.cos(theta),color='.45',lw=.5,alpha=.5)
            ax.axhline(0,color='.5',lw=.3);ax.axvline(0,color='.5',lw=.3)
            for name,x,y in labels:
                ax.plot(x,y,'k+',ms=6)
                ax.annotate(name,(x,y),xytext=(4,-14 if y>.9 else 4),
                            textcoords='offset points',fontsize=8)
            ax.set_title(('XX','YY')[pi]+' — '+('Calibrated data','Matched sky model','Data minus model')[column])
            ax.set_aspect('equal');ax.set_xlabel('East direction cosine');ax.set_ylabel('North direction cosine')
    fig.suptitle(f'LAMBDA all-sky dirty images | {stamp.isot} UTC\n{freq[channels[0]]/1e6:.2f}–{freq[channels[-1]]/1e6:.2f} MHz; {len(channels)} sampled clean fine channels; baselines >7 m',fontsize=14)
    fig.subplots_adjust(left=.05,right=.9,bottom=.09,top=.88,wspace=.15,hspace=.2)
    fig.colorbar(im,cax=fig.add_axes([.92,.19,.015,.59]),label='Relative correlated brightness (not Jy)')
    fig.text(.5,.015,'Phase-only correction; normalized covariances; model retains fitted receptor amplitudes.\nSame sampling and scale in every panel. No CLEAN/deconvolution; sidelobes remain. Crab is below the horizon if not labelled.',ha='center',fontsize=10)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=160)
    np.savez_compressed(args.output.with_suffix('.npz'),images=images,sky_mask=sky,l_axis=axis,m_axis=axis,
                        channels=channels,frequencies_hz=freq[channels],utc=stamp.isot)
    print(json.dumps({'output':str(args.output),'utc':stamp.isot,'sampled_channels':channels.tolist(),
                      'frequency_mhz':(freq[channels]/1e6).tolist(),'residual_image_rms':rms},indent=2))


if __name__=='__main__':main()
