#!/usr/bin/env python3
"""Plot independent solar calibration scores, spectra, and gain phase drift."""
import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def coarse_median(values):
    return np.nanmedian(values.reshape(40, 28), axis=1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('early_fit',type=Path)
    parser.add_argument('transfer_scores',type=Path)
    parser.add_argument('late_fit',type=Path)
    parser.add_argument('capture',type=Path)
    parser.add_argument('output_dir',type=Path)
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    with np.load(args.early_fit) as early, np.load(args.transfer_scores) as transfer, \
         np.load(args.late_fit) as late:
        freq=early['frequencies_hz']/1e6
        centers=freq.reshape(40,28).mean(axis=1)
        fig,axes=plt.subplots(2,2,figsize=(14,8),sharex=True,constrained_layout=True)
        for pi,pol in enumerate(('XX','YY')):
            axes[pi,0].plot(centers,coarse_median(early['metrics'][:,pi,1]),label='Held-out baselines')
            axes[pi,0].plot(centers,coarse_median(early['metrics'][:,pi,3]),label='Later window')
            axes[pi,0].plot(centers,coarse_median(transfer['scores'][:,pi,1]),label='Independent capture')
            axes[pi,0].axhline(.65,color='gray',lw=.7,ls='--')
            axes[pi,0].set(title=f'{pol}: prediction score',ylabel='Real normalized coherence',ylim=(-.1,1.05))
            axes[pi,1].plot(centers,early['good'][:,pi].reshape(40,28).mean(axis=1),label='First capture')
            axes[pi,1].plot(centers,late['good'][:,pi].reshape(40,28).mean(axis=1),label='Five-minute capture')
            axes[pi,1].set(title=f'{pol}: fraction of accepted fine channels',ylabel='Accepted fraction',ylim=(-.05,1.05))
            for ax in axes[pi]:ax.grid(alpha=.3);ax.legend(fontsize=8)
        for ax in axes[-1]:ax.set_xlabel('Frequency (MHz)')
        fig.suptitle('281–312 MHz solar calibration and transfer | third-order FM harmonic hypothesis spans 264–324 MHz')
        fig.savefig(args.output_dir/'frequency_quality.png',dpi=170)
        plt.close(fig)

        order=late['antenna_order'];phase=late['phase_change'];metrics=late['metrics']
        fig,axes=plt.subplots(2,1,figsize=(13,7),sharex=True,constrained_layout=True)
        for pi,pol in enumerate(('XX','YY')):
            candidate=(metrics[:,pi,0]>.7)&(metrics[:,pi,1]>.55)
            drift=np.rad2deg(np.median(abs(phase[candidate,pi]),axis=0))
            drift[np.isin(order,[9,14])]=np.nan
            axes[pi].bar(order,drift,width=.8,color=('tab:blue','tab:orange')[pi])
            axes[pi].axhline(15,color='black',ls='--',lw=.8)
            axes[pi].set(title=f'{pol}: first-to-last phase change, {candidate.sum()} initially plausible bins',
                         ylabel='Median absolute phase change (deg)')
            axes[pi].grid(axis='y',alpha=.3)
        axes[-1].set(xlabel='Physical antenna',xticks=np.arange(1,37))
        fig.suptitle('Five-minute solar capture: which receptors lose phase stability?')
        fig.savefig(args.output_dir/'antenna_phase_drift.png',dpi=170)
        plt.close(fig)

    with h5py.File(args.capture) as capture:
        pairs=capture['baseline_antenna_ids'][:]
        use=np.flatnonzero((pairs[:,0]==pairs[:,1])&(pairs[:,0]>0)&
                           (pairs[:,0]<=36)&(~np.isin(pairs[:,0],[9,14])))
        autos=[]
        for ti in (10,30,60,90,110):
            block=capture['visibilities'][ti,:,use]
            autos.append([np.median(block[:, :, hand,hand,0],axis=1)
                          for hand in (1,0)])
        power=np.median(np.asarray(autos),axis=0)
        power_db=10*np.log10(np.maximum(power,1e-30)/np.median(power,axis=1)[:,None])
    fig,axes=plt.subplots(2,1,figsize=(13,6),sharex=True,constrained_layout=True)
    for pi,pol in enumerate(('XX','YY')):
        axes[pi].plot(freq,power_db[pi],lw=.7)
        axes[pi].set(title=f'{pol}: median receptor autocorrelation',ylabel='Relative power (dB)')
        axes[pi].grid(alpha=.3)
    axes[-1].set_xlabel('Frequency (MHz)')
    fig.suptitle('Measured band shape; FM third harmonic is a hypothesis, not established by this plot')
    fig.savefig(args.output_dir/'band_autopower.png',dpi=170)
    plt.close(fig)
    folded=power.reshape(2,40,28)
    folded_db=10*np.log10(np.maximum(folded,1e-30)/np.median(folded,axis=2)[:,:,None])
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True,constrained_layout=True)
    for pi,pol in enumerate(('XX','YY')):
        median=np.median(folded_db[pi],axis=0)
        axes[pi].plot(np.arange(28),median,'o-',ms=3)
        axes[pi].axhline(0,color='black',lw=.7)
        axes[pi].set(title=pol,xlabel='Retained fine-bin position inside coarse channel',
                     ylabel='Median relative power (dB)',ylim=(-1.2,.4))
        axes[pi].grid(alpha=.3)
    fig.suptitle('Repeated channel-edge dip at every 0.78125-MHz coarse boundary; no de-ripple configured')
    fig.savefig(args.output_dir/'coarse_folded_bandpass.png',dpi=170)
    plt.close(fig)
    print('Wrote frequency_quality.png, antenna_phase_drift.png, band_autopower.png, coarse_folded_bandpass.png')


if __name__=='__main__':main()
