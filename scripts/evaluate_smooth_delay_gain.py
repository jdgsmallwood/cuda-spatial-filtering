#!/usr/bin/env python3
"""Test a hierarchical FPGA/cable phase-slope model on independent visibilities.

Per-receptor slopes come from the circular delay search. The FPGA term is a
cross-polarization weighted circular mean; antenna/cable terms are residuals.
We smooth each receptor's gain phase within each coarse channel, then compare
held-out transfer scores with unsmoothed gains. The coarse-channel phase is a
nuisance parameter, so this cannot identify a physical clock delay absolutely.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from diagnose_sky_calibration import score


def wrap(value, period):
    return (value + period/2) % period - period/2


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fit',type=Path)
    parser.add_argument('transfer',type=Path)
    parser.add_argument('delay_terms',type=Path)
    parser.add_argument('output_dir',type=Path)
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    with np.load(args.fit) as training, np.load(args.transfer) as validation:
        freq=training['frequencies_hz']; gains=training['gains'];good=training['good']
        joint=training['joint'];order=training['antenna_order'];selected=training['selected']
        model=validation['model'];data=validation['data']
    source=json.loads(args.delay_terms.read_text())
    period=source['delay_alias_period_ns']
    delay=np.full((2,36),np.nan);coherence=np.full((2,36),np.nan);fpga=np.zeros(36,int)
    for term in source['antenna_terms']:
        pi=('XX','YY').index(term['polarization']);ai=int(np.flatnonzero(order==term['antenna'])[0])
        delay[pi,ai]=term['relative_delay_ns'];coherence[pi,ai]=term['within_coarse_phase_coherence']
        fpga[ai]=term['fpga']
    # Joint fit of the FPGA common delay to both polarizations and all reliable
    # receptors. This is a circular least-squares fit modulo the fine-bin alias.
    common=np.zeros(4)
    for board in range(4):
        use=(fpga[None,:]==board)&(coherence>.55)&np.isfinite(delay)
        weight=coherence[use]**2
        common[board]=np.angle(np.sum(weight*np.exp(2j*np.pi*delay[use]/period))) * period/(2*np.pi)
    common=wrap(common-common[1],period)
    antenna=np.full_like(delay,np.nan)
    for pi in range(2):
        antenna[pi]=wrap(delay[pi]-common[fpga],period)
    variants={'raw':None,'independent_slopes':1.0,'half_cable_slope':.5,'fpga_only':0.0}
    scores={}
    for name,shrink in variants.items():
        corrected=np.empty_like(gains)
        if shrink is None:
            corrected[:]=gains
        else:
            for pi in range(2):
                for ai in range(36):
                    if not np.isfinite(delay[pi,ai]):
                        corrected[:,pi,ai]=gains[:,pi,ai]
                        continue
                    tau=common[fpga[ai]]+shrink*antenna[pi,ai]
                    for coarse in range(40):
                        sl=slice(coarse*28,(coarse+1)*28)
                        f=freq[sl]; center=float(np.mean(f));use=good[sl,pi]
                        original=gains[sl,pi,ai];unit=original/np.maximum(abs(original),1e-30)
                        if use.sum()<3:
                            corrected[sl,pi,ai]=original
                            continue
                        derotated=unit[use]*np.exp(-2j*np.pi*1e-9*tau*(f[use]-center))
                        phase=np.angle(derotated.sum())
                        corrected[sl,pi,ai]=abs(original)*np.exp(
                            1j*(phase+2*np.pi*1e-9*tau*(f-center)))
        values=np.full((len(freq),2,3),np.nan)
        for ci in np.flatnonzero(joint):
            for pi in range(2):
                gain=corrected[ci,pi]
                for wi in range(3):
                    predicted=model[wi,ci,pi]*gain[:,None]*gain.conj()[None,:]
                    values[ci,pi,wi]=score(data[wi,ci,pi],predicted,selected)
        scores[name]=values
    report={'fit':str(args.fit),'transfer':str(args.transfer),
            'delay_alias_period_ns':period,'joint_fpga_delay_ns_relative_fpga1':common.tolist(),
            'gauge':'FPGA term is weighted circular mean of fitted receptor slopes; cable term is the residual',
            'transfer_scores_on_training_joint_channels':{
                name:{pol:np.nanmedian(values[:,pi],axis=0).tolist()
                      for pi,pol in enumerate(('XX','YY'))}
                for name,values in scores.items()}}
    (args.output_dir/'hierarchical_delay_test.json').write_text(json.dumps(report,indent=2)+'\n')
    fig,axes=plt.subplots(2,1,figsize=(13,7),sharex=True,constrained_layout=True)
    centers=freq.reshape(40,28).mean(axis=1)/1e6
    for pi,pol in enumerate(('XX','YY')):
        for name,values in scores.items():
            axes[pi].plot(centers,np.nanmedian(values[:,pi,1].reshape(40,28),axis=1),label=name)
        axes[pi].set(title=f'{pol}: independent transfer, middle window',ylabel='Real normalized coherence')
        axes[pi].grid(alpha=.3);axes[pi].legend(fontsize=8)
    axes[-1].set_xlabel('Frequency (MHz)')
    fig.suptitle('Per-antenna phase slopes = FPGA common + cable residual; test on independent capture')
    fig.savefig(args.output_dir/'hierarchical_delay_transfer.png',dpi=170)
    plt.close(fig)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
