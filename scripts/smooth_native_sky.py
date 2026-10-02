#!/usr/bin/env python3
"""Validate smooth instrumental phases using native calibration diagnostics.

Whole coarse channels are withheld from the phase fit. The optional later
capture must be processed by calibrate_native_sky.py on the same frequency grid.
This tool writes diagnostics, not automatically approved observing weights.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from fine_frequency_axis import science_channel_mask


def fit_phase_curve(frequency, gains, selected, degree=1, max_delay_s=2e-6,
                    coarse_offset=None):
    """Fit wrapped phases without unwrapping across flagged frequency gaps.

    A delay search initializes a robust circular polynomial fit. Frequency is
    scaled to [-1, 1]; coefficients are ascending powers in radians.
    """
    frequency = np.asarray(frequency)
    centre = .5*(frequency[0]+frequency[-1])
    span = .5*np.ptp(frequency)
    x = (frequency-centre)/span
    valid = selected & np.isfinite(gains) & (abs(gains)>0)
    if valid.sum() < max(10, 4*(degree+1)):
        raise ValueError('too few independent phase samples')
    z = gains[valid]/abs(gains[valid])
    search_frequency=frequency
    search_z=z
    local_slope=0.
    coarse_width=0.
    if coarse_offset is not None:
        adjacent=valid[1:]&valid[:-1]&(np.diff(coarse_offset)>0)&(np.diff(frequency)>0)
        if adjacent.sum()>=3:
            step=np.diff(coarse_offset)[adjacent]
            coarse_width=float(np.median(np.diff(frequency)[adjacent]/step))
            # Estimate the within-coarse slope FIRST. It can wrap through many
            # turns per coarse channel, making a search over the whole band
            # alone select the wrong alias and trap the nonlinear fit.
            phase_step=np.angle(gains[1:]*gains[:-1].conj())[adjacent]
            local_slope=float(np.median(phase_step/step))
            search_frequency=frequency-coarse_offset*coarse_width
            search_z=z*np.exp(-1j*local_slope*coarse_offset[valid])
    delay = np.linspace(-max_delay_s, max_delay_s,
                        int(np.ceil(16*max_delay_s*np.ptp(frequency)))+1)
    coherent = np.exp(-2j*np.pi*delay[:,None]*(search_frequency[valid]-centre)) @ search_z
    best = np.argmax(abs(coherent))
    initial = np.zeros(degree+1)
    initial[:2] = [np.angle(coherent[best]), 2*np.pi*delay[best]*span]
    design = np.polynomial.polynomial.polyvander(x[valid], degree)
    full_design = np.polynomial.polynomial.polyvander(x, degree)
    if coarse_offset is not None:
        # A channelized receiver may have a phase ramp that resets at each
        # coarse boundary. Fit one repeated ramp, not an independent phase for
        # each coarse channel; entire held-out coarse channels remain tests.
        design=np.column_stack((design,coarse_offset[valid]))
        full_design=np.column_stack((full_design,coarse_offset))
        initial=np.r_[initial,local_slope-2*np.pi*delay[best]*coarse_width]
    def residual(coeff):
        r = np.exp(1j*(design@coeff))*z.conj()-1
        return np.r_[r.real,r.imag]
    fit = least_squares(residual,initial,loss='soft_l1',f_scale=.15,
                        max_nfev=200)
    return np.exp(1j*(full_design@fit.x)),fit.x


def fit_amplitudes(data, model, phases, mask):
    """Fit positive nuisance amplitudes while keeping every phase fixed."""
    predicted = model*phases[:,None]*phases[None,:].conj()
    cross = np.where(mask,(data*predicted.conj()).real,0)
    power = np.where(mask,abs(predicted)**2,0)
    scale = np.sqrt(np.sum(abs(data[mask])*abs(predicted[mask])) /
                    max(np.sum(power),1e-30))
    amp = np.full(len(phases),scale)
    for _ in range(200):
        new = np.maximum((cross@amp)/np.maximum(power@amp**2,1e-30),0)
        new = .5*(new+amp)
        if np.linalg.norm(new-amp)<1e-7*max(np.linalg.norm(amp),1e-30):
            amp=new
            break
        amp=new
    return amp


def cosine(data, predicted, mask):
    d,p=data[...,mask],predicted[...,mask]
    return np.sum((p.conj()*d).real,axis=-1)/np.maximum(
        np.sqrt(np.sum(abs(p)**2,axis=-1)*np.sum(abs(d)**2,axis=-1)),1e-30)


def export_phase_weights(template, phases, antenna_order, frequencies, good, prefix, report):
    """Preserve physical receptor labels while exporting unit voltage gains."""
    document=json.loads(Path(template).read_text())
    keys=sorted(document['weights'],key=int)
    if len(keys)!=len(frequencies):
        raise ValueError('template and phase solution have different channel counts')
    metadata=document['provenance']
    selected=science_channel_mask(frequencies)
    np.testing.assert_allclose([metadata['frequency_first_hz'],metadata['frequency_last_hz']],
                               frequencies[selected][[0,-1]],atol=1e-3,rtol=0)
    for ci,key in enumerate(keys):
        for pi,pol in enumerate(('XX','YY')):
            for ant,physical_id in enumerate(antenna_order):
                slot=str(physical_id)
                if slot in document['weights'][key][pol]:
                    g=phases[ci,pi,ant]
                    document['weights'][key][pol][slot]={'real':float(g.real),'imag':float(g.imag)}
    metadata['calibration_kind']='smooth phase only; repeated coarse ramp' if report['coarse_ramp'] else 'smooth phase only'
    metadata['smooth_phase_degree']=report['degree']
    metadata['channel_mask_required']=True
    metadata['independent_capture_validated']=True
    metadata['validated_joint_bandwidth_mhz']=report['smooth_joint_bandwidth_mhz']
    unique=science_channel_mask(frequencies)
    mask={'frequency_hz':frequencies[unique].tolist(),'good':good.all(axis=1)[unique].tolist(),
          'runtime_channel_order':'ascending frequency','required':True}
    destinations=[(Path(str(prefix)+'_gains.json'),document),
                  (Path(str(prefix)+'_channel_mask.json'),mask)]
    if any(path.exists() for path,_ in destinations):
        raise FileExistsError('refusing to overwrite exported observing files')
    for path,contents in destinations:path.write_text(json.dumps(contents,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('validation',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--transfer',type=Path)
    p.add_argument('--degree',type=int,choices=(1,2,3),default=1)
    p.add_argument('--coarse-ramp',action='store_true',help='Fit one repeating within-coarse phase ramp per receptor')
    p.add_argument('--export-template',type=Path,help='Native gain JSON providing physical labels; requires independent transfer validation')
    args=p.parse_args()
    if args.export_template and not args.transfer:p.error('export requires --transfer')
    original=np.load(args.validation)
    freq=original['frequencies_hz'];nf=len(freq)
    phase_train=original['good'].copy()
    held_frequency=(np.arange(nf)//28)%5==2
    phase_train[held_frequency]=False
    live=original['selected'].any(axis=0)|original['selected'].any(axis=1)
    phases=np.ones_like(original['gains'])
    coefficients=np.zeros((2,36,args.degree+1+int(args.coarse_ramp)))
    coarse_offset=(np.arange(nf)%28-13.5)/27 if args.coarse_ramp else None
    phase_error=np.zeros_like(phases.real)
    reference_ids={}
    for pi in range(2):
        # The absolute phase is unconstrained. Re-reference before smoothing
        # to a consistently strong live receptor, rather than inheriting a
        # possibly decorrelated reference from the independent channel fits.
        strength=np.median(abs(original['gains'][phase_train[:,pi],pi]),axis=0)
        reference_index=int(np.argmax(np.where(live,strength,-1)))
        reference_ids[('XX','YY')[pi]]=int(original['antenna_order'][reference_index])
        reference=original['gains'][:,pi,reference_index]
        referenced=original['gains'][:,pi]*reference.conj()[:,None]
        for ant in np.flatnonzero(live):
            phases[:,pi,ant],coefficients[pi,ant]=fit_phase_curve(
                freq,referenced[:,ant],phase_train[:,pi],args.degree,
                coarse_offset=coarse_offset)
        phase_error[:,pi]=np.angle(phases[:,pi]*referenced.conj())
    data=original['data'];model=original['model'];train=original['train']
    gains=phases.copy()
    metrics=np.full((nf,2,3),np.nan)
    for pi in range(2):
        for ci in range(nf):
            if not np.isfinite(data[:,ci,pi][:,train|original['selected']]).all():continue
            gains[ci,pi]*=fit_amplitudes(data[0,ci,pi],model[0,ci,pi],phases[ci,pi],train)
        predicted=model[:,:,pi]*gains[None,:,pi,:,None]*gains[None,:,pi,None,:].conj()
        metrics[:,pi,0]=cosine(data[0,:,pi],predicted[0],original['withheld'])
        metrics[:,pi,1]=cosine(data[1,:,pi],predicted[1],original['selected'])
        metrics[:,pi,2]=cosine(data[2,:,pi],predicted[2],original['selected'])
    good=(metrics[:,:,0]>.55)&(metrics[:,:,1]>.65)&(metrics[:,:,2]>.65)
    edge=np.arange(nf)%28==0
    good[edge]=False
    transfer_metrics=np.array([])
    if args.transfer:
        later=np.load(args.transfer)
        np.testing.assert_allclose(later['frequencies_hz'],freq,atol=1e-3,rtol=0)
        np.testing.assert_array_equal(later['antenna_order'],original['antenna_order'])
        prediction=later['model']*gains[None,:,:,:,None]*gains.conj()[None,:,:,None,:]
        transfer_metrics=cosine(later['data'],prediction,original['selected'])
        good&=(np.min(transfer_metrics,axis=0)>.65)
    unique=science_channel_mask(freq)
    df=(freq[1]-freq[0])/1e6
    report={'degree':args.degree,'coarse_ramp':args.coarse_ramp,'reference_antenna_by_hand':reference_ids,'phase_training':'accepted channels excluding every fifth coarse channel',
        'phase_training_channels':phase_train.sum(axis=0).tolist(),
        'original_joint_bandwidth_mhz':float(np.sum(original['joint']&unique)*df),
        'smooth_joint_bandwidth_mhz':float(np.sum(good.all(axis=1)&unique)*df),
        'smooth_bandwidth_per_hand_mhz':(np.sum(good&unique[:,None],axis=0)*df).tolist(),
        'held_frequency_phase_error_deg':{pol:float(np.rad2deg(np.median(abs(phase_error[held_frequency&original['good'][:,pi],pi][:,live])))) for pi,pol in enumerate(('XX','YY'))},
        'median_validation_on_original_good':{pol:np.nanmedian(metrics[original['good'][:,pi],pi],axis=0).tolist() for pi,pol in enumerate(('XX','YY'))},
        'transfer_median_on_original_good':{pol:np.nanmedian(transfer_metrics[:,original['good'][:,pi],pi],axis=1).tolist() for pi,pol in enumerate(('XX','YY'))} if args.transfer else None}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(str(args.output)+'.npz',phases=phases,gains=gains,
        coefficients=coefficients,metrics=metrics,good=good,phase_error=phase_error,
        transfer_metrics=transfer_metrics,phase_train=phase_train,frequencies_hz=freq)
    Path(str(args.output)+'.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    if args.export_template:
        if report['smooth_joint_bandwidth_mhz']<1:
            raise ValueError('less than 1 MHz passes independent validation; no weights exported')
        export_phase_weights(args.export_template,phases,original['antenna_order'],
                             freq,good,args.output,report)


if __name__=='__main__':main()
