#!/usr/bin/env python3
"""Fit native voltage phases against a diffuse sky, with independent validation.

Uses the existing Haslam/AEP predictor and a scaled alternating least-squares
solver. The output channel mask is mandatory: gains interpolated across rejected
channels are not validated measurements. Requires lambda_commissioning.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz, get_sun
from astropy import units as u
from astropy.utils import iers
from scipy.interpolate import interp1d
from scipy.special import j1

from diagnose_sky_calibration import fit_gains, score
from lambda_commissioning.modelling import calc_sky_model, calc_sun_model, calc_lmn
from lambda_commissioning.capture_timing import timing_pcap_fpga, visibility_clock
from lambda_commissioning.point_calibration import integration_centres
from lambda_commissioning.utils import (_LAMBDA_PACKET_DTYPE, get_udp_payload_bytes,
                                         make_telescope_model, read_hdf5_time_freq)
from fine_frequency_axis import (processing_frequencies_hz,
                                 science_channel_indices,
                                 unique_processing_frequencies_hz)
from lambda_commissioning.constants import NARRIBRI


def solar_model(stamp, frequency_hz, array, location, radius_deg):
    """Unit-flux Sun, optionally as a uniform disk, in the runtime array frame."""
    instant=Time([stamp.jd],format='jd')
    model=calc_sun_model(instant,frequency_hz,array,Sapp=1,location=location)[0]
    if radius_deg==0:
        return model
    solar=get_sun(instant).transform_to(AltAz(obstime=instant,location=location))
    l,m,n=calc_lmn(solar.alt.deg,solar.az.deg)
    l,m,n=float(l[0]),float(m[0]),float(n[0])
    baseline_squared=array.uu_m**2+array.vv_m**2+array.ww_m**2
    radial=array.uu_m*l+array.vv_m*m+array.ww_m*n
    projected=np.sqrt(np.maximum(0,baseline_squared-radial**2))
    x=2*np.pi*np.deg2rad(radius_deg)*projected*frequency_hz/299792458.0
    disk=np.ones_like(x)
    nonzero=x>1e-10
    disk[nonzero]=2*j1(x[nonzero])/x[nonzero]
    return model*disk


def select_parallel_hand(block, antenna_a, antenna_b, swapped, hand):
    """Select the recorded product for a physical parallel-hand baseline.

    ``block`` is [frequency, baseline, recorded hand, recorded hand, real/imag].
    Each antenna can independently reverse its two recorded hands. No
    conjugation or global XX/YY convention change is performed here.
    """
    first=np.where(swapped[antenna_a],1-hand,hand)
    second=np.where(swapped[antenna_b],1-hand,hand)
    result=np.empty(block.shape[:2],complex)
    for hp in (0,1):
        for hq in (0,1):
            use=(first==hp)&(second==hq)
            if np.any(use):
                result[:,use]=(block[:,use,hp,hq,0]
                               +1j*block[:,use,hp,hq,1])
    return result


def packet_clock_times(capture, timing_pcap, location):
    """Use the audited FPGA delay snapshot to locate integration centres in UTC."""
    packet_times, payloads = get_udp_payload_bytes(str(timing_pcap), packets=12000)
    packets = np.frombuffer(b"".join(payloads), dtype=_LAMBDA_PACKET_DTYPE)
    if not len(packets):
        raise ValueError("empty timing PCAP")
    epochs = np.asarray(packet_times, dtype=float) - packets["seq_no"] * 1.08e-6
    with h5py.File(capture) as hdf:
        manifest = json.loads(hdf["audit/run_manifest_json"][()])
        clock = visibility_clock(manifest, float(np.median(epochs)),
                                 timing_pcap_fpga(timing_pcap))
        sequence = hdf["vis_seq_nums"][:]
    first_start = Time(clock["epoch_unix"] + int(sequence[0, 0]) * 1.08e-6,
                       format="unix", scale="utc").mjd
    centres, _ = integration_centres(sequence, first_start)
    return Time(centres, format="mjd", scale="utc", location=location)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('capture',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--antenna-id-swaps',default='')
    p.add_argument('--swap-pol-antennas',default='',
                   help='Additional swap of RECORDED hands for these antenna IDs; '
                        'swapping an ID already reversed by the captured input map undoes that reversal')
    p.add_argument('--samples',type=int,default=30)
    p.add_argument('--exclude-antennas',default='14')
    p.add_argument('--nside',type=int,default=64)
    p.add_argument('--min-baseline',type=float,default=7)
    p.add_argument('--config-file',type=Path,help='Use the exact runtime array location')
    p.add_argument('--timing-pcap',type=Path,required=True,
                   help='Single-FPGA timing PCAP from the capture boot; HDF5 mjd_start is only writer construction time')
    p.add_argument('--model',choices=('haslam','sun-point','sun-disk'),default='haslam')
    p.add_argument('--sun-radius-deg',type=float,default=0.27,
                   help='Uniform radio-Sun radius for sun-disk model (default: 0.27 degrees)')
    p.add_argument('--model-only',action='store_true',help='Save independent data/model windows without fitting gains')
    args=p.parse_args()
    if not np.isfinite(args.sun_radius_deg) or args.sun_radius_deg<0:
        p.error('--sun-radius-deg must be finite and nonnegative')
    args.output.mkdir(parents=True,exist_ok=True)
    iers.conf.auto_download=False; iers.conf.auto_max_age=None
    location=NARRIBRI
    if args.config_file:
        site=json.loads(args.config_file.read_text())['array_location']
        location=EarthLocation.from_geodetic(site['longitude_deg']*u.deg,
            site['latitude_deg']*u.deg,site['height_m']*u.m)
    _,freqs=read_hdf5_time_freq(str(args.capture));nf=len(freqs);ns=args.samples
    times=packet_clock_times(args.capture,args.timing_pcap,location)
    with h5py.File(args.capture) as capture:
        min_channel=int(capture.attrs['min_channel'])
        coarse_channels=int(capture.attrs['max_channel'])-min_channel+1
        audit=json.loads(capture['audit/run_manifest_json'][()])
        captured_hand_swaps=sorted({row['antenna_id'] for row in audit['forward_mapping']
            if row['antenna_id']>0 and row['raw_polarization']!=row['canonical_polarization']})
    fine_grid=nf==coarse_channels*28
    unique_fine_grid=nf==coarse_channels*27
    if fine_grid:
        freqs=processing_frequencies_hz(min_channel,coarse_channels)
    elif unique_fine_grid:
        freqs=unique_processing_frequencies_hz(min_channel,coarse_channels)
    starts=[3,len(times)//2-ns//2,len(times)-ns-3]
    if starts[1]<starts[0]+ns or starts[2]<starts[1]+ns:
        p.error('capture too short for three independent windows')
    windows=[np.arange(s,s+ns) for s in starts]
    stamps=Time([times[w].jd.mean() for w in windows],format='jd')
    order=np.arange(36)
    for pair in filter(None,args.antenna_id_swaps.split(',')):
        a,b=[int(v)-1 for v in pair.split(':')];order[a],order[b]=order[b],order[a]
    excluded={int(v) for v in args.exclude_antennas.split(',') if v}
    swapped_hands={int(v) for v in args.swap_pol_antennas.split(',') if v}
    if any(v<1 or v>36 for v in swapped_hands):
        p.error('--swap-pol-antennas IDs must be 1..36')
    print('Hand reversals already applied at capture:',captured_hand_swaps,flush=True)
    print('Additional recorded-hand reversals:',sorted(swapped_hands),flush=True)
    live=np.array([v+1 not in excluded for v in order])
    raw=np.zeros((3,nf,2,36,36),complex)
    with h5py.File(args.capture) as f:
        min_channel=int(f.attrs['min_channel'])
        pairs=f['baseline_antenna_ids'][:]
        good=(pairs>0).all(axis=1)&(pairs<=36).all(axis=1)
        a,b=pairs[good].T-1
        hand_swapped=np.isin(np.arange(36)+1,list(swapped_hands))
        for wi,win in enumerate(windows):
            v=np.zeros((nf,len(a),2),complex)
            for ti in win:
                block=f['visibilities'][int(ti)][:,good]
                for pi,hand in enumerate((1,0)):
                    v[:,:,pi]+=select_parallel_hand(
                        block,a,b,hand_swapped,hand)
            for pi in range(2):
                raw[wi,:,pi,a,b]=(v[:,:,pi]/ns).T
                raw[wi,:,pi,b,a]=(v[:,:,pi].conj()/ns).T
            print('read window',wi,stamps[wi].isot,flush=True)
    auto=raw.diagonal(axis1=-2,axis2=-1).real
    norm=np.sqrt(np.maximum(auto,1e-30))
    data=raw/(norm[..., :,None]*norm[...,None,:])
    array=make_telescope_model(np.arange(1,37))
    lengths=np.hypot(array.uu_m,array.vv_m)[order][:,order]
    upper=np.triu(np.ones((36,36),bool),1)
    selected=upper&(lengths>args.min_baseline)&live[:,None]&live[None,:]
    ii,jj=np.where(selected)
    withheld=np.zeros_like(selected);withheld[ii[::5],jj[::5]]=True
    train=selected&~withheld;train|=train.T
    node_indices=np.unique(np.r_[np.arange(0,nf,14),nf-1])
    nodes=freqs[node_indices]
    model=np.zeros_like(raw)
    for wi,stamp in enumerate(stamps):
        for pi,pol in enumerate(('XX','YY')):
            mats=[]
            for fi,freq in enumerate(nodes):
                if args.model=='haslam':
                    m=calc_sky_model(stamp,freq,array,pol,nside=args.nside,location=location)
                else:
                    m=solar_model(stamp,freq,array,location,
                                  0 if args.model=='sun-point' else args.sun_radius_deg)
                mats.append(m/np.diag(m).real.mean())
            model[wi,:,pi]=interp1d(nodes,np.asarray(mats),axis=0)(freqs)[:,order][:,:,order]
            print('modelled',wi,pol,len(nodes),'frequency nodes',flush=True)
    if args.model_only:
        np.savez_compressed(args.output/'validation.npz',frequencies_hz=freqs,
            times_jd=stamps.jd,antenna_order=order+1,model=model,data=data,
            train=train,withheld=withheld,selected=selected)
        print('Saved independent data/model validation windows; no gains fitted',flush=True)
        return
    gains=np.ones((nf,2,36),complex)
    later_gains=np.ones_like(gains)
    metrics=np.zeros((nf,2,5))
    phase_change=np.zeros((nf,2,36))
    for pi,pol in enumerate(('XX','YY')):
        for ci in range(nf):
            if not np.isfinite(data[:,ci,pi][:,selected|selected.T]).all():
                metrics[ci,pi]=np.nan
                phase_change[ci,pi]=np.pi
                continue
            g=fit_gains(data[0,ci,pi],model[0,ci,pi],train)
            gl=fit_gains(data[2,ci,pi],model[2,ci,pi],train)
            gains[ci,pi]=g;later_gains[ci,pi]=gl
            predicted=model[:,ci,pi]*g[None,:,None]*g.conj()[None,None,:]
            metrics[ci,pi]=[score(data[0,ci,pi],predicted[0],train&upper),
                score(data[0,ci,pi],predicted[0],withheld),
                score(data[1,ci,pi],predicted[1],selected),
                score(data[2,ci,pi],predicted[2],selected),
                float(np.median(abs(g[live])))]
            delta=np.angle(gl*g.conj());delta-=np.angle(np.mean(np.exp(1j*delta[live])))
            phase_change[ci,pi]=np.angle(np.exp(1j*delta))
        print('solved',pol,flush=True)
    # Thresholds concern prediction of data never used in the gain fit.
    good=(metrics[:,:,1]>.55)&(metrics[:,:,2]>.65)&(metrics[:,:,3]>.65)
    good&=np.median(abs(phase_change[:,:,live]),axis=2)<np.deg2rad(15)
    # The science axis retains original GPU bins 3..29. Bin 2 is lower in
    # measured power and duplicates the previous coarse's bin 29.
    edge=(np.arange(nf)%28==0) if fine_grid else np.zeros(nf,bool)
    good[edge]=False
    physical_excluded=[]
    for pi,pol in enumerate(('XX','YY')):
        if good[:,pi].sum()<10:
            print('INSUFFICIENT accepted channels',pol,int(good[:,pi].sum()),flush=True)
            continue
        stability=np.median(abs(phase_change[good[:,pi],pi]),axis=0)
        weak=np.median(abs(gains[good[:,pi],pi]),axis=0)
        keep=live&(stability<np.deg2rad(25))&(weak>.08*np.median(weak[live]))
        for ant in np.flatnonzero(live&~keep):physical_excluded.append({'antenna_id':int(order[ant]+1),'polarization':pol})
    joint=good.all(axis=1)
    # Save all fit evidence before exporting an operational artifact.
    np.savez_compressed(args.output/'validation.npz',gains=gains,later_gains=later_gains,
        metrics=metrics,phase_change=phase_change,good=good,joint=joint,
        frequencies_hz=freqs,times_jd=stamps.jd,antenna_order=order+1,
        model=model,data=data,train=train,withheld=withheld,selected=selected)
    unique=np.zeros(nf,bool)
    if fine_grid:
        unique[science_channel_indices(coarse_channels)]=True
    else:
        unique[:]=True
    bandwidth=float(np.sum(joint&unique)*(freqs[1]-freqs[0])/1e6)
    report={'capture':str(args.capture),'utc':stamps.isot.tolist(),
        'antenna_id_swaps':args.antenna_id_swaps,
        'swap_pol_antennas':sorted(swapped_hands),
        'captured_input_hand_reversals':captured_hand_swaps,
        'effective_raw_hand_reversals':sorted(set(captured_hand_swaps)^swapped_hands),
        'model':('Haslam + receptor AEP; no duplicate point sources' if args.model=='haslam'
                 else args.model),'sun_radius_deg':args.sun_radius_deg if args.model=='sun-disk' else None,
        'min_baseline_m':args.min_baseline,'excluded_antennas':sorted(excluded),
        'excluded_receptors':physical_excluded,'joint_usable_bandwidth_mhz':bandwidth,
        'accepted_channels':{pol:int(good[:,pi].sum()) for pi,pol in enumerate(('XX','YY'))},
        'median_validation':{pol:[float(v) for v in np.median(metrics[good[:,pi],pi,:4],axis=0)]
                              if good[:,pi].any() else None for pi,pol in enumerate(('XX','YY'))},
        'metric_order':['fit','withheld_baselines','withheld_middle_time','withheld_late_time'],
        'required_runtime_polarization_order':['XX','YY'],
        'channel_mask_required':True}
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    if bandwidth<1 or any(good[:,pi].sum()<10 for pi in range(2)):
        raise RuntimeError('insufficient validated bandwidth; diagnostic evidence saved, no beam gains exported')
    # Keep phase only: avoid amplifying weak receptor noise. Rejected frequencies
    # receive interpolated phases solely to satisfy the runtime's complete table;
    # they MUST be removed using the accompanying mask after folding.
    effective=np.ones_like(gains)
    for pi in range(2):
        for ant in range(36):
            values=gains[good[:,pi],pi,ant]
            phases=np.unwrap(np.angle(values))
            effective[:,pi,ant]=np.exp(1j*np.interp(freqs,freqs[good[:,pi]],phases))
    weights={}
    for ci in range(nf):
        record={}
        for pi,pol in enumerate(('XX','YY')):
            record[pol]={str(order[a]+1):{'real':float(effective[ci,pi,a].real),'imag':float(effective[ci,pi,a].imag)} for a in range(36) if live[a]}
        weights[str(min_channel+ci)]=record
    provenance={'voltage_gain_convention':'native HDF5 covariance V_ij = g_i M_ij conj(g_j)',
        'calibration_kind':'phase only','required_runtime_polarization_order':['XX','YY'],
        'unique_frequency_channels':int(unique.sum()),'processing_channels_with_boundary_duplicates':nf,
        'frequency_first_hz':float(freqs[unique][0]),'frequency_last_hz':float(freqs[unique][-1]),
        'channel_mask_required':True,'excluded_antennas':sorted(excluded),
        'antenna_id_swaps':args.antenna_id_swaps,
        'captured_input_hand_reversals':captured_hand_swaps,
        'effective_raw_hand_reversals':sorted(set(captured_hand_swaps)^swapped_hands),
        'swap_pol_antennas':sorted(swapped_hands)}
    (args.output/'native_voltage_gains.json').write_text(json.dumps({'channel_axis':'fine_processing_index',
        'excluded_receptors':physical_excluded,'weights':weights,'provenance':provenance},indent=2)+'\n')
    unique_good=joint[unique]
    (args.output/'rejected_unique_channels.txt').write_text(' '.join(str(x) for x in np.flatnonzero(~unique_good))+'\n')
    (args.output/'channel_mask.json').write_text(json.dumps({'frequency_hz':freqs[unique].tolist(),
        'good':unique_good.tolist(),'runtime_channel_order':'ascending frequency','required':True},indent=2)+'\n')


if __name__=='__main__':main()
