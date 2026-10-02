#!/usr/bin/env python3
"""Make phase-calibrated solar dirty and small-window Högbom CLEAN images.

This diagnostic uses a tangent-plane DFT of native normalized covariances,
equal visibility weights, and a CLEAN mask centered on the known Sun. Images
are relative units, not a primary-beam-corrected flux measurement.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import AltAz, EarthLocation, get_sun
from astropy.time import Time
from astropy import units as u
from astropy.utils import iers
from scipy.ndimage import gaussian_filter

from lambda_commissioning.utils import make_telescope_model
from fine_frequency_axis import science_channel_mask


def dirty_image(ucoord, vcoord, visibility, axis):
    yy, xx = np.meshgrid(axis, axis, indexing='ij')
    pixels = np.column_stack((xx.ravel(), yy.ravel()))
    result = np.zeros(len(pixels), dtype=float)
    for start in range(0, len(pixels), 256):
        offset = pixels[start:start + 256]
        phase = 2j * np.pi * (ucoord[:, None] * offset[None, :, 0] +
                              vcoord[:, None] * offset[None, :, 1])
        result[start:start + 256] = np.real(visibility @ np.exp(phase))
    return result.reshape(xx.shape) / len(visibility)


def clean(dirty, psf, axis, clean_radius_deg=4, gain=.1, iterations=400):
    n = len(axis); center = n // 2
    x, y = np.meshgrid(axis, axis)
    mask = x*x + y*y < np.deg2rad(clean_radius_deg)**2
    residual = dirty.copy(); components = np.zeros_like(dirty)
    initial = abs(dirty[mask]).max()
    count = 0
    for count in range(iterations):
        masked = np.where(mask, abs(residual), -1)
        iy, ix = np.unravel_index(np.argmax(masked), masked.shape)
        peak = residual[iy, ix]
        if abs(peak) < .04 * initial:
            break
        component = gain * peak
        components[iy, ix] += component
        y0 = max(0, iy-center); y1 = min(n, iy+center+1)
        x0 = max(0, ix-center); x1 = min(n, ix+center+1)
        residual[y0:y1, x0:x1] -= component * psf[
            y0-iy+center:y1-iy+center, x0-ix+center:x1-ix+center]
    # Estimate the central-lobe width from the PSF second moment within 2 deg.
    inner = (x*x+y*y < np.deg2rad(2)**2) & (psf > 0)
    beam_sigma = np.sqrt(np.sum(psf[inner]*(x[inner]**2+y[inner]**2)) /
                            max(2*np.sum(psf[inner]), 1e-30))
    sigma_pixels = max(1.0, beam_sigma / (axis[1]-axis[0]))
    restored = gaussian_filter(components, sigma_pixels) * (2*np.pi*sigma_pixels**2) + residual
    return components, residual, restored, count + 1, sigma_pixels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('validation', type=Path)
    parser.add_argument('report', type=Path)
    parser.add_argument('config', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--channels', type=int, default=64)
    # The restored FWHM is about 2.2 deg here: 0.5-deg pixels give roughly
    # 4--5 pixels across the beam without oversampling a narrow CLEAN spike.
    parser.add_argument('--pixels', type=int, default=49)
    args = parser.parse_args()
    iers.conf.auto_download=False; iers.conf.auto_max_age=None
    with np.load(args.validation) as fit:
        freq=fit['frequencies_hz'];good=fit['joint'];order=fit['antenna_order']
        selected=fit['selected'];gains=fit['gains'];data=fit['data'][2]
        stamp=Time(fit['times_jd'][2],format='jd')
    report=json.loads(args.report.read_text())
    unique=science_channel_mask(freq)
    available=np.flatnonzero(good&unique)
    if not len(available):
        parser.error('no jointly accepted channels to image')
    channels=available[np.unique(np.linspace(0,len(available)-1,
                                              min(args.channels,len(available))).astype(int))]
    site=json.loads(args.config.read_text())['array_location']
    location=EarthLocation.from_geodetic(site['longitude_deg']*u.deg,
        site['latitude_deg']*u.deg,site['height_m']*u.m)
    sun=get_sun(stamp).transform_to(AltAz(obstime=stamp,location=location))
    alt,az=sun.alt.rad,sun.az.rad
    # East, north, up basis with tangent vectors in azimuth and altitude.
    east=np.array([np.cos(alt)*np.sin(az),np.cos(alt)*np.cos(az),np.sin(alt)])
    e_az=np.array([np.cos(az),-np.sin(az),0.])
    e_alt=np.array([-np.sin(alt)*np.sin(az),-np.sin(alt)*np.cos(az),np.cos(alt)])
    array=make_telescope_model(np.arange(1,37));idx=order-1
    baseline=np.stack((array.uu_m[idx][:,idx],array.vv_m[idx][:,idx],
                       array.ww_m[idx][:,idx]),axis=-1)
    axis=np.linspace(-np.deg2rad(12),np.deg2rad(12),args.pixels)
    planes=[];stats={}
    for pi,pol in enumerate(('XX','YY')):
        use=selected.copy()
        excluded={item['antenna_id'] for item in report['excluded_receptors']
                  if item['polarization']==pol}
        # The independent transfer capture identified A30 X as unstable.
        if pol=='XX': excluded.add(30)
        for ant in excluded:
            ai=np.flatnonzero(order==ant)
            if len(ai): use[ai[0],:]=False;use[:,ai[0]]=False
        ii,jj=np.where(use)
        b=baseline[ii,jj]
        projected_u=b @ e_az;projected_v=b @ e_alt
        u_list=[];v_list=[];vis_list=[]
        for ci in channels:
            wave=freq[ci]/299792458.
            phase=np.exp(2j*np.pi*wave*(b@east-b[:,2]))
            gain=gains[ci,pi];unit=gain/np.maximum(abs(gain),1e-30)
            corrected=data[ci,pi,ii,jj]*unit[ii].conj()*unit[jj]*phase
            u_list.append(projected_u*wave);v_list.append(projected_v*wave)
            vis_list.append(corrected)
        uu=np.concatenate(u_list);vv=np.concatenate(v_list);vis=np.concatenate(vis_list)
        dirty=dirty_image(uu,vv,vis,axis)
        psf=dirty_image(uu,vv,np.ones_like(vis),axis)
        components,residual,restored,niter,sigma=clean(dirty,psf,axis)
        planes.append((dirty,psf,components,residual,restored))
        stats[pol]={'baselines':len(ii),'channels':len(channels),
                    'sun_elevation_deg':float(sun.alt.deg),
                    'dirty_peak':float(dirty.max()),
                    'pixel_deg':float(np.rad2deg(axis[1]-axis[0])),
                    'residual_rms_outside_clean_window':float(np.std(residual[
                        np.hypot(*np.meshgrid(axis,axis))>np.deg2rad(4)])),
                    'clean_iterations':niter,'clean_beam_sigma_deg':float(np.rad2deg(sigma*(axis[1]-axis[0])))}
    fig,axs=plt.subplots(2,5,figsize=(18,7),sharex=True,sharey=True,constrained_layout=True)
    extent=np.rad2deg([axis[0],axis[-1],axis[0],axis[-1]])
    for pi,pol in enumerate(('XX','YY')):
        dirty,psf,components,residual,restored=planes[pi]
        for col,(title,image) in enumerate((('Dirty',dirty),('PSF',psf),
                                            ('CLEAN components',components),
                                            ('Residual',residual),
                                            ('Restored',restored))):
            axs[pi,col].imshow(image,origin='lower',extent=extent,cmap='RdBu_r')
            axs[pi,col].set_title(f'{pol}: {title}')
            axs[pi,col].plot(0,0,'k+',ms=8)
            axs[pi,col].set_xlabel('Azimuth tangent offset (deg)')
            axs[pi,col].set_ylabel('Altitude tangent offset (deg)')
    fig.suptitle(f'Sun at {stamp.isot} UTC, {sun.alt.deg:.1f}° elevation | {len(channels)} validated fine channels\n'
                 'Phase-only calibration; 4° Sun-centered CLEAN mask; relative visibility units')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=170)
    plt.close(fig)
    np.savez_compressed(args.output.with_suffix('.npz'),axis_rad=axis,
                        xx=np.asarray(planes[0]),yy=np.asarray(planes[1]),
                        frequencies_hz=freq[channels],utc=stamp.isot)
    args.output.with_suffix('.json').write_text(json.dumps(stats,indent=2)+'\n')
    print(json.dumps(stats,indent=2))


if __name__=='__main__':
    main()
