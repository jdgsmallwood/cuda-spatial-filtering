"""Known-signal checks for native sky fitting and voltage calibration export."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'


def load_script(name):
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name+'.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_solver_predicts_baselines_not_used_in_fit():
    pytest.importorskip('lambda_commissioning')
    fit = load_script('diagnose_sky_calibration').fit_gains
    rng = np.random.default_rng(712)
    sources = rng.normal(size=(36,9)) + 1j*rng.normal(size=(36,9))
    sky = sources @ sources.conj().T
    gains = rng.uniform(.2,2,36) * np.exp(1j*rng.uniform(-np.pi,np.pi,36))
    measured = gains[:,None]*sky*gains[None,:].conj()
    train = ~np.eye(36,dtype=bool)
    train[0::4,1::4] = False
    train &= train.T
    fitted = fit(measured,sky,train)
    predicted = fitted[:,None]*sky*fitted[None,:].conj()
    withheld = (~train)&(~np.eye(36,dtype=bool))
    np.testing.assert_allclose(predicted[withheld],measured[withheld],rtol=2e-6,atol=2e-6)


def test_casa_export_cancels_native_voltage_phases():
    pytest.importorskip('casacore')
    convert = load_script('export_casa_fine_beam_gains').casa_to_voltage_gains
    gains = np.array([.8*np.exp(.7j),1.3*np.exp(-1.8j),.5*np.exp(2.3j)])
    native = gains[:,None]*gains[None,:].conj()
    ms_data = native.conj()
    casa = gains.conj()
    np.testing.assert_allclose(casa[:,None]*casa[None,:].conj(),ms_data)
    exported = convert(casa)
    beam_weights = exported.conj()/abs(exported)**2
    np.testing.assert_allclose(beam_weights*gains,np.ones(3),atol=1e-12)


def test_observation_map_matches_physical_gain_polarizations(tmp_path):
    entries=[]
    for slot,ant in enumerate((5,29,8,12,11,13)):
        for hand in (0,1):
            entries.append({'stream':2*slot+hand,'antenna_id':ant,
                            'polarization':hand^(ant in (8,12))})
    source=tmp_path/'native.json';dest=tmp_path/'pulsar.json'
    source.write_text(json.dumps({'fpgas':{'0':{'streams':entries}}}))
    subprocess.run([sys.executable,str(SCRIPTS/'prepare_pulsar_stream_map.py'),
                    str(source),str(dest),'--antenna-id-swaps','5:29,11:13',
                    '--physical-xx-first'],check=True)
    result=json.loads(dest.read_text())
    output=result['fpgas']['0']['streams']
    swap={5:29,29:5,11:13,13:11}
    for original,new in zip(entries,output):
        assert new['antenna_id']==swap.get(original['antenna_id'],original['antenna_id'])
        assert new['polarization']==1-original['polarization']
    assert result['pulsar_observation']['physical_polarization_order']==['XX','YY']


def test_smooth_phase_recovers_delay_across_gaps_and_outliers():
    fit=load_script('smooth_native_sky').fit_phase_curve
    rng=np.random.default_rng(98)
    freq=np.linspace(280e6,312e6,1120)
    truth=np.exp(1j*(1.3+2*np.pi*(freq-freq.mean())*237e-9))
    observed=truth*np.exp(1j*rng.normal(0,.035,len(freq)))
    observed[::13]*=np.exp(1j*rng.uniform(-np.pi,np.pi,len(observed[::13])))
    selected=(np.arange(len(freq))//28)%5!=2
    selected&=~((freq>290e6)&(freq<296e6))
    predicted,_=fit(freq,observed,selected)
    assert np.max(abs(np.angle(predicted*truth.conj())))<.02


def test_fixed_phase_amplitudes_preserve_known_visibility():
    fit=load_script('smooth_native_sky').fit_amplitudes
    rng=np.random.default_rng(104)
    sources=rng.normal(size=(12,7))+1j*rng.normal(size=(12,7))
    model=sources@sources.conj().T
    phase=np.exp(1j*rng.uniform(-np.pi,np.pi,12))
    amplitude=rng.uniform(.5,2,12)
    gain=phase*amplitude
    data=model*gain[:,None]*gain.conj()[None,:]
    fitted=fit(data,model,phase,~np.eye(12,dtype=bool))
    np.testing.assert_allclose(fitted,amplitude,rtol=2e-5)


def test_smooth_phase_predicts_unseen_coarse_channels_with_repeated_ramp():
    fit=load_script('smooth_native_sky').fit_phase_curve
    coarse=np.repeat(np.arange(40),28)
    offset=np.tile(np.linspace(-.5,.5,28),40)
    freq=300e6+(coarse+offset)*781250
    phase=2*np.pi*(freq-freq.mean())*860e-9-2.3*offset+.4
    truth=np.exp(1j*phase)
    selected=(coarse%5!=2)&(abs(offset)<.43)
    predicted,_=fit(freq,truth,selected,coarse_offset=offset)
    np.testing.assert_allclose(predicted[~selected],truth[~selected],atol=2e-5)


def test_smooth_export_keeps_physical_antenna_mapping_and_joint_mask(tmp_path):
    export=load_script('smooth_native_sky').export_phase_weights
    freq=np.array([300e6,301e6,301e6,302e6])
    order=np.array([29,2,5])
    phases=np.exp(1j*np.arange(24).reshape(4,2,3)/7)
    weights={str(384+i):{p:{str(a):{'real':1.,'imag':0.} for a in order}
                         for p in ('XX','YY')} for i in range(4)}
    template=tmp_path/'original.json'
    template.write_text(json.dumps({'weights':weights,'provenance':{
        'frequency_first_hz':freq[0],'frequency_last_hz':freq[-1]}}))
    good=np.array([[True,True],[True,False],[True,True],[True,True]])
    export(template,phases,order,freq,good,tmp_path/'smooth',{
        'coarse_ramp':True,'degree':2,'smooth_joint_bandwidth_mhz':2.})
    result=json.loads((tmp_path/'smooth_gains.json').read_text())
    g=result['weights']['385']['YY']['29']
    np.testing.assert_allclose(g['real']+1j*g['imag'],phases[1,1,0])
    mask=json.loads((tmp_path/'smooth_channel_mask.json').read_text())
    assert mask['frequency_hz']==[300e6,301e6,302e6]
    assert mask['good']==[True,False,True]


def test_repeated_phase_fit_handles_many_turns_within_one_coarse_channel():
    fit=load_script('smooth_native_sky').fit_phase_curve
    coarse=np.repeat(np.arange(40),28)
    offset=np.tile(np.linspace(-.5,.5,28),40)
    freq=225e6+(coarse+offset)*781250
    phase=.8+2*np.pi*(freq-freq.mean())*350e-9+45*offset
    truth=np.exp(1j*phase)
    selected=(coarse%5!=2)&(abs(offset)<.43)
    predicted,_=fit(freq,truth,selected,coarse_offset=offset)
    np.testing.assert_allclose(predicted[~selected],truth[~selected],atol=2e-5)
