"""Counter-aware sampling and residual delay sign/recovery checks."""
import importlib.util
from pathlib import Path
import sys
import numpy as np

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
spec=importlib.util.spec_from_file_location('delay_samples',SCRIPTS/'analyze_delay_samples.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_packet_counters_clip_partial_packets_keep_gaps_and_ignore_duplicates():
    p=np.zeros(4,dtype=m.PACKET);p['channel']=300;p['scales']=2
    p['sample']=[1000,1064,1192,1064];p['iq'][...,0]=3;p['iq'][...,1]=-1
    x,mask,duplicates=m.assemble(p,300,17,1000,256)
    expected=np.zeros(256,bool);expected[:111]=True;expected[175:239]=True
    np.testing.assert_array_equal(mask,expected)
    np.testing.assert_array_equal(x[mask],np.full((expected.sum(),20),6-2j))
    assert not np.any(x[~mask]);assert duplicates==1


def test_lag_search_recovers_nonzero_residual_in_both_time_halves():
    rng=np.random.default_rng(117)
    windows,n,delay=8,1024,7
    sky=(rng.normal(size=(windows,n+64))+1j*rng.normal(size=(windows,n+64))).astype(np.complex64)
    x=sky[:,32:32+n,None]*np.exp(1j*np.array([.7,-1.2]))
    y=sky[:,32-delay:32-delay+n,None]*np.exp(1j*np.array([2.,-.5,.3]))
    x+=.3*(rng.normal(size=x.shape)+1j*rng.normal(size=x.shape))
    y+=.3*(rng.normal(size=y.shape)+1j*rng.normal(size=y.shape))
    groups=[np.arange(8),np.arange(4),np.arange(4,8)]
    result=m.lag_correlations(x,y,24,groups)
    lags=np.arange(-24,25)
    np.testing.assert_array_equal(lags[np.argmax(abs(result),axis=1)],np.full((3,2,3),-delay))
    # Seed_j-seed_i=100: true counter difference is 100 - (-7).
    assert 100-int(lags[np.argmax(abs(result[0,:,0,0]))])==107


def test_ensemble_recovers_six_edges_and_exact_graph_closure():
    rng=np.random.default_rng(12);n=1024;truth=np.array([4,0,-7,11]);seed=np.array([3,0,-5,8])
    residual=truth-seed;sky=rng.normal(size=(8,n+64))+1j*rng.normal(size=(8,n+64))
    streams=[sky[:,32-r:32-r+n,None]*np.exp(1j*b) for b,r in enumerate(residual)]
    edges=[]
    for i,j in m.PAIRS:
        c=m.lag_correlations(streams[i],streams[j],24,[np.arange(8)])
        lag=np.arange(-24,25)[np.argmax(abs(c[0,:,0,0]))]
        edges.append({'first':i,'second':j,'delay':int(seed[j]-seed[i]-lag)})
    fitted,error=m.fit_graph(edges,reference=1,max_residual=0)
    assert fitted=={str(i):int(d) for i,d in enumerate(truth)}
    assert not np.any(error)


def test_competing_peak_and_search_boundary_are_reported():
    lags=np.arange(-12,13);score=np.ones(25);score[0]=20;score[12]=19
    r=m.summarize_score(lags,score)
    assert r['at_search_boundary']
    assert r['peak_to_runner_ratio']<1.1


def test_end_to_end_packet_files_recover_wrong_seed_without_applying_it(tmp_path):
    import json,subprocess
    rng=np.random.default_rng(91)
    n=8192;start=100000;channels=[300,301]
    seed=[-123,0,134,-33];truth=[-120,0,137,-36]
    sky=rng.integers(-20,21,size=(2,n+1024,2,2),dtype=np.int8)
    manifest={'format':'lambda-delay-packets-v1','packet_bytes':m.PACKET.itemsize,
        'start_sample':start,'samples':n,'channels':channels,'complete':True,
        'sample_seconds':1.08e-6,'fpgas':{}}
    mapping={'fpgas':{}}
    for b in range(4):
        rows=[]
        for ci,ch in enumerate(channels):
            counters=np.arange((start+seed[b])//64*64,start+seed[b]+n,64)
            rows_ch=np.zeros(len(counters),dtype=m.PACKET)
            rows_ch['sample']=counters;rows_ch['channel']=ch;rows_ch['scales']=1
            for k,counter in enumerate(counters):
                sky_index=counter-truth[b]-start+512+np.arange(64)
                rows_ch['iq'][k,:,:2]=sky[ci,sky_index]
            rows.append(rows_ch)
        name=f'fpga{b}.packets';np.concatenate(rows).tofile(tmp_path/name)
        manifest['fpgas'][str(b)]={'file':name,'seed_delay':seed[b],
            'capacity_rejections':0,'malformed':0}
        mapping['fpgas'][str(b)]={'streams':[{'stream':p,'antenna_id':b*2+p+1,'polarization':p} for p in (0,1)]}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    (tmp_path/'map.json').write_text(json.dumps(mapping))
    output=tmp_path/'result.json'
    subprocess.run([sys.executable,str(SCRIPTS/'analyze_delay_samples.py'),str(tmp_path),
        '--stream-map',str(tmp_path/'map.json'),'--output',str(output),
        '--block-samples','1024','--max-lag','24'],check=True,capture_output=True)
    result=json.loads(output.read_text())
    assert result['validated']
    for partition in result['partitions']:
        assert partition['closure']['offsets']=={str(b):truth[b] for b in range(4)}
    assert not (tmp_path/'alveo_delays.json').exists()


def test_optional_gpu_matches_cpu_complex_correlations():
    import os,pytest
    library=os.environ.get('SPATIAL_DELAY_GPU_LIBRARY')
    if not library:pytest.skip('set SPATIAL_DELAY_GPU_LIBRARY to test the CUDA backend')
    rng=np.random.default_rng(73)
    x=(rng.normal(size=(8,512,3))+1j*rng.normal(size=(8,512,3))).astype(np.complex64)
    y=(rng.normal(size=(8,512,2))+1j*rng.normal(size=(8,512,2))).astype(np.complex64)
    x[:,17:39]=0;y[:,93:101]=0
    groups=[np.array([0,1,3,4,6,7]),np.array([0,1,3]),np.array([4,6,7])]
    expected=m.lag_correlations(x,y,64,groups)
    actual=m.GpuCorrelator(library)(x,y,64,groups)
    np.testing.assert_allclose(actual,expected,rtol=5e-4,atol=2e-7)
