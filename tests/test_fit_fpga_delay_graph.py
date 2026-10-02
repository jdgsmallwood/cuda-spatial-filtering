import pytest

from scripts.fit_fpga_delay_graph import fit_delays


def test_six_pair_fit_recovers_offsets_and_closure():
    true = {0: 8738838, 1: 0, 2: -12252948, 3: -21117431}
    pairs = [
        {"first": i, "second": j, "delay": true[j] - true[i]}
        for i in range(4) for j in range(i + 1, 4)
    ]
    fitted, residuals = fit_delays(pairs)
    assert fitted == {str(k): v for k, v in true.items()}
    assert max(abs(r) for r in residuals) < 1e-6


def test_inconsistent_pair_is_rejected():
    pairs = [
        {"first": i, "second": j, "delay": j - i}
        for i in range(4) for j in range(i + 1, 4)
    ]
    pairs[-1]["delay"] += 64
    with pytest.raises(ValueError, match="fail closure"):
        fit_delays(pairs)


def test_one_small_false_peak_does_not_shift_other_fpga_offsets():
    true={0:8739060,1:0,2:-12252732,3:-21117141}
    pairs=[{'first':i,'second':j,'delay':true[j]-true[i]}
           for i in range(4) for j in range(i+1,4)]
    pairs[3]['delay']-=2
    fitted,residuals=fit_delays(pairs)
    assert fitted=={str(k):v for k,v in true.items()}
    assert list(residuals)==[0,0,0,2,0,0]


def test_two_disagreeing_peaks_require_recapture():
    pairs=[{'first':i,'second':j,'delay':j-i}
           for i in range(4) for j in range(i+1,4)]
    pairs[0]['delay']+=2
    pairs[-1]['delay']-=2
    with pytest.raises(ValueError,match='fail closure'):
        fit_delays(pairs)
