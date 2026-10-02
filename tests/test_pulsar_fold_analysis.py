"""Independent dispersed pulse test for archive diagnostics (no GPU required)."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

fits = pytest.importorskip('astropy.io.fits')
pytest.importorskip('matplotlib')
spec = importlib.util.spec_from_file_location(
    'fold_analysis', Path(__file__).parents[1] / 'scripts/analyze_pulsar_fold.py')
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def simulated_fold(path, dedispersed=True):
    nsub, nchan, nbin = 4, 64, 256
    period, archive_dm, true_dm = 0.03385, 56.71, 62.5
    frequencies = np.linspace(225, 255, nchan)
    # Positive residual cold-plasma delays make low frequencies arrive later.
    delays = 4148.808 * (true_dm - archive_dm) * (
        frequencies ** -2 - frequencies.max() ** -2)
    centers = (0.31 + delays / period) % 1
    distance = (np.arange(nbin)[None] / nbin - centers[:, None] + 0.5) % 1 - 0.5
    signal = 5 * np.exp(-0.5 * (distance / (2 / nbin)) ** 2)
    rng = np.random.default_rng(812)
    data = rng.normal(size=(nsub, nchan, nbin)) + signal
    weights = np.ones((nsub, nchan), dtype=np.float32)
    weights[:, :4] = 0
    # A strong spurious profile in excluded channels must not set the best DM.
    data[:, :4, 7] += 1000
    columns = [
        fits.Column(name='DATA', format=f'{nchan * nbin}I', dim=f'({nbin},{nchan},1)',
                    array=np.round(data * 10).astype(np.int16)),
        fits.Column(name='DAT_SCL', format=f'{nchan}E', array=np.full((nsub, nchan), 0.1)),
        fits.Column(name='DAT_OFFS', format=f'{nchan}E', array=np.zeros((nsub, nchan))),
        fits.Column(name='DAT_WTS', format=f'{nchan}E', array=weights),
        fits.Column(name='DAT_FREQ', format=f'{nchan}D', array=np.tile(frequencies, (nsub, 1))),
        fits.Column(name='TSUBINT', format='D', array=np.full(nsub, 10.0)),
        fits.Column(name='OFFS_SUB', format='D', array=np.arange(nsub) * 10.0 + 5),
        # Exercise PSRCHIVE's zero-PERIOD / predictor metadata case.
        fits.Column(name='PERIOD', format='D', array=np.zeros(nsub)),
    ]
    sub = fits.BinTableHDU.from_columns(columns, name='SUBINT')
    sub.header.update(NCHAN=nchan, NBIN=nbin, NPOL=1, DM=archive_dm)
    history = fits.BinTableHDU.from_columns([
        fits.Column(name='DEDISP', format='I', array=[int(dedispersed)]),
        fits.Column(name='TBIN', format='D', array=[period / nbin]),
    ], name='HISTORY')
    primary = fits.PrimaryHDU()
    primary.header.update(OBS_MODE='PSR', SRC_NAME='synthetic')
    fits.HDUList([primary, history, sub]).writeto(path)
    return true_dm, period


def test_recovers_dispersion_with_masked_interference_and_zero_period(tmp_path):
    path = tmp_path / 'pulse.fits'
    dm, period = simulated_fold(path)
    result = analysis.analyze(path, tmp_path / 'diagnostic')
    assert result['period_seconds'] == pytest.approx(period)
    assert result['retained_channels'] == 60
    assert abs(result['best_trial_dm'] - dm) < 0.51
    assert result['best_trial_dm_score'] > 20
    assert (tmp_path / 'diagnostic.png').is_file()


def test_refuses_archive_without_dedispersion(tmp_path):
    path = tmp_path / 'pulse.fits'
    simulated_fold(path, dedispersed=False)
    with pytest.raises(ValueError, match='dedispersed first'):
        analysis.analyze(path, tmp_path / 'diagnostic')
