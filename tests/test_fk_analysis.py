"""FK analysis, beamform() and the hand-off of FK estimates to location."""
import numpy as np
import pytest
from obspy import Stream, Trace, UTCDateTime

from beam_driver import TraditionalBeamformer, _adopt_fk_result
from beam.core.coherence import align_traces, semblance, semblance_at

SR = 100.0
LAT0, LON0 = 33.0, -116.0


def _array(n=7, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-1.0, 1.0, (n, 2))  # km
    coslat = np.cos(np.radians(LAT0))
    return {f'S{i}': (LAT0 + y / 111.19, LON0 + x / (111.19 * coslat), 0.0)
            for i, (x, y) in enumerate(xy)}


def _plane_wave(bf, baz, slowness, noise=0.01, seed=1, freq=8.0, dur=20.0):
    """Stream of a plane wave crossing ``bf.array_geometry``; pulse at t=10 s at the centre."""
    rng = np.random.default_rng(seed)
    sx, sy = slowness * np.sin(np.radians(baz)), slowness * np.cos(np.radians(baz))
    t = np.arange(0, dur, 1 / SR)
    st = Stream()
    for sta, (x, y, _) in bf.array_geometry.items():
        tau = -(sx * x + sy * y)  # arrival relative to the array centre
        d = (np.exp(-((t - 10 - tau) / 0.08) ** 2) * np.sin(2 * np.pi * freq * (t - 10 - tau))
             + noise * rng.standard_normal(t.size))
        tr = Trace(d)
        tr.stats.station, tr.stats.sampling_rate, tr.stats.starttime = sta, SR, UTCDateTime(0)
        st.append(tr)
    return st


@pytest.fixture
def bf():
    b = TraditionalBeamformer(data_dir='.', use_gpu=False)
    b.set_station_coords(_array())
    return b


def test_beamform_returns_a_trace(bf):
    st = _plane_wave(bf, 60.0, 0.17, noise=0.0)
    beam = bf.beamform(st, 0.17, 60.0, use_envelope=False)
    assert beam is not None, "beamform() must return the beam, not None"
    # aligned on the true wave, the normalised beam keeps most of the unit
    # pulse amplitude (beamform() shifts by whole samples, so not all of it)
    assert beam.data.max() > 0.8
    # and a wrong direction does not
    off = bf.beamform(st, 0.17, 240.0, use_envelope=False)
    assert off.data.max() < 0.5 * beam.data.max()


@pytest.mark.parametrize('baz,slow,smax', [(60.0, 0.17, 0.3),
                                           (203.7, 0.125, 0.3),
                                           (311.0, 0.5, 0.7)])   # slow surface wave
def test_fk_recovers_plane_wave(bf, baz, slow, smax):
    freq = 8.0 if slow < 0.3 else 3.0
    st = _plane_wave(bf, baz, slow, freq=freq)
    res = bf.fk_analysis(st, UTCDateTime(10), window_length=3.0, slowness_range=smax)
    assert res is not None
    err = (res['backazimuth'] - baz + 180) % 360 - 180
    assert abs(err) < 1.0
    assert res['slowness'] == pytest.approx(slow, abs=0.005)
    assert res['semblance'] > 0.9
    assert np.isfinite(res['power']) and res['n_stations'] == 7


def test_fk_peak_is_refined_below_the_grid(bf):
    # 0.02 s/km cells are ~6.7 deg of azimuth at 0.17 s/km; refinement must beat half that
    st = _plane_wave(bf, 47.3, 0.17)
    res = bf.fk_analysis(st, UTCDateTime(10), slowness_range=0.3, slowness_step=0.02)
    assert abs(res['backazimuth'] - 47.3) < 1.5


def test_fk_needs_three_stations(bf):
    st = _plane_wave(bf, 60.0, 0.17)[:2]
    assert bf.fk_analysis(st, UTCDateTime(10)) is None


def test_adopt_fk_result_replaces_grid_values_and_keeps_them():
    det = {'backazimuth': 30.0, 'velocity': 6.0, 'slowness': 1 / 6.0}
    _adopt_fk_result(det, {'backazimuth': 31.2, 'velocity': 5.5, 'power': 1.0, 'semblance': 0.8})
    assert det['backazimuth'] == 31.2 and det['velocity'] == 5.5
    assert det['slowness'] == pytest.approx(1 / 5.5)
    assert det['grid_backazimuth'] == 30.0 and det['grid_velocity'] == 6.0
    assert det['fk_semblance'] == 0.8 and det['estimate'] == 'fk'


def test_locators_receive_the_fk_backazimuth(monkeypatch):
    bf = TraditionalBeamformer(data_dir='.', use_gpu=False)
    bf.set_station_coords({'ST1': (0.0, 0.0, 0.0), 'ST2': (0.01, 0.0, 0.0), 'ST3': (0.0, 0.01, 0.0)})
    bf.subarrays = [['ST1', 'ST2', 'ST3']]
    st = Stream()
    for s in ('ST1', 'ST2', 'ST3'):
        tr = Trace(np.zeros(100))
        tr.stats.station, tr.stats.sampling_rate = s, 20.0
        st.append(tr)
    bf.loader.load_day = lambda date: st
    raw = [{'time': 100.0, 'snr': 9.0, 'velocity': 3.0, 'backazimuth': 30.0, 'slowness': 1 / 3.0}]
    monkeypatch.setattr(bf, 'grid_search_detection', lambda *a, **k: raw)
    seen = {}

    def fake_fk(stream, t, **kwargs):
        seen.update(kwargs)
        return {'velocity': 3.5, 'backazimuth': 31.0, 'power': 10.0}

    monkeypatch.setattr(bf, 'fk_analysis', fake_fk)
    dets = bf.process_single_day('20200101', velocity_range=(1.5, 5.0), plot=False)
    assert dets[0]['backazimuth'] == 31.0 and dets[0]['grid_backazimuth'] == 30.0
    # slowness limit follows the slowest velocity searched, so slow phases are not clipped
    assert seen['slowness_range'] == pytest.approx(1.05 / 1.5)


# ------------------------------------------------------------- coherence ---
def test_semblance_bounds():
    rng = np.random.default_rng(3)
    s = rng.standard_normal(500)
    assert semblance(np.tile(s, (6, 1))) == pytest.approx(1.0)
    noise = rng.standard_normal((50, 5000))
    assert semblance(noise) == pytest.approx(1 / 50, rel=0.3)
    # a fractional shift of band-limited traces keeps semblance <= 1
    x = np.tile(np.sin(2 * np.pi * 5 * np.arange(400) / 100.0) * np.hanning(400), (4, 1))
    val = semblance_at(x, np.array([0.0, 0.013, -0.027, 0.004]), 100.0)
    assert 0.0 < val <= 1.0 + 1e-12


def test_align_traces_undoes_a_delay():
    fs = 100.0
    t = np.arange(512) / fs
    pulse = lambda t0: np.exp(-((t - t0) / 0.05) ** 2)
    x = np.vstack([pulse(2.0), pulse(2.1)])
    a = align_traces(x, np.array([0.1, 0.0]), fs)[:, :512]
    assert np.argmax(a[0]) == np.argmax(a[1])
