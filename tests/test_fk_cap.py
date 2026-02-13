import types
from beam_driver import TraditionalBeamformer
import numpy as np
from obspy import Stream, Trace, UTCDateTime


def _make_simple_stream(n_stations=3, n_samples=100, sr=20.0):
    stream = Stream()
    for i in range(n_stations):
        data = 0.01 * np.random.randn(n_samples).astype('float32')
        tr = Trace(data)
        tr.stats.station = f'ST{i+1}'
        tr.stats.sampling_rate = sr
        tr.stats.starttime = UTCDateTime(0)
        stream.append(tr)
    return stream


def test_fk_cap_top_k(monkeypatch):
    bf = TraditionalBeamformer(data_dir='.', use_gpu=False)
    # simple station set
    coords = {
        'ST1': (0.0, 0.0, 0.0),
        'ST2': (0.01, 0.0, 0.0),
        'ST3': (0.0, 0.01, 0.0)
    }
    bf.set_station_coords(coords)
    bf.subarrays = [['ST1', 'ST2', 'ST3']]

    # mock loader to return a stream
    bf.loader.load_day = lambda date: _make_simple_stream()

    # create 10 fake detections with increasing SNR; ensure times spaced > cluster window
    raw = []
    for i in range(10):
        raw.append({'time': float(i * 20), 'snr': float(i), 'velocity': 3.0, 'backazimuth': 30.0, 'slowness': 1.0/3.0})

    # patch grid_search_detection to return our raw detections (no heavy work)
    monkeypatch.setattr(bf, 'grid_search_detection', lambda *args, **kwargs: raw)

    # instrument fk_analysis to count calls
    calls = []

    def fake_fk(stream, t):
        calls.append(t)
        return {'velocity': 3.5, 'backazimuth': 31.0, 'power': 10.0}

    monkeypatch.setattr(bf, 'fk_analysis', fake_fk)

    # run with fk_max_per_subarray=3
    dets = bf.process_single_day('20200101', max_beams=8, decimate=1, plot=False, plot_dir=None, fk_max_per_subarray=3, fk_min_snr=0.0)
    # we expect only up to 3 FK calls
    assert len(calls) <= 3

    # check that at most 3 returned detections include fk_* keys
    fk_cnt = sum(1 for d in dets if 'fk_velocity' in d)
    assert fk_cnt <= 3


def test_fk_cap_threshold(monkeypatch):
    bf = TraditionalBeamformer(data_dir='.', use_gpu=False)
    coords = {'ST1': (0,0,0), 'ST2':(0.01,0,0), 'ST3':(0,0.01,0)}
    bf.set_station_coords(coords)
    bf.subarrays = [['ST1','ST2','ST3']]
    bf.loader.load_day = lambda date: _make_simple_stream()

    # create detections with varying SNR and times spaced to avoid clustering
    snrs = [0.2, 0.8, 1.2, 2.0, 5.0, 0.1, 0.9]
    raw = [{'time': float(i * 20), 'snr': float(s), 'velocity':3.0, 'backazimuth':10.0, 'slowness':1/3.0} for i, s in enumerate(snrs)]
    monkeypatch.setattr(bf, 'grid_search_detection', lambda *args, **kwargs: raw)

    recorded = []
    def fake_fk(stream, t):
        recorded.append(t)
        return {'velocity': 3.1, 'backazimuth': 11.0, 'power': 5.0}

    monkeypatch.setattr(bf, 'fk_analysis', fake_fk)

    # set min_snr to 1.0; only detections with snr >= 1.0 should be considered (three of them)
    dets = bf.process_single_day('20200101', max_beams=50, decimate=1, plot=False, plot_dir=None, fk_max_per_subarray=10, fk_min_snr=1.0)
    assert len(recorded) == 3
    # Check that returned detections have fk fields only on those that passed threshold
    for d in dets:
        if d.get('snr', 0.0) < 1.0:
            assert 'fk_velocity' not in d
