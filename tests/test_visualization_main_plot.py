import os
import numpy as np
from obspy import Stream, Trace, UTCDateTime

from beam.visualization import plot_daily_detections
from beam_driver import TraditionalBeamformer


def _make_short_stream(n_stations=4, n_samples=200, sr=50.0):
    st = Stream()
    for i in range(n_stations):
        data = 0.01 * np.random.randn(n_samples).astype('float32')
        tr = Trace(data)
        tr.stats.station = f'S{i+1:02d}'
        tr.stats.sampling_rate = sr
        tr.stats.starttime = UTCDateTime(0)
        st.append(tr)
    return st


def test_plot_daily_detections_heatmap(tmp_path, monkeypatch):
    bf = TraditionalBeamformer(data_dir='.', use_gpu=False)
    # stations
    coords = {f'S{i+1:02d}': (0.0 + 0.01*i, -120.0 + 0.01*i, 0.0) for i in range(4)}
    bf.set_station_coords(coords)

    st = _make_short_stream()

    # create some detections across azimuths with snr values
    dets = []
    for az, snr in [(0.0, 12.0), (10.0, 8.0), (90.0, 15.0), (270.0, 5.0), (350.0, 11.0)]:
        dets.append({'time': 10.0, 'velocity': 4.0, 'backazimuth': az, 'snr': snr, 'duration': 1.0})

    # monkeypatch beamform for quick, simple beam traces
    def fake_beamform(stream, slowness, azimuth, normalize=True, use_envelope=True, cf_method='envelope'):
        # return a Trace with known data
        n = len(stream[0].data)
        # create a small bump centered in middle
        t = np.zeros(n)
        mid = n // 2
        t[mid-3:mid+3] = 1.0
        tr = Trace(np.array(t, dtype='float32'))
        tr.stats.sampling_rate = stream[0].stats.sampling_rate
        tr.stats.starttime = stream[0].stats.starttime
        return tr

    monkeypatch.setattr(bf, 'beamform', fake_beamform)

    outdir = str(tmp_path)
    outfile = plot_daily_detections('TEST', st, dets, bf, outdir=outdir)

    assert outfile is not None
    assert os.path.exists(outfile)
