import os
import json
import numpy as np
from obspy import UTCDateTime, Stream, Trace

from beam_driver import TraditionalBeamformer


def _make_synthetic_stream(station_coords, event_latlon, event_time=10.0,
                           sr=50.0, duration=30.0, freq=3.0, slowness=0.3,
                           backazimuth=45.0):
    """Create a synthetic Stream for the provided station coordinates.

    station_coords: dict station_code -> (lat, lon, elev)
    event_latlon: (lat, lon) for the source
    The function computes delays for each station relative to the array
    center using the given slowness and backazimuth, and constructs
    traces with a small sinusoidal burst at the delayed time.
    """
    # sampling
    n_samples = int(duration * sr)
    t = np.arange(n_samples) / sr

    # center of provided coordinates
    lats = np.array([v[0] for v in station_coords.values()])
    lons = np.array([v[1] for v in station_coords.values()])
    center_lat = float(np.mean(lats))
    center_lon = float(np.mean(lons))

    # convert latlon to local xy for delays (km)
    def latlon_to_xy(lat, lon, o_lat, o_lon):
        mean_lat = np.radians(o_lat)
        dx = (lon - o_lon) * 111.32 * np.cos(mean_lat)
        dy = (lat - o_lat) * 110.54
        return dx, dy

    # slowness vector
    az = np.radians(backazimuth)
    sx = slowness * np.sin(az)
    sy = slowness * np.cos(az)

    base_wave = np.sin(2 * np.pi * freq * t) * np.exp(-((t - event_time) ** 2) / (2.0 * 0.5 ** 2))

    stream = Stream()
    for i, (sta, (lat, lon, elev)) in enumerate(station_coords.items()):
        x, y = latlon_to_xy(lat, lon, center_lat, center_lon)
        delay = sx * x + sy * y
        # apply delay in samples (allow fractional) by shifting the base waveform
        samples_shift = int(round(delay * sr))
        data = np.zeros_like(base_wave)
        if samples_shift >= 0:
            if samples_shift < len(base_wave):
                data[samples_shift:] = base_wave[:len(base_wave)-samples_shift]
        else:
            s2 = -samples_shift
            if s2 < len(base_wave):
                data[:len(base_wave)-s2] = base_wave[s2:]

        # Add small noise
        data += 0.01 * np.random.randn(len(data))

        tr = Trace(data.astype(np.float32))
        tr.stats.station = sta
        tr.stats.sampling_rate = sr
        tr.stats.starttime = UTCDateTime(0)
        tr.stats.network = 'XX'
        tr.stats.channel = 'BHZ'
        tr.stats.location = ''
        stream.append(tr)

    return stream


def test_single_day_gpu_triangulation(tmp_path):
    # Create beamformer configured to use GPU if available
    bf = TraditionalBeamformer(data_dir='.', use_gpu=True)

    # Build a small station set split into two subarrays
    station_coords = {
        'A1': (0.0, 0.0, 0.0),
        'A2': (0.0, 0.01, 0.0),
        'B1': (0.1, -0.05, 0.0),
        'B2': (0.1, 0.05, 0.0)
    }

    # set station coordinates explicitly and define subarrays
    bf.set_station_coords(station_coords)
    bf.subarrays = [['A1', 'A2'], ['B1', 'B2']]

    # also build subarray geometries (center-relative x,y) so process_single_day
    # can compute delays correctly. We mimic the same computation used in driver.
    bf.subarray_geometries = {}
    for i, group in enumerate(bf.subarrays):
        lat_vals = np.array([station_coords[s][0] for s in group])
        lon_vals = np.array([station_coords[s][1] for s in group])
        center_lat = float(np.mean(lat_vals))
        center_lon = float(np.mean(lon_vals))
        geom = {}
        for sta, lat, lon in zip(group, lat_vals, lon_vals):
            x = (lon - center_lon) * 111.32 * np.cos(np.radians(center_lat))
            y = (lat - center_lat) * 110.54
            geom[sta] = (x, y, 0.0)
        bf.subarray_geometries[i] = geom

    # Create synthetic stream consistent with a source roughly northeast
    event = (0.05, 0.02)
    stream = _make_synthetic_stream(station_coords, event, event_time=8.0, sr=50.0, duration=25.0, freq=5.0, slowness=0.3, backazimuth=45.0)

    # Monkey patch loader.load_day to return our synthetic stream
    def fake_load_day(date_str):
        return stream

    bf.loader.load_day = fake_load_day

    outdir = tmp_path / 'plots'
    # Run single day processing (use small grid to keep runtime modest)
    dets = bf.process_single_day('20200101', velocity_range=(3.0, 5.0), azimuth_range=(0, 360), velocity_step=1.0, azimuth_step=90, sta_len=0.2, lta_len=1.0, threshold=1.5, use_envelope=True, cf_method='envelope', max_beams=8, decimate=1, plot=True, plot_dir=str(outdir))

    # ensure function returned without error and wrote triangulation/plot files
    assert isinstance(dets, list)

    # per-day detections json should have been created in the plot_dir
    per_day_json = outdir / 'detections_20200101.json'
    assert per_day_json.exists()
    # triangulation results JSON may or may not be created depending on detections
    jsonfile = outdir / 'locations_20200101.json'
    # Accept either scenario (file exists or not) but ensure no exception and
    # that if it exists it contains valid JSON list
    if jsonfile.exists():
        obj = json.loads(jsonfile.read_text())
        assert isinstance(obj, list)


