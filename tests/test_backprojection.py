"""Back-projection locator: synthetic sources must be recovered."""
import numpy as np
import pytest
from obspy import Stream, Trace, UTCDateTime
from obspy.geodetics import gps2dist_azimuth

from beam.core import backprojection as bpm
from beam_driver import TraditionalBeamformer

LAT0, LON0 = 33.6, -116.5
COSLAT = np.cos(np.radians(LAT0))
FS = 50.0


def _ll(x_km, y_km):
    return LAT0 + y_km / 111.19, LON0 + x_km / (111.19 * COSLAT)


def _network(seed=1):
    """Three 5-station sub-arrays ~60 km apart."""
    rng = np.random.default_rng(seed)
    coords, groups = {}, []
    for a, (dx, dy) in enumerate([(-30, -20), (35, -10), (0, 35)]):
        g = []
        for k in range(5):
            lat, lon = _ll(dx + rng.uniform(-1, 1), dy + rng.uniform(-1, 1))
            coords[f'A{a}S{k}'] = (lat, lon, 500.0)
            g.append(f'A{a}S{k}')
        groups.append(g)
    return coords, groups


def _synthetic(coords, src, t_origin, dur=80.0, vp=6.0, vs=3.5, seed=2):
    rng = np.random.default_rng(seed)
    stations = list(coords)
    tp = _point_tt(coords, stations, src, vp)
    ts = _point_tt(coords, stations, src, vs)
    t = np.arange(0, dur, 1 / FS)
    X = 0.2 * rng.standard_normal((len(stations), t.size))
    for i in range(len(stations)):
        gain = rng.uniform(0.3, 3.0)  # station gains differ; CFs must not care
        for tt, amp in ((tp[i], 1.0), (ts[i], 2.0)):
            a = t_origin + tt + 0.05 * rng.standard_normal()
            X[i] += gain * amp * np.exp(-(t - a) / 0.8) * (t > a) * np.sin(2 * np.pi * 6 * (t - a))
    return stations, X


def _point_tt(coords, stations, src, velocity):
    """Travel times from exactly ``src`` (the centre node of a tiny grid)."""
    g = bpm.make_source_grid(src[0], src[1], 1.0, 1.0, [src[2]])
    centre = int(np.argmin(np.hypot(g.x_km, g.y_km)))
    return bpm.travel_time_table(g, coords, stations, velocity=velocity)[centre]


def _epi_km(res, src):
    return gps2dist_azimuth(res['lat'] if isinstance(res, dict) else res.lat,
                            res['lon'] if isinstance(res, dict) else res.lon,
                            src[0], src[1])[0] / 1000.0


def test_locates_a_synthetic_source():
    coords, groups = _network()
    src = (*_ll(-12, 8), 9.0)
    stations, X = _synthetic(coords, src, t_origin=20.0)
    grid = bpm.grid_around_stations(coords, margin_km=10, step_km=2.0, depths_km=[0, 4, 8, 12, 16])
    ttp = bpm.travel_time_table(grid, coords, stations, velocity=6.0)
    tts = bpm.travel_time_table(grid, coords, stations, velocity=3.5)
    cf, rate = bpm.characteristic_functions(X, FS, sta_s=0.3, lta_s=5.0, smooth_s=0.4, target_rate=10)
    res = bpm.backproject([cf, cf], [ttp, tts], rate, 0.0, grid,
                          weights=bpm.subarray_weights(stations, groups))
    assert _epi_km(res, src) < 2.0 * grid.step_km
    assert res.origin_time == pytest.approx(20.0, abs=0.5)
    assert abs(res.depth_km - 9.0) <= 4.0
    assert 0.0 < res.coherence <= 1.0 and res.contrast > 1.5
    assert not res.on_edge
    assert res.err_x_km >= grid.step_km / np.sqrt(12) - 1e-9
    assert res.brightness.shape == (grid.n_nodes,)


def test_p_only_stack_is_fooled_by_s_and_p_plus_s_is_not():
    """Why S is stacked by default: P-only locks P at one sub-array onto S at
    another and gives a coherent, sharply peaked, wrong location."""
    coords, groups = _network()
    src = (*_ll(-12, 8), 5.0)
    stations, X = _synthetic(coords, src, t_origin=25.0, dur=100.0)
    grid = bpm.grid_around_stations(coords, margin_km=5, step_km=3.0, depths_km=[5.0])
    ttp = bpm.travel_time_table(grid, coords, stations, velocity=6.0)
    tts = bpm.travel_time_table(grid, coords, stations, velocity=3.5)
    cf, rate = bpm.characteristic_functions(X, FS, sta_s=0.3, lta_s=5.0, target_rate=10)
    w = bpm.subarray_weights(stations, groups)
    p_only = bpm.backproject(cf, ttp, rate, 0.0, grid, weights=w)
    both = bpm.backproject([cf, cf], [ttp, tts], rate, 0.0, grid, weights=w)
    assert _epi_km(p_only, src) > 3 * grid.step_km
    assert _epi_km(both, src) < grid.step_km


def test_maximum_on_the_boundary_is_flagged():
    coords, _ = _network()
    stations = list(coords)
    grid = bpm.grid_around_stations(coords, margin_km=5, step_km=3.0, depths_km=[5.0])
    tt = bpm.travel_time_table(grid, coords, stations, velocity=6.0)
    corner = 0  # first node is a grid corner
    rate, n = 10.0, 600
    cf = np.zeros((len(stations), n))
    cf[np.arange(len(stations)), np.rint((10.0 + tt[corner]) * rate).astype(int)] = 1.0
    res = bpm.backproject(cf, tt, rate, 0.0, grid)
    assert res.node == corner and res.on_edge
    assert res.origin_time == pytest.approx(10.0)
    assert res.coherence == pytest.approx(1.0)


def test_subarray_weights_sum_to_one_per_group():
    stations = ['a1', 'a2', 'a3', 'b1', 'lone']
    w = bpm.subarray_weights(stations, [['a1', 'a2', 'a3'], ['b1']])
    assert w[:3].sum() == pytest.approx(1.0) and w[3] == 1.0 and w[4] == 1.0


def test_grid_geometry():
    g = bpm.make_source_grid(LAT0, LON0, 10.0, 2.0, [0, 5])
    assert g.shape == (2, 11, 11) and g.n_nodes == 242
    centre = np.argmin(np.hypot(g.x_km, g.y_km) + g.depth_km)
    assert g.lat[centre] == pytest.approx(LAT0) and g.lon[centre] == pytest.approx(LON0)


def test_constant_velocity_uses_3d_distance():
    coords = {'X': (LAT0, LON0, 0.0)}
    assert _point_tt(coords, ['X'], (LAT0, LON0, 8.0), 4.0)[0] == pytest.approx(2.0)


def test_taup_table_is_physical():
    coords = {'N': _ll(0, 20) + (0.0,), 'F': _ll(0, 60) + (0.0,)}
    g = bpm.make_source_grid(LAT0, LON0, 1.0, 1.0, [10.0])
    centre = int(np.argmin(np.hypot(g.x_km, g.y_km)))
    tt = bpm.travel_time_table(g, coords, ['N', 'F'], model='iasp91', phase='P', n_distances=25)[centre]
    assert tt[1] > tt[0]
    # iasp91 upper crust is 5.8 km/s
    assert tt[0] == pytest.approx(np.hypot(20, 10) / 5.8, rel=0.1)


def test_travel_time_table_needs_one_method():
    g = bpm.make_source_grid(LAT0, LON0, 1.0, 1.0)
    with pytest.raises(ValueError):
        bpm.travel_time_table(g, {'X': (LAT0, LON0, 0.0)}, ['X'])


def test_driver_backprojects_detection_groups(tmp_path):
    coords, groups = _network()
    src = (*_ll(5, -6), 6.0)
    t0 = UTCDateTime(2020, 1, 1, 0, 1, 0)
    stations, X = _synthetic(coords, src, t_origin=30.0, dur=90.0)
    st = Stream()
    for sta, d in zip(stations, X):
        tr = Trace(d)
        tr.stats.station, tr.stats.sampling_rate, tr.stats.starttime = sta, FS, t0
        st.append(tr)
    bf = TraditionalBeamformer(data_dir='.', use_gpu=False)
    bf.set_station_coords(coords)
    bf.subarrays = groups
    # one detection per sub-array near the P arrival, as the grid search would give
    dets = [{'time': t0 + 30.0 + 5.0 + k, 'subarray_id': k} for k in range(3)]
    out = bf.backproject_detections(
        st, dets, {'step_km': 2.0, 'margin_km': 10.0, 'depths_km': '0,4,8,12',
                   's_velocity': 3.5, 'sta_s': 0.3, 'lta_s': 5.0},
        date_str='20200101', plot=True, plot_dir=str(tmp_path))
    assert len(out) == 1
    loc = out[0]
    assert _epi_km(loc, src) < 4.0
    assert loc['origin_time'] == pytest.approx(float(t0) + 30.0, abs=0.5)
    assert loc['subarrays'] == [0, 1, 2] and loc['phases'] == ['P', 'S']
    assert (tmp_path / 'locations_bp_20200101.json').exists()
    assert list((tmp_path / 'backprojection').glob('bp_*.png'))
    # S is stacked by default (velocity/sqrt(3)); 0 means P only
    default = bf.backproject_detections(st, dets, {'step_km': 4.0, 'margin_km': 5.0, 'depths_km': '6'})
    assert default[0]['phases'] == ['P', 'S']
    p_only = bf.backproject_detections(st, dets, {'step_km': 4.0, 'margin_km': 5.0,
                                                  'depths_km': '6', 's_velocity': 0})
    assert p_only[0]['phases'] == ['P']
    # a group seen by only one sub-array is skipped under the default min_subarrays=2
    assert bf.backproject_detections(st, dets[:1], {'step_km': 4.0, 'margin_km': 5.0}) == []
