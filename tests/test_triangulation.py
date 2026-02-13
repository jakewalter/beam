import math

from beam.core.triangulation import latlon_to_xy, xy_to_latlon, triangulate_two_arrays


def _angle_deg_from_vector(dx, dy):
    # returns azimuth in degrees where 0 = North, 90 = East (clockwise)
    # Our xy coords: x = East, y = North
    theta = math.atan2(dx, dy)  # atan2(x, y) gives azimuth from North
    az = math.degrees(theta) % 360.0
    return az


def test_roundtrip_latlon_xy():
    origin = (10.0, -120.0)
    lat, lon = 10.123, -119.987
    x, y = latlon_to_xy(lat, lon, origin[0], origin[1])
    lat2, lon2 = xy_to_latlon(x, y, origin[0], origin[1])
    assert abs(lat - lat2) < 1e-6
    assert abs(lon - lon2) < 1e-6


def test_triangulation_pair():
    # two arrays separated east-west by 0.2 degrees (roughly 22 km)
    center_a = (0.0, 0.0)
    center_b = (0.0, 0.2)

    # choose a source to the northeast of both arrays
    source = (0.05, 0.1)

    # compute backazimuths (direction from source -> array) for each
    ax, ay = latlon_to_xy(center_a[0], center_a[1], source[0], source[1])
    bx, by = latlon_to_xy(center_b[0], center_b[1], source[0], source[1])

    # For array->source vector (sx, sy) = source - center
    s_ax = (latlon_to_xy(source[0], source[1], center_a[0], center_a[1]))
    s_bx = (latlon_to_xy(source[0], source[1], center_b[0], center_b[1]))

    # compute array->source bearing then backazimuth = (bearing + 180) % 360
    vax, vay = s_ax
    vbx, vby = s_bx
    bearing_a = _angle_deg_from_vector(vax, vay)
    bearing_b = _angle_deg_from_vector(vbx, vby)
    backaz_a = (bearing_a + 180.0) % 360.0
    backaz_b = (bearing_b + 180.0) % 360.0

    # prepare detection entries with nearly identical times
    t = 1000.0
    det_a = {'time': t, 'backazimuth': backaz_a}
    det_b = {'time': t + 1.0, 'backazimuth': backaz_b}

    results = triangulate_two_arrays([det_a], [det_b], center_a, center_b, origin=None, time_tolerance=5.0)
    assert len(results) == 1

    r = results[0]
    # result should be near the source
    assert abs(r['lat'] - source[0]) < 0.01
    assert abs(r['lon'] - source[1]) < 0.01


def test_triangulation_time_tolerance():
    # If times are too far apart we should get no results
    center_a = (0.0, 0.0)
    center_b = (0.0, 0.2)

    # make perfectly pointing backazimuths
    det_a = {'time': 1000.0, 'backazimuth': 200.0}
    det_b = {'time': 2000.0, 'backazimuth': 20.0}

    results = triangulate_two_arrays([det_a], [det_b], center_a, center_b, time_tolerance=1.0)
    assert len(results) == 0


def test_triangulation_lsq_two_array():
    # Test that providing slowness allows the LSQ locator to run for two arrays
    center_a = (0.0, 0.0)
    center_b = (0.0, 0.2)

    # source location
    source = (0.05, 0.1)

    # compute array->source bearing
    s_ax = (latlon_to_xy(source[0], source[1], center_a[0], center_a[1]))
    s_bx = (latlon_to_xy(source[0], source[1], center_b[0], center_b[1]))
    bearing_a = _angle_deg_from_vector(s_ax[0], s_ax[1])
    bearing_b = _angle_deg_from_vector(s_bx[0], s_bx[1])
    backaz_a = (bearing_a + 180.0) % 360.0
    backaz_b = (bearing_b + 180.0) % 360.0

    # assume wave speed 6 km/s (slowness ~0.1667 s/km)
    s = 1.0 / 6.0
    t = 1000.0
    det_a = {'time': t, 'backazimuth': backaz_a, 'slowness': s, 'subarray_id': 0}
    # small time offset due to range difference
    det_b = {'time': t + 0.2, 'backazimuth': backaz_b, 'slowness': s, 'subarray_id': 1}

    results = triangulate_two_arrays([det_a], [det_b], center_a, center_b, time_tolerance=5.0, use_lsq_if_available=True)
    assert len(results) == 1
    r = results[0]
    assert 'origin_time' in r or r.get('method') == 'lsq_2array'
    # location should be near the true source
    assert abs(r['lat'] - source[0]) < 0.05
    assert abs(r['lon'] - source[1]) < 0.05


def test_lsq_vel_snr_gating():
    # velocities mismatch should skip LSQ
    center_a = (0.0, 0.0)
    center_b = (0.0, 0.2)
    source = (0.05, 0.1)
    s_ax = (latlon_to_xy(source[0], source[1], center_a[0], center_a[1]))
    s_bx = (latlon_to_xy(source[0], source[1], center_b[0], center_b[1]))
    bearing_a = _angle_deg_from_vector(s_ax[0], s_ax[1])
    bearing_b = _angle_deg_from_vector(s_bx[0], s_bx[1])
    backaz_a = (bearing_a + 180.0) % 360.0
    backaz_b = (bearing_b + 180.0) % 360.0

    # set dramatically different velocities: 1.5 km/s vs 4.5 km/s
    det_a = {'time': 1000.0, 'backazimuth': backaz_a, 'velocity': 1.5, 'snr': 20.0, 'subarray_id': 0}
    det_b = {'time': 1000.2, 'backazimuth': backaz_b, 'velocity': 4.5, 'snr': 20.0, 'subarray_id': 1}

    # with strict lsq_vel_tol small, LSQ should be skipped and we get intersection_mc
    results = triangulate_two_arrays([det_a], [det_b], center_a, center_b, time_tolerance=5.0, use_lsq_if_available=True, lsq_vel_tol=0.1)
    assert len(results) == 1
    assert results[0].get('method') == 'intersection_mc'

    # now make velocities compatible and SNR too low -> still skip due to SNR gate
    det_b2 = det_b.copy(); det_b2['velocity'] = 1.6; det_b2['snr'] = 2.0
    results2 = triangulate_two_arrays([det_a], [det_b2], center_a, center_b, time_tolerance=5.0, use_lsq_if_available=True, lsq_vel_tol=0.5, min_snr=30.0)
    assert len(results2) == 1
    assert results2[0].get('method') == 'intersection_mc'

    # when velocities compatible and SNR high, LSQ should run
    det_b3 = det_b.copy(); det_b3['velocity'] = 1.6; det_b3['snr'] = 30.0
    results3 = triangulate_two_arrays([det_a], [det_b3], center_a, center_b, time_tolerance=5.0, use_lsq_if_available=True, lsq_vel_tol=1.0, min_snr=10.0)
    assert len(results3) == 1
    assert results3[0].get('method') == 'lsq_2array'
