import numpy as np
from beam.core.locator import locate_multarray_least_squares


def _latlon_to_xy(lat, lon, origin_lat, origin_lon):
    mean_lat = np.radians(origin_lat)
    dx = (lon - origin_lon) * 111.32 * np.cos(mean_lat)
    dy = (lat - origin_lat) * 110.54
    return dx, dy


def test_locator_recovers_source():
    # three arrays forming a modest triangle around the source
    centers = {
        0: (0.0, 0.0),
        1: (0.0, 0.05),
        2: (0.06, 0.02)
    }

    # source and origin time
    source = (0.02, 0.02)
    t0 = 1000.0

    # assume constant velocity 4 km/s -> slowness 0.25 s/km
    s = 0.25

    # build detections with times and backazimuths
    dets = []
    for aid, (latc, lonc) in centers.items():
        x, y = _latlon_to_xy(source[0], source[1], latc, lonc)
        # vector from array center -> source
        vx, vy = _latlon_to_xy(source[0], source[1], latc, lonc)
        # compute distance (km)
        r = (vx**2 + vy**2) ** 0.5
        # compute backazimuth: direction from source -> array
        # vector array - source
        ax, ay = _latlon_to_xy(latc, lonc, source[0], source[1])
        theta = np.degrees(np.arctan2(ax, ay)) % 360.0

        ti = t0 + s * r
        dets.append({'subarray_id': aid, 'time': float(ti), 'backazimuth': float(theta), 'slowness': float(s)})

    res = locate_multarray_least_squares(dets, centers, origin=None, time_weight=1.0, dir_weight=0.5)

    assert res['success']
    # lat/lon recovered within about 0.02 degrees (~2 km)
    assert abs(res['lat'] - source[0]) < 0.03
    assert abs(res['lon'] - source[1]) < 0.03
    assert abs(res['origin_time'] - t0) < 0.5
