"""
Triangulation helpers for combining two-array detections into location estimates.

This module provides small utility functions to convert lat/lon to local
cartesian coordinates (km), compute intersections of bearing lines from array
centers, and a convenience function that accepts paired detections and
returns intersection-based location estimates.

The implementation uses a simple equirectangular approximation which is
accurate enough for regional scales (tens to a few hundreds of km).
"""

import math
from typing import Optional, Tuple, Dict, List
import random
import statistics


def latlon_to_xy(lat: float, lon: float, origin_lat: float, origin_lon: float) -> Tuple[float, float]:
    """Convert lat/lon degrees to local x,y in kilometers using equirectangular approx.

    Parameters
    ----------
    lat, lon: coordinates to convert
    origin_lat, origin_lon: origin latitude/longitude used as reference

    Returns
    -------
    (x_km, y_km): tuple of floats
    """
    # mean latitude for scaling longitude degrees
    mean_lat = math.radians(origin_lat)
    dx = (lon - origin_lon) * 111.32 * math.cos(mean_lat)
    dy = (lat - origin_lat) * 110.54
    return dx, dy


def xy_to_latlon(x: float, y: float, origin_lat: float, origin_lon: float) -> Tuple[float, float]:
    """Inverse of latlon_to_xy

    Returns (lat, lon)
    """
    mean_lat = math.radians(origin_lat)
    lon = origin_lon + x / (111.32 * math.cos(mean_lat))
    lat = origin_lat + y / 110.54
    return lat, lon


def intersection_of_two_bearings(p1: Tuple[float, float], az1_deg: float,
                                 p2: Tuple[float, float], az2_deg: float,
                                 ) -> Optional[Tuple[float, float]]:
    """Compute intersection point of two infinite lines defined by points and azimuths.

    Azimuths given in degrees are treated as forward bearings (0 = north, 90 = east)
    and measured clockwise.

    Returns (x, y) if intersection exists or None when lines are nearly parallel.
    """
    # convert azimuths to radians and to standard mathematical angle (from x-axis)
    # in our coordinate system x = East, y = North. Standard math angle theta
    # from +x axis counterclockwise equals: theta = 90deg - azimuth
    x1, y1 = p1
    x2, y2 = p2
    th1 = math.radians(90.0 - az1_deg)
    th2 = math.radians(90.0 - az2_deg)

    # line parametric: p1 + t * v1, p2 + s * v2, where v = (cos(th), sin(th))
    v1x, v1y = math.cos(th1), math.sin(th1)
    v2x, v2y = math.cos(th2), math.sin(th2)

    denom = v1x * v2y - v1y * v2x

    if abs(denom) < 1e-6:
        # nearly parallel
        return None

    # solve for t such that p1 + t*v1 intersects p2 + s*v2
    t = ((x2 - x1) * v2y - (y2 - y1) * v2x) / denom

    xi = x1 + t * v1x
    yi = y1 + t * v1y

    return xi, yi


def triangulate_two_arrays(dets_a: List[Dict], dets_b: List[Dict],
                           center_a: Tuple[float, float], center_b: Tuple[float, float],
                           origin: Tuple[float, float] = None,
                           time_tolerance: float = 5.0,
                           min_angle_deg: float = 15.0,
                           use_lsq_if_available: bool = True,
                           mc_az_sigma_deg: float = 3.0,
                           mc_samples: int = 200,
                           min_snr: float = 0.0,
                           lsq_vel_min: float = 1.0,
                           lsq_vel_max: float = 6.0,
                           lsq_vel_tol: float = 0.5,
                           lsq_force_velocity: float = None) -> List[Dict]:
    """Triangulate location estimates from two sets of per-array detections.

    Parameters
    ----------
    dets_a, dets_b : list of detection dicts
        Detection dicts must contain: 'time' (UTC epoch seconds) and
        'backazimuth' (degrees). 'velocity' or 'slowness' are optional.
    center_a, center_b : (lat, lon)
        Geographic centers of the two arrays (degrees)
    origin : (lat, lon), optional
        Reference origin for the local projection. If None, use midpoint of
        the two array centers.
    time_tolerance : float
        Maximum separation in seconds between detections to be considered a pair.

    Returns
    -------
    results : list of dicts
    min_snr: float = 0.0,
    lsq_vel_min: float = 1.0,
    lsq_vel_max: float = 6.0,
    lsq_vel_tol: float = 0.5) -> List[Dict]:
        Each dict contains 'time', 'lat', 'lon', 'error_km', 'arrays', 'backazimuths'
    """
    if origin is None:
        origin = ((center_a[0] + center_b[0]) / 2.0, (center_a[1] + center_b[1]) / 2.0)

    # precompute centers in local xy
    a_x, a_y = latlon_to_xy(center_a[0], center_a[1], origin[0], origin[1])
    b_x, b_y = latlon_to_xy(center_b[0], center_b[1], origin[0], origin[1])

    results = []

    # helper: small angle normalization
    def _azimuth_angle_diff(a, b):
        d = abs((a - b + 180.0) % 360.0 - 180.0)
        return d

    for da in dets_a:
        for db in dets_b:
            tdiff = abs(da['time'] - db['time'])
            if tdiff > time_tolerance:
                continue

            # convert backazimuth (direction from source -> array) to bearing from
            # array -> source: bearing = (backazimuth + 180) % 360
            if 'backazimuth' not in da or 'backazimuth' not in db:
                continue

            az_a = (da['backazimuth'] + 180.0) % 360.0
            az_b = (db['backazimuth'] + 180.0) % 360.0

            # require a minimum angular separation between bearings; skip if nearly parallel/antipodal
            angle_sep = _azimuth_angle_diff(az_a, az_b)
            if angle_sep < min_angle_deg or angle_sep > (180.0 - min_angle_deg):
                continue

            inter = intersection_of_two_bearings((a_x, a_y), az_a, (b_x, b_y), az_b)

            if inter is None:
                # skip nearly parallel cases
                continue

            xi, yi = inter
            lat, lon = xy_to_latlon(xi, yi, origin[0], origin[1])

            # simple error metric: distance of intersection to each bearing line
            # (we use distances along perpendicular to line)
            # compute perpendicular distances (km)
            def point_line_distance(px, py, lx, ly, az_deg):
                th = math.radians(90.0 - az_deg)
                vx, vy = math.cos(th), math.sin(th)
                # vector from line point to point
                rx, ry = px - lx, py - ly
                # perpendicular component magnitude = |r x v|
                return abs(rx * vy - ry * vx)

            err_a = point_line_distance(xi, yi, a_x, a_y, az_a)
            err_b = point_line_distance(xi, yi, b_x, b_y, az_b)
            err = max(err_a, err_b)

            res = {
                'time': 0.5 * (da['time'] + db['time']),
                'lat': lat,
                'lon': lon,
                'error_km': err,
                'arrays': (da.get('subarray_id', None), db.get('subarray_id', None)),
                'backazimuths': (da['backazimuth'], db['backazimuth']),
                # include SNR info for downstream QC/plotting
                'snr': None if ('snr' not in da and 'snr' not in db) else float(((da.get('snr', 0.0) or 0.0) + (db.get('snr', 0.0) or 0.0)) / 2.0),
                'time_diff': tdiff
            }

            # if slowness/velocity are present, attempt LSQ to get better origin/time and position
            def _get_slowness(d):
                if 'slowness' in d:
                    return float(d['slowness'])
                if 'velocity' in d and d['velocity']:
                    return 1.0 / float(d['velocity'])
                return None
            s_a = _get_slowness(da)
            s_b = _get_slowness(db)
            # Optional force: override slowness to a fixed value if specified
            if lsq_force_velocity is not None:
                try:
                    forced_s = 1.0 / float(lsq_force_velocity)
                    s_a = forced_s
                    s_b = forced_s
                except Exception:
                    pass
            if use_lsq_if_available and s_a is not None and s_b is not None:
                try:
                    from .locator import locate_multarray_least_squares
                    # build detection dicts for LSQ
                    aid = da.get('subarray_id', 0)
                    bid = db.get('subarray_id', 1)
                    meas = [
                        {'subarray_id': aid, 'time': float(da['time']), 'backazimuth': float(da['backazimuth']), 'slowness': float(s_a)},
                        {'subarray_id': bid, 'time': float(db['time']), 'backazimuth': float(db['backazimuth']), 'slowness': float(s_b)}
                    ]
                    centers = {aid: center_a, bid: center_b}
                    # Check velocities and SNR gating before running LSQ
                    v_a = 1.0 / float(s_a)
                    v_b = 1.0 / float(s_b)
                    mean_v = 0.5 * (v_a + v_b)
                    vel_ok = (lsq_vel_min <= mean_v <= lsq_vel_max) and (abs(v_a - v_b) <= lsq_vel_tol)
                    if da.get('snr') is not None or db.get('snr') is not None:
                        svals = [(da.get('snr', 0.0) or 0.0), (db.get('snr', 0.0) or 0.0)]
                        avg_snr = float(sum(svals) / len(svals))
                    else:
                        avg_snr = 0.0

                    if vel_ok and avg_snr >= min_snr:
                        lsq = locate_multarray_least_squares(meas, centers, origin=origin, use_multiseed=False)
                    else:
                        lsq = None
                    if lsq.get('success'):
                        # override position from LSQ
                        res['lat'] = lsq['lat']
                        res['lon'] = lsq['lon']
                        res['origin_time'] = lsq.get('origin_time')
                        res['residual_norm'] = lsq.get('residual_norm')
                        res['method'] = 'lsq_2array'
                        # optionally compute error_km from covariance diagonal if present
                        cov = lsq.get('covariance')
                        if cov:
                            # covariance corresponds to x,y,t0 but in xy units; take sqrt of first two diagonal entries
                            try:
                                err_xy = math.sqrt(abs(cov[0][0]) + abs(cov[1][1]))
                                # convert small xy km to approximate error
                                res['error_km'] = err_xy
                            except Exception:
                                pass
                except Exception:
                    # if locator not available or fails, continue with intersection
                    pass

            # when using backazimuths only: compute a Monte Carlo std_km if needed
            if 'method' not in res:
                # sample perturbed azimuths to estimate uncertainty
                from math import radians
                try:
                    mc = monte_carlo_intersection((a_x, a_y), az_a, (b_x, b_y), az_b, origin, n_samples=mc_samples, az_sigma_deg=mc_az_sigma_deg)
                    if mc is not None:
                        res['lat'] = mc[0]
                        res['lon'] = mc[1]
                        res['error_km'] = mc[2]
                        res['method'] = 'intersection_mc'
                except Exception:
                    pass

            results.append(res)

    return results


def monte_carlo_intersection(p1, az1_deg, p2, az2_deg, origin, n_samples=200, az_sigma_deg=3.0):
    """Monte Carlo sampling for 2-array intersection uncertainty.
    Returns (median_lat, median_lon, std_km) or None if sampling fails.
    """
    samples = []
    for _ in range(n_samples):
        a1 = az1_deg + random.gauss(0.0, az_sigma_deg)
        a2 = az2_deg + random.gauss(0.0, az_sigma_deg)
        inter = intersection_of_two_bearings(p1, a1, p2, a2)
        if inter is None:
            continue
        xi, yi = inter
        lat, lon = xy_to_latlon(xi, yi, origin[0], origin[1])
        samples.append((lat, lon))
    if not samples:
        return None
    lats = [s[0] for s in samples]
    lons = [s[1] for s in samples]
    med_lat = statistics.median(lats)
    med_lon = statistics.median(lons)

    # compute distances to median
    def _haversine_km(a_lat, a_lon, b_lat, b_lon):
        R = 6371.0
        dlat = math.radians(b_lat - a_lat)
        dlon = math.radians(b_lon - a_lon)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(a_lat)) * math.cos(math.radians(b_lat)) * math.sin(dlon/2)**2
        return 2 * R * math.asin(math.sqrt(a))

    dists = [_haversine_km(med_lat, med_lon, s[0], s[1]) for s in samples]
    std_km = statistics.pstdev(dists) if len(dists) > 1 else 0.0
    return (med_lat, med_lon, std_km)
