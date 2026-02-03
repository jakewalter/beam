"""
Multi-array least-squares location estimator.

This module implements a small non-linear least-squares fitter that
combines per-array detections containing time-of-arrival, backazimuth
and slowness (or velocity) to estimate a single source location and
origin time.

Model
-----
For each array i with center at (xi, yi) and detection providing
arrival time ti (seconds), backazimuth bi (degrees, direction from
source->array) and slowness si (s/km), we assume the source location
is at (x, y) and origin time t0 and that travel time to array center
is approximately r_i / v_i where r_i = distance((x,y),(xi,yi)) and
v_i = 1/si. Thus:

    t_i = t0 + s_i * r_i

Backazimuth gives an independent directional constraint — the bearing
from array->source should be (bi + 180) degrees. We encode this as a
cross-track perpendicular distance residual.

The fitter minimizes squared residuals combining TOA errors (seconds)
and cross-track distances (km) using reasonable default weights.
"""

from typing import List, Dict, Tuple, Optional
import math
import numpy as np
from scipy.optimize import least_squares

from .triangulation import latlon_to_xy, xy_to_latlon


def _deg_to_bearing_unit(az_deg: float) -> Tuple[float, float]:
    """Return unit vector (ux, uy) for bearing measured from North clockwise.

    x = East, y = North coordinate system.
    """
    th = math.radians(90.0 - az_deg)
    return math.cos(th), math.sin(th)


def locate_multarray_least_squares(detections: List[Dict],
                                   centers: Dict[int, Tuple[float, float]],
                                   origin: Optional[Tuple[float, float]] = None,
                                   time_weight: float = 1.0,
                                   dir_weight: float = 1.0,
                                   max_iter: int = 100,
                                   use_multiseed: bool = False,
                                   seed_radii_km: Optional[List[float]] = None,
                                   seeds_per_ring: int = 8) -> Dict:
    """Estimate source (lat, lon, origin_time) from multi-array detections.

    Parameters
    ----------
    detections : list of dict
        Each detection dict must contain:
            - 'subarray_id' (or array id matching keys of centers)
            - 'time' (float epoch seconds)
            - 'backazimuth' (degrees) direction from source->array
            - 'slowness' (s/km) OR 'velocity' (km/s)
    centers : dict
        Mapping array_id -> (lat, lon) center coordinates
    origin : (lat, lon) or None
        Reference for local projection. Defaults to centroid of centers.
    time_weight : float
        Weight (inverse std) applied to TOA residuals (seconds)
    dir_weight : float
        Weight applied to cross-track residuals (km)

    Returns
    -------
    result : dict
        {'lat', 'lon', 'origin_time', 'success', 'message', 'residuals', 'covariance'}
    """
    if len(detections) == 0:
        return {'success': False, 'message': 'no detections provided'}

    # Group detections by array ID and ensure we have numerical slowness
    meas = []
    for d in detections:
        aid = d.get('subarray_id')
        if aid is None:
            continue
        if aid not in centers:
            continue
        if 'slowness' in d:
            s = float(d['slowness'])
        elif 'velocity' in d and d['velocity'] is not None:
            s = 1.0 / float(d['velocity'])
        else:
            # skip detection without slowness/velocity
            continue

        if 'backazimuth' not in d or 'time' not in d:
            continue

        meas.append({'array': aid, 'time': float(d['time']), 'backazimuth': float(d['backazimuth']), 'slowness': s})

    if len(meas) < 2:
        return {'success': False, 'message': 'need at least two detections to locate'}

    # choose projection origin
    if origin is None:
        lats = [c[0] for c in centers.values()]
        lons = [c[1] for c in centers.values()]
        origin = (float(np.mean(lats)), float(np.mean(lons)))

    # centers in xy
    centers_xy = {aid: latlon_to_xy(lat, lon, origin[0], origin[1]) for aid, (lat, lon) in centers.items()}

    # initial guess: use backazimuth intersections for x,y (first pair) or centroid
    x0 = None
    if len(meas) >= 2:
        # pick two distinct arrays with non-parallel backazimuths
        for i in range(len(meas)):
            for j in range(i+1, len(meas)):
                a = meas[i]['array']; b = meas[j]['array']
                if a == b:
                    continue
                p1 = centers_xy[a]; p2 = centers_xy[b]
                # convert backazimuth (source->array) to array->source bearing
                az1 = (meas[i]['backazimuth'] + 180.0) % 360.0
                az2 = (meas[j]['backazimuth'] + 180.0) % 360.0
                # try intersection using simple math
                from .triangulation import intersection_of_two_bearings
                inter = intersection_of_two_bearings(p1, az1, p2, az2)
                if inter is not None:
                    x0 = inter
                    break
            if x0 is not None:
                break

    if x0 is None:
        # fallback: centroid of center positions
        xs = np.array([v[0] for v in centers_xy.values()])
        ys = np.array([v[1] for v in centers_xy.values()])
        x0 = (float(np.mean(xs)), float(np.mean(ys)))

    # initial origin time guess: earliest arrival minus mean slowness*distance between centroid and centers
    times = np.array([m['time'] for m in meas])
    centroid = x0
    dists = np.array([math.hypot(centroid[0] - centers_xy[m['array']][0], centroid[1] - centers_xy[m['array']][1]) for m in meas])
    mean_s = np.mean([m['slowness'] for m in meas])
    t0_guess = float(times.min() - mean_s * np.mean(dists))

    x_init = np.array([x0[0], x0[1], t0_guess])

    # Residual function: combine cross-track (km) and TOA (s) residuals
    def residuals(params):
        x, y, t0 = params
        res = []
        for m in meas:
            aid = m['array']
            xi, yi = centers_xy[aid]
            dx = x - xi; dy = y - yi
            r = math.hypot(dx, dy)
            # time residual
            pred_time = t0 + m['slowness'] * r
            tres = (m['time'] - pred_time) / max(1e-6, time_weight)
            # cross-track distance: perpendicular distance from (x,y) to bearing line
            bearing = (m['backazimuth'] + 180.0) % 360.0
            ux, uy = _deg_to_bearing_unit(bearing)
            # perpendicular (cross) distance magnitude
            cross = abs(dx * uy - dy * ux)
            dres = cross / max(1e-6, dir_weight)
            res.append(tres)
            res.append(dres)
        return np.array(res)

    # Run least squares
    def run_lsq_from_guess(x_guess):
        try:
            ls = least_squares(residuals, x_guess, max_nfev=max_iter)
            return ls
        except Exception:
            return None

    # Determine seed radii
    if seed_radii_km is None:
        seed_radii_km = [0.0]

    # If multi-seed is requested, evaluate multiple initial guesses and pick the best
    ls_candidates = []
    if use_multiseed:
        # ensure we have at least one seed radius (0 = centroid)
        for r_km in seed_radii_km:
            for i in range(max(1, seeds_per_ring)):
                ang = (2 * math.pi * i) / max(1, seeds_per_ring)
                xg = x_init.copy()
                # shift centroid in local xy coordinates
                xg[0] = x0[0] + r_km * math.cos(ang)
                xg[1] = x0[1] + r_km * math.sin(ang)
                ls = run_lsq_from_guess(xg)
                if ls is not None and ls.success:
                    ls_candidates.append(ls)
    else:
        ls = run_lsq_from_guess(x_init)
        if ls is not None:
            ls_candidates.append(ls)

    if len(ls_candidates) == 0:
        return {'success': False, 'message': 'optimizer failure or no seed produced a solution'}

    # choose candidate with lowest cost
    ls = min(ls_candidates, key=lambda _ls: float(_ls.cost))

    x_opt, y_opt, t0_opt = ls.x

    # estimate covariance ~ (J^T J)^-1 * residual_variance
    cov = None
    try:
        J = ls.jac
        JTJ = J.T.dot(J)
        # pseudo-inverse in case singular
        cov_mat = np.linalg.pinv(JTJ)
        cov = cov_mat.tolist()
    except Exception:
        cov = None

    lat_opt, lon_opt = xy_to_latlon(x_opt, y_opt, origin[0], origin[1])

    return {
        'success': bool(ls.success),
        'message': ls.message,
        'lat': float(lat_opt),
        'lon': float(lon_opt),
        'origin_time': float(t0_opt),
        'residual_norm': float(ls.cost),
        'jac_shape': list(ls.jac.shape) if hasattr(ls, 'jac') else None,
        'covariance': cov
    }
