#!/usr/bin/env python3
"""
Estimate triangulation location uncertainty using Monte Carlo perturbation of backazimuths.

Reads a `locations_summary_*.csv` (triangulated rows) and `centers.json` mapping
and for each row computes a Monte Carlo estimate of the location distribution
by perturbing each array's backazimuth with Gaussian noise (sigma deg).

Outputs a CSV with additional columns: mc_lat_std_km, mc_lon_std_km, mc_radial_std_km
and optionally writes a per-event small report or a sample visualization.
"""
import argparse
import csv
import json
import math
import os
import statistics
from typing import List, Tuple

import numpy as np
from plot_beam_polar import read_csv
from beam.core.triangulation import latlon_to_xy, xy_to_latlon, intersection_of_two_bearings


def parse_arrays_backaz(arrays_str: str, bazs_str: str):
    arrays = [s.strip() for s in str(arrays_str).split(';') if s.strip()]
    bazs = [float(s) for s in str(bazs_str).split(';') if s.strip()]
    # Map backaz values to arrays; if lengths mismatch, partition bazs equally
    if len(bazs) == len(arrays):
        mapping = {arrays[i]: bazs[i] for i in range(len(arrays))}
    else:
        # partition bazs into len(arrays) groups
        n = len(bazs)
        per = max(1, n // len(arrays))
        mapping = {}
        idx = 0
        for i, a in enumerate(arrays):
            group = bazs[idx: idx + per]
            if not group and idx < n:
                group = [bazs[idx]]
            if group:
                mapping[a] = sum(group) / len(group)
            else:
                mapping[a] = None
            idx += per
    return mapping


def km_distance(x1, y1, x2, y2):
    return math.hypot(x1 - x2, y1 - y2)


def estimate_error_for_row(arrays_mapping, centers, origin, n_samples=500, sigma_deg=5.0):
    # arrays_mapping: dict array_id -> backazimuth(deg)
    arr_ids = list(arrays_mapping.keys())
    if len(arr_ids) < 2:
        return None

    # precompute centers in local xy
    centers_xy = {aid: latlon_to_xy(centers[str(aid)][0], centers[str(aid)][1], origin[0], origin[1]) for aid in arr_ids}

    samples_lat = []
    samples_lon = []
    for _ in range(n_samples):
        pert = {}
        for a in arr_ids:
            baz = arrays_mapping[a]
            if baz is None:
                pert[a] = None
            else:
                pert[a] = (baz + np.random.normal(scale=sigma_deg)) % 360.0

        # For each unique pair compute intersection
        inters = []
        for i in range(len(arr_ids)):
            for j in range(i+1, len(arr_ids)):
                a1 = arr_ids[i]; a2 = arr_ids[j]
                if pert[a1] is None or pert[a2] is None:
                    continue
                # convert to array->source bearing
                az1 = (pert[a1] + 180.0) % 360.0
                az2 = (pert[a2] + 180.0) % 360.0
                p1 = centers_xy[a1]; p2 = centers_xy[a2]
                inter = intersection_of_two_bearings(p1, az1, p2, az2)
                if inter is not None:
                    xi, yi = inter
                    lat, lon = xy_to_latlon(xi, yi, origin[0], origin[1])
                    inters.append((lat, lon))
        if not inters:
            continue
        # combine intersections by simple mean
        lat = sum([i[0] for i in inters]) / len(inters)
        lon = sum([i[1] for i in inters]) / len(inters)
        samples_lat.append(lat)
        samples_lon.append(lon)

    if not samples_lat:
        return None

    # compute standard deviations in km using local xy at origin
    xs = []
    ys = []
    for lat, lon in zip(samples_lat, samples_lon):
        x, y = latlon_to_xy(lat, lon, origin[0], origin[1])
        xs.append(x); ys.append(y)
    meanx = statistics.mean(xs); meany = statistics.mean(ys)
    stdx = statistics.pstdev(xs)
    stdy = statistics.pstdev(ys)
    radial_std = statistics.pstdev([math.hypot(x-meanx, y-meany) for x,y in zip(xs, ys)])
    return {'std_x_km': stdx, 'std_y_km': stdy, 'radial_std_km': radial_std}


def main():
    parser = argparse.ArgumentParser(description='Estimate triangulation uncertainty via Monte Carlo')
    parser.add_argument('--csv', required=True, help='locations_summary_triangulated_*.csv')
    parser.add_argument('--centers', required=True, help='centers.json mapping array_id -> [lat, lon]')
    parser.add_argument('--out', default='plots/triangulation_error_summary.csv')
    parser.add_argument('--n-samples', type=int, default=500)
    parser.add_argument('--limit', type=int, default=0, help='Limit number of rows to process (0 = all)')
    parser.add_argument('--sigma-deg', type=float, default=5.0, help='backazimuth perturbation sigma in degrees')
    args = parser.parse_args()

    rows = read_csv(args.csv)
    with open(args.centers, 'r') as fh:
        centers = json.load(fh)

    # choose origin as mean of all centers
    all_lats = [v[0] for v in centers.values()]
    all_lons = [v[1] for v in centers.values()]
    origin = (sum(all_lats)/len(all_lats), sum(all_lons)/len(all_lons))

    out_rows = []
    all_rows = rows if isinstance(rows, list) else rows.to_dict(orient='records')
    if args.limit and args.limit > 0:
        all_rows = all_rows[:args.limit]
    for r in all_rows:
        arrays_str = r.get('arrays', '')
        bazs_str = r.get('backazimuths', '')
        mapping = parse_arrays_backaz(arrays_str, bazs_str)
        est = estimate_error_for_row(mapping, centers, origin, n_samples=args.n_samples, sigma_deg=args.sigma_deg)
        out = dict(r)
        if est is None:
            out.update({'mc_std_x_km': '', 'mc_std_y_km': '', 'mc_radial_std_km': ''})
        else:
            out.update({'mc_std_x_km': f"{est['std_x_km']:.3f}", 'mc_std_y_km': f"{est['std_y_km']:.3f}", 'mc_radial_std_km': f"{est['radial_std_km']:.3f}"})
        out_rows.append(out)

    # write CSV
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    keys = list(out_rows[0].keys()) if out_rows else []
    with open(args.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    print('Wrote', args.out)


if __name__ == '__main__':
    main()
