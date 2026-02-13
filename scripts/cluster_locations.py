#!/usr/bin/env python3
"""
Cluster triangulation results into unique events and write a CSV summary.

Usage:
  python scripts/cluster_locations.py --locations locations.json --out locations_summary.csv --cluster-km 20.0 --min-members 1

This script expects each location entry to have 'time', 'lat', 'lon', 'error_km', 'arrays', 'backazimuths'.
"""
import argparse
import json
import math
import csv
from statistics import median
from datetime import datetime, timezone
import os


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def greedy_cluster(locs, eps_km=20.0):
    clusters = []
    for loc in locs:
        lat = loc['lat']
        lon = loc['lon']
        placed = False
        for cl in clusters:
            d = haversine(lat, lon, cl['centroid_lat'], cl['centroid_lon'])
            if d <= eps_km:
                cl['members'].append(loc)
                # update centroid
                cl['centroid_lat'] = sum(m['lat'] for m in cl['members']) / len(cl['members'])
                cl['centroid_lon'] = sum(m['lon'] for m in cl['members']) / len(cl['members'])
                placed = True
                break
        if not placed:
            clusters.append({'centroid_lat': lat, 'centroid_lon': lon, 'members': [loc]})
    return clusters


def write_summary_csv(clusters, out_csv, save_triangulated_only=True, triangulated_min_arrays=2, strict_triangulated=False):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    
    # Write all locations CSV
    with open(out_csv, 'w', newline='') as fh:
        writer = csv.writer(fh)
        # include mean_snr, azimuth_variance, method and residual for easy plotting/QA
        writer.writerow(['time_epoch', 'time_iso', 'lat', 'lon', 'error_km', 'count', 'arrays', 'union_arrays_count', 'backazimuths', 'time_span_s', 'triangulated', 'mean_snr', 'azimuth_variance', 'method', 'residual_norm'])
        for cl in clusters:
            mem = cl['members']
            times = [m['time'] for m in mem]
            lats = [m['lat'] for m in mem]
            lons = [m['lon'] for m in mem]
            errs = [m.get('error_km', 0) for m in mem]
            arrays = set()
            bazs = []
            for m in mem:
                arrays.update(m.get('arrays', []))
                b = m.get('backazimuths')
                if b:
                    bazs.extend([b[0], b[1]])
            epoch_median = median(times)
            dt = datetime.fromtimestamp(epoch_median, tz=timezone.utc).isoformat()
            lat_med = sum(lats) / len(lats)
            lon_med = sum(lons) / len(lons)
            err_med = median(errs)
            count = len(mem)
            arrays_str = ';'.join(sorted([str(a) for a in arrays if a is not None]))
            union_arrays_count = len(arrays)
            bazs_str = ';'.join([str(round(float(b), 1)) for b in bazs])
            # compute mean SNR across members (if provided) and azimuth circular variance
            snrs = [m.get('snr') for m in mem if m.get('snr') is not None]
            mean_snr = round(sum([float(x) for x in snrs]) / len(snrs), 3) if snrs else ''
            # circular variance for azimuths
            import numpy as _np
            if bazs:
                a = _np.radians([float(b) for b in bazs])
                R = _np.sqrt((_np.sum(_np.cos(a))**2 + _np.sum(_np.sin(a))**2)) / len(a)
                az_var = round(1.0 - float(R), 4)
            else:
                az_var = ''
            time_span = max(times) - min(times) if len(times) > 1 else 0.0
            # determine method and mean residual_norm across members
            methods = [m.get('method') for m in mem if m.get('method')]
            method = ';'.join(sorted(set(methods))) if methods else ''
            resid_vals = [float(m.get('residual_norm')) for m in mem if m.get('residual_norm') is not None]
            mean_resid = round(sum(resid_vals) / len(resid_vals), 3) if resid_vals else ''
            writer.writerow([round(epoch_median, 3), dt, round(lat_med, 6), round(lon_med, 6), round(err_med, 3), count, arrays_str, union_arrays_count, bazs_str, round(time_span, 3), union_arrays_count >= triangulated_min_arrays, mean_snr, az_var, method, mean_resid])
    print('Wrote CSV (all locations):', out_csv)
    
    # Write triangulated-only CSV (≥2 arrays means actual triangulation)
    if save_triangulated_only:
        # Triangulation modes:
        # - strict_triangulated=True: cluster is triangulated if the UNION of arrays across members >= triangulated_min_arrays
        # - strict_triangulated=False: cluster is triangulated if ANY member has arrays length >= triangulated_min_arrays
        triangulated_clusters = []
        for c in clusters:
            if strict_triangulated:
                union_arrays = set()
                for m in c['members']:
                    union_arrays.update(m.get('arrays', []))
                if len(union_arrays) >= triangulated_min_arrays:
                    triangulated_clusters.append(c)
            else:
                for m in c['members']:
                    arrays = m.get('arrays', [])
                    if isinstance(arrays, list) and len(arrays) >= triangulated_min_arrays:
                        triangulated_clusters.append(c)
                        break
            if triangulated_clusters:
                base, ext = os.path.splitext(out_csv)
                triang_csv = f"{base}_triangulated{ext}"
                with open(triang_csv, 'w', newline='') as fh:
                    writer = csv.writer(fh)
                writer.writerow(['time_epoch', 'time_iso', 'lat', 'lon', 'error_km', 'count', 'arrays', 'union_arrays_count', 'backazimuths', 'time_span_s', 'triangulated', 'mean_snr', 'azimuth_variance', 'method', 'residual_norm'])
                for cl in triangulated_clusters:
                    mem = cl['members']
                    times = [m['time'] for m in mem]
                    lats = [m['lat'] for m in mem]
                    lons = [m['lon'] for m in mem]
                    errs = [m.get('error_km', 0) for m in mem]
                    arrays = set()
                    bazs = []
                    for m in mem:
                        arrays.update(m.get('arrays', []))
                        b = m.get('backazimuths')
                        if b:
                            bazs.extend([b[0], b[1]])
                    epoch_median = median(times)
                    dt = datetime.fromtimestamp(epoch_median, tz=timezone.utc).isoformat()
                    lat_med = sum(lats) / len(lats)
                    lon_med = sum(lons) / len(lons)
                    err_med = median(errs)
                    count = len(mem)
                    arrays_str = ';'.join(sorted([str(a) for a in arrays if a is not None]))
                    union_arrays_count = len(arrays)
                    bazs_str = ';'.join([str(round(float(b), 1)) for b in bazs])
                    # compute mean_snr and az var for this triangulated cluster
                    snrs = [m.get('snr') for m in mem if m.get('snr') is not None]
                    mean_snr = round(sum([float(x) for x in snrs]) / len(snrs), 3) if snrs else ''
                    import numpy as _np
                    if bazs:
                        a = _np.radians([float(b) for b in bazs])
                        R = _np.sqrt((_np.sum(_np.cos(a))**2 + _np.sum(_np.sin(a))**2)) / len(a)
                        az_var = round(1.0 - float(R), 4)
                    else:
                        az_var = ''
                    time_span = max(times) - min(times) if len(times) > 1 else 0.0
                    methods = [m.get('method') for m in mem if m.get('method')]
                    method = ';'.join(sorted(set(methods))) if methods else ''
                    resid_vals = [float(m.get('residual_norm')) for m in mem if m.get('residual_norm') is not None]
                    mean_resid = round(sum(resid_vals) / len(resid_vals), 3) if resid_vals else ''
                    writer.writerow([round(epoch_median, 3), dt, round(lat_med, 6), round(lon_med, 6), round(err_med, 3), count, arrays_str, union_arrays_count, bazs_str, round(time_span, 3), union_arrays_count >= triangulated_min_arrays, mean_snr, az_var, method, mean_resid])
                    print(f'Wrote CSV (triangulated only, ≥{triangulated_min_arrays} arrays): {triang_csv} ({len(triangulated_clusters)} clusters)')
                else:
                    print('No triangulated locations (≥2 arrays) found - skipping triangulated CSV')


def main():
    parser = argparse.ArgumentParser(description='Cluster locations JSON into summary CSV')
    parser.add_argument('--locations', required=True, help='Path to locations JSON file (list of dicts)')
    parser.add_argument('--out', required=True, help='CSV output file path')
    parser.add_argument('--cluster-km', type=float, default=20.0, help='Cluster radius (km) for merging close location estimates')
    parser.add_argument('--min-members', type=int, default=1, help='Minimum members required for a cluster to be retained')
    parser.add_argument('--no-triangulated-csv', action='store_true', help='Do not save separate triangulated-only CSV')
    parser.add_argument('--triangulated-min-arrays', type=int, default=2, help='Minimum number of arrays to consider a cluster triangulated (default 2)')
    parser.add_argument('--strict-triangulated', action='store_true', help='Require union of arrays across cluster members to reach triangulated-min-arrays (default: any member having arrays >= min)')
    args = parser.parse_args()

    locs = json.load(open(args.locations))
    clusters = greedy_cluster(locs, eps_km=args.cluster_km)
    # filter by min members
    clusters = [c for c in clusters if len(c['members']) >= args.min_members]
    write_summary_csv(clusters, args.out, save_triangulated_only=not args.no_triangulated_csv, triangulated_min_arrays=args.triangulated_min_arrays, strict_triangulated=args.strict_triangulated)


if __name__ == '__main__':
    main()
