#!/usr/bin/env python3
"""
Post-process merged/per-day location outputs.

Features:
 - Prefer LSQ solutions over intersection-based ones when events are nearby in time+space
 - Remove `intersection_mc` solutions that are within a given radius of any center (likely ring/artifact)
 - Output filtered JSON and CSV and report simple stats

Usage:
 python scripts/post_process_locations.py --json plots/locations_test_lsq3_0_all.json --centers /scratch2/.../centers.json --out plots/locations_test_lsq3_0_all_filtered.json --prefer-radius-km 5 --filter-intersection-km 10
"""
import argparse
import json
import math
import os
from datetime import datetime
from collections import defaultdict


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def load_centers(fp):
    c = json.load(open(fp))
    return {str(k): (float(v[0]), float(v[1])) for k, v in c.items()}


def nearest_center_dist(lat, lon, centers):
    return min(haversine_km(lat, lon, clat, clon) for clat, clon in centers.values()) if centers else float('nan')


def group_nearby(events, time_tol_s=2.0, dist_tol_km=5.0):
    # Simple greedy grouping by time then spatial filter for small N
    events_sorted = sorted(events, key=lambda e: e.get('time', 0.0))
    groups = []
    for e in events_sorted:
        placed = False
        et = e.get('time', 0.0)
        elat = e.get('lat'); elon = e.get('lon')
        for g in groups:
            # compare to group's representative time/loc (first member)
            rep = g[0]
            if abs(rep.get('time', 0.0) - et) <= time_tol_s:
                d = haversine_km(float(rep.get('lat')), float(rep.get('lon')), float(elat), float(elon))
                if d <= dist_tol_km:
                    g.append(e)
                    placed = True
                    break
        if not placed:
            groups.append([e])
    return groups


def prefer_lsq(groups):
    keep = []
    for g in groups:
        # if any LSQ in group, keep LSQ entries only
        lsqs = [e for e in g if e.get('method') and 'lsq' in e.get('method')]
        if lsqs:
            keep.extend(lsqs)
        else:
            keep.extend(g)
    return keep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', required=True, help='Merged locations JSON file (list of dicts)')
    parser.add_argument('--centers', required=True, help='centers.json mapping')
    parser.add_argument('--out-json', required=True, help='Output filtered JSON')
    parser.add_argument('--out-csv', required=False, help='Optional output CSV path')
    parser.add_argument('--prefer-radius-km', type=float, default=5.0, help='Time+space tolerance to group events (s, km) for LSQ preference (spatial threshold in km)')
    parser.add_argument('--time-tol-s', type=float, default=2.0, help='Time tolerance in seconds for grouping')
    parser.add_argument('--filter-intersection-km', type=float, default=10.0, help='Drop intersection_mc events within this many km of nearest center')
    args = parser.parse_args()

    events = json.load(open(args.json))
    centers = load_centers(args.centers)

    # First, compute nearest-center distance for all
    for e in events:
        try:
            e['_nearest_center_km'] = nearest_center_dist(float(e.get('lat') or 0.0), float(e.get('lon') or 0.0), centers)
        except Exception:
            e['_nearest_center_km'] = float('nan')

    # Filter intersection_mc within centers
    filtered = []
    dropped_intersection = 0
    for e in events:
        if (e.get('method') or '').startswith('intersection') and e.get('_nearest_center_km', float('inf')) <= args.filter_intersection_km:
            dropped_intersection += 1
            continue
        filtered.append(e)

    # Now group nearby to prefer LSQ
    groups = group_nearby(filtered, time_tol_s=args.time_tol_s, dist_tol_km=args.prefer_radius_km)
    pref = prefer_lsq(groups)

    # Save results
    os.makedirs(os.path.dirname(args.out_json) or '.', exist_ok=True)
    with open(args.out_json, 'w') as fh:
        json.dump(pref, fh, indent=2)

    # Optional CSV
    if args.out_csv:
        import csv
        with open(args.out_csv, 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['time_epoch','time_iso','lat','lon','method','residual_norm','error_km','snr','nearest_center_km'])
            from datetime import datetime, timezone
            for e in pref:
                te = e.get('time','')
                iso = datetime.fromtimestamp(float(te), tz=timezone.utc).isoformat() if te else ''
                w.writerow([te, iso, e.get('lat',''), e.get('lon',''), e.get('method',''), e.get('residual_norm',''), e.get('error_km',''), e.get('snr',''), e.get('_nearest_center_km','')])

    print('Input count:', len(events))
    print('Dropped intersection_mc near centers (<=', args.filter_intersection_km, 'km):', dropped_intersection)
    print('Kept after preference:', len(pref))
    print('Wrote', args.out_json)


if __name__ == '__main__':
    main()
