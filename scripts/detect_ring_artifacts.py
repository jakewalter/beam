#!/usr/bin/env python3
"""
Detect ring artifacts by measuring distribution of distances from array centers.

For each triangulated location, compute distance to the nearest array center.
For each method, compute histogram of nearest distances and report peaks.

Outputs:
- prints summary tables
- writes `plots/ring_artifact_summary.csv` with per-method, per-distance-bin counts
- writes `plots/ring_artifact_example_locations.csv` showing sample points on detected rings

Usage:
  PYTHONPATH=. python3 scripts/detect_ring_artifacts.py --json plots/locations_bench_tri_30s_min_snr8_all.json --centers /scratch2/time/day_volumes/bench_june2020_snr10_drsc_subarrays/centers.json --out-dir plots

"""
import argparse
import json
import math
import os
from collections import defaultdict, Counter

try:
    import numpy as np
except Exception:
    print('numpy required')
    raise


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1); dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(max(0.0,1.0-a)))
    return R*c


def read_centers(fp):
    with open(fp, 'r') as fh:
        centers = json.load(fh)
    # centers likely a dict: id -> [lat, lon]
    out = []
    for k,v in centers.items():
        try:
            out.append((k, float(v[0]), float(v[1])))
        except Exception:
            pass
    return out


def read_json(fp):
    with open(fp, 'r') as fh:
        return json.load(fh)


def compute_nearest_center_dist(lat, lon, centers):
    best = None
    best_id=None
    for cid, clat, clon in centers:
        d = haversine_km(lat, lon, clat, clon)
        if best is None or d < best:
            best = d; best_id = cid
    return best, best_id


def analyze(json_fp, centers_fp, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    centers = read_centers(centers_fp)
    data = read_json(json_fp)

    by_method = defaultdict(list)
    entries = []
    for row in data:
        method = (row.get('method') or '').strip() or '(blank)'
        try:
            lat=float(row['lat']); lon=float(row['lon'])
        except Exception:
            continue
        dist, cid = compute_nearest_center_dist(lat, lon, centers)
        by_method[method].append((lat, lon, dist, cid, row))
        entries.append((method, lat, lon, dist, cid, row))

    # For each method, find histogram of distances and peaks
    summaries=[]
    global_max_bins = {}
    for method, arr in by_method.items():
        dists = [x[2] for x in arr]
        if not dists:
            continue
        # histogram: bins width 10 km up to 1000 km
        maxd = max(dists)
        binw = 10.0
        nbins = int(max(1, math.ceil(maxd/binw)))
        counts, edges = np.histogram(dists, bins=nbins, range=(0, nbins*binw))
        # find the top peaks (local maxima)
        peak_bins = []
        for i in range(len(counts)):
            left = counts[i-1] if i>0 else 0
            right = counts[i+1] if i < len(counts)-1 else 0
            if counts[i] > left and counts[i] > right and counts[i] >= max(3, np.mean(counts) + 2*np.std(counts)):
                peak_bins.append((i, counts[i]))
        # also report top bins
        top_bins = sorted([(i, counts[i]) for i in range(len(counts))], key=lambda kv: -kv[1])[:5]
        summaries.append((method, len(dists), top_bins, peak_bins, edges))
        global_max_bins[method] = (np.argmax(counts), counts.max(), edges)

    # write CSV summarizing top bins
    out_csv = os.path.join(out_dir, 'ring_artifact_summary.csv')
    with open(out_csv, 'w') as fh:
        fh.write('method,total_count,top_bin_index,top_bin_count,bin_start_km,bin_end_km\n')
        for method, n, top_bins, peak_bins, edges in summaries:
            for (i, c) in top_bins:
                fh.write(','.join(map(str, [method, n, i, c, edges[i], edges[i+1]])) + '\n')
    print('Wrote', out_csv)

    # write example points on detected rings (peaks)
    out_examples = os.path.join(out_dir, 'ring_artifact_example_locations.csv')
    with open(out_examples, 'w') as fh:
        fh.write('method,center_id,lat,lon,dist_km,time_epoch,time_iso,error_km,mean_snr,source\n')
        for method, n, top_bins, peak_bins, edges in summaries:
            # prefer explicit peak_bins, else top_bins
            targets = peak_bins if peak_bins else top_bins[:2]
            for i, c in targets:
                # find points within that bin
                bin_start = edges[i]; bin_end = edges[i+1]
                hits = [x for x in by_method.get(method, []) if x[2] >= bin_start and x[2] < bin_end]
                # take up to 10 examples
                for h in hits[:10]:
                    lat, lon, dist, cid, row = h
                    fh.write(','.join(map(str, [method, cid, lat, lon, dist, row.get('time_epoch',''), row.get('time_iso',''), row.get('error_km',''), row.get('mean_snr',''), row.get('source_file','')])) + '\n')
    print('Wrote', out_examples)

    # Print a compact summary
    for method, n, top_bins, peak_bins, edges in summaries:
        print('\nMethod:', method)
        print('Total', n)
        print('Top bins (index,count,start_km,end_km):')
        for i,c in top_bins:
            print(i, c, edges[i], edges[i+1])
        if peak_bins:
            print('Significant peaks at bins:')
            for i,c in peak_bins:
                print(i, c, edges[i], edges[i+1])

    # additionally, compute fraction of points with distance equal to a small set of repeated radii (candidate rings)
    ring_report = []
    for method, n, top_bins, peak_bins, edges in summaries:
        # compute fraction of points that live in any of top 3 bins as a proxy for ring
        top_inds = [i for i,c in top_bins[:3]]
        hits = sum([sum(1 for x in by_method[method] if x[2] >= edges[i] and x[2] < edges[i+1]) for i in top_inds])
        frac = hits / float(n) if n>0 else 0.0
        ring_report.append((method, n, hits, frac, top_inds, [(edges[i],edges[i+1]) for i in top_inds]))

    print('\nRing summary (method, total, hits_in_top3_bins, fraction):')
    for m in ring_report:
        print(m)

    return summaries, ring_report

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', required=True, help='Merged locations JSON (deduped)')
    parser.add_argument('--centers', required=True)
    parser.add_argument('--out-dir', default='plots')
    args = parser.parse_args()
    analyze(args.json, args.centers, args.out_dir)
