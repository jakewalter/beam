#!/usr/bin/env python3
"""
Run multiseed LSQ localization across per-day detections and save results.

This script mirrors the LSQ grouping logic in `beam_driver` but forces
`use_multiseed=True` with configurable seed radii and seeds per ring.
"""
import argparse
import json
import os
import glob
import numpy as np
from beam.core.locator import locate_multarray_least_squares


def load_detections_file(fp):
    data = json.load(open(fp))
    # flatten dict or list
    rows = []
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, list): rows.extend(v)
    elif isinstance(data, list):
        rows = data
    return rows


def group_by_time(dets, window_s=5.0):
    groups = []
    if not dets:
        return groups
    sorted_all = sorted(dets, key=lambda x: x['time'])
    cur = [sorted_all[0]]
    for d in sorted_all[1:]:
        if d['time'] - cur[-1]['time'] <= window_s:
            cur.append(d)
        else:
            groups.append(cur)
            cur = [d]
    if cur:
        groups.append(cur)
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detections-dir', required=True, help='Top-level plot dir with detections_YYYYMMDD.json files')
    parser.add_argument('--centers', required=True)
    parser.add_argument('--out-dir', default='plots', help='Output dir for LSQ results')
    parser.add_argument('--seed-radii', default='0,10,50', help='Comma-separated radii in km')
    parser.add_argument('--seeds-per-ring', type=int, default=8)
    parser.add_argument('--time-window', type=float, default=5.0)
    args = parser.parse_args()

    centers = json.load(open(args.centers))
    files = glob.glob(os.path.join(args.detections_dir, 'detections_*.json'))
    files.sort()
    os.makedirs(args.out_dir, exist_ok=True)
    radii = [float(x) for x in args.seed_radii.split(',') if x.strip()]

    for fp in files:
        try:
            dets = load_detections_file(fp)
        except Exception as e:
            print('Failed to load', fp, e)
            continue
        groups = group_by_time(dets, window_s=args.time_window)
        lsq_results = []
        for g in groups:
            arr_ids = set([d.get('subarray_id', None) for d in g if d.get('subarray_id', None) is not None])
            if len(arr_ids) < 3:
                continue
            # build centers mapping for arrays present
            c = {}
            for aid in arr_ids:
                sid = str(aid)
                if sid in centers:
                    c[aid] = tuple(centers[sid])
            if len(c) < 3:
                continue
            res = locate_multarray_least_squares(g, c, use_multiseed=True, seed_radii_km=radii, seeds_per_ring=args.seeds_per_ring)
            if res and res.get('success'):
                res['member_count'] = len(g)
                lsq_results.append(res)
        if lsq_results:
            basename = os.path.basename(fp).replace('detections_','').replace('.json','')
            outpath = os.path.join(args.out_dir, f'locations_lsq_{basename}_multiseed.json')
            with open(outpath, 'w') as fh:
                json.dump(lsq_results, fh, indent=2)
            print('Wrote', outpath, len(lsq_results))

if __name__ == '__main__':
    main()
