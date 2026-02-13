#!/usr/bin/env python3
"""
Summarize triangulation method usage and simple QC stats from a merged locations CSV.
"""
import argparse
import csv
import glob
import json
from collections import defaultdict
import math

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def summarize(csv_path, json_dir=None):
    counts = defaultdict(int)
    sum_err = defaultdict(float)
    sum_res = defaultdict(float)
    sum_snr = defaultdict(float)
    n_err = defaultdict(int)
    n_res = defaultdict(int)
    n_snr = defaultdict(int)

    if csv_path.endswith('.json') or json_dir is not None:
        files = []
        if csv_path.endswith('.json'):
            files = [csv_path]
        elif json_dir:
            files = sorted(glob.glob(f'{json_dir}/locations_*.json'))
        for f in files:
            try:
                data = json.load(open(f))
                for row in data:
                    method = row.get('method', 'intersection')
                    counts[method] += 1
                    err = safe_float(row.get('error_km'))
                    if err is not None:
                        sum_err[method] += err; n_err[method] += 1
                    res = safe_float(row.get('residual_norm'))
                    if res is not None:
                        sum_res[method] += res; n_res[method] += 1
                    snr = safe_float(row.get('mean_snr') or row.get('snr'))
                    if snr is not None:
                        sum_snr[method] += snr; n_snr[method] += 1
            except Exception:
                print('Failed to read', f)
        # and fall through to printing
    with open(csv_path, 'r', newline='') as fh:
        r = csv.DictReader(fh)
        for row in r:
            method = row.get('method','intersection')
            counts[method] += 1
            err = safe_float(row.get('error_km'))
            if err is not None:
                sum_err[method] += err; n_err[method] += 1
            res = safe_float(row.get('residual_norm'))
            if res is not None:
                sum_res[method] += res; n_res[method] += 1
            snr = safe_float(row.get('mean_snr') or row.get('snr'))
            if snr is not None:
                sum_snr[method] += snr; n_snr[method] += 1

    print('Summary for', csv_path)
    print('Method | Count | mean_error_km | mean_residual_norm | mean_snr')
    for m, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        me = (sum_err[m] / n_err[m]) if n_err[m] else float('nan')
        mr = (sum_res[m] / n_res[m]) if n_res[m] else float('nan')
        ms = (sum_snr[m] / n_snr[m]) if n_snr[m] else float('nan')
        print(f'{m:12s} | {c:6d} | {me:14.3f} | {mr:18.3f} | {ms:8.2f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True, help='Merged locations CSV (triangulated)')
    parser.add_argument('--json-dir', default=None, help='Optional directory containing locations_YYYYMMDD.json files')
    args = parser.parse_args()
    summarize(args.csv, json_dir=args.json_dir)

