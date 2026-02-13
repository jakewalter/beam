#!/usr/bin/env python3
"""
Merge detection JSON files from a top-level `plot_dir` (or nested) into one combined JSON or CSV.

The script recursively finds files named `detections_*.json` under the provided folder
and merges their detection records into a single output file.

Output format: JSON (list of dicts) or CSV (if --out-csv provided)

Options:
  --input-dir, --in-dir  : Directory to search under (e.g., /scratch2/time/day_volumes)
  --out-json             : Path to output JSON file (default: plots/detections_merged.json)
  --out-csv              : Path to output CSV if you prefer CSV
  --min-snr              : Filter detections with snr < min_snr
  --limit                : Limit total number of detection records merged (0 = no limit)
"""
import argparse
import os
import json
import csv
from glob import iglob
from typing import List


def find_detection_files(input_dir: str):
    # find any file under input_dir named detections_*.json
    pattern = os.path.join(input_dir, '**', 'detections_*.json')
    for p in iglob(pattern, recursive=True):
        yield p


def load_detections_from_file(fp: str):
    with open(fp, 'r') as fh:
        data = json.load(fh)
    # data may be dict mapping date->list, or list of dicts
    out = []
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, list):
                out.extend(v)
    elif isinstance(data, list):
        out.extend(data)
    return out


def write_csv(rows: List[dict], out_csv: str):
    # determine header from union of keys
    keys = set()
    for r in rows:
        keys.update(r.keys())
    keys = sorted(keys)
    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    with open(out_csv, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in keys})


def main():
    parser = argparse.ArgumentParser(description='Merge detection JSONs into a single file')
    parser.add_argument('--in-dir', '--input-dir', dest='input_dir', required=True)
    parser.add_argument('--out-json', default='plots/detections_merged.json')
    parser.add_argument('--out-csv', default=None)
    parser.add_argument('--min-snr', type=float, default=0.0)
    parser.add_argument('--limit', type=int, default=0, help='Limit number of merged detections (0 = all)')
    args = parser.parse_args()

    all_rows = []
    files = list(find_detection_files(args.input_dir))
    files.sort()
    print(f'Found {len(files)} detection files under {args.input_dir}')
    for fp in files:
        try:
            rows = load_detections_from_file(fp)
        except Exception as e:
            print(f'Warning: could not read {fp}: {e}')
            continue
        for r in rows:
            # ensure 'snr' exists and is a float
            try:
                snr = float(r.get('snr', 0.0) or 0.0)
            except Exception:
                snr = 0.0
            if snr < args.min_snr:
                continue
            all_rows.append(r)
            if args.limit and args.limit > 0 and len(all_rows) >= args.limit:
                break
        if args.limit and args.limit > 0 and len(all_rows) >= args.limit:
            break

    print(f'Collected {len(all_rows)} detections')
    out_json = args.out_json
    os.makedirs(os.path.dirname(out_json) or '.', exist_ok=True)
    with open(out_json, 'w') as fh:
        json.dump(all_rows, fh)
    print('Wrote', out_json)
    if args.out_csv:
        write_csv(all_rows, args.out_csv)
        print('Wrote CSV', args.out_csv)


if __name__ == '__main__':
    main()
