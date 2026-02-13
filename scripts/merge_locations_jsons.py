#!/usr/bin/env python3
"""
Merge multiple per-day 'locations_YYYYMMDD.json' arrays into a single JSON array.

Usage:
  PYTHONPATH=. python3 scripts/merge_locations_jsons.py --input-dir /path --out merged.json --pattern 'locations_*.json' --unique-by time_epoch,lat,lon

Options
  --input-dir: dir to search for JSON files (default: current directory)
  --pattern: glob pattern to match JSON files
  --out: output file path (default: locations_all.json)
  --unique-by: comma-separated list of keys to dedupe by (exact equality)
  --force: overwrite output if exists
  --verbose: print progress

This reads each file as a JSON array and writes a single merged array.
"""
import argparse
import glob
import json
import os


def find_json_files(input_dir, pattern):
    p = os.path.join(input_dir, pattern)
    return sorted(glob.glob(p))


def merge_jsons(files, outpath, unique_by=None, force=False, verbose=False):
    if not files:
        raise RuntimeError('No files to merge')

    if os.path.exists(outpath) and not force:
        raise RuntimeError(f'Output exists: {outpath} (use --force to overwrite)')

    unique_by_cols = []
    if unique_by:
        unique_by_cols = [c.strip() for c in unique_by.split(',') if c.strip()]

    merged = []
    seen_keys = set()
    total = 0

    for f in files:
        if verbose:
            print('Reading', f)
        try:
            with open(f, 'r') as fh:
                arr = json.load(fh)
                for row in arr:
                    total += 1
                    if unique_by_cols:
                        key = tuple(row.get(c, None) for c in unique_by_cols)
                        if key in seen_keys:
                            continue
                        seen_keys.add(key)
                    merged.append(row)
        except Exception as e:
            print('Failed to read', f, e)

    if verbose:
        print(f'Writing {len(merged)} of {total} items to {outpath}')
    with open(outpath, 'w') as outfh:
        json.dump(merged, outfh)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', default='.', help='Directory to search for JSON files')
    parser.add_argument('--pattern', default='locations_*.json', help='Glob pattern')
    parser.add_argument('--out', default='locations_all.json', help='Output JSON path')
    parser.add_argument('--unique-by', default=None, help='Comma-separated list of keys to deduplicate by')
    parser.add_argument('--force', action='store_true', help='Overwrite output if exists')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()

    files = find_json_files(args.input_dir, args.pattern)
    if args.verbose:
        print(f'Found {len(files)} files')
    try:
        merge_jsons(files, args.out, unique_by=args.unique_by, force=args.force, verbose=args.verbose)
    except Exception as e:
        print('Error:', e)
        exit(1)

    print('Merged', len(files), 'files into', args.out)
