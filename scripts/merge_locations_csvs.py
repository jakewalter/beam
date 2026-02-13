#!/usr/bin/env python3
"""
Merge multiple 'locations_summary_YYYYMMDD.csv' files into a single CSV.

Usage:
  PYTHONPATH=. python3 scripts/merge_locations_csvs.py --input-dir /path/to/plots --out merged.csv --pattern 'locations_summary_*.csv' --add-source

Options
  --input-dir: Directory to search for CSV files (default: current directory)
  --pattern: Glob pattern to match CSV files
  --out: Output file path (default: locations_summary_all.csv)
  --add-source: Add a column 'source_file' with the basename of the source CSV
  --unique-by: Comma-separated column names to deduplicate on (exact equality of column values)
  --force: Overwrite output if exists
  --verbose: Print processing info

This uses only the Python standard library and handles mismatching headers by creating a union of columns.
"""

import argparse
import csv
import glob
import os
import sys
import logging


def find_csv_files(input_dir, pattern):
    p = os.path.join(input_dir, pattern)
    return sorted(glob.glob(p))


def collect_headers(files):
    headers = []
    seen = set()
    for f in files:
        try:
            with open(f, 'r', newline='') as fh:
                reader = csv.reader(fh)
                hdr = next(reader, None)
                if hdr is None:
                    continue
                for c in hdr:
                    if c not in seen:
                        seen.add(c)
                        headers.append(c)
        except Exception as e:
            logging.warning(f"Failed to read header from {f}: {e}")
    return headers


def merge_csvs(files, outpath, add_source=False, unique_by=None, force=False, verbose=False):
    if not files:
        raise RuntimeError("No files to merge")

    if os.path.exists(outpath) and not force:
        raise RuntimeError(f"Output exists: {outpath} (use --force to overwrite)")

    headers = collect_headers(files)
    if add_source and 'source_file' not in headers:
        headers.append('source_file')

    unique_by_cols = []
    if unique_by:
        unique_by_cols = [c.strip() for c in unique_by.split(',') if c.strip()]
        for c in unique_by_cols:
            if c not in headers:
                logging.warning(f"unique-by column '{c}' not found in headers; dedupe will be exact on available columns")

    seen_keys = set()
    total_rows = 0

    with open(outpath, 'w', newline='') as outfh:
        writer = csv.DictWriter(outfh, fieldnames=headers)
        writer.writeheader()

        for f in files:
            if verbose:
                print(f"Merging {f}")
            try:
                with open(f, 'r', newline='') as fh:
                    reader = csv.DictReader(fh)
                    for row in reader:
                        # Normalize row to contain all headers
                        outrow = {k: row.get(k, '') for k in headers}
                        if add_source:
                            outrow['source_file'] = os.path.basename(f)

                        if unique_by_cols:
                            key = tuple(outrow.get(c, '') for c in unique_by_cols)
                            if key in seen_keys:
                                continue
                            seen_keys.add(key)

                        writer.writerow(outrow)
                        total_rows += 1
            except Exception as e:
                logging.warning(f"Failed to process {f}: {e}")

    if verbose:
        print(f"Wrote {total_rows} rows to {outpath}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge locations CSV files')
    parser.add_argument('--input-dir', default='.', help='Directory to search for CSVs')
    parser.add_argument('--pattern', default='locations_summary_*.csv', help='Glob pattern to match files')
    parser.add_argument('--out', default='locations_summary_all.csv', help='Output CSV path')
    parser.add_argument('--add-source', action='store_true', help='Add a source_file column containing the basename of the input file')
    parser.add_argument('--unique-by', default=None, help='Comma-separated list of columns to deduplicate by')
    parser.add_argument('--force', action='store_true', help='Overwrite output if exists')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    files = find_csv_files(args.input_dir, args.pattern)
    if args.verbose:
        print(f"Found {len(files)} files matching {args.pattern} in {args.input_dir}")
    if not files:
        print('No files found - exiting')
        sys.exit(0)

    try:
        merge_csvs(files, args.out, add_source=args.add_source, unique_by=args.unique_by, force=args.force, verbose=args.verbose)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"Merged {len(files)} files into {args.out}")
