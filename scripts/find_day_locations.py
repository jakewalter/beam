#!/usr/bin/env python3
"""
Simple helper script to inspect triangulation/location JSONs created by beam_driver

Usage:
  python scripts/find_day_locations.py --plot-dir plots/detections_20200601 --date 20200601

This script prints the contents of `locations_<date>.json` and `locations_lsq_<date>.json`
if they exist.
"""
import argparse
import os
import json


def main():
    parser = argparse.ArgumentParser(description='Find and display day locations JSONs')
    parser.add_argument('--plot-dir', required=True, help='Plot dir path used by beam_driver')
    parser.add_argument('--date', required=True, help='Date (YYYYMMDD)')
    args = parser.parse_args()

    plot_dir = args.plot_dir
    date_str = args.date

    files_searched = []
    json_regular = os.path.join(plot_dir, f"locations_{date_str}.json")
    json_lsq = os.path.join(plot_dir, f"locations_lsq_{date_str}.json")

    for f in [json_regular, json_lsq]:
        files_searched.append(f)
        if os.path.exists(f):
            print(f"Found: {f}")
            j = json.load(open(f, 'r'))
            print(f"Entries: {len(j)}")
            print(json.dumps(j, indent=2)[:1000])
            print('\n---\n')
        else:
            print(f"Missing: {f}")

    print('\nSearched files:')
    for p in files_searched:
        print(' -', p)

if __name__ == '__main__':
    main()
