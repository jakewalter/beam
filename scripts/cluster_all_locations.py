#!/usr/bin/env python3
"""
Cluster all locations_YYYYMMDD.json files in a directory and write per-day CSVs.

Usage:
  PYTHONPATH=. python3 scripts/cluster_all_locations.py \
    --locations-dir /path/to/plot_dir --out-dir /path/to/plot_dir --cluster-km 20.0 --min-members 1 \
    [--strict-triangulated] [--triangulated-min-arrays 2] [--parallel 4]

This script calls `scripts/cluster_locations.py` for each `locations_*.json` found
and writes `{out_dir}/locations_summary_<YYYYMMDD>.csv` and the triangulated
version `{out_dir}/locations_summary_<YYYYMMDD>_triangulated.csv`.
"""

import argparse
import glob
import os
import subprocess
from multiprocessing.pool import ThreadPool


def cluster_one(args_tuple):
    locations_file, out_file, cluster_km, min_members, strict, min_arrays = args_tuple
    cmd = [
        'python3', 'scripts/cluster_locations.py',
        '--locations', locations_file,
        '--out', out_file,
        '--cluster-km', str(cluster_km),
        '--min-members', str(min_members),
    ]
    if strict:
        cmd.append('--strict-triangulated')
    if min_arrays and int(min_arrays) != 2:
        cmd.extend(['--triangulated-min-arrays', str(min_arrays)])
    print('Executing:', ' '.join(cmd))
    try:
        subprocess.run(cmd, check=True)
        return (locations_file, True, None)
    except subprocess.CalledProcessError as e:
        return (locations_file, False, str(e))


def main():
    parser = argparse.ArgumentParser(description='Cluster all locations_*.json files in a directory')
    parser.add_argument('--locations-dir', required=True, help='Directory containing locations_YYYYMMDD.json files')
    parser.add_argument('--out-dir', default=None, help='Directory to write CSVs (defaults to locations-dir)')
    parser.add_argument('--pattern', default='locations_*.json', help='Glob pattern for location files')
    parser.add_argument('--cluster-km', type=float, default=20.0, help='Cluster radius (km)')
    parser.add_argument('--min-members', type=int, default=1, help='Minimum members for cluster')
    parser.add_argument('--strict-triangulated', action='store_true', help='Require union-of-arrays to mark cluster as triangulated')
    parser.add_argument('--triangulated-min-arrays', type=int, default=2, help='Minimum arrays to consider triangulated')
    parser.add_argument('--parallel', type=int, default=1, help='Number of parallel workers (default 1)')
    args = parser.parse_args()

    base = args.locations_dir
    outdir = args.out_dir or base
    os.makedirs(outdir, exist_ok=True)

    pattern = os.path.join(base, args.pattern)
    files = sorted(glob.glob(pattern))
    if not files:
        print('No files found matching', pattern)
        return 1

    tasks = []
    for f in files:
        # derive date
        bn = os.path.basename(f)
        # keep same naming as cluster script (locations_summary_<date>.csv)
        date_part = bn.replace('locations_', '').replace('.json', '')
        out_file = os.path.join(outdir, f'locations_summary_{date_part}.csv')
        tasks.append((f, out_file, args.cluster_km, args.min_members, args.strict_triangulated, args.triangulated_min_arrays))

    if args.parallel and args.parallel > 1:
        pool = ThreadPool(args.parallel)
        results = pool.map(cluster_one, tasks)
        pool.close()
        pool.join()
    else:
        results = [cluster_one(t) for t in tasks]

    success = [r for r in results if r[1]]
    failed = [r for r in results if not r[1]]
    print('\nSummary:')
    print(f'  Total files: {len(results)}')
    print(f'  Succeeded: {len(success)}')
    if failed:
        print(f'  Failed: {len(failed)}')
        for f, ok, err in failed:
            print('   -', f, err)
    else:
        print('  All succeeded')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
