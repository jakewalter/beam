#!/usr/bin/env python3
"""
Run forced LSQ velocity across full date range (from config) for multiple velocities.

This script invokes `scripts/run_triangulate_force_vel_full.py` for each velocity
and merges+post-processes results. It writes per-velocity merged JSONs and a
summary JSON and then optionally creates basic plots.
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime


def run_full_for_vel(cfg, vel, outdir, time_tol=30.0, min_snr=8.0, lsq_vel_min=1.5, lsq_vel_max=4.5, lsq_vel_tol=0.5):
    os.makedirs(outdir, exist_ok=True)
    # Run the existing full-run script for this velocity
    cmd = [
        'python3', 'scripts/run_triangulate_force_vel_full.py',
        '--config', cfg,
        '--force-vel', str(vel),
        '--outdir', outdir,
        '--time-tol', str(time_tol),
        '--min-snr', str(min_snr),
        '--lsq-vel-min', str(lsq_vel_min),
        '--lsq-vel-max', str(lsq_vel_max),
        '--lsq-vel-tol', str(lsq_vel_tol)
    ]
    print('Running:', ' '.join(cmd))
    env = os.environ.copy()
    env['PYTHONPATH'] = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    ret = subprocess.run(cmd, env=env)
    return ret.returncode == 0


def merge_and_post(outdir, centers_path, vel, post_args=None):
    # Merge per-date locations into a single merged json
    merged = os.path.join(outdir, f'locations_vel_{vel}.json')
    cmd_merge = ['python3', 'scripts/merge_locations_jsons.py', '--input-dir', outdir, '--out', merged, '--pattern', 'locations_*.json', '--force']
    subprocess.run(cmd_merge, check=True)

    # Run the post-processor
    post_out = merged.replace('.json', '.post.json')
    cmd_post = ['python3', 'scripts/post_process_locations.py', '--json', merged, '--centers', centers_path, '--out-json', post_out]
    if post_args:
        cmd_post.extend(post_args)
    subprocess.run(cmd_post, check=True)
    return merged, post_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='bench config JSON')
    parser.add_argument('--vels', required=True, help='CSV of velocities, e.g., 2.5,3.0,3.5')
    parser.add_argument('--outdir', required=True, help='Base output dir')
    parser.add_argument('--time-tol', type=float, default=30.0)
    parser.add_argument('--min-snr', type=float, default=8.0)
    parser.add_argument('--lsq-vel-min', type=float, default=1.5)
    parser.add_argument('--lsq-vel-max', type=float, default=4.5)
    parser.add_argument('--lsq-vel-tol', type=float, default=0.5)
    parser.add_argument('--post-args', default='', help='Extra post_process args')
    args = parser.parse_args()

    cfg_path = args.config
    cfg = json.load(open(cfg_path))
    plot_dir = cfg.get('plot_dir', './plots')
    centers_path = os.path.join(plot_dir, 'centers.json')
    os.makedirs(args.outdir, exist_ok=True)
    vels = [float(v.strip()) for v in args.vels.split(',') if v.strip()]

    summary = {}
    for v in vels:
        od = os.path.join(args.outdir, f'vel_{v:.2f}')
        ok = run_full_for_vel(cfg_path, v, od, time_tol=args.time_tol, min_snr=args.min_snr, lsq_vel_min=args.lsq_vel_min, lsq_vel_max=args.lsq_vel_max, lsq_vel_tol=args.lsq_vel_tol)
        if not ok:
            print('Run failed for vel', v, file=sys.stderr)
            continue
        merged, post_out = merge_and_post(od, centers_path, v, post_args=(args.post_args.split() if args.post_args else None))
        # compute counts and nearest center stats for summary
        arr = json.load(open(post_out))
        counts = {}
        lsq_dists = []
        inter_dists = []
        for e in arr:
            m = e.get('method', 'unknown')
            counts[m] = counts.get(m, 0) + 1
            if e.get('_nearest_center_km') is not None:
                if (m or '').startswith('lsq'):
                    lsq_dists.append(float(e.get('_nearest_center_km') or float('nan')))
                elif (m or '').startswith('intersection'):
                    inter_dists.append(float(e.get('_nearest_center_km') or float('nan')))
        import statistics
        def mean(a):
            return statistics.mean(a) if a else float('nan')
        def median(a):
            return statistics.median(a) if a else float('nan')
        summary[v] = {
            'n_total': len(arr),
            'counts': counts,
            'lsq_mean_center_km': mean(lsq_dists),
            'lsq_median_center_km': median(lsq_dists),
            'inter_mean_center_km': mean(inter_dists),
            'inter_median_center_km': median(inter_dists)
        }

    sf = os.path.join(args.outdir, 'vel_grid_full_summary.json')
    with open(sf, 'w') as fh:
        json.dump(summary, fh, indent=2)
    print('Wrote full summary to', sf)


if __name__ == '__main__':
    main()
