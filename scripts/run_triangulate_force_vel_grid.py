#!/usr/bin/env python3
"""
Run triangulation for a small grid of LSQ forced velocities on a set of sample dates.

This is useful for sensitivity tests (e.g., vel = 2.5, 3.0, 3.5, 4.0).

For each velocity the script:
 - runs `scripts/triangulate_from_detections_json.py` for each requested date
 - merges per-date outputs into a single JSON using `merge_locations_jsons.py`
 - optionally runs `post_process_locations.py` to prefer LSQ and filter intersections
 - writes a small summary JSON with counts per method and simple stats

Usage example:
 python scripts/run_triangulate_force_vel_grid.py --config configs/bench_tri_30s_min_snr8.json --dates 20200109,20200110 --vels 2.5,3.0,3.5,4.0 --outdir plots/vel_grid_sample
"""
import argparse
import json
import os
import subprocess
import tempfile
from datetime import datetime
import math


def run_for_velocity(cfg, dates, vel, outdir, time_tol=30.0, min_snr=8.0, lsq_vel_min=2.5, lsq_vel_max=5.0):
    os.makedirs(outdir, exist_ok=True)
    dets_folder = cfg.get('plot_dir', './plots')
    logf = os.path.join(outdir, f'vel_{vel:.2f}.log')
    with open(logf, 'a') as logfh:
        logfh.write(f"Running vel {vel}\n")
        for date in dates:
            dets = os.path.join(dets_folder, f'detections_{date}.json')
            centers = os.path.join(dets_folder, 'centers.json')
            if not os.path.exists(dets):
                logfh.write(f"{date}: no detections file {dets}\n")
                continue
            cmd = [
                'python3', 'scripts/triangulate_from_detections_json.py',
                '--detections', dets,
                '--centers', centers,
                '--date', date,
                '--outdir', outdir,
                '--time-tol', str(time_tol),
                '--min-snr', str(min_snr),
                '--lsq-force-vel', str(vel),
                '--lsq-vel-min', str(lsq_vel_min),
                '--lsq-vel-max', str(lsq_vel_max)
            ]
            logfh.write('Running: ' + ' '.join(cmd) + '\n')
            try:
                env = os.environ.copy()
                env['PYTHONPATH'] = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
                subprocess.run(cmd, check=True, env=env)
                logfh.write(f"{date}: success\n")
            except subprocess.CalledProcessError as e:
                logfh.write(f"{date}: error {e}\n")


def merge_and_summarize(vel_dir, out_dir, centers_path, run_post=False, post_args=None):
    # merge per-date files
    merged = os.path.join(out_dir, f'locations_vel_{os.path.basename(vel_dir)}.json')
    cmd_merge = ['python3', 'scripts/merge_locations_jsons.py', '--input-dir', vel_dir, '--out', merged, '--pattern', 'locations_*.json', '--force']
    subprocess.run(cmd_merge, check=True)

    # optional post-process
    post_out = merged
    if run_post:
        post_out = merged.replace('.json', '.post.json')
        cmd_post = ['python3', 'scripts/post_process_locations.py', '--json', merged, '--centers', centers_path, '--out-json', post_out]
        if post_args:
            cmd_post.extend(post_args)
        subprocess.run(cmd_post, check=True)

    # summarize
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

    mean = lambda a: sum(a)/len(a) if a else float('nan')
    stats = {
        'n_total': len(arr),
        'counts': counts,
        'lsq_mean_center_km': mean(lsq_dists),
        'lsq_median_center_km': sorted(lsq_dists)[len(lsq_dists)//2] if lsq_dists else float('nan'),
        'inter_mean_center_km': mean(inter_dists),
        'inter_median_center_km': sorted(inter_dists)[len(inter_dists)//2] if inter_dists else float('nan')
    }
    return stats, post_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='JSON config used for plot_dir/dates')
    parser.add_argument('--dates', required=True, help='Comma-separated list of sample dates (YYYYMMDD)')
    parser.add_argument('--vels', required=True, help='Comma-separated list of velocities (km/s), e.g., 2.5,3.0,3.5')
    parser.add_argument('--outdir', required=True, help='Directory to write velocity dirs and summaries')
    parser.add_argument('--time-tol', type=float, default=30.0)
    parser.add_argument('--min-snr', type=float, default=8.0)
    parser.add_argument('--lsq-vel-min', type=float, default=2.5)
    parser.add_argument('--lsq-vel-max', type=float, default=5.0)
    parser.add_argument('--post-process', action='store_true', help='Run post_process_locations on merged output')
    parser.add_argument('--post-args', default='', help='Additional args to post_process_locations (space-separated)')
    args = parser.parse_args()

    cfg = json.load(open(args.config))
    centers_path = os.path.join(cfg.get('plot_dir', './plots'), 'centers.json')
    dates = [d.strip() for d in args.dates.split(',') if d.strip()]
    vels = [float(v.strip()) for v in args.vels.split(',') if v.strip()]
    os.makedirs(args.outdir, exist_ok=True)

    summary = {}
    for v in vels:
        vd = os.path.join(args.outdir, f'vel_{v:.2f}')
        run_for_velocity(cfg, dates, v, vd, time_tol=args.time_tol, min_snr=args.min_snr, lsq_vel_min=args.lsq_vel_min, lsq_vel_max=args.lsq_vel_max)
        stats, outjson = merge_and_summarize(vd, args.outdir, centers_path, run_post=args.post_process, post_args=(args.post_args.split() if args.post_args else None))
        summary[v] = stats

    # write summary
    sfile = os.path.join(args.outdir, 'vel_grid_summary.json')
    with open(sfile, 'w') as fh:
        json.dump(summary, fh, indent=2)
    print('Wrote summary to', sfile)


if __name__ == '__main__':
    main()
