#!/usr/bin/env python3
"""
Run triangulation across date range using a forced LSQ velocity, writing outputs into an outdir.

Usage:
  python scripts/run_triangulate_force_vel_full.py --config configs/bench_tri_30s_min_snr8.json --force-vel 3.5 --outdir plots/test_lsq_force_full_3_5

This script iterates across dates in the config and invokes
scripts/triangulate_from_detections_json.py for each date, reading the per-day
detections file under the original `plot_dir` in the config and writing
locations into the provided outdir.
"""
import argparse
import json
import os
import subprocess
from datetime import datetime, timedelta


def dates_in_range(start, end):
    d0 = datetime.strptime(start, '%Y%m%d')
    d1 = datetime.strptime(end, '%Y%m%d')
    cur = d0
    out = []
    while cur <= d1:
        out.append(cur.strftime('%Y%m%d'))
        cur += timedelta(days=1)
    return out


def main():
    parser = argparse.ArgumentParser(description='Batch-run two-array triangulate with forced LSQ velocity')
    parser.add_argument('--config', required=True, help='Path to JSON config (used for date range and detection folder)')
    parser.add_argument('--force-vel', type=float, required=True, help='lsq_force_vel to pass to triangulate')
    parser.add_argument('--outdir', required=True, help='Directory to write per-date outputs')
    parser.add_argument('--time-tol', type=float, default=30.0)
    parser.add_argument('--min-snr', type=float, default=8.0)
    parser.add_argument('--lsq-vel-min', type=float, default=3.0)
    parser.add_argument('--lsq-vel-max', type=float, default=5.0)
    parser.add_argument('--lsq-vel-tol', type=float, default=0.5)
    args = parser.parse_args()

    cfg = json.load(open(args.config))
    data_dir = cfg['data_dir']
    # original plot dir where per-day detection files live
    dets_folder = cfg.get('plot_dir', './plots')
    start = cfg['start']
    end = cfg['end']
    dates = dates_in_range(start, end)
    os.makedirs(args.outdir, exist_ok=True)
    logf = os.path.join(os.path.dirname(args.outdir), os.path.basename(args.outdir) + '.log')
    with open(logf, 'a') as logfh:
        logfh.write(f"Starting forced LSQ run force_vel={args.force_vel}\n")
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
                '--outdir', args.outdir,
                '--time-tol', str(args.time_tol),
                '--min-snr', str(args.min_snr),
                '--lsq-force-vel', str(args.force_vel),
                '--lsq-vel-min', str(args.lsq_vel_min),
                '--lsq-vel-max', str(args.lsq_vel_max),
                '--lsq-vel-tol', str(args.lsq_vel_tol)
            ]
            logfh.write('Running: ' + ' '.join(cmd) + '\n')
            try:
                env = os.environ.copy()
                env['PYTHONPATH'] = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
                subprocess.run(cmd, check=True, env=env)
                logfh.write(f"{date}: success\n")
            except subprocess.CalledProcessError as e:
                logfh.write(f"{date}: error {e}\n")
    print('Wrote log to', logf)


if __name__ == '__main__':
    main()
