#!/usr/bin/env python3
"""
Pipeline wrapper for running BEAM beamforming across a date range and running follow-up algorithms.

Usage:
  PYTHONPATH=. python3 scripts/run_pipeline.py --config configs/template_config.json

This script loads a JSON configuration file and then: 
  - iterates across days in the date range (inclusive)
  - runs `beam_driver.py` per day (or per range) to write per-subarray detections and per-day `detections.json`
  - writes `centers.json` based on config subarray groups (if not already written)
  - runs `triangulate_from_detections_json.py` for each day
  - runs `cluster_locations.py` to cluster pairwise intersections into events and write a CSV summary

CLI Flags:
  --config CONFIG    Path to JSON config (required)
  --pythpath PYTHONPATH (optional) default '.'
  --skip-beam        Skip the beamforming stage
  --skip-triangulate Skip triangulation
  --skip-cluster     Skip clustering
  --force-centers    Force recompute centers.json from the config and inventory

"""
import argparse
import json
import subprocess
import os
import sys
from datetime import datetime, timedelta

# Ensure the package dir is on the import path so we can import beam.*
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def dates_in_range(start, end):
    d0 = datetime.strptime(start, '%Y%m%d')
    d1 = datetime.strptime(end, '%Y%m%d')
    cur = d0
    out = []
    while cur <= d1:
        out.append(cur.strftime('%Y%m%d'))
        cur += timedelta(days=1)
    return out


def write_centers_from_config(config, inventory_folder, outdir, pattern='*.xml', tag=None, name=None):
    # This implementation mirrors the quick script used earlier; we compute
    # per-subarray mean lat/lon and write centers.json
    from beam.io.inventory import load_inventory, get_station_coords_dict
    inv = load_inventory(inventory_folder, pattern=pattern, tag=tag, name=name)
    coords = get_station_coords_dict(inv)
    centers = {}
    for i, group in enumerate(config.get('subarrays', [])):
        lat_vals = [coords[s][0] for s in group if s in coords]
        lon_vals = [coords[s][1] for s in group if s in coords]
        if len(lat_vals) > 0:
            centers[str(i)] = [float(sum(lat_vals) / len(lat_vals)), float(sum(lon_vals) / len(lon_vals))]
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, 'centers.json'), 'w') as fh:
        json.dump(centers, fh, indent=2)
    print('Wrote centers.json with keys:', list(centers.keys()))
    return os.path.join(outdir, 'centers.json')


def main():
    parser = argparse.ArgumentParser(description='Run the BEAM pipeline and follow-up triangulation/clustering')
    parser.add_argument('--config', required=True, help='Path to JSON config')
    parser.add_argument('--pythonpath', default='.', help='Python path prefix for beam_driver and scripts')
    parser.add_argument('--skip-beam', action='store_true', help='Skip beamforming stage')
    parser.add_argument('--skip-triangulate', action='store_true', help='Skip triangulation stage')
    parser.add_argument('--skip-cluster', action='store_true', help='Skip clustering stage')
    parser.add_argument('--force-centers', action='store_true', help='Force recompute centers.json')
    parser.add_argument('--dry-run', action='store_true', help='Print commands that will be executed, but do not run them')
    parser.add_argument('--inventory-pattern', default=None, help='Optional inventory glob pattern or tag to help find inventory files')
    parser.add_argument('--inventory-tag', default=None, help='Optional substring to filter inventory filenames when discovering in folder')
    parser.add_argument('--inventory-name', default=None, help='Optional exact inventory filename to load (basename)')
    args = parser.parse_args()

    cfg = json.load(open(args.config))
    data_dir = cfg['data_dir']
    plot_dir = cfg.get('plot_dir', './plots')
    config_start = cfg['start']
    config_end = cfg['end']

    # Safety clamp: triangulated_min_arrays should not exceed the number of subarrays
    if 'triangulated_min_arrays' in cfg:
        subarr_count = len(cfg.get('subarrays', []))
        if subarr_count > 0 and cfg['triangulated_min_arrays'] > subarr_count:
            print(f"Warning: config 'triangulated_min_arrays' ({cfg['triangulated_min_arrays']}) > number of subarrays ({subarr_count}). Clamping to {subarr_count}.")
            cfg['triangulated_min_arrays'] = subarr_count

    dates = dates_in_range(config_start, config_end)

    # ensure centers.json exists
    centers_path = os.path.join(plot_dir, 'centers.json')
    if not os.path.exists(centers_path) or args.force_centers:
        print('Writing centers.json from config subarrays and inventory')
        centers_path = write_centers_from_config(cfg, data_dir, plot_dir, pattern=(cfg.get('inventory_pattern') or args.inventory_pattern or '*.xml'), tag=(cfg.get('inventory_tag') or args.inventory_tag), name=(cfg.get('inventory_name') or args.inventory_name))

    # Run beam driver per day
    for date_str in dates:
        if not args.skip_beam:
            cmd = [
                'python3', 'beam_driver.py',
                '--mode', cfg.get('mode', 'traditional'),
                '--data-dir', data_dir,
                '--start', date_str, '--end', date_str,
            ]
            if cfg.get('gpu', False):
                cmd.extend(['--gpu'])
            if cfg.get('surface_wave', False):
                cmd.extend(['--surface-wave'])
            if 'gpu_safety_factor' in cfg:
                cmd.extend(['--gpu-safety-factor', str(cfg['gpu_safety_factor'])])
            # override and pass plotting
            if cfg.get('plot', False):
                cmd.extend(['--plot', '--plot-dir', plot_dir])
            # constructive pass of the FK/beam options
            for k in ['vel_min', 'vel_max', 'vel_step', 'az_step', 'fk_max_per_subarray', 'fk_min_snr', 'min_snr_output']:
                if k in cfg:
                    key = k.replace('_', '-')
                    cmd.extend([f'--{key}', str(cfg[k])])
            # subarrays -> pass inline as semicolon separated groups
            if 'subarrays' in cfg:
                inline = ';'.join([','.join(g) for g in cfg['subarrays']])
                cmd.extend(['--subarrays', inline])
            # pass inventory file or folder if present in config
            if 'inventory' in cfg and cfg.get('inventory'):
                cmd.extend(['--inventory', str(cfg.get('inventory'))])
            # pass inventory pattern and tag if present in config
            # prefer CLI args over config, then config, then nothing
            inv_pattern = args.inventory_pattern or cfg.get('inventory_pattern')
            inv_tag = args.inventory_tag or cfg.get('inventory_tag')
            inv_name = args.inventory_name or cfg.get('inventory_name')
            if inv_pattern:
                cmd.extend(['--inventory-pattern', str(inv_pattern)])
            if inv_tag:
                cmd.extend(['--inventory-tag', str(inv_tag)])
            if inv_name:
                cmd.extend(['--inventory-name', str(inv_name)])
            # Set output path for detections.txt to plot_dir with date
            detections_txt_path = os.path.join(plot_dir, f'detections_{date_str}.txt')
            cmd.extend(['--output', detections_txt_path])
            # execute
            print('Executing:', ' '.join(cmd))
            if args.dry_run:
                print('Dry run - not executing beam command')
                continue
            env = os.environ.copy()
            # set python path to repo root so `beam` package loads reliably
            env['PYTHONPATH'] = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            # forward gpu safety factor from config into the subprocess env so spawned
            # workers and imports read the same value for BEAM_GPU_SAFETY_FACTOR
            if 'gpu_safety_factor' in cfg:
                env['BEAM_GPU_SAFETY_FACTOR'] = str(cfg['gpu_safety_factor'])
            subprocess.run(cmd, shell=False, check=True, env=env)

    # Triangulate per date. Prefer per-day detection files if present,
    # otherwise fall back to the aggregated detections.json
    if not args.skip_triangulate:
        for date_str in dates:
            outdir = plot_dir
            # prefer per-day file
            dets_per_day = os.path.join(plot_dir, f'detections_{date_str}.json')
            dets_agg = os.path.join(plot_dir, 'detections.json')
            dets_to_use = dets_per_day if os.path.exists(dets_per_day) else dets_agg
            cmd = [
                'python3', 'scripts/triangulate_from_detections_json.py',
                '--detections', dets_to_use,
                '--centers', centers_path,
                '--date', date_str,
                '--outdir', outdir,
                '--time-tol', str(cfg.get('time_tol', 20.0))
            ]
            # LSQ gating options (optional in config)
            if 'min_snr' in cfg:
                cmd.extend(['--min-snr', str(cfg.get('min_snr'))])
            if 'lsq_vel_min' in cfg:
                cmd.extend(['--lsq-vel-min', str(cfg.get('lsq_vel_min'))])
            if 'lsq_vel_max' in cfg:
                cmd.extend(['--lsq-vel-max', str(cfg.get('lsq_vel_max'))])
            if 'lsq_vel_tol' in cfg:
                cmd.extend(['--lsq-vel-tol', str(cfg.get('lsq_vel_tol'))])
            if 'lsq_force_vel' in cfg and cfg.get('lsq_force_vel') is not None:
                cmd.extend(['--lsq-force-vel', str(cfg.get('lsq_force_vel'))])
            print('Executing:', ' '.join(cmd))
            if args.dry_run:
                print('Dry run - not executing triangulate command')
                continue
            env = os.environ.copy()
            # set python path to repo root so `beam` package loads reliably
            env['PYTHONPATH'] = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            if 'gpu_safety_factor' in cfg:
                env['BEAM_GPU_SAFETY_FACTOR'] = str(cfg['gpu_safety_factor'])
            subprocess.run(cmd, shell=False, check=True, env=env)

    # Cluster and summarize per date
    if not args.skip_cluster:
        for date_str in dates:
            locations_file = os.path.join(plot_dir, f'locations_{date_str}.json')
            if not os.path.exists(locations_file):
                print('No locations file for', date_str, '-- skipping cluster')
                continue
            csv_out = os.path.join(plot_dir, f'locations_summary_{date_str}.csv')
            cmd = [
                'python3', 'scripts/cluster_locations.py',
                '--locations', locations_file,
                '--out', csv_out,
                '--cluster-km', str(cfg.get('cluster_radius_km', 20.0)),
                '--min-members', str(cfg.get('cluster_min_members', 1))
            ]
            # optional: pass triangulation thresholds from config
            if 'triangulated_min_arrays' in cfg:
                cmd.extend(['--triangulated-min-arrays', str(cfg.get('triangulated_min_arrays'))])
            if 'strict_triangulated' in cfg and cfg.get('strict_triangulated'):
                cmd.extend(['--strict-triangulated'])
            print('Executing:', ' '.join(cmd))
            if args.dry_run:
                print('Dry run - not executing cluster command')
                continue
            env = os.environ.copy()
            env['PYTHONPATH'] = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            if 'gpu_safety_factor' in cfg:
                env['BEAM_GPU_SAFETY_FACTOR'] = str(cfg['gpu_safety_factor'])
            subprocess.run(cmd, shell=False, check=True, env=env)

    print('Pipeline completed')


if __name__ == '__main__':
    main()
