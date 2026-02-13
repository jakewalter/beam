#!/usr/bin/env python3
"""
Lightweight triangulation helper

Usage:
  python scripts/triangulate_from_detections_json.py --detections detections.json --centers centers.json --date 20200601 --outdir plots/detections_20200601

- `detections.json` should be a dict mapping date_str -> list of detection dicts with at least 'time', 'backazimuth', 'subarray_id'.
- `centers.json` should map subarray_id -> [lat, lon]

The script writes `locations_<date>.json` using `beam.core.triangulation.triangulate_two_arrays`.
"""
import argparse
import json
import os
from beam.core.triangulation import triangulate_two_arrays


def main():
    parser = argparse.ArgumentParser(description='Triangulate detections JSON into locations')
    parser.add_argument('--detections', required=True, help='Path to detections JSON (dict: date -> list[det])')
    parser.add_argument('--centers', required=True, help='JSON mapping subarray_id -> [lat, lon]')
    parser.add_argument('--date', required=True, help='YYYYMMDD date to triangulate')
    parser.add_argument('--outdir', required=True, help='Directory to write locations JSON')
    parser.add_argument('--time-tol', type=float, default=5.0, help='Time tolerance (s) to pair detections')
    parser.add_argument('--min-snr', type=float, default=0.0, help='Minimum average SNR for LSQ fallback')
    parser.add_argument('--lsq-vel-min', type=float, default=1.0, help='Minimum mean velocity (km/s) to allow LSQ')
    parser.add_argument('--lsq-vel-max', type=float, default=6.0, help='Maximum mean velocity (km/s) to allow LSQ')
    parser.add_argument('--lsq-vel-tol', type=float, default=0.5, help='Maximum allowed difference in velocities (km/s) between pair')
    parser.add_argument('--lsq-force-vel', type=float, default=None, help='Force LSQ to assume a fixed velocity (km/s) rather than using detection-provided velocities')
    parser.add_argument('--min-angle', type=float, default=15.0, help='Minimum bearing separation (deg) to attempt intersection')
    parser.add_argument('--no-lsq', action='store_true', help='Disable LSQ fallback for two-array pairs (use intersection only)')
    parser.add_argument('--mc-az-sigma', type=float, default=3.0, help='Azimuth uncertainty (deg) to use for Monte Carlo intersection')
    parser.add_argument('--mc-samples', type=int, default=200, help='Number of Monte Carlo samples for intersection uncertainty')
    args = parser.parse_args()

    dets = json.load(open(args.detections))
    centers_map = json.load(open(args.centers))

    if args.date not in dets:
        print(f"Date {args.date} not in detections JSON")
        return

    day_dets = dets[args.date]
    # build sub_map
    sub_map = {}
    for d in day_dets:
        sid = str(d.get('subarray_id', 'None'))
        if sid == 'None':
            continue
        sub_map.setdefault(sid, []).append(d)

    sub_centers = {}
    for sid, latlon in centers_map.items():
        sub_centers[sid] = tuple(latlon)

    location_estimates = []
    ids = sorted(list(sub_map.keys()))
    for i in range(len(ids)):
        for j in range(i+1, len(ids)):
            ida = ids[i]
            idb = ids[j]
            if ida not in sub_centers or idb not in sub_centers:
                continue
            res = triangulate_two_arrays(sub_map.get(ida, []), sub_map.get(idb, []), sub_centers[ida], sub_centers[idb], origin=None, time_tolerance=args.time_tol, min_angle_deg=args.min_angle, use_lsq_if_available=(not args.no_lsq), mc_az_sigma_deg=args.mc_az_sigma, mc_samples=args.mc_samples, min_snr=args.min_snr, lsq_vel_min=args.lsq_vel_min, lsq_vel_max=args.lsq_vel_max, lsq_vel_tol=args.lsq_vel_tol, lsq_force_velocity=args.lsq_force_vel)
            if res:
                for r in res:
                    r['subarrays'] = (ida, idb)
                    location_estimates.append(r)

    if len(location_estimates) == 0:
        print(f"No triangulation results for {args.date}")
    else:
        os.makedirs(args.outdir, exist_ok=True)
        outpath = os.path.join(args.outdir, f"locations_{args.date}.json")
        with open(outpath, 'w') as fh:
            json.dump(location_estimates, fh, indent=2)
        print(f"Wrote {outpath} ({len(location_estimates)} results)")

if __name__ == '__main__':
    main()
