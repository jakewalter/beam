#!/usr/bin/env python3
"""
Compare merged post-processed files to raw per-day files and report events that were removed by post-processing.

Writes a CSV with removed event times, lat/lon, method, nearest_center_km and velocity bucket.
"""
import argparse
import glob
import json
import os
import csv


def load_post(fp):
    return set((e.get('time'), float(e.get('lat', 'nan')), float(e.get('lon','nan'))) for e in json.load(open(fp)))


def load_perday(dirpath):
    s=set()
    for f in sorted(glob.glob(os.path.join(dirpath, 'locations_*.json'))):
        try:
            arr=json.load(open(f))
        except Exception:
            continue
        for e in arr:
            s.add((e.get('time'), float(e.get('lat','nan')), float(e.get('lon','nan'))))
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', required=True, help='e.g., plots/vel_grid_sample')
    parser.add_argument('--out', required=False, help='CSV output path')
    args = parser.parse_args()

    out = args.out or os.path.join(args.indir, 'plots', 'dropped_events.csv')
    rows=[]
    for postfp in sorted(glob.glob(os.path.join(args.indir, 'locations_vel_vel_*.post.json'))):
        vstr=os.path.basename(postfp).replace('locations_vel_vel_','').replace('.post.json','')
        daydir=os.path.join(args.indir, f'vel_{float(vstr):.2f}')
        post_set=load_post(postfp)
        per_set=load_perday(daydir) if os.path.isdir(daydir) else set()
        dropped = per_set - post_set
        # fetch details of dropped events from per-day files
        details={}
        for f in sorted(glob.glob(os.path.join(daydir,'locations_*.json'))):
            try:
                arr=json.load(open(f))
            except Exception:
                continue
            for e in arr:
                key=(e.get('time'), float(e.get('lat','nan')), float(e.get('lon','nan')))
                if key in dropped:
                    details[key]=e
        for k,e in details.items():
            rows.append({'vel':vstr,'time':e.get('time'),'lat':e.get('lat'),'lon':e.get('lon'),'method':e.get('method'),'nearest_center_km':e.get('_nearest_center_km')})

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out,'w',newline='') as fh:
        w=csv.DictWriter(fh, fieldnames=['vel','time','lat','lon','method','nearest_center_km'])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print('Wrote', out)


if __name__=='__main__':
    main()
