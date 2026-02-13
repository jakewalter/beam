#!/usr/bin/env python3
"""
Create summary plots for a velocity-grid sample directory.

For each velocity (looks for `locations_vel_vel_<v>.post.json`) we plot:
 - Top: lat/lon scatter colored by method
 - Bottom: histogram of nearest-center distances for LSQ vs intersection

Also writes a combined counts+mean-distance plot.
"""
import argparse
import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt


def load_post_json(fp):
    return json.load(open(fp))


def make_plots(indir, outdir=None, include_days=False):
    outdir = outdir or indir
    os.makedirs(outdir, exist_ok=True)
    # Accept either post-processed merged files or (fallback) merged files.
    files = sorted(glob.glob(os.path.join(indir, 'locations_vel_vel_*.post.json')))
    if not files:
        files = sorted(glob.glob(os.path.join(indir, 'locations_vel_vel_*.json')))
    if not files:
        raise RuntimeError('No merged JSON files found in ' + indir)

    vels = []
    datasets = {}
    for f in files:
        # parse velocity from filename
        b = os.path.basename(f)
        vstr = b.replace('locations_vel_vel_', '').replace('.post.json','')
        try:
            v = float(vstr)
        except Exception:
            v = vstr
        vels.append(v)
        datasets[v] = load_post_json(f)

    vels = sorted(vels)
    n = len(vels)

    # per-velocity panels: 2 rows x n columns
    fig, axs = plt.subplots(2, n, figsize=(4*n, 8), constrained_layout=True)
    counts = {}
    lsq_means = {}

    for i, v in enumerate(vels):
        arr = datasets[v]
        # optionally include per-day files found under a vel-specific subdirectory
        if include_days:
            day_dir = os.path.join(indir, f'vel_{v:.2f}')
            if os.path.isdir(day_dir):
                day_files = sorted(glob.glob(os.path.join(day_dir, 'locations_*.json')))
                for df in day_files:
                    try:
                        extra = json.load(open(df))
                        if isinstance(extra, list):
                            arr.extend(extra)
                    except Exception:
                        # skip corrupt files
                        continue
        # Normalize lat/lon types and filter out entries without coordinates
        pts = []
        for e in arr:
            try:
                lat = float(e.get('lat'))
                lon = float(e.get('lon'))
                pts.append((lat, lon, e))
            except Exception:
                continue
        lats = [p[0] for p in pts]
        lons = [p[1] for p in pts]
        methods = [p[2].get('method','') for p in pts]
        methods = [e.get('method','') for e in arr]
        # scatter
        ax = axs[0,i]
        colors = {'lsq_2array':'C0','lsq_multi':'C0','intersection_mc':'C1','intersection':'C1'}
        for m in sorted(set(methods)):
            idx = [j for j,mm in enumerate(methods) if mm==m]
            ax.scatter([lons[j] for j in idx], [lats[j] for j in idx], label=m, alpha=0.8, s=20, c=colors.get(m,'k'))
        ax.set_title(f'vel {v} km/s (n={len(arr)})')
        ax.set_xlabel('lon'); ax.set_ylabel('lat')
        ax.legend(fontsize='small')

        # histogram of nearest center distances
        ax2 = axs[1,i]
        lsq_d = [float(e.get('_nearest_center_km')) for _,_,e in pts if (e.get('method') or '').startswith('lsq') and e.get('_nearest_center_km') is not None]
        inter_d = [float(e.get('_nearest_center_km')) for _,_,e in pts if (e.get('method') or '').startswith('intersection') and e.get('_nearest_center_km') is not None]
        if lsq_d:
            ax2.hist(lsq_d, bins=20, alpha=0.7, label='lsq')
        if inter_d:
            ax2.hist(inter_d, bins=20, alpha=0.7, label='intersection')
        ax2.set_xlabel('nearest-center km')
        ax2.set_ylabel('count')
        ax2.legend()

        counts[v] = len(pts)
        lsq_means[v] = np.nanmean(lsq_d) if lsq_d else np.nan

    fig.savefig(os.path.join(outdir, 'vel_grid_sample_panels.png'), dpi=150)

    # combined summary plots: counts bar and lsq mean distance
    fig2, ax = plt.subplots(figsize=(8,4))
    xs = list(counts.keys())
    ys = [counts[x] for x in xs]
    ax.bar([str(x) for x in xs], ys)
    ax.set_xlabel('vel (km/s)')
    ax.set_ylabel('count')
    ax.set_title('Total events per velocity')
    fig2.tight_layout()
    fig2.savefig(os.path.join(outdir, 'vel_grid_sample_counts.png'), dpi=150)

    fig3, ax = plt.subplots(figsize=(8,4))
    ax.plot([str(x) for x in xs], [lsq_means[x] for x in xs], '-o')
    ax.set_xlabel('vel (km/s)'); ax.set_ylabel('LSQ mean nearest-center km')
    fig3.tight_layout()
    fig3.savefig(os.path.join(outdir, 'vel_grid_sample_lsq_mean_center.png'), dpi=150)

    return os.path.join(outdir, 'vel_grid_sample_panels.png')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', required=True, help='velocity sample dir (e.g., plots/vel_grid_sample)')
    parser.add_argument('--outdir', required=False, help='output directory for plots')
    parser.add_argument('--include-days', action='store_true', help='Also include per-day files under vel_<v> subdirectories')
    args = parser.parse_args()
    out = make_plots(args.indir, args.outdir, include_days=args.include_days)
    print('Wrote', out)


if __name__ == '__main__':
    main()
