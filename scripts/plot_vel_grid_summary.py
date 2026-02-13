#!/usr/bin/env python3
"""
Plot summary results from a velocity-grid full run summary file (JSON produced by run_triangulate_force_vel_full_grid.py).

Produces:
 - Bar chart: total events & LSQ counts per velocity
 - Boxplots/histograms: nearest-center distances for LSQ and intersection per velocity

Outputs PNGs into the same folder as the summary JSON.
"""
import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', required=True, help='vel_grid_full_summary.json')
    parser.add_argument('--outdir', required=False, help='Output dir (default: same dir as summary)')
    args = parser.parse_args()

    s = json.load(open(args.summary))
    outdir = args.outdir or os.path.dirname(args.summary) or '.'
    os.makedirs(outdir, exist_ok=True)

    vels = sorted([float(k) for k in s.keys()])
    totals = [s[str(v)]['n_total'] for v in vels]
    lsq_counts = [s[str(v)]['counts'].get('lsq_2array', 0) + s[str(v)]['counts'].get('lsq_multi', 0) + s[str(v)]['counts'].get('lsq_3array',0) for v in vels]

    # bar chart
    x = np.arange(len(vels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(x - width/2, totals, width, label='total')
    ax.bar(x + width/2, lsq_counts, width, label='lsq')
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in vels])
    ax.set_xlabel('LSQ forced velocity (km/s)')
    ax.set_ylabel('Count')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'vel_grid_counts.png'), dpi=150)

    # nearest-center stats scatter/box via summary medians
    lsq_means = [s[str(v)]['lsq_mean_center_km'] for v in vels]
    inter_means = [s[str(v)]['inter_mean_center_km'] for v in vels]
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(vels, lsq_means, '-o', label='LSQ mean nearest center (km)')
    ax.plot(vels, inter_means, '-o', label='Intersection mean nearest center (km)')
    ax.set_xlabel('LSQ forced velocity (km/s)')
    ax.set_ylabel('Mean nearest-center distance (km)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'vel_grid_center_means.png'), dpi=150)

    print('Wrote plots to', outdir)


if __name__ == '__main__':
    main()
