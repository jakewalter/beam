#!/usr/bin/env python3
"""
Plot polar histograms (rose diagrams) of beamforming backazimuth detections.

Accepts either a detections JSON file (per-day detection lists) or a CSV
with 'subarray_id' and 'backazimuth' fields. Plots either a combined histogram
or per-array subplots. Optionally weight by SNR and set bin size.
"""
import argparse
import json
import os
import math
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    import pandas as pd
    PANDAS = True
except Exception:
    PANDAS = False


def read_detections_json(fp):
    with open(fp, 'r') as fh:
        data = json.load(fh)
    # If dict with dates -> lists, flatten into a list
    rows = []
    if isinstance(data, dict):
        for k, lst in data.items():
            if isinstance(lst, list):
                rows.extend(lst)
    elif isinstance(data, list):
        rows = data
    return rows


def read_csv(fp):
    if PANDAS:
        return pd.read_csv(fp)
    with open(fp, 'r') as fh:
        return list(csv.DictReader(fh))


def parse_rows(rows):
    # rows: list of dicts containing 'backazimuth' and 'subarray_id' and optionally 'snr'
    out = {}
    for r in rows:
        try:
            if isinstance(r, dict):
                baz = float(r.get('backazimuth'))
                sid = str(r.get('subarray_id')) if 'subarray_id' in r else '0'
                snr = float(r.get('snr', 0.0) or 0.0)
            else:
                # pandas Series
                baz = float(r['backazimuth'])
                sid = str(r['subarray_id'])
                snr = float(r.get('snr', 0.0) or 0.0)
        except Exception:
            continue
        if sid not in out:
            out[sid] = {'bazs': [], 'snrs': []}
        out[sid]['bazs'].append(baz % 360.0)
        out[sid]['snrs'].append(snr)
    return out


def plot_rose(ax, angles_deg, weights=None, bins=36, title=None, color='C0'):
    # angles_deg list of floats 0-360 (clockwise from North)
    # We'll plot with 0 at North and angles increasing clockwise so that
    # the polar plot directly matches the backazimuth convention.
    theta = np.radians(angles_deg)
    theta = np.mod(theta, 2 * np.pi)

    # Histogram
    counts, bin_edges = np.histogram(theta, bins=bins, weights=weights)
    widths = np.diff(bin_edges)
    ax.bar(bin_edges[:-1], counts, width=widths, align='edge', color=color, edgecolor='k', alpha=0.8)
    # Set zero to North and direction clockwise (-1) so 0deg points up
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    if title:
        ax.set_title(title)
    return ax


def circular_mean_deg(angles_deg, weights=None):
    # Return circular mean in degrees (0-360), angles in degrees
    a = np.radians(angles_deg)
    if weights is None:
        w = np.ones_like(a)
    else:
        w = np.array(weights)
    s = np.sum(w * np.sin(a))
    c = np.sum(w * np.cos(a))
    mean_angle = np.degrees(np.arctan2(s, c)) % 360.0
    return mean_angle


def main():
    parser = argparse.ArgumentParser(description='Polar histogram for beamforming backazimuths')
    parser.add_argument('--detections', help='JSON detections file produced by the pipeline')
    parser.add_argument('--csv', help='CSV with backazimuths and subarray_id')
    parser.add_argument('--out', default='plots/beam_polar.png')
    parser.add_argument('--bins', type=int, default=36)
    parser.add_argument('--per-array', action='store_true', help='Create a subplot per array')
    parser.add_argument('--snr-weight', action='store_true', help='Weight histogram bins by SNR')
    parser.add_argument('--snr-threshold', type=float, default=0.0, help='Minimum SNR to include a detection')
    parser.add_argument('--annotate-mean', action='store_true', help='Annotate mean azimuth on the polar plot')
    parser.add_argument('--top-bin-annot', type=int, default=0, help='Annotate top N bins by count/SNR (0 = none)')
    args = parser.parse_args()

    rows = None
    if args.detections:
        rows = read_detections_json(args.detections)
    elif args.csv:
        rows = read_csv(args.csv)
    else:
        parser.error('either --detections or --csv required')

    data = rows if isinstance(rows, list) else rows.to_dict(orient='records') if PANDAS else rows
    parsed = parse_rows(data)
    # Apply SNR threshold filtering
    if args.snr_threshold and args.snr_threshold > 0.0:
        for sid, v in list(parsed.items()):
            new_angles = []
            new_snrs = []
            for ang, sn in zip(v['bazs'], v['snrs']):
                if sn >= args.snr_threshold:
                    new_angles.append(ang)
                    new_snrs.append(sn)
            parsed[sid]['bazs'] = new_angles
            parsed[sid]['snrs'] = new_snrs
    # Choose layout
    if args.per_array:
        n = len(parsed)
        cols = min(4, n)
        rows_plot = math.ceil(n / cols)
        fig, axes = plt.subplots(rows_plot, cols, subplot_kw={'projection': 'polar'}, figsize=(4*cols, 3*rows_plot))
        axes = np.array(axes).reshape(-1)
        for idx, (sid, v) in enumerate(sorted(parsed.items(), key=lambda x:int(x[0]))):
            angles = v['bazs']
            weights = v['snrs'] if args.snr_weight else None
            plot_rose(axes[idx], angles, weights=weights, bins=args.bins, title=f'Array {sid}')
            if args.annotate_mean and angles:
                mean_az = circular_mean_deg(angles, weights)
                # Convert to plot theta
                theta = np.radians(mean_az)
                axes[idx].annotate('', xy=(theta, max(axes[idx].get_ylim())*0.9), xytext=(0,0),
                                   arrowprops=dict(facecolor='red', width=2, headwidth=8),
                                   )
                axes[idx].text(theta, max(axes[idx].get_ylim())*0.95, f"{mean_az:.1f}\u00B0", color='red', fontsize=8)
            if args.top_bin_annot and angles:
                # annotate the top bins by count (or weighted sum)
                counts, bin_edges = np.histogram(np.radians(angles), bins=args.bins, weights=(v['snrs'] if args.snr_weight else None))
                top_idx = np.argsort(counts)[-args.top_bin_annot:]
                for ti in top_idx:
                    ang_mid = (bin_edges[ti] + bin_edges[ti+1]) / 2.0
                    axes[idx].text(ang_mid, max(counts)*0.9, f"{counts[ti]:.1f}", fontsize=7, color='black')
        # hide extra axes
        for ax in axes[len(parsed):]:
            ax.axis('off')
    else:
        # Combined diagram
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(1, 1, 1, projection='polar')
        combined_angles = []
        combined_weights = []
        for sid, v in parsed.items():
            combined_angles.extend(v['bazs'])
            if args.snr_weight:
                combined_weights.extend(v['snrs'])
            else:
                combined_weights.extend([1.0] * len(v['bazs']))
        plot_rose(ax, combined_angles, weights=combined_weights if args.snr_weight else None, bins=args.bins, title='Combined backazimuths')
        if args.annotate_mean and combined_angles:
            mean_az = circular_mean_deg(combined_angles, combined_weights if args.snr_weight else None)
            theta = np.radians(mean_az)
            ax.annotate('', xy=(theta, max(ax.get_ylim())*0.9), xytext=(0,0), arrowprops=dict(facecolor='red', width=2, headwidth=8))
            ax.text(theta, max(ax.get_ylim())*0.95, f"{mean_az:.1f}\u00B0", color='red', fontsize=10)
        if args.top_bin_annot and combined_angles:
            counts, bin_edges = np.histogram(np.radians(combined_angles), bins=args.bins, weights=(combined_weights if args.snr_weight else None))
            top_idx = np.argsort(counts)[-args.top_bin_annot:]
            for ti in top_idx:
                ang_mid = (bin_edges[ti] + bin_edges[ti+1]) / 2.0
                ax.text(ang_mid, max(counts)*0.9, f"{counts[ti]:.1f}", fontsize=7, color='black')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print('Wrote', args.out)


if __name__ == '__main__':
    main()
