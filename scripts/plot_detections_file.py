#!/usr/bin/env python3
"""Simple plotting utility for detections saved in detections.txt

Usage: python scripts/plot_detections_file.py detections.txt --outdir plots

Creates several diagnostic plots saved to the outdir:
 - timeline.png : event times (scatter) color-coded by SNR
 - snr_hist.png : histogram of SNR values
 - azimuth_polar.png : polar histogram of backazimuth
 - velocity_hist.png : histogram of velocity values
 - duration_hist.png : histogram of event durations
"""

import os
import sys
from datetime import datetime
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_detections_file(path):
    times = []
    velocities = []
    azs = []
    snrs = []
    durs = []

    with open(path, 'r') as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 5:
                continue
            tstr, vel, az, snr, dur = parts[:5]

            # parse time like 2020-01-15T00:24:03.875000Z
            try:
                if tstr.endswith('Z'):
                    tstr = tstr[:-1]
                t = datetime.fromisoformat(tstr)
            except Exception:
                try:
                    t = datetime.strptime(tstr, '%Y-%m-%dT%H:%M:%S.%f')
                except Exception:
                    continue

            try:
                velocities.append(float(vel))
                azs.append(float(az))
                snrs.append(float(snr))
                durs.append(float(dur))
                times.append(t)
            except Exception:
                continue

    return {
        'times': np.array(times),
        'velocities': np.array(velocities),
        'az': np.array(azs),
        'snr': np.array(snrs),
        'dur': np.array(durs)
    }


def plot_all(stats, outdir):
    os.makedirs(outdir, exist_ok=True)

    times = stats['times']
    vel = stats['velocities']
    az = stats['az']
    snr = stats['snr']
    dur = stats['dur']

    # Timeline scatter
    if len(times) > 0:
        fig, ax = plt.subplots(figsize=(12, 3))
        xs = matplotlib.dates.date2num(times)
        sc = ax.scatter(xs, snr, c=snr, cmap='viridis', s=25, edgecolor='k')
        ax.set_xlabel('Time')
        ax.set_ylabel('SNR')
        ax.xaxis_date()
        fig.autofmt_xdate()
        plt.colorbar(sc, ax=ax, label='SNR')
        fig.tight_layout()
        f = os.path.join(outdir, 'timeline.png')
        fig.savefig(f, dpi=150)
        plt.close(fig)

    # SNR histogram
    if len(snr) > 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(snr, bins=40, color='C0', alpha=0.8)
        ax.set_xlabel('SNR')
        ax.set_ylabel('Count')
        ax.grid(True)
        fig.tight_layout()
        f = os.path.join(outdir, 'snr_hist.png')
        fig.savefig(f, dpi=150)
        plt.close(fig)

    # azimuth polar: heatmap-style (counts or SNR-weighted) + scatter overlay
    if len(az) > 0:
        # allow flexible binning
        n_bins = 36
        theta = np.radians(az)

        # compute counts and s/n-weighted sums
        counts, edges = np.histogram(theta, bins=n_bins, range=(0, 2*np.pi))
        snr_sums, _ = np.histogram(theta, bins=edges, weights=snr)
        # avoid division by zero
        mean_snr = snr_sums / np.where(counts == 0, 1.0, counts)

        # build a radial heatmap plot: we map each bin to an angular patch
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, polar=True)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)

        # radial coordinates for the bar heights
        radii = counts.astype(float)
        width = (2 * np.pi) / n_bins
        bin_centers = edges[:-1] + width / 2.0

        cmap = plt.get_cmap('viridis')

        # color by mean SNR (use minimum zero for bins with no detections)
        colors = mean_snr.copy()
        colors[counts == 0] = np.nan
        # normalize to range for color mapping
        cmin = np.nanmin(colors) if np.isfinite(np.nanmin(colors)) else 0.0
        cmax = np.nanmax(colors) if np.isfinite(np.nanmax(colors)) else 1.0

        # map colors (use masked array so empty bins appear faint)
        norm = plt.Normalize(vmin=cmin, vmax=cmax)
        mapped = cmap(norm(colors))

        bars = ax.bar(bin_centers, radii, width=width, bottom=0.0, color=mapped, edgecolor='k', alpha=0.9)

        # scatter overlay for individual detections (alpha scaled by snr)
        scatter_r = np.ones_like(theta) * (0.05 * max(radii.max(), 1.0))
        snr_norm = (snr - snr.min()) / max(1e-6, snr.max() - snr.min())
        ax.scatter(theta, scatter_r, c=snr, cmap='inferno', s=30 + 50 * snr_norm, alpha=0.8, edgecolor='k')

        # add colorbar for mean-SNR coloring
        mappable = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
        mappable.set_array(colors)
        cbar = fig.colorbar(mappable, ax=ax, pad=0.1, orientation='vertical')
        cbar.set_label('Mean SNR (per azimuth bin)')

        ax.set_title('Azimuthal detections (bar height = count; color = mean SNR)')
        fig.tight_layout()
        f = os.path.join(outdir, 'azimuth_polar_heatmap.png')
        fig.savefig(f, dpi=150)
        plt.close(fig)

    # velocity histogram
    if len(vel) > 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(vel, bins=np.arange(min(vel) - 0.5, max(vel) + 0.5, 0.25), color='C2', alpha=0.8)
        ax.set_xlabel('Velocity (km/s)')
        ax.set_ylabel('Count')
        ax.grid(True)
        fig.tight_layout()
        f = os.path.join(outdir, 'velocity_hist.png')
        fig.savefig(f, dpi=150)
        plt.close(fig)

    # duration histogram
    if len(dur) > 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(dur, bins=40, color='C3', alpha=0.8)
        ax.set_xlabel('Duration (s)')
        ax.set_ylabel('Count')
        ax.grid(True)
        fig.tight_layout()
        f = os.path.join(outdir, 'duration_hist.png')
        fig.savefig(f, dpi=150)
        plt.close(fig)

    return True


def main():
    parser = argparse.ArgumentParser(description='Plot detections file')
    parser.add_argument('detections_file')
    parser.add_argument('--outdir', default='plots', help='Output directory for images')
    args = parser.parse_args()

    stats = parse_detections_file(args.detections_file)
    plot_all(stats, args.outdir)
    print('Wrote plots to', args.outdir)


if __name__ == '__main__':
    main()
