#!/usr/bin/env python3
"""
Plot angular kernel-density (smoothed histograms) of beam backazimuths split
by velocity bands and per-array (subarray_id). Outputs polar annulus plots
where each radial band corresponds to a velocity bin and the radial thickness
represents density (smoothed counts, optionally weighted by SNR).

Usage:
  python scripts/plot_beam_kde.py --detections detections.json --out plots/beam_kde.png

Options:
  --bins N           : number of angular bins (default 360)
  --nbands K         : number of velocity bands (default 4)
  --bands "a-b,c-d"  : explicit band edges
  --sigma-bins S     : Gaussian smoothing sigma in bins (default 3)
  --snr-weight       : weight counts by SNR
  --per-array        : create separate subplot for each array (default True)
"""
import argparse
import json
import math
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm


def read_detections_json(fp):
    with open(fp, 'r') as fh:
        data = json.load(fh)
    rows = []
    if isinstance(data, dict):
        for k, lst in data.items():
            if isinstance(lst, list):
                rows.extend(lst)
    elif isinstance(data, list):
        rows = data
    return rows


def parse_bands_arg(bands_str, nbands, velocities):
    if bands_str:
        parts = [p.strip() for p in bands_str.split(',') if p.strip()]
        bands = []
        for p in parts:
            lo, hi = [float(x) for x in p.split('-')]
            bands.append((lo, hi))
        return bands
    # auto compute nbands between min and max velocity
    vmin = float(np.min(velocities))
    vmax = float(np.max(velocities))
    edges = np.linspace(vmin, vmax, nbands+1)
    bands = [(float(edges[i]), float(edges[i+1])) for i in range(len(edges)-1)]
    return bands


def gaussian_kernel(sigma_bins, radius=6):
    # return 1D gaussian kernel
    L = max(int(radius * sigma_bins), 3)
    xs = np.arange(-L, L+1)
    kern = np.exp(-0.5 * (xs / float(sigma_bins))**2)
    kern /= kern.sum()
    return kern


def angular_smooth_hist(angles_deg, weights, bins, sigma_bins):
    # angles_deg in [0,360)
    hist, edges = np.histogram(angles_deg, bins=bins, range=(0.0, 360.0), weights=weights)
    kern = gaussian_kernel(sigma_bins)
    # circular convolution, pad and wrap
    padded = np.concatenate([hist, hist, hist])
    conv = np.convolve(padded, kern, mode='same')
    mid = len(hist)
    out = conv[mid:mid+len(hist)]
    # normalize to max 1
    if out.max() > 0:
        out = out / out.max()
    theta_centers = (edges[:-1] + edges[1:]) / 2.0
    return theta_centers, out


def plot_kde_per_array(rows, out, bins=360, nbands=4, bands=None, sigma_bins=3, snr_weight=False):
    # group by array
    arr_map = {}
    velocities = []
    for r in rows:
        try:
            sid = str(int(r.get('subarray_id', 0)))
            baz = float(r.get('backazimuth')) % 360.0
            vel = float(r.get('velocity', np.nan))
            snr = float(r.get('snr', 0.0) or 0.0)
        except Exception:
            continue
        arr_map.setdefault(sid, []).append({'baz': baz, 'vel': vel, 'snr': snr})
        velocities.append(vel)

    if not bands:
        bands = parse_bands_arg(None, nbands, np.array(velocities))

    n_arrays = len(arr_map)
    cols = min(2, n_arrays)
    rows_plot = int(math.ceil(n_arrays / cols)) if n_arrays > 1 else 1
    fig, axes = plt.subplots(rows_plot, cols, subplot_kw={'projection': 'polar'}, figsize=(5*cols, 4*rows_plot))
    axes = np.array(axes).reshape(-1)
    cmap = cm.get_cmap('plasma', len(bands))

    for idx, (sid, items) in enumerate(sorted(arr_map.items(), key=lambda x: int(x[0]))):
        ax = axes[idx]
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        # For each band compute smoothed density
        for bidx, (lo, hi) in enumerate(bands):
            angles = [it['baz'] for it in items if it['vel'] >= lo and it['vel'] < hi]
            if not angles:
                dens = np.zeros(bins)
                theta = np.linspace(0, 360, bins, endpoint=False)
            else:
                weights = [it['snr'] if snr_weight else 1.0 for it in items if it['vel'] >= lo and it['vel'] < hi]
                theta, dens = angular_smooth_hist(angles, weights, bins, sigma_bins)
            # radial placement for band
            base = bidx
            scale = 0.8
            r_outer = base + dens * scale
            r_base = np.ones_like(r_outer) * base
            theta_rad = np.radians(theta)
            ax.fill_between(theta_rad, r_base, r_outer, color=cmap(bidx), alpha=0.7)
        ax.set_title(f'Array {sid}')
        # legend boxes for bands
        legend_patches = [plt.Rectangle((0,0),1,1, facecolor=cmap(i), alpha=0.7) for i in range(len(bands))]
        band_labels = [f'{lo:.2f}-{hi:.2f} km/s' for (lo,hi) in bands]
        ax.legend(legend_patches, band_labels, loc='lower left', bbox_to_anchor=(1.05, 0.1), fontsize=8)

    # hide unused axes
    for i in range(len(arr_map), len(axes)):
        axes[i].axis('off')

    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print('Wrote', out)


def main():
    parser = argparse.ArgumentParser(description='Plot KDE-style polar backazimuths by velocity band')
    parser.add_argument('--detections', required=True)
    parser.add_argument('--out', default='plots/beam_kde.png')
    parser.add_argument('--bins', type=int, default=360)
    parser.add_argument('--nbands', type=int, default=4)
    parser.add_argument('--bands', type=str, default=None, help='Comma-separated bands like "0-1,1-2,2-3"')
    parser.add_argument('--sigma-bins', type=float, default=3.0)
    parser.add_argument('--snr-weight', action='store_true')
    args = parser.parse_args()

    rows = read_detections_json(args.detections)
    bands = None
    if args.bands:
        parts = [p.strip() for p in args.bands.split(',') if p.strip()]
        bands = []
        for p in parts:
            lo, hi = [float(x) for x in p.split('-')]
            bands.append((lo, hi))

    plot_kde_per_array(rows, args.out, bins=args.bins, nbands=args.nbands, bands=bands, sigma_bins=args.sigma_bins, snr_weight=args.snr_weight)


if __name__ == '__main__':
    main()
