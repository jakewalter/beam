#!/usr/bin/env python3
"""Create a 2x2 comparison PDF of LSQ-only combined plots across velocities.

Usage:
  python3 scripts/plot_lsq_comparison_panel.py --base plots/vel_grid_full_relaxed --vels 2.5,3.0,3.5,4.0 --out plots/vel_grid_full_relaxed/suite/lsq_comparison.pdf --size-by mean_snr --color-by azimuth_variance --size-range 6,40
"""
import argparse, json, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    import packaging
    import importlib
    if not hasattr(packaging, 'version'):
        packaging.version = importlib.import_module('packaging.version')
except Exception:
    pass
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    CARTOPY = True
except Exception:
    CARTOPY = False


def compute_field(e, field):
    if field == 'mean_snr':
        return float(e.get('snr') or 0.0)
    if field in ('azimuth_variance', 'azi_var'):
        bazs = e.get('backazimuths') or []
        try:
            vals = [float(b) for b in bazs]
            if not vals:
                return 0.0
            a = np.deg2rad(np.array(vals))
            R = np.sqrt((np.mean(np.cos(a)))**2 + (np.mean(np.sin(a)))**2)
            return float(1.0 - R)
        except Exception:
            return 0.0
    return float(e.get(field) or 0.0)


def _compute_sizes(vals, size_range=(6,40), clip_percentiles=(1,99)):
    a = np.array([np.nan if v is None else float(v) for v in vals], dtype=float)
    lo, hi = np.nanpercentile(a, clip_percentiles)
    a_clipped = np.clip(a, lo, hi)
    sval = (a_clipped - lo) / (hi - lo) if hi > lo else np.zeros_like(a_clipped)
    smin, smax = size_range
    sizes = smin + sval * (smax - smin)
    sizes[np.isnan(sizes)] = smin
    return sizes


def load_lsq_points(fp):
    arr = json.load(open(fp))
    pts = [e for e in arr if (e.get('method') or '').startswith('lsq')]
    return pts


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', required=True)
    p.add_argument('--vels', default='2.5,3.0,3.5,4.0')
    p.add_argument('--out', required=True)
    p.add_argument('--size-by', default='mean_snr')
    p.add_argument('--color-by', default='azimuth_variance')
    p.add_argument('--size-range', default='6,40')
    args = p.parse_args()

    base = Path(args.base)
    vels = [float(x) for x in args.vels.split(',') if x.strip()]
    size_min, size_max = [float(x) for x in args.size_range.split(',')]

    # Collect data per velocity
    data = {}
    all_colors = []
    all_sizes = []
    all_lons = []
    all_lats = []
    for v in vels:
        fp = base / f'locations_vel_{v:.2f}.post.json'
        if not fp.exists():
            print('Missing', fp)
            data[v] = []
            continue
        pts = load_lsq_points(str(fp))
        rows = []
        for e in pts:
            if e.get('lat') is None or e.get('lon') is None:
                continue
            lat = float(e['lat']); lon = float(e['lon'])
            cval = compute_field(e, args.color_by)
            sval = compute_field(e, args.size_by)
            rows.append({'lat':lat, 'lon':lon, 'c':cval, 'sval':sval})
            all_colors.append(cval); all_sizes.append(sval)
            all_lons.append(lon); all_lats.append(lat)
        data[v] = rows

    if not any(len(data[v])>0 for v in vels):
        raise SystemExit('No LSQ points found for any velocity')

    # Common color scale (vmin/vmax as pctiles)
    vmin, vmax = np.nanpercentile(all_colors, [2,98]) if all_colors else (0,1)
    # Prepare sizes using combined percentile clipping
    combined_sizes = np.array(all_sizes)
    lo, hi = np.nanpercentile(combined_sizes, [1,99]) if len(combined_sizes)>0 else (0,1)

    # Prepare figure: 2x2
    fig, axs = plt.subplots(2,2, figsize=(10,10), subplot_kw={'projection': ccrs.SouthPolarStereo() if CARTOPY and np.mean(all_lats)<-60 else None})
    axs = axs.flatten()

    for ax, v in zip(axs, vels):
        pts = data[v]
        if not pts:
            ax.set_title(f'v={v:.2f} (no points)')
            continue
        lons = np.array([p['lon'] for p in pts])
        lats = np.array([p['lat'] for p in pts])
        cvals = np.array([p['c'] for p in pts])
        sval = np.array([p['sval'] for p in pts])
        # clip sizes
        svalc = np.clip(sval, lo, hi)
        sizes = size_min + (svalc - lo) / (hi - lo + 1e-12) * (size_max - size_min)
        sizes = np.nan_to_num(sizes, nan=size_min)

        if CARTOPY:
            proj = ccrs.SouthPolarStereo() if np.mean(lats) < -60 else ccrs.PlateCarree()
            ax = fig.add_subplot(ax.get_subplotspec(), projection=proj)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
            ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.6)
            ax.set_extent([min(all_lons)-0.5, max(all_lons)+0.5, min(all_lats)-0.5, max(all_lats)+0.5], crs=ccrs.PlateCarree())
            sc = ax.scatter(lons, lats, c=cvals, s=sizes, cmap='viridis', vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree(), edgecolors='k', linewidth=0.2)
        else:
            sc = ax.scatter(lons, lats, c=cvals, s=sizes, cmap='viridis', vmin=vmin, vmax=vmax, edgecolors='k', linewidth=0.2)
            ax.set_xlim(min(all_lons)-0.5, max(all_lons)+0.5)
            ax.set_ylim(min(all_lats)-0.5, max(all_lats)+0.5)
        # annotate
        cnt = len(pts)
        med_c = float(np.median(cvals))
        med_s = float(np.median(sval))
        ax.set_title(f'v={v:.2f} n={cnt}')
        # place annotation box
        bbox = dict(facecolor='white', alpha=0.75, edgecolor='black')
        ax.text(0.01, 0.98, f'n={cnt}\nmedian {args.color_by}={med_c:.3g}\nmedian {args.size_by}={med_s:.3g}', transform=ax.transAxes, va='top', fontsize=8, bbox=bbox)

    # colorbar
    cax = fig.add_axes([0.92, 0.2, 0.02, 0.6])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=vmin, vmax=vmax)), cax=cax)
    cb.set_label(args.color_by)

    plt.suptitle('LSQ-only comparison (size=%s, color=%s)'%(args.size_by, args.color_by))
    plt.tight_layout(rect=[0,0,0.9,0.95])
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(outp))
    print('Wrote', outp)


if __name__ == '__main__':
    main()
