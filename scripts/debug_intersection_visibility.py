#!/usr/bin/env python3
"""Quick debug utility to re-plot per-velocity locations with larger markers

Generates simple Matplotlib scatter plots and (if Cartopy is available)
Cartopy-based maps with larger markers to help confirm whether points
are present but invisible in the default plots.

Usage:
  python3 scripts/debug_intersection_visibility.py --base plots/vel_grid_full_relaxed --out plots/vel_grid_full_relaxed/plots
"""
import os
import json
import argparse
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import importlib

# Ensure packaging.version attribute exists (some environments ship a packaging
# package where the submodule isn't exposed as an attribute, which breaks
# Cartopy's runtime import). This mirrors the monkeypatch used in the
# plotting utilities.
try:
    import packaging
    if not hasattr(packaging, 'version'):
        packaging.version = importlib.import_module('packaging.version')
except Exception:
    pass


def load(path):
    try:
        return json.load(open(path))
    except Exception:
        return []


def simple_scatter(v, arr, outdir):
    lats = [e['lat'] for e in arr if e.get('lat') is not None and math.isfinite(e.get('lat'))]
    lons = [e['lon'] for e in arr if e.get('lon') is not None and math.isfinite(e.get('lon'))]
    if not lats:
        print(f'[debug] vel {v}: no finite coords')
        return None
    plt.figure(figsize=(6,6))
    plt.scatter(lons, lats, s=24, c='red', edgecolor='k', linewidth=0.3, alpha=0.9)
    plt.xlabel('lon'); plt.ylabel('lat')
    plt.title(f'debug large scatter vel {v} n={len(arr)}')
    xmin, xmax = min(lons), max(lons)
    ymin, ymax = min(lats), max(lats)
    padx = max((xmax-xmin)*0.02, 0.2)
    pady = max((ymax-ymin)*0.02, 0.2)
    plt.xlim(xmin-padx, xmax+padx)
    plt.ylim(ymin-pady, ymax+pady)
    out = Path(outdir)/f'debug_large_scatter_vel_{float(v):.2f}.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print('[debug] wrote', out)
    return out


def cartopy_map(v, arr, outdir):
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except Exception as e:
        print('[debug] cartopy not available:', e)
        return None
    lats = [e['lat'] for e in arr if e.get('lat') is not None and math.isfinite(e.get('lat'))]
    lons = [e['lon'] for e in arr if e.get('lon') is not None and math.isfinite(e.get('lon'))]
    if not lats:
        return None
    # choose stereographic if southern and span small
    meanlat = sum(lats)/len(lats)
    if meanlat < -60:
        proj = ccrs.SouthPolarStereo()
    else:
        proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1,1,1, projection=proj)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
    ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.6)
    ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
    ax.gridlines(draw_labels=False, alpha=0.4)
    lon_min, lon_max = min(lons), max(lons)
    lat_min, lat_max = min(lats), max(lats)
    ax.set_extent([lon_min-0.5, lon_max+0.5, max(lat_min-0.5,-90), min(lat_max+0.5,90)], crs=ccrs.PlateCarree())
    # plot large, visible points
    ax.scatter(lons, lats, s=60, c='red', edgecolors='k', linewidth=0.4, transform=ccrs.PlateCarree(), zorder=10)
    out = Path(outdir)/f'debug_cartopy_large_vel_{float(v):.2f}.png'
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print('[debug] wrote', out)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', required=True)
    p.add_argument('--out', default=None)
    args = p.parse_args()
    base = Path(args.base)
    outdir = Path(args.out) if args.out else base/ 'plots'
    outdir.mkdir(parents=True, exist_ok=True)
    for fn in sorted(base.glob('locations_vel_*.post.json')):
        v = float(fn.stem.replace('locations_vel_','').replace('.post',''))
        arr = load(fn)
        print(f'[debug] vel {v:.2f} total {len(arr)}')
        simple_scatter(v, arr, outdir)
        cartopy_map(v, arr, outdir)


if __name__ == '__main__':
    main()
