#!/usr/bin/env python3
"""Plot a horizontal pair: (left) Intersection map colored by error_km and (right) LSQ map + offsets.

Produces a side-by-side PNG for a given velocity using the post JSON.
"""
import os
import json
import argparse
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import importlib

# packaging.version monkeypatch (same as other scripts)
try:
    import packaging
    if not hasattr(packaging, 'version'):
        packaging.version = importlib.import_module('packaging.version')
except Exception:
    pass


def haversine_km(a, b):
    # a,b are (lat,lon) in degrees
    import math
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    R = 6371.0
    h = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*(math.sin(dlon/2)**2)
    return 2*R*math.asin(min(1, math.sqrt(h)))


def compute_azvar(e):
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


def load_post(base, vel):
    fn = Path(base)/f'locations_vel_{float(vel):.2f}.post.json'
    return json.load(open(fn))


def match_nearest(inter_points, lsq_points, max_km=50.0):
    # Return list of tuples (inter, lsq, dist_km)
    matches = []
    lsq_used = set()
    lsq_coords = [(i, (e['lat'], e['lon'])) for i, e in enumerate(lsq_points)]
    for ie in inter_points:
        ic = (ie['lat'], ie['lon'])
        best = None
        bestd = float('inf')
        for j, lc in lsq_coords:
            if j in lsq_used:
                continue
            d = haversine_km(ic, lc)
            if d < bestd:
                bestd = d
                best = j
        if best is not None and bestd <= max_km:
            matches.append((ie, lsq_points[best], bestd))
            lsq_used.add(best)
    return matches


def make_pair_plot(base, vel, outdir, max_match_km=50.0, min_line_offset_km=2.0):
    arr = load_post(base, vel)
    inter = [e for e in arr if (e.get('method') or '').startswith('intersection')]
    lsq = [e for e in arr if (e.get('method') or '').startswith('lsq')]
    if not inter or not lsq:
        print('not enough points for', vel)
        return None

    # build arrays
    int_lats = [e['lat'] for e in inter]
    int_lons = [e['lon'] for e in inter]
    int_err = [float(e.get('error_km') or 0.0) for e in inter]
    int_snr = [float(e.get('snr') or 0.0) for e in inter]
    int_azvar = [compute_azvar(e) for e in inter]

    lsq_lats = [e['lat'] for e in lsq]
    lsq_lons = [e['lon'] for e in lsq]
    lsq_snr = [float(e.get('snr') or 0.0) for e in lsq]
    lsq_resid = [float(e.get('residual_norm') or 0.0) for e in lsq]

    matches = match_nearest(inter, lsq, max_km=max_match_km)
    match_dists = [d for (_, _, d) in matches]

    # figure
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        CARTOPY = True
    except Exception:
        CARTOPY = False

    fig = plt.figure(figsize=(12, 6))
    # Left: intersection map
    if CARTOPY:
        ax0 = fig.add_subplot(1, 2, 1, projection=ccrs.SouthPolarStereo() if np.mean(int_lats) < -60 else ccrs.PlateCarree())
        ax0.add_feature(cfeature.COASTLINE, linewidth=0.6)
        ax0.add_feature(cfeature.LAND, color='lightgray', alpha=0.6)
        ax0.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
        ax0.set_extent([min(int_lons)-0.5, max(int_lons)+0.5, max(min(int_lats)-0.5,-90), min(max(int_lats)+0.5,90)], crs=ccrs.PlateCarree())
        sc = ax0.scatter(int_lons, int_lats, c=int_err, cmap='viridis', s=np.clip((np.array(int_snr)-np.min(int_snr)+1)*8, 6, 200), transform=ccrs.PlateCarree(), edgecolors='k', linewidth=0.2, alpha=0.9)
        cbar = fig.colorbar(sc, ax=ax0, orientation='vertical', shrink=0.7)
        cbar.set_label('intersection error (km)')
    else:
        ax0 = fig.add_subplot(1,2,1)
        sc = ax0.scatter(int_lons, int_lats, c=int_err, cmap='viridis', s=np.clip((np.array(int_snr)-np.min(int_snr)+1)*8,6,200), edgecolors='k', linewidth=0.2, alpha=0.9)
        cbar = fig.colorbar(sc, ax=ax0)
        cbar.set_label('intersection error (km)')

    ax0.set_title(f'Vel {vel:.2f} — Intersection (n={len(inter)})')
    ax0.set_xlabel('lon'); ax0.set_ylabel('lat')

    # Right: LSQ map with lines to matched intersections
    if CARTOPY:
        ax1 = fig.add_subplot(1,2,2, projection=ccrs.SouthPolarStereo() if np.mean(lsq_lats) < -60 else ccrs.PlateCarree())
        ax1.add_feature(cfeature.COASTLINE, linewidth=0.6)
        ax1.add_feature(cfeature.LAND, color='lightgray', alpha=0.6)
        ax1.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
        ax1.set_extent([min(int_lons)-0.5, max(int_lons)+0.5, max(min(int_lats)-0.5,-90), min(max(int_lats)+0.5,90)], crs=ccrs.PlateCarree())
        # draw lines for matches *only* for offsets above the configured threshold
        lines_to_plot = [(ie, le, d) for ie, le, d in matches if d >= min_line_offset_km]
        for ie, le, d in lines_to_plot:
            ax1.plot([ie['lon'], le['lon']], [ie['lat'], le['lat']], color='lightgray', alpha=0.25, linewidth=0.4, transform=ccrs.PlateCarree(), zorder=2)
        # draw LSQ points on top of the lines so they are not overplotted
        sc2 = ax1.scatter(lsq_lons, lsq_lats, c=lsq_resid, cmap='plasma', s=np.clip((np.array(lsq_snr)-np.min(lsq_snr)+1)*8,6,200), transform=ccrs.PlateCarree(), edgecolors='k', linewidth=0.2, alpha=0.95, zorder=10)
        cbar2 = fig.colorbar(sc2, ax=ax1, orientation='vertical', shrink=0.7)
        cbar2.set_label('LSQ residual norm')
    else:
        ax1 = fig.add_subplot(1,2,2)
        # plot only significant lines first
        lines_to_plot = [(ie, le, d) for ie, le, d in matches if d >= min_line_offset_km]
        for ie, le, d in lines_to_plot:
            ax1.plot([ie['lon'], le['lon']], [ie['lat'], le['lat']], color='lightgray', alpha=0.25, linewidth=0.4)
        sc2 = ax1.scatter(lsq_lons, lsq_lats, c=lsq_resid, cmap='plasma', s=np.clip((np.array(lsq_snr)-np.min(lsq_snr)+1)*8,6,200), edgecolors='k', linewidth=0.2, alpha=0.95, zorder=10)
        cbar2 = fig.colorbar(sc2, ax=ax1)
        cbar2.set_label('LSQ residual norm')

    ax1.set_title(f'LSQ (n={len(lsq)}) — matched pairs: {len(matches)}')
    ax1.set_xlabel('lon'); ax1.set_ylabel('lat')

    # small diagnostics panel inset: histogram of match distances
    axh = fig.add_axes([0.55, 0.07, 0.35, 0.2])  # x,y,w,h in figure coords
    if match_dists:
        axh.hist(match_dists, bins=40, color='gray', alpha=0.8)
        axh.axvline(np.median(match_dists), color='C1', linestyle='--', label=f'median {np.median(match_dists):.2f} km')
        axh.set_xlabel('intersection→LSQ (km)')
        axh.set_ylabel('count')
        axh.legend(fontsize=8)
    else:
        axh.text(0.1, 0.5, 'No matches within threshold', va='center')
        axh.set_axis_off()

    # save
    outfn = Path(outdir)/f'vel_{float(vel):.2f}_intersection_vs_lsq.png'
    fig.tight_layout()
    fig.savefig(outfn, dpi=150)
    plt.close(fig)
    print('wrote', outfn)
    # return some summary
    out_summary = {'vel':vel, 'n_inter':len(inter), 'n_lsq':len(lsq), 'n_matches':len(matches), 'match_median_km': float(np.median(match_dists)) if match_dists else None}
    return outfn, out_summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', required=True)
    p.add_argument('--vel', required=True, type=float)
    p.add_argument('--outdir', default=None)
    p.add_argument('--max-match-km', default=50.0, type=float)
    p.add_argument('--min-line-offset-km', default=2.0, type=float, help='Only draw connecting lines for matches with offset >= this (km)')
    args = p.parse_args()
    outdir = Path(args.outdir) if args.outdir else Path(args.base)/'plots'
    outdir.mkdir(parents=True, exist_ok=True)
    outfn, summary = make_pair_plot(args.base, args.vel, outdir, max_match_km=args.max_match_km, min_line_offset_km=args.min_line_offset_km)
    print('summary:', summary)


if __name__ == '__main__':
    main()
