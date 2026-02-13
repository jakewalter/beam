#!/usr/bin/env python3
"""
Plot locations (CSV) with triangulation error circles (Monte Carlo estimate) and optional rays.

This script reads a merged locations CSV (with `lat`,`lon`,`triangulated`,`arrays`,`backazimuths`, etc.)
and a separate error CSV created by `estimate_triangulation_error.py` that includes
`mc_radial_std_km` per row (matching by `time_epoch` or `lat/lon`).

It plots triangulated points colored by `mc_radial_std_km` and draws a thin circle
around each triangulated point with radius equal to the MC radial std. It also optionally
plots rays for single-array detections (non-triangulated rows) using `centers.json`.
"""
import argparse
import csv
import json
import os
import sys
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    CARTOPY = True
except Exception:
    CARTOPY = False
try:
    import pandas as pd
    PANDAS = True
except Exception:
    PANDAS = False


def read_csv(fp):
    if PANDAS:
        return pd.read_csv(fp)
    with open(fp, 'r') as fh:
        return list(csv.DictReader(fh))


def km_to_deg(km):
    # Rough conversion for visualization: 1 deg ~ 111 km
    return km / 111.0


def find_error_for_row(r, errors):
    # errors: list of dicts with mc_radial_std_km and time_epoch or lat/lon
    # Try to match by time_epoch first
    t = r.get('time_epoch')
    if t is not None:
        try:
            t_float = float(t)
            for e in errors:
                if 'time_epoch' in e and float(e['time_epoch']) == t_float:
                    return e
        except Exception:
            pass
    # fallback to lat/lon match within small tol
    lat = float(r.get('lat', 0))
    lon = float(r.get('lon', 0))
    for e in errors:
        try:
            if abs(float(e.get('lat', 1e9)) - lat) < 1e-6 and abs(float(e.get('lon', 1e9)) - lon) < 1e-6:
                return e
        except Exception:
            continue
    return None


def destination_point(lat, lon, bearing_deg, distance_km):
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    bearing = math.radians(bearing_deg)
    d_by_r = distance_km / 6371.0
    lat2 = math.asin(math.sin(lat1) * math.cos(d_by_r) + math.cos(lat1) * math.sin(d_by_r) * math.cos(bearing))
    lon2 = lon1 + math.atan2(math.sin(bearing) * math.sin(d_by_r) * math.cos(lat1), math.cos(d_by_r) - math.sin(lat1) * math.sin(lat2))
    return math.degrees(lat2), math.degrees(lon2)


def plot_error_map(rows, errors, centers, out, snr_threshold=0.0, show_rays=False):
    # Build arrays
    tri_rows = [r for r in rows if (str(r.get('triangulated', 'False')).lower() in ('true','1') or int(float(r.get('union_arrays_count', 0))) >= 2)]
    nontri_rows = [r for r in rows if r not in tri_rows]

    # Setup map
    lats = [float(r.get('lat')) for r in rows if r.get('lat')]
    lons = [float(r.get('lon')) for r in rows if r.get('lon')]
    if not lats:
        print('No valid coordinates found. Exiting')
        return
    avg_lat = np.mean(lats)
    if CARTOPY:
        if avg_lat < -60:
            projection = ccrs.SouthPolarStereo(); figsize=(15, 12)
        else:
            projection = ccrs.PlateCarree(); figsize=(11, 6)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1, projection=projection)
        lon_pad = max((max(lons) - min(lons)) * 0.02, 0.05)
        lat_pad = max((max(lats) - min(lats)) * 0.02, 0.05)
        ax.set_extent([min(lons)-lon_pad, max(lons)+lon_pad, min(lats)-lat_pad, max(lats)+lat_pad], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.7)
        ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
        ax.gridlines(draw_labels=True)
    else:
        fig, ax = plt.subplots(figsize=(11,6))
        ax.set_aspect('equal', adjustable='box')
    # Plot triangulated points colored by radial std
    tri_lats = [float(r.get('lat')) for r in tri_rows]
    tri_lons = [float(r.get('lon')) for r in tri_rows]
    tri_rad = []
    for r in tri_rows:
        e = find_error_for_row(r, errors)
        val = float(e['mc_radial_std_km']) if e and e.get('mc_radial_std_km') not in (None, '') else 0.0
        tri_rad.append(val)
    sc = None
    if tri_lats:
        if CARTOPY:
            sc = ax.scatter(tri_lons, tri_lats, c=tri_rad, cmap='viridis', s=40, transform=ccrs.PlateCarree(), zorder=6, edgecolors='white')
        else:
            sc = ax.scatter(tri_lons, tri_lats, c=tri_rad, cmap='viridis', s=40, zorder=6, edgecolors='white')
    if sc is not None:
        cbar = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label('MC radial std (km)')

    # Draw error circles
    for r, rad_km in zip(tri_rows, tri_rad):
        if not rad_km or rad_km <= 0:
            continue
        lat = float(r.get('lat'))
        lon = float(r.get('lon'))
        deg = km_to_deg(rad_km)
        circ = plt.Circle((lon, lat), deg, fill=False, edgecolor='orange', linewidth=0.8, alpha=0.8)
        if CARTOPY:
            ax.add_patch(circ)
            circ.set_transform(ccrs.PlateCarree())
        else:
            ax.add_patch(circ)

    # Plot non-triangulated rays if asked
    if show_rays and centers:
        for r in nontri_rows:
            try:
                arrays = [s.strip() for s in str(r.get('arrays', '')).split(';') if s.strip()]
                bazs = [float(s) for s in str(r.get('backazimuths', '')).split(';') if s.strip()]
            except Exception:
                continue
            if not arrays or not bazs:
                continue
            arr0 = arrays[0]
            if arr0 not in centers:
                continue
            lat0, lon0 = centers[arr0]
            baz = bazs[0]
            dest_lat, dest_lon = destination_point(lat0, lon0, baz, 2000.0)
            if CARTOPY:
                ax.plot([lon0, dest_lon], [lat0, dest_lat], color='red', alpha=0.25, linewidth=0.6, transform=ccrs.PlateCarree())
            else:
                ax.plot([lon0, dest_lon], [lat0, dest_lat], color='red', alpha=0.25, linewidth=0.6)

    ax.set_title('Triangulated sources with MC error (radial std)')
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('Wrote', out)


def main():
    parser = argparse.ArgumentParser(description='Plot locations with triangulation error circles and rays')
    parser.add_argument('--csv', required=True)
    parser.add_argument('--error-csv', help='CSV produced by estimate_triangulation_error.py (matching rows)')
    parser.add_argument('--centers', help='centers.json mapping array_id->[lat,lon]')
    parser.add_argument('--out', default='plots/locations_error_map.png')
    parser.add_argument('--show-rays', action='store_true')
    args = parser.parse_args()
    rows = read_csv(args.csv)
    errors = read_csv(args.error_csv) if args.error_csv else []
    centers = None
    if args.centers:
        with open(args.centers) as fh:
            centers = json.load(fh)
            centers = {str(k): (v[0], v[1]) for k, v in centers.items()}
    plot_error_map(rows if isinstance(rows, list) else rows.to_dict(orient='records'), errors if isinstance(errors, list) else errors.to_dict(orient='records'), centers, args.out, show_rays=args.show_rays)


if __name__ == '__main__':
    main()
