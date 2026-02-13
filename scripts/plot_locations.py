#!/usr/bin/env python3
"""
Plot locations (JSON or CSV) as a simple lat/lon scatter and optionally
overlay single-array detection rays.

Usage (JSON input):
    python scripts/plot_locations.py --locations /path/to/locations_YYYYMMDD.json --out /tmp/locations_map.png --centers /path/to/centers.json

Usage (CSV input):
    python scripts/plot_locations.py --csv /path/to/locations_summary_all.csv --out /tmp/locations_map.png --centers /path/to/centers.json --triangulated-only --show-rays

This script defaults to plotting only triangulated events (multi-array
support). Use `--triangulated-only` to hide non-triangulated points, or
`--show-rays` to draw backazimuth rays for single-array detections.

Cartopy support:
    If Cartopy is installed, this script will draw a basemap using
    PlateCarree projection for global / mid-latitude data and
    SouthPolarStereo projection for Antarctic-focused data (avg lat < -60).
    If Cartopy is not available, the script falls back to a plain Matplotlib
    projection without coastlines.
"""
import argparse
import csv
import json
import math
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    CARTOPY = True
except Exception:
    CARTOPY = False
    print("[INFO] cartopy not found; falling back to plain Matplotlib plotting (no basemap). Install cartopy for nicer maps.")

try:
    import pandas as pd
    PANDAS = True
except Exception:
    PANDAS = False


def destination_point(lat, lon, bearing_deg, distance_km):
    # Spherical earth forward formula
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    bearing = math.radians(bearing_deg)
    d_by_r = distance_km / 6371.0
    lat2 = math.asin(math.sin(lat1) * math.cos(d_by_r) + math.cos(lat1) * math.sin(d_by_r) * math.cos(bearing))
    lon2 = lon1 + math.atan2(math.sin(bearing) * math.sin(d_by_r) * math.cos(lat1), math.cos(d_by_r) - math.sin(lat1) * math.sin(lat2))
    return math.degrees(lat2), math.degrees(lon2)


def read_csv(fp):
    if PANDAS:
        return pd.read_csv(fp)
    with open(fp, 'r') as fh:
        return list(csv.DictReader(fh))


def parse_bazs(bazs_str):
    if not bazs_str:
        return []
    parts = [p for p in str(bazs_str).split(';') if p.strip()]
    out = []
    for p in parts:
        try:
            out.append(float(p))
        except Exception:
            pass
    return out


def classify_row_r(df_row, min_union_arrays=2):
    # df_row may be a dict (csv) or pandas Series
    tri = False
    if 'triangulated' in df_row:
        val = df_row.get('triangulated') if isinstance(df_row, dict) else df_row['triangulated']
        if str(val).lower() in ('true', '1'):
            tri = True
    if not tri and 'union_arrays_count' in df_row:
        try:
            if int(float(df_row.get('union_arrays_count') if isinstance(df_row, dict) else df_row['union_arrays_count'] or 0)) >= min_union_arrays:
                tri = True
        except Exception:
            pass
    return tri


def main():
    parser = argparse.ArgumentParser(description='Plot locations (JSON or merged CSV)')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--locations', help='Path to locations JSON file (list of dicts)')
    group.add_argument('--csv', help='Path to locations CSV (merged or per-day)')
    parser.add_argument('--centers', help='Optional centers.json mapping subarray_id -> [lat, lon]')
    parser.add_argument('--out', default='locations_map.png', help='Output PNG path')
    parser.add_argument('--triangulated-only', action='store_true', help='Plot only triangulated rows (default False)')
    parser.add_argument('--show-rays', action='store_true', help='Draw backazimuth rays for single-array detections (requires --centers)')
    parser.add_argument('--rays-km', type=float, default=1000.0, help='Length of rays (km) to draw')
    parser.add_argument('--min-union-arrays', type=int, default=2, help='Min union_arrays_count to consider triangulated')
    parser.add_argument('--color-by', type=str, default=None, help='Field to color triangulated points by: error_km,count,union_arrays_count,time_span_s')
    args = parser.parse_args()

    rows = None
    if args.locations:
        with open(args.locations, 'r') as fh:
            rows = json.load(fh)
    else:
        rows = read_csv(args.csv)

    # Robust empty check: DataFrames are ambiguous in boolean context
    if rows is None or (PANDAS and getattr(rows, 'empty', True)) or (not PANDAS and not rows):
        print('No locations found')
        sys.exit(0)

    centers = None
    if args.centers:
        with open(args.centers, 'r') as fh:
            centers = json.load(fh)
            centers = {str(k): (v[0], v[1]) for k, v in centers.items()}

    # Normalize iterator
    it = rows if isinstance(rows, list) else rows.to_dict(orient='records')

    lats = []
    lons = []
    errs = []
    counts = []
    arrays_list = []
    bazs_list = []
    tri_mask = []

    for r in it:
        try:
            lat = float(r.get('lat'))
            lon = float(r.get('lon'))
        except Exception:
            continue
        lats.append(lat)
        lons.append(lon)
        errs.append(float(r.get('error_km', 0.0) or 0.0))
        counts.append(int(float(r.get('count', 1) or 1)))
        arrays_list.append(r.get('arrays', ''))
        bazs_list.append(parse_bazs(r.get('backazimuths', '')))
        tri_mask.append(classify_row_r(r, args.min_union_arrays))

    # Plot
    avg_lat = np.mean(lats) if lats else 0.0
    if CARTOPY:
        # Use polar stereographic for Antarctic-focused data, else PlateCarree
        if avg_lat < -60:
            projection = ccrs.SouthPolarStereo()
            fig = plt.figure(figsize=(15, 12))
            ax = fig.add_subplot(1, 1, 1, projection=projection)
            # For Antarctic ensure lat extent does not go below -90
            lat_min = max(min(lats) if lats else -90, -90)
            lat_max = min(max(lats) if lats else -60, -60 + 30)
            lon_pad = max((max(lons) - min(lons)) * 0.02 if lons else 0.05, 0.05)
            lat_pad = max((lat_max - lat_min) * 0.02 if lats else 0.05, 0.05)
            ax.set_extent([
                min(lons) - lon_pad, max(lons) + lon_pad,
                max(lat_min - lat_pad, -90), min(lat_max + lat_pad, -60)
            ], crs=ccrs.PlateCarree())
        else:
            projection = ccrs.PlateCarree()
            fig = plt.figure(figsize=(11, 6))
            ax = fig.add_subplot(1, 1, 1, projection=projection)
            lon_pad = max((max(lons) - min(lons)) * 0.02 if lons else 0.05, 0.05)
            lat_pad = max((max(lats) - min(lats)) * 0.02 if lats else 0.05, 0.05)
            ax.set_extent([
                min(lons) - lon_pad, max(lons) + lon_pad,
                min(lats) - lat_pad, max(lats) + lat_pad
            ], crs=ccrs.PlateCarree())
            lat_pad = max((max(lats) - min(lats)) * 0.02 if lats else 0.05, 0.05)
            ax.set_extent([
                min(lons) - lon_pad, max(lons) + lon_pad,
                min(lats) - lat_pad, max(lats) + lat_pad
            ], crs=ccrs.PlateCarree())
        # Add map features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, alpha=0.5)
        ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
        ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.5)
        if avg_lat < -60:
            try:
                ax.add_feature(cfeature.NaturalEarthFeature('physical', 'antarctic_ice_shelves_polys', '50m', facecolor='white', alpha=0.8))
            except Exception:
                pass
        ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.5)
    else:
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.set_aspect('equal', adjustable='box')

    # triangulated vs non
    tri_lats = [lat for lat, t in zip(lats, tri_mask) if t]
    tri_lons = [lon for lon, t in zip(lons, tri_mask) if t]
    tri_err = [e for e, t in zip(errs, tri_mask) if t]
    tri_counts = [c for c, t in zip(counts, tri_mask) if t]

    non_lats = [lat for lat, t in zip(lats, tri_mask) if not t]
    non_lons = [lon for lon, t in zip(lons, tri_mask) if not t]

    sizes = [max(10, min(200, 10 * c)) for c in tri_counts]
    # Decide color mapping for triangulated points
    color_by = args.color_by
    tri_color = None
    cbar_label = None
    if color_by is None:
        if any([v > 0 for v in tri_err]):
            color_by = 'error_km'
        else:
            color_by = 'count'

    if color_by == 'error_km':
        tri_color = tri_err
        cbar_label = 'error_km'
    elif color_by == 'count':
        tri_color = tri_counts
        cbar_label = 'count'
    elif color_by == 'union_arrays_count':
        tri_color = [int(float(r.get('union_arrays_count') or 0)) for r in it if classify_row_r(r, args.min_union_arrays)]
        cbar_label = 'union_arrays_count'
    elif color_by == 'mean_snr':
        tri_color = [float(r.get('mean_snr') or 0.0) for r in it if classify_row_r(r, args.min_union_arrays)]
        cbar_label = 'mean_snr'
    elif color_by in ('azimuth_variance', 'azi_var'):
        tri_color = [float(r.get('azimuth_variance') or 0.0) for r in it if classify_row_r(r, args.min_union_arrays)]
        cbar_label = 'azimuth_variance'
    elif color_by == 'time_span_s':
        tri_color = [float(r.get('time_span_s') or 0.0) for r in it if classify_row_r(r, args.min_union_arrays)]
        cbar_label = 'time_span_s'
    else:
        tri_color = tri_err if tri_err else tri_counts
        cbar_label = color_by

    if CARTOPY:
        sc = ax.scatter(tri_lons, tri_lats, c=tri_color, cmap='viridis', s=sizes, alpha=0.9, edgecolor='k', label='triangulated', transform=ccrs.PlateCarree())
    else:
        sc = ax.scatter(tri_lons, tri_lats, c=tri_color, cmap='viridis', s=sizes, alpha=0.9, edgecolor='k', label='triangulated')
    # Add colorbar if using a numeric color
    try:
        import numpy as _np
        if sc is not None and hasattr(sc, 'get_array') and _np.any(_np.isfinite(_np.array(sc.get_array()))):
            cbar = fig.colorbar(sc, ax=ax)
            cbar.set_label(cbar_label)
    except Exception:
        pass

    if not args.triangulated_only and non_lats:
        if CARTOPY:
            ax.scatter(non_lons, non_lats, s=20, color='grey', alpha=0.5, marker='x', label='single-array', transform=ccrs.PlateCarree())
        else:
            ax.scatter(non_lons, non_lats, s=20, color='grey', alpha=0.5, marker='x', label='single-array')

    if centers:
        for sid, (lat0, lon0) in centers.items():
            if CARTOPY:
                ax.scatter(lon0, lat0, c='k', marker='^', s=80, transform=ccrs.PlateCarree())
                ax.text(lon0 + 0.02, lat0 + 0.02, str(sid), fontsize=8, transform=ccrs.PlateCarree())
            else:
                ax.scatter(lon0, lat0, c='k', marker='^', s=80)
                ax.text(lon0 + 0.02, lat0 + 0.02, str(sid), fontsize=8)

    if args.show_rays and centers:
        # Draw rays only for non-triangulated rows (single-array detections)
        # This ensures triangulated-only plots remain uncluttered
        for idx, (arrays_str, bazs) in enumerate(zip(arrays_list, bazs_list)):
            # skip triangulated rows
            if tri_mask and idx < len(tri_mask) and tri_mask[idx]:
                continue
            arrs = [s.strip() for s in str(arrays_str).split(';') if s.strip()]
            if len(arrs) == 0:
                continue
            arr0 = arrs[0]
            if arr0 not in centers:
                continue
            if not bazs:
                continue
            center_lat, center_lon = centers[arr0]
            baz = bazs[0]
            dest_lat, dest_lon = destination_point(center_lat, center_lon, baz, args.rays_km)
            if CARTOPY:
                ax.plot([center_lon, dest_lon], [center_lat, dest_lat], color='red', alpha=0.4, linewidth=0.8, transform=ccrs.PlateCarree())
            else:
                ax.plot([center_lon, dest_lon], [center_lat, dest_lat], color='red', alpha=0.4, linewidth=0.8)

    ax.legend(loc='upper right')
    if lats and lons:
        minlat, maxlat = min(lats), max(lats)
        minlon, maxlon = min(lons), max(lons)
        lat_margin = 0.1 * max(0.1, maxlat - minlat)
        lon_margin = 0.1 * max(0.1, maxlon - minlon)
        # If not using Cartopy, manually set limits; Cartopy extent already set above
        if not CARTOPY:
            ax.set_ylim(minlat - lat_margin, maxlat + lat_margin)
            ax.set_xlim(minlon - lon_margin, maxlon + lon_margin)

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    title = 'Locations map (triangulated only)' if args.triangulated_only else 'Locations map'
    if avg_lat < -60:
        title = 'Antarctic ' + title
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print('Wrote', args.out)


if __name__ == '__main__':
    main()
