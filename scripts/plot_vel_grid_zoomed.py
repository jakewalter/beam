#!/usr/bin/env python3
"""
Create zoomed scatter maps for per-velocity post-processed location JSONs.

Usage examples:
  python3 scripts/plot_vel_grid_zoomed.py --base plots/vel_grid_full_relaxed --outdir plots/vel_grid_full_relaxed/plots
  python3 scripts/plot_vel_grid_zoomed.py --base plots/vel_grid_full_relaxed --vels 2.5,3.0 --zooms 100,50,25

The script looks for files named `locations_vel_{v:.2f}.post.json` under `--base` and
creates a set of PNGs per-velocity at different zoom percentiles.
"""
import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import importlib

# Some environments have a packaging package where the submodule
# `packaging.version` isn't available as an attribute on the
# `packaging` package (which breaks Cartopy's use of
# `packaging.version.parse`). Ensure the attribute exists so
# Cartopy can reliably access `packaging.version`.
try:
    import packaging
    if not hasattr(packaging, 'version'):
        packaging.version = importlib.import_module('packaging.version')
except Exception:
    # If this fails, Cartopy will still be attempted and the script
    # will fall back to Matplotlib plotting on failure.
    pass


def load_locations(path):
    try:
        return json.load(open(path))
    except Exception:
        return []


def make_zoom_boxes(lats, lons, zoom_percents=(100, 80, 50, 25)):
    """Compute bounding boxes (latmin, latmax, lonmin, lonmax) for zoom percentiles.
    We use central percentiles: e.g., zoom=50 -> use the 25th-75th percentile interval."""
    if len(lats) == 0:
        return []
    boxes = []
    a_lat = np.array(lats)
    a_lon = np.array(lons)
    for z in zoom_percents:
        if z >= 100:
            lat_min, lat_max = a_lat.min(), a_lat.max()
            lon_min, lon_max = a_lon.min(), a_lon.max()
        else:
            low = (100 - z) / 200.0 * 100
            high = 100 - low
            lat_min, lat_max = np.percentile(a_lat, [low, high])
            lon_min, lon_max = np.percentile(a_lon, [low, high])
        # add a small padding (5%) so points at edge are visible
        lat_pad = (lat_max - lat_min) * 0.05 if (lat_max - lat_min) > 0 else 0.01
        lon_pad = (lon_max - lon_min) * 0.05 if (lon_max - lon_min) > 0 else 0.01
        boxes.append((lat_min - lat_pad, lat_max + lat_pad, lon_min - lon_pad, lon_max + lon_pad))
    return boxes


def _compute_sizes(vals, size_range=(20, 200), clip_percentiles=(1, 99)):
    """Scale numeric values `vals` into marker sizes (pt^2-ish) between size_range.
    vals: list or array of numeric values (may contain None)
    Returns list of sizes (floats) with same length. None/NaN -> minimum size.
    """
    import numpy as np
    a = np.array([np.nan if v is None else float(v) for v in vals], dtype=float)
    # compute percentile-based clipping to avoid huge outliers
    lo, hi = np.nanpercentile(a, clip_percentiles)
    a_clipped = np.clip(a, lo, hi)
    sval = (a_clipped - lo) / (hi - lo) if hi > lo else np.zeros_like(a_clipped)
    smin, smax = size_range
    sizes = smin + sval * (smax - smin)
    # NaNs -> minimum size
    sizes[np.isnan(sizes)] = smin
    return sizes.tolist()


def plot_one(vel, arr, outdir, zooms=(100, 80, 50, 25), size_by='mean_snr', size_range='20,200'):
    # split methods
    lats_lsq = [e['lat'] for e in arr if e.get('method','').startswith('lsq')]
    lons_lsq = [e['lon'] for e in arr if e.get('method','').startswith('lsq')]
    lats_int = [e['lat'] for e in arr if e.get('method','').startswith('intersection')]
    lons_int = [e['lon'] for e in arr if e.get('method','').startswith('intersection')]

    all_lats = [e['lat'] for e in arr]
    all_lons = [e['lon'] for e in arr]
    if not all_lats or not all_lons:
        print(f"No locations for vel {vel}")
        return []
    # Filter out any entries with non-finite lat/lon at the start to avoid plotting issues
    import math
    arr = [e for e in arr if e.get('lat') is not None and e.get('lon') is not None and math.isfinite(e.get('lat')) and math.isfinite(e.get('lon'))]
    all_lats = [e['lat'] for e in arr]
    all_lons = [e['lon'] for e in arr]
    if not all_lats or not all_lons:
        print(f"No finite locations for vel {vel}")
        return []

    boxes = make_zoom_boxes(all_lats, all_lons, zoom_percents=zooms)
    out_files = []
    # parse size_range string into numeric bounds
    try:
        if isinstance(size_range, str):
            size_min, size_max = [float(x) for x in size_range.split(',')]
        else:
            size_min, size_max = list(size_range)
    except Exception:
        size_min, size_max = 20.0, 200.0

    # Attempt to use the same basemap style as other scripts (cartopy PlateCarree)
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        CARTOPY = True
    except Exception:
        CARTOPY = False

    for z, box in zip(zooms, boxes):
        lat_min, lat_max, lon_min, lon_max = box
        if CARTOPY:
            try:
                # Use a south-pole stereographic projection for the
                # 100% zoom level so these zoom_100 maps match the
                # other maps in the folder (which use a southern
                # stereographic view). For other zooms use the
                # standard PlateCarree projection.
                if int(z) == 100:
                    # Match `final_plotting.py` behavior: use a plain
                    # SouthPolarStereo (no custom central_longitude) and
                    # set extent clipped to typical Antarctic bounds so
                    # orientation and framing match other plots.
                    proj = ccrs.SouthPolarStereo()
                else:
                    proj = ccrs.PlateCarree()
                fig = plt.figure(figsize=(6, 6))
                ax = fig.add_subplot(1, 1, 1, projection=proj)
                ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
                ax.add_feature(cfeature.BORDERS, alpha=0.5)
                ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
                ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.5)
                ax.gridlines(draw_labels=False, alpha=0.5)
                # set_extent expects the extents in a geographic CRS
                if int(z) == 100 and np.mean(all_lats) < -60:
                    # Use the same lat-clipping logic as `final_plotting.py`
                    lon_pad = max((max(all_lons) - min(all_lons)) * 0.02, 0.05)
                    lat_pad = max((max(all_lats) - min(all_lats)) * 0.02, 0.05)
                    ax.set_extent([
                        min(all_lons) - lon_pad, max(all_lons) + lon_pad,
                        max(min(all_lats) - lat_pad, -90), min(max(all_lats) + lat_pad, -60)
                    ], crs=ccrs.PlateCarree())
                else:
                    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
                # prepare size mapping if requested
                size_min, size_max = [float(x) for x in str(size_range).split(',')] if isinstance(size_range, str) else list(size_range)
                # compute sizes for all points
                pts = [e for e in arr if e.get('method','').startswith('lsq') or e.get('method','').startswith('intersection')]
                if size_by and size_by.lower() != 'none':
                    size_vals = [compute_field(e, size_by) for e in pts]
                    sizes = _compute_sizes(size_vals, size_range=(size_min, size_max))
                else:
                    sizes = [8 for _ in pts]

                # split sizes by method for plotting
                lsq_pts = [p for p in pts if p.get('method','').startswith('lsq')]
                int_pts = [p for p in pts if p.get('method','').startswith('intersection')]
                # filter out non-finite points for background context
                import math
                lsq_coords = [(p['lon'], p['lat']) for p in lsq_pts if math.isfinite(p.get('lon', float('nan'))) and math.isfinite(p.get('lat', float('nan')))]
                int_coords = [(p['lon'], p['lat']) for p in int_pts if math.isfinite(p.get('lon', float('nan'))) and math.isfinite(p.get('lat', float('nan')))]
                lsq_lons = [c[0] for c in lsq_coords]
                lsq_lats = [c[1] for c in lsq_coords]
                int_lons = [c[0] for c in int_coords]
                int_lats = [c[1] for c in int_coords]
                lsq_sizes = [s for p,s in zip(pts, sizes) if p in lsq_pts]
                int_sizes = [s for p,s in zip(pts, sizes) if p in int_pts]

                # For color/size rendering we will show points sized/colored per the
                # requested fields in the color-by loop; here draw faint outlines
                # of the point types to provide context but keep them subtle.
                ax.scatter(int_lons, int_lats, s=[max(2,min(8,s)) for s in int_sizes] or 6, c='C1', label='intersection', alpha=0.25, transform=ccrs.PlateCarree(), zorder=1, linewidths=0)
                ax.scatter(lsq_lons, lsq_lats, s=[max(2,min(8,s)) for s in lsq_sizes] or 6, c='C0', label='lsq', alpha=0.25, transform=ccrs.PlateCarree(), zorder=2, linewidths=0)
            except Exception as e:
                # If cartopy fails at runtime, fall back to plain plotting
                print('[WARN] cartopy projection failed; falling back to Matplotlib:', e)
                fig, ax = plt.subplots(figsize=(6, 6))
                # filter non-finite before plotting fallback
                import math
                filt_int = [(lon,lat) for lon,lat in zip(lons_int, lats_int) if math.isfinite(lon) and math.isfinite(lat)]
                filt_lsq = [(lon,lat) for lon,lat in zip(lons_lsq, lats_lsq) if math.isfinite(lon) and math.isfinite(lat)]
                if filt_int:
                    ax.scatter([p[0] for p in filt_int], [p[1] for p in filt_int], s=8, c='C1', label='intersection', alpha=0.6)
                if filt_lsq:
                    ax.scatter([p[0] for p in filt_lsq], [p[1] for p in filt_lsq], s=10, c='C0', label='lsq', alpha=0.8)
                ax.set_xlim(lon_min, lon_max)
                ax.set_ylim(lat_min, lat_max)
        else:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(lons_int, lats_int, s=8, c='C1', label='intersection', alpha=0.6)
            ax.scatter(lons_lsq, lats_lsq, s=10, c='C0', label='lsq', alpha=0.8)
            ax.set_xlim(lon_min, lon_max)
            ax.set_ylim(lat_min, lat_max)

        ax.set_xlabel('lon')
        ax.set_ylabel('lat')
        ax.set_title(f'vel {vel} zoom {z}% (n={len(arr)})')
        ax.legend()
        fn = os.path.join(outdir, f'vel_{float(vel):.2f}_zoom_{z}.png')
        fig.tight_layout()
        fig.savefig(fn, dpi=150)
        plt.close(fig)
        out_files.append(fn)
    return out_files


def compute_field(e, field):
    """Return numeric value for coloring fields: mean_snr uses 'snr'; azimuth_variance computed from backazimuths list."""
    if field == 'mean_snr':
        return float(e.get('snr') or 0.0)
    if field in ('azimuth_variance', 'azi_var'):
        bazs = e.get('backazimuths') or []
        try:
            vals = [float(b) for b in bazs]
            if not vals:
                return 0.0
            # Convert to radians, compute circular variance
            a = np.deg2rad(np.array(vals))
            R = np.sqrt((np.mean(np.cos(a)))**2 + (np.mean(np.sin(a)))**2)
            circ_var = 1.0 - R
            # scale to degrees variance proxy (not exact)
            return float(circ_var)
        except Exception:
            return 0.0
    # fallback
    return float(e.get(field) or 0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True, help='Base dir containing per-velocity post JSONs')
    parser.add_argument('--outdir', help='Output dir for plots (default: <base>/plots)')
    parser.add_argument('--vels', default='', help='Comma-separated velocities to process (default: all found)')
    parser.add_argument('--zooms', default='100,80,50,25', help='Comma-separated zoom percentiles to generate')
    parser.add_argument('--color-bys', default='mean_snr,azimuth_variance', help='Comma-separated fields to color by (e.g., mean_snr,azimuth_variance)')
    parser.add_argument('--size-by', default='mean_snr', help='Field to control marker size (e.g., mean_snr, azimuth_variance). Use "none" to disable sizing.')
    parser.add_argument('--size-range', default='6,60', help='Min,max marker size in points (e.g., 6,60). Use small sizes to show more detail.')
    parser.add_argument('--only-method', default='', help='If set, only include locations whose method starts with this string (e.g., "lsq")')
    args = parser.parse_args()

    base = args.base
    outdir = args.outdir or os.path.join(base, 'plots')
    os.makedirs(outdir, exist_ok=True)

    zooms = [int(x) for x in args.zooms.split(',') if x.strip()]

    if args.vels:
        vels = [float(v.strip()) for v in args.vels.split(',') if v.strip()]
    else:
        # detect all files matching pattern
        vels = []
        for fn in os.listdir(base):
            if fn.startswith('locations_vel_') and fn.endswith('.post.json'):
                try:
                    v = fn.replace('locations_vel_','').replace('.post.json','')
                    vels.append(float(v))
                except Exception:
                    continue
        vels = sorted(vels)

    print('Found velocities:', vels)
    generated = []
    for v in vels:
        pj = os.path.join(base, f'locations_vel_{v:.2f}.post.json')
        if not os.path.exists(pj):
            print('Missing post json for', v, pj)
            continue
        arr = load_locations(pj)
        # optionally filter by method prefix (e.g., lsq)
        if args.only_method:
            arr = [e for e in arr if (e.get('method') or '').startswith(args.only_method)]
        # produce baseline zoomed scatter on basemap
        out_files = plot_one(v, arr, outdir, zooms=zooms, size_by=args.size_by, size_range=args.size_range)
        generated.extend(out_files)

        # additional color-by plots (mean_snr, azimuth_variance, etc.)
        color_bys = [c.strip() for c in (args.color_bys.split(',') if args.color_bys else []) if c.strip()]
        for cb in color_bys:
            for z, box in zip(zooms, make_zoom_boxes([e['lat'] for e in arr], [e['lon'] for e in arr], zoom_percents=zooms)):
                lat_min, lat_max, lon_min, lon_max = box
                try:
                    import cartopy.crs as ccrs
                    import cartopy.feature as cfeature
                    CARTOPY = True
                except Exception:
                    CARTOPY = False

                # prepare color arrays for triangulated points
                tri = [e for e in arr if e.get('method','').startswith('lsq') or e.get('method','').startswith('intersection')]
                if not tri:
                    continue
                tri_lats = [e['lat'] for e in tri]
                tri_lons = [e['lon'] for e in tri]
                tri_vals = [compute_field(e, cb) for e in tri]

                # compute sizes for tri points (from args.size_by) when requested
                if args.size_by and args.size_by.lower() != 'none':
                    tri_size_vals = [compute_field(e, args.size_by) for e in tri]
                    try:
                        size_min, size_max = [float(x) for x in str(args.size_range).split(',')]
                    except Exception:
                        size_min, size_max = 20.0, 200.0
                    tri_sizes = _compute_sizes(tri_size_vals, size_range=(size_min, size_max))
                else:
                    tri_size_vals = []
                    tri_sizes = [20 for _ in tri]

                if CARTOPY:
                    import cartopy.crs as ccrs
                    import cartopy.feature as cfeature
                    # Select projection for color plots like the baseline plots
                    if int(z) == 100 and np.mean(tri_lats) < -60:
                        proj = ccrs.SouthPolarStereo()
                    else:
                        proj = ccrs.PlateCarree()
                    try:
                        fig = plt.figure(figsize=(6, 6))
                        ax = fig.add_subplot(1, 1, 1, projection=proj)
                        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
                        ax.add_feature(cfeature.BORDERS, alpha=0.5)
                        ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
                        ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.5)
                        if int(z) == 100 and np.mean(tri_lats) < -60:
                            ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.5)
                            lon_pad = max((max(tri_lons) - min(tri_lons)) * 0.02, 0.05)
                            lat_pad = max((max(tri_lats) - min(tri_lats)) * 0.02, 0.05)
                            ax.set_extent([
                                min(tri_lons) - lon_pad, max(tri_lons) + lon_pad,
                                max(min(tri_lats) - lat_pad, -90), min(max(tri_lats) + lat_pad, -60)
                            ], crs=ccrs.PlateCarree())
                            try:
                                ax.add_feature(cfeature.NaturalEarthFeature('physical', 'antarctic_ice_shelves_polys', '50m', facecolor='white', alpha=0.8))
                            except Exception:
                                pass
                        else:
                            ax.gridlines(draw_labels=False, alpha=0.5)
                            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
                        # tri_sizes/tri_size_vals were computed above (from args.size_by)
                        # filter out any non-finite points
                        import math
                        filtered = [(lon, lat, val, s) for lon, lat, val, s in zip(tri_lons, tri_lats, tri_vals, tri_sizes) if (math.isfinite(lon) and math.isfinite(lat))]
                        if not filtered:
                            continue
                        flons, flats, fvals, fsizes = zip(*filtered)
                        sc = ax.scatter(list(flons), list(flats), c=list(fvals), cmap='viridis', s=list(fsizes), transform=ccrs.PlateCarree(), zorder=4, edgecolors='k', linewidth=0.2, alpha=0.9)
                        # add a simple size legend (three quantiles) if possible
                        try:
                            import numpy as _np
                            if len(tri_size_vals) > 0:
                                qv = _np.percentile([v for v in tri_size_vals if v is not None], [25, 50, 75])
                                qsizes = _compute_sizes(qv.tolist(), size_range=(size_min, size_max))
                                x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
                                # ensure finite axis limits before attempting to place legend markers
                                import math
                                if not (math.isfinite(x0) and math.isfinite(x1) and math.isfinite(y0) and math.isfinite(y1)):
                                    raise ValueError('non-finite axis limits, skipping legend')
                                lx = x0 + 0.02*(x1-x0)
                                ly = y1 - 0.08*(y1-y0)
                                for i, (qs, qvval) in enumerate(zip(qsizes, qv)):
                                    if CARTOPY:
                                        ax.scatter([lx], [ly - i*0.06*(y1-y0)], s=qs, c='gray', alpha=0.7, transform=ccrs.PlateCarree(), zorder=6, edgecolors='k', linewidth=0.2)
                                        ax.text(lx + 0.02*(x1-x0), ly - i*0.06*(y1-y0), f'{float(qvval):.2g}', va='center', transform=ccrs.PlateCarree(), fontsize=8)
                                    else:
                                        ax.scatter([lx], [ly - i*0.06*(y1-y0)], s=qs, c='gray', alpha=0.7, zorder=6, edgecolors='k', linewidth=0.2)
                                        ax.text(lx + 0.02*(x1-x0), ly - i*0.06*(y1-y0), f'{float(qvval):.2g}', va='center', fontsize=8)
                                # annotate which field the sizes represent
                                try:
                                    if CARTOPY:
                                        ax.text(lx, ly + 0.02*(y1-y0), f'size={args.size_by}', transform=ccrs.PlateCarree(), fontsize=8)
                                    else:
                                        ax.text(lx, ly + 0.02*(y1-y0), f'size={args.size_by}', fontsize=8)
                                except Exception:
                                    pass
                        except Exception:
                            pass
                    except Exception as e:
                        print('[WARN] cartopy failed for color plot; falling back:', e)
                        fig, ax = plt.subplots(figsize=(6, 6))
                        # filter non-finite tri points before fallback plotting
                        import math
                        filt = [(lon,lat,val,s) for lon,lat,val,s in zip(tri_lons, tri_lats, tri_vals, tri_sizes) if math.isfinite(lon) and math.isfinite(lat)]
                        if filt:
                            flons, flats, fvals, fsizes = zip(*filt)
                            try:
                                sc = ax.scatter(list(flons), list(flats), c=list(fvals), cmap='viridis', s=list(fsizes))
                            except Exception:
                                sc = ax.scatter(list(flons), list(flats), c=list(fvals), cmap='viridis', s=20)
                        else:
                            sc = ax.scatter([], [], c=[], cmap='viridis', s=20)
                else:
                    fig, ax = plt.subplots(figsize=(6, 6))
                    sc = ax.scatter(tri_lons, tri_lats, c=tri_vals, cmap='viridis', s=20)
                    ax.set_xlim(lon_min, lon_max)
                    ax.set_ylim(lat_min, lat_max)

                fig.colorbar(sc, ax=ax, label=cb)
                fn = os.path.join(outdir, f'vel_{float(v):.2f}_zoom_{z}_{cb}.png')
                fig.tight_layout()
                fig.savefig(fn, dpi=150)
                plt.close(fig)
                generated.append(fn)
            # Free figures to avoid memory blowups
            plt.close('all')

    print('Generated', len(generated), 'plots in', outdir)


if __name__ == '__main__':
    main()
