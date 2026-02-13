#!/usr/bin/env python3
"""Analyze intersection_mc solutions and produce maps and error diagnostics.

Produces per-velocity:
 - map of intersection_mc points colored by `error_km` (or `std_km` if present)
 - histogram of `error_km`
 - scatter of azimuth_variance vs error_km (color by snr)
 - histogram of nearest-center distances
 - summary JSON with median/mean/percentiles counts

Outputs in: plots/vel_grid_full_relaxed/plots/
"""
import argparse, json, os
from pathlib import Path
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
    # ensure packaging.version exists in broken envs similar to other scripts
    try:
        import packaging, importlib
        if not hasattr(packaging, 'version'):
            packaging.version = importlib.import_module('packaging.version')
    except Exception:
        pass


def haversine_km(a_lat, a_lon, b_lat, b_lon):
    import math
    R = 6371.0
    dlat = math.radians(b_lat - a_lat)
    dlon = math.radians(b_lon - a_lon)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(a_lat)) * math.cos(math.radians(b_lat)) * math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def azimuth_variance_from_bazs(bazs):
    import numpy as _np
    if not bazs:
        return np.nan
    a = np.deg2rad(np.array([float(x) for x in bazs]))
    R = np.sqrt((np.mean(np.cos(a)))**2 + (np.mean(np.sin(a)))**2)
    return float(1.0 - R)


def summarize_stats(arr, key):
    vals = np.array([x for x in arr if np.isfinite(x)])
    if vals.size == 0:
        return {}
    return {
        'count': int(vals.size),
        'median': float(np.median(vals)),
        'mean': float(np.mean(vals)),
        'p25': float(np.percentile(vals, 25)),
        'p75': float(np.percentile(vals, 75)),
        'p90': float(np.percentile(vals, 90)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', default='plots/vel_grid_full_relaxed', help='Base dir with post JSONs and plots')
    p.add_argument('--outdir', default='plots/vel_grid_full_relaxed/plots', help='Output plots')
    p.add_argument('--vels', default='', help='Comma separated velocities (default: discover)')
    args = p.parse_args()

    base = Path(args.base)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.vels:
        vels = [float(x) for x in args.vels.split(',') if x.strip()]
    else:
        vels = []
        for fn in base.glob('locations_vel_*.post.json'):
            try:
                v = float(fn.stem.replace('locations_vel_','').replace('.post',''))
                vels.append(v)
            except Exception:
                continue
        vels = sorted(vels)

    summary = {}

    for v in vels:
        fp = base / f'locations_vel_{v:.2f}.post.json'
        if not fp.exists():
            print(f'Missing {fp}, skipping')
            continue
        arr = json.load(open(fp))
        # focus on intersection_mc entries
        ints = [e for e in arr if (e.get('method') or '').startswith('intersection')]
        if not ints:
            print(f'No intersection entries for v={v}')
            continue
        lats = np.array([float(e.get('lat', np.nan)) for e in ints])
        lons = np.array([float(e.get('lon', np.nan)) for e in ints])
        errs = np.array([float(e.get('error_km', np.nan)) for e in ints])
        # monte carlo std if present
        mc_std = np.array([float(e.get('std_km', np.nan)) if e.get('std_km') is not None else float(e.get('mc_std_km', np.nan)) for e in ints])
        snrs = np.array([float(e.get('snr', np.nan)) if e.get('snr') is not None else np.nan for e in ints])
        azvars = np.array([azimuth_variance_from_bazs(e.get('backazimuths') or []) for e in ints])
        nearest = np.array([float(e.get('_nearest_center_km', np.nan)) for e in ints])

        # Map colored by errs (or mc_std if errs missing)
        cmap_field = 'error_km'
        color_vals = errs.copy()
        mask_finite = np.isfinite(color_vals)
        if not np.any(mask_finite):
            # fall back to mc_std
            color_vals = mc_std.copy(); cmap_field = 'mc_std_km'
        vmin, vmax = np.nanpercentile(color_vals, [2,98])

        fig = plt.figure(figsize=(6,6))
        sc = None
        try:
            if CARTOPY:
                proj = ccrs.SouthPolarStereo() if np.nanmean(lats) < -60 else ccrs.PlateCarree()
                ax = fig.add_subplot(1,1,1, projection=proj)
                ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
                ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.5)
                ax.set_extent([np.nanmin(lons)-0.5, np.nanmax(lons)+0.5, np.nanmin(lats)-0.5, np.nanmax(lats)+0.5], crs=ccrs.PlateCarree())
                sc = ax.scatter(lons[mask_finite], lats[mask_finite], c=color_vals[mask_finite], cmap='plasma', vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree(), s=15, edgecolors='k', linewidth=0.1)
            else:
                ax = fig.add_subplot(1,1,1)
                sc = ax.scatter(lons[mask_finite], lats[mask_finite], c=color_vals[mask_finite], cmap='plasma', vmin=vmin, vmax=vmax, s=15, edgecolors='k', linewidth=0.1)
                ax.set_xlim(np.nanmin(lons)-0.5, np.nanmax(lons)+0.5)
                ax.set_ylim(np.nanmin(lats)-0.5, np.nanmax(lats)+0.5)
        except Exception as e:
            print('Map plotting failed for v', v, e)
            sc = None
        # if scatter not created (cartopy/other error), create a mappable for colorbar
        if sc is not None:
            plt.colorbar(sc, label=cmap_field)
        else:
            try:
                mappable = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=vmin, vmax=vmax))
                plt.colorbar(mappable, label=cmap_field)
            except Exception:
                pass
        plt.title(f'v={v:.2f} intersection (color={cmap_field}) n={len(ints)}')
        fig.tight_layout()
        pmap = outdir / f'vel_{v:.2f}_intersection_map_{cmap_field}.png'
        fig.savefig(pmap, dpi=150)
        plt.close(fig)

        # Histogram of error_km
        fig = plt.figure(figsize=(6,3))
        plt.hist(errs[np.isfinite(errs)], bins=50, alpha=0.8)
        plt.xlabel('error_km'); plt.ylabel('count'); plt.title(f'v={v:.2f} intersection error_km')
        ph = outdir / f'vel_{v:.2f}_intersection_error_hist.png'
        fig.tight_layout(); fig.savefig(ph, dpi=150); plt.close(fig)

        # Scatter azvar vs error_km colored by snr
        fig = plt.figure(figsize=(5,4))
        mask = np.isfinite(errs) & np.isfinite(azvars)
        if np.any(mask):
            sc = plt.scatter(snrs[mask], azvars[mask], c=errs[mask], cmap='plasma', s=10, vmin=vmin, vmax=vmax)
            plt.colorbar(sc, label='error_km')
            plt.xlabel('SNR'); plt.ylabel('azimuth_variance'); plt.title(f'v={v:.2f} azvar vs SNR (color=error_km)')
            ps = outdir / f'vel_{v:.2f}_intersection_azvar_vs_snr.png'
            fig.tight_layout(); fig.savefig(ps, dpi=150); plt.close(fig)

        # nearest center histogram
        fig = plt.figure(figsize=(5,3))
        plt.hist(nearest[np.isfinite(nearest)], bins=50, alpha=0.8)
        plt.xlabel('nearest_center_km'); plt.ylabel('count'); plt.title(f'v={v:.2f} nearest center dist (intersection)')
        pn = outdir / f'vel_{v:.2f}_intersection_nearest_center_hist.png'
        fig.tight_layout(); fig.savefig(pn, dpi=150); plt.close(fig)

        # summary
        summary[v] = {
            'n_intersection': len(ints),
            'error_stats': summarize_stats(errs, 'error_km'),
            'mc_std_stats': summarize_stats(mc_std, 'mc_std_km'),
            'nearest_center_stats': summarize_stats(nearest, 'nearest_center_km'),
            'azvar_stats': summarize_stats(azvars, 'azimuth_variance')
        }
    # save summary
    with open(outdir / 'intersection_mc_summary.json', 'w') as fh:
        json.dump(summary, fh, indent=2)
    print('Wrote plots and summary to', outdir)

if __name__ == '__main__':
    main()
