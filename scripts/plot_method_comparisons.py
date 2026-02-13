#!/usr/bin/env python3
"""
Plot triangulated locations split by method and produce error/difference summaries.

Creates:
 - map scatter colored by method
 - polar backazimuth hist overlay by method
 - histograms of error_km by method
 - residual_norm vs mean_snr scatter by method
 - matches between methods with deltas, output CSV, and histograms of positional differences

Usage:
 PYTHONPATH=. python3 scripts/plot_method_comparisons.py --csv plots/locations_summary_bench_tri_30s_min_snr8_all_dedup.csv --json plots/locations_bench_tri_30s_min_snr8_all.json --out-dir plots

"""
import argparse
import csv
import json
import math
import os
import sys
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


def read_csv(fp):
    rows=[]
    with open(fp, 'r') as fh:
        r=csv.DictReader(fh)
        for row in r:
            rows.append(row)
    return rows


def read_json(fp):
    with open(fp, 'r') as fh:
        return json.load(fh)


def haversine_km(lat1, lon1, lat2, lon2):
    # haversine formula
    R=6371.0
    phi1=math.radians(lat1); phi2=math.radians(lat2)
    dphi=math.radians(lat2-lat1); dlambda=math.radians(lon2-lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(max(0.0,1.0-a)))
    return R*c


def prepare_method_groups(rows):
    groups={}
    for r in rows:
        method=(r.get('method') or '').strip()
        if not method:
            method='(blank)'
        if method not in groups:
            groups[method]=[]
        try:
            r['latf']=float(r.get('lat') or 0.0)
            r['lonf']=float(r.get('lon') or 0.0)
            r['errorf']=float(r.get('error_km') or 0.0)
            r['snrf']=float(r.get('mean_snr') or r.get('snr') or 0.0)
            r['resf']=float(r.get('residual_norm') or 0.0)
            r['tepochf']=float(r.get('time_epoch') or 0.0)
        except Exception:
            continue
        groups[method].append(r)
    return groups


def plot_map(groups, outpath, centers=None, extent=None):
    # Combine all lat/lon to set extents
    all_lats=[]; all_lons=[]
    for g in groups.values():
        for r in g:
            all_lats.append(r['latf']); all_lons.append(r['lonf'])
    if not all_lats:
        print('No locations to plot')
        return
    avg_lat=sum(all_lats)/len(all_lats)

    if CARTOPY:
        projection=ccrs.SouthPolarStereo() if avg_lat<-60 else ccrs.PlateCarree()
        fig=plt.figure(figsize=(12,8)); ax=fig.add_subplot(1,1,1, projection=projection)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
        ax.add_feature(cfeature.BORDERS, linewidth=0.4)
        ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
        ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.5)
    else:
        fig, ax = plt.subplots(figsize=(12,8))

    # categorical colors for methods
    methods=sorted(groups.keys(), key=lambda s: (-len(groups[s]), s))
    cmap = plt.get_cmap('tab10')

    for i, m in enumerate(methods):
        pts=groups[m]
        lats=[r['latf'] for r in pts]; lons=[r['lonf'] for r in pts]
        sizes=[max(10, min(200, 10*max(1, r.get('snrf') or 1))) for r in pts]
        c = cmap(i % 10)
        if CARTOPY:
            ax.scatter(lons, lats, s=sizes, c=[c], alpha=0.8, transform=ccrs.PlateCarree(), label=f'{m} ({len(pts)})')
        else:
            ax.scatter(lons, lats, s=sizes, c=[c], alpha=0.8, label=f'{m} ({len(pts)})')

    # Apply extent if provided, otherwise use a small padded bounding box around all points
    if extent is not None:
        minlon, maxlon, minlat, maxlat = extent
        if CARTOPY:
            ax.set_extent([minlon, maxlon, minlat, maxlat], crs=ccrs.PlateCarree())
        else:
            ax.set_xlim(minlon, maxlon)
            ax.set_ylim(minlat, maxlat)
    else:
        if CARTOPY:
            lon_pad=max(0.05, (max(all_lons)-min(all_lons))*0.02)
            lat_pad=max(0.05, (max(all_lats)-min(all_lats))*0.02)
            ax.set_extent([min(all_lons)-lon_pad, max(all_lons)+lon_pad, min(all_lats)-lat_pad, max(all_lats)+lat_pad], crs=ccrs.PlateCarree())
        else:
            ax.set_xlim(min(all_lons)-0.1, max(all_lons)+0.1);
            ax.set_ylim(min(all_lats)-0.1, max(all_lats)+0.1);

    ax.legend(loc='best', fontsize='small')
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    ax.set_title('Locations by method (color-coded)')
    fig.tight_layout(); fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print('Wrote', outpath)


def polar_hist_by_method(groups, outpath, bins=36):
    fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(1,1,1, projection='polar')
    ax.set_theta_zero_location('N'); ax.set_theta_direction(-1)
    cmap=plt.get_cmap('tab10')
    methods=sorted(groups.keys(), key=lambda s: (-len(groups[s]), s))
    max_counts=0
    for i,m in enumerate(methods):
        angles=[]
        for r in groups[m]:
            bazlist=[p for p in (r.get('backazimuths') or '').split(';') if p.strip()]
            for p in bazlist:
                try: angles.append(float(p)%360)
                except: pass
        if not angles: continue
        theta = np.radians(angles)
        counts, edges = np.histogram(theta, bins=bins, range=(0,2*math.pi))
        max_counts=max(max_counts, counts.max())
        widths = np.diff(edges)
        ax.step(edges, np.append(counts, counts[0]), where='post', color=cmap(i%10), linewidth=1.2, label=f'{m} ({len(groups[m])})')
    if max_counts>0:
        ax.set_rlim(0, max_counts*1.1)
    ax.set_title('Backazimuth distribution by method')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    fig.tight_layout(); fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print('Wrote', outpath)


def error_hist_by_method(groups, outpath):
    fig, ax = plt.subplots(figsize=(8,6))
    methods=sorted(groups.keys(), key=lambda s: (-len(groups[s]), s))
    for i,m in enumerate(methods):
        errs=[r['errorf'] for r in groups[m] if r.get('errorf') is not None]
        if not errs: continue
        ax.hist(errs, bins=50, alpha=0.5, label=f'{m} ({len(errs)})')
    ax.set_xlabel('error_km'); ax.set_ylabel('count')
    ax.set_title('Error (km) distribution by method')
    ax.legend(loc='best')
    fig.tight_layout(); fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print('Wrote', outpath)


def residual_vs_snr(groups, outpath):
    fig, ax = plt.subplots(figsize=(8,6))
    methods=sorted(groups.keys(), key=lambda s: (-len(groups[s]), s))
    for i,m in enumerate(methods):
        xs=[r['snrf'] for r in groups[m] if r.get('snrf') is not None]
        ys=[r['resf'] for r in groups[m] if r.get('resf') is not None]
        if not xs or not ys: continue
        ax.scatter(xs, ys, alpha=0.5, label=f'{m} ({len(xs)})')
    ax.set_xlabel('mean_snr'); ax.set_ylabel('residual_norm')
    ax.set_title('Residual norm vs mean SNR by method')
    ax.set_yscale('symlog')
    ax.legend(loc='best')
    fig.tight_layout(); fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print('Wrote', outpath)


def match_methods_and_diffs(groups, outdir, max_time_diff_s=1.0, max_dist_km=100.0):
    # Find pairs of events from different methods within time and distance thresholds
    items=[]
    for m, lst in groups.items():
        for r in lst:
            items.append({'method':m, 'lat':r['latf'], 'lon':r['lonf'], 'te':r['tepochf'], 'error':r['errorf'], 'res':r['resf'], 'snr':r['snrf']})
    # brute force matching: O(N^2) but N~10k so acceptable; restrict mapping for speed by binning
    matches=[]
    n=len(items)
    for i in range(n):
        a=items[i]
        for j in range(i+1,n):
            b=items[j]
            if a['method']==b['method']: continue
            if abs(a['te']-b['te'])>max_time_diff_s: continue
            dist=haversine_km(a['lat'], a['lon'], b['lat'], b['lon'])
            if dist>max_dist_km: continue
            matches.append({'method_a':a['method'], 'method_b':b['method'], 'dist_km':dist, 'delta_error':a['error']-b['error'], 'delta_res':a['res']-b['res'], 'snr_a':a['snr'], 'snr_b':b['snr']})
    # Save matches to CSV
    outcsv=os.path.join(outdir, 'method_matches_diffs.csv')
    with open(outcsv, 'w') as fh:
        keys=['method_a','method_b','dist_km','delta_error','delta_res','snr_a','snr_b']
        fh.write(','.join(keys)+'\n')
        for m in matches:
            fh.write(','.join(str(m[k]) for k in keys)+'\n')
    print('Wrote', outcsv, 'with', len(matches), 'pairs')
    # plot hist of dist_km
    if matches:
        dists=[m['dist_km'] for m in matches]
        plt.figure(figsize=(6,4)); plt.hist(dists, bins=50); plt.xlabel('pair distance (km)'); plt.ylabel('count'); plt.title('Distance between matched pairs'); plt.tight_layout(); plt.savefig(os.path.join(outdir, 'match_pair_distance_hist.png'), dpi=150); plt.close()
        # delta error hist
        deltas=[m['delta_error'] for m in matches if m['delta_error'] is not None]
        plt.figure(figsize=(6,4)); plt.hist(deltas, bins=50); plt.xlabel('error_km difference'); plt.ylabel('count'); plt.title('Delta error (method A - method B)'); plt.tight_layout(); plt.savefig(os.path.join(outdir, 'match_delta_error_hist.png'), dpi=150); plt.close()
        # delta residual
        delres=[m['delta_res'] for m in matches if m['delta_res'] is not None]
        plt.figure(figsize=(6,4)); plt.hist(delres, bins=50); plt.xlabel('residual difference'); plt.ylabel('count'); plt.title('Delta residual (method A - method B)'); plt.tight_layout(); plt.savefig(os.path.join(outdir, 'match_delta_residual_hist.png'), dpi=150); plt.close()
    return matches


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--json', required=False, default=None)
    parser.add_argument('--centers', required=False)
    parser.add_argument('--out-dir', default='plots')
    parser.add_argument('--per-method-maps', action='store_true', help='Write separate maps for each method')
    parser.add_argument('--zoom-percentile', type=float, default=None, help='If set, zoom to the given percentile (e.g., 5 -> 5th-95th percentile)')
    parser.add_argument('--zoom-bbox', type=str, default=None, help='Comma list: minlon,maxlon,minlat,maxlat to explicitly set bbox')
    parser.add_argument('--zoom-center', type=str, default=None, help='Center for zoom: lat,lon')
    parser.add_argument('--zoom-radius-km', type=float, default=None, help='Radius in km for zoom center')
    parser.add_argument('--matched-zoom-radius-km', type=float, default=50.0, help='Radius in km for matched-pairs focus map')
    args=parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows=read_csv(args.csv)
    if args.json:
        # json contains full fields; but csv is okay
        pass
    groups=prepare_method_groups(rows)
    # find matched pairs so we can optionally create a matched-area zoom
    matches=match_methods_and_diffs(groups, args.out_dir, max_time_diff_s=1.0, max_dist_km=50.0)

    # main map (full extent)
    plot_map(groups, os.path.join(args.out_dir, 'method_locations_map.png'), args.centers)

    # If requested, make a zoomed map based on percentile, bbox, or center+radius
    extent = None
    if args.zoom_bbox:
        try:
            minlon,maxlon,minlat,maxlat = [float(x) for x in args.zoom_bbox.split(',')]
            extent = (minlon,maxlon,minlat,maxlat)
        except Exception:
            print('Invalid --zoom-bbox value; ignoring')
    elif args.zoom_center and args.zoom_radius_km:
        try:
            lat, lon = [float(x) for x in args.zoom_center.split(',')]
            rad = float(args.zoom_radius_km)
            dlat = rad/111.0
            dlon = rad/(111.0 * max(0.1, math.cos(math.radians(lat))))
            extent = (lon-dlon, lon+dlon, lat-dlat, lat+dlat)
        except Exception:
            print('Invalid --zoom-center or --zoom-radius-km; ignoring')
    elif args.zoom_percentile is not None:
        # compute percentiles across all points
        all_lats=[]; all_lons=[]
        for g in groups.values():
            for r in g:
                all_lats.append(r['latf']); all_lons.append(r['lonf'])
        p_low=args.zoom_percentile; p_high=100.0-p_low
        lon_min=float(np.percentile(all_lons,p_low)); lon_max=float(np.percentile(all_lons,p_high))
        lat_min=float(np.percentile(all_lats,p_low)); lat_max=float(np.percentile(all_lats,p_high))
        extent=(lon_min,lon_max,lat_min,lat_max)

    if extent is not None:
        zoom_out = os.path.join(args.out_dir, 'method_locations_map_zoom.png')
        plot_map(groups, zoom_out, args.centers, extent=extent)

    # Produce per-method maps if requested
    if args.per_method_maps:
        for method, items in groups.items():
            safe_name = method.replace(' ', '_').replace(';', '_').replace('/', '_')
            outp = os.path.join(args.out_dir, f'method_locations_{safe_name}.png')
            plot_map({method: items}, outp, args.centers, extent=extent)

    # If we found matches, create a matched-pairs focused zoom around median midpoint
    if matches:
        # fallback: center on median of all lat/lons of matched items
        mid_lats = []
        mid_lons = []
        # re-create items mapping for efficient lookups
        items=[]
        for m, lst in groups.items():
            for r in lst:
                items.append({'method':m, 'lat':r['latf'], 'lon':r['lonf'], 'te':r['tepochf']})
        # for speed, compute medians over items' lat/lon
        if items:
            lat_med = float(np.median([r['lat'] for r in items])); lon_med = float(np.median([r['lon'] for r in items]))
            rad = float(args.matched_zoom_radius_km or 50.0)
            dlat = rad/111.0
            dlon = rad/(111.0 * max(0.1, math.cos(math.radians(lat_med))))
            extent_mp = (lon_med-dlon, lon_med+dlon, lat_med-dlat, lat_med+dlat)
            plot_map(groups, os.path.join(args.out_dir, 'method_locations_map_matched_zoom.png'), args.centers, extent=extent_mp)
    polar_hist_by_method(groups, os.path.join(args.out_dir, 'method_backazimuth_polar.png'))
    error_hist_by_method(groups, os.path.join(args.out_dir, 'method_error_hist.png'))
    residual_vs_snr(groups, os.path.join(args.out_dir, 'method_residual_vs_snr.png'))
    # (already computed matches above and written to CSV)
    print('Methods found:', {m:len(g) for m,g in groups.items()})
    print('Matching pairs (see method_matches_diffs.csv):', os.path.join(args.out_dir, 'method_matches_diffs.csv'))

if __name__=='__main__':
    main()
