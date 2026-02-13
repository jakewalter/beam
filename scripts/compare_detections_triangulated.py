#!/usr/bin/env python3
"""
Compare angular distributions: raw beam detections vs triangulated event bearings.

Produces an overlaid polar histogram: raw detections (lighter fill) and
triangulated-derived mean backazimuth per event (darker outline), so you see
how the triangulated subset differs from raw detections.
"""
import argparse
import json
import csv
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_detections_json(fp):
    with open(fp) as fh:
        data=json.load(fh)
    rows=[]
    if isinstance(data, dict):
        for k,v in data.items():
            if isinstance(v,list): rows.extend(v)
    elif isinstance(data,list):
        rows=data
    return rows


def read_triangulated_csv(fp):
    rows=[]
    with open(fp) as fh:
        r=csv.DictReader(fh)
        for row in r:
            if row.get('triangulated') and row['triangulated'].lower() in ('true','1'):
                rows.append(row)
    return rows


def polar_hist_overlay(dets, tris, out, bins=36):
    # dets: list of detections dicts with 'backazimuth' and optional 'snr'
    # tris: list of triangulated rows with 'backazimuths' (semicolon) field
    det_angles = [float(d['backazimuth'])%360 for d in dets if 'backazimuth' in d]
    det_weights = [float(d.get('snr',1.0) or 1.0) for d in dets if 'backazimuth' in d]

    tri_angles=[]
    for r in tris:
        bstr=r.get('backazimuths','')
        parts=[p for p in bstr.split(';') if p.strip()]
        vals=[float(p) for p in parts] if parts else []
        if vals:
            mean_val = sum(vals)/len(vals)
            tri_angles.append(mean_val%360)

    # convert to radians and prepare hist
    theta_det = np.radians(det_angles)
    theta_tri = np.radians(tri_angles)

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(1,1,1, projection='polar')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    # det histogram (filled)
    counts_det, edges = np.histogram(theta_det, bins=bins, range=(0,2*math.pi), weights=det_weights)
    widths = np.diff(edges)
    ax.bar(edges[:-1], counts_det, width=widths, bottom=0.0, color='C0', alpha=0.4, align='edge', edgecolor='none')

    # tri histogram (outline)
    counts_tri, edges = np.histogram(theta_tri, bins=bins, range=(0,2*math.pi))
    # normalize tri counts to det max so visible
    if counts_tri.max()>0 and counts_det.max()>0:
        scale = counts_det.max()/counts_tri.max()
    else:
        scale = 1.0
    ax.step(edges, np.append(counts_tri*scale, counts_tri[0]*scale), where='post', color='C1', linewidth=1.5, label='Triangulated (scaled)')

    # annotate means
    def circ_mean_deg(angles_deg):
        a=np.radians(angles_deg)
        s=np.sum(np.sin(a)); c=np.sum(np.cos(a))
        return (math.degrees(math.atan2(s,c))%360)

    if det_angles:
        mdet=circ_mean_deg(det_angles)
        ax.annotate('', xy=(math.radians(mdet), max(counts_det)*0.9), xytext=(0,0), arrowprops=dict(facecolor='C0', width=2, headwidth=8))
        ax.text(math.radians(mdet), max(counts_det)*0.95, f'Det mean {mdet:.1f}°', color='C0')
    if tri_angles:
        mtri=circ_mean_deg(tri_angles)
        ax.annotate('', xy=(math.radians(mtri), max(counts_det)*0.8), xytext=(0,0), arrowprops=dict(facecolor='C1', width=2, headwidth=8))
        ax.text(math.radians(mtri), max(counts_det)*0.85, f'Tri mean {mtri:.1f}°', color='C1')

    ax.set_title('Detections (filled) vs Triangulated (outline) backazimuths')
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print('Wrote', out)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--detections', required=True)
    parser.add_argument('--triangulated-csv', required=True)
    parser.add_argument('--out', default='plots/det_vs_tri_polar.png')
    parser.add_argument('--bins', type=int, default=36)
    args=parser.parse_args()
    dets=read_detections_json(args.detections)
    tris=read_triangulated_csv(args.triangulated_csv)
    polar_hist_overlay(dets, tris, args.out, bins=args.bins)

if __name__ == '__main__':
    main()
