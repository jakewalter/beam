#!/usr/bin/env python3
"""
Diagnostic script to analyze detection pairing and triangulation quality.

Usage:
    PYTHONPATH=. python3 scripts/diagnose_triangulation.py \\
        --detections-dir /path/to/detections \\
        --time-tol 30.0 \\
        [--date YYYYMMDD]
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


def load_json(filepath):
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def analyze_detections(detections_file, centers_file, time_tol):
    """Analyze detection pairing potential."""
    
    print(f"\n{'='*80}")
    print(f"ANALYZING: {os.path.basename(detections_file)}")
    print(f"{'='*80}\n")
    
    # Load data
    detections_raw = load_json(detections_file)
    centers = load_json(centers_file)
    
    # Handle both formats: list or dict with date keys
    if isinstance(detections_raw, dict):
        # Extract detections from first (and likely only) date key
        detections = []
        for date_key, date_detections in detections_raw.items():
            detections.extend(date_detections)
    else:
        detections = detections_raw
    
    # Group by subarray
    by_subarray = defaultdict(list)
    for det in detections:
        subarray_id = det.get('subarray_id', 'unknown')
        by_subarray[subarray_id].append(det)
    
    print(f"DETECTION SUMMARY:")
    print(f"  Total detections: {len(detections)}")
    print(f"  Number of subarrays: {len(by_subarray)}")
    print(f"  Time tolerance: {time_tol} seconds\n")
    
    print(f"DETECTIONS PER SUBARRAY:")
    for subarray_id in sorted(by_subarray.keys()):
        count = len(by_subarray[subarray_id])
        print(f"  Subarray {subarray_id}: {count} detections")
    
    # Check for multi-array pairing potential
    print(f"\n{'='*80}")
    print(f"PAIRING ANALYSIS (time_tol = {time_tol}s)")
    print(f"{'='*80}\n")
    
    # Convert all detection times to float
    det_times = {}
    for subarray_id, dets in by_subarray.items():
        times = []
        for det in dets:
            time_val = det.get('time')
            if time_val:
                # Handle both timestamp float and ISO string
                if isinstance(time_val, (int, float)):
                    times.append(float(time_val))
                else:
                    try:
                        dt = datetime.fromisoformat(time_val.replace('Z', '+00:00'))
                        times.append(dt.timestamp())
                    except:
                        pass
        det_times[subarray_id] = np.array(sorted(times))
    
    # Count potential pairs
    subarray_ids = sorted(by_subarray.keys())
    n_subarrays = len(subarray_ids)
    
    if n_subarrays < 2:
        print(f"WARNING: Only {n_subarrays} subarray(s) detected.")
        print(f"  Triangulation requires at least 2 arrays!")
        print(f"  Single-array detections only provide backazimuth (direction),")
        print(f"  not range -> produces ring/arc patterns around array.\n")
        return
    
    print(f"Pairwise pairing potential:")
    total_pairs = 0
    pair_matrix = np.zeros((n_subarrays, n_subarrays), dtype=int)
    
    for i, sid1 in enumerate(subarray_ids):
        for j, sid2 in enumerate(subarray_ids):
            if i >= j:
                continue
            
            times1 = det_times[sid1]
            times2 = det_times[sid2]
            
            # Count coincident detections
            pairs = 0
            for t1 in times1:
                time_diffs = np.abs(times2 - t1)
                if np.any(time_diffs <= time_tol):
                    pairs += 1
            
            pair_matrix[i, j] = pairs
            total_pairs += pairs
            
            if pairs > 0:
                print(f"  Subarray {sid1} <-> {sid2}: {pairs} potential pairs")
            else:
                print(f"  Subarray {sid1} <-> {sid2}: 0 pairs (arrays not detecting common events)")
    
    print(f"\nTotal potential pairwise triangulations: {total_pairs}")
    
    if total_pairs == 0:
        print(f"\n⚠️  WARNING: NO PAIRING POSSIBLE")
        print(f"  Possible causes:")
        print(f"    1. time_tol ({time_tol}s) too strict for surface waves")
        print(f"    2. Arrays detecting different events (no overlap)")
        print(f"    3. Detection threshold too high (missing weak signals)")
        print(f"  Recommendations:")
        print(f"    - Increase --time-tol to 60-120s for surface waves")
        print(f"    - Lower detection threshold (--threshold)")
        print(f"    - Check that arrays have sufficient common aperture\n")
    else:
        print(f"\n✓ Pairing is possible with current time_tol")
    
    # LSQ analysis
    print(f"\n{'='*80}")
    print(f"LSQ MULTI-ARRAY ANALYSIS")
    print(f"{'='*80}\n")
    
    if n_subarrays < 3:
        print(f"LSQ requires ≥3 arrays. You have {n_subarrays}.")
        print(f"  Only pairwise triangulation available.\n")
    else:
        # Count events detected by 3+ arrays
        all_times = []
        for times in det_times.values():
            all_times.extend(times)
        all_times = np.array(sorted(all_times))
        
        multi_array_events = 0
        for t in all_times:
            n_arrays_detecting = 0
            for times in det_times.values():
                if np.any(np.abs(times - t) <= time_tol):
                    n_arrays_detecting += 1
            if n_arrays_detecting >= 3:
                multi_array_events += 1
        
        print(f"Events detected by ≥3 arrays: {multi_array_events}")
        if multi_array_events > 0:
            print(f"  ✓ LSQ multi-array localization possible")
            print(f"  LSQ provides better constraints than pairwise triangulation\n")
        else:
            print(f"  No events with ≥3 array detections")
            print(f"  Consider increasing time_tol or lowering threshold\n")
    
    # Geometry analysis
    print(f"{'='*80}")
    print(f"ARRAY GEOMETRY")
    print(f"{'='*80}\n")
    
    center_coords = []
    center_ids = []
    for subarray_id in subarray_ids:
        if subarray_id in centers:
            center_coords.append([centers[subarray_id]['lat'], centers[subarray_id]['lon']])
            center_ids.append(subarray_id)
    
    if len(center_coords) >= 2:
        center_coords = np.array(center_coords)
        
        print(f"Array centers:")
        for i, sid in enumerate(center_ids):
            print(f"  Subarray {sid}: lat={center_coords[i,0]:.4f}, lon={center_coords[i,1]:.4f}")
        
        # Approximate inter-array distances
        print(f"\nInter-array distances (approximate, degrees):")
        for i in range(len(center_ids)):
            for j in range(i+1, len(center_ids)):
                dist = np.sqrt((center_coords[i,0] - center_coords[j,0])**2 + 
                              (center_coords[i,1] - center_coords[j,1])**2)
                dist_km = dist * 111.0  # rough conversion
                print(f"  {center_ids[i]} <-> {center_ids[j]}: {dist:.3f}° (~{dist_km:.1f} km)")
        
        # Check for linear geometry
        if len(center_coords) >= 3:
            # Simple linearity check: compute area of triangle
            c = center_coords[:3]
            area = 0.5 * abs((c[1,0] - c[0,0]) * (c[2,1] - c[0,1]) - 
                            (c[2,0] - c[0,0]) * (c[1,1] - c[0,1]))
            if area < 0.001:  # very small area
                print(f"\n⚠️  WARNING: Arrays may be nearly collinear")
                print(f"    Triangle area: {area:.6f} deg²")
                print(f"    Linear geometry gives poor location constraints\n")
    
    print(f"{'='*80}\n")


def check_location_files(detections_dir, date=None):
    """Check for existing location output files."""
    
    print(f"\n{'='*80}")
    print(f"CHECKING FOR LOCATION OUTPUT FILES")
    print(f"{'='*80}\n")
    
    if date:
        patterns = [
            f"locations_{date}.json",
            f"locations_lsq_{date}.json",
            f"locations_summary_{date}.csv"
        ]
    else:
        patterns = [
            "locations_*.json",
            "locations_lsq_*.json",
            "locations_summary_*.csv"
        ]
    
    found_any = False
    for pattern in patterns:
        if '*' in pattern:
            import glob
            files = glob.glob(os.path.join(detections_dir, pattern))
            if files:
                found_any = True
                print(f"Found {len(files)} files matching {pattern}:")
                for f in sorted(files)[:5]:  # show first 5
                    print(f"  {os.path.basename(f)}")
                if len(files) > 5:
                    print(f"  ... and {len(files)-5} more")
        else:
            filepath = os.path.join(detections_dir, pattern)
            if os.path.exists(filepath):
                found_any = True
                size = os.path.getsize(filepath)
                print(f"✓ Found: {pattern} ({size} bytes)")
                
                # If JSON, peek at content
                if pattern.endswith('.json'):
                    try:
                        data = load_json(filepath)
                        if isinstance(data, list):
                            print(f"    Contains {len(data)} location entries")
                            if len(data) > 0:
                                print(f"    Sample keys: {list(data[0].keys())}")
                    except:
                        pass
            else:
                print(f"✗ Not found: {pattern}")
    
    if not found_any:
        print(f"\n⚠️  NO LOCATION FILES FOUND")
        print(f"  This suggests triangulation has not been run.")
        print(f"  Single-array detections only provide backazimuth -> ring patterns!")
        print(f"\nTo run triangulation:")
        print(f"  PYTHONPATH=. python3 scripts/triangulate_from_detections_json.py \\")
        print(f"    --detections {detections_dir}/detections_YYYYMMDD.json \\")
        print(f"    --centers {detections_dir}/centers.json \\")
        print(f"    --date YYYYMMDD --time-tol 30.0 --outdir {detections_dir}\n")
    
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose detection pairing and triangulation quality"
    )
    parser.add_argument(
        '--detections-dir',
        required=True,
        help='Directory containing detection JSON files'
    )
    parser.add_argument(
        '--date',
        help='Specific date (YYYYMMDD) to analyze. If omitted, analyzes all detections_*.json'
    )
    parser.add_argument(
        '--time-tol',
        type=float,
        default=30.0,
        help='Time tolerance for pairing (seconds), default=30.0'
    )
    
    args = parser.parse_args()
    
    detections_dir = args.detections_dir
    
    # Find detection files
    if args.date:
        det_file = os.path.join(detections_dir, f'detections_{args.date}.json')
        if not os.path.exists(det_file):
            print(f"ERROR: {det_file} not found")
            return 1
        det_files = [det_file]
    else:
        import glob
        det_files = sorted(glob.glob(os.path.join(detections_dir, 'detections_*.json')))
        if not det_files:
            print(f"ERROR: No detections_*.json files found in {detections_dir}")
            return 1
    
    # Find centers file
    centers_file = os.path.join(detections_dir, 'centers.json')
    if not os.path.exists(centers_file):
        print(f"ERROR: {centers_file} not found")
        return 1
    
    # Check for location files first
    check_location_files(detections_dir, date=args.date)
    
    # Analyze each detection file
    for det_file in det_files:
        analyze_detections(det_file, centers_file, args.time_tol)
    
    print(f"\n{'='*80}")
    print(f"RECOMMENDATIONS")
    print(f"{'='*80}\n")
    print(f"If you see ring patterns in your locations:")
    print(f"  1. Verify triangulation ran (check for locations_*.json files)")
    print(f"  2. If no pairing: increase --time-tol to 60-120s for surface waves")
    print(f"  3. Use LSQ locations (locations_lsq_*.json) if ≥3 arrays available")
    print(f"  4. Filter by error_km when clustering to remove poor intersections")
    print(f"  5. Check array geometry (avoid linear configurations)\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
