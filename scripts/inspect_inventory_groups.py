#!/usr/bin/env python3
"""
Small utility to inspect station inventory and check explicit subarray group membership.
"""
import sys, json
sys.path.insert(0, '.')
from beam_driver import TraditionalBeamformer

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--inventory', default='.', help='Inventory file/folder')
    parser.add_argument('--inventory-pattern', default='*.xml', help='Glob pattern to find inventory files')
    parser.add_argument('--inventory-tag', default=None, help='Optional substring tag to filter inventory filenames')
    parser.add_argument('--inventory-name', default=None, help='Optional exact inventory filename to load (basename)')
    parser.add_argument('--subarrays-file', default=None, help='JSON file listing groups')
    parser.add_argument('--subarrays', default=None, help='Inline groups: STA1,STA2;STA3,STA4')
    args = parser.parse_args()

    bf = TraditionalBeamformer(data_dir='.')
    bf.load_inventory_from_folder(args.inventory, pattern=args.inventory_pattern, tag=args.inventory_tag, name=args.inventory_name)

    print('Station codes in inventory:')
    codes = sorted(list(bf.station_coords.keys()))
    print('\n'.join(codes))
    print('\nTotal:', len(codes))

    groups = None
    if args.subarrays_file:
        groups = json.load(open(args.subarrays_file))
    elif args.subarrays:
        groups = [g.strip() for g in args.subarrays.split(';')]
        groups = [[s.strip() for s in g.split(',') if s.strip()] for g in groups]

    if groups is None:
        print('No groups provided; printing station centers:')
        for sta, (lat, lon, elev) in bf.station_coords.items():
            print(sta, lat, lon, elev)
        sys.exit(0)

    print('\nVerifying provided groups:')
    all_stations = set(codes)
    missing = []
    for i, g in enumerate(groups):
        print(f'Group {i}: {g}')
        for s in g:
            if s not in all_stations:
                missing.append(s)
    if missing:
        print('\nStations not found in inventory:', missing)
    else:
        print('\nAll provided stations present in inventory')
        # Also compute leftover stations
        grouped = set([s for g in groups for s in g])
        leftovers = sorted(list(all_stations - grouped))
        print('Leftover stations not in any group:', leftovers)
