#!/usr/bin/env python3
"""
Create a combined figure: left = map of triangulated points + rays, right = combined polar histogram.
Uses existing scripts' functionality to avoid duplicating logic.
"""
import argparse
import subprocess
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detections', help='Detections JSON for polar plot', required=True)
    parser.add_argument('--locations-csv', help='Locations CSV for map', required=True)
    parser.add_argument('--centers', help='centers.json', required=True)
    parser.add_argument('--out', default='plots/combined_map_polar.png')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    # Create map with rays (all events)
    map_out = args.out.replace('.png', '_map.png')
    subprocess.check_call(['python3', 'scripts/plot_locations.py', '--csv', args.locations_csv, '--centers', args.centers, '--out', map_out, '--show-rays'])

    # Create combined polar
    polar_out = args.out.replace('.png', '_polar.png')
    subprocess.check_call(['python3', 'scripts/plot_beam_polar.py', '--detections', args.detections, '--out', polar_out, '--bins', '36', '--annotate-mean'])

    # Combine using PIL
    from PIL import Image
    map_img = Image.open(map_out)
    polar_img = Image.open(polar_out)
    # Resize to same height
    h = max(map_img.height, polar_img.height)
    map_img = map_img.resize((int(map_img.width * h / map_img.height), h))
    polar_img = polar_img.resize((int(polar_img.width * h / polar_img.height), h))
    neww = map_img.width + polar_img.width
    new = Image.new('RGB', (neww, h), (255,255,255))
    new.paste(map_img, (0,0))
    new.paste(polar_img, (map_img.width, 0))
    new.save(args.out)
    print('Wrote', args.out)

if __name__ == '__main__':
    main()
