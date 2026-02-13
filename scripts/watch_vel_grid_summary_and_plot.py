#!/usr/bin/env python3
"""
Wait for `vel_grid_full_summary.json` to appear and run `plot_vel_grid_summary.py`.

This is useful to automatically generate final plots after the long-running full-grid run finishes.
"""
import time
import os
import subprocess
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', required=True, help='Path to vel_grid_full_summary.json to watch for')
    args = parser.parse_args()
    s = args.summary
    print('Watching for', s)
    while True:
        if os.path.exists(s):
            print('Found summary, running plot script')
            cmd = ['python3', 'scripts/plot_vel_grid_summary.py', '--summary', s]
            subprocess.run(cmd, check=True)
            print('Plotting done')
            break
        time.sleep(30)


if __name__ == '__main__':
    main()
