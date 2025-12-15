#!/usr/bin/env bash
# Run LSQ-only per-velocity plotting (wrapping the suite scripts under plots/ for convenience)
set -euo pipefail
BASE=plots/vel_grid_full_relaxed
VELS="2.5 3.0 3.5 4.0"
ZOOM=100
COLOR_BYS="mean_snr,azimuth_variance"
for v in ${VELS}; do
  outdir="$BASE/suite/lsq_only_vel_${v}"
  mkdir -p "$outdir"
  echo "Running lsq-only plot for vel=$v -> $outdir"
  python3 scripts/plot_vel_grid_zoomed.py --base "$BASE" --outdir "$outdir" --vels "$v" --zooms "$ZOOM" --color-bys "$COLOR_BYS" --only-method lsq
  echo "Finished $outdir"
done
